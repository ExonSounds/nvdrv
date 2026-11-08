// SPDX-License-Identifier: GPL-2.0
/*
 * The subset of nouveau's DRM uAPI (include/uapi/drm/nouveau_drm.h) that Mesa's NVK
 * Vulkan driver uses, on top of GSP-RM: GETPARAM, a few NVIF object calls (device info,
 * class list, engine objects), GEM objects in VRAM or system memory, VM_INIT/VM_BIND
 * on a per-file GPU address space, CHANNEL_ALLOC/FREE and EXEC with DRM syncobjs.
 *
 * Model (nouveau's, simplified): each DRM file gets its own RM client and an externally
 * owned VA space whose page tables nvdrv writes (src/fifo.c). VM_BIND runs synchronously.
 * EXEC waits for its input fences in the ioctl, writes the user's push buffers into the
 * channel's GPFIFO followed by a host semaphore release, and returns a dma_fence that a
 * polling hrtimer signals once the semaphore reaches its value (no interrupts yet).
 *
 * Enabled with the module parameter nvk=1, which also names the DRM driver "nouveau":
 * Mesa only binds NVK to devices of that name.
 */
#include <linux/dma-fence.h>
#include <linux/dma-fence-chain.h>
#include <linux/dma-mapping.h>
#include <linux/highmem.h>
#include <linux/hrtimer.h>
#include <linux/idr.h>
#include <linux/iopoll.h>
#include <linux/mm.h>
#include <linux/module.h>
#include <linux/slab.h>

#include <drm/drm_device.h>
#include <drm/drm_drv.h>
#include <drm/drm_file.h>
#include <drm/drm_gem.h>
#include <drm/drm_ioctl.h>
#include <drm/drm_prime.h>
#include <drm/drm_print.h>
#include <drm/drm_syncobj.h>
#include <uapi/drm/nouveau_drm.h>

#include "nvdrv.h"

bool nv_nvk;
module_param_named(nvk, nv_nvk, bool, 0444);
MODULE_PARM_DESC(nvk, "Offer nouveau's uAPI for Mesa NVK (DRM driver named \"nouveau\"; experimental)");

static bool dump_push;
module_param(dump_push, bool, 0644);
MODULE_PARM_DESC(dump_push, "Debug: print the dwords of every EXEC push buffer (GART objects only)");

#define UAPI_GPFIFO_ENTRIES	512		/* must match fifo.c */
#define UAPI_PUSH_MAX		(UAPI_GPFIFO_ENTRIES / 2 - 1)
#define UAPI_POLL_NS		(100 * NSEC_PER_USEC)
#define UAPI_MAX_CHANS		16
#define H_CLIENT_BASE		0xc1e00000	/* + file number */
#define H_UCHAN(i)		(0xd000c000 + (i))
#define H_UOBJ(chid, i)		(0xd1000000 + ((chid) << 8) + (i))

/* engine classes NVK looks for on a GR channel (ga102 GR + GRCE) */
#define AMPERE_DMA_COPY_B		0xc7b5
#define FERMI_TWOD_A			0x902d
#define AMPERE_B			0xc797
#define KEPLER_INLINE_TO_MEMORY_B	0xa140
#define AMPERE_COMPUTE_B		0xc7c0
static const u32 gr_classes[] = { AMPERE_DMA_COPY_B, FERMI_TWOD_A, AMPERE_B, KEPLER_INLINE_TO_MEMORY_B,
				  AMPERE_COMPUTE_B };

/* ---- per-file state ---------------------------------------------------------- */

struct nv_cli {
	struct nvdrv_dev *nv;
	struct mutex lock;		/* vm, vmas, channels */
	struct nv_vmm vm;
	bool vm_ready;
	u64 kernel_va, kernel_size;
	struct list_head vmas;		/* struct nv_vma, sorted by addr */
	struct idr chans;		/* struct nv_uchan */
};

struct nv_uchan {
	struct nv_chan ch;
	struct nv_cli *cli;
	int id;
	u32 nobj;
	bool gr_ctx;			/* GR context promoted (on the first GR class object) */
	u32 submitted;			/* last fence seqno pushed */
	u64 fence_ctx;
	struct list_head pending;	/* struct nv_fence, under nv->uapi.lock */
	struct list_head active;	/* in nv->uapi.active while pending is not empty */
	struct dma_fence *last;		/* most recent fence */
	bool dead;
};

struct nv_fence {
	struct dma_fence base;
	struct list_head link;
	u32 seqno;
};

struct nv_bo {
	struct drm_gem_object base;
	struct nvdrv_dev *nv;
	u32 domain, tile_mode, tile_flags;
	bool vram;
	struct drm_mm_node node;	/* VRAM */
	struct drm_mm_node bar1;	/* BAR1 VA while CPU-mappable */
	struct page **pages;		/* GART */
	dma_addr_t *dma;		/* GART: bus address per page */
	struct sg_table *sgt;
};

struct nv_vma {
	struct list_head link;
	u64 addr, range, bo_offset;
	struct nv_bo *bo;		/* NULL: sparse region */
	u8 kind;
};

static inline struct nv_bo *to_nv_bo(struct drm_gem_object *obj)
{
	return container_of(obj, struct nv_bo, base);
}

#define dev_nv nv_drm_nv

/* ---- fences --------------------------------------------------------------------- */

static const char *fence_driver_name(struct dma_fence *f)
{
	return "nvdrv";
}

static const char *fence_timeline_name(struct dma_fence *f)
{
	return "nvdrv-channel";
}

static const struct dma_fence_ops nv_fence_ops = {
	.get_driver_name = fence_driver_name,
	.get_timeline_name = fence_timeline_name,
};

/* Signal what the semaphore has passed; caller holds uapi.lock. */
static bool uchan_poll_locked(struct nv_uchan *u)
{
	u32 sem = READ_ONCE(*(u32 *)u->ch.sem.cpu);
	const u32 *n = u->ch.errnotif.cpu;
	struct nv_fence *f, *tmp;
	bool dead = n && (n[3] >> 16);

	if (dead && !u->dead) {
		u->dead = true;
		pr_warn("nvdrv: channel %u recovered by RM: error %u (info16 0x%x), semaphore %u/%u, GP_GET %u/%u\n",
			u->ch.chid, n[2], n[3] & 0xffff, sem, u->submitted, ioread32(u->ch.userd_cpu + 0x88), u->ch.gp_put);
	}
	list_for_each_entry_safe(f, tmp, &u->pending, link) {
		if (!dead && (s32)(sem - f->seqno) < 0)
			break;
		if (dead)
			dma_fence_set_error(&f->base, -EIO);
		dma_fence_signal_locked(&f->base);
		list_del_init(&f->link);
		dma_fence_put(&f->base);
	}
	return !list_empty(&u->pending);
}

static enum hrtimer_restart uapi_poll(struct hrtimer *t)
{
	struct nv_uapi *ua = container_of(t, struct nv_uapi, timer);
	struct nv_uchan *u, *tmp;
	unsigned long flags;
	bool more;

	spin_lock_irqsave(&ua->lock, flags);
	list_for_each_entry_safe(u, tmp, &ua->active, active)
		if (!uchan_poll_locked(u))
			list_del_init(&u->active);
	more = !list_empty(&ua->active);
	ua->polling = more;
	spin_unlock_irqrestore(&ua->lock, flags);
	if (!more)
		return HRTIMER_NORESTART;
	hrtimer_forward_now(t, ns_to_ktime(UAPI_POLL_NS));
	return HRTIMER_RESTART;
}

/* A fence for seqno on u, already queued for polling. */
static struct dma_fence *uchan_fence(struct nvdrv_dev *nv, struct nv_uchan *u, u32 seqno)
{
	struct nv_uapi *ua = &nv->uapi;
	struct nv_fence *f = kzalloc(sizeof(*f), GFP_KERNEL);
	unsigned long flags;

	if (!f)
		return NULL;
	f->seqno = seqno;
	dma_fence_init(&f->base, &nv_fence_ops, &ua->lock, u->fence_ctx, seqno);
	spin_lock_irqsave(&ua->lock, flags);
	dma_fence_get(&f->base);	/* the pending list's reference */
	list_add_tail(&f->link, &u->pending);
	if (list_empty(&u->active))
		list_add_tail(&u->active, &ua->active);
	if (!ua->polling) {	/* the callback clears it under this lock when it stops */
		ua->polling = true;
		hrtimer_start(&ua->timer, ns_to_ktime(UAPI_POLL_NS), HRTIMER_MODE_REL);
	}
	spin_unlock_irqrestore(&ua->lock, flags);
	return &f->base;
}

/* Wait for everything submitted on u (up to 2 s), then fail what is left. */
static void uchan_drain(struct nvdrv_dev *nv, struct nv_uchan *u)
{
	struct nv_uapi *ua = &nv->uapi;
	unsigned long flags;

	if (u->last)
		dma_fence_wait_timeout(u->last, false, 2 * HZ);
	spin_lock_irqsave(&ua->lock, flags);
	u->dead = true;
	{
		struct nv_fence *f, *tmp;

		list_for_each_entry_safe(f, tmp, &u->pending, link) {
			dma_fence_set_error(&f->base, -ENODEV);
			dma_fence_signal_locked(&f->base);
			list_del_init(&f->link);
			dma_fence_put(&f->base);
		}
	}
	if (!list_empty(&u->active))
		list_del_init(&u->active);
	spin_unlock_irqrestore(&ua->lock, flags);
	dma_fence_put(u->last);
	u->last = NULL;
}

/* Wait for every channel of the client to go idle: 0, -EBUSY (timeout) or an error. */
static int cli_wait_idle(struct nv_cli *cli, bool intr, long timeout)
{
	struct nv_uchan *u;
	int id;

	idr_for_each_entry(&cli->chans, u, id) {
		long t;

		if (!u->last)
			continue;
		t = dma_fence_wait_timeout(u->last, intr, timeout);
		if (t <= 0)
			return t ?: -EBUSY;
	}
	return 0;
}

/* ---- GEM objects ------------------------------------------------------------- */

static int bar1_pool(struct nvdrv_dev *nv)
{
	struct nv_uapi *ua = &nv->uapi;
	u64 start = ALIGN(NV_FB_MAX + SZ_64K, SZ_2M);	/* after scanout surface and USERD */

	if (ua->bar1_ready)
		return 0;
	if (!nv->bar1.size || nv->bar1.size <= start)
		return -ENODEV;
	drm_mm_init(&ua->bar1_mm, start, nv->bar1.size - start);
	ua->bar1_ready = true;
	return 0;
}

static void nv_bo_free(struct drm_gem_object *obj)
{
	struct nv_bo *bo = to_nv_bo(obj);
	struct nvdrv_dev *nv = bo->nv;

	if (drm_mm_node_allocated(&bo->bar1)) {
		mutex_lock(&nv->gsp_lock);
		nv_bar1_set(nv, bo->bar1.start, NV_BAR1_UNMAP, bo->bar1.size);
		mutex_unlock(&nv->gsp_lock);
		mutex_lock(&nv->uapi.bar1_lock);
		drm_mm_remove_node(&bo->bar1);
		mutex_unlock(&nv->uapi.bar1_lock);
	}
	nv_mem_put(nv, &bo->node);
	if (bo->sgt) {
		dma_unmap_sgtable(&nv->pdev->dev, bo->sgt, DMA_BIDIRECTIONAL, 0);
		sg_free_table(bo->sgt);
		kfree(bo->sgt);
	}
	kvfree(bo->dma);
	if (bo->pages)
		drm_gem_put_pages(obj, bo->pages, true, false);
	if (obj->import_attach)
		drm_prime_gem_destroy(obj, NULL);
	drm_gem_object_release(obj);
	kfree(bo);
}

static vm_fault_t nv_bo_fault(struct vm_fault *vmf)
{
	struct vm_area_struct *vma = vmf->vma;
	struct nv_bo *bo = to_nv_bo(vma->vm_private_data);
	pgoff_t pg = (vmf->address - vma->vm_start) >> PAGE_SHIFT;

	pg += vma->vm_pgoff - drm_vma_node_start(&bo->base.vma_node);
	if (pg >= bo->base.size >> PAGE_SHIFT)
		return VM_FAULT_SIGBUS;
	if (bo->vram)
		return vmf_insert_pfn(vma, vmf->address,
				      (pci_resource_start(bo->nv->pdev, 1) + bo->bar1.start) / PAGE_SIZE + pg);
	return vmf_insert_pfn(vma, vmf->address, page_to_pfn(bo->pages[pg]));
}

static const struct vm_operations_struct nv_bo_vm_ops = {
	.fault = nv_bo_fault,
	.open = drm_gem_vm_open,
	.close = drm_gem_vm_close,
};

/* CPU mappings: VRAM through BAR1 (mapped on first mmap, write-combined), GART cached. */
static int nv_bo_mmap(struct drm_gem_object *obj, struct vm_area_struct *vma)
{
	struct nv_bo *bo = to_nv_bo(obj);
	struct nvdrv_dev *nv = bo->nv;
	int ret = 0;

	if (bo->vram) {
		mutex_lock(&nv->uapi.bar1_lock);
		if (!drm_mm_node_allocated(&bo->bar1)) {
			ret = bar1_pool(nv);
			if (!ret)
				ret = drm_mm_insert_node_generic(&nv->uapi.bar1_mm, &bo->bar1, bo->node.size, SZ_64K,
								 0, DRM_MM_INSERT_BEST);
			if (!ret) {
				mutex_lock(&nv->gsp_lock);
				ret = nv_bar1_set(nv, bo->bar1.start, bo->node.start, bo->node.size);
				mutex_unlock(&nv->gsp_lock);
				if (ret)
					drm_mm_remove_node(&bo->bar1);
			}
		}
		mutex_unlock(&nv->uapi.bar1_lock);
		if (ret)
			return ret;
		vma->vm_page_prot = pgprot_writecombine(vm_get_page_prot(vma->vm_flags));
	} else {
		vma->vm_page_prot = vm_get_page_prot(vma->vm_flags);
	}
	vm_flags_set(vma, VM_PFNMAP | VM_IO | VM_DONTEXPAND | VM_DONTDUMP);
	vma->vm_ops = &nv_bo_vm_ops;
	return 0;
}

static struct sg_table *nv_bo_get_sg_table(struct drm_gem_object *obj)
{
	struct nv_bo *bo = to_nv_bo(obj);

	if (bo->vram)
		return ERR_PTR(-EOPNOTSUPP);	/* VRAM objects are shared within this device only */
	return drm_prime_pages_to_sg(obj->dev, bo->pages, obj->size >> PAGE_SHIFT);
}

static const struct drm_gem_object_funcs nv_bo_funcs = {
	.free = nv_bo_free,
	.mmap = nv_bo_mmap,
	.get_sg_table = nv_bo_get_sg_table,
};

static struct nv_bo *nv_bo_new(struct drm_device *dev, u64 size, u32 domain, u64 align)
{
	struct nvdrv_dev *nv = dev_nv(dev);
	struct nv_bo *bo;
	u64 i, npages;
	int ret;

	bo = kzalloc(sizeof(*bo), GFP_KERNEL);
	if (!bo)
		return ERR_PTR(-ENOMEM);
	bo->nv = nv;
	bo->base.funcs = &nv_bo_funcs;
	bo->vram = domain & NOUVEAU_GEM_DOMAIN_VRAM;
	bo->domain = bo->vram ? NOUVEAU_GEM_DOMAIN_VRAM : NOUVEAU_GEM_DOMAIN_GART;
	size = ALIGN(size, bo->vram ? SZ_64K : PAGE_SIZE);
	npages = size >> PAGE_SHIFT;
	if (bo->vram) {
		drm_gem_private_object_init(dev, &bo->base, size);
		mutex_lock(&nv->gsp_lock);
		ret = nv_fifo_init(nv);	/* CE channel; also sets up the pool under gsp_lock */
		if (!ret)
			ret = nv_mem_get(nv, &bo->node, size, max_t(u64, align, SZ_64K));
		if (!ret)	/* never hand out old VRAM contents */
			ret = nv_ce_clear(nv, bo->node.start, size);
		mutex_unlock(&nv->gsp_lock);
	} else {
		ret = drm_gem_object_init(dev, &bo->base, size);
		if (ret) {
			kfree(bo);
			return ERR_PTR(ret);
		}
		bo->pages = drm_gem_get_pages(&bo->base);	/* zeroed shmem pages */
		ret = PTR_ERR_OR_ZERO(bo->pages);
		if (ret)
			bo->pages = NULL;
		if (!ret) {
			bo->sgt = drm_prime_pages_to_sg(dev, bo->pages, npages);
			ret = PTR_ERR_OR_ZERO(bo->sgt);
			if (ret)
				bo->sgt = NULL;
		}
		if (!ret)
			ret = dma_map_sgtable(&nv->pdev->dev, bo->sgt, DMA_BIDIRECTIONAL, 0);
		if (!ret) {
			struct sg_dma_page_iter it;

			bo->dma = kvmalloc_array(npages, sizeof(*bo->dma), GFP_KERNEL);
			ret = bo->dma ? 0 : -ENOMEM;
			i = 0;
			if (!ret)
				for_each_sgtable_dma_page(bo->sgt, &it, 0)
					bo->dma[i++] = sg_page_iter_dma_address(&it);
		}
	}
	if (!ret)
		ret = drm_gem_create_mmap_offset(&bo->base);
	if (ret) {
		drm_gem_object_put(&bo->base);
		return ERR_PTR(ret);
	}
	return bo;
}

static void bo_info(struct drm_file *file, struct nv_bo *bo, u32 handle, struct drm_nouveau_gem_info *info)
{
	info->handle = handle;
	info->domain = bo->domain;
	info->size = bo->base.size;
	info->offset = 0;
	info->map_handle = drm_vma_node_offset_addr(&bo->base.vma_node);
	info->tile_mode = bo->tile_mode;
	info->tile_flags = bo->tile_flags;
}

static int ioctl_gem_new(struct drm_device *dev, void *data, struct drm_file *file)
{
	struct drm_nouveau_gem_new *req = data;
	struct nv_bo *bo;
	u32 handle;
	int ret;

	if (!req->info.size || !(req->info.domain & (NOUVEAU_GEM_DOMAIN_VRAM | NOUVEAU_GEM_DOMAIN_GART)))
		return -EINVAL;
	bo = nv_bo_new(dev, req->info.size, req->info.domain, req->align);
	if (IS_ERR(bo))
		return PTR_ERR(bo);
	bo->tile_mode = req->info.tile_mode;
	bo->tile_flags = req->info.tile_flags;
	ret = drm_gem_handle_create(file, &bo->base, &handle);
	if (!ret)
		bo_info(file, bo, handle, &req->info);
	drm_gem_object_put(&bo->base);
	return ret;
}

static int ioctl_gem_info(struct drm_device *dev, void *data, struct drm_file *file)
{
	struct drm_nouveau_gem_info *req = data;
	struct drm_gem_object *obj = drm_gem_object_lookup(file, req->handle);

	if (!obj)
		return -ENOENT;
	if (obj->funcs != &nv_bo_funcs) {
		drm_gem_object_put(obj);
		return -EINVAL;
	}
	bo_info(file, to_nv_bo(obj), req->handle, req);
	drm_gem_object_put(obj);
	return 0;
}

/* No per-object fences: wait for the whole client to be idle. */
static int ioctl_gem_cpu_prep(struct drm_device *dev, void *data, struct drm_file *file)
{
	struct drm_nouveau_gem_cpu_prep *req = data;
	struct nv_cli *cli = file->driver_priv;
	long t;

	mutex_lock(&cli->lock);
	t = cli_wait_idle(cli, true, req->flags & NOUVEAU_GEM_CPU_PREP_NOWAIT ? 0 : 30 * HZ);
	mutex_unlock(&cli->lock);
	return t;
}

static int ioctl_gem_cpu_fini(struct drm_device *dev, void *data, struct drm_file *file)
{
	return 0;
}

/* ---- VM ------------------------------------------------------------------------ */

static int ioctl_vm_init(struct drm_device *dev, void *data, struct drm_file *file)
{
	struct drm_nouveau_vm_init *req = data;
	struct nv_cli *cli = file->driver_priv;
	struct nvdrv_dev *nv = cli->nv;
	static atomic_t next_client = ATOMIC_INIT(0);
	int ret;

	if (!req->kernel_managed_size || (req->kernel_managed_addr | req->kernel_managed_size) & (SZ_2M - 1) ||
	    req->kernel_managed_addr + req->kernel_managed_size > (1ULL << 49))
		return -EINVAL;
	mutex_lock(&cli->lock);
	if (cli->vm_ready) {
		mutex_unlock(&cli->lock);
		return -ENOSPC;
	}
	mutex_lock(&nv->gsp_lock);
	ret = nv_fifo_init(nv);	/* golden GR context, nvdrv's CE channel */
	if (!ret)
		ret = nv_vmm_client_init(nv, &cli->vm, H_CLIENT_BASE + (atomic_inc_return(&next_client) & 0xffff),
					 req->kernel_managed_addr);
	mutex_unlock(&nv->gsp_lock);
	if (!ret) {
		cli->kernel_va = req->kernel_managed_addr;
		cli->kernel_size = req->kernel_managed_size;
		cli->vm_ready = true;
	}
	mutex_unlock(&cli->lock);
	return ret;
}

static bool in_kernel_range(struct nv_cli *cli, u64 addr, u64 range)
{
	return addr < cli->kernel_va + cli->kernel_size && addr + range > cli->kernel_va;
}

/* PTE for an unmapped page: sparse (reads 0, writes dropped) inside a sparse region. */
static u64 hole_pte(struct nv_cli *cli, u64 va)
{
	struct nv_vma *v;

	list_for_each_entry(v, &cli->vmas, link)
		if (!v->bo && va >= v->addr && va < v->addr + v->range)
			return NV_PTE_VOL;
	return 0;
}

static int write_ptes(struct nv_cli *cli, struct nv_vma *v, u64 addr, u64 range)
{
	struct nvdrv_dev *nv = cli->nv;
	u64 off;
	int ret = 0;

	for (off = 0; off < range && !ret; off += SZ_4K) {
		u64 va = addr + off, pte;

		if (!v) {
			pte = hole_pte(cli, va);
		} else if (!v->bo) {
			pte = NV_PTE_VOL;
		} else {
			u64 boff = v->bo_offset + (va - v->addr);

			if (v->bo->vram)
				pte = (v->bo->node.start + boff) >> 4;
			else
				pte = (v->bo->dma[boff >> PAGE_SHIFT] + (boff & ~PAGE_MASK)) >> 4 |
				      NV_PTE_APER_SYS_COH | NV_PTE_VOL;
			pte |= BIT_ULL(0) | (u64)v->kind << NV_PTE_KIND_SHIFT;
		}
		ret = nv_vmm_set_pte(nv, &cli->vm, va, pte);
	}
	return ret;
}

/* Remove BO mappings (sparse = false) or sparse regions (sparse = true) inside [addr, end). */
static int vma_remove(struct nv_cli *cli, u64 addr, u64 range, bool sparse)
{
	u64 end = addr + range;
	struct nv_vma *v, *tmp;

	list_for_each_entry_safe(v, tmp, &cli->vmas, link) {
		u64 vend = v->addr + v->range;

		if (!!v->bo == sparse || vend <= addr || v->addr >= end)
			continue;
		if (v->addr < addr && vend > end) {	/* split */
			struct nv_vma *t = kmemdup(v, sizeof(*v), GFP_KERNEL);

			if (!t)
				return -ENOMEM;
			t->addr = end;
			t->range = vend - end;
			t->bo_offset += end - v->addr;
			if (t->bo)
				drm_gem_object_get(&t->bo->base);
			list_add(&t->link, &v->link);
			v->range = addr - v->addr;
		} else if (v->addr < addr) {
			v->range = addr - v->addr;
		} else if (vend > end) {
			v->bo_offset += end - v->addr;
			v->range = vend - end;
			v->addr = end;
		} else {
			list_del(&v->link);
			if (v->bo)
				drm_gem_object_put(&v->bo->base);
			kfree(v);
		}
	}
	return 0;
}

static int vm_bind_op(struct drm_file *file, struct nv_cli *cli, const struct drm_nouveau_vm_bind_op *op)
{
	bool sparse = op->flags & DRM_NOUVEAU_VM_BIND_SPARSE;
	struct nv_vma *v = NULL;
	int ret;

	if (!op->range || (op->addr | op->range | op->bo_offset) & (SZ_4K - 1) ||
	    op->addr + op->range > cli->kernel_va || in_kernel_range(cli, op->addr, op->range))
		return -EINVAL;

	if (op->op == DRM_NOUVEAU_VM_BIND_OP_UNMAP) {
		/* the GPU may still use the range: no per-object tracking yet, so wait for idle */
		ret = cli_wait_idle(cli, false, 10 * HZ);
		if (ret)
			return ret;
		ret = vma_remove(cli, op->addr, op->range, false);
		if (!ret && sparse)
			ret = vma_remove(cli, op->addr, op->range, true);
		return ret ?: write_ptes(cli, NULL, op->addr, op->range);
	}
	if (op->op != DRM_NOUVEAU_VM_BIND_OP_MAP)
		return -EINVAL;

	v = kzalloc(sizeof(*v), GFP_KERNEL);
	if (!v)
		return -ENOMEM;
	v->addr = op->addr;
	v->range = op->range;
	v->kind = op->flags & 0xff;
	if (!sparse) {
		struct drm_gem_object *obj = drm_gem_object_lookup(file, op->handle);

		if (!obj || obj->funcs != &nv_bo_funcs || op->bo_offset + op->range > obj->size) {
			if (obj)
				drm_gem_object_put(obj);
			kfree(v);
			return -EINVAL;
		}
		v->bo = to_nv_bo(obj);
		v->bo_offset = op->bo_offset;
	}
	ret = write_ptes(cli, v, v->addr, v->range);
	if (ret) {
		if (v->bo)
			drm_gem_object_put(&v->bo->base);
		kfree(v);
		return ret;
	}
	list_add_tail(&v->link, &cli->vmas);
	return 0;
}

/* Wait for a syncobj point (in-fence) to signal. */
static int sync_wait(struct drm_file *file, const struct drm_nouveau_sync *s)
{
	struct dma_fence *f;
	u64 point = (s->flags & DRM_NOUVEAU_SYNC_TYPE_MASK) == DRM_NOUVEAU_SYNC_TIMELINE_SYNCOBJ ?
		    s->timeline_value : 0;
	long t;
	int ret;

	ret = drm_syncobj_find_fence(file, s->handle, point, DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT, &f);
	if (ret)
		return ret;
	t = dma_fence_wait_timeout(f, true, 30 * HZ);
	dma_fence_put(f);
	return t < 0 ? t : t == 0 ? -ETIMEDOUT : 0;
}

/* Attach fence to the signal syncobjs. */
static int sync_signal(struct drm_file *file, const struct drm_nouveau_sync *s, u32 n, struct dma_fence *fence)
{
	u32 i;

	for (i = 0; i < n; i++) {
		struct drm_syncobj *so = drm_syncobj_find(file, s[i].handle);

		if (!so)
			return -ENOENT;
		if ((s[i].flags & DRM_NOUVEAU_SYNC_TYPE_MASK) == DRM_NOUVEAU_SYNC_TIMELINE_SYNCOBJ) {
			struct dma_fence_chain *chain = dma_fence_chain_alloc();

			if (!chain) {
				drm_syncobj_put(so);
				return -ENOMEM;
			}
			drm_syncobj_add_point(so, chain, fence, s[i].timeline_value);
		} else {
			drm_syncobj_replace_fence(so, fence);
		}
		drm_syncobj_put(so);
	}
	return 0;
}

static void *copy_array(u64 uptr, u32 n, size_t size)
{
	if (!n)
		return NULL;
	return vmemdup_array_user(u64_to_user_ptr(uptr), n, size);
}

static int ioctl_vm_bind(struct drm_device *dev, void *data, struct drm_file *file)
{
	struct drm_nouveau_vm_bind *req = data;
	struct nv_cli *cli = file->driver_priv;
	struct nvdrv_dev *nv = cli->nv;
	struct drm_nouveau_vm_bind_op *ops = NULL;
	struct drm_nouveau_sync *waits = NULL, *sigs = NULL;
	struct dma_fence *stub;
	u32 i;
	int ret = 0;

	if (!cli->vm_ready)
		return -EINVAL;
	ops = copy_array(req->op_ptr, req->op_count, sizeof(*ops));
	waits = copy_array(req->wait_ptr, req->wait_count, sizeof(*waits));
	sigs = copy_array(req->sig_ptr, req->sig_count, sizeof(*sigs));
	if (IS_ERR(ops) || IS_ERR(waits) || IS_ERR(sigs)) {
		ret = -EFAULT;
		goto out;
	}
	for (i = 0; i < req->wait_count && !ret; i++)
		ret = sync_wait(file, &waits[i]);
	if (ret)
		goto out;
	mutex_lock(&cli->lock);
	mutex_lock(&nv->gsp_lock);
	for (i = 0; i < req->op_count && !ret; i++) {
		ret = vm_bind_op(file, cli, &ops[i]);
		pr_debug("nvdrv: vm_bind %s 0x%llx+0x%llx handle %u off 0x%llx flags 0x%x: %d\n",
			 ops[i].op ? "unmap" : "map", ops[i].addr, ops[i].range, ops[i].handle, ops[i].bo_offset,
			 ops[i].flags, ret);
	}
	if (req->op_count)
		ret = nv_vmm_flush(nv, &cli->vm) ?: ret;
	mutex_unlock(&nv->gsp_lock);
	mutex_unlock(&cli->lock);
	if (!ret) {
		stub = dma_fence_get_stub();
		ret = sync_signal(file, sigs, req->sig_count, stub);
		dma_fence_put(stub);
	}
out:
	if (!IS_ERR(ops))
		kvfree(ops);
	if (!IS_ERR(waits))
		kvfree(waits);
	if (!IS_ERR(sigs))
		kvfree(sigs);
	return ret;
}

/* ---- channels -------------------------------------------------------------------- */

static void uchan_free(struct nv_cli *cli, struct nv_uchan *u)
{
	struct nvdrv_dev *nv = cli->nv;

	uchan_drain(nv, u);
	mutex_lock(&nv->gsp_lock);
	nv_chan_destroy(nv, &u->ch);
	mutex_unlock(&nv->gsp_lock);
	kfree(u);
}

static int ioctl_channel_alloc(struct drm_device *dev, void *data, struct drm_file *file)
{
	struct drm_nouveau_channel_alloc *req = data;
	struct nv_cli *cli = file->driver_priv;
	struct nvdrv_dev *nv = cli->nv;
	struct nv_fifo *f = &nv->fifo;
	struct nv_uchan *u;
	int id, chid, ret;

	if (!cli->vm_ready)
		return -EINVAL;	/* VM_BIND uAPI only */
	if (req->fb_ctxdma_handle != ~0u || req->tt_ctxdma_handle != NOUVEAU_FIFO_ENGINE_GR)
		return -ENOSYS;	/* GR channels only for now (no transfer queue) */
	u = kzalloc(sizeof(*u), GFP_KERNEL);
	if (!u)
		return -ENOMEM;
	u->cli = cli;
	INIT_LIST_HEAD(&u->pending);
	INIT_LIST_HEAD(&u->active);
	u->fence_ctx = dma_fence_context_alloc(1);

	mutex_lock(&cli->lock);
	id = idr_alloc(&cli->chans, u, 0, UAPI_MAX_CHANS, GFP_KERNEL);
	if (id < 0) {
		mutex_unlock(&cli->lock);
		kfree(u);
		return id;
	}
	u->id = id;
	mutex_lock(&nv->gsp_lock);
	chid = nv_chid_get(nv);
	ret = chid < 0 ? chid : 0;
	if (!ret) {
		ret = nv_chan_create(nv, &u->ch, &cli->vm, H_UCHAN(id), chid, 0x01 /* GR0 */, f->gr_runlist,
				     f->gr_rl_base);
		if (ret && !u->ch.chid)
			nv_chid_put(nv, chid);
	}
	mutex_unlock(&nv->gsp_lock);
	if (ret) {
		idr_remove(&cli->chans, id);
		mutex_unlock(&cli->lock);
		mutex_lock(&nv->gsp_lock);
		nv_chan_destroy(nv, &u->ch);
		mutex_unlock(&nv->gsp_lock);
		kfree(u);
		return ret;
	}
	mutex_unlock(&cli->lock);
	req->channel = id;
	req->pushbuf_domains = NOUVEAU_GEM_DOMAIN_VRAM | NOUVEAU_GEM_DOMAIN_GART;
	req->notifier_handle = 0;
	req->nr_subchan = 0;
	drm_dbg(dev, "uapi: channel %d = chid %u (GR)\n", id, u->ch.chid);
	return 0;
}

static int ioctl_channel_free(struct drm_device *dev, void *data, struct drm_file *file)
{
	struct drm_nouveau_channel_free *req = data;
	struct nv_cli *cli = file->driver_priv;
	struct nv_uchan *u;

	mutex_lock(&cli->lock);
	u = idr_remove(&cli->chans, req->channel);
	mutex_unlock(&cli->lock);
	if (!u)
		return -ENOENT;
	uchan_free(cli, u);
	return 0;
}

static void push_dump(struct nv_cli *cli, u32 chid, u64 va, u32 len);
#define HOST_SEM_ADDR_LO	0x5c
#define HOST_SEM_EXECUTE_RELEASE_WFI	(0x1 | 1 << 20)	/* RELEASE, RELEASE_WFI, 32-bit payload */

static int ioctl_exec(struct drm_device *dev, void *data, struct drm_file *file)
{
	struct drm_nouveau_exec *req = data;
	struct nv_cli *cli = file->driver_priv;
	struct nvdrv_dev *nv = cli->nv;
	struct drm_nouveau_exec_push *push = NULL;
	struct drm_nouveau_sync *waits = NULL, *sigs = NULL;
	struct dma_fence *fence = NULL;
	struct nv_uchan *u;
	struct nv_chan *ch;
	u32 i, *gp, free, spins = 0;
	int ret = 0;

	if (req->push_count > UAPI_PUSH_MAX)
		return -EINVAL;
	push = copy_array(req->push_ptr, req->push_count, sizeof(*push));
	waits = copy_array(req->wait_ptr, req->wait_count, sizeof(*waits));
	sigs = copy_array(req->sig_ptr, req->sig_count, sizeof(*sigs));
	if (IS_ERR(push) || IS_ERR(waits) || IS_ERR(sigs)) {
		ret = -EFAULT;
		goto out;
	}
	for (i = 0; i < req->wait_count && !ret; i++)
		ret = sync_wait(file, &waits[i]);
	if (ret)
		goto out;

	mutex_lock(&cli->lock);
	u = idr_find(&cli->chans, req->channel);
	if (!u || u->dead) {
		ret = u ? -ENODEV : -ENOENT;
		goto unlock;
	}
	ch = &u->ch;
	/* room for the pushes and the fence entry */
	for (;;) {
		u32 get = ioread32(ch->userd_cpu + 0x88);

		free = (get - ch->gp_put - 1 + UAPI_GPFIFO_ENTRIES) % UAPI_GPFIFO_ENTRIES;
		if (free >= req->push_count + 1)
			break;
		if (++spins > 100000) {
			ret = -EBUSY;
			goto unlock;
		}
		usleep_range(20, 50);
	}
	for (i = 0; i < req->push_count; i++) {
		if (!push[i].va_len || push[i].va_len & 3 || (push[i].va_len >> 2) >= BIT(21) ||
		    push[i].va & 3 || push[i].va >> 40) {
			ret = -EINVAL;
			goto unlock;
		}
	}
	if (dump_push)
		for (i = 0; i < req->push_count; i++)
			push_dump(cli, ch->chid, push[i].va, push[i].va_len);
	gp = ch->gpfifo.cpu;
	for (i = 0; i < req->push_count; i++) {
		u64 va = push[i].va;

		gp[ch->gp_put * 2] = lower_32_bits(va) & ~3u;
		gp[ch->gp_put * 2 + 1] = upper_32_bits(va) | BIT(9) | (push[i].va_len >> 2) << 10 |
					 (push[i].flags & DRM_NOUVEAU_EXEC_PUSH_NO_PREFETCH ? BIT(31) : 0);
		ch->gp_put = (ch->gp_put + 1) % UAPI_GPFIFO_ENTRIES;
	}
	/* fence: host semaphore release after the work, then GP_PUT + doorbell (nv_kick) */
	{
		u32 sem[5] = { lower_32_bits(ch->sem_va), upper_32_bits(ch->sem_va), u->submitted + 1, 0,
			       HOST_SEM_EXECUTE_RELEASE_WFI };

		fence = uchan_fence(nv, u, u->submitted + 1);
		if (!fence) {
			ret = -ENOMEM;
			goto unlock;
		}
		u->submitted++;
		nv_push(ch, 0, HOST_SEM_ADDR_LO, sem, 5);
		nv_kick(nv, ch);
	}
	dma_fence_put(u->last);
	u->last = dma_fence_get(fence);
unlock:
	mutex_unlock(&cli->lock);
	if (!ret)
		ret = sync_signal(file, sigs, req->sig_count, fence);
out:
	dma_fence_put(fence);
	if (!IS_ERR(push))
		kvfree(push);
	if (!IS_ERR(waits))
		kvfree(waits);
	if (!IS_ERR(sigs))
		kvfree(sigs);
	return ret;
}


/* Print a submitted push buffer (debug aid; system-memory objects only). */
static void push_dump(struct nv_cli *cli, u32 chid, u64 va, u32 len)
{
	struct nv_vma *v;

	list_for_each_entry(v, &cli->vmas, link) {
		u64 off, i;

		if (!v->bo || va < v->addr || va + len > v->addr + v->range)
			continue;
		if (v->bo->vram) {
			pr_info("nvdrv: chid %u push 0x%llx+%u is in VRAM, not dumped\n", chid, va, len);
			return;
		}
		off = v->bo_offset + va - v->addr;
		pr_info("nvdrv: chid %u push 0x%llx, %u dwords:\n", chid, va, len / 4);
		for (i = 0; i < len / 4; i += 8) {
			u32 d[8] = {}, j;

			for (j = 0; j < 8 && i + j < len / 4; j++) {
				u64 o = off + (i + j) * 4;
				u32 *pg = kmap_local_page(v->bo->pages[o >> PAGE_SHIFT]);

				d[j] = pg[(o & ~PAGE_MASK) / 4];
				kunmap_local(pg);
			}
			pr_info("  %04llx: %08x %08x %08x %08x %08x %08x %08x %08x\n", i, d[0], d[1], d[2], d[3],
				d[4], d[5], d[6], d[7]);
		}
		return;
	}
	pr_info("nvdrv: chid %u push 0x%llx+%u not in any mapping\n", chid, va, len);
}

/* ---- GETPARAM ---------------------------------------------------------------------- */

static void gr_units(struct nvdrv_dev *nv)
{
	struct nv_uapi *ua = &nv->uapi;
	__le32 p[6] = {};
	u32 mask, gpc;

	if (ua->gpcs)
		return;
	mutex_lock(&nv->gsp_lock);
	if (!nv_rm_control(nv, nv->rm.h_client, nv->rm.h_subdevice, 0x2080122a /* GR_GET_GPC_MASK */, p, 24)) {
		mask = le32_to_cpu(p[4]);
		for (gpc = 0; gpc < 32; gpc++) {
			if (!(mask & BIT(gpc)))
				continue;
			memset(p, 0, sizeof(p));
			p[4] = cpu_to_le32(gpc);
			if (!nv_rm_control(nv, nv->rm.h_client, nv->rm.h_subdevice, 0x2080122b /* GET_TPC_MASK */,
					   p, 24))
				ua->tpcs += hweight32(le32_to_cpu(p[5]));
			ua->gpcs++;
		}
	}
	mutex_unlock(&nv->gsp_lock);
}

static int ioctl_getparam(struct drm_device *dev, void *data, struct drm_file *file)
{
	struct drm_nouveau_getparam *req = data;
	struct nvdrv_dev *nv = dev_nv(dev);

	switch (req->param) {
	case NOUVEAU_GETPARAM_CHIPSET_ID:
		req->value = nv->chipset;
		break;
	case NOUVEAU_GETPARAM_PCI_VENDOR:
		req->value = nv->pdev->vendor;
		break;
	case NOUVEAU_GETPARAM_PCI_DEVICE:
		req->value = nv->pdev->device;
		break;
	case NOUVEAU_GETPARAM_BUS_TYPE:
		req->value = 2;	/* PCIe */
		break;
	case NOUVEAU_GETPARAM_FB_SIZE:
		req->value = nv->mem.ready ? nv->mem.size : nv->gsp_info.fb_length / 4 * 3;
		break;
	case NOUVEAU_GETPARAM_AGP_SIZE:
		req->value = (u64)totalram_pages() << PAGE_SHIFT >> 1;
		break;
	case NOUVEAU_GETPARAM_VM_VRAM_BASE:
		req->value = 0;
		break;
	case NOUVEAU_GETPARAM_PTIMER_TIME: {
		u32 hi, lo;

		do {
			hi = nv_rd32(nv, 0x9410);
			lo = nv_rd32(nv, 0x9400);
		} while (hi != nv_rd32(nv, 0x9410));
		req->value = (u64)hi << 32 | lo;
		break;
	}
	case NOUVEAU_GETPARAM_HAS_BO_USAGE:
	case NOUVEAU_GETPARAM_HAS_PAGEFLIP:
	case NOUVEAU_GETPARAM_HAS_VMA_TILEMODE:
		req->value = 1;
		break;
	case NOUVEAU_GETPARAM_GRAPH_UNITS:
		gr_units(nv);
		req->value = nv->uapi.gpcs | (u64)nv->uapi.tpcs << 8;
		break;
	case NOUVEAU_GETPARAM_EXEC_PUSH_MAX:
		req->value = UAPI_PUSH_MAX;
		break;
	case NOUVEAU_GETPARAM_VRAM_BAR_SIZE:
		req->value = nv->bar1.size > ALIGN(NV_FB_MAX + SZ_64K, SZ_2M) ?
			     nv->bar1.size - ALIGN(NV_FB_MAX + SZ_64K, SZ_2M) : 0;
		break;
	case NOUVEAU_GETPARAM_VRAM_USED:
		req->value = nv->mem.used;
		break;
	default:
		return -EINVAL;
	}
	return 0;
}

static int ioctl_zcull(struct drm_device *dev, void *data, struct drm_file *file)
{
	return -EOPNOTSUPP;
}

/* ---- NVIF (the parts NVK uses) ----------------------------------------------------- */

struct nvif_hdr {
	u8 version, type, pad02[4], owner, route;
	u64 token, object;
};

struct nvif_new {
	u8 version, pad01[6], route;
	u64 token, object;
	u32 handle;
	s32 oclass;
};

struct nvif_sclass {
	u8 version, count, pad02[6];
	struct { s32 oclass; s16 minver, maxver; } oclass[];
};

struct nvif_mthd {
	u8 version, method, pad02[6];
};

struct nvif_device_info {
	u8 version, platform;
	u16 chipset;
	u8 revision, family, pad06[2];
	u64 ram_size, ram_user;
	char chip[16], name[64];
};

#define NVIF_SCLASS	0x01
#define NVIF_NEW	0x02
#define NVIF_DEL	0x03
#define NVIF_MTHD	0x04
#define NVIF_ROUTE_HIDDEN	0xff	/* abi16: token = channel */

static int nvif_engine_obj(struct nv_cli *cli, struct nv_uchan *u, s32 oclass)
{
	struct nvdrv_dev *nv = cli->nv;
	struct { __le32 version, engine_type; } ce = { cpu_to_le32(1), 0 };
	bool known = false;
	int i, ret;

	for (i = 0; i < ARRAY_SIZE(gr_classes); i++)
		known |= gr_classes[i] == oclass;
	if (!known)
		return -EINVAL;
	if (oclass == AMPERE_DMA_COPY_B) {
		if (!nv->fifo.gr_ce)
			return -ENODEV;
		ce.engine_type = cpu_to_le32(0x09 /* COPY0 */ + nv->fifo.gr_ce - 1);
	}
	mutex_lock(&nv->gsp_lock);
	/*
	 * Like nouveau (r535_gr_chan_new), promote the GR context right before the first
	 * object: RM initialises it on that allocation.
	 */
	ret = 0;
	if (!u->gr_ctx) {
		ret = nv_gr_chan_ctx(nv, &u->ch);
		u->gr_ctx = !ret;
	}
	if (!ret)
		ret = nv_rm_alloc(nv, cli->vm.h_client, u->ch.handle, H_UOBJ(u->ch.chid, u->nobj), oclass,
			  oclass == AMPERE_DMA_COPY_B ? &ce : NULL, oclass == AMPERE_DMA_COPY_B ? sizeof(ce) : 0);
	mutex_unlock(&nv->gsp_lock);
	if (!ret)
		u->nobj++;
	pr_debug("nvdrv: chid %u: object %04x: %d\n", u->ch.chid, oclass, ret);
	return ret;
}

static int nvif_ioctl(struct drm_file *file, void *buf, u32 size)
{
	struct nv_cli *cli = file->driver_priv;
	struct nvdrv_dev *nv = cli->nv;
	struct nvif_hdr *h = buf;
	void *body = h + 1;
	u32 left = size - sizeof(*h);
	struct nv_uchan *u = NULL;
	int ret, i;

	if (size < sizeof(*h) || h->version)
		return -EINVAL;
	if (h->route == NVIF_ROUTE_HIDDEN) {
		mutex_lock(&cli->lock);
		u = idr_find(&cli->chans, (int)h->token);
		mutex_unlock(&cli->lock);
		if (!u)
			return -ENOENT;
	}
	switch (h->type) {
	case NVIF_SCLASS: {
		struct nvif_sclass *sc = body;
		u8 max;

		if (!u || left < sizeof(*sc))
			return -EINVAL;
		max = min_t(u32, sc->count, (left - sizeof(*sc)) / sizeof(sc->oclass[0]));
		for (i = 0; i < ARRAY_SIZE(gr_classes) && i < max; i++) {
			sc->oclass[i].oclass = gr_classes[i];
			sc->oclass[i].minver = sc->oclass[i].maxver = -1;
		}
		sc->count = ARRAY_SIZE(gr_classes);
		return 0;
	}
	case NVIF_NEW: {
		struct nvif_new *n = body;

		if (left < sizeof(*n))
			return -EINVAL;
		if (!u)
			return n->oclass == 0x80 /* NV_DEVICE */ ? 0 : -EINVAL;
		mutex_lock(&cli->lock);
		ret = nvif_engine_obj(cli, u, n->oclass);
		mutex_unlock(&cli->lock);
		return ret;
	}
	case NVIF_DEL:
		return 0;	/* engine objects go with their channel */
	case NVIF_MTHD: {
		struct nvif_mthd *m = body;
		struct nvif_device_info *info = (void *)(m + 1);

		if (u || left < sizeof(*m) + sizeof(*info) || m->method != 0x00 /* NV_DEVICE_V0_INFO */)
			return -EINVAL;
		memset(info, 0, sizeof(*info));
		info->platform = 0x03;		/* PCIE */
		info->chipset = nv->chipset;
		info->revision = nv->boot0 & 0xff;
		info->family = 0x0d;		/* AMPERE */
		info->ram_size = nv->gsp_info.fb_length;
		info->ram_user = nv->mem.ready ? nv->mem.size : nv->gsp_info.fb_length / 4 * 3;
		snprintf(info->chip, sizeof(info->chip), "GA10%x", nv->chipset & 0xf);	/* 0x174 = GA104 */
		strscpy(info->name, nv->gsp_info.name[0] ? nv->gsp_info.name : "NVIDIA GPU", sizeof(info->name));
		return 0;
	}
	default:
		return -EINVAL;
	}
}

/* DRM_NOUVEAU_NVIF has a variable size; nouveau handles it outside drm_ioctl() too. */
long nv_uapi_ioctl(struct file *filp, unsigned int cmd, unsigned long arg)
{
	struct drm_file *file = filp->private_data;
	u32 size = _IOC_SIZE(cmd);
	void *buf;
	long ret;

	if (!nv_nvk || _IOC_TYPE(cmd) != DRM_IOCTL_BASE || _IOC_NR(cmd) != DRM_COMMAND_BASE + DRM_NOUVEAU_NVIF)
		return drm_ioctl(filp, cmd, arg);
	if (!file->driver_priv || size < sizeof(struct nvif_hdr) || size > SZ_4K)
		return -EINVAL;
	buf = memdup_user((void __user *)arg, size);
	if (IS_ERR(buf))
		return PTR_ERR(buf);
	ret = nvif_ioctl(file, buf, size);
	if (!ret && (_IOC_DIR(cmd) & _IOC_READ) && copy_to_user((void __user *)arg, buf, size))
		ret = -EFAULT;
	kfree(buf);
	return ret;
}

/* ---- file open / close ------------------------------------------------------------- */

int nv_uapi_open(struct drm_device *dev, struct drm_file *file)
{
	struct nv_cli *cli;

	if (!nv_nvk)
		return 0;
	cli = kzalloc(sizeof(*cli), GFP_KERNEL);
	if (!cli)
		return -ENOMEM;
	cli->nv = dev_nv(dev);
	mutex_init(&cli->lock);
	INIT_LIST_HEAD(&cli->vmas);
	INIT_LIST_HEAD(&cli->vm.pts);
	idr_init(&cli->chans);
	file->driver_priv = cli;
	return 0;
}

void nv_uapi_postclose(struct drm_device *dev, struct drm_file *file)
{
	struct nv_cli *cli = file->driver_priv;
	struct nvdrv_dev *nv;
	struct nv_uchan *u;
	struct nv_vma *v, *tmp;
	int id;

	if (!cli)
		return;
	nv = cli->nv;
	idr_for_each_entry(&cli->chans, u, id)
		uchan_free(cli, u);
	idr_destroy(&cli->chans);
	list_for_each_entry_safe(v, tmp, &cli->vmas, link) {
		if (v->bo)
			drm_gem_object_put(&v->bo->base);
		kfree(v);
	}
	if (cli->vm_ready) {
		mutex_lock(&nv->gsp_lock);
		nv_vmm_client_fini(nv, &cli->vm);
		mutex_unlock(&nv->gsp_lock);
	}
	mutex_destroy(&cli->lock);
	kfree(cli);
	file->driver_priv = NULL;
}

#define IOCTL(n, f) DRM_IOCTL_DEF_DRV(NOUVEAU_##n, f, DRM_RENDER_ALLOW)
static const struct drm_ioctl_desc nv_uapi_ioctls[] = {
	IOCTL(GETPARAM, ioctl_getparam),
	IOCTL(CHANNEL_ALLOC, ioctl_channel_alloc),
	IOCTL(CHANNEL_FREE, ioctl_channel_free),
	IOCTL(VM_INIT, ioctl_vm_init),
	IOCTL(VM_BIND, ioctl_vm_bind),
	IOCTL(EXEC, ioctl_exec),
	IOCTL(GET_ZCULL_INFO, ioctl_zcull),
	IOCTL(GEM_NEW, ioctl_gem_new),
	IOCTL(GEM_CPU_PREP, ioctl_gem_cpu_prep),
	IOCTL(GEM_CPU_FINI, ioctl_gem_cpu_fini),
	IOCTL(GEM_INFO, ioctl_gem_info),
};
const struct drm_ioctl_desc *const nv_uapi_ioctls_ptr = nv_uapi_ioctls;
const int nv_uapi_num_ioctls = ARRAY_SIZE(nv_uapi_ioctls);
static_assert(ARRAY_SIZE(nv_uapi_ioctls) == DRM_NOUVEAU_GEM_INFO + 1);

void nv_uapi_init(struct nvdrv_dev *nv)
{
	struct nv_uapi *ua = &nv->uapi;

	spin_lock_init(&ua->lock);
	INIT_LIST_HEAD(&ua->active);
	mutex_init(&ua->bar1_lock);
	hrtimer_setup(&ua->timer, uapi_poll, CLOCK_MONOTONIC, HRTIMER_MODE_REL);
}

void nv_uapi_fini(struct nvdrv_dev *nv)
{
	struct nv_uapi *ua = &nv->uapi;

	hrtimer_cancel(&ua->timer);
	if (ua->bar1_ready && drm_mm_clean(&ua->bar1_mm))
		drm_mm_takedown(&ua->bar1_mm);
	ua->bar1_ready = false;
	ua->gpcs = ua->tpcs = 0;
}
