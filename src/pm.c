// SPDX-License-Identifier: GPL-2.0
/*
 * System suspend/resume (S3) with GSP-RM preserving its state, as NVIDIA's driver does
 * (gpu_suspend.c gpuPowerManagementEnter/Resume, kgspUnloadRm SR_SUSPEND, kgspBootstrap
 * SR_RESUME, fbsr_gm107.c _fbsrInitGsp) and nouveau's r570 path (gsp.c r535_gsp_fini /
 * tu102_gsp_fini/init, r570/fbsr.c).
 *
 * Suspend:
 *   1. the display is off (DRM suspend helper); its channels are freed with RM's
 *      PRESERVE_HW free flag (nv_evo_suspend)
 *   2. no submissions, GPU idle (nv_uapi_pm)
 *   3. channel scheduling off (INTERNAL_FIFO_TOGGLE_ACTIVE_CHANNEL_SCHEDULING): every channel
 *      is preempted, GR contexts and RAMFCs written back
 *   4. nvdrv's own VRAM - the carve-out minus the unused VRAM pool - copied to system pages
 *      by a new kernel copy channel (made after the toggle, so it runs), then freed
 *   5. GspFwSRMeta + SR data (radix-3 in sysmem, WPR2 start to FRTS end) and the FBSR
 *      buffer for RM's VRAM (end of the client reserve to end of the non-WPR heap, plus the
 *      VGA workspace), handed to RM as a memory list (ALLOC_MEMORY RPC) and FBSR_INIT
 *   6. UNLOADING_GUEST_DRIVER (PM transition, level 3), FWSEC-SB, Booter-Unload with the
 *      SR meta: GSP-RM saves its VRAM into the FBSR buffer, Booter saves WPR2
 * Resume:
 *   1. GFW boot done, sysmem flush buffer, Booter-Load with the SR meta, INIT_DONE
 *      (Booter restores WPR2 and FRTS, GSP-RM its VRAM; no FWSEC-FRTS, no queue reset)
 *   2. nvdrv's VRAM back, CPU writes: the non-pool carve-out (BAR1 page tables among it)
 *      through PRAMIN, then the pool through its 1:1 BAR1 mapping
 *   3. scheduling on, a new kernel copy channel, interrupts, display (DRM resume helper)
 */
#include <linux/delay.h>
#include <linux/highmem.h>
#include <linux/iopoll.h>
#include <linux/pci.h>
#include <linux/slab.h>
#include <linux/unaligned.h>

#include "nvdrv.h"
#include "nv_regs.h"

#define NV_RPC_ALLOC_MEMORY				4
#define NV01_MEMORY_LIST_SYSTEM				0x81
#define NV2080_CTRL_CMD_INTERNAL_FBSR_INIT		0x20800ac2
#define NV2080_CTRL_CMD_INTERNAL_FIFO_TOGGLE_ACTIVE_CHANNEL_SCHEDULING	0x20800ac3
#define NVDRV_H_FBSR_MEM				0xd000fb5e

#define GSP_FW_SR_META_MAGIC		0x8a3bb9e6c6c39d93ULL
#define GSP_FW_SR_META_REVISION		2

/* GspFwSRMeta (arch/nvalloc/common/inc/gsp/gsp_fw_sr_meta.h), one 256-byte DMA chunk */
struct gsp_fw_sr_meta {
	__le64 magic, revision;
	__le64 sysmem_addr_sr_data, size_sr_data;
	__le32 internal[32];
	__le32 flags, subrevision;
	__le32 padding[22];
};
static_assert(sizeof(struct gsp_fw_sr_meta) == 256);

/* ---- system pages for a VRAM range ---------------------------------------------- */

static void region_free(struct nvdrv_dev *nv, struct nv_pm_region *r)
{
	u32 i;

	if (r->mapped)
		dma_unmap_sgtable(&nv->pdev->dev, &r->sgt, DMA_BIDIRECTIONAL, 0);
	if (r->sgt.sgl)
		sg_free_table(&r->sgt);
	for (i = 0; r->pages && i < r->npages; i++)
		if (r->pages[i])
			__free_page(r->pages[i]);
	kvfree(r->pages);
	memset(r, 0, sizeof(*r));
}

static int region_alloc(struct nvdrv_dev *nv, struct nv_pm_region *r, u64 vram, u64 size)
{
	u32 i;
	int ret;

	r->vram = vram;
	r->size = size;
	r->npages = DIV_ROUND_UP(size, PAGE_SIZE);
	r->pages = kvcalloc(r->npages, sizeof(*r->pages), GFP_KERNEL);
	if (!r->pages)
		return -ENOMEM;
	for (i = 0; i < r->npages; i++) {
		r->pages[i] = alloc_page(GFP_KERNEL | __GFP_NORETRY | __GFP_NOWARN);
		if (!r->pages[i]) {
			ret = -ENOMEM;
			goto err;
		}
	}
	ret = sg_alloc_table_from_pages(&r->sgt, r->pages, r->npages, 0, (size_t)r->npages << PAGE_SHIFT,
					GFP_KERNEL);
	if (!ret)
		ret = dma_map_sgtable(&nv->pdev->dev, &r->sgt, DMA_BIDIRECTIONAL, 0);
	if (ret)
		goto err;
	r->mapped = true;
	return 0;
err:
	region_free(nv, r);
	return ret;
}

static void regions_free(struct nvdrv_dev *nv)
{
	struct nv_pm *pm = &nv->pm;
	u32 i;

	for (i = 0; i < pm->nregs; i++)
		region_free(nv, &pm->regs[i]);
	kfree(pm->regs);
	pm->regs = NULL;
	pm->nregs = 0;
}

/*
 * nvdrv's VRAM: the carve-out [vram_next, vram_top) without the unused part of the pool:
 * below the pool, above it, and the pool's allocated nodes (adjacent ones merged).
 */
static int regions_collect(struct nvdrv_dev *nv)
{
	struct nv_evo *e = &nv->evo;
	struct nv_mem *mem = &nv->mem;
	struct nv_pm *pm = &nv->pm;
	struct drm_mm_node *node;
	u64 lo = e->vram_next, hi = e->vram_top, cur_s = 0, cur_e = 0;
	u32 cap = 16, n = 0;
	struct nv_pm_region *r;

	r = kcalloc(cap, sizeof(*r), GFP_KERNEL);
	if (!r)
		return -ENOMEM;
#define ADD(s, sz) do {								\
		if ((sz) == 0)							\
			break;							\
		if (n == cap) {							\
			struct nv_pm_region *t = krealloc_array(r, cap * 2, sizeof(*r), GFP_KERNEL); \
			if (!t) {						\
				kfree(r);					\
				return -ENOMEM;					\
			}							\
			memset(t + cap, 0, cap * sizeof(*r));			\
			r = t;							\
			cap *= 2;						\
		}								\
		r[n].vram = (s);						\
		r[n].size = (sz);						\
		n++;								\
	} while (0)

	if (!mem->ready || mem->base < lo || mem->base + mem->size > hi) {
		ADD(lo, hi - lo);
	} else {
		ADD(lo, mem->base - lo);
		mutex_lock(&mem->lock);
		drm_mm_for_each_node(node, &mem->mm) {
			if (node->start == cur_e) {
				cur_e += node->size;
				continue;
			}
			if (cur_e > cur_s)
				ADD(cur_s, cur_e - cur_s);
			cur_s = node->start;
			cur_e = node->start + node->size;
		}
		mutex_unlock(&mem->lock);
		if (cur_e > cur_s)
			ADD(cur_s, cur_e - cur_s);
		ADD(mem->base + mem->size, hi - (mem->base + mem->size));
	}
#undef ADD
	pm->regs = r;
	pm->nregs = n;
	return 0;
}

static bool in_pool(struct nvdrv_dev *nv, const struct nv_pm_region *r)
{
	return nv->mem.ready && r->vram >= nv->mem.base && r->vram + r->size <= nv->mem.base + nv->mem.size;
}

/* CPU copy of a saved region back into VRAM: BAR1 for the 1:1-mapped pool, else PRAMIN. */
static int region_restore(struct nvdrv_dev *nv, struct nv_pm_region *r)
{
	void __iomem *bar1 = NULL;
	u64 off = 0;
	u32 i;

	if (in_pool(nv, r) && nv->bar1.pool_mapped)
		bar1 = ioremap_wc(pci_resource_start(nv->pdev, 1) + r->vram, r->size);
	for (i = 0; i < r->npages && off < r->size; i++) {
		u64 n = min_t(u64, PAGE_SIZE, r->size - off);
		void *src = kmap_local_page(r->pages[i]);

		if (bar1)
			memcpy_toio(bar1 + off, src, n);
		else
			nv_vram_write(nv, r->vram + off, src, n);
		kunmap_local(src);
		off += n;
	}
	if (bar1) {
		wmb();
		(void)ioread32(bar1);
		iounmap(bar1);
	}
	return 0;
}

/* ---- RM side ------------------------------------------------------------------- */

static int toggle_scheduling(struct nvdrv_dev *nv, bool disable)
{
	u8 p = disable;	/* NV2080_CTRL_CMD_INTERNAL_FIFO_TOGGLE_ACTIVE_CHANNEL_SCHEDULING_PARAMS */

	return nv_rm_control(nv, nv->gsp_info.h_internal_client, nv->gsp_info.h_internal_subdevice,
			     NV2080_CTRL_CMD_INTERNAL_FIFO_TOGGLE_ACTIVE_CHANNEL_SCHEDULING, &p, sizeof(p));
}

/* rpc_alloc_memory_v13_01: NV01_MEMORY_LIST_SYSTEM over @r's pages (memdescSendMemDescToGSP). */
static int memlist_sysmem(struct nvdrv_dev *nv, u32 handle, struct nv_pm_region *r)
{
	u32 pages = r->size >> 12, len = 56 + pages * 8, i = 0;
	struct sg_dma_page_iter it;
	u8 *m;
	int ret;

	if (pages > 0xffff)	/* pteDesc.length is 16 bits */
		return -E2BIG;
	m = kvzalloc(len, GFP_KERNEL);
	if (!m)
		return -ENOMEM;
	put_unaligned_le32(nv->rm.h_client, m + 0);
	put_unaligned_le32(nv->rm.h_device, m + 4);
	put_unaligned_le32(handle, m + 8);
	put_unaligned_le32(NV01_MEMORY_LIST_SYSTEM, m + 12);
	/* NVOS02_FLAGS: PHYSICALITY NONCONTIGUOUS (7:4 = 1), LOCATION PCI (0), MAPPING NO_MAP (31:30 = 1) */
	put_unaligned_le32(1 << 4 | 1 << 30, m + 16);
	put_unaligned_le64(r->size, m + 32);
	put_unaligned_le32(pages, m + 40);
	put_unaligned_le32(pages << 16, m + 48);	/* idr 0, length */
	for_each_sgtable_dma_page(&r->sgt, &it, 0) {
		dma_addr_t a = sg_page_iter_dma_address(&it);
		u32 k;

		for (k = 0; k < PAGE_SIZE / SZ_4K && i < pages; k++, i++)
			put_unaligned_le64((a + k * SZ_4K) >> 12, m + 56 + i * 8);
		if (i >= pages)
			break;
	}
	ret = nv_rpc_call_large(nv, NV_RPC_ALLOC_MEMORY, m, len);
	kvfree(m);
	return ret;
}

static void pm_buffers_free(struct nvdrv_dev *nv)
{
	struct nv_pm *pm = &nv->pm;

	nv_radix3_free(nv, &pm->sr_data);
	if (pm->sr_meta.cpu)
		nv_dma_buf_free(nv, &pm->sr_meta);
	region_free(nv, &pm->fbsr);
	regions_free(nv);
}

/* GspFwSRMeta + SR data, the FBSR buffer, FBSR_INIT (kgspPrepareSuspendResumeData, _fbsrInitGsp). */
static int rm_suspend_prepare(struct nvdrv_dev *nv)
{
	struct device *dev = &nv->pdev->dev;
	const struct gsp_fw_wpr_meta *w = nv->wpr_meta;
	struct nv_pm *pm = &nv->pm;
	struct gsp_fw_sr_meta *sr;
	struct { __le32 h_client, h_sys_mem; u8 gcoff, pad[7]; __le64 sr_meta; } fi = {};
	u64 non_wpr_end = le64_to_cpu(w->non_wpr_heap_offset) + le64_to_cpu(w->non_wpr_heap_size);
	u64 sr_size = le64_to_cpu(w->frts_offset) + le64_to_cpu(w->frts_size) - non_wpr_end;
	u64 fbsr_size = non_wpr_end - nv->evo.vram_top + le64_to_cpu(w->vga_workspace_size);
	int ret;

	ret = nv_dma_buf_alloc(nv, &pm->sr_meta, SZ_4K);
	if (!ret)
		ret = nv_radix3_build(nv, &pm->sr_data, NULL, sr_size);
	if (ret)
		return ret;
	sr = pm->sr_meta.cpu;
	memset(sr, 0, sizeof(*sr));
	sr->magic = cpu_to_le64(GSP_FW_SR_META_MAGIC);
	sr->revision = cpu_to_le64(GSP_FW_SR_META_REVISION);
	sr->sysmem_addr_sr_data = cpu_to_le64(pm->sr_data.root);
	sr->size_sr_data = cpu_to_le64(sr_size);
	sr->flags = cpu_to_le32(w->flags);
	wmb();

	ret = region_alloc(nv, &pm->fbsr, 0, ALIGN(fbsr_size, PAGE_SIZE));
	if (ret)
		return ret;
	ret = memlist_sysmem(nv, NVDRV_H_FBSR_MEM, &pm->fbsr);
	if (ret) {
		dev_err(dev, "pm: FBSR memory list: %d\n", ret);
		return ret;
	}
	fi.h_client = cpu_to_le32(nv->rm.h_client);
	fi.h_sys_mem = cpu_to_le32(NVDRV_H_FBSR_MEM);
	fi.sr_meta = cpu_to_le64(pm->sr_meta.dma);
	ret = nv_rm_control(nv, nv->gsp_info.h_internal_client, nv->gsp_info.h_internal_subdevice,
			    NV2080_CTRL_CMD_INTERNAL_FBSR_INIT, &fi, sizeof(fi));
	nv_rm_free(nv, nv->rm.h_client, NVDRV_H_FBSR_MEM);
	dev_info(dev, "pm: SR data %llu MiB, FBSR %llu MiB: %d\n", sr_size >> 20, fbsr_size >> 20, ret);
	return ret;
}

/* ---- suspend / resume ------------------------------------------------------------ */

/*
 * The display objects are new (and after a resume the display came back from a reset: no SOR
 * assignment, no trained link, RM wants DisplayPort library mode again). Power-on then does
 * the full bring-up, display-common object included.
 */
static void disp_forget_link(struct nvdrv_dev *nv)
{
	u32 i;

	for (i = 0; i < nv->disp.ndpy; i++) {
		nv->disp.dpy[i].link_lanes = 0;
		nv->disp.dpy[i].sor = ~0u;
	}
	nv->disp.manual_dp = false;
}

int nv_pm_suspend(struct nvdrv_dev *nv)
{
	struct device *dev = &nv->pdev->dev;
	struct nv_pm *pm = &nv->pm;
	bool sched_off = false;
	u64 bytes = 0;
	u32 i;
	int ret;

	if (pm->suspended)
		return 0;
	pm->t_suspend = ktime_get();
	cancel_work_sync(&nv->disp.hpd_work);
	ret = nv_uapi_pm(nv, true);
	if (ret) {
		dev_err(dev, "pm: GPU did not go idle: %d\n", ret);
		return ret;
	}
	mutex_lock(&nv->gsp_lock);
	if (!nv->gsp_running) {
		ret = -ENODEV;
		goto err_unlock;
	}
	nv_evo_suspend(nv);

	/*
	 * Kernel RM's order (kfifoStateUnload, then the FBSR save on a new CeUtils channel):
	 * disable scheduling of the active channels first. That preempts them, so GR contexts
	 * and RAMFCs are written back to VRAM and the snapshot holds them current (a resident
	 * context is only saved on preemption; restored stale, the channel takes a GR
	 * exception). Channels allocated after it still run: the kernel channels are made anew
	 * for the copy.
	 */
	ret = toggle_scheduling(nv, true);
	sched_off = !ret;
	if (nv->chan.handle)
		nv_chan_destroy(nv, &nv->chan);
	if (!ret)
		ret = nv_fifo_init(nv);	/* a new kernel copy channel does the saving */
	if (!ret)
		ret = regions_collect(nv);
	for (i = 0; !ret && i < pm->nregs; i++) {
		struct nv_pm_region *r = &pm->regs[i];
		u64 vram = r->vram, size = r->size;

		ret = region_alloc(nv, r, vram, size);
		if (!ret)
			ret = nv_ce_copy_to_sg(nv, vram, &r->sgt, size);
		bytes += size;
	}
	if (ret) {
		dev_err(dev, "pm: saving VRAM failed: %d\n", ret);
		goto err_unlock;
	}
	/*
	 * The copy channel ran while the snapshot was taken (its own RAMFC is in it, from before
	 * the copies: restored, the channel would replay them into the freed buffers). Made new
	 * on resume. The kernel GR channel stays: it is preempted and holds the golden context's
	 * global buffers that every client GR channel maps.
	 */
	if (nv->chan.handle)
		nv_chan_destroy(nv, &nv->chan);
	ret = rm_suspend_prepare(nv);
	if (!ret) {
		msleep(200);	/* nouveau r535_gsp_fini: without it channels reset after resume */
		nv_bar1_detach(nv);
		ret = nv_gsp_suspend_locked(nv, nv->pm.sr_meta.dma);
	}
	if (ret) {
		dev_err(dev, "pm: GSP-RM suspend failed: %d\n", ret);
		goto err_unlock;
	}
	pm->suspended = true;
	mutex_unlock(&nv->gsp_lock);
	nv_intr_fini(nv);	/* interrupts stay on until GSP-RM is down (nouveau keeps them) */
	dev_info(dev, "pm: suspended: %u regions, %llu MiB of VRAM saved, %lld ms\n", pm->nregs, bytes >> 20,
		 ktime_ms_delta(ktime_get(), pm->t_suspend));
	return 0;

err_unlock:
	/* GSP-RM still up (the unload is the last step): back to running as before */
	if (nv->gsp_running) {
		if (sched_off && toggle_scheduling(nv, false))
			dev_warn(dev, "pm: scheduling on after failed suspend failed\n");
		nv_bar1_resume(nv);
		disp_forget_link(nv);
		if (nv_fifo_init(nv) || nv_evo_init(nv))
			dev_warn(dev, "pm: recovery after failed suspend incomplete\n");
	}
	mutex_unlock(&nv->gsp_lock);
	pm_buffers_free(nv);
	nv_uapi_pm(nv, false);
	return ret;
}

int nv_pm_resume(struct nvdrv_dev *nv)
{
	struct device *dev = &nv->pdev->dev;
	struct nv_pm *pm = &nv->pm;
	ktime_t t0 = ktime_get();
	bool done;
	u32 gfw, i;
	int ret;

	if (!pm->suspended)
		return 0;
	mutex_lock(&nv->gsp_lock);
	nv->evo.pramin_base = ~0ULL;	/* the window register lost its value */
	if (read_poll_timeout(nv_gfw_boot_completed, done, done, 1000, 2050000, false, nv, &gfw))
		dev_warn(dev, "pm: GFW boot not complete (0x%x)\n", gfw);
	ret = nv_gsp_resume_locked(nv, pm->sr_meta.dma);
	if (ret) {
		dev_err(dev, "pm: GSP-RM resume failed: %d\n", ret);
		goto out;
	}
	/*
	 * Carve-out outside the pool first (PRAMIN): BAR1 page tables are among it. Only after
	 * GSP-RM is up: written before its boot, RM removes a GR channel (RC 45) on resume.
	 */
	for (i = 0; i < pm->nregs; i++)
		if (!in_pool(nv, &pm->regs[i]))
			region_restore(nv, &pm->regs[i]);
	nv_bar1_resume(nv);
	for (i = 0; i < pm->nregs; i++)
		if (in_pool(nv, &pm->regs[i]))
			region_restore(nv, &pm->regs[i]);
	ret = toggle_scheduling(nv, false);
	if (ret)
		dev_warn(dev, "pm: scheduling on: %d\n", ret);
	ret = nv_fifo_init(nv);	/* kernel copy channel, new */
	if (ret)
		dev_warn(dev, "pm: kernel channels: %d\n", ret);
	nv_intr_init(nv);
	disp_forget_link(nv);
	ret = nv_evo_init(nv);	/* display object and core channel again; DRM resume does the rest */
	if (ret)
		dev_warn(dev, "pm: display init: %d\n", ret);
	ret = 0;
out:
	pm->suspended = false;
	mutex_unlock(&nv->gsp_lock);
	pm_buffers_free(nv);
	nv_uapi_pm(nv, false);
	if (ret)
		nv_uapi_device_lost(nv, "resume failed");
	dev_info(dev, "pm: resumed: %d (%lld ms)\n", ret, ktime_ms_delta(ktime_get(), t0));
	return ret;
}

/*
 * debugfs pm_test: suspend and resume without sleeping. mode 1: as is (VRAM keeps its
 * contents); mode 2: FLR in between, which resets the GPU the way losing power does.
 */
int nv_pm_test(struct nvdrv_dev *nv, u32 mode)
{
	int ret;

	if (mode == 3) {	/* display left on (channels freed under a running head), FLR, bring-up */
		ret = nv_pm_suspend(nv);
		if (ret)
			return ret;
		ret = pci_reset_function(nv->pdev);
		dev_info(&nv->pdev->dev, "pm: test FLR: %d\n", ret);
		ret = nv_pm_resume(nv);
		if (!ret)
			ret = nv_disp_bringup(nv);
		dev_info(&nv->pdev->dev, "pm: test bring-up: %d\n", ret);
		return ret;
	}
	ret = nv_drm_suspend(nv);
	if (ret)
		return ret;
	ret = nv_pm_suspend(nv);
	if (ret) {
		nv_drm_resume(nv);
		return ret;
	}
	if (mode == 2) {
		ret = pci_reset_function(nv->pdev);
		dev_info(&nv->pdev->dev, "pm: test FLR: %d\n", ret);
	}
	ret = nv_pm_resume(nv);
	nv_drm_resume(nv);
	return ret;
}
