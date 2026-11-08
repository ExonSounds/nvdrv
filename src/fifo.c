// SPDX-License-Identifier: GPL-2.0
/*
 * GPU address space, channels and the copy engine on top of GSP-RM (Phase 8/9).
 *
 * The approach is nouveau's (drivers/gpu/drm/nouveau/nvkm/subdev/gsp/rm/r535/{vmm,fifo,ce}.c,
 * MIT): the host writes the GPU page tables itself, allocates FERMI_VASPACE_A in GSP-RM
 * and tells RM where the page directories for its server-reserved range are
 * (NV90F1_CTRL_CMD_VASPACE_COPY_SERVER_RESERVED_PDES), then allocates a GPFIFO channel
 * with host-provided instance block, USERD and method buffer, binds and schedules it,
 * and allocates the engine object (AMPERE_DMA_COPY_B) on it. Work is submitted by
 * writing GPFIFO entries, GP_PUT in USERD, and the channel's work-submit token to the
 * usermode doorbell.
 *
 * Layouts: tools/fifo_struct_probe.sh (compiled from NVIDIA's SDK headers, 570.144).
 */
#include <linux/delay.h>
#include <linux/dma-mapping.h>
#include <linux/iopoll.h>
#include <linux/lockdep.h>
#include <linux/seq_file.h>
#include <linux/slab.h>

#include "nvdrv.h"

static void rl_state(struct nvdrv_dev *nv, const char *when);

/* ---- GMMU page tables (GP100+ format, see bar1.c) ------------------------- */

#define PDE_VRAM		(1ULL << 1)
#define PTE_VALID		BIT_ULL(0)
#define PTE_APER_SYS_COH	(2ULL << 1)
#define PTE_VOL			BIT_ULL(3)

#define MMU_INVALIDATE_PDB_LO	0x00b830a0
#define MMU_INVALIDATE_PDB_HI	0x00b830a4
#define MMU_INVALIDATE		0x00b830b0

/* VA layout: RM's server-reserved range at 4 GiB, nvdrv's own mappings from 8 GiB. */
#define SPLIT_VAS_START		0x100000000ULL
#define SPLIT_VAS_SIZE		0x20000000ULL
#define NV_VA_BASE		0x200000000ULL

static void wr64(struct nvdrv_dev *nv, u64 addr, u64 v)
{
	nv_vram_wr32(nv, addr, lower_32_bits(v));
	nv_vram_wr32(nv, addr + 4, upper_32_bits(v));
}

static u64 rd64(struct nvdrv_dev *nv, u64 addr)
{
	return nv_vram_rd32(nv, addr) | (u64)nv_vram_rd32(nv, addr + 4) << 32;
}

/* A zeroed 4K page-table page from the VRAM carve-out. */
static int pt_page(struct nvdrv_dev *nv, u64 *out)
{
	int ret = nv_vram_alloc(nv, SZ_4K, SZ_4K, out);

	if (!ret)
		nv_vram_fill(nv, *out, SZ_4K, 0, NULL, NULL);
	return ret;
}

/* Next-level table behind entry (addr), allocated on first use. */
static int pd_next(struct nvdrv_dev *nv, u64 entry, u64 *next)
{
	u64 e = rd64(nv, entry);
	int ret;

	if ((e >> 1) & 3) {
		*next = (e & ~0xfULL) << 4;
		return 0;
	}
	ret = pt_page(nv, next);
	if (!ret)
		wr64(nv, entry, (*next >> 4) | PDE_VRAM);
	return ret;
}

/*
 * Map one 4 KiB page at va. 4K pages use PD0 entry [1] (the small page table, 512 x 8 B);
 * aperture: VRAM (pte_flags 0) or coherent sysmem (PTE_APER_SYS_COH | PTE_VOL).
 */
static int vmm_map_4k(struct nvdrv_dev *nv, struct nv_vmm *vm, u64 va, u64 pa, u64 pte_flags)
{
	u64 pd2, pd1, pd0, pt;
	int ret;

	ret = pd_next(nv, vm->root + ((va >> 47) & 3) * 8, &pd2);
	if (!ret)
		ret = pd_next(nv, pd2 + ((va >> 38) & 511) * 8, &pd1);
	if (!ret)
		ret = pd_next(nv, pd1 + ((va >> 29) & 511) * 8, &pd0);
	if (!ret)
		ret = pd_next(nv, pd0 + ((va >> 21) & 255) * 16 + 8, &pt);
	if (ret)
		return ret;
	wr64(nv, pt + ((va >> 12) & 511) * 8, (pa >> 4) | PTE_VALID | pte_flags);
	return 0;
}

static int vmm_flush(struct nvdrv_dev *nv, struct nv_vmm *vm)
{
	u32 v;

	(void)nv_vram_rd32(nv, vm->root);	/* PRAMIN writes landed */
	nv_wr32(nv, MMU_INVALIDATE_PDB_LO, lower_32_bits(vm->root >> 8));
	nv_wr32(nv, MMU_INVALIDATE_PDB_HI, upper_32_bits(vm->root >> 8));
	nv_wr32(nv, MMU_INVALIDATE, BIT(31) | 0x1);	/* PAGE_ALL */
	return readl_poll_timeout(nv->bar0 + MMU_INVALIDATE, v, !(v & BIT(31)), 10, 2000000);
}

/* Map a VRAM range (4K granularity) at a fresh VA; returns the VA. */
static int vmm_map_vram(struct nvdrv_dev *nv, struct nv_vmm *vm, u64 vram, u64 size, u64 *va)
{
	u64 off;
	int ret = 0;

	*va = vm->next;
	for (off = 0; off < size && !ret; off += SZ_4K)
		ret = vmm_map_4k(nv, vm, *va + off, vram + off, 0);
	vm->next = ALIGN(vm->next + size, SZ_2M);
	return ret;
}

/* Map a coherent sysmem buffer (contiguous bus addresses) at a fresh VA. */
static int vmm_map_sysmem(struct nvdrv_dev *nv, struct nv_vmm *vm, const struct nv_dma_buf *b, u64 *va)
{
	u64 off;
	int ret = 0;

	*va = vm->next;
	for (off = 0; off < b->size && !ret; off += SZ_4K)
		ret = vmm_map_4k(nv, vm, *va + off, b->dma + off, PTE_APER_SYS_COH | PTE_VOL);
	vm->next = ALIGN(vm->next + b->size, SZ_2M);
	return ret;
}

/*
 * Point a channel instance block at the address space (nouveau gp100_vmm_join +
 * gv100_vmm_join): PDB at 0x200 (VER2, 64 KiB big pages, VRAM), VA limit at 0x208,
 * subcontext 0 PDB at 0x2a0, subcontexts 1..63 invalid, valid mask at 0x298.
 */
static void vmm_join(struct nvdrv_dev *nv, struct nv_vmm *vm, u64 inst)
{
	u64 base = vm->root | BIT_ULL(10) | BIT_ULL(11);
	int i;

	wr64(nv, inst + 0x200, base);
	wr64(nv, inst + 0x208, (1ULL << 49) - 1);
	nv_vram_wr32(nv, inst + 0x21c, 0);
	for (i = 0; i < 64; i++) {
		nv_vram_wr32(nv, inst + 0x2a4 + i * 0x10, i ? 1 : upper_32_bits(base));
		nv_vram_wr32(nv, inst + 0x2a0 + i * 0x10, i ? 1 : lower_32_bits(base));
		nv_vram_wr32(nv, inst + 0x2a8 + i * 0x10, 0);
	}
	nv_vram_wr32(nv, inst + 0x298, 1);
	nv_vram_wr32(nv, inst + 0x29c, 0);
}

/* ---- RM objects ------------------------------------------------------------ */

#define FERMI_VASPACE_A				0x90f1
#define AMPERE_CHANNEL_GPFIFO_A			0xc56f
#define AMPERE_DMA_COPY_B			0xc7b5
#define NV90F1_CTRL_CMD_VASPACE_COPY_SERVER_RESERVED_PDES	0x90f10106
#define NV2080_CTRL_CMD_FIFO_GET_DEVICE_INFO_TABLE	0x20801112
#define NV2080_CTRL_CMD_CE_GET_FAULT_METHOD_BUFFER_SIZE	0x20802a08
#define NV2080_CTRL_CMD_CE_GET_CE_PCE_MASK		0x20802a02
#define NVA06F_CTRL_CMD_GPFIFO_SCHEDULE		0xa06f0103
#define NVA06F_CTRL_CMD_BIND			0xa06f0104
#define NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN	0xc36f0108

#define NVDRV_H_VASPACE		0xd0009001
#define NVDRV_H_CHAN(i)		(0xd000c000 + (i))
#define NVDRV_H_CE(i)		(0xd000c100 + (i))

#define ENGINE_INFO_TYPE_RM_ENGINE_TYPE	2
#define ENGINE_INFO_TYPE_RUNLIST	3
#define ENGINE_INFO_TYPE_RUNLIST_PRI_BASE	11
#define RM_ENGINE_TYPE_GR0		0x01
#define RM_ENGINE_TYPE_COPY0		0x09	/* rm_engine_type.h */
#define NV2080_ENGINE_TYPE_COPY0	0x09

#define USERD_SIZE		0x200
#define USERD_GP_GET		0x88
#define USERD_GP_PUT		0x8c
#define CHAN_INST_SIZE		0x1000
#define GPFIFO_ENTRIES		512
#define PUSHBUF_SIZE		SZ_64K
#define NV_USERMODE_DOORBELL	(0xb80000 + 0x030000 + 0x90)	/* AMPERE_USERMODE_A + NOTIFY_CHANNEL_PENDING */
#define NV_RSVD_CHIDS		1				/* r570 */

struct nv_vaspace_alloc {
	__le32 index, flags;
	__le64 va_size, va_start_internal, va_limit_internal;
	__le32 big_page_size, pad;
	__le64 va_base;
};
static_assert(sizeof(struct nv_vaspace_alloc) == 48);

struct nv_copy_server_pdes {
	__le32 h_subdevice, subdevice_id;
	__le64 page_size, virt_addr_lo, virt_addr_hi;
	__le32 num_levels, pad;
	struct {
		__le64 phys_address, size;
		__le32 aperture;
		u8 page_shift, pad[3];
	} levels[6];
};
static_assert(sizeof(struct nv_copy_server_pdes) == 184);

struct nv_memdesc {
	__le64 base, size;
	__le32 address_space, cache_attrib;
};

struct nv_channel_alloc {
	__le32 h_object_error, h_object_buffer;
	__le64 gpfifo_offset;
	__le32 gpfifo_entries, flags, h_context_share, h_vaspace;
	__le32 h_userd_memory[8];
	__le64 userd_offset[8];
	__le32 engine_type, cid, sub_device_id, h_object_ecc_error;
	struct nv_memdesc instance_mem, userd_mem, ramfc_mem, mthdbuf_mem;
	__le32 h_phys_channel_group, internal_flags;
	struct nv_memdesc error_notifier_mem, ecc_error_notifier_mem;
	__le32 process_id, sub_process_id;
	u8 encrypt_iv[12], decrypt_iv[12], hmac_nonce[32];
	u8 pad[8];
};
static_assert(offsetof(struct nv_channel_alloc, engine_type) == 128);
static_assert(offsetof(struct nv_channel_alloc, instance_mem) == 144);
static_assert(offsetof(struct nv_channel_alloc, internal_flags) == 244);
static_assert(offsetof(struct nv_channel_alloc, process_id) == 296);
static_assert(sizeof(struct nv_channel_alloc) == 368);

#define NVOS04_FLAGS_BASE	0x00200020	/* PHYSICAL, PRIVILEGED, USERD_INDEX_PAGE_FIXED, ... */
#define NVOS04_RUNQUEUE_SHIFT	4
#define NVOS04_USERD_IDX_SHIFT	8
#define NVOS04_USERD_PAGE_SHIFT	12
/*
 * NV_KERNELCHANNEL_ALLOC_INTERNALFLAGS: PRIVILEGE (1:0) ADMIN = 1, ERROR_NOTIFIER_TYPE (3:2),
 * ECC_ERROR_NOTIFIER_TYPE (5:4); types UNKNOWN 0 (CPU-RM-only placeholder), NONE 1, CTXDMA 2,
 * MEMORY 3. nvdrv asks for a memory error notifier so RM reports why it recovers a channel.
 */
#define KCHAN_INTERNAL_FLAGS	(0x1 | 3 << 2 | 1 << 4)

/*
 * The GPU address space nvdrv's channels run in (r535_mmu_vaspace_new, not external):
 * root page directory in VRAM, RM told which PD3/PD2/PD1 pages cover its reserved
 * 512 MiB at 4 GiB so it can place its own mappings there.
 */
static int vmm_init(struct nvdrv_dev *nv, struct nv_vmm *vm)
{
	struct nv_vaspace_alloc a = {};
	struct nv_copy_server_pdes c = {};
	u64 pd2, pd1;
	int ret;

	ret = pt_page(nv, &vm->root);
	if (!ret)
		ret = pd_next(nv, vm->root + ((SPLIT_VAS_START >> 47) & 3) * 8, &pd2);
	if (!ret)
		ret = pd_next(nv, pd2 + ((SPLIT_VAS_START >> 38) & 511) * 8, &pd1);
	if (ret)
		return ret;
	vm->next = NV_VA_BASE;

	a.index = 0;		/* NV_VASPACE_ALLOCATION_INDEX_GPU_NEW */
	ret = nv_rm_alloc(nv, nv->rm.h_client, nv->rm.h_device, NVDRV_H_VASPACE, FERMI_VASPACE_A, &a, sizeof(a));
	if (ret)
		return ret;
	vm->handle = NVDRV_H_VASPACE;

	c.page_size = cpu_to_le64(SPLIT_VAS_SIZE);
	c.virt_addr_lo = cpu_to_le64(SPLIT_VAS_START);
	c.virt_addr_hi = cpu_to_le64(SPLIT_VAS_START + SPLIT_VAS_SIZE - 1);
	c.num_levels = cpu_to_le32(3);
	c.levels[0].phys_address = cpu_to_le64(vm->root);	/* PD3: 4 x 8 B, VA 48:47 */
	c.levels[0].size = cpu_to_le64(4 * 8);
	c.levels[0].aperture = cpu_to_le32(1);
	c.levels[0].page_shift = 47;
	c.levels[1].phys_address = cpu_to_le64(pd2);		/* PD2: 512 x 8 B, VA 46:38 */
	c.levels[1].size = cpu_to_le64(512 * 8);
	c.levels[1].aperture = cpu_to_le32(1);
	c.levels[1].page_shift = 38;
	c.levels[2].phys_address = cpu_to_le64(pd1);		/* PD1: 512 x 8 B, VA 37:29 */
	c.levels[2].size = cpu_to_le64(512 * 8);
	c.levels[2].aperture = cpu_to_le32(1);
	c.levels[2].page_shift = 29;
	ret = nv_rm_control(nv, nv->rm.h_client, vm->handle, NV90F1_CTRL_CMD_VASPACE_COPY_SERVER_RESERVED_PDES,
			    &c, sizeof(c));
	dev_info(&nv->pdev->dev, "fifo: VA space 0x%08x, root PD at VRAM 0x%llx, server PDEs copied: %d\n",
		 vm->handle, vm->root, ret);
	return ret;
}

/*
 * Pick an async copy engine: a COPYn on a runlist of its own (not the GR runlist, whose
 * GRCE nouveau also avoids for async copies). FIFO_GET_DEVICE_INFO_TABLE on the
 * internal subdevice.
 */
static int ce_pick(struct nvdrv_dev *nv, u32 *runlist, u32 *ce, u32 *rl_base)
{
	u32 gr_runlist = ~0u;
	struct nv_gsp_static_info *si = &nv->gsp_info;
	u8 *t;
	u32 n, i;
	int ret;

	t = kzalloc(3212, GFP_KERNEL);
	if (!t)
		return -ENOMEM;
	ret = nv_rm_control(nv, si->h_internal_client, si->h_internal_subdevice,
			    NV2080_CTRL_CMD_FIFO_GET_DEVICE_INFO_TABLE, t, 3212);
	ret = ret ?: -ENODEV;
	n = le32_to_cpu(*(__le32 *)(t + 4));
	for (i = 0; i < n && i < 32; i++) {
		const __le32 *data = (const __le32 *)(t + 12 + i * 100);

		if (le32_to_cpu(data[ENGINE_INFO_TYPE_RM_ENGINE_TYPE]) == RM_ENGINE_TYPE_GR0)
			gr_runlist = le32_to_cpu(data[ENGINE_INFO_TYPE_RUNLIST]);
	}
	for (i = 0; i < n && i < 32; i++) {
		const __le32 *data = (const __le32 *)(t + 12 + i * 100);
		u32 type = le32_to_cpu(data[ENGINE_INFO_TYPE_RM_ENGINE_TYPE]);
		u32 rl = le32_to_cpu(data[ENGINE_INFO_TYPE_RUNLIST]);

		dev_info(&nv->pdev->dev, "fifo: engine %u: RM type 0x%02x, runlist %u, name %.16s\n", i, type, rl,
			 (const char *)(t + 12 + i * 100 + 84));
		if (type >= RM_ENGINE_TYPE_COPY0 && type < RM_ENGINE_TYPE_COPY0 + 10 && rl != gr_runlist) {
			/* an LCE without physical copy engines raises INVALID_CONFIG on first use */
			__le32 pce[2] = { cpu_to_le32(NV2080_ENGINE_TYPE_COPY0 + type - RM_ENGINE_TYPE_COPY0), 0 };
			int r = nv_rm_control(nv, nv->rm.h_client, nv->rm.h_subdevice,
					      NV2080_CTRL_CMD_CE_GET_CE_PCE_MASK, pce, sizeof(pce));

			dev_info(&nv->pdev->dev, "fifo: CE%u PCE mask 0x%x (%d)\n", type - RM_ENGINE_TYPE_COPY0,
				 le32_to_cpu(pce[1]), r);
			if (r || !pce[1] || !ret)
				continue;
			*runlist = rl;
			*rl_base = le32_to_cpu(data[ENGINE_INFO_TYPE_RUNLIST_PRI_BASE]);
			*ce = type - RM_ENGINE_TYPE_COPY0;
			ret = 0;
		}
	}
	if (!ret)
		dev_info(&nv->pdev->dev, "fifo: using CE%u on runlist %u (GR runlist %u)\n", *ce, *runlist, gr_runlist);
	kfree(t);
	return ret;
}

static int dma_buf_alloc(struct nvdrv_dev *nv, struct nv_dma_buf *b, size_t size)
{
	b->size = size;
	b->cpu = dma_alloc_coherent(&nv->pdev->dev, size, &b->dma, GFP_KERNEL);
	if (!b->cpu)
		return -ENOMEM;
	memset(b->cpu, 0, size);
	return 0;
}

static void dma_buf_free(struct nvdrv_dev *nv, struct nv_dma_buf *b)
{
	if (b->cpu)
		dma_free_coherent(&nv->pdev->dev, b->size, b->cpu, b->dma);
	b->cpu = NULL;
}

/*
 * A privileged GPFIFO channel on the CE0 runlist with the copy engine bound to
 * subchannel 4 (r535_chan_alloc + r535_chan_ramfc_write + r535_ce_alloc).
 */
static int chan_init(struct nvdrv_dev *nv, struct nv_chan *ch)
{
	struct device *dev = &nv->pdev->dev;
	struct nv_gsp_static_info *si = &nv->gsp_info;
	struct nv_channel_alloc *a;
	struct { __le32 version, engine_type; } ce;
	__le32 v32;
	u32 chid = NV_RSVD_CHIDS, mthdbuf_size;
	u64 userd_bar1 = NV_FB_MAX;	/* BAR1 VA right after the scanout surface */
	int ret;

	ret = ce_pick(nv, &ch->runlist, &ch->ce, &ch->rl_base);
	if (ret)
		return ret;
	ce.version = cpu_to_le32(1);
	ce.engine_type = cpu_to_le32(NV2080_ENGINE_TYPE_COPY0 + ch->ce);
	v32 = 0;
	ret = nv_rm_control(nv, si->h_internal_client, si->h_internal_subdevice,
			    NV2080_CTRL_CMD_CE_GET_FAULT_METHOD_BUFFER_SIZE, &v32, sizeof(v32));
	if (ret)
		return ret;
	mthdbuf_size = le32_to_cpu(v32);

	/* VRAM: instance block, USERD page; sysmem: GPFIFO, push buffer, method buffer, semaphores */
	ret = nv_vram_alloc(nv, CHAN_INST_SIZE, SZ_4K, &ch->inst);
	if (!ret)
		ret = nv_vram_alloc(nv, SZ_64K, SZ_64K, &ch->userd);
	if (ret)
		return ret;
	nv_vram_fill(nv, ch->inst, CHAN_INST_SIZE, 0, NULL, NULL);
	nv_vram_fill(nv, ch->userd, SZ_64K, 0, NULL, NULL);
	vmm_join(nv, &nv->vmm, ch->inst);
	ch->userd_cpu = nv_bar1_map(nv, userd_bar1, ch->userd, SZ_64K);
	if (!ch->userd_cpu)
		return -EIO;
	ch->userd_cpu += chid * USERD_SIZE;
	if ((ret = dma_buf_alloc(nv, &ch->gpfifo, GPFIFO_ENTRIES * 8)) ||
	    (ret = dma_buf_alloc(nv, &ch->pb, PUSHBUF_SIZE)) ||
	    (ret = dma_buf_alloc(nv, &ch->mthdbuf, PAGE_ALIGN(mthdbuf_size))) ||
	    (ret = dma_buf_alloc(nv, &ch->sem, SZ_4K)) ||
	    (ret = dma_buf_alloc(nv, &ch->errnotif, SZ_4K)))
		return ret;
	if ((ret = vmm_map_sysmem(nv, &nv->vmm, &ch->gpfifo, &ch->gpfifo_va)) ||
	    (ret = vmm_map_sysmem(nv, &nv->vmm, &ch->pb, &ch->pb_va)) ||
	    (ret = vmm_map_sysmem(nv, &nv->vmm, &ch->sem, &ch->sem_va)) ||
	    (ret = vmm_flush(nv, &nv->vmm)))
		return ret;

	a = kzalloc(sizeof(*a), GFP_KERNEL);
	if (!a)
		return -ENOMEM;
	a->gpfifo_offset = cpu_to_le64(ch->gpfifo_va);
	a->gpfifo_entries = cpu_to_le32(GPFIFO_ENTRIES);
	a->flags = cpu_to_le32(NVOS04_FLAGS_BASE | 0 << NVOS04_RUNQUEUE_SHIFT |
			       (chid % 8) << NVOS04_USERD_IDX_SHIFT | (chid / 8) << NVOS04_USERD_PAGE_SHIFT);
	a->h_vaspace = cpu_to_le32(nv->vmm.handle);
	a->engine_type = cpu_to_le32(NV2080_ENGINE_TYPE_COPY0 + ch->ce);
	a->instance_mem = (struct nv_memdesc){ cpu_to_le64(ch->inst), cpu_to_le64(CHAN_INST_SIZE), cpu_to_le32(2), cpu_to_le32(1) };
	a->userd_mem = (struct nv_memdesc){ cpu_to_le64(ch->userd + chid * USERD_SIZE), cpu_to_le64(USERD_SIZE), cpu_to_le32(2), cpu_to_le32(1) };
	a->ramfc_mem = (struct nv_memdesc){ cpu_to_le64(ch->inst), cpu_to_le64(0x200), cpu_to_le32(2), cpu_to_le32(1) };
	a->mthdbuf_mem = (struct nv_memdesc){ cpu_to_le64(ch->mthdbuf.dma), cpu_to_le64(mthdbuf_size), cpu_to_le32(1), 0 };
	a->internal_flags = cpu_to_le32(KCHAN_INTERNAL_FLAGS);
	a->error_notifier_mem = (struct nv_memdesc){ cpu_to_le64(ch->errnotif.dma), cpu_to_le64(SZ_4K), cpu_to_le32(1), 0 };
	ret = nv_rm_alloc(nv, nv->rm.h_client, nv->rm.h_device, NVDRV_H_CHAN(0), AMPERE_CHANNEL_GPFIFO_A, a, sizeof(*a));
	kfree(a);
	if (ret)
		return ret;
	ch->handle = NVDRV_H_CHAN(0);
	ch->chid = chid;

	v32 = cpu_to_le32(NV2080_ENGINE_TYPE_COPY0 + ch->ce);
	ret = nv_rm_control(nv, nv->rm.h_client, ch->handle, NVA06F_CTRL_CMD_BIND, &v32, sizeof(v32));
	if (ret)
		return ret;
	{
		u8 sched[2] = { 1, 0 };	/* NVA06F_CTRL_GPFIFO_SCHEDULE_PARAMS: bEnable, bSkipSubmit */

		ret = nv_rm_control(nv, nv->rm.h_client, ch->handle, NVA06F_CTRL_CMD_GPFIFO_SCHEDULE, sched, 2);
	}
	if (ret)
		return ret;
	/*
	 * Doorbell token: the runlist's hardware doorbell ID (RUNLIST_DOORBELL_CONFIG at
	 * runlist base + 0x008, bits 31:16; nouveau ga100_runl_new / ga100_chan_doorbell_handle),
	 * which is not necessarily the runlist number.
	 */
	ch->token = (nv_rd32(nv, ch->rl_base + 0x008) >> 16) << 16 | ch->chid;
	dev_info(dev, "fifo: runlist %u at 0x%x: chcfg 0x%08x dbcfg 0x%08x pbcfg 0x%08x/0x%08x -> token 0x%x\n",
		 ch->runlist, ch->rl_base, nv_rd32(nv, ch->rl_base + 0x004), nv_rd32(nv, ch->rl_base + 0x008),
		 nv_rd32(nv, ch->rl_base + 0x010), nv_rd32(nv, ch->rl_base + 0x014), ch->token);
	ret = nv_rm_alloc(nv, nv->rm.h_client, ch->handle, NVDRV_H_CE(0), AMPERE_DMA_COPY_B, &ce, sizeof(ce));
	if (ret)
		return ret;
	dev_info(dev, "fifo: channel %u (runlist %u) up: inst 0x%llx, USERD 0x%llx, GPFIFO va 0x%llx, token 0x%x, CE bound\n",
		 ch->chid, ch->runlist, ch->inst, ch->userd + chid * USERD_SIZE, ch->gpfifo_va, ch->token);
	return 0;
}

/* ---- push buffer and submission -------------------------------------------- */

static void push(struct nv_chan *ch, u32 subc, u32 mthd, const u32 *data, u32 n)
{
	u32 *pb = ch->pb.cpu;

	pb[ch->pb_put++] = 0x20000000 | n << 16 | subc << 13 | mthd >> 2;	/* incrementing method */
	while (n--)
		pb[ch->pb_put++] = *data++;
}

static void push1(struct nv_chan *ch, u32 subc, u32 mthd, u32 v)
{
	push(ch, subc, mthd, &v, 1);
}

/* Submit everything pushed since the last kick as one GPFIFO entry. */
static void kick(struct nvdrv_dev *nv, struct nv_chan *ch)
{
	u64 va = ch->pb_va + ch->pb_start * 4;
	u32 len = ch->pb_put - ch->pb_start, *gp = ch->gpfifo.cpu;

	gp[ch->gp_put * 2] = lower_32_bits(va) & ~3u;
	gp[ch->gp_put * 2 + 1] = (upper_32_bits(va) & 0xff) | len << 10;
	ch->gp_put = (ch->gp_put + 1) % GPFIFO_ENTRIES;
	ch->pb_start = ch->pb_put;
	wmb();
	iowrite32(ch->gp_put, ch->userd_cpu + USERD_GP_PUT);
	wmb();
	(void)ioread32(ch->userd_cpu + USERD_GP_PUT);
	nv_wr32(nv, NV_USERMODE_DOORBELL, ch->token);
}

#define SUBC_CE		4
#define C7B5_SET_SEMAPHORE_A	0x240
#define C7B5_LAUNCH_DMA		0x300
#define C7B5_OFFSET_IN_UPPER	0x400
#define C7B5_LINE_LENGTH_IN	0x418
#define LAUNCH_COPY_SEM		0x18e	/* non-pipelined, flush, pitch->pitch, virtual, release 1-word semaphore */

/* GPU copy: dst_va <- src_va, bytes; then semaphore (sem_va) <- seq. */
static void ce_copy(struct nv_chan *ch, u64 dst_va, u64 src_va, u32 bytes, u32 seq)
{
	u32 in_out[4] = { upper_32_bits(src_va), lower_32_bits(src_va), upper_32_bits(dst_va), lower_32_bits(dst_va) };
	u32 sem[3] = { upper_32_bits(ch->sem_va), lower_32_bits(ch->sem_va), seq };

	push(ch, SUBC_CE, C7B5_OFFSET_IN_UPPER, in_out, 4);
	push1(ch, SUBC_CE, C7B5_LINE_LENGTH_IN, bytes);
	push(ch, SUBC_CE, C7B5_SET_SEMAPHORE_A, sem, 3);
	push1(ch, SUBC_CE, C7B5_LAUNCH_DMA, LAUNCH_COPY_SEM);
}

static void rl_state(struct nvdrv_dev *nv, const char *when)
{
	struct nv_chan *ch = &nv->chan;
	u32 chram = (nv_rd32(nv, ch->rl_base + 0x004) & 0xfffffff0) + ch->chid * 4;

	{
		const u32 *n = ch->errnotif.cpu;

		if (n && (n[2] || n[3]))
			dev_info(&nv->pdev->dev, "fifo: [%s] error notifier: info32 %u (0x%x) info16 0x%04x status 0x%04x\n",
				 when, n[2], n[2], n[3] & 0xffff, n[3] >> 16);
	}
	dev_info(&nv->pdev->dev, "fifo: [%s] runlist base 0x%x%08x len %u info 0x%x, chram 0x%03x, pbdma GP_GET %u\n",
		 when, nv_rd32(nv, ch->rl_base + 0x084), nv_rd32(nv, ch->rl_base + 0x080),
		 nv_rd32(nv, ch->rl_base + 0x088), nv_rd32(nv, ch->rl_base + 0x08c), nv_rd32(nv, chram),
		 nv_rd32(nv, 0x40000 + 9 * 0x800 + 0x14));
}

int nv_fifo_init(struct nvdrv_dev *nv)
{
	int ret;

	lockdep_assert_held(&nv->gsp_lock);
	if (nv->chan.handle)
		return 0;
	if (!nv->rm.h_device)
		return -ENODEV;
	ret = nv_bar1_init(nv);
	if (!ret && !nv->vmm.handle)
		ret = vmm_init(nv, &nv->vmm);
	if (!ret)
		ret = chan_init(nv, &nv->chan);
	if (!ret) {
		push1(&nv->chan, SUBC_CE, 0, AMPERE_DMA_COPY_B);	/* SET_OBJECT */
		kick(nv, &nv->chan);
	}
	return ret;
}

/*
 * Copy-engine test: GPU-copy a 1 MiB sysmem pattern into the middle of the scanout
 * surface, wait for the semaphore, verify through PRAMIN.
 */
int nv_ce_test(struct nvdrv_dev *nv)
{
	struct device *dev = &nv->pdev->dev;
	struct nv_chan *ch = &nv->chan;
	struct nv_evo *e = &nv->evo;
	struct nv_dma_buf src = {};
	u64 src_va, dst_va, dst = 0;
	u32 *s, i, bad = 0, sem, seq;
	int ret;

	mutex_lock(&nv->gsp_lock);
	ret = nv_fifo_init(nv);
	if (ret || !e->fb)
		goto out;
	ret = dma_buf_alloc(nv, &src, SZ_1M);
	if (ret)
		goto out;
	for (s = src.cpu, i = 0; i < SZ_1M / 4; i++)
		s[i] = 0xff000000 | (i * 2654435761u >> 8);	/* hashed colours */
	dst = e->fb + (e->fb_size / 2 & ~(u64)(SZ_1M - 1));
	ret = vmm_map_sysmem(nv, &nv->vmm, &src, &src_va);
	if (!ret)
		ret = vmm_map_vram(nv, &nv->vmm, dst, SZ_1M, &dst_va);
	if (!ret)
		ret = vmm_flush(nv, &nv->vmm);
	if (ret)
		goto out_free;

	seq = ++ch->seq;
	ce_copy(ch, dst_va, src_va, SZ_1M, seq);
	kick(nv, ch);
	ret = read_poll_timeout(READ_ONCE, sem, sem == seq, 50, 2000000, false, *(u32 *)ch->sem.cpu);
	for (i = 0; i < 64; i++) {
		u32 idx = (i * 4099) % (SZ_1M / 4);

		bad += nv_vram_rd32(nv, dst + idx * 4) != s[idx];
	}
	dev_info(dev, "ce test: 1 MiB sysmem -> VRAM 0x%llx by the copy engine: semaphore %u/%u (%d), GP_GET %u/%u, %u/64 mismatches\n",
		 dst, sem, seq, ret, ioread32(ch->userd_cpu + USERD_GP_GET), ch->gp_put, bad);
	if (!ret && bad)
		ret = -EIO;
out_free:
	dma_buf_free(nv, &src);	/* GPU done with it (or the GPU gets reset before reuse) */
out:
	mutex_unlock(&nv->gsp_lock);
	return ret;
}

/*
 * debugfs chan_kick: ring one doorbell flavour and report the PBDMA's cached GP_PUT/GP_GET.
 * 0 user doorbell (NV_FUNC_DOORBELL, token), 1 privileged doorbell (NV_FUNC_PRIV_DOORBELL,
 * token), 2 runlist-internal doorbell (NV_RUNLIST_INTERNAL_DOORBELL, chid).
 */
int nv_chan_kick(struct nvdrv_dev *nv, u32 mode)
{
	struct nv_chan *ch = &nv->chan;
	u32 pb = 0x40000 + 9 * 0x800, before_put, before_get;	/* PBDMA 9 (runlist 8 on GA104) */

	if (!ch->handle)
		return -ENODEV;
	before_put = nv_rd32(nv, pb + 0x00);
	before_get = nv_rd32(nv, pb + 0x14);
	switch (mode) {
	case 0: nv_wr32(nv, 0xb80000 + 0x30090, ch->token); break;
	case 1: nv_wr32(nv, 0xb80000 + 0x2200, ch->token); break;
	case 2: nv_wr32(nv, ch->rl_base + 0x090, ch->chid); break;
	case 3:	/* diagnostic: set ENABLE in channel RAM (nouveau ga100_chan_start), then user doorbell */
		nv_wr32(nv, (nv_rd32(nv, ch->rl_base + 0x004) & 0xfffffff0) + ch->chid * 4, 0x2);
		nv_wr32(nv, 0xb80000 + 0x30090, ch->token);
		break;
	default: return -EINVAL;
	}
	msleep(20);
	dev_info(&nv->pdev->dev, "chan kick %u: chram 0x%03x, PBDMA GP_PUT %u -> %u, GP_GET %u -> %u, USERD GP_GET %u GP_PUT %u, sem %u\n",
		 mode, nv_rd32(nv, (nv_rd32(nv, ch->rl_base + 0x004) & 0xfffffff0) + ch->chid * 4),
		 before_put, nv_rd32(nv, pb + 0x00), before_get, nv_rd32(nv, pb + 0x14),
		 ioread32(ch->userd_cpu + USERD_GP_GET), ioread32(ch->userd_cpu + USERD_GP_PUT),
		 READ_ONCE(*(u32 *)ch->sem.cpu));
	return 0;
}

/* debugfs chan_status */
void nv_fifo_show(struct nvdrv_dev *nv, struct seq_file *m)
{
	struct nv_chan *ch = &nv->chan;
	u32 *gp = ch->gpfifo.cpu, *pb = ch->pb.cpu, i;

	if (!ch->handle) {
		seq_puts(m, "no channel\n");
		return;
	}
	seq_printf(m, "chid %u runlist %u CE%u token 0x%x gp_put %u pb_put %u seq %u sem %u\n", ch->chid,
		   ch->runlist, ch->ce, ch->token, ch->gp_put, ch->pb_put, ch->seq, READ_ONCE(*(u32 *)ch->sem.cpu));
	seq_puts(m, "USERD 0x40..0x9c:");
	for (i = 0x40; i < 0xa0; i += 4)
		seq_printf(m, " %08x", ioread32(ch->userd_cpu + i));
	seq_puts(m, "\nGPFIFO:");
	for (i = 0; i < ch->gp_put && i < 8; i++)
		seq_printf(m, " [%08x %08x]", gp[2 * i], gp[2 * i + 1]);
	seq_puts(m, "\npush buffer:");
	for (i = 0; i < ch->pb_put && i < 24; i++)
		seq_printf(m, " %08x", pb[i]);
	seq_printf(m, "\nva: gpfifo 0x%llx pb 0x%llx sem 0x%llx\n", ch->gpfifo_va, ch->pb_va, ch->sem_va);
}

/* After the GPU reset in .remove: host memory only (RM objects died with the client). */
void nv_fifo_fini(struct nvdrv_dev *nv)
{
	struct nv_chan *ch = &nv->chan;

	if (ch->userd_cpu)
		iounmap(ch->userd_cpu - ch->chid * USERD_SIZE);
	dma_buf_free(nv, &ch->gpfifo);
	dma_buf_free(nv, &ch->pb);
	dma_buf_free(nv, &ch->mthdbuf);
	dma_buf_free(nv, &ch->sem);
	dma_buf_free(nv, &ch->errnotif);
	memset(ch, 0, sizeof(*ch));
	memset(&nv->vmm, 0, sizeof(nv->vmm));
}
