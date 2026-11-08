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

static void __maybe_unused rl_state(struct nvdrv_dev *nv, const char *when);

/* ---- GMMU page tables (GP100+ format, see bar1.c) ------------------------- */

#define PDE_VRAM		(1ULL << 1)
#define PTE_VALID		BIT_ULL(0)
#define PTE_APER_SYS_COH	(2ULL << 1)
#define PTE_VOL			BIT_ULL(3)
#define PTE_PRIV		BIT_ULL(5)
#define PTE_RO			BIT_ULL(6)

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

/* A zeroed 4K page-table page: carve-out for nvdrv's VM, the freeable pool for client VMs. */
static int pt_page(struct nvdrv_dev *nv, struct nv_vmm *vm, u64 *out)
{
	int ret;

	if (vm->external) {
		struct nv_pt *pt = kzalloc(sizeof(*pt), GFP_KERNEL);

		ret = pt ? nv_mem_get(nv, &pt->node, SZ_4K, SZ_4K) : -ENOMEM;
		if (ret) {
			kfree(pt);
			return ret;
		}
		list_add(&pt->head, &vm->pts);
		*out = pt->node.start;
	} else {
		ret = nv_vram_alloc(nv, SZ_4K, SZ_4K, out);
		if (ret)
			return ret;
	}
	nv_vram_fill(nv, *out, SZ_4K, 0, NULL, NULL);
	return 0;
}

/* Next-level table behind entry (addr), allocated on first use. */
static int pd_next(struct nvdrv_dev *nv, struct nv_vmm *vm, u64 entry, u64 *next)
{
	u64 e = rd64(nv, entry);
	int ret;

	if ((e >> 1) & 3) {
		*next = (e & ~0xfULL) << 4;
		return 0;
	}
	ret = pt_page(nv, vm, next);
	if (!ret)
		wr64(nv, entry, (*next >> 4) | PDE_VRAM);
	return ret;
}

/*
 * Map one 4 KiB page at va. 4K pages use PD0 entry [1] (the small page table, 512 x 8 B);
 * aperture: VRAM (pte_flags 0) or coherent sysmem (PTE_APER_SYS_COH | PTE_VOL).
 */
/* Write the 4K PTE for va (allocating the tables on the way); pte is the full entry. */
int nv_vmm_set_pte(struct nvdrv_dev *nv, struct nv_vmm *vm, u64 va, u64 pte)
{
	u64 pd2, pd1, pd0, pt;
	int ret;

	ret = pd_next(nv, vm, vm->root + ((va >> 47) & 3) * 8, &pd2);
	if (!ret)
		ret = pd_next(nv, vm, pd2 + ((va >> 38) & 511) * 8, &pd1);
	if (!ret)
		ret = pd_next(nv, vm, pd1 + ((va >> 29) & 511) * 8, &pd0);
	if (!ret)
		ret = pd_next(nv, vm, pd0 + ((va >> 21) & 255) * 16 + 8, &pt);
	if (ret)
		return ret;
	wr64(nv, pt + ((va >> 12) & 511) * 8, pte);
	return 0;
}

int nv_vmm_map_4k(struct nvdrv_dev *nv, struct nv_vmm *vm, u64 va, u64 pa, u64 pte_flags)
{
	return nv_vmm_set_pte(nv, vm, va, (pa >> 4) | PTE_VALID | pte_flags);
}

int nv_vmm_flush(struct nvdrv_dev *nv, struct nv_vmm *vm)
{
	u32 v;

	(void)nv_vram_rd32(nv, vm->root);	/* PRAMIN writes landed */
	nv_wr32(nv, MMU_INVALIDATE_PDB_LO, lower_32_bits(vm->root >> 8));
	nv_wr32(nv, MMU_INVALIDATE_PDB_HI, upper_32_bits(vm->root >> 8));
	nv_wr32(nv, MMU_INVALIDATE, BIT(31) | 0x1);	/* PAGE_ALL */
	return readl_poll_timeout(nv->bar0 + MMU_INVALIDATE, v, !(v & BIT(31)), 10, 2000000);
}

/* Map a VRAM range (4K granularity) at a fresh VA aligned to align; returns the VA. */
int nv_vmm_map_vram_flags(struct nvdrv_dev *nv, struct nv_vmm *vm, u64 vram, u64 size, u64 align,
			      u64 pte_flags, u64 *va)
{
	u64 off;
	int ret = 0;

	*va = ALIGN(vm->next, max_t(u64, align, SZ_4K));
	for (off = 0; off < size && !ret; off += SZ_4K)
		ret = nv_vmm_map_4k(nv, vm, *va + off, vram + off, pte_flags);
	vm->next = ALIGN(*va + size, SZ_2M);
	return ret;
}

static int nv_vmm_map_vram(struct nvdrv_dev *nv, struct nv_vmm *vm, u64 vram, u64 size, u64 *va)
{
	return nv_vmm_map_vram_flags(nv, vm, vram, size, SZ_4K, 0, va);
}

/* Map a coherent sysmem buffer (contiguous bus addresses) at a fresh VA. */
int nv_vmm_map_sysmem(struct nvdrv_dev *nv, struct nv_vmm *vm, const struct nv_dma_buf *b, u64 *va)
{
	u64 off;
	int ret = 0;

	*va = vm->next;
	for (off = 0; off < b->size && !ret; off += SZ_4K)
		ret = nv_vmm_map_4k(nv, vm, *va + off, b->dma + off, PTE_APER_SYS_COH | PTE_VOL);
	vm->next = ALIGN(vm->next + b->size, SZ_2M);
	return ret;
}

/*
 * Point a channel instance block at the address space (nouveau gp100_vmm_join +
 * gv100_vmm_join): PDB at 0x200 (VER2, 64 KiB big pages, VRAM), VA limit at 0x208,
 * subcontext 0 PDB at 0x2a0, subcontexts 1..63 invalid, valid mask at 0x298.
 */
void nv_vmm_join(struct nvdrv_dev *nv, struct nv_vmm *vm, u64 inst)
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
#define NVDRV_H_GR3D		0xd000c200
#define NVDRV_H_GRCOMPUTE	0xd000c201
#define AMPERE_B		0xc797
#define AMPERE_COMPUTE_B	0xc7c0
#define NV2080_ENGINE_TYPE_GR0	0x01
#define NV2080_CTRL_CMD_GPU_PROMOTE_CTX	0x2080012b
#define NV2080_CTRL_CMD_INTERNAL_STATIC_KGR_GET_CONTEXT_BUFFERS_INFO	0x20800a32
#define CHID_CE			1
#define CHID_GR			2

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
#define KCHAN_PRIVILEGE_MASK	0x3	/* USER = 0 */
#define NVOS04_FLAGS_PRIVILEGED	BIT(5)

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

	vm->h_client = nv->rm.h_client;
	vm->h_device = nv->rm.h_device;
	vm->h_subdevice = nv->rm.h_subdevice;
	ret = pt_page(nv, vm, &vm->root);
	if (!ret)
		ret = pd_next(nv, vm, vm->root + ((SPLIT_VAS_START >> 47) & 3) * 8, &pd2);
	if (!ret)
		ret = pd_next(nv, vm, pd2 + ((SPLIT_VAS_START >> 38) & 511) * 8, &pd1);
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

#define NV_VASPACE_FLAGS_EXTERNALLY_OWNED	BIT(3)
#define NV0080_CTRL_CMD_DMA_SET_PAGE_DIRECTORY	0x00801813
#define NV0080_CTRL_CMD_DMA_UNSET_PAGE_DIRECTORY	0x00801814

struct nv_set_page_directory {
	__le64 phys_address;
	__le32 num_entries, flags, h_vaspace, chid, sub_device_id, pasid;
};
static_assert(sizeof(struct nv_set_page_directory) == 32);

/*
 * A client GPU address space (r535_mmu_vaspace_new with external = true): its own RM
 * client/device/subdevice (h_client), an externally owned FERMI_VASPACE_A whose page
 * directory nvdrv writes and hands to RM with DMA_SET_PAGE_DIRECTORY. No RM-reserved
 * range; kernel objects (GPFIFOs, push buffers, GR context) go from kernel_va upwards.
 */
int nv_vmm_client_init(struct nvdrv_dev *nv, struct nv_vmm *vm, u32 h_client, u64 kernel_va)
{
	struct nv_vaspace_alloc a = { .flags = cpu_to_le32(NV_VASPACE_FLAGS_EXTERNALLY_OWNED) };
	struct nv_set_page_directory pd = {};
	int ret;

	lockdep_assert_held(&nv->gsp_lock);
	memset(vm, 0, sizeof(*vm));
	INIT_LIST_HEAD(&vm->pts);
	vm->external = true;
	vm->next = kernel_va;
	ret = nv_rm_client_alloc(nv, h_client);
	if (ret)
		return ret;
	vm->h_client = h_client;
	vm->h_device = NVDRV_H_DEVICE;
	vm->h_subdevice = NVDRV_H_SUBDEVICE;
	ret = pt_page(nv, vm, &vm->root);
	if (!ret)
		ret = nv_rm_alloc(nv, h_client, vm->h_device, NVDRV_H_VASPACE, FERMI_VASPACE_A, &a, sizeof(a));
	if (ret)
		goto fail;
	vm->handle = NVDRV_H_VASPACE;
	pd.phys_address = cpu_to_le64(vm->root);
	pd.num_entries = cpu_to_le32(4);		/* PD3: VA bits 48:47 */
	pd.flags = 0;					/* APERTURE_VIDMEM */
	pd.h_vaspace = cpu_to_le32(vm->handle);
	ret = nv_rm_control(nv, h_client, vm->h_device, NV0080_CTRL_CMD_DMA_SET_PAGE_DIRECTORY, &pd, sizeof(pd));
	if (!ret)
		return 0;
fail:
	nv_vmm_client_fini(nv, vm);
	return ret;
}

void nv_vmm_client_fini(struct nvdrv_dev *nv, struct nv_vmm *vm)
{
	struct nv_pt *pt, *tmp;

	lockdep_assert_held(&nv->gsp_lock);
	if (vm->h_client && nv->gsp_running) {
		if (vm->handle) {
			__le32 un[2] = { cpu_to_le32(vm->handle), 0 };

			nv_rm_control(nv, vm->h_client, vm->h_device, NV0080_CTRL_CMD_DMA_UNSET_PAGE_DIRECTORY,
				      un, sizeof(un));
		}
		nv_rm_free(nv, vm->h_client, vm->h_client);
	}
	if (vm->pts.next)
		list_for_each_entry_safe(pt, tmp, &vm->pts, head) {
			nv_mem_put(nv, &pt->node);
			kfree(pt);
		}
	memset(vm, 0, sizeof(*vm));
	INIT_LIST_HEAD(&vm->pts);
}

int nv_chid_get(struct nvdrv_dev *nv)
{
	unsigned long *map = nv->fifo.chids;
	int chid = find_next_zero_bit(map, 128, 1);

	if (chid >= 128)
		return -ENOSPC;
	set_bit(chid, map);
	return chid;
}

void nv_chid_put(struct nvdrv_dev *nv, u32 chid)
{
	clear_bit(chid, nv->fifo.chids);
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

		if (le32_to_cpu(data[ENGINE_INFO_TYPE_RM_ENGINE_TYPE]) == RM_ENGINE_TYPE_GR0) {
			gr_runlist = le32_to_cpu(data[ENGINE_INFO_TYPE_RUNLIST]);
			nv->fifo.gr_runlist = gr_runlist;
			nv->fifo.gr_rl_base = le32_to_cpu(data[ENGINE_INFO_TYPE_RUNLIST_PRI_BASE]);
		}
	}
	for (i = 0; i < n && i < 32; i++) {
		const __le32 *data = (const __le32 *)(t + 12 + i * 100);
		u32 type = le32_to_cpu(data[ENGINE_INFO_TYPE_RM_ENGINE_TYPE]);
		u32 rl = le32_to_cpu(data[ENGINE_INFO_TYPE_RUNLIST]);

		if (type >= RM_ENGINE_TYPE_COPY0 && type < RM_ENGINE_TYPE_COPY0 + 10 && rl == gr_runlist &&
		    !nv->fifo.gr_ce)
			nv->fifo.gr_ce = type - RM_ENGINE_TYPE_COPY0 + 1;	/* +1: 0 = none */

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

int nv_dma_buf_alloc(struct nvdrv_dev *nv, struct nv_dma_buf *b, size_t size)
{
	b->size = size;
	b->cpu = dma_alloc_coherent(&nv->pdev->dev, size, &b->dma, GFP_KERNEL);
	if (!b->cpu)
		return -ENOMEM;
	memset(b->cpu, 0, size);
	return 0;
}

void nv_dma_buf_free(struct nvdrv_dev *nv, struct nv_dma_buf *b)
{
	if (b->cpu)
		dma_free_coherent(&nv->pdev->dev, b->size, b->cpu, b->dma);
	b->cpu = NULL;
}

/* The USERD page all nvdrv channels share (0x200 per chid), mapped through BAR1. */
static int userd_page(struct nvdrv_dev *nv)
{
	struct nv_fifo *f = &nv->fifo;
	int ret;

	if (f->userd_cpu)
		return 0;
	ret = nv_vram_alloc(nv, SZ_64K, SZ_64K, &f->userd);
	if (ret)
		return ret;
	nv_vram_fill(nv, f->userd, SZ_64K, 0, NULL, NULL);
	f->userd_cpu = nv_bar1_map(nv, NV_FB_MAX, f->userd, SZ_64K);	/* BAR1 VA right after scanout */
	return f->userd_cpu ? 0 : -EIO;
}

/*
 * A privileged GPFIFO channel for one engine (r535_chan_alloc + r535_chan_ramfc_write):
 * host-provided instance block (VRAM), USERD slot, method buffer; sysmem GPFIFO, push
 * buffer and semaphore page mapped into nvdrv's VA space; then BIND and SCHEDULE.
 */
int nv_chan_create(struct nvdrv_dev *nv, struct nv_chan *ch, struct nv_vmm *vm, u32 handle, u32 chid,
		   u32 engine, u32 runlist, u32 rl_base)
{
	struct device *dev = &nv->pdev->dev;
	struct nv_gsp_static_info *si = &nv->gsp_info;
	struct nv_fifo *f = &nv->fifo;
	struct nv_channel_alloc *a;
	__le32 v32 = 0;
	u32 mthdbuf_size;
	u8 sched[2] = { 1, 0 };	/* NVA06F_CTRL_GPFIFO_SCHEDULE_PARAMS: bEnable, bSkipSubmit */
	int ret;

	ret = nv_rm_control(nv, si->h_internal_client, si->h_internal_subdevice,
			    NV2080_CTRL_CMD_CE_GET_FAULT_METHOD_BUFFER_SIZE, &v32, sizeof(v32));
	if (!ret)
		ret = userd_page(nv);
	if (!ret)
		ret = nv_vram_alloc(nv, CHAN_INST_SIZE, SZ_4K, &ch->inst);
	if (ret)
		return ret;
	mthdbuf_size = le32_to_cpu(v32);
	ch->vm = vm;
	ch->chid = chid;
	ch->engine = engine;
	ch->runlist = runlist;
	ch->rl_base = rl_base;
	nv_vram_fill(nv, ch->inst, CHAN_INST_SIZE, 0, NULL, NULL);
	nv_vmm_join(nv, vm, ch->inst);
	ch->userd_cpu = f->userd_cpu + chid * USERD_SIZE;
	memset_io(ch->userd_cpu, 0, USERD_SIZE);
	if ((ret = nv_dma_buf_alloc(nv, &ch->gpfifo, GPFIFO_ENTRIES * 8)) ||
	    (ret = nv_dma_buf_alloc(nv, &ch->pb, PUSHBUF_SIZE)) ||
	    (ret = nv_dma_buf_alloc(nv, &ch->mthdbuf, PAGE_ALIGN(mthdbuf_size))) ||
	    (ret = nv_dma_buf_alloc(nv, &ch->sem, SZ_4K)) ||
	    (ret = nv_dma_buf_alloc(nv, &ch->errnotif, SZ_4K)))
		return ret;
	if ((ret = nv_vmm_map_sysmem(nv, vm, &ch->gpfifo, &ch->gpfifo_va)) ||
	    (ret = nv_vmm_map_sysmem(nv, vm, &ch->pb, &ch->pb_va)) ||
	    (ret = nv_vmm_map_sysmem(nv, vm, &ch->sem, &ch->sem_va)) ||
	    (ret = nv_vmm_flush(nv, vm)))
		return ret;

	a = kzalloc(sizeof(*a), GFP_KERNEL);
	if (!a)
		return -ENOMEM;
	a->gpfifo_offset = cpu_to_le64(ch->gpfifo_va);
	a->gpfifo_entries = cpu_to_le32(GPFIFO_ENTRIES);
	/*
	 * Channels in client VMs run userspace push buffers: unprivileged (NVOS04
	 * PRIVILEGED_CHANNEL, bit 5, clear; internal PRIVILEGE_USER), as nouveau's user
	 * channels. Besides keeping privileged methods away from userspace, RM gives USER
	 * contexts the restricted priv access map, which is the one promoted for them.
	 */
	a->flags = cpu_to_le32((NVOS04_FLAGS_BASE & ~(vm->external ? NVOS04_FLAGS_PRIVILEGED : 0)) |
			       0 << NVOS04_RUNQUEUE_SHIFT |
			       (chid % 8) << NVOS04_USERD_IDX_SHIFT | (chid / 8) << NVOS04_USERD_PAGE_SHIFT);
	a->h_vaspace = cpu_to_le32(vm->handle);
	a->engine_type = cpu_to_le32(engine);
	a->instance_mem = (struct nv_memdesc){ cpu_to_le64(ch->inst), cpu_to_le64(CHAN_INST_SIZE), cpu_to_le32(2), cpu_to_le32(1) };
	a->userd_mem = (struct nv_memdesc){ cpu_to_le64(f->userd + chid * USERD_SIZE), cpu_to_le64(USERD_SIZE), cpu_to_le32(2), cpu_to_le32(1) };
	a->ramfc_mem = (struct nv_memdesc){ cpu_to_le64(ch->inst), cpu_to_le64(0x200), cpu_to_le32(2), cpu_to_le32(1) };
	a->mthdbuf_mem = (struct nv_memdesc){ cpu_to_le64(ch->mthdbuf.dma), cpu_to_le64(mthdbuf_size), cpu_to_le32(1), 0 };
	a->internal_flags = cpu_to_le32(KCHAN_INTERNAL_FLAGS & ~(vm->external ? KCHAN_PRIVILEGE_MASK : 0));
	a->error_notifier_mem = (struct nv_memdesc){ cpu_to_le64(ch->errnotif.dma), cpu_to_le64(SZ_4K), cpu_to_le32(1), 0 };
	ret = nv_rm_alloc(nv, vm->h_client, vm->h_device, handle, AMPERE_CHANNEL_GPFIFO_A, a, sizeof(*a));
	kfree(a);
	if (ret)
		return ret;
	ch->handle = handle;

	v32 = cpu_to_le32(engine);
	ret = nv_rm_control(nv, vm->h_client, ch->handle, NVA06F_CTRL_CMD_BIND, &v32, sizeof(v32));
	if (!ret)
		ret = nv_rm_control(nv, vm->h_client, ch->handle, NVA06F_CTRL_CMD_GPFIFO_SCHEDULE, sched, 2);
	if (ret)
		return ret;
	/*
	 * Doorbell token: the runlist's hardware doorbell ID (RUNLIST_DOORBELL_CONFIG at
	 * runlist base + 0x008, bits 31:16; nouveau ga100_runl_new / ga100_chan_doorbell_handle).
	 * GPFIFO_GET_WORK_SUBMIT_TOKEN returns INVALID_STATE for these privileged channels.
	 */
	ch->token = (nv_rd32(nv, rl_base + 0x008) >> 16) << 16 | chid;
	dev_info(dev, "fifo: channel %u (engine 0x%x, runlist %u) up: inst 0x%llx, GPFIFO va 0x%llx, token 0x%x\n",
		 chid, engine, runlist, ch->inst, ch->gpfifo_va, ch->token);
	return 0;
}

/* Free a client channel (RM preempts and unbinds it) and everything the host gave it. */
void nv_chan_destroy(struct nvdrv_dev *nv, struct nv_chan *ch)
{
	lockdep_assert_held(&nv->gsp_lock);
	if (ch->handle && nv->gsp_running)
		nv_rm_free(nv, ch->vm->h_client, ch->handle);
	nv_dma_buf_free(nv, &ch->gpfifo);
	nv_dma_buf_free(nv, &ch->pb);
	nv_dma_buf_free(nv, &ch->mthdbuf);
	nv_dma_buf_free(nv, &ch->sem);
	nv_dma_buf_free(nv, &ch->errnotif);
	nv_mem_put(nv, &ch->ctx[0]);
	nv_mem_put(nv, &ch->ctx[1]);
	if (ch->chid)
		nv_chid_put(nv, ch->chid);
	memset(ch, 0, sizeof(*ch));
}

/* The copy-engine channel: a CE with physical engines on a runlist of its own. */
static int ce_chan_init(struct nvdrv_dev *nv, struct nv_chan *ch)
{
	struct { __le32 version, engine_type; } ce;
	u32 runlist, rl_base, n;
	int ret;

	ret = ce_pick(nv, &runlist, &n, &rl_base);
	if (!ret)
		ret = nv_chan_create(nv, ch, &nv->vmm, NVDRV_H_CHAN(0), CHID_CE, NV2080_ENGINE_TYPE_COPY0 + n, runlist, rl_base);
	if (ret)
		return ret;
	ch->ce = n;
	ce.version = cpu_to_le32(1);
	ce.engine_type = cpu_to_le32(NV2080_ENGINE_TYPE_COPY0 + n);
	return nv_rm_alloc(nv, nv->rm.h_client, ch->handle, NVDRV_H_CE(0), AMPERE_DMA_COPY_B, &ce, sizeof(ce));
}

/* ---- push buffer and submission -------------------------------------------- */

void nv_push(struct nv_chan *ch, u32 subc, u32 mthd, const u32 *data, u32 n)
{
	u32 *pb = ch->pb.cpu;

	pb[ch->pb_put++] = 0x20000000 | n << 16 | subc << 13 | mthd >> 2;	/* incrementing method */
	while (n--)
		pb[ch->pb_put++] = *data++;
}

void nv_push1(struct nv_chan *ch, u32 subc, u32 mthd, u32 v)
{
	nv_push(ch, subc, mthd, &v, 1);
}

/* Submit everything pushed since the last kick as one GPFIFO entry. */
void nv_kick(struct nvdrv_dev *nv, struct nv_chan *ch)
{
	u32 spins = 0;

	/* room for one more GPFIFO entry */
	while ((ch->gp_put + 1) % GPFIFO_ENTRIES == ioread32(ch->userd_cpu + USERD_GP_GET) && spins++ < 200000)
		udelay(10);
	u64 va = ch->pb_va + ch->pb_start * 4;
	u32 len = ch->pb_put - ch->pb_start, *gp = ch->gpfifo.cpu;

	gp[ch->gp_put * 2] = lower_32_bits(va) & ~3u;
	gp[ch->gp_put * 2 + 1] = (upper_32_bits(va) & 0xff) | len << 10;
	ch->gp_put = (ch->gp_put + 1) % GPFIFO_ENTRIES;
	ch->pb_start = ch->pb_put;
	/*
	 * Kernel push buffers wrap once 3/4 full: by then the GPFIFO (512 entries) has been
	 * waited on far more often than the oldest segment could still be pending.
	 */
	if (ch->pb_put > PUSHBUF_SIZE / 4 * 3 / 4)
		ch->pb_put = ch->pb_start = 0;
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

	nv_push(ch, SUBC_CE, C7B5_OFFSET_IN_UPPER, in_out, 4);
	nv_push1(ch, SUBC_CE, C7B5_LINE_LENGTH_IN, bytes);
	nv_push(ch, SUBC_CE, C7B5_SET_SEMAPHORE_A, sem, 3);
	nv_push1(ch, SUBC_CE, C7B5_LAUNCH_DMA, LAUNCH_COPY_SEM);
}

/* ---- graphics engine context (nouveau r535 gr.c) --------------------------- */

#define CTXPROP_COUNT		0x1a	/* NV0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_ENGINE_ID_COUNT */
#define PROMOTE_ID_MAIN		0
#define PROMOTE_ID_ATTRIBUTE_CB	5
#define PROMOTE_ID_PRIV_ACCESS_MAP	10
#define PROMOTE_ID_UNRESTRICTED_PRIV_ACCESS_MAP	11

struct nv_promote_entry {
	__le64 gpu_phys_addr, gpu_virt_addr, size;
	__le32 phys_attr;
	__le16 buffer_id;
	u8 initialize, nonmapped;
};

struct nv_promote_ctx {
	__le32 engine_type, h_client, chid, h_chan_client, h_object, h_virt_memory;
	__le64 virt_address, size;
	__le32 entry_count, pad;
	struct nv_promote_entry entry[16];
};
static_assert(sizeof(struct nv_promote_ctx) == 560);

/*
 * RM's GR context-buffer table (engine context property id -> promote buffer id), with
 * nouveau's flags: global (shared by all channels), init (RM initialises it through its
 * physical address), ro (mapped read-only).
 */
static const struct { u8 prop, id; bool global, init, ro; } gr_ctxbufs[] = {
	{ 0x00, PROMOTE_ID_MAIN,	false, true,  false },	/* GRAPHICS */
	{ 0x10, 2,			false, true,  false },	/* GRAPHICS_PATCH */
	{ 0x11, 3,			true,  false, false },	/* GRAPHICS_BUNDLE_CB */
	{ 0x0d, 4,			true,  false, false },	/* GRAPHICS_PAGEPOOL */
	{ 0x13, PROMOTE_ID_ATTRIBUTE_CB, true, false, false },	/* GRAPHICS_ATTRIBUTE_CB */
	{ 0x14, 6,			true,  false, false },	/* GRAPHICS_RTV_CB_GLOBAL */
	{ 0x17, 9,			true,  true,  false },	/* GRAPHICS_FECS_EVENT */
	{ 0x18, PROMOTE_ID_PRIV_ACCESS_MAP, true, true, true },	/* GRAPHICS_PRIV_ACCESS_MAP */
};

/* Allocate and map one context buffer; nouveau's size, page and alignment rules. */
static int gr_ctxbuf_add(struct nvdrv_dev *nv, u16 id, u32 size, bool init, bool ro, bool nonmapped)
{
	struct nv_fifo *f = &nv->fifo;
	u32 page, align;
	u64 vram, va = 0;
	int ret;

	if (f->ctxbuf_nr == NV_GR_MAX_CTXBUFS)
		return -ENOSPC;
	if (id == PROMOTE_ID_MAIN)
		size = ALIGN(size, SZ_4K) + 64 * SZ_4K;	/* per-subcontext headers */
	page = size >= SZ_2M ? SZ_2M : size >= SZ_64K ? SZ_64K : SZ_4K;
	align = id == PROMOTE_ID_ATTRIBUTE_CB ? roundup_pow_of_two(size) : page;
	size = ALIGN(size, SZ_4K);
	ret = nv_vram_alloc(nv, size, page, &vram);
	if (ret)
		return ret;
	if (init)
		nv_vram_fill(nv, vram, size, 0, NULL, NULL);
	if (!nonmapped) {
		ret = nv_vmm_map_vram_flags(nv, &nv->vmm, vram, size, align, PTE_PRIV | (ro ? PTE_RO : 0), &va);
		if (ret)
			return ret;
	}
	f->ctxbuf[f->ctxbuf_nr].vram = vram;
	f->ctxbuf[f->ctxbuf_nr].va = va;
	f->ctxbuf[f->ctxbuf_nr].size = size;
	f->ctxbuf[f->ctxbuf_nr].id = id;
	f->ctxbuf[f->ctxbuf_nr].nonmapped = nonmapped;
	f->ctxbuf_nr++;
	dev_info(&nv->pdev->dev, "gr: ctxbuf %2u: %8u KiB at VRAM 0x%llx, va 0x%llx%s%s\n", id, size >> 10, vram, va,
		 init ? ", init" : "", nonmapped ? ", unmapped" : ro ? ", ro" : "");
	return 0;
}

/*
 * Give RM the GR context of the GR channel (r535_gr_oneinit + r535_gr_promote_ctx with
 * golden = true): every buffer is allocated here, initialisable ones are passed with their
 * physical address. Allocating the 3D class on the channel then makes RM build the golden
 * context. nvdrv has one GR channel, so the golden channel is also the working channel.
 */
static int gr_ctx_promote(struct nvdrv_dev *nv, struct nv_chan *ch)
{
	struct nv_gsp_static_info *si = &nv->gsp_info;
	struct nv_fifo *f = &nv->fifo;
	struct nv_promote_ctx *pc;
	__le32 *info;
	int i, j, ret;

	info = kzalloc(8 * CTXPROP_COUNT * 8, GFP_KERNEL);	/* [GR_MAX_ENGINES][COUNT] { size, align } */
	pc = kzalloc(sizeof(*pc), GFP_KERNEL);
	ret = info && pc ? 0 : -ENOMEM;
	if (!ret)
		ret = nv_rm_control(nv, si->h_internal_client, si->h_internal_subdevice,
				    NV2080_CTRL_CMD_INTERNAL_STATIC_KGR_GET_CONTEXT_BUFFERS_INFO,
				    info, 8 * CTXPROP_COUNT * 8);
	for (i = 0; !ret && i < ARRAY_SIZE(gr_ctxbufs); i++) {
		u32 size = le32_to_cpu(info[gr_ctxbufs[i].prop * 2]);
		bool pam = gr_ctxbufs[i].id == PROMOTE_ID_PRIV_ACCESS_MAP;

		ret = gr_ctxbuf_add(nv, gr_ctxbufs[i].id, size, gr_ctxbufs[i].init, gr_ctxbufs[i].ro, pam);
		if (!ret && pam)	/* plus a mapped, unrestricted copy */
			ret = gr_ctxbuf_add(nv, PROMOTE_ID_UNRESTRICTED_PRIV_ACCESS_MAP, size, true, true, false);
	}
	if (!ret)
		ret = nv_vmm_flush(nv, &nv->vmm);
	if (ret)
		goto out;

	pc->engine_type = cpu_to_le32(NV2080_ENGINE_TYPE_GR0);
	pc->h_chan_client = cpu_to_le32(nv->rm.h_client);
	pc->h_object = cpu_to_le32(ch->handle);
	for (i = 0; i < f->ctxbuf_nr; i++) {
		struct nv_promote_entry *e = &pc->entry[i];
		bool init = f->ctxbuf[i].id == PROMOTE_ID_UNRESTRICTED_PRIV_ACCESS_MAP;

		for (j = 0; j < ARRAY_SIZE(gr_ctxbufs); j++)
			if (gr_ctxbufs[j].id == f->ctxbuf[i].id)
				init = gr_ctxbufs[j].init;
		e->buffer_id = cpu_to_le16(f->ctxbuf[i].id);
		e->initialize = init;
		e->nonmapped = f->ctxbuf[i].nonmapped;
		e->gpu_virt_addr = cpu_to_le64(f->ctxbuf[i].va);
		if (init) {
			e->gpu_phys_addr = cpu_to_le64(f->ctxbuf[i].vram);
			e->size = cpu_to_le64(f->ctxbuf[i].size);
			e->phys_attr = cpu_to_le32(4);
		}
	}
	pc->entry_count = cpu_to_le32(f->ctxbuf_nr);
	ret = nv_rm_control(nv, nv->rm.h_client, nv->rm.h_subdevice, NV2080_CTRL_CMD_GPU_PROMOTE_CTX, pc, sizeof(*pc));
	if (!ret)	/* golden context init happens in RM on the first GR object */
		ret = nv_rm_alloc(nv, nv->rm.h_client, ch->handle, NVDRV_H_GR3D, AMPERE_B, NULL, 0);
	if (!ret)
		ret = nv_rm_alloc(nv, nv->rm.h_client, ch->handle, NVDRV_H_GRCOMPUTE, AMPERE_COMPUTE_B, NULL, 0);
	dev_info(&nv->pdev->dev, "gr: %u context buffers promoted, AMPERE_B + AMPERE_COMPUTE_B on channel %u: %d\n",
		 f->ctxbuf_nr, ch->chid, ret);
out:
	kfree(pc);
	kfree(info);
	return ret;
}

/*
 * GR context for a client channel (r535_gr_chan_new: promote with golden = false):
 * fresh MAIN and PATCH buffers from the pool (RM initialises them from the golden
 * context), the global buffers from the golden setup mapped into the client's VM,
 * PRIV_ACCESS_MAP mapped this time and its unrestricted copy left out.
 */
int nv_gr_chan_ctx(struct nvdrv_dev *nv, struct nv_chan *ch)
{
	struct nv_fifo *f = &nv->fifo;
	struct nv_vmm *vm = ch->vm;
	struct nv_promote_ctx *pc;
	int i, n = 0, ret = 0;

	lockdep_assert_held(&nv->gsp_lock);
	if (!f->gr_ready)
		return -ENODEV;
	pc = kzalloc(sizeof(*pc), GFP_KERNEL);
	if (!pc)
		return -ENOMEM;
	for (i = 0; i < f->ctxbuf_nr && !ret; i++) {
		struct nv_promote_entry *e = &pc->entry[n];
		u16 id = f->ctxbuf[i].id;
		u64 size = f->ctxbuf[i].size, va, vram = f->ctxbuf[i].vram, align;
		bool ro = false;
		int j, k = id == PROMOTE_ID_MAIN ? 0 : id == 2 ? 1 : -1;

		if (id == PROMOTE_ID_UNRESTRICTED_PRIV_ACCESS_MAP)
			continue;
		for (j = 0; j < ARRAY_SIZE(gr_ctxbufs); j++)
			if (gr_ctxbufs[j].id == id)
				ro = gr_ctxbufs[j].ro;
		align = id == PROMOTE_ID_ATTRIBUTE_CB ? roundup_pow_of_two(size) :
			size >= SZ_2M ? SZ_2M : size >= SZ_64K ? SZ_64K : SZ_4K;
		if (k >= 0) {	/* per-channel */
			ret = nv_mem_get(nv, &ch->ctx[k], size, align);
			if (ret)
				break;
			vram = ch->ctx[k].start;
			nv_vram_fill(nv, vram, size, 0, NULL, NULL);
			e->initialize = 1;
			e->gpu_phys_addr = cpu_to_le64(vram);
			e->size = cpu_to_le64(size);
			e->phys_attr = cpu_to_le32(4);
		}
		ret = nv_vmm_map_vram_flags(nv, vm, vram, size, align, NV_PTE_PRIV | (ro ? NV_PTE_RO : 0), &va);
		e->buffer_id = cpu_to_le16(id);
		e->gpu_virt_addr = cpu_to_le64(va);
		n++;
	}
	if (!ret)
		ret = nv_vmm_flush(nv, vm);
	if (!ret) {
		pc->engine_type = cpu_to_le32(NV2080_ENGINE_TYPE_GR0);
		pc->h_chan_client = cpu_to_le32(vm->h_client);
		pc->h_object = cpu_to_le32(ch->handle);
		pc->entry_count = cpu_to_le32(n);
		ret = nv_rm_control(nv, vm->h_client, vm->h_subdevice, NV2080_CTRL_CMD_GPU_PROMOTE_CTX, pc, sizeof(*pc));
	}
	kfree(pc);
	return ret;
}

#define SUBC_3D			0
#define SUBC_COMPUTE		1
#define C7C0_SET_REPORT_SEMAPHORE_A	0x1b00
#define REPORT_SEM_RELEASE_1WORD	(1u << 28)	/* OPERATION RELEASE, STRUCTURE_SIZE ONE_WORD */

static int gr_chan_init(struct nvdrv_dev *nv, struct nv_chan *ch)
{
	struct nv_fifo *f = &nv->fifo;
	int ret;

	if (!f->gr_rl_base)
		return -ENODEV;
	ret = nv_chan_create(nv, ch, &nv->vmm, NVDRV_H_CHAN(1), CHID_GR, NV2080_ENGINE_TYPE_GR0, f->gr_runlist, f->gr_rl_base);
	if (!ret)
		ret = gr_ctx_promote(nv, ch);
	if (ret)
		return ret;
	nv_push1(ch, SUBC_3D, 0, AMPERE_B);		/* SET_OBJECT */
	nv_push1(ch, SUBC_COMPUTE, 0, AMPERE_COMPUTE_B);
	nv_kick(nv, ch);
	f->gr_ready = true;
	return 0;
}

/* The compute engine writes a semaphore: proof that GR runs our channel's context. */
int nv_gr_test(struct nvdrv_dev *nv)
{
	struct nv_chan *ch = &nv->grch;
	u32 sem, seq, *n;
	int ret;

	mutex_lock(&nv->gsp_lock);
	ret = nv_fifo_init(nv);
	if (ret)
		goto out;
	seq = ++ch->seq;
	{
		u32 d[4] = { upper_32_bits(ch->sem_va), lower_32_bits(ch->sem_va), seq, REPORT_SEM_RELEASE_1WORD };

		nv_push(ch, SUBC_COMPUTE, C7C0_SET_REPORT_SEMAPHORE_A, d, 4);
	}
	nv_kick(nv, ch);
	ret = read_poll_timeout(READ_ONCE, sem, sem == seq, 50, 2000000, false, *(u32 *)ch->sem.cpu);
	n = ch->errnotif.cpu;
	dev_info(&nv->pdev->dev, "gr test: compute semaphore %u/%u (%d), GP_GET %u/%u, error notifier %u/0x%x\n",
		 sem, seq, ret, ioread32(ch->userd_cpu + USERD_GP_GET), ch->gp_put, n[2], n[3] >> 16);
out:
	mutex_unlock(&nv->gsp_lock);
	return ret;
}

/*
 * Zero [vram, vram + size) with the copy engine in physical mode (constant remap), and
 * wait for it. Caller holds gsp_lock (nvdrv's CE channel).
 */
#define C7B5_SET_DST_PHYS_MODE	0x264
#define C7B5_OFFSET_OUT_UPPER	0x408
#define C7B5_SET_REMAP_CONST_A	0x700
#define C7B5_SET_REMAP_COMPONENTS	0x708
#define REMAP_CONST_A_4BYTE	0x30004		/* DST_X = CONST_A, 4-byte components, 1 src/dst */
#define LAUNCH_CLEAR_PHYS_SEM	0x258e		/* LAUNCH_COPY_SEM | REMAP | DST_TYPE PHYSICAL */
int nv_ce_clear(struct nvdrv_dev *nv, u64 vram, u64 size)
{
	struct nv_chan *ch = &nv->chan;
	u32 seq = 0, sem;
	u64 off;

	lockdep_assert_held(&nv->gsp_lock);
	if (!ch->handle || (vram | size) & 3)
		return -ENODEV;
	nv_push1(ch, SUBC_CE, C7B5_SET_REMAP_CONST_A, 0);
	nv_push1(ch, SUBC_CE, C7B5_SET_REMAP_COMPONENTS, REMAP_CONST_A_4BYTE);
	nv_push1(ch, SUBC_CE, C7B5_SET_DST_PHYS_MODE, 0);	/* LOCAL_FB */
	for (off = 0; off < size; off += SZ_1G) {
		u64 dst = vram + off, n = min_t(u64, size - off, SZ_1G);
		u32 out[2] = { upper_32_bits(dst), lower_32_bits(dst) };
		u32 semd[3];

		seq = ++ch->seq;
		semd[0] = upper_32_bits(ch->sem_va);
		semd[1] = lower_32_bits(ch->sem_va);
		semd[2] = seq;
		nv_push(ch, SUBC_CE, C7B5_OFFSET_OUT_UPPER, out, 2);
		nv_push1(ch, SUBC_CE, C7B5_LINE_LENGTH_IN, n / 4);
		nv_push(ch, SUBC_CE, C7B5_SET_SEMAPHORE_A, semd, 3);
		nv_push1(ch, SUBC_CE, C7B5_LAUNCH_DMA, LAUNCH_CLEAR_PHYS_SEM);
	}
	nv_kick(nv, ch);
	return read_poll_timeout(READ_ONCE, sem, (s32)(sem - seq) >= 0, 20, 2000000, false, *(u32 *)ch->sem.cpu);
}

static void __maybe_unused rl_state(struct nvdrv_dev *nv, const char *when)
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
	if (nv->fifo.gr_ready)
		return 0;
	if (!nv->rm.h_device)
		return -ENODEV;
	ret = nv_bar1_init(nv);
	set_bit(CHID_CE, nv->fifo.chids);
	set_bit(CHID_GR, nv->fifo.chids);
	if (!ret && !nv->vmm.handle)
		ret = vmm_init(nv, &nv->vmm);
	if (!ret && !nv->mem.ready) {	/* the VRAM pool takes its range from the carve-out */
		struct drm_mm_node n;

		ret = nv_mem_get(nv, &n, SZ_4K, 0);
		if (!ret)
			nv_mem_put(nv, &n);
	}
	if (!ret && !nv->chan.handle) {
		ret = ce_chan_init(nv, &nv->chan);
		if (!ret) {
			nv_push1(&nv->chan, SUBC_CE, 0, AMPERE_DMA_COPY_B);	/* SET_OBJECT */
			nv_kick(nv, &nv->chan);
		}
	}
	if (!ret && !nv->grch.handle)
		ret = gr_chan_init(nv, &nv->grch);
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
	if (!nv->chan.handle || !e->fb)
		goto out;	/* a GR setup failure alone does not stop the CE test */
	ret = nv_dma_buf_alloc(nv, &src, SZ_1M);
	if (ret)
		goto out;
	for (s = src.cpu, i = 0; i < SZ_1M / 4; i++)
		s[i] = 0xff000000 | (i * 2654435761u >> 8);	/* hashed colours */
	dst = e->fb + (e->fb_size / 2 & ~(u64)(SZ_1M - 1));
	ret = nv_vmm_map_sysmem(nv, &nv->vmm, &src, &src_va);
	if (!ret)
		ret = nv_vmm_map_vram(nv, &nv->vmm, dst, SZ_1M, &dst_va);
	if (!ret)
		ret = nv_vmm_flush(nv, &nv->vmm);
	if (ret)
		goto out_free;

	seq = ++ch->seq;
	ce_copy(ch, dst_va, src_va, SZ_1M, seq);
	nv_kick(nv, ch);
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
	nv_dma_buf_free(nv, &src);	/* GPU done with it (or the GPU gets reset before reuse) */
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
	struct nv_chan *chans[] = { &nv->chan, &nv->grch };
	int i;

	for (i = 0; i < ARRAY_SIZE(chans); i++) {
		struct nv_chan *ch = chans[i];

		nv_dma_buf_free(nv, &ch->gpfifo);
		nv_dma_buf_free(nv, &ch->pb);
		nv_dma_buf_free(nv, &ch->mthdbuf);
		nv_dma_buf_free(nv, &ch->sem);
		nv_dma_buf_free(nv, &ch->errnotif);
		memset(ch, 0, sizeof(*ch));
	}
	if (nv->fifo.userd_cpu)
		iounmap(nv->fifo.userd_cpu);
	memset(&nv->fifo, 0, sizeof(nv->fifo));
	memset(&nv->vmm, 0, sizeof(nv->vmm));
}
