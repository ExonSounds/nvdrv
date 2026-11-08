// SPDX-License-Identifier: GPL-2.0
/*
 * BAR1: CPU access to VRAM through the GPU MMU.
 *
 * With GSP-RM, BAR1 runs in virtual mode and GSP-RM owns the root page directory
 * (GspStaticConfigInfo.bar1PdeBase). Like nouveau (nvkm/subdev/gsp/rm/r535/bar.c,
 * r535_bar_bar1_init), nvdrv takes over entry 0 of that root and hangs its own
 * PD2 -> PD1 -> PD0 -> 64 KiB page tables below it, all in nvdrv's VRAM carve-out.
 *
 * Page-table formats are GP100+ (nouveau nvkm/subdev/mmu/vmmgp100.c):
 *   levels (VA bits)  PD3/root 48:47 (GSP's page), PD2 46:38, PD1 37:29,
 *                     PD0 28:21 (16-byte dual entries: [0] 64K table, [1] 4K table;
 *                     one PD0 page per 512 MiB, up to the whole BAR: 8 GiB with ReBAR),
 *                     PT 20:16 (32 x 8-byte PTEs for 64 KiB pages)
 *   PDE  = (table address >> 4) | APERTURE_VIDEO_MEMORY (1 << 1)
 *   PTE  = (page address >> 4)  | VALID, aperture 0 = video memory, kind 0 = pitch
 * TLB invalidate: tu102_vmm_flush (NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE*).
 */
#include <linux/io.h>
#include <linux/ktime.h>
#include <linux/iopoll.h>
#include <linux/lockdep.h>

#include "nvdrv.h"

#define MMU_INVALIDATE_PDB_LO	0x00b830a0
#define MMU_INVALIDATE_PDB_HI	0x00b830a4
#define MMU_INVALIDATE		0x00b830b0
#define   INVALIDATE_TRIGGER	BIT(31)
#define   INVALIDATE_ALL	0x7	/* PAGE_ALL | HUB_ONLY | ALL_PDB */

#define PDE_VRAM		(1ULL << 1)
#define PTE_VALID		BIT_ULL(0)

#define BAR1_PT_BYTES		0x100		/* 32 PTEs */
#define BAR1_PD0_SPAN		(256ULL << 21)	/* one PD0 page: 256 x 2 MiB = 512 MiB */
#define BAR1_MAX_PD0		16		/* 8 GiB: all of a Resizable BAR on GA104 */
/* PD2, PD1, n PD0 pages, then 256 page tables per PD0 page */
#define BAR1_TABLES_SIZE(n)	((2 + (n)) * SZ_4K + (n) * 256 * BAR1_PT_BYTES)

static void wr64(struct nvdrv_dev *nv, u64 addr, u64 v)
{
	nv_vram_wr32(nv, addr, lower_32_bits(v));
	nv_vram_wr32(nv, addr + 4, upper_32_bits(v));
}

static int bar1_invalidate(struct nvdrv_dev *nv)
{
	u64 pdb = nv->gsp_info.bar1_pde_base;
	u32 v;

	(void)nv_vram_rd32(nv, nv->bar1.tables);	/* PRAMIN writes have landed */
	nv_wr32(nv, MMU_INVALIDATE_PDB_LO, lower_32_bits(pdb >> 8));
	nv_wr32(nv, MMU_INVALIDATE_PDB_HI, upper_32_bits(pdb >> 8));
	nv_wr32(nv, MMU_INVALIDATE, INVALIDATE_TRIGGER | INVALIDATE_ALL);
	return readl_poll_timeout(nv->bar0 + MMU_INVALIDATE, v, !(v & INVALIDATE_TRIGGER), 10, 2000000);
}

int nv_bar1_init(struct nvdrv_dev *nv)
{
	struct nv_bar1 *b = &nv->bar1;
	u64 root = nv->gsp_info.bar1_pde_base, pd2, pd1, i;
	int ret;

	lockdep_assert_held(&nv->gsp_lock);
	if (b->tables)
		return 0;
	if (!nv->gsp_info.valid || !root)
		return -ENODEV;
	b->npd0 = clamp_t(u64, DIV_ROUND_UP(pci_resource_len(nv->pdev, 1), BAR1_PD0_SPAN), 1, BAR1_MAX_PD0);
	ret = nv_vram_alloc(nv, BAR1_TABLES_SIZE(b->npd0), SZ_64K, &b->tables);
	if (ret)
		return ret;
	nv_vram_fill(nv, b->tables, BAR1_TABLES_SIZE(b->npd0), 0, NULL, NULL);
	pd2 = b->tables;
	pd1 = b->tables + SZ_4K;
	b->pd0 = b->tables + 2 * SZ_4K;
	b->pts = b->pd0 + b->npd0 * SZ_4K;

	for (i = 0; i < b->npd0; i++)	/* PD1 entry i: BAR1 VA [i * 512 MiB, +512 MiB) */
		wr64(nv, pd1 + i * 8, ((b->pd0 + i * SZ_4K) >> 4) | PDE_VRAM);
	wr64(nv, pd2, (pd1 >> 4) | PDE_VRAM);
	b->old_root0 = (u64)nv_vram_rd32(nv, root + 4) << 32 | nv_vram_rd32(nv, root);
	wr64(nv, root, (pd2 >> 4) | PDE_VRAM);
	ret = bar1_invalidate(nv);
	dev_info(&nv->pdev->dev, "bar1: tables at VRAM 0x%llx, root 0x%llx entry 0: 0x%llx -> 0x%llx, invalidate %d\n",
		 b->tables, root, b->old_root0, (pd2 >> 4) | PDE_VRAM, ret);
	if (ret)
		return ret;

	b->size = min_t(u64, pci_resource_len(nv->pdev, 1), b->npd0 * BAR1_PD0_SPAN);
	return 0;
}

/*
 * Point BAR1 [va, va + size) at [vram, vram + size), or unmap it (vram == NV_BAR1_UNMAP).
 * All 64 KiB aligned. Caller holds gsp_lock (PRAMIN).
 */
int nv_bar1_set(struct nvdrv_dev *nv, u64 va, u64 vram, u64 size)
{
	struct nv_bar1 *b = &nv->bar1;
	bool unmap = vram == NV_BAR1_UNMAP;
	u64 off;

	lockdep_assert_held(&nv->gsp_lock);
	if (!b->tables || (va | size | (unmap ? 0 : vram)) & (SZ_64K - 1) || va + size > b->size)
		return -EINVAL;
	for (off = 0; off < size; off += SZ_64K) {
		u64 v = va + off, pt = b->pts + (v >> 21) * BAR1_PT_BYTES;

		/* PD0 dual entry ([0] 64 KiB table, [1] 4 KiB table: none), PD0 page v >> 29; once per 2 MiB */
		if (!off || !(v & (SZ_2M - 1)))
			wr64(nv, b->pd0 + (v >> 29) * SZ_4K + ((v >> 21) & 255) * 16, (pt >> 4) | PDE_VRAM);
		wr64(nv, pt + ((v >> 16) & 31) * 8, unmap ? 0 : ((vram + off) >> 4) | PTE_VALID);
	}
	return bar1_invalidate(nv);
}

/*
 * With a large (resizable) BAR: map the whole VRAM pool once at BAR1 offset = VRAM address,
 * so every pool object is CPU-reachable without per-object BAR1 space or PTE updates. The
 * pool starts above the scanout surface and USERD at the bottom of BAR1.
 */
void nv_bar1_map_pool(struct nvdrv_dev *nv)
{
	struct nv_bar1 *b = &nv->bar1;
	ktime_t t0 = ktime_get();
	int ret;

	lockdep_assert_held(&nv->gsp_lock);
	if (b->pool_mapped || !nv->mem.ready || !b->tables || nv->mem.base + nv->mem.size > b->size ||
	    nv->mem.base < NV_FB_MAX + SZ_64K)
		return;
	ret = nv_bar1_set(nv, nv->mem.base, nv->mem.base, nv->mem.size);
	b->pool_mapped = !ret;
	dev_info(&nv->pdev->dev, "bar1: VRAM pool 0x%llx+%llu MiB mapped 1:1 (%d, %lld ms)\n", nv->mem.base,
		 nv->mem.size >> 20, ret, ktime_ms_delta(ktime_get(), t0));
}

/* Map and return a write-combined CPU mapping of it. */
void __iomem *nv_bar1_map(struct nvdrv_dev *nv, u64 va, u64 vram, u64 size)
{
	if (nv_bar1_set(nv, va, vram, size))
		return NULL;
	return ioremap_wc(pci_resource_start(nv->pdev, 1) + va, size);
}
