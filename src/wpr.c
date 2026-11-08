// SPDX-License-Identifier: GPL-2.0
/*
 * GspFwWprMeta (Phase 5): the boot contract between this driver, Booter and
 * the GSP bootloader. Filled as kgspCalculateFbLayout_TU102() does (570.144)
 * and placed in one coherent sysmem page (kgspAllocBootArgs_TU102: 4K, 4K-aligned).
 * Host memory only; nothing is written to the GPU here.
 */
#include <linux/dma-mapping.h>
#include <linux/stddef.h>

#include "nvdrv.h"

static_assert(sizeof(struct gsp_fw_wpr_meta) == 256);
static_assert(offsetof(struct gsp_fw_wpr_meta, boot_count) == 200);
static_assert(offsetof(struct gsp_fw_wpr_meta, sysmem_addr_crash_report_queue) == 224);
static_assert(offsetof(struct gsp_fw_wpr_meta, flags) == 241);
static_assert(offsetof(struct gsp_fw_wpr_meta, pmu_reserved_size) == 244);
static_assert(offsetof(struct gsp_fw_wpr_meta, verified) == 248);

int nv_wpr_meta_build(struct nvdrv_dev *nv)
{
	struct device *dev = &nv->pdev->dev;
	struct nv_gsp_fw *g = &nv->gsp_fw;
	struct nv_fb_layout *fb = &nv->fb;
	struct gsp_fw_wpr_meta *m;
	int ret;

	if (!g->radix3.root || !g->bl || !g->sig)
		return -ENODATA;

	ret = nv_fb_layout_gsp(nv, g->bl_size, g->radix3.size);
	if (ret)
		return ret;

	m = dma_alloc_coherent(dev, SZ_4K, &nv->wpr_meta_dma, GFP_KERNEL);
	if (!m)
		return -ENOMEM;
	memset(m, 0, SZ_4K);

	/* Sysmem: where Booter DMAs the images from */
	m->sysmem_addr_radix3_elf = cpu_to_le64(g->radix3.root);
	m->size_radix3_elf = cpu_to_le64(g->radix3.size);
	m->sysmem_addr_bootloader = cpu_to_le64(g->bl_dma);
	m->size_bootloader = cpu_to_le64(g->bl_size);
	m->bootloader_code_offset = cpu_to_le64(g->bl_desc.monitorCodeOffset);
	m->bootloader_data_offset = cpu_to_le64(g->bl_desc.monitorDataOffset);
	m->bootloader_manifest_offset = cpu_to_le64(g->bl_desc.manifestOffset);
	m->sysmem_addr_signature = cpu_to_le64(g->sig_dma);
	m->size_signature = cpu_to_le64(ALIGN(g->sig_size, 256));

	/* FB layout */
	m->gsp_fw_rsvd_start = cpu_to_le64(fb->rsvd_start);
	m->non_wpr_heap_offset = cpu_to_le64(fb->non_wpr_heap_offset);
	m->non_wpr_heap_size = cpu_to_le64(fb->non_wpr_heap_size);
	m->gsp_fw_wpr_start = cpu_to_le64(fb->wpr_start);
	m->gsp_fw_heap_offset = cpu_to_le64(fb->heap_offset);
	m->gsp_fw_heap_size = cpu_to_le64(fb->heap_size);
	m->gsp_fw_offset = cpu_to_le64(fb->gsp_fw_offset);
	m->boot_bin_offset = cpu_to_le64(fb->boot_bin_offset);
	m->frts_offset = cpu_to_le64(fb->frts_offset);
	m->frts_size = cpu_to_le64(fb->frts_size);
	m->gsp_fw_wpr_end = cpu_to_le64(fb->wpr_end);
	m->fb_size = cpu_to_le64(fb->fb_size);
	m->vga_workspace_offset = cpu_to_le64(fb->vga_ws_offset);
	m->vga_workspace_size = cpu_to_le64(fb->vga_ws_size);

	/*
	 * GA104: no CrashCat sysmem queue (kgspGetCrashcatSysmemBufferSize_4a4dee
	 * returns 0), no VF partitions, pmuReservedSize is Hopper-only, and
	 * bBootGspRmWithBoostClocks is true for Turing..Ada.
	 */
	m->flags = GSP_FW_FLAGS_CLOCK_BOOST;
	m->boot_count = 0;
	m->verified = 0;
	m->revision = cpu_to_le64(GSP_FW_WPR_META_REVISION);
	m->magic = cpu_to_le64(GSP_FW_WPR_META_MAGIC);

	nv->wpr_meta = m;
	dev_info(dev, "wpr meta at %pad: radix3 %pad, bootloader %pad, signature %pad\n",
		 &nv->wpr_meta_dma, &g->radix3.root, &g->bl_dma, &g->sig_dma);
	return 0;
}

void nv_wpr_meta_fini(struct nvdrv_dev *nv)
{
	if (nv->wpr_meta)
		dma_free_coherent(&nv->pdev->dev, SZ_4K, nv->wpr_meta, nv->wpr_meta_dma);
	nv->wpr_meta = NULL;
}
