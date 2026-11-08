// SPDX-License-Identifier: GPL-2.0
/*
 * Boot status and framebuffer layout (Phase 4, read-only part).
 *
 * The layout follows kgspCalculateFbLayout_TU102() (570.144) from the top of
 * VRAM down to the FRTS region. The rest (boot binary, GSP ELF, heaps, WPR
 * start) depends on firmware sizes and is added once the firmware is loaded.
 */
#include <linux/bitfield.h>
#include <linux/minmax.h>

#include "nvdrv.h"
#include "nv_regs.h"

#define VBIOS_WORKSPACE_SIZE	0x20000
#define FRTS_SIZE		SZ_1M

bool nv_gfw_boot_completed(struct nvdrv_dev *nv, u32 *progress)
{
	/* FWSEC lowers the PLM once it is done; until then the status is unreadable. */
	if (!(nv_rd32(nv, NV_PGC6_AON_SECURE_SCRATCH_GROUP_05_PRIV_LEVEL_MASK) &
	      NV_PGC6_AON_SSG05_PLM_READ_L0)) {
		*progress = 0;
		return false;
	}
	*progress = FIELD_GET(NV_PGC6_GFW_BOOT_PROGRESS,
			      nv_rd32(nv, NV_PGC6_AON_SECURE_SCRATCH_GROUP_05_0_GFW_BOOT));
	return *progress == NV_PGC6_GFW_BOOT_PROGRESS_COMPLETED;
}

static u64 mmu_addr(struct nvdrv_dev *nv, u32 reg)
{
	return (u64)FIELD_GET(NV_PFB_PRI_MMU_ADDR_VAL, nv_rd32(nv, reg)) << NV_PFB_PRI_MMU_ADDR_SHIFT;
}

u64 nv_wpr2_lo(struct nvdrv_dev *nv) { return mmu_addr(nv, NV_PFB_PRI_MMU_WPR2_ADDR_LO); }
u64 nv_wpr2_hi(struct nvdrv_dev *nv) { return mmu_addr(nv, NV_PFB_PRI_MMU_WPR2_ADDR_HI); }

int nv_fb_layout_calc(struct nvdrv_dev *nv)
{
	struct nv_fb_layout *fb = &nv->fb;
	struct device *dev = &nv->pdev->dev;
	u64 vbios_reserved;
	u32 vga;

	fb->fb_size = (u64)nv_rd32(nv, NV_USABLE_FB_SIZE_IN_MB) << 20;
	if (!fb->fb_size || fb->fb_size > nv->vram_lmr) {
		dev_err(dev, "fb: implausible usable FB size %llu MiB\n", fb->fb_size >> 20);
		return -EINVAL;
	}

	/* VGA workspace (kdispGetVgaWorkspaceBase_v04_00 on TU10x/GA10x/AD10x). */
	vga = nv_rd32(nv, NV_PDISP_VGA_WORKSPACE_BASE);
	if (!(nv_rd32(nv, NV_FUSE_STATUS_OPT_DISPLAY) & NV_FUSE_STATUS_OPT_DISPLAY_DISABLED) &&
	    (vga & NV_PDISP_VGA_WORKSPACE_BASE_STATUS_VALID)) {
		fb->vga_ws_from_hw = true;
		fb->vga_ws_offset = (u64)FIELD_GET(NV_PDISP_VGA_WORKSPACE_BASE_ADDR, vga) << 16;
		if (fb->vga_ws_offset < fb->fb_size - NV_PRAMIN_SIZE)
			fb->vga_ws_offset = fb->fb_size - VBIOS_WORKSPACE_SIZE;
	} else {
		fb->vga_ws_offset = fb->fb_size - NV_PRAMIN_SIZE;
	}
	fb->vga_ws_size = fb->fb_size - fb->vga_ws_offset;

	/*
	 * MMU-locked region: only GA100 reads it (memmgrReadMmuLock_GA100); every
	 * other chip uses a stub that reports it invalid. Read it anyway for debugfs.
	 */
	if (nv_rd32(nv, NV_PFB_PRI_MMU_LOCK_CFG_PRIV_LEVEL_MASK) & NV_PFB_PRI_MMU_LOCK_PLM_READ_L0) {
		fb->mmu_lock_lo = mmu_addr(nv, NV_PFB_PRI_MMU_LOCK_ADDR_LO);
		fb->mmu_lock_hi = mmu_addr(nv, NV_PFB_PRI_MMU_LOCK_ADDR_HI);
	}
	fb->mmu_lock_valid = nv->chipset == 0x170 && fb->mmu_lock_hi >= fb->mmu_lock_lo;

	vbios_reserved = fb->mmu_lock_valid ? min(fb->mmu_lock_lo, fb->vga_ws_offset)
					    : fb->vga_ws_offset;

	/* RM would wrap here if MMU_LOCK read back as 0..0; refuse instead. */
	if (vbios_reserved < fb->fb_size / 2) {
		dev_err(dev, "fb: VBIOS reserved base 0x%llx is implausibly low\n", vbios_reserved);
		return -EINVAL;
	}

	/* Normal boot: no WPR end margin (kgspGetWprEndMargin() returns 0). */
	fb->wpr_end = ALIGN_DOWN(vbios_reserved, 0x20000);
	fb->frts_size = FRTS_SIZE;
	fb->frts_offset = fb->wpr_end - fb->frts_size;

	dev_info(dev, "fb: size 0x%llx, VGA ws 0x%llx%s, MMU lock %s 0x%llx-0x%llx, FRTS 0x%llx\n",
		 fb->fb_size, fb->vga_ws_offset, fb->vga_ws_from_hw ? "" : " (default)",
		 fb->mmu_lock_valid ? "used" : "ignored", fb->mmu_lock_lo, fb->mmu_lock_hi,
		 fb->frts_offset);
	return 0;
}

/* ---- GSP carve-outs below FRTS (rest of kgspCalculateFbLayout_TU102) ---- */

/* GA10x values, each checked against the chip's HAL in g_kernel_gsp_nvoc.c (570.144). */
#define HEAP_OS_CARVEOUT	(22ULL << 20)	/* kgspGetFwHeapParamOsCarveoutSize_4b5307 (LIBOS3 baremetal) */
#define HEAP_BASE_RM		(8ULL << 20)	/* fwHeapParamBaseSize, Turing..Ada */
#define HEAP_PER_GB_FB		(96ULL << 10)	/* GSP_FW_HEAP_PARAM_SIZE_PER_GB_FB */
#define HEAP_CLIENT_ALLOC	((48ULL << 10) * 2048)	/* GSP_FW_HEAP_PARAM_CLIENT_ALLOC_SIZE */
#define HEAP_MIN		(88ULL << 20)	/* kgspGetMinWprHeapSizeMB_647ce6 */
#define NON_WPR_HEAP_SIZE	SZ_1M		/* kgspGetNonWprHeapSize_ed6b8b */
#define PRESCRUBBED_TOP_FB	(256ULL << 20)	/* kgspGetPrescrubbedTopFbSize_e1e623 */
#define WPR_META_SIZE		256		/* sizeof(GspFwWprMeta) */

/*
 * _kgspCalculateFwHeapSize, limited to the pre-scrubbed top of FB because
 * GA10x has no scrubber ucode (bScrubberUcodeSupported is AD10x only).
 */
static int fw_heap_size(struct nv_fb_layout *fb, u64 pre, u64 post, u64 *size)
{
	u64 fb_gb = ALIGN(fb->fb_size, SZ_1G) >> 30;
	u64 heap = HEAP_OS_CARVEOUT + HEAP_BASE_RM +
		   ALIGN(HEAP_PER_GB_FB * fb_gb, SZ_1M) + ALIGN(HEAP_CLIENT_ALLOC, SZ_1M);
	u64 max_mb;

	if (pre + post >= PRESCRUBBED_TOP_FB)
		return -ENOSPC;
	max_mb = (PRESCRUBBED_TOP_FB - (pre + post)) >> 20;
	heap = max(heap, HEAP_MIN);
	heap = min(heap, max_mb << 20);
	*size = heap;
	return 0;
}

int nv_fb_layout_gsp(struct nvdrv_dev *nv, u64 bootloader_size, u64 elf_size)
{
	struct nv_fb_layout *fb = &nv->fb;
	struct device *dev = &nv->pdev->dev;
	u64 meta = ALIGN(WPR_META_SIZE, SZ_1M);
	u64 non_wpr = ALIGN(NON_WPR_HEAP_SIZE, SZ_1M);
	u64 pre, post, heap;
	int ret;

	if (!fb->frts_offset || !bootloader_size || !elf_size)
		return -EINVAL;

	fb->boot_bin_size = bootloader_size;
	fb->boot_bin_offset = ALIGN_DOWN(fb->frts_offset - bootloader_size, SZ_4K);
	fb->gsp_fw_size = elf_size;
	fb->gsp_fw_offset = ALIGN_DOWN(fb->boot_bin_offset - elf_size, SZ_64K);

	pre = meta + non_wpr;
	post = ALIGN(fb->fb_size - fb->gsp_fw_offset, SZ_1M);
	ret = fw_heap_size(fb, pre, post, &heap);
	if (ret) {
		dev_err(dev, "fb: GSP image does not fit in the pre-scrubbed region\n");
		return ret;
	}

	fb->heap_offset = ALIGN_DOWN(fb->gsp_fw_offset - heap, SZ_1M);
	fb->heap_size = ALIGN_DOWN(fb->gsp_fw_offset - fb->heap_offset, SZ_1M);
	fb->wpr_start = fb->heap_offset - meta;
	fb->non_wpr_heap_size = non_wpr;
	fb->non_wpr_heap_offset = fb->wpr_start - non_wpr;
	fb->rsvd_start = fb->non_wpr_heap_offset;

	/* Everything GSP uses must sit in the 256M the VBIOS already scrubbed. */
	if (fb->rsvd_start < fb->fb_size - PRESCRUBBED_TOP_FB) {
		dev_err(dev, "fb: GSP reservation 0x%llx starts below the pre-scrubbed region\n",
			fb->rsvd_start);
		return -ENOSPC;
	}

	dev_info(dev, "fb: GSP reserves 0x%llx-0x%llx (%llu MiB): non-WPR 0x%llx, WPR 0x%llx-0x%llx, heap %llu MiB, ELF 0x%llx, boot 0x%llx\n",
		 fb->rsvd_start, fb->fb_size, (fb->fb_size - fb->rsvd_start) >> 20,
		 fb->non_wpr_heap_offset, fb->wpr_start, fb->wpr_end, fb->heap_size >> 20,
		 fb->gsp_fw_offset, fb->boot_bin_offset);
	return 0;
}
