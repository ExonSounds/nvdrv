/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Register offsets used by nvdrv. Sources (open-gpu-kernel-modules 570.144,
 * src/common/inc/swref/published/...) are noted next to each group.
 */
#ifndef NV_REGS_H
#define NV_REGS_H

#include <linux/bits.h>

/* PMC: chip identification (ampere/ga102/dev_boot.h) */
#define NV_PMC_BOOT_0				0x00000000
#define   NV_PMC_BOOT_0_MINOR_REVISION		GENMASK(3, 0)
#define   NV_PMC_BOOT_0_MAJOR_REVISION		GENMASK(7, 4)
#define   NV_PMC_BOOT_0_CHIPSET			GENMASK(28, 20)	/* arch:impl */

#define NV_PMC_BOOT_42				0x00000a00
#define   NV_PMC_BOOT_42_MINOR_REVISION		GENMASK(15, 12)
#define   NV_PMC_BOOT_42_MAJOR_REVISION		GENMASK(19, 16)
#define   NV_PMC_BOOT_42_CHIPSET		GENMASK(29, 20)	/* arch:impl */

/* VBIOS scratch (turing/tu102/dev_bus.h) */
#define NV_PBUS_VBIOS_SCRATCH(i)		(0x00001400 + (i) * 4)
#define   NV_VBIOS_FWSECLIC_SCRATCH_INDEX_0E	0x0e
#define   NV_VBIOS_FWSECLIC_FRTS_ERR_CODE	GENMASK(31, 16)
#define   NV_VBIOS_FWSECLIC_SCRATCH_INDEX_15	0x15
#define   NV_VBIOS_FWSECLIC_SB_ERR_CODE		GENMASK(15, 0)

/* PFB (ampere/ga100/dev_fb.h, turing/tu102/dev_fb.h) */
#define NV_PFB_PRI_MMU_LOCAL_MEMORY_RANGE	0x00100ce0
#define   NV_PFB_LMR_LOWER_SCALE		GENMASK(3, 0)
#define   NV_PFB_LMR_LOWER_MAG			GENMASK(9, 4)
#define   NV_PFB_LMR_ECC_MODE			BIT(30)
#define NV_PFB_PRI_MMU_LOCK_CFG_PRIV_LEVEL_MASK	0x001fa7c8
#define   NV_PFB_PRI_MMU_LOCK_PLM_READ_L0	BIT(0)
#define NV_PFB_PRI_MMU_WPR2_ADDR_LO		0x001fa824
#define NV_PFB_PRI_MMU_WPR2_ADDR_HI		0x001fa828
#define NV_PFB_PRI_MMU_LOCK_ADDR_LO		0x001fa82c
#define NV_PFB_PRI_MMU_LOCK_ADDR_HI		0x001fa830
/* All four ADDR registers: VAL in 31:4, units of 4 KiB (ALIGNMENT 12). */
#define   NV_PFB_PRI_MMU_ADDR_VAL		GENMASK(31, 4)
#define   NV_PFB_PRI_MMU_ADDR_SHIFT		12

/* Sysmem flush buffer (ampere/ga100/dev_fb.h; kmemsysProgramSysmemFlushBuffer_GA100) */
#define NV_PFB_NISO_FLUSH_SYSMEM_ADDR		0x00100c10	/* addr >> 8, bits 39:8 */
#define NV_PFB_NISO_FLUSH_SYSMEM_ADDR_HI	0x00100c40	/* addr >> 8 >> 32, 7 bits */
#define NV_PFB_NISO_FLUSH_SYSMEM_ADDR_SHIFT	8

/* GSP reload handshake (ampere/ga102/dev_gc6_island*.h) */
#define NV_PGC6_BSI_SECURE_SCRATCH_14		0x001180f8
#define   NV_PGC6_BSI_SS14_BOOT_STAGE_3_HANDOFF	BIT(26)

/*
 * GSP doorbell (ampere/ga102/dev_gsp.h; kgspSetCmdQueueHead_TU102): any write to
 * QUEUE_HEAD(0) after a new command-queue element tells GSP-RM to look at the queue.
 */
#define NV_PGSP_QUEUE_HEAD(i)			(0x00110c00 + (i) * 8)
#define   NV_PGSP_RPC_QUEUE_IDX			0	/* RPC_TASK_RM_QUEUE_IDX */

/* PGC6 always-on scratch (ampere/ga102/dev_gc6_island*.h) */
#define NV_PGC6_AON_SECURE_SCRATCH_GROUP_05_PRIV_LEVEL_MASK	0x00118128
#define   NV_PGC6_AON_SSG05_PLM_READ_L0		BIT(0)
#define NV_PGC6_AON_SECURE_SCRATCH_GROUP_05_0_GFW_BOOT		0x00118234
#define   NV_PGC6_GFW_BOOT_PROGRESS		GENMASK(7, 0)
#define   NV_PGC6_GFW_BOOT_PROGRESS_COMPLETED	0xff
#define NV_USABLE_FB_SIZE_IN_MB			0x001183a4

/* Fuses (ampere/ga100/dev_fuse.h; used for all GA10x by RM's HAL) */
#define NV_FUSE_OPT_SECURE_GSP_DEBUG_DIS	0x0082074c
#define   NV_FUSE_OPT_SECURE_GSP_DEBUG_DIS_YES	BIT(0)
#define NV_FUSE_STATUS_OPT_DISPLAY		0x00820c04
#define   NV_FUSE_STATUS_OPT_DISPLAY_DISABLED	BIT(0)		/* 0 = display present */
#define NV_FUSE_OPT_FPF_SEC2_UCODE1_VERSION	0x00824140	/* +4*(ucode_id-1), 16 entries */
#define NV_FUSE_OPT_FPF_GSP_UCODE1_VERSION	0x008241c0	/* +4*(ucode_id-1), 16 entries */

/* Display (disp/v04_00/dev_disp.h) */
#define NV_PDISP_VGA_WORKSPACE_BASE		0x00625f04
#define   NV_PDISP_VGA_WORKSPACE_BASE_STATUS_VALID	BIT(3)
#define   NV_PDISP_VGA_WORKSPACE_BASE_ADDR	GENMASK(31, 8)	/* << 16 */

/* PROM: VBIOS window in BAR0 (turing/tu102/dev_ext_devices.h) */
#define NV_PROM_DATA				0x00300000
#define NV_PROM_SIZE				0x00100000

/* Size of the PRAMIN window (DRF_SIZE(NV_PRAMIN)). */
#define NV_PRAMIN_SIZE				0x00100000

/* Value returned by a device that has dropped off the bus. */
#define NV_DEAD					0xffffffff

#endif
