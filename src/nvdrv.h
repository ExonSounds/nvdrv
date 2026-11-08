/* SPDX-License-Identifier: GPL-2.0 */
#ifndef NVDRV_H
#define NVDRV_H

#include <drm/drm_mm.h>
#include <linux/debugfs.h>
#include <linux/io.h>
#include <linux/mutex.h>
#include <linux/pci.h>
#include <linux/types.h>

#define DRV_NAME "nvdrv"

/* The one GSP firmware release this driver speaks (RPC ABI is version-locked). */
#define NV_GSP_FW_VERSION "570.144"

#include <linux/scatterlist.h>

/*
 * Signed falcon "boot from HS" ucode ready to DMA into a falcon (FWSEC, Booter).
 * image holds IMEM contents at code_offset and DMEM contents at dmem_offset.
 */
struct nv_hs_ucode {
	u8 *image;
	u32 image_size;
	u32 code_offset;
	u32 imem_size, imem_pa, imem_va;
	u32 dmem_offset, dmem_size, dmem_pa;
	u32 pkc_data_offset;	/* DMEM offset of the signature (BROM_PARAADDR0) */
	u16 engine_id_mask;
	u8 ucode_id;
};

/* RM_RISCV_UCODE_DESC (arch/nvalloc/common/inc/rmRiscvUcode.h), 21 x u32. */
struct rm_riscv_ucode_desc {
	u32 version;
	u32 bootloaderOffset, bootloaderSize;
	u32 bootloaderParamOffset, bootloaderParamSize;
	u32 riscvElfOffset, riscvElfSize;
	u32 appVersion;
	u32 manifestOffset, manifestSize;
	u32 monitorDataOffset, monitorDataSize;
	u32 monitorCodeOffset, monitorCodeSize;
	u32 bIsMonitorEnabled;
	u32 swbromCodeOffset, swbromCodeSize;
	u32 swbromDataOffset, swbromDataSize;
	u32 fbReservedSize;
	u32 bSignedAsCode;
};
static_assert(sizeof(struct rm_riscv_ucode_desc) == 84);

/* GSP ELF image behind a 3-level page table in sysmem (kgspCreateRadix3). */
struct nv_radix3 {
	struct page **pages;	/* [L0][L1][L2][data], one 4K page each */
	dma_addr_t *dma;	/* bus address of each page */
	u32 npages;
	u32 lvl_pages[4], lvl_first[4];
	void *vaddr;		/* CPU mapping of all pages */
	struct sg_table sgt;
	bool mapped;
	u64 size;		/* image bytes */
	dma_addr_t root;	/* L0 page: sysmemAddrOfRadix3Elf */
};

struct nv_gsp_fw {
	struct nv_hs_ucode booter_load, booter_unload;	/* signature already patched */

	struct rm_riscv_ucode_desc bl_desc;
	void *bl;		/* RISC-V bootloader image, coherent sysmem */
	dma_addr_t bl_dma;
	u32 bl_size, bl_alloc;

	struct nv_radix3 radix3;	/* .fwimage */
	void *sig;		/* .fwsignature_ga10x, coherent sysmem */
	dma_addr_t sig_dma;
	u32 sig_size;
	u32 img_crc, sig_crc;
};

/* VBIOS image copied out of PROM (Phase 2). */
struct nv_vbios {
	u8 *data;
	u32 size;		/* through the end of the last ROM image */
	u32 exp_rom_offset;	/* added to falcon-table pointers (gap left by EFI) */
	u32 bit_offset;
};

/* FWSEC ucode extracted from the VBIOS, desc v3 "boot from HS" (Phase 2). */
struct nv_fwsec {
	bool debug;		/* FWSEC_DBG (debug-fused board) vs FWSEC_PROD */
	u32 desc_offset;	/* in VBIOS, for reference */

	u8 *image;		/* IMEM then DMEM, align256(stored_size) bytes */
	u32 image_size;
	u32 imem_size, imem_pa, imem_va;
	u32 dmem_offset;	/* DMEM's offset inside image (== imem_size) */
	u32 dmem_size, dmem_pa;
	u32 pkc_data_offset;	/* DMEM offset where the chosen signature is patched */
	u32 interface_offset;	/* DMEM offset of the application interface table */
	u32 dmem_mapper;	/* DMEM offset of the DMAP structure */
	u16 engine_id_mask;
	u8 ucode_id;
	u8 sig_count;
	u16 sig_versions;	/* bitmask; bit n set = signature for fuse version n */
	u8 *sigs;		/* sig_count * NV_FWSEC_SIG_SIZE */
	u32 sigs_size;
};

#define NV_FWSEC_SIG_SIZE	384	/* RSA-3K (BCRT30_RSA3K_SIG_SIZE) */

/* Framebuffer carve-outs at the top of VRAM (Phase 4, computed read-only). */
struct nv_fb_layout {
	u64 fb_size;
	u64 vga_ws_offset, vga_ws_size;
	bool vga_ws_from_hw;
	bool mmu_lock_valid;
	u64 mmu_lock_lo, mmu_lock_hi;
	u64 wpr_end;		/* gspFwWprEnd, 128K aligned */
	u64 frts_offset, frts_size;

	/* Below FRTS; needs firmware sizes (nv_fb_layout_gsp). 0 until computed. */
	u64 boot_bin_offset, boot_bin_size;	/* RISC-V bootloader, 4K aligned */
	u64 gsp_fw_offset, gsp_fw_size;		/* GSP ELF (.fwimage), 64K aligned */
	u64 heap_offset, heap_size;		/* GSP-RM heap inside WPR, 1M aligned */
	u64 wpr_start;				/* GspFwWprMeta lives here (1M) */
	u64 non_wpr_heap_offset, non_wpr_heap_size;
	u64 rsvd_start;				/* lowest byte GSP owns */
};

/*
 * GspFwWprMeta (arch/nvalloc/common/inc/gsp/gsp_fw_wpr_meta.h, 570.144):
 * handed to Booter in sysmem; Booter copies it into WPR at gspFwWprStart.
 */
struct gsp_fw_wpr_meta {
	__le64 magic;
	__le64 revision;
	__le64 sysmem_addr_radix3_elf;
	__le64 size_radix3_elf;
	__le64 sysmem_addr_bootloader;
	__le64 size_bootloader;
	__le64 bootloader_code_offset;
	__le64 bootloader_data_offset;
	__le64 bootloader_manifest_offset;
	__le64 sysmem_addr_signature;		/* union with heap free-list offset (resume) */
	__le64 size_signature;
	__le64 gsp_fw_rsvd_start;
	__le64 non_wpr_heap_offset;
	__le64 non_wpr_heap_size;
	__le64 gsp_fw_wpr_start;
	__le64 gsp_fw_heap_offset;
	__le64 gsp_fw_heap_size;
	__le64 gsp_fw_offset;
	__le64 boot_bin_offset;
	__le64 frts_offset;
	__le64 frts_size;
	__le64 gsp_fw_wpr_end;
	__le64 fb_size;
	__le64 vga_workspace_offset;
	__le64 vga_workspace_size;
	__le64 boot_count;
	__le32 partition_rpc_pad[4];		/* union: partitionRpc* fields */
	__le64 sysmem_addr_crash_report_queue;
	__le32 size_crash_report_queue;
	__le32 ls_ucode_version_pad;
	u8 gsp_fw_heap_vf_partition_count;
	u8 flags;
	u8 padding[2];
	__le32 pmu_reserved_size;
	__le64 verified;
} __packed;

#define GSP_FW_WPR_META_MAGIC		0xdc3aae21371a60b3ULL
#define GSP_FW_WPR_META_REVISION	1
#define GSP_FW_WPR_META_VERIFIED	0xa0a0a0a0a0a0a0a0ULL
#define GSP_FW_FLAGS_CLOCK_BOOST	BIT(0)

/* A coherent sysmem buffer shared with the GPU. */
struct nv_dma_buf {
	void *cpu;
	dma_addr_t dma;
	size_t size;
};

#define NV_LIBOS_NUM_LOGS	5	/* INIT, INTR, RM, MNOC, KRNL (GA10x) */

/* LibOS boot arguments and the RM RPC queues (src/libos.c). */
struct nv_libos {
	struct nv_dma_buf logs[NV_LIBOS_NUM_LOGS];
	struct nv_dma_buf args;		/* LibosMemoryRegionInitArgument[]; GSP gets this address */
	struct nv_dma_buf rmargs;	/* GSP_ARGUMENTS_CACHED */
	struct nv_dma_buf shm;		/* [PTE page][cmd queue][status queue] */
	struct nv_dma_buf flush;	/* sysmem flush page (NV_PFB_NISO_FLUSH_SYSMEM_ADDR) */
	struct nv_dma_buf notify_op;	/* NotifyOpSharedSurface */
	u32 pte_count, pt_size, cmdq_size, statq_size;
};

/* rpc_message_header_v03_00 */
struct nv_rpc_header {
	__le32 header_version;
	__le32 signature;
	__le32 length;		/* header + payload */
	__le32 function;
	__le32 rpc_result;
	__le32 rpc_result_private;
	__le32 sequence;
	__le32 spare;
};

/* inc/kernel/vgpu/rpc_global_enums.h */
#define NV_RPC_FREE				10
#define NV_RPC_GET_GSP_STATIC_INFO		65
#define NV_RPC_GSP_RM_CONTROL			76
#define NV_RPC_GSP_RM_ALLOC			103
#define NV_RPC_GSP_SET_SYSTEM_INFO		72
#define NV_RPC_SET_REGISTRY			73
#define NV_RPC_EVENT_GSP_INIT_DONE		0x1001
#define NV_RPC_EVENT_RUN_CPU_SEQUENCER		0x1002
#define NV_RPC_EVENT_OS_ERROR_LOG		0x1006

/* Host-side queue state (src/rpc.c). */
struct nv_rpc {
	void *stage;		/* 16 x 4K staging buffer for one message */
	u32 tx_count, tx_entry_off, tx_rxhdr_off, tx_write, tx_seq;
	bool rx_linked;
	u32 rx_count, rx_entry_off, rx_rxhdr_off, rx_read, rx_seq;
	u32 last_result, last_result_private;	/* rpc_result of the last nv_rpc_call */
};

/*
 * GspStaticConfigInfo (inc/kernel/gpu/gsp/gsp_static_config.h, 570.144): what GSP-RM
 * reports about the GPU in reply to GET_GSP_STATIC_INFO. Kept raw (debugfs) plus the
 * fields nvdrv uses, decoded at offsets from tools/gsp_static_info_probe.sh.
 */
#define GSP_STATIC_INFO_SIZE	1656
#define GSP_STATIC_FB_REGIONS	16

struct nv_gsp_fb_region {
	u64 base, limit, reserved;
	u32 performance;
	bool compressed, iso, is_protected;
};

struct nv_gsp_static_info {
	bool valid;
	u8 raw[GSP_STATIC_INFO_SIZE];

	char name[64], short_name[64];
	u8 gid[16];			/* SHA-1 based GPU UUID, binary */
	char chip_sku[9], project[5], project_sku[5];
	u64 fb_length, fbio_mask;
	u32 engine_caps[3];		/* bitmask over RM_ENGINE_TYPE */
	u32 num_fb_regions;
	struct nv_gsp_fb_region fb_regions[GSP_STATIC_FB_REGIONS];
	u64 bar1_pde_base, bar2_pde_base;
	bool vbios_valid;
	u32 vbios_sub_vendor, vbios_sub_device;
	u32 h_internal_client, h_internal_device, h_internal_subdevice;
	u64 fw_non_wpr_heap_offset, fw_frts_offset;	/* fwWprLayoutOffset */
};

/* Objects nvdrv owns inside GSP-RM, and what it learned through them (src/rm.c). */
struct nv_rm {
	u32 h_client, h_device, h_subdevice;	/* 0 = not allocated */

	/* NV2080_CTRL_CMD_GPU_GET_CHIP_DETAILS on our subdevice */
	bool chip_valid;
	u32 pci_dev_id, chip_major, chip_minor;
	char chip_sku[5];

	/* NV2080_CTRL_CMD_INTERNAL_MEMSYS_GET_STATIC_CONFIG on GSP's internal subdevice */
	bool memsys_valid;
	u64 l2_cache_size;
	u32 ram_type, ltc_count, lts_per_ltc, compr_page_size;
	bool ecc_fbpa, fbpa_present;
};

/* Display discovery through GSP-RM (src/disp.c). */
#define NV_DISP_MAX_DPY		8
#define NV_EDID_MAX		2048

struct nv_mode {
	bool valid;
	u32 clock_khz;
	u32 hdisplay, hsync_start, hsync_end, htotal;
	u32 vdisplay, vsync_start, vsync_end, vtotal;
	bool hsync_pos, vsync_pos;
};

struct nv_disp_dpy {
	u32 id;			/* NV0073 displayId (one bit) */
	bool connected, lit_by_vbios;
	u32 type, or_type, or_index, protocol, dcb_index;
	u32 edid_size;
	u8 edid[NV_EDID_MAX];
	struct nv_mode mode;	/* EDID preferred (first detailed timing) */
	/* DisplayPort sink and link */
	u8 dpcd_rev, max_link_bw, max_lanes;
	bool enhanced_framing;
	u32 sor;		/* assigned SOR, ~0 = none */
	u32 link_lanes, link_bw;
};

struct nv_disp {
	u32 h_common;		/* NV04_DISPLAY_COMMON, 0 = not allocated */
	u32 num_heads, supported, connected;
	u32 ndpy;
	struct nv_disp_dpy dpy[NV_DISP_MAX_DPY];
};

/* Display channels (src/evo.c). */
struct nv_evo_chan {
	u32 handle, chn;
	struct nv_dma_buf pb;	/* 4 KiB push buffer, coherent sysmem */
	void __iomem *user;	/* BAR0 user area: PUT @0, GET @4 */
	u32 put;
	bool overflow;
};

struct nv_evo {
	u64 vram_bottom, vram_top, vram_next;	/* nvdrv's VRAM carve-out (top-down) */
	u64 pramin_base;			/* current BAR0 window (VRAM, 64K units) */
	u64 instmem;				/* display instance memory (VRAM) */
	u32 inst_next;				/* next free instance, 16-byte units */
	u32 h_disp;				/* NVC670_DISPLAY */
	struct nv_evo_chan core;
	struct nv_evo_chan win[2];
	u64 fb, fb_size;			/* scanout surface (VRAM), NV_FB_MAX */
	void __iomem *fb_cpu;			/* its BAR1 mapping (write-combined) */
	u32 fb_w, fb_h;				/* mode the test pattern was drawn for */
	bool pattern_dirty;
	struct nv_dma_buf sysfb;		/* scanout test surface in sysmem */
	bool fb_bound, win_mapped, active;
	bool pool_ctxdma;			/* window ctxdma over the VRAM pool exists */
	u64 scanout_cur;			/* surface window 0 shows (0 = own framebuffer) */
	u32 fb_pitch;				/* own framebuffer pitch for the current mode */
	u32 sor, head;				/* of the active modeset */
};

/* BAR1 page tables under GSP-RM's root (src/bar1.c). */
struct nv_bar1 {
	u64 tables, pd0, pts;	/* VRAM */
	u64 old_root0;		/* GSP's original root entry 0 */
	u64 size;		/* usable BAR1 VA */
};

struct nv_drm;

/* GPU address space and channel (src/fifo.c). */
struct nv_vmm {
	u32 handle;		/* FERMI_VASPACE_A */
	u32 h_client, h_device, h_subdevice;	/* RM objects it lives under */
	u64 root;		/* root page directory (VRAM) */
	u64 next;		/* next free VA for kernel objects (bump) */
	bool external;		/* client VM: externally owned, page tables from the pool */
	struct list_head pts;	/* struct nv_pt, external only */
};

/* A freeable VRAM allocation from the pool (src/mem.c). */
struct nv_pt {
	struct list_head head;
	struct drm_mm_node node;
};

struct nv_mem {
	struct mutex lock;
	struct drm_mm mm;
	u64 base, size, used;
	bool ready;
};

struct nv_chan {
	struct nv_vmm *vm;
	u32 handle, chid, runlist, token, ce, rl_base, engine;
	struct drm_mm_node ctx[2];		/* client GR channels: MAIN, PATCH from the pool */
	u64 inst;				/* VRAM */
	void __iomem *userd_cpu;		/* this channel's USERD through BAR1 */
	struct nv_dma_buf gpfifo, pb, mthdbuf, sem, errnotif;
	u64 gpfifo_va, pb_va, sem_va;
	u32 gp_put, pb_put, pb_start, seq;
};

#define NV_GR_MAX_CTXBUFS	16
struct nv_fifo {
	u64 userd;				/* shared USERD page (VRAM, 0x200 per chid) */
	void __iomem *userd_cpu;		/* ... through BAR1 */
	u32 gr_runlist, gr_rl_base;
	/* GR context buffers (promoted to RM once, for the GR channel) */
	u32 ctxbuf_nr;
	struct { u64 vram, va, size; u16 id; bool nonmapped; } ctxbuf[NV_GR_MAX_CTXBUFS];
	bool gr_ready;
	unsigned long chids[2];			/* chids in use (0 = reserved by RM) */
	u32 gr_ce;				/* a copy engine on the GR runlist (GRCE) */
};

/* nouveau-compatible uAPI state (src/uapi.c) */
struct nv_uapi {
	spinlock_t lock;			/* fence lock: pending lists, active channels */
	struct list_head active;		/* channels with unsignalled fences */
	struct hrtimer timer;			/* polls their semaphores */
	bool polling;				/* timer armed or running (under lock) */
	struct mutex bar1_lock;
	struct drm_mm bar1_mm;			/* BAR1 VA for CPU-mapped VRAM objects */
	bool bar1_ready;
	u32 gpcs, tpcs;
};

struct nvdrv_dev {
	struct pci_dev *pdev;
	void __iomem *bar0;
	resource_size_t bar0_len;

	u32 boot0, boot42;
	u16 chipset;		/* e.g. 0x174 = GA104 */
	u8 rev;
	const char *name;
	u64 vram_lmr;		/* from LOCAL_MEMORY_RANGE */

	struct nv_vbios vbios;
	struct nv_fwsec fwsec;
	struct nv_fb_layout fb;
	struct nv_gsp_fw gsp_fw;

	struct gsp_fw_wpr_meta *wpr_meta;	/* 4K coherent sysmem page */
	dma_addr_t wpr_meta_dma;		/* what Booter gets in MAILBOX0/1 */
	struct nv_libos libos;
	struct nv_rpc rpc;
	struct mutex gsp_lock;	/* GSP boot and every RPC exchange after it */
	struct nv_gsp_static_info gsp_info;
	struct nv_rm rm;
	struct nv_disp disp;
	struct nv_evo evo;
	struct nv_bar1 bar1;
	struct nv_drm *drm;	/* DRM device once registered (src/drm.c) */
	struct nv_vmm vmm;
	struct nv_chan chan;		/* copy engine */
	struct nv_chan grch;		/* graphics/compute */
	struct nv_fifo fifo;
	struct nv_mem mem;
	struct nv_uapi uapi;
	bool gsp_touched;	/* Booter/FWSEC ran: GSP may DMA into our memory */
	bool gsp_running;	/* GSP_INIT_DONE received */

	/* FWSEC image patched for FRTS, kept for inspection (fwsec_frts.bin). */
	u8 *fwsec_frts;
	struct debugfs_blob_wrapper vbios_blob, gsp_info_blob, fwsec_frts_blob, booter_load_blob, wpr_meta_blob, cmdq_blob;
	struct debugfs_blob_wrapper libos_blobs[4 + NV_LIBOS_NUM_LOGS];	/* args, rmargs, PTE page, cmdq hdr, log page 0s */

	struct dentry *dbg;
};

static inline u32 nv_rd32(struct nvdrv_dev *nv, u32 reg)
{
	return ioread32(nv->bar0 + reg);
}

static inline void nv_wr32(struct nvdrv_dev *nv, u32 reg, u32 val)
{
	iowrite32(val, nv->bar0 + reg);
}

struct seq_file;

/* falcon.c (register writes: only via debugfs triggers, only with allow_hw_writes=1) */
enum nv_falcon_id { NV_FALCON_GSP, NV_FALCON_SEC2 };
bool nv_hw_writes_allowed(void);
int nv_falcon_hs_exec(struct nvdrv_dev *nv, enum nv_falcon_id id,
		      const struct nv_hs_ucode *u, u32 *mbox0, u32 *mbox1);
int nv_gsp_reset_into_riscv(struct nvdrv_dev *nv);
void nv_falcon_write_mailboxes(struct nvdrv_dev *nv, enum nv_falcon_id id, u32 m0, u32 m1);
void nv_falcon_read_mailboxes(struct nvdrv_dev *nv, enum nv_falcon_id id, u32 *m0, u32 *m1);
void nv_gsp_write_os(struct nvdrv_dev *nv, u32 val);
bool nv_gsp_riscv_active(struct nvdrv_dev *nv);
int nv_gsp_core_reset(struct nvdrv_dev *nv);
void nv_gsp_core_start(struct nvdrv_dev *nv);
void nv_falcon_start(struct nvdrv_dev *nv, enum nv_falcon_id id);
int nv_gsp_core_wait_halt(struct nvdrv_dev *nv);
int nv_fwsec_run_frts(struct nvdrv_dev *nv);
void nv_falcon_dump(struct nvdrv_dev *nv, struct seq_file *m);

/* vbios.c */
int nv_vbios_load(struct nvdrv_dev *nv);
int nv_fwsec_extract(struct nvdrv_dev *nv);
int nv_fwsec_prepare_frts(struct nvdrv_dev *nv, u64 frts_offset, u8 **out);
void nv_vbios_fini(struct nvdrv_dev *nv);
u32 nv_fuse_ucode_version(struct nvdrv_dev *nv, u8 ucode_id);

/* gsp.c */
int nv_gsp_boot(struct nvdrv_dev *nv);
void nv_gsp_fini(struct nvdrv_dev *nv);
int nv_gsp_handle_event(struct nvdrv_dev *nv, u32 fn, struct nv_rpc_header *h, void *payload, u32 len);
void nv_gsp_dump_state(struct nvdrv_dev *nv, const char *why);

/* rm.c */
int nv_rm_get_static_info(struct nvdrv_dev *nv);
void nv_rm_static_info_show(struct nvdrv_dev *nv, struct seq_file *m);
int nv_rm_alloc(struct nvdrv_dev *nv, u32 h_client, u32 h_parent, u32 h_object, u32 cls,
		void *params, u32 params_size);
int nv_rm_control(struct nvdrv_dev *nv, u32 h_client, u32 h_object, u32 cmd,
		  void *params, u32 params_size);
#define NVDRV_H_DEVICE		0xd0000080	/* in every nvdrv RM client */
#define NVDRV_H_SUBDEVICE	0xd0002080
int nv_rm_client_alloc(struct nvdrv_dev *nv, u32 h_client);
int nv_rm_free(struct nvdrv_dev *nv, u32 h_client, u32 h_object);
int nv_rm_init_objects(struct nvdrv_dev *nv);
void nv_rm_fini_objects(struct nvdrv_dev *nv);
void nv_rm_objects_show(struct nvdrv_dev *nv, struct seq_file *m);
int nv_rm_selftest(struct nvdrv_dev *nv);
int nv_rm_perf_boost(struct nvdrv_dev *nv, u32 seconds);
#define NV_PERF_BOOST_FOREVER	U32_MAX
void nv_rm_perf_show(struct nvdrv_dev *nv, struct seq_file *m);

/* disp.c */
int nv_disp_probe(struct nvdrv_dev *nv);
void nv_disp_forget(struct nvdrv_dev *nv);
int nv_disp_link(struct nvdrv_dev *nv);
int nv_disp_retrain(struct nvdrv_dev *nv);
int nv_disp_bringup(struct nvdrv_dev *nv);
void nv_disp_dp_status_show(struct nvdrv_dev *nv, struct seq_file *m);
void nv_disp_show(struct nvdrv_dev *nv, struct seq_file *m);

/* evo.c */
int nv_vram_alloc(struct nvdrv_dev *nv, u64 size, u64 align, u64 *out);
void nv_vram_wr32(struct nvdrv_dev *nv, u64 off, u32 val);
u32 nv_vram_rd32(struct nvdrv_dev *nv, u64 off);
void nv_vram_fill(struct nvdrv_dev *nv, u64 off, u64 len, u32 val,
		  u32 (*fn)(void *ctx, u64 i), void *ctx);
int nv_evo_ctxdma(struct nvdrv_dev *nv, u32 handle, u64 base, u64 size, u32 chn, bool sysmem);
void nv_evo_mthd(struct nv_evo_chan *c, u32 mthd, const u32 *data, u32 n);
int nv_evo_kick(struct nvdrv_dev *nv, struct nv_evo_chan *c);
int nv_evo_kick_nowait(struct nvdrv_dev *nv, struct nv_evo_chan *c);
int nv_evo_begin(struct nvdrv_dev *nv, struct nv_evo_chan *c, u32 need);
int nv_evo_wait(struct nvdrv_dev *nv, struct nv_evo_chan *c);

/* bar1.c */
int nv_bar1_init(struct nvdrv_dev *nv);
#define NV_BAR1_UNMAP	(~0ULL)
int nv_bar1_set(struct nvdrv_dev *nv, u64 va, u64 vram, u64 size);
void __iomem *nv_bar1_map(struct nvdrv_dev *nv, u64 va, u64 vram, u64 size);

/* fifo.c */
/* src/mem.c: VRAM pool for GEM objects, client page tables, channel contexts */
int nv_mem_get(struct nvdrv_dev *nv, struct drm_mm_node *n, u64 size, u64 align);
void nv_mem_put(struct nvdrv_dev *nv, struct drm_mm_node *n);
void nv_mem_fini(struct nvdrv_dev *nv);

/* src/fifo.c helpers shared with src/uapi.c (caller holds gsp_lock) */
#define NV_PTE_APER_SYS_COH	(2ULL << 1)
#define NV_PTE_VOL		BIT_ULL(3)
#define NV_PTE_PRIV		BIT_ULL(5)
#define NV_PTE_RO		BIT_ULL(6)
#define NV_PTE_KIND_SHIFT	56
int nv_vmm_map_4k(struct nvdrv_dev *nv, struct nv_vmm *vm, u64 va, u64 pa, u64 pte_flags);
int nv_vmm_set_pte(struct nvdrv_dev *nv, struct nv_vmm *vm, u64 va, u64 pte);
int nv_vmm_flush(struct nvdrv_dev *nv, struct nv_vmm *vm);
void nv_vmm_join(struct nvdrv_dev *nv, struct nv_vmm *vm, u64 inst);
int nv_vmm_map_vram_flags(struct nvdrv_dev *nv, struct nv_vmm *vm, u64 vram, u64 size, u64 align,
			  u64 pte_flags, u64 *va);
int nv_vmm_map_sysmem(struct nvdrv_dev *nv, struct nv_vmm *vm, const struct nv_dma_buf *b, u64 *va);
int nv_vmm_client_init(struct nvdrv_dev *nv, struct nv_vmm *vm, u32 h_client, u64 kernel_va);
void nv_vmm_client_fini(struct nvdrv_dev *nv, struct nv_vmm *vm);
int nv_dma_buf_alloc(struct nvdrv_dev *nv, struct nv_dma_buf *b, size_t size);
void nv_dma_buf_free(struct nvdrv_dev *nv, struct nv_dma_buf *b);
int nv_chan_create(struct nvdrv_dev *nv, struct nv_chan *ch, struct nv_vmm *vm, u32 handle, u32 chid,
		   u32 engine, u32 runlist, u32 rl_base);
void nv_chan_destroy(struct nvdrv_dev *nv, struct nv_chan *ch);
int nv_gr_chan_ctx(struct nvdrv_dev *nv, struct nv_chan *ch);
int nv_ce_clear(struct nvdrv_dev *nv, u64 vram, u64 size);
int nv_chid_get(struct nvdrv_dev *nv);
void nv_chid_put(struct nvdrv_dev *nv, u32 chid);
void nv_push(struct nv_chan *ch, u32 subc, u32 mthd, const u32 *data, u32 n);
void nv_push1(struct nv_chan *ch, u32 subc, u32 mthd, u32 v);
void nv_kick(struct nvdrv_dev *nv, struct nv_chan *ch);
/* src/uapi.c */
struct drm_file;
struct drm_device;
struct file;
extern bool nv_nvk;
struct nvdrv_dev *nv_drm_nv(struct drm_device *dev);
void nv_uapi_init(struct nvdrv_dev *nv);
void nv_uapi_fini(struct nvdrv_dev *nv);
int nv_uapi_open(struct drm_device *dev, struct drm_file *file);
void nv_uapi_postclose(struct drm_device *dev, struct drm_file *file);
long nv_uapi_ioctl(struct file *filp, unsigned int cmd, unsigned long arg);
struct drm_ioctl_desc;
extern const struct drm_ioctl_desc *const nv_uapi_ioctls_ptr;
extern const int nv_uapi_num_ioctls;
struct drm_gem_object;
bool nv_uapi_bo_vram(struct drm_gem_object *obj, u64 *vram);

int nv_fifo_init(struct nvdrv_dev *nv);
int nv_ce_test(struct nvdrv_dev *nv);
int nv_gr_test(struct nvdrv_dev *nv);
void nv_fifo_show(struct nvdrv_dev *nv, struct seq_file *m);
int nv_chan_kick(struct nvdrv_dev *nv, u32 mode);
void nv_fifo_fini(struct nvdrv_dev *nv);

/* drm.c */
int nv_drm_register(struct nvdrv_dev *nv);
void nv_drm_unregister(struct nvdrv_dev *nv);
void nv_drm_stats_show(struct nvdrv_dev *nv, struct seq_file *m);

/* modeset.c */
int nv_disp_modeset(struct nvdrv_dev *nv, u32 flags);
int nv_disp_modeset_locked(struct nvdrv_dev *nv, u32 flags);
int nv_disp_detach_locked(struct nvdrv_dev *nv);
int dp_watermark(const struct nv_mode *m, u32 lanes, u32 link_bw_270m, bool ef, bool increased,
		 u32 *tu, u32 *wm, u32 *hblank_sym, u32 *vblank_sym);
bool nv_disp_mode_fits(struct nvdrv_dev *nv, const struct nv_mode *m);
int nv_disp_set_mode(struct nvdrv_dev *nv, const struct nv_mode *m);
#define NV_FB_MAX		(64ULL << 20)	/* fits 4096x4096 XRGB8888 */
/*
 * Experiment flags for disp_modeset / disp_flip. 0 = the configuration known to
 * work on GA104: composition bypass, identity FMT, CSC/LUT methods not sent.
 */
#define NV_FLIP_COMPOSE			BIT(0)	/* normal blending: needs an input LUT (err 0x2d without) */
#define NV_MODESET_NO_STREAM_CONFIG	BIT(1)
#define NV_FLIP_MCLK_WAR		BIT(2)	/* bookend UPDATE with SW_SET_MCLK_SWITCH */
#define NV_FLIP_NO_FMT			BIT(3)
#define NV_FLIP_CSC			BIT(4)	/* send CSC/ILUT/TMO disables (err 0x2 with bypass) */
#define NV_FLIP_SYSMEM			BIT(5)	/* scan out a test surface from coherent sysmem */
int nv_disp_scanout(struct nvdrv_dev *nv, u64 vram, u32 pitch);
int nv_disp_flip(struct nvdrv_dev *nv, u32 flags);
int nv_bar1_test(struct nvdrv_dev *nv);
int nv_bar1_bench(struct nvdrv_dev *nv);
int nv_evo_init(struct nvdrv_dev *nv);
int nv_evo_window_init(struct nvdrv_dev *nv, u32 win);
void nv_evo_fini(struct nvdrv_dev *nv);
void nv_evo_show(struct nvdrv_dev *nv, struct seq_file *m);

/* rpc.c */
int nv_rpc_init(struct nvdrv_dev *nv);
void nv_rpc_fini(struct nvdrv_dev *nv);
int nv_rpc_send(struct nvdrv_dev *nv, u32 function, const void *payload, u32 len);
int nv_rpc_link_statq(struct nvdrv_dev *nv);
int nv_rpc_recv(struct nvdrv_dev *nv, struct nv_rpc_header **hdr, void **payload, u32 *len);
int nv_rpc_queue_preboot(struct nvdrv_dev *nv);
int nv_rpc_wait(struct nvdrv_dev *nv, u32 fn, unsigned int timeout_ms,
		struct nv_rpc_header **hdr, void **payload, u32 *len);
int nv_rpc_call(struct nvdrv_dev *nv, u32 fn, const void *req, u32 req_len,
		void *reply, u32 reply_len);

/* libos.c */
int nv_libos_init(struct nvdrv_dev *nv);
void nv_libos_reset(struct nvdrv_dev *nv);
void nv_libos_fini(struct nvdrv_dev *nv);

/* wpr.c */
int nv_wpr_meta_build(struct nvdrv_dev *nv);
void nv_wpr_meta_fini(struct nvdrv_dev *nv);

/* fw.c */
int nv_fw_load(struct nvdrv_dev *nv);
void nv_fw_fini(struct nvdrv_dev *nv);

/* fb.c */
bool nv_gfw_boot_completed(struct nvdrv_dev *nv, u32 *progress);
int nv_fb_layout_calc(struct nvdrv_dev *nv);
int nv_fb_layout_gsp(struct nvdrv_dev *nv, u64 bootloader_size, u64 elf_size);
u64 nv_wpr2_lo(struct nvdrv_dev *nv);
u64 nv_wpr2_hi(struct nvdrv_dev *nv);

#endif
