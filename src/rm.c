// SPDX-License-Identifier: GPL-2.0
/*
 * RM-level calls into GSP-RM (Phase 6/7), on top of the RPC transport in rpc.c.
 *
 *   GET_GSP_STATIC_INFO  rpcGetGspStaticInfo_v14_00 (vgpu/rpc.c), consumed by
 *                        _kgspInitGpuProperties / gpu_gspclient.c in RM
 *   GSP_RM_ALLOC         rpcRmApiAlloc_GSP
 *   GSP_RM_CONTROL       rpcRmApiControl_GSP
 *   FREE                 rpcRmApiFree_GSP
 *   root/device/subdevice  NV_RM_RPC_ALLOC_SHARE_DEVICE_FWCLIENT (vgpu/rpc.h), subdevice construct
 */
#include <linux/lockdep.h>
#include <linux/seq_file.h>
#include <linux/slab.h>
#include <linux/string.h>
#include <linux/unaligned.h>

#include "nvdrv.h"

/*
 * GspStaticConfigInfo field offsets, 570.144 (tools/gsp_static_info_probe.sh).
 * A hand-written struct would have to replicate seven nested control-param structs
 * just to reach the fields below, so the reply is decoded by offset instead.
 */
#define SCI_GID_INFO			24	/* NV2080_CTRL_GPU_GET_GID_INFO_PARAMS */
#define   GID_LENGTH			8
#define   GID_DATA			12
#define SCI_SKU_INFO			292	/* NV2080_CTRL_BIOS_GET_SKU_INFO_PARAMS */
#define   SKU_CHIP_SKU			4	/* char[9] */
#define   SKU_PROJECT			24	/* char[5] */
#define   SKU_PROJECT_SKU		29	/* char[5] */
#define SCI_FB_REGION_INFO		344	/* NV2080_CTRL_CMD_FB_GET_FB_REGION_INFO_PARAMS */
#define   FBR_ENTRIES			8
#define   FBR_ENTRY_SIZE		48
#define   FBR_BASE			0
#define   FBR_LIMIT			8
#define   FBR_RESERVED			16
#define   FBR_PERFORMANCE		24
#define   FBR_COMPRESSED		28
#define   FBR_ISO			29
#define   FBR_PROTECTED			30
#define SCI_ENGINE_CAPS			1204	/* NvU32[3], NV2080_ENGINE_TYPE bits (kern_bus.c) */
#define SCI_FB_LENGTH			1224
#define SCI_FBIO_MASK			1232
/*
 * 1240..1259 (fb_bus_width, fb_ram_type, fbp_mask, l2_cache_size in the header) are not
 * decoded: on GA104 they hold 0, 0x100, 0x11, 0xf, which does not match the header
 * (MEMSYS_GET_STATIC_CONFIG says GDDR6 = 0x11, L2 3 MiB). RM never reads them either.
 */
#define SCI_GPU_NAME			1260	/* NvU8[64] */
#define SCI_GPU_SHORT_NAME		1324	/* NvU8[64] */
#define SCI_BAR1_PDE_BASE		1536
#define SCI_BAR2_PDE_BASE		1544
#define SCI_VBIOS_VALID			1552
#define SCI_VBIOS_SUB_VENDOR		1556
#define SCI_VBIOS_SUB_DEVICE		1560
#define SCI_H_INTERNAL_CLIENT		1600
#define SCI_H_INTERNAL_DEVICE		1604
#define SCI_H_INTERNAL_SUBDEVICE	1608
#define SCI_FW_WPR_LAYOUT		1640	/* {nonWprHeapOffset, frtsOffset} */

static_assert(SCI_FB_REGION_INFO + FBR_ENTRIES + GSP_STATIC_FB_REGIONS * FBR_ENTRY_SIZE == 1120);
static_assert(SCI_FW_WPR_LAYOUT + 16 == GSP_STATIC_INFO_SIZE);

static void copy_str(char *dst, size_t dst_size, const u8 *src, size_t src_size)
{
	size_t n = strnlen(src, min(src_size, dst_size - 1));

	memcpy(dst, src, n);
	dst[n] = 0;
}

static void decode_static_info(struct nv_gsp_static_info *si)
{
	const u8 *b = si->raw;
	u32 i;

	copy_str(si->name, sizeof(si->name), b + SCI_GPU_NAME, 64);
	copy_str(si->short_name, sizeof(si->short_name), b + SCI_GPU_SHORT_NAME, 64);
	if (get_unaligned_le32(b + SCI_GID_INFO + GID_LENGTH) >= sizeof(si->gid))
		memcpy(si->gid, b + SCI_GID_INFO + GID_DATA, sizeof(si->gid));
	copy_str(si->chip_sku, sizeof(si->chip_sku), b + SCI_SKU_INFO + SKU_CHIP_SKU, 9);
	copy_str(si->project, sizeof(si->project), b + SCI_SKU_INFO + SKU_PROJECT, 5);
	copy_str(si->project_sku, sizeof(si->project_sku), b + SCI_SKU_INFO + SKU_PROJECT_SKU, 5);

	si->fb_length = get_unaligned_le64(b + SCI_FB_LENGTH);
	si->fbio_mask = get_unaligned_le64(b + SCI_FBIO_MASK);
	for (i = 0; i < ARRAY_SIZE(si->engine_caps); i++)
		si->engine_caps[i] = get_unaligned_le32(b + SCI_ENGINE_CAPS + 4 * i);

	si->num_fb_regions = min_t(u32, get_unaligned_le32(b + SCI_FB_REGION_INFO),
				   GSP_STATIC_FB_REGIONS);
	for (i = 0; i < si->num_fb_regions; i++) {
		const u8 *e = b + SCI_FB_REGION_INFO + FBR_ENTRIES + i * FBR_ENTRY_SIZE;
		struct nv_gsp_fb_region *r = &si->fb_regions[i];

		r->base = get_unaligned_le64(e + FBR_BASE);
		r->limit = get_unaligned_le64(e + FBR_LIMIT);
		r->reserved = get_unaligned_le64(e + FBR_RESERVED);
		r->performance = get_unaligned_le32(e + FBR_PERFORMANCE);
		r->compressed = e[FBR_COMPRESSED];
		r->iso = e[FBR_ISO];
		r->is_protected = e[FBR_PROTECTED];
	}

	si->bar1_pde_base = get_unaligned_le64(b + SCI_BAR1_PDE_BASE);
	si->bar2_pde_base = get_unaligned_le64(b + SCI_BAR2_PDE_BASE);
	si->vbios_valid = b[SCI_VBIOS_VALID];
	si->vbios_sub_vendor = get_unaligned_le32(b + SCI_VBIOS_SUB_VENDOR);
	si->vbios_sub_device = get_unaligned_le32(b + SCI_VBIOS_SUB_DEVICE);
	si->h_internal_client = get_unaligned_le32(b + SCI_H_INTERNAL_CLIENT);
	si->h_internal_device = get_unaligned_le32(b + SCI_H_INTERNAL_DEVICE);
	si->h_internal_subdevice = get_unaligned_le32(b + SCI_H_INTERNAL_SUBDEVICE);
	si->fw_non_wpr_heap_offset = get_unaligned_le64(b + SCI_FW_WPR_LAYOUT);
	si->fw_frts_offset = get_unaligned_le64(b + SCI_FW_WPR_LAYOUT + 8);
}

/*
 * GET_GSP_STATIC_INFO: the request carries a zeroed GspStaticConfigInfo, the reply
 * the filled-in one. Cross-checks it against what nvdrv computed on its own.
 */
int nv_rm_get_static_info(struct nvdrv_dev *nv)
{
	struct device *dev = &nv->pdev->dev;
	struct nv_gsp_static_info *si = &nv->gsp_info;
	u8 *req;
	int ret;

	req = kzalloc(GSP_STATIC_INFO_SIZE, GFP_KERNEL);
	if (!req)
		return -ENOMEM;
	si->valid = false;
	ret = nv_rpc_call(nv, NV_RPC_GET_GSP_STATIC_INFO, req, GSP_STATIC_INFO_SIZE,
			  si->raw, GSP_STATIC_INFO_SIZE);
	kfree(req);
	if (ret) {
		dev_err(dev, "rm: GET_GSP_STATIC_INFO failed: %d (rpc_result 0x%x)\n", ret,
			nv->rpc.last_result);
		return ret;
	}
	decode_static_info(si);
	si->valid = true;

	dev_info(dev, "rm: \"%s\" (%s), %llu MiB, engine caps %08x %08x, internal client 0x%08x\n",
		 si->name, si->short_name, si->fb_length >> 20, si->engine_caps[0],
		 si->engine_caps[1], si->h_internal_client);
	return 0;
}

static const char *ram_type_name(u32 t)
{
	switch (t) {
	case 0x08: return "GDDR5";
	case 0x10: return "GDDR5X";
	case 0x11: return "GDDR6";
	case 0x12: return "GDDR6X";
	case 0x0f: return "HBM2";
	case 0x14: return "HBM3";
	case 0x15: return "GDDR7";
	default:   return "?";
	}
}

/* NV2080_ENGINE_TYPE_* (class/cl2080_notification.h): the numbering engineCaps uses. */
static const char *const nv2080_engine_names[] = {
	[0x01] = "GR", [0x09] = "CE0", [0x0a] = "CE1", [0x0b] = "CE2", [0x0c] = "CE3",
	[0x0d] = "CE4", [0x0e] = "CE5", [0x0f] = "CE6", [0x10] = "CE7", [0x11] = "CE8",
	[0x12] = "CE9", [0x13] = "NVDEC0", [0x1b] = "NVENC0", [0x22] = "SW",
	[0x26] = "SEC2", [0x33] = "OFA",
};

/* Caller holds nv->gsp_lock. */
void nv_rm_static_info_show(struct nvdrv_dev *nv, struct seq_file *m)
{
	const struct nv_gsp_static_info *si = &nv->gsp_info;
	const u8 *g = si->gid;
	u32 i;

	seq_puts(m, "\n[gsp static info]\n");
	if (!si->valid) {
		seq_puts(m, "not received\n");
		return;
	}
	seq_printf(m, "name:            %s\n", si->name);
	seq_printf(m, "short name:      %s\n", si->short_name);
	seq_printf(m, "uuid:            GPU-%*phN-%*phN-%*phN-%*phN-%*phN\n",
		   4, g, 2, g + 4, 2, g + 6, 2, g + 8, 6, g + 10);
	seq_printf(m, "sku:             chip %s, project %s-%s\n", si->chip_sku, si->project,
		   si->project_sku);
	seq_printf(m, "vbios:           %svalid, sub ids %04x:%04x\n", si->vbios_valid ? "" : "not ",
		   si->vbios_sub_vendor, si->vbios_sub_device);
	seq_printf(m, "fb:              %llu MiB, fbio mask 0x%llx\n", si->fb_length >> 20, si->fbio_mask);
	seq_printf(m, "engine caps:     %08x %08x %08x:", si->engine_caps[0], si->engine_caps[1],
		   si->engine_caps[2]);
	for (i = 0; i < 32 * ARRAY_SIZE(si->engine_caps); i++) {
		if (!(si->engine_caps[i / 32] & BIT(i % 32)))
			continue;
		if (i < ARRAY_SIZE(nv2080_engine_names) && nv2080_engine_names[i])
			seq_printf(m, " %s", nv2080_engine_names[i]);
		else
			seq_printf(m, " #%u", i);
	}
	seq_putc(m, '\n');
	for (i = 0; i < si->num_fb_regions; i++) {
		const struct nv_gsp_fb_region *r = &si->fb_regions[i];

		seq_printf(m, "fb region %u:     0x%llx - 0x%llx, reserved 0x%llx, perf %u%s%s%s\n", i,
			   r->base, r->limit, r->reserved, r->performance,
			   r->compressed ? " compressible" : "", r->iso ? " iso" : "",
			   r->is_protected ? " protected" : "");
	}
	seq_printf(m, "bar1/bar2 pde:   0x%llx / 0x%llx\n", si->bar1_pde_base, si->bar2_pde_base);
	seq_printf(m, "internal hdls:   client 0x%08x device 0x%08x subdevice 0x%08x\n",
		   si->h_internal_client, si->h_internal_device, si->h_internal_subdevice);
	/* Filled only where ACR places WPR (PDB_PROP_GPU_FW_WPR_OFFSET_SET_BY_ACR); 0 on GA10x. */
	seq_printf(m, "wpr layout:      non-WPR heap 0x%llx, FRTS 0x%llx%s\n",
		   si->fw_non_wpr_heap_offset, si->fw_frts_offset,
		   si->fw_frts_offset || si->fw_non_wpr_heap_offset ? "" : " (not set by ACR on this chip)");
}

/* ---- RM API over RPC ---------------------------------------------------- */

/* rpc_gsp_rm_alloc_v03_00, rpc_gsp_rm_control_v03_00, NVOS00_PARAMETERS_v03_00 */
struct gsp_rm_alloc {
	__le32 h_client, h_parent, h_object, h_class, status, params_size, flags;
	u8 reserved[4];
	u8 params[];
};
static_assert(sizeof(struct gsp_rm_alloc) == 32);

struct gsp_rm_control {
	__le32 h_client, h_object, cmd, status, params_size, flags;
	u8 params[];
};
static_assert(sizeof(struct gsp_rm_control) == 24);

struct gsp_rm_free {
	__le32 h_root, h_object_parent, h_object_old, status;
};

/* Largest params nvdrv sends in one message (no "large RPC" continuation yet). */
#define RM_PARAMS_MAX	(SZ_4K * 15)

/* nvstatuscodes.h */
static const char *nv_status_name(u32 st)
{
	switch (st) {
	case 0x00: return "NV_OK";
	case 0x19: return "INSERT_DUPLICATE_NAME";
	case 0x1b: return "INSUFFICIENT_PERMISSIONS";
	case 0x1f: return "INVALID_ARGUMENT";
	case 0x22: return "INVALID_CLASS";
	case 0x23: return "INVALID_CLIENT";
	case 0x24: return "INVALID_COMMAND";
	case 0x26: return "INVALID_DEVICE";
	case 0x33: return "INVALID_OBJECT_HANDLE";
	case 0x36: return "INVALID_OBJECT_PARENT";
	case 0x3a: return "INVALID_PARAM_STRUCT";
	case 0x40: return "INVALID_STATE";
	case 0x51: return "NO_MEMORY";
	case 0x56: return "NOT_SUPPORTED";
	case 0x57: return "OBJECT_NOT_FOUND";
	case 0x63: return "STATE_IN_USE";
	case 0xffff: return "GENERIC";
	default: return "?";
	}
}

/*
 * The RM status lives in the reply payload; rpc_result in the header says whether the
 * call got through. Either can be nonzero on failure (rpcRmApiAlloc_GSP reads both).
 */
static int rm_status(struct nvdrv_dev *nv, int ret, u32 status, const char *what)
{
	u32 st = status ?: (ret == -EREMOTEIO ? nv->rpc.last_result : 0);

	if (!ret && !status)
		return 0;
	if (ret && ret != -EREMOTEIO)
		dev_err(&nv->pdev->dev, "rm: %s: rpc error %d\n", what, ret);
	else
		dev_err(&nv->pdev->dev, "rm: %s: NV_ERR_%s (status 0x%x, rpc_result 0x%x)\n", what,
			nv_status_name(st), status, ret ? nv->rpc.last_result : 0);
	return ret && ret != -EREMOTEIO ? ret : -EREMOTEIO;
}

/* RmAllocWithHandle through GSP-RM. params is sent and the reply copied back into it. */
int nv_rm_alloc(struct nvdrv_dev *nv, u32 h_client, u32 h_parent, u32 h_object, u32 cls,
		void *params, u32 params_size)
{
	struct gsp_rm_alloc *m;
	u32 size = sizeof(*m) + params_size;
	char what[64];
	int ret;

	if (params_size > RM_PARAMS_MAX)
		return -E2BIG;
	m = kzalloc(size, GFP_KERNEL);
	if (!m)
		return -ENOMEM;
	m->h_client = cpu_to_le32(h_client);
	m->h_parent = cpu_to_le32(h_parent);
	m->h_object = cpu_to_le32(h_object);
	m->h_class = cpu_to_le32(cls);
	m->params_size = cpu_to_le32(params_size);
	if (params_size)
		memcpy(m->params, params, params_size);

	ret = nv_rpc_call(nv, NV_RPC_GSP_RM_ALLOC, m, size, m, size);
	snprintf(what, sizeof(what), "alloc class 0x%04x handle 0x%08x", cls, h_object ?: h_client);
	ret = rm_status(nv, ret, le32_to_cpu(m->status), what);
	if (!ret && params_size)
		memcpy(params, m->params, params_size);
	kfree(m);
	return ret;
}

/* RmControl through GSP-RM. params is sent and the reply copied back into it. */
int nv_rm_control(struct nvdrv_dev *nv, u32 h_client, u32 h_object, u32 cmd,
		  void *params, u32 params_size)
{
	struct gsp_rm_control *m;
	u32 size = sizeof(*m) + params_size;
	char what[48];
	int ret;

	if (params_size > RM_PARAMS_MAX)
		return -E2BIG;
	m = kzalloc(size, GFP_KERNEL);
	if (!m)
		return -ENOMEM;
	m->h_client = cpu_to_le32(h_client);
	m->h_object = cpu_to_le32(h_object);
	m->cmd = cpu_to_le32(cmd);
	m->params_size = cpu_to_le32(params_size);
	memcpy(m->params, params, params_size);

	ret = nv_rpc_call(nv, NV_RPC_GSP_RM_CONTROL, m, size, m, size);
	snprintf(what, sizeof(what), "control 0x%08x on 0x%08x", cmd, h_object);
	ret = rm_status(nv, ret, le32_to_cpu(m->status), what);
	if (!ret)
		memcpy(params, m->params, params_size);
	kfree(m);
	return ret;
}

/* RmFree; freeing a client frees everything under it. */
int nv_rm_free(struct nvdrv_dev *nv, u32 h_client, u32 h_object)
{
	struct gsp_rm_free m = {
		.h_root = cpu_to_le32(h_client),
		.h_object_old = cpu_to_le32(h_object),
	};
	char what[32];
	int ret;

	ret = nv_rpc_call(nv, NV_RPC_FREE, &m, sizeof(m), &m, sizeof(m));
	snprintf(what, sizeof(what), "free 0x%08x", h_object);
	return rm_status(nv, ret, le32_to_cpu(m.status), what);
}

/* ---- nvdrv's own client, device and subdevice --------------------------- */

#define NV01_ROOT		0x0000
#define NV01_DEVICE_0		0x0080
#define NV20_SUBDEVICE_0	0x2080
#define KERNEL_PID		0xffffffff	/* processID for kernel clients */

/*
 * Handles are ours to pick. The client uses the range CPU-RM gives its clients
 * (0xc1d0xxxx); GSP-RM's internal client is 0xc2xxxxxx. Object handles are free-form.
 */
#define NVDRV_H_CLIENT		0xc1d00001
#define NVDRV_H_DEVICE		0xd0000080
#define NVDRV_H_SUBDEVICE	0xd0002080

/* NV0000_ALLOC_PARAMETERS, NV0080_ALLOC_PARAMETERS, NV2080_ALLOC_PARAMETERS (tools/rm_struct_probe.sh) */
struct nv0000_alloc {
	__le32 h_client;
	__le32 process_id;
	char process_name[100];
	__le64 os_pid_info;
};
static_assert(sizeof(struct nv0000_alloc) == 120);
static_assert(offsetof(struct nv0000_alloc, os_pid_info) == 112);

struct nv0080_alloc {
	__le32 device_id, h_client_share, h_target_client, h_target_device, flags, pad0;
	__le64 va_space_size, va_start_internal, va_limit_internal;
	__le32 va_mode, pad1;
};
static_assert(sizeof(struct nv0080_alloc) == 56);
static_assert(offsetof(struct nv0080_alloc, va_space_size) == 24);
static_assert(offsetof(struct nv0080_alloc, va_mode) == 48);

struct nv2080_alloc {
	__le32 sub_device_id;
};

/* NV2080_CTRL_GPU_GET_CHIP_DETAILS_PARAMS */
#define NV2080_CTRL_CMD_GPU_GET_CHIP_DETAILS	0x208001a4
struct nv2080_chip_details {
	__le32 pci_dev_id;
	u8 chip_sku[4];
	__le32 chip_major, chip_minor;
};
static_assert(sizeof(struct nv2080_chip_details) == 16);

/* NV2080_CTRL_INTERNAL_MEMSYS_GET_STATIC_CONFIG_PARAMS */
#define NV2080_CTRL_CMD_INTERNAL_MEMSYS_GET_STATIC_CONFIG	0x20800a1c
struct nv2080_memsys_static_config {
	u8 one_to_one_comptag, one_to_four_comptag, raw_mode_comptag, disable_compbit_backing;
	u8 disable_post_l2_compression, ecc_fbpa, l2_prefill, pad0;
	__le64 l2_cache_size;
	u8 fbpa_present, pad1[3];
	__le32 compr_page_size, compr_page_shift, ram_type, ltc_count, lts_per_ltc;
};
static_assert(sizeof(struct nv2080_memsys_static_config) == 40);
static_assert(offsetof(struct nv2080_memsys_static_config, l2_cache_size) == 8);
static_assert(offsetof(struct nv2080_memsys_static_config, ram_type) == 28);
static_assert(offsetof(struct nv2080_memsys_static_config, lts_per_ltc) == 36);

/*
 * Caller holds gsp_lock, GSP-RM running. Allocates root client -> device ->
 * subdevice the way CPU-RM does for its first device, then asks two questions:
 * one routed control on our own subdevice, one internal control on GSP-RM's
 * internal subdevice (handles from the static info), which is how CPU-RM talks
 * to "physical" RM.
 */
int nv_rm_init_objects(struct nvdrv_dev *nv)
{
	struct device *dev = &nv->pdev->dev;
	struct nv_rm *rm = &nv->rm;
	struct nv0000_alloc root = {
		.h_client = cpu_to_le32(NVDRV_H_CLIENT),
		.process_id = cpu_to_le32(KERNEL_PID),
		.process_name = DRV_NAME,
	};
	struct nv0080_alloc device = {
		.device_id = 0,
		.h_client_share = cpu_to_le32(NVDRV_H_CLIENT),
	};
	struct nv2080_alloc subdevice = { .sub_device_id = 0 };
	struct nv2080_chip_details chip = {};
	struct nv2080_memsys_static_config ms = {};
	int ret;

	lockdep_assert_held(&nv->gsp_lock);
	memset(rm, 0, sizeof(*rm));

	ret = nv_rm_alloc(nv, NVDRV_H_CLIENT, 0, 0, NV01_ROOT, &root, sizeof(root));
	if (ret)
		return ret;
	rm->h_client = NVDRV_H_CLIENT;
	ret = nv_rm_alloc(nv, rm->h_client, rm->h_client, NVDRV_H_DEVICE, NV01_DEVICE_0,
			  &device, sizeof(device));
	if (ret)
		return ret;
	rm->h_device = NVDRV_H_DEVICE;
	ret = nv_rm_alloc(nv, rm->h_client, rm->h_device, NVDRV_H_SUBDEVICE, NV20_SUBDEVICE_0,
			  &subdevice, sizeof(subdevice));
	if (ret)
		return ret;
	rm->h_subdevice = NVDRV_H_SUBDEVICE;
	dev_info(dev, "rm: client 0x%08x -> device 0x%08x -> subdevice 0x%08x allocated\n",
		 rm->h_client, rm->h_device, rm->h_subdevice);

	ret = nv_rm_control(nv, rm->h_client, rm->h_subdevice, NV2080_CTRL_CMD_GPU_GET_CHIP_DETAILS,
			    &chip, sizeof(chip));
	if (ret)
		return ret;
	rm->pci_dev_id = le32_to_cpu(chip.pci_dev_id);
	rm->chip_major = le32_to_cpu(chip.chip_major);
	rm->chip_minor = le32_to_cpu(chip.chip_minor);
	copy_str(rm->chip_sku, sizeof(rm->chip_sku), chip.chip_sku, sizeof(chip.chip_sku));
	rm->chip_valid = true;

	ret = nv_rm_control(nv, nv->gsp_info.h_internal_client, nv->gsp_info.h_internal_subdevice,
			    NV2080_CTRL_CMD_INTERNAL_MEMSYS_GET_STATIC_CONFIG, &ms, sizeof(ms));
	if (ret)
		return ret;
	rm->l2_cache_size = le64_to_cpu(ms.l2_cache_size);
	rm->ram_type = le32_to_cpu(ms.ram_type);
	rm->ltc_count = le32_to_cpu(ms.ltc_count);
	rm->lts_per_ltc = le32_to_cpu(ms.lts_per_ltc);
	rm->compr_page_size = le32_to_cpu(ms.compr_page_size);
	rm->ecc_fbpa = ms.ecc_fbpa;
	rm->fbpa_present = ms.fbpa_present;
	rm->memsys_valid = true;

	dev_info(dev, "rm: chip details: pci dev 0x%08x, sku %s, rev %x.%x; memsys: L2 %llu KiB, %s, %u LTC x %u LTS\n",
		 rm->pci_dev_id, rm->chip_sku, rm->chip_major, rm->chip_minor,
		 rm->l2_cache_size >> 10, ram_type_name(rm->ram_type), rm->ltc_count, rm->lts_per_ltc);
	return 0;
}

/* Caller holds gsp_lock. Frees the client (and with it device and subdevice). */
void nv_rm_fini_objects(struct nvdrv_dev *nv)
{
	struct nv_rm *rm = &nv->rm;

	lockdep_assert_held(&nv->gsp_lock);
	if (rm->h_client && nv->gsp_running && !nv_rm_free(nv, rm->h_client, rm->h_client))
		dev_info(&nv->pdev->dev, "rm: client 0x%08x freed\n", rm->h_client);
	memset(rm, 0, sizeof(*rm));
	nv_disp_forget(nv);
}

/* Caller holds gsp_lock. */
void nv_rm_objects_show(struct nvdrv_dev *nv, struct seq_file *m)
{
	const struct nv_rm *rm = &nv->rm;

	seq_puts(m, "\n[rm objects]\n");
	if (!rm->h_client) {
		seq_puts(m, "none\n");
		return;
	}
	seq_printf(m, "handles:         client 0x%08x device 0x%08x subdevice 0x%08x\n",
		   rm->h_client, rm->h_device, rm->h_subdevice);
	if (rm->chip_valid)
		seq_printf(m, "chip details:    pci dev 0x%08x, sku %s, rev %x.%x\n", rm->pci_dev_id,
			   rm->chip_sku, rm->chip_major, rm->chip_minor);
	if (rm->memsys_valid)
		seq_printf(m, "memsys:          L2 %llu KiB, ram type 0x%x (%s), %u LTC x %u LTS, compr page %u KiB, ECC %s, FBPA %s\n",
			   rm->l2_cache_size >> 10, rm->ram_type, ram_type_name(rm->ram_type),
			   rm->ltc_count, rm->lts_per_ltc, rm->compr_page_size >> 10,
			   rm->ecc_fbpa ? "on" : "off", rm->fbpa_present ? "present" : "absent");
}

/*
 * Error-path check (debugfs rm_selftest): requests GSP-RM must refuse, plus one
 * alloc/free round trip. Logs where the error was reported (rpc_result vs status).
 */
int nv_rm_selftest(struct nvdrv_dev *nv)
{
	struct device *dev = &nv->pdev->dev;
	struct nv_rm *rm = &nv->rm;
	struct nv0000_alloc root = {
		.h_client = cpu_to_le32(rm->h_client),
		.process_id = cpu_to_le32(KERNEL_PID),
	};
	u32 dummy = 0;
	int ret, fails = 0;

	mutex_lock(&nv->gsp_lock);
	if (!nv->gsp_running || !rm->h_subdevice) {
		mutex_unlock(&nv->gsp_lock);
		return -ENODEV;
	}

	ret = nv_rm_alloc(nv, rm->h_client, 0, 0, NV01_ROOT, &root, sizeof(root));
	dev_info(dev, "rm selftest: duplicate client -> %d (rpc_result 0x%x)\n", ret, nv->rpc.last_result);
	fails += ret != -EREMOTEIO;

	ret = nv_rm_control(nv, rm->h_client, rm->h_subdevice, 0x2080ffff, &dummy, sizeof(dummy));
	dev_info(dev, "rm selftest: unknown control -> %d (rpc_result 0x%x)\n", ret, nv->rpc.last_result);
	fails += ret != -EREMOTEIO;

	ret = nv_rm_control(nv, rm->h_client, 0xdeadbeef, NV2080_CTRL_CMD_GPU_GET_CHIP_DETAILS,
			    &(struct nv2080_chip_details){}, sizeof(struct nv2080_chip_details));
	dev_info(dev, "rm selftest: bad handle -> %d (rpc_result 0x%x)\n", ret, nv->rpc.last_result);
	fails += ret != -EREMOTEIO;

	ret = nv_rm_control(nv, rm->h_client, rm->h_subdevice, NV2080_CTRL_CMD_GPU_GET_CHIP_DETAILS,
			    &dummy, sizeof(dummy));
	dev_info(dev, "rm selftest: wrong params size -> %d (rpc_result 0x%x)\n", ret, nv->rpc.last_result);
	fails += ret != -EREMOTEIO;

	/* A second client, freed again: the queues must still be in sync afterwards. */
	root.h_client = cpu_to_le32(NVDRV_H_CLIENT + 1);
	ret = nv_rm_alloc(nv, NVDRV_H_CLIENT + 1, 0, 0, NV01_ROOT, &root, sizeof(root));
	if (!ret)
		ret = nv_rm_free(nv, NVDRV_H_CLIENT + 1, NVDRV_H_CLIENT + 1);
	dev_info(dev, "rm selftest: second client alloc+free -> %d\n", ret);
	fails += ret != 0;

	ret = nv_rm_control(nv, rm->h_client, rm->h_subdevice, NV2080_CTRL_CMD_GPU_GET_CHIP_DETAILS,
			    &(struct nv2080_chip_details){}, sizeof(struct nv2080_chip_details));
	dev_info(dev, "rm selftest: good control afterwards -> %d\n", ret);
	fails += ret != 0;
	mutex_unlock(&nv->gsp_lock);

	dev_info(dev, "rm selftest: %s\n", fails ? "FAILED" : "PASS");
	return fails ? -EIO : 0;
}

/* ---- performance state ------------------------------------------------ */

#define NV2080_CTRL_CMD_INTERNAL_PERF_BOOST_SET_2X	0x20800a9a
#define NV2080_CTRL_CMD_PERF_GET_CURRENT_PSTATE		0x20802068
#define PERF_BOOST_CLEAR				0
#define PERF_BOOST_TO_MAX				2
#define PERF_BOOST_DURATION_INFINITE			0xffffffff

/* NV2080_CTRL_INTERNAL_PERF_BOOST_SET_PARAMS_2X: note flags is an NvBool */
struct nv2080_perf_boost_2x {
	u8 flags, pad[3];
	__le32 duration;
};
static_assert(sizeof(struct nv2080_perf_boost_2x) == 8);

/*
 * kperfBoostSet: what CPU-RM does for NV2080_CTRL_CMD_PERF_BOOST. GSP-RM's governor
 * only sees engine load, not CPU traffic through BAR1, so with nothing but scanout
 * and CPU-drawn frames it parks the GPU in P8 with the PCIe link at 2.5 GT/s.
 */
/* seconds: 0 clears the boost, NV_PERF_BOOST_FOREVER lasts until cleared */
int nv_rm_perf_boost(struct nvdrv_dev *nv, u32 seconds)
{
	bool max = seconds != 0;
	struct nv2080_perf_boost_2x p = {
		.flags = max ? PERF_BOOST_TO_MAX : PERF_BOOST_CLEAR,
		.duration = cpu_to_le32(seconds == NV_PERF_BOOST_FOREVER ? PERF_BOOST_DURATION_INFINITE : seconds),
	};
	int ret;

	mutex_lock(&nv->gsp_lock);
	ret = nv->rm.h_subdevice ? nv_rm_control(nv, nv->rm.h_client, nv->rm.h_subdevice,
						 NV2080_CTRL_CMD_INTERNAL_PERF_BOOST_SET_2X, &p, sizeof(p)) : -ENODEV;
	mutex_unlock(&nv->gsp_lock);
	dev_dbg(&nv->pdev->dev, "rm: perf boost %u s: %d\n", seconds, ret);
	return ret;
}

void nv_rm_perf_show(struct nvdrv_dev *nv, struct seq_file *m)
{
	__le32 pstate = 0;
	u16 lnksta = 0;
	int ret = -ENODEV, i;

	mutex_lock(&nv->gsp_lock);
	if (nv->rm.h_subdevice)
		ret = nv_rm_control(nv, nv->rm.h_client, nv->rm.h_subdevice,
				    NV2080_CTRL_CMD_PERF_GET_CURRENT_PSTATE, &pstate, sizeof(pstate));
	mutex_unlock(&nv->gsp_lock);
	pcie_capability_read_word(nv->pdev, PCI_EXP_LNKSTA, &lnksta);
	i = ffs(le32_to_cpu(pstate));
	if (ret)
		seq_printf(m, "pstate:  unknown (%d)\n", ret);
	else
		seq_printf(m, "pstate:  P%d (0x%x)\n", i ? i - 1 : -1, le32_to_cpu(pstate));
	seq_printf(m, "pcie:    gen%u x%u\n", lnksta & PCI_EXP_LNKSTA_CLS,
		   (lnksta & PCI_EXP_LNKSTA_NLW) >> PCI_EXP_LNKSTA_NLW_SHIFT);
}
