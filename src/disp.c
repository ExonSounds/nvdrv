// SPDX-License-Identifier: GPL-2.0
/*
 * Display (Phase 12, pulled forward): discovery through GSP-RM's display classes.
 *
 * Mirrors what nvkms does at startup (nvidia-modeset/src/nvkms-rm.c):
 *   NV04_DISPLAY_COMMON (0x73) under the device, then
 *   NV0073_CTRL_CMD_SYSTEM_GET_NUM_HEADS / GET_SUPPORTED / GET_CONNECT_STATE,
 *   NV0073_CTRL_CMD_SPECIFIC_GET_TYPE / OR_GET_INFO / GET_EDID_V2 per display.
 * All of these are ROUTE_TO_PHYSICAL controls, i.e. GSP-RM answers them.
 */
#include <linux/delay.h>
#include <linux/seq_file.h>
#include <linux/slab.h>

#include "nvdrv.h"

#define NV04_DISPLAY_COMMON	0x0073
#define NVDRV_H_DISP_COMMON	0xd0000073

#define NV0073_CTRL_CMD_SYSTEM_GET_NUM_HEADS		0x730102
#define NV0073_CTRL_CMD_SYSTEM_GET_SUPPORTED		0x730107
#define NV0073_CTRL_CMD_SYSTEM_GET_CONNECT_STATE	0x730108
#define NV0073_CTRL_CMD_SPECIFIC_GET_TYPE		0x730240
#define NV0073_CTRL_CMD_SPECIFIC_GET_EDID_V2		0x730245
#define NV0073_CTRL_CMD_SPECIFIC_OR_GET_INFO		0x73028b

struct nv0073_num_heads {
	__le32 sub_device_instance, flags, num_heads;
};
struct nv0073_supported {
	__le32 sub_device_instance, display_mask, display_mask_ddc;
};
struct nv0073_connect_state {
	__le32 sub_device_instance, flags, display_mask, retry_time_ms;
};
struct nv0073_get_type {
	__le32 sub_device_instance, display_id, display_type;
};
struct nv0073_or_info {
	__le32 sub_device_instance, display_id, index, type, protocol;
	__le32 dither_type, dither_algo, location, root_port_id, dcb_index;
	__le64 vbios_address;
	u8 lit_by_vbios, disp_dynamic, pad[6];
};
static_assert(sizeof(struct nv0073_or_info) == 56);
static_assert(offsetof(struct nv0073_or_info, vbios_address) == 40);

struct nv0073_edid_v2 {
	__le32 sub_device_instance, display_id, buffer_size, flags;
	u8 edid[NV_EDID_MAX];
};
static_assert(sizeof(struct nv0073_edid_v2) == 16 + 2048);

static const char *or_type_name(u32 t)
{
	static const char *const n[] = { "none", "DAC", "SOR", "PIOR", "?", "DSI" };

	return t < ARRAY_SIZE(n) ? n[t] : "?";
}

static const char *sor_protocol_name(u32 p)
{
	switch (p) {
	case 0x0: return "LVDS";
	case 0x1: return "TMDS-A";
	case 0x2: return "TMDS-B";
	case 0x5: return "dual TMDS";
	case 0x8: return "DP-A";
	case 0x9: return "DP-B";
	case 0x10: return "DSI";
	default: return "?";
	}
}

/* EDID base block, first detailed timing descriptor (the preferred mode). */
static void edid_preferred_mode(struct nv_disp_dpy *d)
{
	const u8 *t = d->edid + 54;
	struct nv_mode *m = &d->mode;
	u32 hblank, vblank;

	if (d->edid_size < 128 || !(t[0] | t[1]))
		return;
	m->clock_khz = (t[0] | t[1] << 8) * 10;
	m->hdisplay = t[2] | (t[4] & 0xf0) << 4;
	hblank = t[3] | (t[4] & 0x0f) << 8;
	m->vdisplay = t[5] | (t[7] & 0xf0) << 4;
	vblank = t[6] | (t[7] & 0x0f) << 8;
	m->hsync_start = m->hdisplay + (t[8] | (t[11] & 0xc0) << 2);
	m->hsync_end = m->hsync_start + (t[9] | (t[11] & 0x30) << 4);
	m->vsync_start = m->vdisplay + ((t[10] >> 4) | (t[11] & 0x0c) << 2);
	m->vsync_end = m->vsync_start + ((t[10] & 0x0f) | (t[11] & 0x03) << 4);
	m->htotal = m->hdisplay + hblank;
	m->vtotal = m->vdisplay + vblank;
	m->hsync_pos = !!(t[17] & 0x02);
	m->vsync_pos = !!(t[17] & 0x04);
	m->valid = m->htotal && m->vtotal;
}

/* Caller holds gsp_lock. */
static int disp_probe_locked(struct nvdrv_dev *nv)
{
	struct device *dev = &nv->pdev->dev;
	struct nv_disp *dp = &nv->disp;
	struct nv_rm *rm = &nv->rm;
	struct nv0073_num_heads nh = {};
	struct nv0073_supported sup = {};
	struct nv0073_connect_state cs = {};
	struct nv0073_edid_v2 *edid;
	u32 h = NVDRV_H_DISP_COMMON, bit;
	int ret;

	if (!nv->gsp_running || !rm->h_device)
		return -ENODEV;
	if (!dp->h_common) {
		ret = nv_rm_alloc(nv, rm->h_client, rm->h_device, h, NV04_DISPLAY_COMMON, NULL, 0);
		if (ret)
			return ret;
		dp->h_common = h;
	}

	ret = nv_rm_control(nv, rm->h_client, h, NV0073_CTRL_CMD_SYSTEM_GET_NUM_HEADS, &nh, sizeof(nh));
	if (!ret)
		ret = nv_rm_control(nv, rm->h_client, h, NV0073_CTRL_CMD_SYSTEM_GET_SUPPORTED,
				    &sup, sizeof(sup));
	if (ret)
		return ret;
	dp->num_heads = le32_to_cpu(nh.num_heads);
	dp->supported = le32_to_cpu(sup.display_mask);

	cs.display_mask = sup.display_mask;
	ret = nv_rm_control(nv, rm->h_client, h, NV0073_CTRL_CMD_SYSTEM_GET_CONNECT_STATE,
			    &cs, sizeof(cs));
	if (ret)
		return ret;
	dp->connected = le32_to_cpu(cs.display_mask);
	dev_info(dev, "disp: %u heads, displays supported 0x%x, connected 0x%x\n",
		 dp->num_heads, dp->supported, dp->connected);

	edid = kzalloc(sizeof(*edid), GFP_KERNEL);
	if (!edid)
		return -ENOMEM;
	dp->ndpy = 0;
	for (bit = 0; bit < 32 && dp->ndpy < NV_DISP_MAX_DPY; bit++) {
		struct nv_disp_dpy *d = &dp->dpy[dp->ndpy];
		struct nv0073_get_type ty = {};
		struct nv0073_or_info oi = {};

		if (!(dp->supported & BIT(bit)))
			continue;
		memset(d, 0, sizeof(*d));
		d->id = BIT(bit);
		d->connected = !!(dp->connected & d->id);
		ty.display_id = cpu_to_le32(d->id);
		oi.display_id = cpu_to_le32(d->id);
		ret = nv_rm_control(nv, rm->h_client, h, NV0073_CTRL_CMD_SPECIFIC_GET_TYPE, &ty, sizeof(ty));
		if (!ret)
			ret = nv_rm_control(nv, rm->h_client, h, NV0073_CTRL_CMD_SPECIFIC_OR_GET_INFO,
					    &oi, sizeof(oi));
		if (ret)
			break;
		d->type = le32_to_cpu(ty.display_type);
		d->or_type = le32_to_cpu(oi.type);
		d->or_index = le32_to_cpu(oi.index);
		d->protocol = le32_to_cpu(oi.protocol);
		d->dcb_index = le32_to_cpu(oi.dcb_index);
		d->lit_by_vbios = oi.lit_by_vbios;

		if (d->connected) {
			memset(edid, 0, sizeof(*edid));
			edid->display_id = cpu_to_le32(d->id);
			edid->buffer_size = cpu_to_le32(NV_EDID_MAX);
			if (!nv_rm_control(nv, rm->h_client, h, NV0073_CTRL_CMD_SPECIFIC_GET_EDID_V2,
					   edid, sizeof(*edid))) {
				d->edid_size = min_t(u32, le32_to_cpu(edid->buffer_size), NV_EDID_MAX);
				memcpy(d->edid, edid->edid, d->edid_size);
				edid_preferred_mode(d);
			}
		}
		dev_info(dev, "disp: display 0x%08x: %s, %s%u %s, dcb %u%s%s, EDID %u bytes, mode %ux%u @ %u kHz\n",
			 d->id, d->connected ? "CONNECTED" : "not connected", or_type_name(d->or_type),
			 d->or_index, d->or_type == 2 ? sor_protocol_name(d->protocol) : "",
			 d->dcb_index, d->lit_by_vbios ? ", lit by VBIOS" : "",
			 d->type == 2 ? ", DFP" : "", d->edid_size, d->mode.hdisplay, d->mode.vdisplay,
			 d->mode.clock_khz);
		dp->ndpy++;
	}
	kfree(edid);
	return ret;
}

int nv_disp_probe(struct nvdrv_dev *nv)
{
	int ret;

	mutex_lock(&nv->gsp_lock);
	ret = disp_probe_locked(nv);
	if (!ret)
		ret = nv_evo_init(nv);
	mutex_unlock(&nv->gsp_lock);
	return ret;
}

/* ---- DisplayPort link (EvoMainLink in common/displayport/src/dp_evoadapter.cpp) ---- */

#define NV0073_CTRL_CMD_DFP_ASSIGN_SOR		0x731152
#define NV0073_CTRL_CMD_DP_AUXCH_CTRL		0x731341
#define NV0073_CTRL_CMD_DP_CTRL			0x731343
#define NV0073_CTRL_CMD_DP_GET_LINK_CONFIG	0x731360

struct nv0073_assign_sor {
	__le32 sub_device_instance, display_id;
	u8 sor_exclude_mask, pad0[3];
	__le32 slave_display_id, force_sublink_config;
	u8 is_2head_1or, pad1[3];
	__le32 sor_assign_list[4];
	struct { __le32 display_mask, sor_type; } sor_assign_list_with_tag[4];
	u8 reserved_sor_mask, pad2[3];
	__le32 flags;
};
static_assert(sizeof(struct nv0073_assign_sor) == 80);
static_assert(offsetof(struct nv0073_assign_sor, sor_assign_list) == 24);
static_assert(offsetof(struct nv0073_assign_sor, flags) == 76);

struct nv0073_auxch {
	__le32 sub_device_instance, display_id;
	u8 addr_only, pad[3];
	__le32 cmd, addr;
	u8 data[16];
	__le32 size, reply_type, retry_time_ms;
};
static_assert(sizeof(struct nv0073_auxch) == 48);

struct nv0073_dp_ctrl {
	__le32 sub_device_instance, display_id, cmd, data, err, retry_time_ms, eight_lane_dpcd_base;
};

struct nv0073_dp_link_config {
	__le32 sub_device_instance, display_id, lane_count, link_bw, dp2_link_bw;
};

#define AUX_CMD_READ	(BIT(3) | 1)	/* TYPE_AUX | REQ_TYPE_READ */
#define AUX_CMD_WRITE	BIT(3)		/* TYPE_AUX | REQ_TYPE_WRITE */

static int dp_aux(struct nvdrv_dev *nv, u32 dpy, bool write, u32 addr, u8 *buf, u32 len)
{
	struct nv0073_auxch a = {};
	int ret, tries;

	if (!len || len > 16)
		return -EINVAL;
	for (tries = 0; tries < 3; tries++) {
		memset(&a, 0, sizeof(a));
		a.display_id = cpu_to_le32(dpy);
		a.cmd = cpu_to_le32(write ? AUX_CMD_WRITE : AUX_CMD_READ);
		a.addr = cpu_to_le32(addr);
		a.size = cpu_to_le32(len - 1);
		if (write)
			memcpy(a.data, buf, len);
		ret = nv_rm_control(nv, nv->rm.h_client, nv->disp.h_common, NV0073_CTRL_CMD_DP_AUXCH_CTRL,
				    &a, sizeof(a));
		if (!ret)
			break;
		msleep(max_t(u32, le32_to_cpu(a.retry_time_ms), 5));
	}
	if (ret)
		return ret;
	if (!write)
		memcpy(buf, a.data, min_t(u32, len, le32_to_cpu(a.size)));
	return le32_to_cpu(a.size) == len ? 0 : -EIO;
}

/*
 * Bring up the DP link of display dpy: read the sink's capabilities, power it up,
 * get a SOR assigned, and have RM train the link at the highest rate that works.
 */
static int dp_link_up(struct nvdrv_dev *nv, struct nv_disp_dpy *d)
{
	static const u8 rates[] = { 0x1e, 0x14, 0x0a, 0x06 };	/* 8.1/5.4/2.7/1.62 Gbps, 270 MHz units */
	struct device *dev = &nv->pdev->dev;
	struct nv_disp *dp = &nv->disp;
	struct nv0073_assign_sor as = {};
	struct nv0073_dp_link_config lc = {};
	u8 caps[16], d0 = 1, st[6];
	u32 i, lanes;
	int ret;

	ret = dp_aux(nv, d->id, false, 0x000, caps, 16);
	if (ret) {
		dev_err(dev, "disp: DPCD read failed: %d\n", ret);
		return ret;
	}
	d->dpcd_rev = caps[0];
	d->max_link_bw = caps[1];
	d->max_lanes = caps[2] & 0x1f;
	d->enhanced_framing = !!(caps[2] & 0x80);
	dev_info(dev, "disp: sink DPCD %u.%u, max link 0x%02x x %u lanes%s, DPCD[0..15] %*ph\n",
		 caps[0] >> 4, caps[0] & 0xf, d->max_link_bw, d->max_lanes,
		 d->enhanced_framing ? ", enhanced framing" : "", 16, caps);
	if (dp_aux(nv, d->id, true, 0x600, &d0, 1))	/* SET_POWER: D0 */
		dev_warn(dev, "disp: could not set sink power state D0\n");
	msleep(1);

	as.display_id = cpu_to_le32(d->id);
	ret = nv_rm_control(nv, nv->rm.h_client, dp->h_common, NV0073_CTRL_CMD_DFP_ASSIGN_SOR, &as, sizeof(as));
	if (ret)
		return ret;
	d->sor = ~0u;
	for (i = 0; i < 4; i++)
		if (le32_to_cpu(as.sor_assign_list[i]) & d->id)
			d->sor = i;
	dev_info(dev, "disp: SOR assignment %08x %08x %08x %08x -> SOR%d\n",
		 le32_to_cpu(as.sor_assign_list[0]), le32_to_cpu(as.sor_assign_list[1]),
		 le32_to_cpu(as.sor_assign_list[2]), le32_to_cpu(as.sor_assign_list[3]), (int)d->sor);
	if (d->sor == ~0u)
		return -ENODEV;

	lanes = d->max_lanes;
	for (i = 0; i < ARRAY_SIZE(rates); i++) {
		struct nv0073_dp_ctrl c = {};
		u32 cmd = BIT(0) | BIT(1) | (d->enhanced_framing ? BIT(7) : 0);

		if (rates[i] > d->max_link_bw)
			continue;
		c.display_id = cpu_to_le32(d->id);
		c.cmd = cpu_to_le32(cmd);
		c.data = cpu_to_le32(lanes | rates[i] << 8);
		ret = nv_rm_control(nv, nv->rm.h_client, dp->h_common, NV0073_CTRL_CMD_DP_CTRL, &c, sizeof(c));
		dev_info(dev, "disp: link training %u lanes @ 0x%02x: %d (err 0x%x)\n", lanes, rates[i], ret,
			 le32_to_cpu(c.err));
		if (!ret)
			break;
	}
	if (ret)
		return ret;

	lc.display_id = cpu_to_le32(d->id);
	if (!nv_rm_control(nv, nv->rm.h_client, dp->h_common, NV0073_CTRL_CMD_DP_GET_LINK_CONFIG, &lc, sizeof(lc))) {
		d->link_lanes = le32_to_cpu(lc.lane_count) & 0xf;
		d->link_bw = le32_to_cpu(lc.link_bw);
	}
	if (!dp_aux(nv, d->id, false, 0x202, st, 6))
		dev_info(dev, "disp: link up: %u lanes x 0x%02x; lane status %*ph\n", d->link_lanes, d->link_bw,
			 6, st);
	return 0;
}

int nv_disp_link(struct nvdrv_dev *nv)
{
	struct nv_disp *dp = &nv->disp;
	int ret = -ENODEV;
	u32 i;

	mutex_lock(&nv->gsp_lock);
	for (i = 0; i < dp->ndpy; i++) {
		struct nv_disp_dpy *d = &dp->dpy[i];

		if (d->connected && (d->protocol == 0x8 || d->protocol == 0x9)) {
			ret = dp_link_up(nv, d);
			break;
		}
	}
	mutex_unlock(&nv->gsp_lock);
	return ret;
}

/* Called from nv_rm_fini_objects: the objects die with the client. */
void nv_disp_forget(struct nvdrv_dev *nv)
{
	memset(&nv->disp, 0, sizeof(nv->disp));
}

/* Caller holds gsp_lock. */
void nv_disp_show(struct nvdrv_dev *nv, struct seq_file *m)
{
	const struct nv_disp *dp = &nv->disp;
	u32 i;

	seq_puts(m, "\n[display]\n");
	if (!dp->h_common) {
		seq_puts(m, "not probed\n");
		return;
	}
	seq_printf(m, "heads:           %u\n", dp->num_heads);
	seq_printf(m, "displays:        supported 0x%x, connected 0x%x\n", dp->supported, dp->connected);
	for (i = 0; i < dp->ndpy; i++) {
		const struct nv_disp_dpy *d = &dp->dpy[i];
		const struct nv_mode *md = &d->mode;

		seq_printf(m, "dpy 0x%08x:    %s %s%u %s dcb %u%s\n", d->id,
			   d->connected ? "connected" : "-", or_type_name(d->or_type), d->or_index,
			   d->or_type == 2 ? sor_protocol_name(d->protocol) : "", d->dcb_index,
			   d->lit_by_vbios ? " lit-by-vbios" : "");
		if (md->valid)
			seq_printf(m, "  preferred:     %ux%u, %u kHz, h %u %u %u, v %u %u %u, %chsync %cvsync\n",
				   md->hdisplay, md->vdisplay, md->clock_khz, md->hsync_start,
				   md->hsync_end, md->htotal, md->vsync_start, md->vsync_end, md->vtotal,
				   md->hsync_pos ? '+' : '-', md->vsync_pos ? '+' : '-');
	}
}
