// SPDX-License-Identifier: GPL-2.0
/*
 * First modeset: one DisplayPort sink, its EDID preferred mode, head 0, window 0,
 * a pitch-linear X8R8G8B8 framebuffer in VRAM filled with a test pattern.
 *
 * Method sequences follow nvkms' NVDisplay C6 HAL (nvidia-modeset/src/nvkms-evo3.c,
 * 570.144): nvEvoInitWindowMappingC5, EvoSetRasterParamsC6, EvoHeadSetControlORC5,
 * EvoSORSetControlC3, EvoSetViewportInOutC5, EvoHeadSetDisplayIdC3, nvEvoFlipC6 /
 * EvoFlipC3Common / EvoFlipC5Common, nvEvoUpdateC3 (window + core interlocked).
 * DP stream parameters follow isModePossibleSST (common/displayport/src/dp_watermark.cpp).
 * Supervisor interrupts (clocks, SOR attach) are handled by GSP-RM.
 */
#include <linux/delay.h>
#include <linux/dma-mapping.h>
#include <linux/iopoll.h>
#include <linux/seq_file.h>

#include "nvdrv.h"

/* NVC67D core channel methods (clc67d.h) */
#define C67D_UPDATE				0x0200
#define C67D_SET_INTERLOCK_FLAGS		0x0218
#define C67D_SET_WINDOW_INTERLOCK_FLAGS		0x021c
#define C67D_GET_RG_SCAN_LINE(h)		(0x0220 + (h) * 4)	/* control area, read */
#define C67D_SOR_SET_CONTROL(s)			(0x0300 + (s) * 0x20)
#define C67D_WINDOW_SET_CONTROL(w)		(0x1000 + (w) * 0x80)
#define C67D_WINDOW_SET_FORMAT_USAGE_BOUNDS(w)	(0x1004 + (w) * 0x80)
#define C67D_WINDOW_SET_MAX_INPUT_SCALE(w)	(0x100c + (w) * 0x80)
#define C67D_WINDOW_SET_USAGE_BOUNDS(w)		(0x1010 + (w) * 0x80)
#define C67D_HEAD(h, m)				((m) + (h) * 0x400)
#define   HEAD_SET_PROCAMP			0x2000
#define   HEAD_SET_CONTROL_OUTPUT_RESOURCE	0x2004
#define   HEAD_SET_CONTROL			0x2008
#define   HEAD_SET_PIXEL_CLOCK_FREQUENCY	0x200c
#define   HEAD_SET_DITHER_CONTROL		0x2018
#define   HEAD_SET_PIXEL_CLOCK_CONFIGURATION	0x201c
#define   HEAD_SET_DISPLAY_ID0			0x2020
#define   HEAD_SET_PIXEL_CLOCK_FREQUENCY_MAX	0x2028
#define   HEAD_SET_MAX_OUTPUT_SCALE_FACTOR	0x202c
#define   HEAD_SET_HEAD_USAGE_BOUNDS		0x2030
#define   HEAD_SET_VIEWPORT_POINT_IN		0x2048
#define   HEAD_SET_VIEWPORT_SIZE_IN		0x204c
#define   HEAD_SET_VIEWPORT_SIZE_OUT		0x2058
#define   HEAD_SET_VIEWPORT_POINT_OUT_ADJUST	0x205c
#define   HEAD_SET_RASTER_SIZE			0x2064
#define   HEAD_SET_RASTER_SYNC_END		0x2068
#define   HEAD_SET_RASTER_BLANK_END		0x206c
#define   HEAD_SET_RASTER_BLANK_START		0x2070
#define   HEAD_SET_MIN_FRAME_IDLE		0x2218
#define   HEAD_SET_TILE_POSITION		0x22d0
#define   HEAD_SET_RASTER_HBLANK_DELAY		0x2364

/* NVC67E window channel methods (clc67e.h) */
#define C67E_UPDATE				0x0200
#define C67E_SET_SIZE				0x0224
#define C67E_SET_STORAGE			0x0228
#define C67E_SET_PARAMS				0x022c
#define C67E_SET_PLANAR_STORAGE(p)		(0x0230 + (p) * 4)
#define C67E_SET_CONTEXT_DMA_ISO(i)		(0x0240 + (i) * 4)
#define C67E_SET_OFFSET(i)			(0x0260 + (i) * 4)
#define C67E_SET_POINT_IN(i)			(0x0290 + (i) * 4)
#define C67E_SET_SIZE_IN			0x0298
#define C67E_SET_SIZE_OUT			0x02a4
#define C67E_SET_CONTROL_INPUT_SCALER		0x02a8
#define C67E_SET_COMPOSITION_CONTROL		0x02ec
#define C67E_SET_COMPOSITION_CONSTANT_ALPHA	0x02f0
#define C67E_SET_COMPOSITION_FACTOR_SELECT	0x02f4
#define C67E_SET_PRESENT_CONTROL		0x0308
#define C67E_SET_SCAN_DIRECTION			0x033c
#define C67E_SET_INTERLOCK_FLAGS		0x0370
#define C67E_SET_WINDOW_INTERLOCK_FLAGS		0x0374
#define C67E_SET_FMT_COEFFICIENT_C00		0x0400
#define C67E_SET_ILUT_CONTROL			0x0440
#define C67E_SET_CONTEXT_DMA_ILUT		0x0444
#define C67E_SET_CSC00CONTROL			0x045c
#define C67E_SET_CSC0LUT_CONTROL		0x04a0
#define C67E_SET_CSC01CONTROL			0x04bc
#define C67E_SET_TMO_CONTROL			0x0500
#define C67E_SET_CONTEXT_DMA_TMO_LUT		0x0528
#define C67E_SET_CSC10CONTROL			0x053c
#define C67E_SET_CSC1LUT_CONTROL		0x0580
#define C67E_SET_CSC11CONTROL			0x059c
#define C67E_FORMAT_X8R8G8B8			0xe6
#define C67E_SW_SET_MCLK_SWITCH			0x02b4	/* NVC57E_SW_SET_MCLK_SWITCH (clc57esw.h), RM-trapped */

#define SCALE_1X				1024	/* NV_EVO_SCALE_FACTOR_1X */

#define NVDRV_H_FB_CTXDMA			0xd000f100
#define NVDRV_H_SYSFB_CTXDMA			0xd000f101

#define NV0073_CTRL_CMD_DP_CONFIG_STREAM	0x731362
#define NV0073_CTRL_CMD_DP_GET_CAPS		0x731369

struct nv0073_dp_config_stream {
	__le32 sub_device_instance, head, sor_index, dp_link;
	u8 enable_override, mst, pad0[2];
	__le32 single_head_multistream_mode, hblank_sym, vblank_sym, color_format;
	u8 enable_two_head_one_or, pad1[3];
	u8 mst_params[28];
	u8 sst_enhanced_framing, pad2[3];
	__le32 sst_tu_size, sst_watermark;
	u8 sst_audio_over_right_panel, pad3[3];
};
static_assert(sizeof(struct nv0073_dp_config_stream) == 84);
static_assert(offsetof(struct nv0073_dp_config_stream, sst_enhanced_framing) == 68);
static_assert(offsetof(struct nv0073_dp_config_stream, sst_watermark) == 76);

struct nv0073_dp_get_caps {
	__le32 sub_device_instance, sor_index, max_link_rate;
	u8 rest[64 - 12];
};

static void m1(struct nv_evo_chan *c, u32 mthd, u32 val)
{
	nv_evo_mthd(c, mthd, &val, 1);
}

/* isModePossibleSST, 8b/10b, no DSC/FEC/audio. */
static int dp_watermark(const struct nv_mode *m, u32 lanes, u32 link_bw_270m, bool ef, bool increased,
			u32 *tu, u32 *wm, u32 *hblank_sym, u32 *vblank_sym)
{
	const u64 pf = 100000, depth = 24, pclk = (u64)m->clock_khz * 1000;
	u64 rate = (u64)link_bw_270m * 270000000ULL;	/* line rate per lane, bit/s */
	u64 lane_data, ratio, wmf, blank_clocks;
	s64 hb, vb;
	u32 adjust = increased ? 8 : 2, minimum = increased ? 22 : 20, blank_bits, min_hblank;

	rate -= rate * 6 / 1000;				/* 0.6% downspread */
	lane_data = rate * 8 / 10 / 8;				/* bytes/s per lane */
	if (pclk * depth >= 8 * lane_data * lanes)
		return -ERANGE;
	*tu = 64;
	ratio = pclk * depth * pf / (8 * lane_data * lanes);
	wmf = ratio * *tu * (pf - ratio) / pf;
	*wm = adjust + (2 * (depth * pf / (8 * lanes)) + wmf) / pf;
	if (*wm > 39)
		return -ERANGE;
	*wm = max(*wm, minimum);

	blank_bits = 3 * 8 * lanes + (ef ? 3 * 8 * lanes : 0) + 3 * 8 * 4;
	if (m->hdisplay % lanes)
		blank_bits += (lanes - m->hdisplay % lanes) * depth;
	blank_clocks = (u64)blank_bits * pf / (8 * lanes);
	min_hblank = blank_clocks * pclk / lane_data / pf + 12;
	if (min_hblank > m->htotal - m->hdisplay)
		return -ERANGE;
	hb = (s64)((u64)(m->htotal - m->hdisplay - min_hblank) * lane_data / pclk) - 1 - 3 -
	     (lanes == 1 ? 9 : lanes == 2 ? 6 : 3);
	*hblank_sym = hb < 0 ? 0 : hb;
	vb = (s64)((u64)(m->hdisplay - 40) * lane_data / pclk) - 1 - (lanes == 1 ? 39 : lanes == 2 ? 21 : 12);
	*vblank_sym = vb < 0 ? 0 : vb;
	return 0;
}

/* Test pattern: 8 vertical colour bars on top, a grey ramp below, white border. */
struct pattern { u32 w, h, pitch_px; };

static u32 pattern_px(void *ctx, u64 i)
{
	static const u32 bars[8] = { 0xffffff, 0xffff00, 0x00ffff, 0x00ff00,
				     0xff00ff, 0xff0000, 0x0000ff, 0x000000 };
	const struct pattern *p = ctx;
	u32 x = i % p->pitch_px, y = i / p->pitch_px, g;

	if (x >= p->w)
		return 0;
	if (x < 8 || y < 8 || x >= p->w - 8 || y >= p->h - 8)
		return 0xffffffff;
	if (y < p->h * 2 / 3)
		return 0xff000000 | bars[x * 8 / p->w];
	g = x * 255 / p->w;
	return 0xff000000 | g << 16 | g << 8 | g;
}

/* Window channel methods for a pitch-linear X8R8G8B8 surface (EvoFlipC3Common/C5Common). */
static void win_methods(struct nv_evo_chan *win, u32 flags, u32 ctxdma, u32 w, u32 h, u32 pitch,
			bool interlock_core)
{
	u32 i;

	m1(win, C67E_SET_PRESENT_CONTROL, 0);		/* MIN_PRESENT_INTERVAL 0, NON_TEARING, MONO */
	for (i = 0; i < 6; i++) {
		m1(win, C67E_SET_CONTEXT_DMA_ISO(i), i ? 0 : ctxdma);
		m1(win, C67E_SET_OFFSET(i), 0);
	}
	m1(win, C67E_SET_SIZE, w | h << 16);
	m1(win, C67E_SET_SIZE_IN, w | h << 16);
	m1(win, C67E_SET_SIZE_OUT, w | h << 16);
	m1(win, C67E_SET_STORAGE, 0);
	m1(win, C67E_SET_PLANAR_STORAGE(0), pitch >> 6);
	m1(win, C67E_SET_PLANAR_STORAGE(1), 0);
	m1(win, C67E_SET_PLANAR_STORAGE(2), 0);
	m1(win, C67E_SET_POINT_IN(0), 0);
	m1(win, C67E_SET_SCAN_DIRECTION, 0);		/* from top, from left */
	m1(win, C67E_SET_PARAMS, C67E_FORMAT_X8R8G8B8);
	if (!(flags & NV_FLIP_NO_FMT))
		for (i = 0; i < 12; i++)		/* FMT: identity, S5.16 */
			m1(win, C67E_SET_FMT_COEFFICIENT_C00 + 4 * i, (i % 5 == 0) ? 0x10000 : 0);
	m1(win, C67E_SET_CONTROL_INPUT_SCALER, 1 | 1 << 4);	/* 2 taps */
	if (flags & NV_FLIP_CSC) {
	m1(win, C67E_SET_CSC00CONTROL, 0);
	m1(win, C67E_SET_CSC0LUT_CONTROL, 0);
	m1(win, C67E_SET_CSC01CONTROL, 0);
	m1(win, C67E_SET_CSC10CONTROL, 0);
	m1(win, C67E_SET_CSC1LUT_CONTROL, 0);
	m1(win, C67E_SET_CSC11CONTROL, 0);
	m1(win, C67E_SET_ILUT_CONTROL, 0);
	m1(win, C67E_SET_CONTEXT_DMA_ILUT, 0);
	m1(win, C67E_SET_TMO_CONTROL, 0);
	m1(win, C67E_SET_CONTEXT_DMA_TMO_LUT, 0);
	}
	if (!(flags & NV_FLIP_COMPOSE)) {
		m1(win, C67E_SET_COMPOSITION_CONTROL, BIT(16));	/* BYPASS */
	} else {
		m1(win, C67E_SET_COMPOSITION_CONTROL, 0);	/* no colour key, depth 0 */
		m1(win, C67E_SET_COMPOSITION_CONSTANT_ALPHA, 0xff);
		m1(win, C67E_SET_COMPOSITION_FACTOR_SELECT, 1 << 0 | 1 << 4);	/* src ONE, dst ZERO */
	}
	m1(win, C67E_SET_INTERLOCK_FLAGS, interlock_core);	/* INTERLOCK_WITH_CORE */
	m1(win, C67E_SET_WINDOW_INTERLOCK_FLAGS, 0);
	if (flags & NV_FLIP_MCLK_WAR)
		m1(win, C67E_SW_SET_MCLK_SWITCH, 0);	/* ENABLE_FALSE (bug 2193096 WAR) */
	m1(win, C67E_UPDATE, 1);			/* RELEASE_ELV */
	if (flags & NV_FLIP_MCLK_WAR)
		m1(win, C67E_SW_SET_MCLK_SWITCH, 1);

}

static int modeset_locked(struct nvdrv_dev *nv, u32 flags)
{
	struct device *dev = &nv->pdev->dev;
	struct nv_disp *dp = &nv->disp;
	struct nv_evo *e = &nv->evo;
	struct nv_evo_chan *core = &e->core, *win = &e->win[0];
	struct nv_disp_dpy *d = NULL;
	struct nv0073_dp_config_stream cs = {};
	struct nv0073_dp_get_caps caps = {};
	const struct nv_mode *m;
	struct pattern pat;
	u32 i, head = 0, w, h, pitch, tu, wm, hbs, vbs, leading, trailing, get, sl;
	u32 hsw, vsw, hbe, vbe;
	u64 fb_size;
	bool increased;
	int ret;

	for (i = 0; i < dp->ndpy; i++)
		if (dp->dpy[i].connected && dp->dpy[i].link_lanes && dp->dpy[i].mode.valid)
			d = &dp->dpy[i];
	if (!d || !core->handle)
		return -ENODEV;
	m = &d->mode;
	w = m->hdisplay;
	h = m->vdisplay;
	pitch = ALIGN(w * 4, 256);
	fb_size = ALIGN((u64)pitch * h, SZ_64K);

	/* 1. framebuffer + test pattern, ctxdma bound to window 0's channel */
	if (!e->fb) {
		ret = nv_vram_alloc(nv, fb_size, SZ_1M, &e->fb);
		if (ret)
			return ret;
		e->fb_size = fb_size;
		pat = (struct pattern){ w, h, pitch / 4 };
		nv_vram_fill(nv, e->fb, fb_size, 0, pattern_px, &pat);
		dev_info(dev, "modeset: framebuffer %ux%u pitch %u at VRAM 0x%llx\n", w, h, pitch, e->fb);
	}
	ret = nv_evo_window_init(nv, 0);
	if (ret)
		return ret;
	if (!e->fb_bound) {
		ret = nv_evo_ctxdma(nv, NVDRV_H_FB_CTXDMA, e->fb, e->fb_size, win->chn, false);
		if (ret)
			return ret;
		e->fb_bound = true;
	}

	/* 2. DP stream parameters for the trained link */
	ret = nv_evo_begin(nv, core, 512);
	if (!ret)
		ret = nv_evo_begin(nv, win, 768);
	if (ret)
		return ret;
	caps.sor_index = cpu_to_le32(d->sor);
	increased = !nv_rm_control(nv, nv->rm.h_client, dp->h_common, NV0073_CTRL_CMD_DP_GET_CAPS,
				   &caps, sizeof(caps)) && caps.rest[26 - 12];
	ret = dp_watermark(m, d->link_lanes, d->link_bw, d->enhanced_framing, increased, &tu, &wm, &hbs, &vbs);
	if (ret) {
		dev_err(dev, "modeset: mode does not fit the link (%u lanes x 0x%x)\n", d->link_lanes, d->link_bw);
		return ret;
	}
	cs.head = cpu_to_le32(head);
	cs.sor_index = cpu_to_le32(d->sor);
	cs.dp_link = cpu_to_le32(d->protocol == 0x9 ? 1 : 0);
	cs.enable_override = 1;
	cs.hblank_sym = cpu_to_le32(hbs);
	cs.vblank_sym = cpu_to_le32(vbs);
	cs.color_format = 0;	/* RGB */
	cs.sst_enhanced_framing = d->enhanced_framing;
	cs.sst_tu_size = cpu_to_le32(tu);
	cs.sst_watermark = cpu_to_le32(wm);
	ret = (flags & NV_MODESET_NO_STREAM_CONFIG) ? -ECANCELED :
	      nv_rm_control(nv, nv->rm.h_client, dp->h_common, NV0073_CTRL_CMD_DP_CONFIG_STREAM, &cs, sizeof(cs));
	dev_info(dev, "modeset: DP stream: tu %u watermark %u hblank %u vblank %u (increased limits %d): %s\n",
		 tu, wm, hbs, vbs, increased, ret ? "refused, continuing with RM defaults" : "set");

	/*
	 * 3a. window -> head mapping in its own core update, not interlocked with the
	 * window (EvoInitWindowMapping3: ownership cannot change in the update that uses it).
	 */
	if (!e->win_mapped) {
		m1(core, C67D_WINDOW_SET_CONTROL(0), head);
		m1(core, C67D_WINDOW_SET_FORMAT_USAGE_BOUNDS(0), BIT(2));	/* RGB_PACKED4BPP */
		m1(core, C67D_WINDOW_SET_MAX_INPUT_SCALE(0), SCALE_1X | SCALE_1X << 16);
		m1(core, C67D_WINDOW_SET_USAGE_BOUNDS(0), (w + 22) | BIT(16) | BIT(28) | 1 << 20);
		m1(core, C67D_SET_INTERLOCK_FLAGS, 0);
		m1(core, C67D_SET_WINDOW_INTERLOCK_FLAGS, 0);
		m1(core, C67D_UPDATE, 1);
		ret = nv_evo_kick(nv, core);
		dev_info(dev, "modeset: window mapping update: %d, core except 0x%08x err 0x%08x\n", ret,
			 nv_rd32(nv, 0x611020), nv_rd32(nv, 0x611028));
		if (ret)
			return ret;
		e->win_mapped = true;
	}

	/* 3. window 0 */
	win_methods(win, flags, NVDRV_H_FB_CTXDMA, w, h, pitch, true);

	/* 4. core: window -> head, SOR, raster, viewport, usage bounds */
	hsw = m->hsync_end - m->hsync_start;
	vsw = m->vsync_end - m->vsync_start;
	hbe = m->htotal - m->hsync_start;
	vbe = m->vtotal - m->vsync_start;
	leading = vbe - 1 + 1;				/* rasterBlankEnd.y + 1, no overscan */
	trailing = m->vtotal - (leading + h);

	m1(core, C67D_SOR_SET_CONTROL(d->sor), BIT(head) | d->protocol << 8);
	m1(core, C67D_HEAD(head, HEAD_SET_PROCAMP), 0);
	m1(core, C67D_HEAD(head, HEAD_SET_CONTROL_OUTPUT_RESOURCE),
	   1 | (!m->hsync_pos) << 2 | (!m->vsync_pos) << 3 | 4 << 4);	/* CRC raster, BPP_24_444 */
	m1(core, C67D_HEAD(head, HEAD_SET_CONTROL), 0);	/* progressive, no lock */
	m1(core, C67D_HEAD(head, HEAD_SET_DITHER_CONTROL), 0);
	m1(core, C67D_HEAD(head, HEAD_SET_DISPLAY_ID0), d->id);
	m1(core, C67D_HEAD(head, HEAD_SET_PIXEL_CLOCK_FREQUENCY), m->clock_khz * 1000);
	m1(core, C67D_HEAD(head, HEAD_SET_PIXEL_CLOCK_CONFIGURATION), 0);
	m1(core, C67D_HEAD(head, HEAD_SET_PIXEL_CLOCK_FREQUENCY_MAX), m->clock_khz * 1000);
	m1(core, C67D_HEAD(head, HEAD_SET_RASTER_SIZE), m->htotal | m->vtotal << 16);
	m1(core, C67D_HEAD(head, HEAD_SET_RASTER_SYNC_END), (hsw - 1) | (vsw - 1) << 16);
	m1(core, C67D_HEAD(head, HEAD_SET_RASTER_BLANK_END), (hbe - 1) | (vbe - 1) << 16);
	m1(core, C67D_HEAD(head, HEAD_SET_RASTER_BLANK_START), (w + hbe - 1) | (h + vbe - 1) << 16);
	m1(core, C67D_HEAD(head, HEAD_SET_MIN_FRAME_IDLE), leading | trailing << 16);
	m1(core, C67D_HEAD(head, HEAD_SET_TILE_POSITION), 0);
	m1(core, C67D_HEAD(head, HEAD_SET_RASTER_HBLANK_DELAY), 0);
	m1(core, C67D_HEAD(head, HEAD_SET_VIEWPORT_POINT_IN), 0);
	m1(core, C67D_HEAD(head, HEAD_SET_VIEWPORT_SIZE_IN), w | h << 16);
	m1(core, C67D_HEAD(head, HEAD_SET_VIEWPORT_POINT_OUT_ADJUST), 0);
	m1(core, C67D_HEAD(head, HEAD_SET_VIEWPORT_SIZE_OUT), w | h << 16);
	m1(core, C67D_HEAD(head, HEAD_SET_MAX_OUTPUT_SCALE_FACTOR), SCALE_1X | SCALE_1X << 16);
	m1(core, C67D_HEAD(head, HEAD_SET_HEAD_USAGE_BOUNDS), 4 | BIT(4) | 1 << 12);
	m1(core, C67D_SET_INTERLOCK_FLAGS, 0);
	m1(core, C67D_SET_WINDOW_INTERLOCK_FLAGS, BIT(0));
	m1(core, C67D_UPDATE, 1);			/* RELEASE_ELV */

	dev_info(dev, "modeset: %ux%u %u kHz on head %u, SOR%u, %u+%u bytes of methods\n",
		 w, h, m->clock_khz, head, d->sor, win->put, core->put);
	/* window first: it waits at its UPDATE for the interlocked core UPDATE */
	ret = nv_evo_kick_nowait(nv, win);
	if (!ret)
		ret = nv_evo_kick(nv, core);
	if (!ret)
		ret = nv_evo_wait(nv, win);
	if (ret)
		return ret;

	/* 5. is the raster generator running? */
	ret = read_poll_timeout(ioread32, sl, (sl & 0xffff) != 0, 1000, 500000, false,
				core->user + C67D_GET_RG_SCAN_LINE(head));
	get = ioread32(core->user + C67D_GET_RG_SCAN_LINE(head));
	msleep(5);
	dev_info(dev, "modeset: done; head %u scanline %u -> %u (vblank %u)\n", head, sl & 0xffff,
		 ioread32(core->user + C67D_GET_RG_SCAN_LINE(head)) & 0xffff, !!(get & BIT(16)));
	for (i = 0; i < 2; i++) {
		u32 chn = i ? win->chn : core->chn;

		dev_info(dev, "modeset: chn %u except 0x%08x arg 0x%08x err 0x%08x\n", chn,
			 nv_rd32(nv, 0x611020 + chn * 12), nv_rd32(nv, 0x611024 + chn * 12),
			 nv_rd32(nv, 0x611028 + chn * 12));
	}
	e->active = true;
	return 0;
}

int nv_disp_modeset(struct nvdrv_dev *nv, u32 flags)
{
	int ret;

	mutex_lock(&nv->gsp_lock);
	ret = modeset_locked(nv, flags);
	mutex_unlock(&nv->gsp_lock);
	return ret;
}

/*
 * A second test surface in coherent system memory (diagonal stripes, so it is easy to
 * tell from the VRAM bars), reached by the display through a PCI_COHERENT ctxdma.
 */
static int sysfb_prepare(struct nvdrv_dev *nv, u32 w, u32 h)
{
	struct nv_evo *e = &nv->evo;
	struct nv_dma_buf *b = &e->sysfb;
	u32 pitch = ALIGN(w * 4, 256), x, y, *px;
	int ret;

	if (b->cpu)
		return 0;
	b->size = ALIGN((u64)pitch * h, PAGE_SIZE);
	b->cpu = dma_alloc_coherent(&nv->pdev->dev, b->size, &b->dma, GFP_KERNEL);
	if (!b->cpu)
		return -ENOMEM;
	for (y = 0; y < h; y++) {
		px = b->cpu + (size_t)y * pitch;
		for (x = 0; x < w; x++)
			px[x] = ((x + y) / 64) & 1 ? 0xff2060ff : 0xffffa000;	/* blue / orange */
	}
	wmb();
	ret = nv_evo_ctxdma(nv, NVDRV_H_SYSFB_CTXDMA, b->dma, b->size, e->win[0].chn, true);
	if (ret) {
		dma_free_coherent(&nv->pdev->dev, b->size, b->cpu, b->dma);
		b->cpu = NULL;
	}
	return ret;
}

/* Re-send only the window state (no core interlock) and report what the hardware thinks. */
int nv_disp_flip(struct nvdrv_dev *nv, u32 flags)
{
	struct nv_evo *e = &nv->evo;
	struct nv_evo_chan *win = &e->win[0];
	struct nv_disp_dpy *d = NULL;
	u32 i, w, h, ctxdma;
	int ret;

	mutex_lock(&nv->gsp_lock);
	for (i = 0; i < nv->disp.ndpy; i++)
		if (nv->disp.dpy[i].connected && nv->disp.dpy[i].mode.valid)
			d = &nv->disp.dpy[i];
	ret = -ENODEV;
	if (!d || !win->handle || !e->active)
		goto out;
	w = d->mode.hdisplay;
	h = d->mode.vdisplay;
	nv_wr32(nv, 0x611020 + win->chn * 12, 0);	/* clear the latched exception info (best effort) */
	ret = nv_evo_begin(nv, win, 768);
	if (ret)
		goto out;
	ctxdma = NVDRV_H_FB_CTXDMA;
	if (flags & NV_FLIP_SYSMEM) {
		ret = sysfb_prepare(nv, w, h);
		if (ret)
			goto out;
		ctxdma = NVDRV_H_SYSFB_CTXDMA;
	}
	win_methods(win, flags, ctxdma, w, h, ALIGN(w * 4, 256), false);
	ret = nv_evo_kick(nv, win);
	msleep(50);
	dev_info(&nv->pdev->dev, "flip 0x%x: %d, window except 0x%08x arg 0x%08x err 0x%08x, scanline %u\n",
		 flags, ret, nv_rd32(nv, 0x611020 + win->chn * 12), nv_rd32(nv, 0x611024 + win->chn * 12),
		 nv_rd32(nv, 0x611028 + win->chn * 12), ioread32(e->core.user + C67D_GET_RG_SCAN_LINE(0)) & 0xffff);
out:
	mutex_unlock(&nv->gsp_lock);
	return ret;
}

/*
 * BAR1 check: map the scanout framebuffer through BAR1, draw a magenta rectangle with
 * a green frame into the middle of it with ordinary CPU writes, and read a few pixels
 * back through PRAMIN to prove both paths see the same VRAM.
 */
int nv_bar1_test(struct nvdrv_dev *nv)
{
	struct device *dev = &nv->pdev->dev;
	struct nv_evo *e = &nv->evo;
	struct nv_disp_dpy *d = NULL;
	u32 i, w, h, pitch, x, y, x0, y0, bad = 0;
	void __iomem *fb;
	int ret;

	mutex_lock(&nv->gsp_lock);
	for (i = 0; i < nv->disp.ndpy; i++)
		if (nv->disp.dpy[i].connected && nv->disp.dpy[i].mode.valid)
			d = &nv->disp.dpy[i];
	ret = -ENODEV;
	if (!d || !e->fb || !e->active)
		goto out;
	ret = nv_bar1_init(nv);
	if (ret)
		goto out;
	ret = -EIO;
	fb = nv_bar1_map(nv, 0, e->fb, e->fb_size);
	if (!fb)
		goto out;
	w = d->mode.hdisplay;
	h = d->mode.vdisplay;
	pitch = ALIGN(w * 4, 256);
	x0 = w / 4;
	y0 = h / 4;
	for (y = y0; y < y0 + h / 2; y++)
		for (x = x0; x < x0 + w / 2; x++) {
			bool edge = x < x0 + 16 || y < y0 + 16 || x >= x0 + w / 2 - 16 || y >= y0 + h / 2 - 16;

			iowrite32(edge ? 0xff00ff00 : 0xffff00ff, fb + (size_t)y * pitch + x * 4);
		}
	wmb();
	(void)ioread32(fb);
	for (i = 0; i < 64; i++) {
		x = x0 + 16 + (i * 37) % (w / 2 - 32);
		y = y0 + 16 + (i * 53) % (h / 2 - 32);
		bad += nv_vram_rd32(nv, e->fb + (u64)y * pitch + x * 4) != 0xffff00ff;
	}
	dev_info(dev, "bar1: framebuffer mapped at BAR1+0 (%llu KiB), rectangle drawn, PRAMIN read-back %u/64 mismatches\n",
		 e->fb_size >> 10, bad);
	iounmap(fb);
	ret = bad ? -EIO : 0;
out:
	mutex_unlock(&nv->gsp_lock);
	return ret;
}
