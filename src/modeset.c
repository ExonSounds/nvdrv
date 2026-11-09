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
#include <linux/vmalloc.h>

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
#define C67E_SET_KEY_ALPHA			0x02f8	/* then RED_CR, GREEN_Y, BLUE_CB */
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
#define NVDRV_H_POOL_CTXDMA			0xd000f102
#define NVDRV_H_CURS_CTXDMA			0xd000f103	/* cursor slots, core channel */
#define NVDRV_H_POOL_CORE_CTXDMA		0xd000f104	/* VRAM pool, core channel */
#define NVDRV_H_CURS				0xd000c67a
#define NVDRV_H_LUT_CTXDMA			0xd000f105	/* LUTs, window channel */
#define NVDRV_H_LUT_CORE_CTXDMA			0xd000f106	/* LUTs, core channel */
#define OLUT_OFFSET				0x4000		/* in the LUT buffer, after the ILUT */

#define C67D_HEAD_SET_OCSC0CONTROL		0x2240
#define C67D_HEAD_SET_OCSC0COEFFICIENT_C00	0x2244
#define C67D_HEAD_SET_OLUT_CONTROL		0x2280
#define   OLUT_INTERPOLATE			BIT(0)
#define   OLUT_DIRECT10				(2 << 2)
#define C67D_HEAD_SET_OLUT_FP_NORM_SCALE	0x2284
#define C67D_HEAD_SET_CONTEXT_DMA_OLUT		0x2288
#define C67D_HEAD_SET_OFFSET_OLUT		0x228c

#define C67E_SET_OFFSET_ILUT			0x0448
#define   ILUT_DIRECT10				(2 << 2)
#define   ILUT_ENTRIES				(4 + 1024 + 1)	/* VSS header + entries + one to interpolate */

/*
 * compose=1: the window goes through the normal composition pipeline (opaque, with an
 * identity input LUT), which the hardware cursor needs; 0: composition bypass as before.
 */
static bool compose = true;
module_param(compose, bool, 0444);
MODULE_PARM_DESC(compose, "Window composition with an identity input LUT (needed for the hardware cursor)");

/* cursor: core channel methods (clc67d.h) and the cursor immediate PIO channel (clc67a.h) */
#define NVC67A_CURSOR_IMM_CHANNEL_PIO		0xc67a
#define C67D_HEAD_SET_CONTEXT_DMA_CURSOR	0x2088
#define C67D_HEAD_SET_OFFSET_CURSOR		0x2090
#define C67D_HEAD_SET_PRESENT_CONTROL_CURSOR	0x2098
#define C67D_HEAD_SET_CONTROL_CURSOR		0x209c
#define   CURSOR_ENABLE				BIT(31)
#define   CURSOR_FORMAT_A8R8G8B8		0xcf
#define C67D_HEAD_SET_CONTROL_CURSOR_COMPOSITION 0x20a0
#define   CURSOR_BLEND_PREMULT			(0xff | 2 << 8 /* K1 */ | 7 << 12 /* NEG_K1_TIMES_SRC */)
#define C67A_USER(head)				(0x006d8000 + (head) * 0x1000)
#define C67A_FREE				0x0008
#define C67A_UPDATE				0x0200
#define C67A_SET_INTERLOCK_FLAGS		0x0204
#define   C67A_INTERLOCK_WITH_CORE		BIT(16)
#define C67A_SET_CURSOR_HOT_SPOT_POINT_OUT	0x0208

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
int dp_watermark(const struct nv_mode *m, u32 lanes, u32 link_bw_270m, bool ef, bool increased,
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
/* nouveau wndwc57e_ilut_load: unsigned 0.16 to FP16 */
static u16 u0_16_to_fp16(u16 fixed)
{
	int exp = 0, man = 0;

	if (fixed) {
		while (--exp && !(fixed & 0x8000))
			fixed <<= 1;
		man = ((fixed << 1) & 0xffc0) >> 6;
		exp += 15;
	}
	return exp << 10 | man;
}

/*
 * Identity input LUT in VRAM (nouveau wndwc57e_ilut with ilut_identity): 32-byte VSS header,
 * 1024 DIRECT10 entries of FP16 R, G, B (8 bytes each), the last one repeated.
 * Caller holds gsp_lock; BAR1 is set up (the framebuffer is mapped first).
 */
static int lut_init(struct nvdrv_dev *nv)
{
	struct nv_evo *e = &nv->evo;
	u32 i;
	int ret;

	if (e->lut_ready)
		return 0;
	if (!e->lut_vram) {
		ret = nv_vram_alloc(nv, NV_LUT_SIZE, SZ_64K, &e->lut_vram);
		if (ret)
			return ret;
	}
	if (!e->lut_cpu) {
		e->lut_cpu = nv_bar1_map(nv, NV_BAR1_LUT_VA, e->lut_vram, NV_LUT_SIZE);
		if (!e->lut_cpu)
			return -EIO;
	}
	memset_io(e->lut_cpu, 0, 0x20);
	for (i = 0; i <= 1024; i++) {
		u16 v = u0_16_to_fp16(min(i, 1023u) * 0xffff / 1023);
		void __iomem *p = e->lut_cpu + 0x20 + i * 8;

		writew(v, p);
		writew(v, p + 2);
		writew(v, p + 4);
		writew(0, p + 6);
	}
	/* identity output LUT (nouveau headc57d_olut_load): unsigned 16-bit entries */
	memset_io(e->lut_cpu + OLUT_OFFSET, 0, 0x20);
	for (i = 0; i <= 1024; i++) {
		u16 v = min(i, 1023u) * 0xffff / 1023;
		void __iomem *p = e->lut_cpu + OLUT_OFFSET + 0x20 + i * 8;

		writew(v, p);
		writew(v, p + 2);
		writew(v, p + 4);
		writew(0, p + 6);
	}
	wmb();
	(void)ioread32(e->lut_cpu);
	ret = nv_evo_ctxdma(nv, NVDRV_H_LUT_CTXDMA, e->lut_vram, NV_LUT_SIZE, e->win[0].chn, false);
	if (!ret)
		ret = nv_evo_ctxdma(nv, NVDRV_H_LUT_CORE_CTXDMA, e->lut_vram, NV_LUT_SIZE, e->core.chn, false);
	if (ret)
		return ret;
	e->lut_ready = true;
	return 0;
}

/* Head output LUT (identity) on the core channel: nouveau headc57d_olut_set. */
/*
 * Head output LUT (identity) on the core channel, nvkms EvoSetOutputLutC5: OCSC0 matrix
 * first, the OLUT, then OCSC0 enabled ("only enable OCSC0 after enabling the OLUT": the
 * specification requires the two together).
 */
static void olut_methods(struct nv_evo_chan *core, u32 head, bool on)
{
	u32 i;

	if (!on) {	/* composition bypass: no OLUT, OCSC0 off (nvSetupOutputLUT5) */
		m1(core, C67D_HEAD(head, C67D_HEAD_SET_OCSC0CONTROL), 0);
		m1(core, C67D_HEAD(head, C67D_HEAD_SET_OLUT_CONTROL), 0);
		m1(core, C67D_HEAD(head, C67D_HEAD_SET_CONTEXT_DMA_OLUT), 0);
		return;
	}
	for (i = 0; i < 12; i++)	/* identity, S5.16 in 20:0 (nvCscCoefConvertS514) */
		m1(core, C67D_HEAD(head, C67D_HEAD_SET_OCSC0COEFFICIENT_C00) + 4 * i, (i % 5 == 0) ? 0x10000 : 0);
	m1(core, C67D_HEAD(head, C67D_HEAD_SET_OLUT_CONTROL), OLUT_INTERPOLATE | OLUT_DIRECT10 | ILUT_ENTRIES << 8);
	m1(core, C67D_HEAD(head, C67D_HEAD_SET_CONTEXT_DMA_OLUT), NVDRV_H_LUT_CORE_CTXDMA);
	m1(core, C67D_HEAD(head, C67D_HEAD_SET_OFFSET_OLUT), OLUT_OFFSET >> 8);
	m1(core, C67D_HEAD(head, C67D_HEAD_SET_OLUT_FP_NORM_SCALE), 0xffffffff);
	m1(core, C67D_HEAD(head, C67D_HEAD_SET_OCSC0CONTROL), 1);	/* ENABLE */
}

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
	/* not TMO_CONTROL = 0: an invalid argument (window error 0x2) */
	}
	if (!(flags & NV_FLIP_COMPOSE)) {
		m1(win, C67E_SET_COMPOSITION_CONTROL, BIT(16));	/* BYPASS */
	} else {
		if (flags & NV_FLIP_ILUT) {	/* identity input LUT: blending needs one (err 0x2d) */
			m1(win, C67E_SET_ILUT_CONTROL, ILUT_ENTRIES << 8 | ILUT_DIRECT10);
			m1(win, C67E_SET_CONTEXT_DMA_ILUT, NVDRV_H_LUT_CTXDMA);
			m1(win, C67E_SET_OFFSET_ILUT, 0);
		}
		/* nouveau wndwc37e_blend_set for an opaque bottom window: depth 255, K1 = 1.0 */
		m1(win, C67E_SET_COMPOSITION_CONTROL, 255 << 4);	/* no colour key */
		m1(win, C67E_SET_COMPOSITION_CONSTANT_ALPHA, 0xff);
		m1(win, C67E_SET_COMPOSITION_FACTOR_SELECT, 0x4422);	/* src K1, dst NEG_K1 */
		for (i = 0; i < 4; i++)				/* colour key ranges: all */
			m1(win, C67E_SET_KEY_ALPHA + 4 * i, 0xffff0000);
	}
	m1(win, C67E_SET_INTERLOCK_FLAGS, interlock_core);	/* INTERLOCK_WITH_CORE */
	m1(win, C67E_SET_WINDOW_INTERLOCK_FLAGS, 0);
	if (flags & NV_FLIP_MCLK_WAR)
		m1(win, C67E_SW_SET_MCLK_SWITCH, 0);	/* ENABLE_FALSE (bug 2193096 WAR) */
	m1(win, C67E_UPDATE, 1);			/* RELEASE_ELV */
	if (flags & NV_FLIP_MCLK_WAR)
		m1(win, C67E_SW_SET_MCLK_SWITCH, 1);

}

static int modeset_win_locked(struct nvdrv_dev *nv, u32 flags)
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
	e->fb_pitch = pitch;
	e->scanout_cur = 0;	/* the modeset points window 0 at the own framebuffer */

	/*
	 * 1. one fixed framebuffer big enough for any mode we drive, mapped through BAR1
	 * once; the test pattern is (re)drawn for the current mode through that mapping.
	 */
	if (fb_size > NV_FB_MAX)
		return -E2BIG;
	if (!e->fb) {
		ret = nv_vram_alloc(nv, NV_FB_MAX, SZ_1M, &e->fb);
		if (ret)
			return ret;
		e->fb_size = NV_FB_MAX;
	}
	if (!e->fb_cpu) {
		ret = nv_bar1_init(nv);
		if (ret)
			return ret;
		e->fb_cpu = nv_bar1_map(nv, 0, e->fb, e->fb_size);
		if (!e->fb_cpu)
			return -EIO;
	}
	if (e->fb_w != w || e->fb_h != h || e->pattern_dirty) {
		u64 px, n = (u64)(pitch / 4) * h;

		pat = (struct pattern){ w, h, pitch / 4 };
		for (px = 0; px < n; px++)
			iowrite32(pattern_px(&pat, px), e->fb_cpu + px * 4);
		wmb();
		(void)ioread32(e->fb_cpu);
		e->fb_w = w;
		e->fb_h = h;
		e->pattern_dirty = false;
		dev_info(dev, "modeset: framebuffer %ux%u pitch %u at VRAM 0x%llx (BAR1)\n", w, h, pitch, e->fb);
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
	ret = nv_evo_begin(nv, core, 640);
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

	/* 3. window 0: normal composition with an identity input LUT unless compose=0 */
	if (compose && !(flags & NV_FLIP_COMPOSE) && !lut_init(nv))
		flags |= NV_FLIP_COMPOSE | NV_FLIP_ILUT | NV_FLIP_CSC;
	win_methods(win, flags, NVDRV_H_FB_CTXDMA, w, h, pitch, true);

	/* 4. core: window -> head, SOR, raster, viewport, usage bounds */
	hsw = m->hsync_end - m->hsync_start;
	vsw = m->vsync_end - m->vsync_start;
	hbe = m->htotal - m->hsync_start;
	vbe = m->vtotal - m->vsync_start;
	leading = vbe - 1 + 1;				/* rasterBlankEnd.y + 1, no overscan */
	trailing = m->vtotal - (leading + h);

	m1(core, C67D_SOR_SET_CONTROL(d->sor), BIT(head) | d->protocol << 8);
	e->sor = d->sor;
	e->head = head;
	m1(core, C67D_HEAD(head, HEAD_SET_PROCAMP), 0);
	m1(core, C67D_HEAD(head, HEAD_SET_CONTROL_OUTPUT_RESOURCE),
	   1 | (!m->hsync_pos) << 2 | (!m->vsync_pos) << 3 | 4 << 4 |	/* CRC raster, BPP_24_444 */
	   0x3f << 26);		/* EXT_PACKET_WIN_NONE (0 = window 0 was an invalid argument, err 4) */
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
	/* composition needs the head's OLUT + OCSC0 in the same (window-interlocked) update */
	olut_methods(core, head, flags & NV_FLIP_COMPOSE);
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
	e->composed = flags & NV_FLIP_COMPOSE;
	return 0;
}

int nv_disp_modeset(struct nvdrv_dev *nv, u32 flags)
{
	int ret;

	mutex_lock(&nv->gsp_lock);
	ret = nv_disp_modeset_locked(nv, flags);
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
/*
 * Point window 0 at another surface of the current mode's size: vram != 0 is a
 * linear XRGB surface in the VRAM pool (a GPU-rendered DRM framebuffer, shown without
 * a copy), vram == 0 nvdrv's own framebuffer. Takes effect at the next vblank.
 */
int nv_disp_scanout(struct nvdrv_dev *nv, u64 vram, u32 pitch)
{
	struct nv_evo *e = &nv->evo;
	struct nv_evo_chan *win = &e->win[0];
	u32 ctxdma = NVDRV_H_FB_CTXDMA;
	u64 offset = 0;
	int ret = 0;

	if (vram && (!nv->mem.ready || vram < nv->mem.base || vram >= nv->mem.base + nv->mem.size ||
		     (vram & 255) || (pitch & 63)))
		return -EINVAL;
	if (vram && !READ_ONCE(e->pool_ctxdma)) {	/* once: instmem + RM, needs gsp_lock */
		mutex_lock(&nv->gsp_lock);
		mutex_lock(&e->win_lock);
		if (!e->pool_ctxdma && win->handle)
			ret = nv_evo_ctxdma(nv, NVDRV_H_POOL_CTXDMA, nv->mem.base, nv->mem.size, win->chn, false);
		if (!ret && win->handle)
			e->pool_ctxdma = true;
		mutex_unlock(&e->win_lock);
		mutex_unlock(&nv->gsp_lock);
		if (ret)
			return ret;
	}
	mutex_lock(&e->win_lock);
	if (!win->handle || !e->active || (vram && !e->pool_ctxdma)) {
		ret = -ENODEV;
		goto out;
	}
	if (vram == e->scanout_cur)
		goto out;
	if (vram) {
		ctxdma = NVDRV_H_POOL_CTXDMA;
		offset = vram - nv->mem.base;
	} else {
		pitch = e->fb_pitch;
	}
	ret = nv_evo_begin(nv, win, 6 * 8);	/* bytes: six single-word methods */
	if (ret)
		goto out;
	m1(win, C67E_SET_CONTEXT_DMA_ISO(0), ctxdma);
	m1(win, C67E_SET_OFFSET(0), offset >> 8);
	m1(win, C67E_SET_PLANAR_STORAGE(0), pitch >> 6);
	/* standalone window update: the modeset left it interlocked with the core channel */
	m1(win, C67E_SET_INTERLOCK_FLAGS, 0);
	m1(win, C67E_SET_WINDOW_INTERLOCK_FLAGS, 0);
	m1(win, C67E_UPDATE, 1);
	ret = nv_evo_kick(nv, win);
	if (!ret)
		e->scanout_cur = vram;
out:
	mutex_unlock(&e->win_lock);
	return ret;
}

/* ---- hardware cursor ----------------------------------------------------- */

/*
 * Cursor of the active head (nouveau headc37d_curs_set, cursc37a): image, size and blending
 * through the core channel, position through the cursor PIO channel. Images that are not in
 * the VRAM pool are copied into one of two slots in nvdrv's carve-out. Caller holds gsp_lock.
 */
int nv_disp_cursor_init(struct nvdrv_dev *nv)
{
	struct nv_evo *e = &nv->evo;
	int ret;

	lockdep_assert_held(&nv->gsp_lock);
	if (e->curs_ready)
		return 0;
	if (!e->active || !e->core.handle)
		return -ENODEV;
	ret = nv_vram_alloc(nv, 2 * NV_CURS_SLOT, SZ_64K, &e->curs_vram);
	if (ret)
		return ret;
	e->curs_cpu = nv_bar1_map(nv, NV_BAR1_CURS_VA, e->curs_vram, 2 * NV_CURS_SLOT);
	if (!e->curs_cpu)
		return -EIO;
	ret = nv_evo_ctxdma(nv, NVDRV_H_CURS_CTXDMA, e->curs_vram, 2 * NV_CURS_SLOT, e->core.chn, false);
	if (ret)
		return ret;
	ret = nv_evo_pio_alloc(nv, NVC67A_CURSOR_IMM_CHANNEL_PIO, NVDRV_H_CURS, e->head);
	if (ret)
		return ret;
	e->curs_user = nv->bar0 + C67A_USER(e->head);
	e->curs_ready = true;
	return 0;
}

static int cursor_core(struct nvdrv_dev *nv, u32 ctl, u32 ctxdma, u64 offset)
{
	struct nv_evo *e = &nv->evo;
	struct nv_evo_chan *core = &e->core;
	u32 head = e->head;
	int ret;

	if (ctl == e->curs_ctl && ctxdma == e->curs_ctxdma && offset == e->curs_offset)
		return 0;
	/*
	 * Needs the window in normal composition (compose=1): with composition bypass GSP-RM
	 * rejects enabling the cursor (INVALID_STATE, error code 0x43). The interlock flags are
	 * sticky; the modeset leaves the core interlocked with window 0.
	 */
	mutex_lock(&e->win_lock);
	ret = nv_evo_begin(nv, core, 8 * 8);
	if (ret)
		goto out;
	m1(core, C67D_HEAD(head, C67D_HEAD_SET_PRESENT_CONTROL_CURSOR), 0);	/* MONO */
	m1(core, C67D_HEAD(head, C67D_HEAD_SET_CONTROL_CURSOR), ctl);
	m1(core, C67D_HEAD(head, C67D_HEAD_SET_CONTROL_CURSOR_COMPOSITION), CURSOR_BLEND_PREMULT);
	m1(core, C67D_HEAD(head, C67D_HEAD_SET_CONTEXT_DMA_CURSOR), ctxdma);
	m1(core, C67D_HEAD(head, C67D_HEAD_SET_OFFSET_CURSOR), offset >> 8);
	m1(core, C67D_SET_INTERLOCK_FLAGS, 0);
	m1(core, C67D_SET_WINDOW_INTERLOCK_FLAGS, 0);
	m1(core, C67D_UPDATE, 1);
	ret = nv_evo_kick(nv, core);
	dev_dbg(&nv->pdev->dev, "cursor: ctl 0x%08x ctxdma 0x%08x offset 0x%llx: %d, core status 0x%08x except 0x%08x err 0x%08x\n",
		ctl, ctxdma, offset, ret, nv_rd32(nv, 0x610630), nv_rd32(nv, 0x611020), nv_rd32(nv, 0x611028));
	if (!ret) {
		e->curs_ctl = ctl;
		e->curs_ctxdma = ctxdma;
		e->curs_offset = offset;
	}
out:
	mutex_unlock(&e->win_lock);
	return ret;
}

/*
 * Show a square ARGB8888 (premultiplied) cursor of @size pixels (32/64/128/256): either
 * the pool object at @vram (packed, 256-byte aligned) or @src (rows @src_pitch apart)
 * copied into a slot.
 */
int nv_disp_cursor_image(struct nvdrv_dev *nv, u64 vram, const void *src, u32 src_pitch, u32 size)
{
	struct nv_evo *e = &nv->evo;
	u32 ctl, ctxdma, code = ilog2(size) - 5;
	u64 offset;
	int ret;

	if (size < 32 || size > 256 || !is_power_of_2(size))
		return -EINVAL;
	mutex_lock(&nv->gsp_lock);
	ret = nv_disp_cursor_init(nv);
	if (ret)
		goto out;
	if (vram) {
		if (!nv->mem.ready || vram < nv->mem.base || vram + size * size * 4 > nv->mem.base + nv->mem.size ||
		    (vram & 255)) {
			ret = -EINVAL;
			goto out;
		}
		if (!e->curs_pool_ctxdma) {
			ret = nv_evo_ctxdma(nv, NVDRV_H_POOL_CORE_CTXDMA, nv->mem.base, nv->mem.size,
					    e->core.chn, false);
			if (ret)
				goto out;
			e->curs_pool_ctxdma = true;
		}
		ctxdma = NVDRV_H_POOL_CORE_CTXDMA;
		offset = vram - nv->mem.base;
	} else {
		u32 y;

		e->curs_slot ^= 1;	/* the other slot: the one on screen stays intact */
		for (y = 0; y < size; y++)
			memcpy_toio(e->curs_cpu + e->curs_slot * NV_CURS_SLOT + y * size * 4,
				    src + y * src_pitch, size * 4);
		wmb();
		(void)ioread32(e->curs_cpu);
		ctxdma = NVDRV_H_CURS_CTXDMA;
		offset = e->curs_slot * NV_CURS_SLOT;
	}
	ctl = CURSOR_ENABLE | CURSOR_FORMAT_A8R8G8B8 | code << 8;
	ret = cursor_core(nv, ctl, ctxdma, offset);
out:
	mutex_unlock(&nv->gsp_lock);
	return ret;
}

int nv_disp_cursor_off(struct nvdrv_dev *nv)
{
	int ret = 0;

	mutex_lock(&nv->gsp_lock);
	if (nv->evo.curs_ready && nv->evo.active)	/* disabled still needs a valid format */
		ret = cursor_core(nv, CURSOR_FORMAT_A8R8G8B8, 0, 0);
	mutex_unlock(&nv->gsp_lock);
	return ret;
}

/* Top-left corner of the cursor on the head (may be negative). */
void nv_disp_cursor_move(struct nvdrv_dev *nv, int x, int y)
{
	struct nv_evo *e = &nv->evo;
	u32 free;

	mutex_lock(&e->curs_lock);
	if (e->curs_ready && !readl_poll_timeout(e->curs_user + C67A_FREE, free, (free & 0x3f) >= 2,
						 1, 20000)) {
		iowrite32((u16)x | (u32)(u16)y << 16, e->curs_user + C67A_SET_CURSOR_HOT_SPOT_POINT_OUT);
		iowrite32(0, e->curs_user + C67A_UPDATE);
	}
	mutex_unlock(&e->curs_lock);
}

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
	mutex_lock(&e->win_lock);
	w = d->mode.hdisplay;
	h = d->mode.vdisplay;
	nv_wr32(nv, 0x611020 + win->chn * 12, 0);	/* clear the latched exception info (best effort) */
	ret = nv_evo_begin(nv, win, 768);
	if (ret)
		goto out_win;
	ctxdma = NVDRV_H_FB_CTXDMA;
	if (flags & NV_FLIP_ILUT) {
		ret = lut_init(nv);
		if (ret)
			goto out_win;
	}
	if (flags & NV_FLIP_SYSMEM) {
		ret = sysfb_prepare(nv, w, h);
		if (ret)
			goto out_win;
		ctxdma = NVDRV_H_SYSFB_CTXDMA;
	}
	if ((flags & NV_FLIP_ILUT) || e->composed != !!(flags & NV_FLIP_COMPOSE)) {
		/* bypass <-> composition: head OLUT/OCSC0 and window together, interlocked */
		struct nv_evo_chan *core = &e->core;

		ret = nv_evo_begin(nv, core, 24 * 8);
		if (ret)
			goto out_win;
		olut_methods(core, e->head, flags & NV_FLIP_COMPOSE);
		m1(core, C67D_SET_INTERLOCK_FLAGS, 0);
		m1(core, C67D_SET_WINDOW_INTERLOCK_FLAGS, BIT(0));
		m1(core, C67D_UPDATE, 1);
		win_methods(win, flags, ctxdma, w, h, ALIGN(w * 4, 256), true);
		ret = nv_evo_kick_nowait(nv, win);
		if (!ret)
			ret = nv_evo_kick(nv, core);
		if (!ret)
			ret = nv_evo_wait(nv, win);
		if (!ret)
			e->composed = flags & NV_FLIP_COMPOSE;
		msleep(50);
		dev_info(&nv->pdev->dev, "flip: with core (OLUT %s): %d, core except 0x%08x err 0x%08x\n",
			 flags & NV_FLIP_COMPOSE ? "on" : "off", ret, nv_rd32(nv, 0x611020), nv_rd32(nv, 0x611028));
	} else {
		win_methods(win, flags, ctxdma, w, h, ALIGN(w * 4, 256), false);
		ret = nv_evo_kick(nv, win);
		msleep(50);
	}
	dev_info(&nv->pdev->dev, "flip 0x%x: %d, window except 0x%08x arg 0x%08x err 0x%08x, scanline %u\n",
		 flags, ret, nv_rd32(nv, 0x611020 + win->chn * 12), nv_rd32(nv, 0x611024 + win->chn * 12),
		 nv_rd32(nv, 0x611028 + win->chn * 12), ioread32(e->core.user + C67D_GET_RG_SCAN_LINE(0)) & 0xffff);
out_win:
	mutex_unlock(&e->win_lock);
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

/*
 * Detach the stream before the link can be retrained (what DPlib does on a long HPD
 * pulse): window without surface, SOR without owner, display ID cleared, one
 * interlocked update. The raster generator of the head stops with it.
 */
static int detach_win_locked(struct nvdrv_dev *nv)
{
	struct nv_evo *e = &nv->evo;
	struct nv_evo_chan *core = &e->core, *win = &e->win[0];
	u32 i;
	int ret;

	if (!e->active)
		return 0;
	ret = nv_evo_begin(nv, core, 128);
	if (!ret)
		ret = nv_evo_begin(nv, win, 128);
	if (ret)
		return ret;
	for (i = 0; i < 6; i++)
		m1(win, C67E_SET_CONTEXT_DMA_ISO(i), 0);
	m1(win, C67E_SET_INTERLOCK_FLAGS, 1);
	m1(win, C67E_SET_WINDOW_INTERLOCK_FLAGS, 0);
	m1(win, C67E_UPDATE, 1);
	m1(core, C67D_SOR_SET_CONTROL(e->sor), 0);
	m1(core, C67D_HEAD(e->head, HEAD_SET_DISPLAY_ID0), 0);
	m1(core, C67D_SET_INTERLOCK_FLAGS, 0);
	m1(core, C67D_SET_WINDOW_INTERLOCK_FLAGS, BIT(0));
	m1(core, C67D_UPDATE, 1);
	ret = nv_evo_kick_nowait(nv, win);
	if (!ret)
		ret = nv_evo_kick(nv, core);
	if (!ret)
		ret = nv_evo_wait(nv, win);
	dev_info(&nv->pdev->dev, "modeset: detached head %u from SOR%u: %d, except core 0x%08x win 0x%08x err 0x%08x\n",
		 e->head, e->sor, ret, nv_rd32(nv, 0x611020), nv_rd32(nv, 0x611020 + win->chn * 12),
		 nv_rd32(nv, 0x611028 + win->chn * 12));
	e->active = false;
	return ret;
}

/* Time full-frame copies into the scanout surface through BAR1 (what each DRM plane update does). */
int nv_bar1_bench(struct nvdrv_dev *nv)
{
	struct nv_evo *e = &nv->evo;
	size_t frame = (size_t)ALIGN(e->fb_w * 4, 256) * e->fb_h;
	void *src;
	u64 t0, ns;
	int i, n = 30;

	if (!e->fb_cpu || !frame)
		return -ENODEV;
	src = vmalloc(frame);
	if (!src)
		return -ENOMEM;
	memset(src, 0x40, frame);
	t0 = ktime_get_ns();
	for (i = 0; i < n; i++)
		memcpy_toio(e->fb_cpu, src, frame);
	wmb();
	(void)ioread32(e->fb_cpu);
	ns = ktime_get_ns() - t0;
	vfree(src);
	dev_info(&nv->pdev->dev, "bar1 bench: %d x %zu KiB in %llu ms = %llu MB/s, %llu frames/s\n", n, frame >> 10,
		 ns / 1000000, (u64)n * frame * 1000 / ns, (u64)n * 1000000000ULL / ns);
	e->pattern_dirty = true;	/* redraw the test pattern on the next modeset */
	return 0;
}

/*
 * Lock order: gsp_lock, then evo.win_lock. Everything that drives window channel 0 holds
 * win_lock; nv_disp_scanout (DRM page flips) holds only win_lock, so long GSP-RM work
 * (channel setup, context promotion) under gsp_lock no longer delays flips.
 */
int nv_disp_modeset_locked(struct nvdrv_dev *nv, u32 flags)
{
	int ret;

	mutex_lock(&nv->evo.win_lock);
	ret = modeset_win_locked(nv, flags);
	mutex_unlock(&nv->evo.win_lock);
	return ret;
}

int nv_disp_detach_locked(struct nvdrv_dev *nv)
{
	int ret;

	mutex_lock(&nv->evo.win_lock);
	ret = detach_win_locked(nv);
	mutex_unlock(&nv->evo.win_lock);
	return ret;
}
