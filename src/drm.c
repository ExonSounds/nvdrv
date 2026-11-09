// SPDX-License-Identifier: GPL-2.0
/*
 * DRM/KMS device on top of the running modeset (src/modeset.c).
 *
 * Structure follows drivers/gpu/drm/tiny/cirrus-qemu.c (GPL-2.0, Red Hat): GEM-shmem
 * buffers, a shadow-buffered primary plane whose damaged rectangles are copied into
 * the scanout surface, fbdev emulation through the client library. Vblank comes from
 * the head-timing interrupt (src/intr.c) with timestamps from the raster position, or
 * from a software timer when interrupts are unavailable. Here the scanout surface is nvdrv's VRAM framebuffer, which the CPU reaches
 * through BAR1 (src/bar1.c). All EDID modes that fit the trained DP link are offered;
 * a mode change in an atomic commit becomes an nvdrv modeset (mode_set_nofb).
 */
#include <linux/aperture.h>
#include <linux/iosys-map.h>
#include <linux/seq_file.h>
#include <linux/workqueue.h>

#include <drm/clients/drm_client_setup.h>
#include <drm/drm_atomic.h>
#include <drm/drm_atomic_helper.h>
#include <drm/drm_color_mgmt.h>
#include <drm/drm_damage_helper.h>
#include <drm/drm_drv.h>
#include <drm/drm_edid.h>
#include <drm/drm_eld.h>
#include <drm/drm_fbdev_shmem.h>
#include <drm/drm_file.h>
#include <drm/drm_format_helper.h>
#include <drm/drm_framebuffer.h>
#include <drm/drm_print.h>
#include <drm/drm_fourcc.h>
#include <drm/drm_gem_atomic_helper.h>
#include <drm/drm_gem_framebuffer_helper.h>
#include <drm/drm_gem_shmem_helper.h>
#include <drm/drm_managed.h>
#include <drm/drm_modeset_helper_vtables.h>
#include <drm/drm_probe_helper.h>
#include <drm/drm_vblank.h>
#include <drm/drm_vblank_helper.h>

#include <uapi/drm/nouveau_drm.h>

#include "nvdrv.h"

struct nv_drm {
	struct drm_device dev;
	struct nvdrv_dev *nv;
	struct drm_plane primary, cursor;
	bool audio_on;			/* ELD handed to the HD-Audio function */
	struct drm_crtc crtc;
	struct drm_encoder encoder;
	struct drm_connector connector;
	void __iomem *scanout;		/* BAR1 mapping of the VRAM framebuffer */
	unsigned int pitch;
	struct drm_display_mode mode;	/* the one mode the head runs */
	const struct drm_edid *edid;
	/*
	 * GSP-RM does not see CPU-drawn frames as load and parks the GPU in P8 / PCIe Gen1
	 * (~2.9 GB/s through BAR1, < 180 full frames/s). Frames being drawn re-arm a short
	 * boost to P0 / Gen3 (~11 GB/s); a static screen lets the GPU idle again.
	 */
	struct work_struct boost_work;
	unsigned long boost_next;	/* jiffies: earliest next boost request */
	unsigned long win_start;	/* jiffies: start of the 1 s traffic window */
	u64 win_bytes;			/* bytes copied in the window */
	atomic64_t updates, bytes;	/* totals, debugfs drm_stats */
};

#define NV_DRM_BOOST_SECONDS	3
#define NV_DRM_BOOST_BYTES	(64ULL << 20)	/* per second; a blinking cursor stays far below */

static void nv_drm_boost_work(struct work_struct *w)
{
	struct nv_drm *d = container_of(w, struct nv_drm, boost_work);

	nv_rm_perf_boost(d->nv, NV_DRM_BOOST_SECONDS);
}

static struct nv_drm *to_nv_drm(struct drm_device *dev)
{
	return container_of(dev, struct nv_drm, dev);
}

static void drm_mode_to_nv(const struct drm_display_mode *dm, struct nv_mode *m)
{
	memset(m, 0, sizeof(*m));
	m->valid = true;
	m->clock_khz = dm->clock;
	m->hdisplay = dm->hdisplay;
	m->hsync_start = dm->hsync_start;
	m->hsync_end = dm->hsync_end;
	m->htotal = dm->htotal;
	m->vdisplay = dm->vdisplay;
	m->vsync_start = dm->vsync_start;
	m->vsync_end = dm->vsync_end;
	m->vtotal = dm->vtotal;
	m->hsync_pos = !(dm->flags & DRM_MODE_FLAG_NHSYNC);
	m->vsync_pos = !(dm->flags & DRM_MODE_FLAG_NVSYNC);
}

static bool nv_drm_mode_ok(struct nv_drm *d, const struct drm_display_mode *dm)
{
	struct nv_mode m;

	if (dm->flags & (DRM_MODE_FLAG_INTERLACE | DRM_MODE_FLAG_DBLSCAN | DRM_MODE_FLAG_3D_MASK))
		return false;
	drm_mode_to_nv(dm, &m);
	return nv_disp_mode_fits(d->nv, &m);
}

/* ---- plane ------------------------------------------------------------- */

static const u32 nv_formats[] = { DRM_FORMAT_XRGB8888, DRM_FORMAT_ARGB8888 };	/* alpha ignored */
/*
 * GPU-rendered framebuffers may also be block linear (scanned out directly; copied ones must
 * be linear). Turing+ layout as nouveau wndwc57e_modifiers / NVK: page kind 0x06, kind
 * generation 2, sector layout 1, no compression, block height 32..1 GOBs.
 */
#define NV_MOD_BL(h)	DRM_FORMAT_MOD_NVIDIA_BLOCK_LINEAR_2D(0, 1, 2, 0x06, h)
static const u64 nv_modifiers[] = {
	NV_MOD_BL(5), NV_MOD_BL(4), NV_MOD_BL(3), NV_MOD_BL(2), NV_MOD_BL(1), NV_MOD_BL(0),
	DRM_FORMAT_MOD_LINEAR, DRM_FORMAT_MOD_INVALID
};

/* log2 block height of a supported block-linear modifier, -1 for linear, -2 otherwise */
static int nv_mod_block_h(u64 mod)
{
	int h;

	if (mod == DRM_FORMAT_MOD_LINEAR)
		return -1;
	for (h = 0; h <= 5; h++)
		if (mod == NV_MOD_BL(h))
			return h;
	return -2;
}

static int nv_plane_atomic_check(struct drm_plane *plane, struct drm_atomic_commit *state)
{
	struct drm_plane_state *ps = drm_atomic_get_new_plane_state(state, plane);
	struct drm_crtc_state *cs = NULL;

	u64 vram;
	int ret;

	if (ps->crtc)
		cs = drm_atomic_get_new_crtc_state(state, ps->crtc);
	ret = drm_atomic_helper_check_plane_state(ps, cs, DRM_PLANE_NO_SCALING,
						  DRM_PLANE_NO_SCALING, false, false);
	if (ret || !ps->fb)
		return ret;
	if (!nv_uapi_bo_vram(ps->fb->obj[0], &vram))	/* copied: linear only */
		return ps->fb->modifier == DRM_FORMAT_MOD_LINEAR ? 0 : -EINVAL;
	/* GPU-rendered VRAM framebuffer: scanned out directly, so it must fit the window */
	if (nv_mod_block_h(ps->fb->modifier) < -1 || (ps->fb->pitches[0] & 63) ||
	    ((vram + ps->fb->offsets[0]) & 255) || (ps->src.x1 >> 16) || (ps->src.y1 >> 16))
		return -EINVAL;
	return 0;
}

/* Async (tearing) flips: only zero-copy VRAM framebuffers, which flip the window directly. */
static int nv_plane_atomic_async_check(struct drm_plane *plane, struct drm_atomic_commit *state, bool flip)
{
	struct drm_plane_state *ps = drm_atomic_get_new_plane_state(state, plane);
	u64 vram;

	if (!ps || !ps->fb || !nv_uapi_bo_vram(ps->fb->obj[0], &vram))
		return -EINVAL;
	return 0;
}

/* The shadow-plane vmap only for framebuffers that are copied (shmem objects). */
static int nv_plane_begin_fb_access(struct drm_plane *plane, struct drm_plane_state *ps)
{
	u64 vram;

	if (ps->fb && nv_uapi_bo_vram(ps->fb->obj[0], &vram))
		return 0;
	return drm_gem_begin_shadow_fb_access(plane, ps);
}

static void nv_plane_end_fb_access(struct drm_plane *plane, struct drm_plane_state *ps)
{
	u64 vram;

	if (ps->fb && nv_uapi_bo_vram(ps->fb->obj[0], &vram))
		return;
	drm_gem_end_shadow_fb_access(plane, ps);
}

static void nv_plane_atomic_update(struct drm_plane *plane, struct drm_atomic_commit *state)
{
	struct nv_drm *d = to_nv_drm(plane->dev);
	struct drm_plane_state *ps = drm_atomic_get_new_plane_state(state, plane);
	struct drm_plane_state *old = drm_atomic_get_old_plane_state(state, plane);
	struct drm_shadow_plane_state *sps = to_drm_shadow_plane_state(ps);
	struct iosys_map vaddr = IOSYS_MAP_INIT_VADDR_IOMEM(d->scanout);
	struct drm_framebuffer *fb = ps->fb;
	struct drm_atomic_helper_damage_iter iter;
	struct drm_rect damage;
	int idx;

	u64 vram;

	if (!fb || !drm_dev_enter(&d->dev, &idx))
		return;
	if (nv_uapi_bo_vram(fb->obj[0], &vram)) {	/* zero copy: flip the window to it */
		struct drm_crtc_state *cs = ps->crtc ? drm_atomic_get_new_crtc_state(state, ps->crtc) : NULL;

		nv_disp_scanout(d->nv, vram + fb->offsets[0], fb->pitches[0], nv_mod_block_h(fb->modifier),
				cs && cs->async_flip);
		atomic64_inc(&d->updates);
		drm_dev_exit(idx);
		return;
	}
	if (d->nv->evo.scanout_cur) {	/* back from a direct framebuffer: copy all of this one */
		nv_disp_scanout(d->nv, 0, 0, -1, false);
		ps->ignore_damage_clips = true;
	}
	drm_atomic_helper_damage_iter_init(&iter, old, ps);
	drm_atomic_for_each_plane_damage(&iter, &damage) {
		struct drm_rect dst_clip = ps->dst;
		struct iosys_map dst;

		/* plane src == dst (no scaling); clip damage to the visible area */
		drm_rect_translate(&damage, ps->dst.x1 - (ps->src.x1 >> 16), ps->dst.y1 - (ps->src.y1 >> 16));
		if (!drm_rect_intersect(&dst_clip, &damage))
			continue;
		dst = IOSYS_MAP_INIT_OFFSET(&vaddr, drm_fb_clip_offset(d->pitch, fb->format, &dst_clip));
		drm_rect_translate(&dst_clip, (ps->src.x1 >> 16) - ps->dst.x1, (ps->src.y1 >> 16) - ps->dst.y1);
		drm_fb_memcpy(&dst, &d->pitch, sps->data, fb, &dst_clip);
		d->win_bytes += (u64)drm_rect_width(&dst_clip) * drm_rect_height(&dst_clip) * 4;
		atomic64_add((u64)drm_rect_width(&dst_clip) * drm_rect_height(&dst_clip) * 4, &d->bytes);
	}
	atomic64_inc(&d->updates);
	/* sustained copy traffic: keep the GPU (and its PCIe link) out of the idle state */
	if (time_after(jiffies, d->win_start + HZ)) {
		d->win_start = jiffies;
		d->win_bytes = 0;
	} else if (d->win_bytes > NV_DRM_BOOST_BYTES && time_after_eq(jiffies, d->boost_next)) {
		d->boost_next = jiffies + HZ;
		schedule_work(&d->boost_work);
	}
	drm_dev_exit(idx);
}

static void nv_plane_atomic_disable(struct drm_plane *plane, struct drm_atomic_commit *state)
{
	struct nv_drm *d = to_nv_drm(plane->dev);
	int idx;

	if (!drm_dev_enter(&d->dev, &idx))
		return;
	memset_io(d->scanout, 0, (size_t)d->pitch * d->mode.vdisplay);	/* black */
	drm_dev_exit(idx);
}

/* ---- cursor plane --------------------------------------------------------- */

static const u32 nv_cursor_formats[] = { DRM_FORMAT_ARGB8888 };
static const u64 nv_cursor_modifiers[] = { DRM_FORMAT_MOD_LINEAR, DRM_FORMAT_MOD_INVALID };

static int nv_cursor_atomic_check(struct drm_plane *plane, struct drm_atomic_commit *state)
{
	struct drm_plane_state *ps = drm_atomic_get_new_plane_state(state, plane);
	struct drm_crtc_state *cs = NULL;
	struct drm_framebuffer *fb = ps->fb;
	u64 vram;
	int ret;

	if (ps->crtc)
		cs = drm_atomic_get_new_crtc_state(state, ps->crtc);
	ret = drm_atomic_helper_check_plane_state(ps, cs, DRM_PLANE_NO_SCALING, DRM_PLANE_NO_SCALING,
						  true, true);
	if (ret || !fb || !ps->visible)
		return ret;
	/* the hardware shows whole square images of 32..256 pixels */
	if (fb->width != fb->height || fb->width < 32 || fb->width > 256 || !is_power_of_2(fb->width) ||
	    ps->src_x || ps->src_y || (ps->src_w >> 16) != fb->width || (ps->src_h >> 16) != fb->height)
		return -EINVAL;
	if (fb->modifier != DRM_FORMAT_MOD_LINEAR)
		return -EINVAL;
	if (nv_uapi_bo_vram(fb->obj[0], &vram) &&
	    (fb->pitches[0] != fb->width * 4 || ((vram + fb->offsets[0]) & 255)))
		return -EINVAL;
	return 0;
}

static void nv_cursor_atomic_update(struct drm_plane *plane, struct drm_atomic_commit *state)
{
	struct nv_drm *d = to_nv_drm(plane->dev);
	struct drm_plane_state *ps = drm_atomic_get_new_plane_state(state, plane);
	struct drm_plane_state *old = drm_atomic_get_old_plane_state(state, plane);
	struct drm_framebuffer *fb = ps->fb;
	struct drm_rect damage;
	u64 vram;
	int idx, ret = 0;

	if (!drm_dev_enter(&d->dev, &idx))
		return;
	if (!fb || !ps->visible) {
		nv_disp_cursor_off(d->nv);
		goto out;
	}
	/*
	 * New image: another framebuffer, the cursor reappearing, or damage on the same one
	 * (dirtyfb). Not the damage helper's default "whole plane" for a plain move: that
	 * would turn every move into a core channel update.
	 */
	if (fb != old->fb || !old->visible ||
	    (ps->fb_damage_clips && drm_atomic_helper_damage_merged(old, ps, &damage))) {
		if (nv_uapi_bo_vram(fb->obj[0], &vram))
			ret = nv_disp_cursor_image(d->nv, vram + fb->offsets[0], NULL, 0, fb->width);
		else
			ret = nv_disp_cursor_image(d->nv, 0, to_drm_shadow_plane_state(ps)->data[0].vaddr +
						   fb->offsets[0], fb->pitches[0], fb->width);
		if (ret)
			drm_err_once(&d->dev, "cursor image: %d\n", ret);
	}
	nv_disp_cursor_move(d->nv, ps->crtc_x, ps->crtc_y);
out:
	drm_dev_exit(idx);
}

static void nv_cursor_atomic_disable(struct drm_plane *plane, struct drm_atomic_commit *state)
{
	struct nv_drm *d = to_nv_drm(plane->dev);
	int idx;

	if (!drm_dev_enter(&d->dev, &idx))
		return;
	nv_disp_cursor_off(d->nv);
	drm_dev_exit(idx);
}

static const struct drm_plane_helper_funcs nv_cursor_helper_funcs = {
	.begin_fb_access = nv_plane_begin_fb_access,
	.end_fb_access = nv_plane_end_fb_access,
	.atomic_check = nv_cursor_atomic_check,
	.atomic_update = nv_cursor_atomic_update,
	.atomic_disable = nv_cursor_atomic_disable,
};

static const struct drm_plane_helper_funcs nv_plane_helper_funcs = {
	.begin_fb_access = nv_plane_begin_fb_access,
	.end_fb_access = nv_plane_end_fb_access,
	.atomic_check = nv_plane_atomic_check,
	.atomic_async_check = nv_plane_atomic_async_check,
	.atomic_update = nv_plane_atomic_update,
	.atomic_disable = nv_plane_atomic_disable,
};

static const struct drm_plane_funcs nv_plane_funcs = {
	.update_plane = drm_atomic_helper_update_plane,
	.disable_plane = drm_atomic_helper_disable_plane,
	.destroy = drm_plane_cleanup,
	DRM_GEM_SHADOW_PLANE_FUNCS,
};

/* ---- crtc / encoder / connector ---------------------------------------- */

static int nv_crtc_atomic_check(struct drm_crtc *crtc, struct drm_atomic_commit *state)
{
	struct nv_drm *d = to_nv_drm(crtc->dev);
	struct drm_crtc_state *cs = drm_atomic_get_new_crtc_state(state, crtc);

	if (!cs->enable)
		return 0;
	if (cs->color_mgmt_changed && cs->gamma_lut) {
		u32 n = drm_color_lut_size(cs->gamma_lut);

		if (n < 2 || n > 4096 || drm_color_lut_check(cs->gamma_lut, 0))
			return -EINVAL;
	}
	if (!nv_drm_mode_ok(d, &cs->mode))
		return -EINVAL;
	return drm_atomic_helper_check_crtc_primary_plane(cs);
}

/* Mode changes: reprogram the head before the planes are updated for the new size. */
static void nv_crtc_mode_set_nofb(struct drm_crtc *crtc)
{
	struct nv_drm *d = to_nv_drm(crtc->dev);
	const struct drm_display_mode *dm = &crtc->state->mode;
	struct nv_mode m;
	int idx;

	if (drm_mode_equal(dm, &d->mode) || !drm_dev_enter(&d->dev, &idx))
		return;
	if (READ_ONCE(d->nv->evo.vrr_active))	/* one-shot mode off for the modeset; flush re-enables */
		nv_disp_vrr(d->nv, false, 0);
	drm_mode_to_nv(dm, &m);
	if (nv_disp_set_mode(d->nv, &m)) {
		drm_err(&d->dev, "modeset to " DRM_MODE_FMT " failed\n", DRM_MODE_ARG(dm));
	} else {
		drm_mode_copy(&d->mode, dm);
		d->pitch = ALIGN(dm->hdisplay * 4, 256);
		drm_info(&d->dev, "mode " DRM_MODE_FMT ", %d Hz\n", DRM_MODE_ARG(dm), drm_mode_vrefresh(dm));
	}
	drm_dev_exit(idx);
}

/* Monitor audio follows the display: ELD from the connector's EDID while the head runs. */
static void nv_drm_audio(struct nv_drm *d, bool on)
{
	const u8 *eld = d->connector.eld;
	u32 size = on ? drm_eld_size(eld) : 0;

	if (on && (size <= DRM_ELD_HEADER_BLOCK_SIZE || size > 96))	/* no audio blocks / too long */
		size = 0;
	/*
	 * The HD-Audio side only reads the ELD on a clean presence change: switching it on
	 * from an unknown state (first commit after load) left eld_valid 0. Clear it first.
	 */
	u32 max_freq = 0, i, n = size ? drm_eld_sad_count(eld) : 0;

	/* highest sample rate in the SADs -> MAX_FREQ_SUPPORTED code (1 = 32 kHz ... 7 = 192 kHz) */
	for (i = 0; i < n; i++) {
		u8 rates = drm_eld_sad(eld)[i * 3 + 1] & 0x7f;

		if (rates)
			max_freq = max_t(u32, max_freq, fls(rates));
	}
	if (size && !d->audio_on)
		nv_disp_audio(d->nv, NULL, 0, 0);
	nv_disp_audio(d->nv, size ? eld : NULL, size, max_freq);
	d->audio_on = size;
}

/* Display power: CRTC active again (DPMS on) brings the head and the sink back. */
static void nv_crtc_atomic_enable(struct drm_crtc *crtc, struct drm_atomic_commit *state)
{
	struct nv_drm *d = to_nv_drm(crtc->dev);
	struct drm_crtc_state *cs = drm_atomic_get_new_crtc_state(state, crtc);
	struct nv_mode m;
	int idx;

	if (!d->nv->evo.active && drm_dev_enter(crtc->dev, &idx)) {
		drm_mode_to_nv(&cs->mode, &m);
		if (nv_disp_power(d->nv, true, &m))
			drm_err(crtc->dev, "display power on failed\n");
		drm_dev_exit(idx);
	}
	drm_crtc_vblank_on(crtc);
	if (drm_dev_enter(crtc->dev, &idx)) {
		nv_drm_audio(d, true);
		drm_dev_exit(idx);
	}
}

/* CRTC inactive (DPMS off, not a mode change): head off, monitor to standby. */
static void nv_crtc_atomic_disable(struct drm_crtc *crtc, struct drm_atomic_commit *state)
{
	struct nv_drm *d = to_nv_drm(crtc->dev);
	struct drm_crtc_state *cs = drm_atomic_get_new_crtc_state(state, crtc);
	int idx;

	drm_crtc_vblank_atomic_disable(crtc, state);
	if (!cs->active && drm_dev_enter(crtc->dev, &idx)) {
		nv_drm_audio(d, false);
		nv_disp_power(d->nv, false, NULL);
		drm_dev_exit(idx);
	}
}

/* ---- hardware vblank ---------------------------------------------------- */

#define PDISP_RG_DPCA(h)		(0x00616330 + (h) * 0x800)	/* vline, latches hline */
#define PDISP_RG_DPCB(h)		(0x00616334 + (h) * 0x800)
#define CORE_ARM_HEAD(h, r)		(0x0068a000 + (h) * 0x400 + (r))	/* nouveau gv100_head_state */

static int nv_crtc_enable_vblank(struct drm_crtc *crtc)
{
	struct nv_drm *d = to_nv_drm(crtc->dev);

	return nv_intr_vblank_enable(d->nv, d->nv->evo.head, true) ? 0 : -ENODEV;
}

static void nv_crtc_disable_vblank(struct drm_crtc *crtc)
{
	struct nv_drm *d = to_nv_drm(crtc->dev);

	nv_intr_vblank_enable(d->nv, d->nv->evo.head, false);
}

/* Called from the interrupt handler on LAST_DATA (start of vblank) of @head. */
void nv_drm_vblank_irq(struct nvdrv_dev *nv, u32 head)
{
	struct nv_drm *d = READ_ONCE(nv->drm);

	if (d && head == nv->evo.head)
		drm_crtc_handle_vblank(&d->crtc);
}

/*
 * Raster position for vblank timestamps (nouveau_display_scanoutpos): vtotal and the
 * blank start/end lines from the core channel's ARMed head state, the line counter
 * from the raster generator; lines from blank start on count as negative.
 */
static bool nv_crtc_get_scanout_position(struct drm_crtc *crtc, bool in_vblank_irq, int *vpos,
					 int *hpos, ktime_t *stime, ktime_t *etime,
					 const struct drm_display_mode *mode)
{
	struct nv_drm *d = to_nv_drm(crtc->dev);
	struct nvdrv_dev *nv = d->nv;
	u32 h = nv->evo.head, total, blank_end, blank_start;
	int vline, hline;

	total = nv_rd32(nv, CORE_ARM_HEAD(h, 0x064)) >> 16;
	blank_end = nv_rd32(nv, CORE_ARM_HEAD(h, 0x06c)) >> 16;
	blank_start = nv_rd32(nv, CORE_ARM_HEAD(h, 0x070)) >> 16;
	if (!total || total == 0xffff)
		return false;
	if (stime)
		*stime = ktime_get();
	vline = nv_rd32(nv, PDISP_RG_DPCA(h)) & 0xffff;
	hline = nv_rd32(nv, PDISP_RG_DPCB(h)) & 0xffff;
	if (etime)
		*etime = ktime_get();
	if (vline >= blank_start)	/* nouveau calc() */
		vline -= total;
	if (blank_end < blank_start)
		vline -= blank_end + 1;
	*vpos = vline;
	*hpos = hline;
	return true;
}

/* ---- colour: GAMMA_LUT -> head OLUT, CTM -> OCSC0 (before the OLUT, as DRM's order) ---- */

/* DRM S31.32 sign-magnitude to OCSC S5.16 in 20:0 (nvCscCoefConvertS514: clamp, round to 4). */
static u32 ctm_coef(u64 v)
{
	s64 x = (s64)((v & ~BIT_ULL(63)) >> 16);

	if (v & BIT_ULL(63))
		x = -x;
	x = clamp_t(s64, x + 2, -0x100000, 0xfffff);
	return (u32)x & 0x1ffffc;
}

static void nv_crtc_color(struct nv_drm *d, struct drm_crtc_state *cs)
{
	u16 *lut = NULL;
	u32 ocsc[12], i, c;
	int ret;

	if (cs->gamma_lut) {
		const struct drm_color_lut *in = cs->gamma_lut->data;
		u32 n = drm_color_lut_size(cs->gamma_lut);

		lut = kmalloc_array(NV_OLUT_ENTRIES * 3, sizeof(*lut), GFP_KERNEL);
		if (!lut)
			return;
		for (i = 0; i < NV_OLUT_ENTRIES; i++) {	/* resample n entries onto 1024 (+1) */
			u32 pos = min(i, 1023u) * (n - 1) * 256 / 1023, a = pos >> 8, f = pos & 255;
			u32 b = min(a + 1, n - 1);

			for (c = 0; c < 3; c++) {
				const u16 *pa = &in[a].red, *pb = &in[b].red;

				lut[i * 3 + c] = (pa[c] * (256 - f) + pb[c] * f) >> 8;
			}
		}
	}
	if (cs->ctm) {
		const struct drm_color_ctm *m = cs->ctm->data;

		for (i = 0; i < 12; i++)	/* 3x4: 3x3 of the CTM, no offsets */
			ocsc[i] = (i % 4 == 3) ? 0 : ctm_coef(m->matrix[(i / 4) * 3 + i % 4]);
	}
	ret = nv_disp_color(d->nv, lut, cs->ctm ? ocsc : NULL);
	if (ret)
		drm_err_once(&d->dev, "colour update: %d (needs compose=1)\n", ret);
	kfree(lut);
}

static void nv_crtc_atomic_flush(struct drm_crtc *crtc, struct drm_atomic_commit *state)
{
	struct drm_crtc_state *cs = drm_atomic_get_new_crtc_state(state, crtc);
	int idx;

	if (cs->color_mgmt_changed && cs->active && drm_dev_enter(crtc->dev, &idx)) {
		nv_crtc_color(to_nv_drm(crtc->dev), cs);
		drm_dev_exit(idx);
	}
	if (cs->active && drm_dev_enter(crtc->dev, &idx)) {	/* VRR_ENABLED -> head one-shot mode */
		struct nv_drm *d = to_nv_drm(crtc->dev);
		const struct drm_display_info *di = &d->connector.display_info;
		bool want = cs->vrr_enabled && di->monitor_range.min_vfreq &&
			    d->connector.state && d->connector.state->crtc == crtc;

		if (want != READ_ONCE(d->nv->evo.vrr_active))
			nv_disp_vrr(d->nv, want, want ? 1000000 / di->monitor_range.min_vfreq : 0);
		drm_dev_exit(idx);
	}
	if (cs->async_flip && cs->event) {	/* tearing flip: done as soon as the window has it */
		spin_lock_irq(&crtc->dev->event_lock);
		drm_crtc_send_vblank_event(crtc, cs->event);
		cs->event = NULL;
		spin_unlock_irq(&crtc->dev->event_lock);
	}
	drm_crtc_vblank_atomic_flush(crtc, state);
}

static const struct drm_crtc_helper_funcs nv_crtc_helper_funcs = {
	.atomic_check = nv_crtc_atomic_check,
	.mode_set_nofb = nv_crtc_mode_set_nofb,
	.atomic_flush = nv_crtc_atomic_flush,
	.atomic_enable = nv_crtc_atomic_enable,
	.atomic_disable = nv_crtc_atomic_disable,
	.get_scanout_position = nv_crtc_get_scanout_position,
};

static const struct drm_crtc_funcs nv_crtc_funcs = {
	.reset = drm_atomic_helper_crtc_reset,
	.destroy = drm_crtc_cleanup,
	.set_config = drm_atomic_helper_set_config,
	.page_flip = drm_atomic_helper_page_flip,
	.atomic_duplicate_state = drm_atomic_helper_crtc_duplicate_state,
	.atomic_destroy_state = drm_atomic_helper_crtc_destroy_state,
	DRM_CRTC_VBLANK_TIMER_FUNCS,
};

static const struct drm_crtc_funcs nv_crtc_funcs_hw = {
	.reset = drm_atomic_helper_crtc_reset,
	.destroy = drm_crtc_cleanup,
	.set_config = drm_atomic_helper_set_config,
	.page_flip = drm_atomic_helper_page_flip,
	.atomic_duplicate_state = drm_atomic_helper_crtc_duplicate_state,
	.atomic_destroy_state = drm_atomic_helper_crtc_destroy_state,
	.enable_vblank = nv_crtc_enable_vblank,
	.disable_vblank = nv_crtc_disable_vblank,
	.get_vblank_timestamp = drm_crtc_vblank_helper_get_vblank_timestamp,
};

static const struct drm_encoder_funcs nv_encoder_funcs = {
	.destroy = drm_encoder_cleanup,
};

/* Hotplug from GSP-RM (disp.c): userspace and the fbdev client re-probe. */
void nv_drm_hotplug(struct nvdrv_dev *nv)
{
	struct nv_drm *d = READ_ONCE(nv->drm);

	if (d)
		drm_kms_helper_hotplug_event(&d->dev);
}

/* The DP display nvdrv drives: connected, and its EDID if it changed. */
static enum drm_connector_status nv_connector_detect(struct drm_connector *connector, bool force)
{
	struct nv_drm *d = to_nv_drm(connector->dev);
	struct nvdrv_dev *nv = d->nv;
	struct nv_disp_dpy *dpy = NULL;
	bool connected;
	u32 i;

	mutex_lock(&nv->gsp_lock);
	for (i = 0; i < nv->disp.ndpy; i++)
		if (nv->disp.dpy[i].connected && (nv->disp.dpy[i].protocol == 0x8 || nv->disp.dpy[i].protocol == 0x9))
			dpy = &nv->disp.dpy[i];
	connected = dpy && dpy->edid_size;
	if (connected && (!d->edid || drm_edid_raw(d->edid)->extensions + 1 != dpy->edid_size / EDID_LENGTH ||
			  memcmp(drm_edid_raw(d->edid), dpy->edid, dpy->edid_size))) {
		drm_edid_free(d->edid);
		d->edid = drm_edid_alloc(dpy->edid, dpy->edid_size);
	}
	mutex_unlock(&nv->gsp_lock);
	return connected ? connector_status_connected : connector_status_disconnected;
}

static int nv_connector_get_modes(struct drm_connector *connector)
{
	struct nv_drm *d = to_nv_drm(connector->dev);
	struct drm_display_mode *m;
	int n;

	drm_edid_connector_update(connector, d->edid);	/* EDID property + display info */
	/* Adaptive-Sync: the sink ignores MSA timing and the EDID has a refresh range */
	drm_connector_set_vrr_capable_property(connector, nv_disp_vrr_capable(d->nv) &&
					       connector->display_info.monitor_range.min_vfreq);
	n = drm_edid_connector_add_modes(connector);	/* base, CTA-861 and DisplayID timings */
	if (n)
		return n;
	m = drm_mode_duplicate(connector->dev, &d->mode);
	if (!m)
		return 0;
	m->type = DRM_MODE_TYPE_DRIVER | DRM_MODE_TYPE_PREFERRED;
	drm_mode_probed_add(connector, m);
	return 1;
}

static enum drm_mode_status nv_connector_mode_valid(struct drm_connector *connector,
						    const struct drm_display_mode *mode)
{
	return nv_drm_mode_ok(to_nv_drm(connector->dev), mode) ? MODE_OK : MODE_CLOCK_HIGH;
}

static const struct drm_connector_helper_funcs nv_connector_helper_funcs = {
	.get_modes = nv_connector_get_modes,
	.mode_valid = nv_connector_mode_valid,
};

static const struct drm_connector_funcs nv_connector_funcs = {
	.detect = nv_connector_detect,
	.fill_modes = drm_helper_probe_single_connector_modes,
	.destroy = drm_connector_cleanup,
	.reset = drm_atomic_helper_connector_reset,
	.atomic_duplicate_state = drm_atomic_helper_connector_duplicate_state,
	.atomic_destroy_state = drm_atomic_helper_connector_destroy_state,
};

/*
 * drm_atomic_helper_commit_tail, except that a tearing flip does not wait for the next vblank
 * before releasing the old framebuffer (it is off screen once the window took the flip);
 * otherwise async flips would be capped at two per frame.
 */
static void nv_commit_tail(struct drm_atomic_commit *state)
{
	struct drm_device *dev = state->dev;
	struct drm_crtc_state *cs;
	struct drm_crtc *crtc;
	bool async = false;
	int i;

	for_each_new_crtc_in_state(state, crtc, cs, i)
		async |= cs->async_flip;
	drm_atomic_helper_commit_modeset_disables(dev, state);
	drm_atomic_helper_commit_planes(dev, state, 0);
	drm_atomic_helper_commit_modeset_enables(dev, state);
	drm_atomic_helper_fake_vblank(state);
	drm_atomic_helper_commit_hw_done(state);
	if (async)
		drm_atomic_helper_wait_for_flip_done(dev, state);
	else
		drm_atomic_helper_wait_for_vblanks(dev, state);
	drm_atomic_helper_cleanup_planes(dev, state);
}

static const struct drm_mode_config_helper_funcs nv_mode_config_helpers = {
	.atomic_commit_tail = nv_commit_tail,
};

static const struct drm_mode_config_funcs nv_mode_config_funcs = {
	.fb_create = drm_gem_fb_create_with_dirty,
	.atomic_check = drm_atomic_helper_check,
	.atomic_commit = drm_atomic_helper_commit,
};

DEFINE_DRM_GEM_FOPS(nv_drm_fops);

static const struct drm_driver nv_drm_driver = {
	.driver_features = DRIVER_MODESET | DRIVER_GEM | DRIVER_ATOMIC,
	.name = DRV_NAME,
	.desc = "nvdrv GSP-RM display",
	.major = 0,
	.minor = 1,
	.fops = &nv_drm_fops,
	DRM_GEM_SHMEM_DRIVER_OPS,
	DRM_FBDEV_SHMEM_DRIVER_OPS,
};

/*
 * nvk=1: the same KMS device plus nouveau's uAPI subset (src/uapi.c) on a render node.
 * Mesa only binds NVK to a DRM driver named "nouveau" with version >= 1.3.1; from 1.4.1
 * on NVK exposes a transfer queue on copy-engine channels, from 1.4.2 (big/small PTE
 * transitions race-free, src/uapi.c write_chunk) it compresses images on Ampere.
 */
static const struct file_operations nv_nvk_fops = {
	.owner = THIS_MODULE,
	.open = drm_open,
	.release = drm_release,
	.unlocked_ioctl = nv_uapi_ioctl,
	.compat_ioctl = drm_compat_ioctl,
	.poll = drm_poll,
	.read = drm_read,
	.llseek = noop_llseek,
	.mmap = drm_gem_mmap,
	.fop_flags = FOP_UNSIGNED_OFFSET,
};

static struct drm_driver nv_nvk_driver = {	/* .ioctls set at registration (another unit) */
	.driver_features = DRIVER_MODESET | DRIVER_GEM | DRIVER_ATOMIC | DRIVER_RENDER |
			   DRIVER_SYNCOBJ | DRIVER_SYNCOBJ_TIMELINE,
	.name = "nouveau",
	.desc = "nvdrv GSP-RM (nouveau uAPI subset)",
	.major = 1,
	.minor = 4,
	.patchlevel = 2,	/* 1.4.1: transfer queue, page sizes; 1.4.2: LPTE/SPTE safe -> compression */
	.fops = &nv_nvk_fops,
	.open = nv_uapi_open,
	.postclose = nv_uapi_postclose,
	.num_ioctls = DRM_NOUVEAU_GEM_INFO + 1,	/* nv_uapi_ioctls[] is indexed by ioctl number */
	DRM_GEM_SHMEM_DRIVER_OPS,
	DRM_FBDEV_SHMEM_DRIVER_OPS,
};

struct nvdrv_dev *nv_drm_nv(struct drm_device *dev)
{
	return to_nv_drm(dev)->nv;
}

static void nv_mode_to_drm(const struct nv_mode *m, struct drm_display_mode *dm)
{
	memset(dm, 0, sizeof(*dm));
	dm->clock = m->clock_khz;
	dm->hdisplay = m->hdisplay;
	dm->hsync_start = m->hsync_start;
	dm->hsync_end = m->hsync_end;
	dm->htotal = m->htotal;
	dm->vdisplay = m->vdisplay;
	dm->vsync_start = m->vsync_start;
	dm->vsync_end = m->vsync_end;
	dm->vtotal = m->vtotal;
	dm->flags = (m->hsync_pos ? DRM_MODE_FLAG_PHSYNC : DRM_MODE_FLAG_NHSYNC) |
		    (m->vsync_pos ? DRM_MODE_FLAG_PVSYNC : DRM_MODE_FLAG_NVSYNC);
	dm->type = DRM_MODE_TYPE_DRIVER | DRM_MODE_TYPE_PREFERRED;
	drm_mode_set_name(dm);
}

/*
 * Register the DRM device for the display nvdrv has already lit (disp_modeset).
 * Caller must not hold gsp_lock (drm_client_setup may draw through atomic commits).
 */
static void nv_drm_put_nv(struct drm_device *dev, void *nv)
{
	nv_put(nv);
}

int nv_drm_register(struct nvdrv_dev *nv)
{
	struct pci_dev *pdev = nv->pdev;
	struct nv_evo *e = &nv->evo;
	struct nv_disp_dpy *dpy = NULL;
	struct drm_device *dev;
	struct nv_drm *d;
	u32 i;
	int ret;

	if (nv->drm)
		return -EBUSY;
	mutex_lock(&nv->gsp_lock);
	for (i = 0; i < nv->disp.ndpy; i++)
		if (nv->disp.dpy[i].connected && nv->disp.dpy[i].mode.valid)
			dpy = &nv->disp.dpy[i];
	ret = -ENODEV;
	if (!dpy || !e->active || !e->fb) {
		mutex_unlock(&nv->gsp_lock);
		return ret;
	}
	mutex_unlock(&nv->gsp_lock);

	/* the firmware framebuffer (simpledrm) is dead since the first FLR; take over */
	ret = aperture_remove_conflicting_pci_devices(pdev, nv_drm_driver.name);
	if (ret)
		return ret;

	nv_nvk_driver.ioctls = nv_uapi_ioctls_ptr;
	d = devm_drm_dev_alloc(&pdev->dev, nv_nvk ? &nv_nvk_driver : &nv_drm_driver, struct nv_drm, dev);
	if (IS_ERR(d))
		return PTR_ERR(d);
	dev = &d->dev;
	d->nv = nv;
	kref_get(&nv->ref);	/* dropped when the DRM device is released (last file closed) */
	ret = drmm_add_action_or_reset(&d->dev, nv_drm_put_nv, nv);
	if (ret)
		return ret;
	INIT_WORK(&d->boost_work, nv_drm_boost_work);
	d->boost_next = jiffies;
	d->win_start = jiffies;
	d->pitch = ALIGN(dpy->mode.hdisplay * 4, 256);
	nv_mode_to_drm(&dpy->mode, &d->mode);
	d->edid = drm_edid_alloc(dpy->edid, dpy->edid_size);

	d->scanout = e->fb_cpu;		/* owned by evo.c, mapped once */
	if (!d->scanout)
		return -EIO;

	ret = drmm_mode_config_init(dev);
	if (ret)
		goto err;
	dev->mode_config.min_width = 32;	/* framebuffers: cursors are 32..256 */
	dev->mode_config.max_width = 4096;
	dev->mode_config.min_height = 32;
	dev->mode_config.max_height = 4096;
	dev->mode_config.preferred_depth = 24;
	dev->mode_config.funcs = &nv_mode_config_funcs;
	dev->mode_config.helper_private = &nv_mode_config_helpers;
	dev->mode_config.async_page_flip = true;	/* tearing flips of VRAM framebuffers */

	ret = drm_universal_plane_init(dev, &d->primary, 0, &nv_plane_funcs, nv_formats,
				       ARRAY_SIZE(nv_formats), nv_modifiers, DRM_PLANE_TYPE_PRIMARY, NULL);
	if (ret)
		goto err;
	drm_plane_helper_add(&d->primary, &nv_plane_helper_funcs);
	drm_plane_enable_fb_damage_clips(&d->primary);

	ret = drm_universal_plane_init(dev, &d->cursor, 0, &nv_plane_funcs, nv_cursor_formats,
				       ARRAY_SIZE(nv_cursor_formats), nv_cursor_modifiers,
				       DRM_PLANE_TYPE_CURSOR, NULL);
	if (ret)
		goto err;
	drm_plane_helper_add(&d->cursor, &nv_cursor_helper_funcs);
	drm_plane_enable_fb_damage_clips(&d->cursor);
	dev->mode_config.cursor_width = 256;
	dev->mode_config.cursor_height = 256;

	ret = drm_crtc_init_with_planes(dev, &d->crtc, &d->primary, &d->cursor,
					nv_intr_has_vblank(nv) ? &nv_crtc_funcs_hw : &nv_crtc_funcs, NULL);
	if (ret)
		goto err;
	drm_crtc_helper_add(&d->crtc, &nv_crtc_helper_funcs);
	if (nv->evo.composed) {	/* gamma/CTM live in the composition pipeline */
		drm_mode_crtc_set_gamma_size(&d->crtc, 1024);
		drm_crtc_enable_color_mgmt(&d->crtc, 0, true, 1024);
	}

	ret = drm_encoder_init(dev, &d->encoder, &nv_encoder_funcs, DRM_MODE_ENCODER_TMDS, NULL);
	if (ret)
		goto err;
	d->encoder.possible_crtcs = drm_crtc_mask(&d->crtc);

	ret = drm_connector_init(dev, &d->connector, &nv_connector_funcs, DRM_MODE_CONNECTOR_DisplayPort);
	if (ret)
		goto err;
	drm_connector_helper_add(&d->connector, &nv_connector_helper_funcs);
	ret = drm_connector_attach_encoder(&d->connector, &d->encoder);
	if (ret)
		goto err;
	drm_connector_attach_vrr_capable_property(&d->connector);
	ret = drm_vblank_init(dev, 1);
	if (ret)
		goto err;

	drm_mode_config_reset(dev);
	ret = drm_dev_register(dev, 0);
	if (ret)
		goto err;
	nv->drm = d;
	drm_client_setup(dev, NULL);
	/* make the fbdev client commit its mode now instead of on the next VT switch */
	drm_kms_helper_hotplug_event(dev);
	drm_info(dev, "%s on %s, scanout VRAM 0x%llx via BAR1, pitch %u, %s vblank\n", d->mode.name,
		 pci_name(pdev), e->fb, d->pitch, nv_intr_has_vblank(nv) ? "hardware" : "timer");
	return 0;
err:
	d->scanout = NULL;
	return ret;
}

/* debugfs drm_stats: plane updates and bytes copied into the scanout surface so far */
void nv_drm_stats_show(struct nvdrv_dev *nv, struct seq_file *m)
{
	struct nv_drm *d = nv->drm;

	if (!d) {
		seq_puts(m, "no DRM device\n");
		return;
	}
	seq_printf(m, "updates %lld bytes %lld mode %s@%d\n", atomic64_read(&d->updates),
		   atomic64_read(&d->bytes), d->mode.name, drm_mode_vrefresh(&d->mode));
}

/* .remove, before the GPU is reset: stop all access to BAR1. */
void nv_drm_unregister(struct nvdrv_dev *nv)
{
	struct nv_drm *d = nv->drm;

	if (!d)
		return;
	nv->unplugged = true;	/* file / GEM teardown from here on: software state only */
	drm_dev_unplug(&d->dev);
	drm_atomic_helper_shutdown(&d->dev);
	WRITE_ONCE(nv->drm, NULL);	/* the vblank interrupt stops using it */
	if (nv->intr.irq > 0)
		synchronize_irq(nv->intr.irq);
	cancel_work_sync(&d->boost_work);
	d->scanout = NULL;
	drm_edid_free(d->edid);
	d->edid = NULL;
	nv->drm = NULL;
}
