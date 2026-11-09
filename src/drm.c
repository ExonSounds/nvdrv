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
#include <drm/drm_damage_helper.h>
#include <drm/drm_drv.h>
#include <drm/drm_edid.h>
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
	struct drm_plane primary;
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
static const u64 nv_modifiers[] = { DRM_FORMAT_MOD_LINEAR, DRM_FORMAT_MOD_INVALID };

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
	if (ret || !ps->fb || !nv_uapi_bo_vram(ps->fb->obj[0], &vram))
		return ret;
	/* GPU-rendered VRAM framebuffer: scanned out directly, so it must fit the window */
	if (ps->fb->modifier != DRM_FORMAT_MOD_LINEAR || (ps->fb->pitches[0] & 63) ||
	    ((vram + ps->fb->offsets[0]) & 255) || (ps->src.x1 >> 16) || (ps->src.y1 >> 16))
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
		nv_disp_scanout(d->nv, vram + fb->offsets[0], fb->pitches[0]);
		atomic64_inc(&d->updates);
		drm_dev_exit(idx);
		return;
	}
	if (d->nv->evo.scanout_cur) {	/* back from a direct framebuffer: copy all of this one */
		nv_disp_scanout(d->nv, 0, 0);
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

static const struct drm_plane_helper_funcs nv_plane_helper_funcs = {
	.begin_fb_access = nv_plane_begin_fb_access,
	.end_fb_access = nv_plane_end_fb_access,
	.atomic_check = nv_plane_atomic_check,
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

static void nv_crtc_atomic_enable(struct drm_crtc *crtc, struct drm_atomic_commit *state)
{
	drm_crtc_vblank_on(crtc);	/* the head itself stays configured by nvdrv */
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

static const struct drm_crtc_helper_funcs nv_crtc_helper_funcs = {
	.atomic_check = nv_crtc_atomic_check,
	.mode_set_nofb = nv_crtc_mode_set_nofb,
	.atomic_flush = drm_crtc_vblank_atomic_flush,
	.atomic_enable = nv_crtc_atomic_enable,
	.atomic_disable = drm_crtc_vblank_atomic_disable,
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

static int nv_connector_get_modes(struct drm_connector *connector)
{
	struct nv_drm *d = to_nv_drm(connector->dev);
	struct drm_display_mode *m;
	int n;

	drm_edid_connector_update(connector, d->edid);	/* EDID property + display info */
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
	.fill_modes = drm_helper_probe_single_connector_modes,
	.destroy = drm_connector_cleanup,
	.reset = drm_atomic_helper_connector_reset,
	.atomic_duplicate_state = drm_atomic_helper_connector_duplicate_state,
	.atomic_destroy_state = drm_atomic_helper_connector_destroy_state,
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
	dev->mode_config.min_width = 320;
	dev->mode_config.max_width = 4096;
	dev->mode_config.min_height = 200;
	dev->mode_config.max_height = 4096;
	dev->mode_config.preferred_depth = 24;
	dev->mode_config.funcs = &nv_mode_config_funcs;

	ret = drm_universal_plane_init(dev, &d->primary, 0, &nv_plane_funcs, nv_formats,
				       ARRAY_SIZE(nv_formats), nv_modifiers, DRM_PLANE_TYPE_PRIMARY, NULL);
	if (ret)
		goto err;
	drm_plane_helper_add(&d->primary, &nv_plane_helper_funcs);
	drm_plane_enable_fb_damage_clips(&d->primary);

	ret = drm_crtc_init_with_planes(dev, &d->crtc, &d->primary, NULL,
					nv_intr_has_vblank(nv) ? &nv_crtc_funcs_hw : &nv_crtc_funcs, NULL);
	if (ret)
		goto err;
	drm_crtc_helper_add(&d->crtc, &nv_crtc_helper_funcs);

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
