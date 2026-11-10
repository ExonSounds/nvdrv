// SPDX-License-Identifier: GPL-2.0
/*
 * In-place GPU reset: when GSP-RM dies (RISC-V stopped, found by the health check or during
 * an RPC), bring the GPU back on the same bind instead of requiring a driver reload.
 *
 *   1. programs get the device lost (every fence fails), new submissions wait
 *   2. with every DRM modeset lock held (no other commit can interleave): pending vblank
 *      events sent, the current display state duplicated, everything disabled
 *   3. nvdrv forgets every RM object of the dead GSP-RM: display channels and objects, kernel
 *      channels, its VM and the GR context buffers (pool memory freed); RM objects of
 *      programs' files are left to their owners, whose frees skip RM (generation check)
 *   4. FLR, then a cold GSP-RM boot (FWSEC-FRTS, Booter-Load, INIT_DONE) and nvdrv's RM
 *      client again
 *   5. BAR1 tables written again (VRAM contents are not trusted after the FLR), kernel
 *      channels, display object; the duplicated state is committed again (full DP bring-up)
 *
 * Programs that had the device lost recreate their devices; fbcon redraws on resume.
 * At most NV_RESET_MAX resets within NV_RESET_WINDOW: a GPU that keeps dying is left lost.
 */
#include <linux/jiffies.h>
#include <linux/module.h>
#include <linux/pci.h>

#include "nvdrv.h"

#define NV_RESET_MAX	3
#define NV_RESET_WINDOW	(10 * 60 * HZ)

static bool auto_reset = true;
module_param(auto_reset, bool, 0644);
MODULE_PARM_DESC(auto_reset, "Reset the GPU in place when GSP-RM dies (default: on)");

/* The hardware part, with every DRM modeset lock held (nv_drm_reset). */
static int reset_hw(struct nvdrv_dev *nv)
{
	struct device *dev = &nv->pdev->dev;
	int ret;

	nv_uapi_reset(nv, true);	/* no submissions */
	nv_intr_fini(nv);
	mutex_lock(&nv->gsp_lock);
	nv_evo_suspend(nv);		/* GSP-RM down: host memory only */
	nv_disp_reset_rm(nv);
	nv_fifo_reset(nv);
	memset(&nv->rm, 0, sizeof(nv->rm));
	nv->rm_gen++;
	ret = pci_try_reset_function(nv->pdev);	/* -EAGAIN while the device is being unbound */
	if (ret)
		dev_err(dev, "reset: FLR failed: %d\n", ret);
	if (!ret)
		ret = nv_gsp_reboot_locked(nv);
	if (!ret) {
		nv_bar1_rebuild(nv);
		ret = nv_fifo_init(nv);
		if (ret)
			dev_err(dev, "reset: kernel channels: %d\n", ret);
	}
	if (!ret && nv_evo_init(nv))
		dev_warn(dev, "reset: display object not back; the modeset tries again\n");
	mutex_unlock(&nv->gsp_lock);
	nv_uapi_reset(nv, false);
	return ret;
}

int nv_gpu_reset(struct nvdrv_dev *nv, const char *why)
{
	struct device *dev = &nv->pdev->dev;
	ktime_t t0 = ktime_get();
	int ret;

	if (!nv->gsp_touched) {
		WRITE_ONCE(nv->reset_pending, false);
		return -ENODEV;
	}
	if (nv->resets >= NV_RESET_MAX && time_before(jiffies, nv->reset_last + NV_RESET_WINDOW)) {
		dev_err(dev, "reset: %u GPU resets within %u minutes, giving up (reload nvdrv)\n",
			nv->resets, NV_RESET_WINDOW / HZ / 60);
		return -EBUSY;	/* reset_pending stays set: no further attempts */
	}
	if (time_after(jiffies, nv->reset_last + NV_RESET_WINDOW))
		nv->resets = 0;
	dev_warn(dev, "reset: GPU reset in place (%s)\n", why);

	nv_uapi_device_lost(nv, why);
	cancel_work_sync(&nv->disp.hpd_work);
	mutex_lock(&nv->gsp_lock);
	nv->gsp_running = false;	/* from here every RM call fails at once */
	mutex_unlock(&nv->gsp_lock);
	ret = nv_drm_reset(nv, reset_hw);
	if (ret)
		nv_uapi_device_lost(nv, "GPU reset failed");
	nv->resets++;
	nv->reset_last = jiffies;
	WRITE_ONCE(nv->reset_pending, false);
	if (ret)
		dev_err(dev, "reset: failed: %d (reload nvdrv)\n", ret);
	else
		dev_warn(dev, "reset: GPU back after %lld ms (reset %u)\n",
			 ktime_ms_delta(ktime_get(), t0), nv->resets);
	return ret;
}

static void reset_work(struct work_struct *w)
{
	struct nvdrv_dev *nv = container_of(w, struct nvdrv_dev, reset_work);

	nv_gpu_reset(nv, nv->reset_why);
}

/* From the health check or an RPC that found GSP-RM dead (any context that may sleep). */
void nv_gpu_reset_schedule(struct nvdrv_dev *nv, const char *why)
{
	if (!auto_reset || xchg(&nv->reset_pending, true))
		return;
	nv->reset_why = why;
	schedule_work(&nv->reset_work);
}

void nv_gpu_reset_init(struct nvdrv_dev *nv)
{
	INIT_WORK(&nv->reset_work, reset_work);
}
