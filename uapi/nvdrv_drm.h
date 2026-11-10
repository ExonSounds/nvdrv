/* SPDX-License-Identifier: MIT */
/*
 * nvdrv additions to the nouveau DRM uAPI. Used by nvdrv (kernel) and nvcuda (userspace).
 *
 * DRM_NOUVEAU_SETPARAM (0x01) is unused in upstream nouveau ("deprecated"); nvdrv gives it
 * per-file parameters of its own. The parameter numbers start with "NV" so that they cannot
 * be mistaken for nouveau's GETPARAM numbers.
 */
#ifndef NVDRV_DRM_H
#define NVDRV_DRM_H

#include <linux/types.h>
#include <drm/drm.h>

struct nvdrv_setparam {
	__u64 param;
	__u64 value;
};

#ifndef DRM_NOUVEAU_SETPARAM
#define DRM_NOUVEAU_SETPARAM		0x01
#endif
#define DRM_IOCTL_NOUVEAU_SETPARAM	DRM_IOWR(DRM_COMMAND_BASE + DRM_NOUVEAU_SETPARAM, struct nvdrv_setparam)

/*
 * value 1: compute client. Every submission on the file's channels keeps the GPU at its highest
 * performance level (P0, PCIe at full speed) for the next few seconds. Without it GSP-RM's
 * governor raises the clocks only after about 2.5 s of sustained load, which leaves short
 * compute jobs at idle clocks (P8: ~20x slower memory, PCIe Gen1).
 */
#define NVDRV_SETPARAM_COMPUTE_BOOST	0x4e560001ull

#endif
