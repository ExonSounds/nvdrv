// SPDX-License-Identifier: GPL-2.0
/*
 * curstest: hardware cursor through the legacy cursor ioctls (what fbcon-era clients and
 * the atomic helpers' legacy path use): a 64x64 dumb buffer with a white arrow on a
 * half-transparent blue square, moved in a circle for N seconds, then hidden.
 *
 *   cc -O2 -o curstest tools/curstest.c $(pkg-config --cflags --libs libdrm) -lm
 *   sudo ./curstest [/dev/dri/card0] [seconds] [size]
 */
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#include <xf86drm.h>
#include <xf86drmMode.h>

int main(int argc, char **argv)
{
	const char *path = argc > 1 ? argv[1] : "/dev/dri/card0";
	double secs = argc > 2 ? atof(argv[2]) : 5;
	int size = argc > 3 ? atoi(argv[3]) : 64, fd, x = 0, y = 0, moves = 0;
	struct drm_mode_create_dumb cd = { .width = size, .height = size, .bpp = 32 };
	struct drm_mode_map_dumb md = {};
	struct timespec t0, t;
	drmModeRes *res;
	drmModeCrtc *crtc;
	uint32_t *px;

	fd = open(path, O_RDWR | O_CLOEXEC);
	if (fd < 0 || drmSetClientCap(fd, DRM_CLIENT_CAP_UNIVERSAL_PLANES, 0))
		return perror(path), 1;
	res = drmModeGetResources(fd);
	crtc = res ? drmModeGetCrtc(fd, res->crtcs[0]) : NULL;
	if (!crtc || !crtc->mode_valid)
		return fprintf(stderr, "no active crtc\n"), 1;
	if (drmIoctl(fd, DRM_IOCTL_MODE_CREATE_DUMB, &cd))
		return perror("create dumb"), 1;
	md.handle = cd.handle;
	if (drmIoctl(fd, DRM_IOCTL_MODE_MAP_DUMB, &md))
		return perror("map dumb"), 1;
	px = mmap(NULL, cd.size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, md.offset);
	if (px == MAP_FAILED)
		return perror("mmap"), 1;
	for (y = 0; y < size; y++)
		for (x = 0; x < size; x++) {
			uint32_t *p = (uint32_t *)((char *)px + y * cd.pitch) + x;

			if (x <= y / 2 && y < size * 3 / 4)			/* arrow: opaque white */
				*p = 0xffffffff;
			else							/* premultiplied 50 % blue */
				*p = 0x80000080;
		}
	if (drmModeSetCursor(fd, crtc->crtc_id, cd.handle, size, size))
		return perror("SetCursor"), 1;
	clock_gettime(CLOCK_MONOTONIC, &t0);
	do {
		double a;

		clock_gettime(CLOCK_MONOTONIC, &t);
		a = (t.tv_sec - t0.tv_sec) + (t.tv_nsec - t0.tv_nsec) / 1e9;
		if (a > secs)
			break;
		x = crtc->mode.hdisplay / 2 + (int)(crtc->mode.vdisplay / 3 * cos(a * 2));
		y = crtc->mode.vdisplay / 2 + (int)(crtc->mode.vdisplay / 3 * sin(a * 2));
		if (drmModeMoveCursor(fd, crtc->crtc_id, x, y))
			return perror("MoveCursor"), 1;
		moves++;
		usleep(4000);
	} while (1);
	printf("%d moves in %.1f s; last position %d,%d\n", moves, secs, x, y);
	if (getenv("CURSTEST_KEEP"))
		return 0;
	drmModeSetCursor(fd, crtc->crtc_id, 0, 0, 0);
	return 0;
}
