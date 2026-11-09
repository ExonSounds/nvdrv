// SPDX-License-Identifier: GPL-2.0
/*
 * vblanktest: wait for N vblanks on CRTC 0 and report the interval between the kernel's
 * vblank timestamps (mean, min, max) and how late the waiter woke up after each.
 *
 *   cc -O2 -o vblanktest tools/vblanktest.c $(pkg-config --cflags --libs libdrm)
 *   sudo ./vblanktest [/dev/dri/card0] [count]
 */
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <xf86drm.h>
#include <xf86drmMode.h>

int main(int argc, char **argv)
{
	const char *path = argc > 1 ? argv[1] : "/dev/dri/card0";
	int n = argc > 2 ? atoi(argv[2]) : 300, fd, i, hz = 0;
	double prev = 0, sum = 0, min = 1e9, max = 0, late_max = 0, late_sum = 0;
	drmModeRes *res;

	fd = open(path, O_RDWR | O_CLOEXEC);
	if (fd < 0) {
		perror(path);
		return 1;
	}
	res = drmModeGetResources(fd);
	if (res && res->count_crtcs) {
		drmModeCrtc *c = drmModeGetCrtc(fd, res->crtcs[0]);

		if (c && c->mode_valid)
			hz = c->mode.vrefresh;
		printf("crtc %u: %dx%d @ %d Hz\n", res->crtcs[0], c ? c->mode.hdisplay : 0,
		       c ? c->mode.vdisplay : 0, hz);
	}
	for (i = 0; i <= n; i++) {
		drmVBlank vb = { .request = { .type = DRM_VBLANK_RELATIVE, .sequence = 1 } };
		struct timespec now;
		double t, late;

		if (drmWaitVBlank(fd, &vb)) {
			perror("drmWaitVBlank");
			return 1;
		}
		clock_gettime(CLOCK_MONOTONIC, &now);
		t = vb.reply.tval_sec * 1e3 + vb.reply.tval_usec / 1e3;
		late = now.tv_sec * 1e3 + now.tv_nsec / 1e6 - t;
		if (i) {
			double d = t - prev;

			sum += d;
			min = d < min ? d : min;
			max = d > max ? d : max;
			late_sum += late;
			late_max = late > late_max ? late : late_max;
		}
		prev = t;
	}
	printf("%d vblanks: interval mean %.4f ms (%.3f Hz), min %.4f, max %.4f; wake-up delay mean %.3f ms, max %.3f ms\n",
	       n, sum / n, 1000.0 / (sum / n), min, max, late_sum / n, late_max);
	if (hz)
		printf("expected interval %.4f ms\n", 1000.0 / hz);
	return 0;
}
