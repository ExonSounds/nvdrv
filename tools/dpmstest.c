// SPDX-License-Identifier: GPL-2.0
/*
 * dpmstest: display power off and on through the CRTC ACTIVE property (what compositors do
 * for DPMS / idle blanking). Off for N seconds, then on; reports how long each commit took.
 *
 *   cc -O2 -o dpmstest tools/dpmstest.c $(pkg-config --cflags --libs libdrm)
 *   sudo ./dpmstest [/dev/dri/card0] [seconds off]   (needs DRM master: no compositor)
 */
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <xf86drm.h>
#include <xf86drmMode.h>

static int fd;

static uint32_t prop(uint32_t obj, uint32_t type, const char *name)
{
	drmModeObjectProperties *p = drmModeObjectGetProperties(fd, obj, type);
	uint32_t i, id = 0;

	for (i = 0; p && i < p->count_props; i++) {
		drmModePropertyRes *r = drmModeGetProperty(fd, p->props[i]);

		if (r && !strcmp(r->name, name))
			id = r->prop_id;
		drmModeFreeProperty(r);
	}
	drmModeFreeObjectProperties(p);
	return id;
}

static double now_ms(void)
{
	struct timespec t;

	clock_gettime(CLOCK_MONOTONIC, &t);
	return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

static int set_active(uint32_t crtc, uint32_t conn, uint32_t mode_blob, int on)
{
	drmModeAtomicReq *req = drmModeAtomicAlloc();
	double t0 = now_ms();
	int ret;

	drmModeAtomicAddProperty(req, crtc, prop(crtc, DRM_MODE_OBJECT_CRTC, "ACTIVE"), on);
	if (on) {
		drmModeAtomicAddProperty(req, crtc, prop(crtc, DRM_MODE_OBJECT_CRTC, "MODE_ID"), mode_blob);
		drmModeAtomicAddProperty(req, conn, prop(conn, DRM_MODE_OBJECT_CONNECTOR, "CRTC_ID"), crtc);
	}
	ret = drmModeAtomicCommit(fd, req, DRM_MODE_ATOMIC_ALLOW_MODESET, NULL);
	drmModeAtomicFree(req);
	printf("display %s: %s (%.0f ms)\n", on ? "on" : "off", ret ? strerror(-ret) : "ok", now_ms() - t0);
	return ret;
}

int main(int argc, char **argv)
{
	const char *path = argc > 1 ? argv[1] : "/dev/dri/card0";
	int secs = argc > 2 ? atoi(argv[2]) : 5;
	drmModeRes *res;
	drmModeCrtc *c;
	uint32_t blob, conn;

	fd = open(path, O_RDWR | O_CLOEXEC);
	if (fd < 0 || drmSetClientCap(fd, DRM_CLIENT_CAP_ATOMIC, 1))
		return perror(path), 1;
	res = drmModeGetResources(fd);
	c = drmModeGetCrtc(fd, res->crtcs[0]);
	conn = res->connectors[0];
	if (!c || !c->mode_valid || drmModeCreatePropertyBlob(fd, &c->mode, sizeof(c->mode), &blob))
		return fprintf(stderr, "no active mode\n"), 1;
	if (set_active(c->crtc_id, conn, blob, 0))
		return 1;
	sleep(secs);
	return !!set_active(c->crtc_id, conn, blob, 1);
}
