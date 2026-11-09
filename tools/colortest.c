// SPDX-License-Identifier: GPL-2.0
/*
 * colortest: CRTC colour management through DRM, the way compositors and night-light tools
 * use it. Each step holds for N seconds, then everything is reset.
 *   gamma   atomic GAMMA_LUT (1024 entries): green and blue at half -> red tint
 *   ctm     atomic CTM: red and blue swapped
 *   legacy  drmModeCrtcSetGamma (the CRTC's gamma size): warm (blue at 60 %)
 *
 *   cc -O2 -o colortest tools/colortest.c $(pkg-config --cflags --libs libdrm)
 *   sudo ./colortest [/dev/dri/card0] [seconds]    (needs DRM master: no compositor running)
 */
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <xf86drm.h>
#include <xf86drmMode.h>

static int fd;
static uint32_t crtc;

static uint32_t prop_id(const char *name)
{
	drmModeObjectProperties *p = drmModeObjectGetProperties(fd, crtc, DRM_MODE_OBJECT_CRTC);
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

static int set_blob(const char *name, const void *data, size_t len)
{
	drmModeAtomicReq *req = drmModeAtomicAlloc();
	uint32_t blob = 0, id = prop_id(name);
	int ret;

	if (!id)
		return fprintf(stderr, "no %s property\n", name), -1;
	if (data && drmModeCreatePropertyBlob(fd, data, len, &blob))
		return perror("CreatePropertyBlob"), -1;
	drmModeAtomicAddProperty(req, crtc, id, blob);
	ret = drmModeAtomicCommit(fd, req, 0, NULL);
	if (ret)
		perror(name);
	drmModeAtomicFree(req);
	return ret;
}

int main(int argc, char **argv)
{
	const char *path = argc > 1 ? argv[1] : "/dev/dri/card0";
	int secs = argc > 2 ? atoi(argv[2]) : 3, i, fails = 0;
	struct drm_color_lut lut[1024];
	struct drm_color_ctm ctm = {};
	uint16_t r[4096], g[4096], b[4096];
	int n;
	drmModeRes *res;

	fd = open(path, O_RDWR | O_CLOEXEC);
	if (fd < 0 || drmSetClientCap(fd, DRM_CLIENT_CAP_ATOMIC, 1))
		return perror(path), 1;
	res = drmModeGetResources(fd);
	crtc = res->crtcs[0];
	printf("GAMMA_LUT %u, CTM %u, GAMMA_LUT_SIZE %u\n", prop_id("GAMMA_LUT"), prop_id("CTM"),
	       prop_id("GAMMA_LUT_SIZE"));

	for (i = 0; i < 1024; i++) {
		uint16_t v = i * 0xffff / 1023;

		lut[i] = (struct drm_color_lut){ .red = v, .green = v / 2, .blue = v / 2 };
	}
	printf("gamma: red tint\n");
	fails += !!set_blob("GAMMA_LUT", lut, sizeof(lut));
	sleep(secs);
	fails += !!set_blob("GAMMA_LUT", NULL, 0);

	/* S31.32 sign-magnitude, row-major: out.r = in.b, out.g = in.g, out.b = in.r */
	ctm.matrix[2] = 1ULL << 32;
	ctm.matrix[4] = 1ULL << 32;
	ctm.matrix[6] = 1ULL << 32;
	printf("ctm: red and blue swapped\n");
	fails += !!set_blob("CTM", &ctm, sizeof(ctm));
	sleep(secs);
	fails += !!set_blob("CTM", NULL, 0);

	{
		drmModeCrtc *c = drmModeGetCrtc(fd, crtc);

		n = c && c->gamma_size > 1 && c->gamma_size <= 4096 ? c->gamma_size : 256;
	}
	for (i = 0; i < n; i++) {
		r[i] = g[i] = i * 0xffff / (n - 1);
		b[i] = r[i] * 6 / 10;
	}
	printf("legacy: warm (%d entries)\n", n);
	if (drmModeCrtcSetGamma(fd, crtc, n, r, g, b))
		perror("CrtcSetGamma"), fails++;
	sleep(secs);
	fails += !!set_blob("GAMMA_LUT", NULL, 0);
	printf("%s\n", fails ? "FAIL" : "done, reset to identity");
	return !!fails;
}
