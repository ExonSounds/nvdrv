// SPDX-License-Identifier: GPL-2.0
/*
 * GPU-rendered frames shown through KMS, the way a compositor's DRM backend does it:
 * GBM scanout surface, EGL/GLES (Zink on NVK with nvdrv nvk=1), drmModeAddFB2 on the
 * front buffer, page flips. With nvdrv the flips are zero-copy (window points at the
 * buffer in VRAM). Draws moving colour bars for N seconds and reports the flip rate.
 *   cc -O2 -o kmsgl tools/kmsgl.c $(pkg-config --cflags --libs gbm egl glesv2 libdrm)
 *   [KMSGL_HZ=refresh] [KMSGL_BL=1] kmsgl [/dev/dri/card0] [seconds]
 */
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>
#include <drm_fourcc.h>
#include <fcntl.h>
#include <gbm.h>
#include <math.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <xf86drm.h>
#include <xf86drmMode.h>

static int fd;
static int pending;

static void flip_done(int fd, unsigned int seq, unsigned int s, unsigned int us, void *data)
{
	pending = 0;
}

static uint32_t fb_for(struct gbm_bo *bo)
{
	uint32_t *id = gbm_bo_get_user_data(bo);
	uint32_t handles[4] = { gbm_bo_get_handle(bo).u32 }, pitches[4] = { gbm_bo_get_stride(bo) }, offs[4] = {};
	uint64_t mods[4] = { gbm_bo_get_modifier(bo) };
	static int shown;

	if (!shown++)
		printf("scanout buffer: modifier 0x%016llx, pitch %u\n", (unsigned long long)mods[0],
		       gbm_bo_get_stride(bo));

	if (id)
		return *id;
	id = malloc(sizeof(*id));
	if (drmModeAddFB2WithModifiers(fd, gbm_bo_get_width(bo), gbm_bo_get_height(bo), gbm_bo_get_format(bo),
				       handles, pitches, offs, mods, id, DRM_MODE_FB_MODIFIERS)) {
		perror("AddFB2");
		exit(1);
	}
	gbm_bo_set_user_data(bo, id, NULL);
	return *id;
}

static double now(void)
{
	struct timespec t;

	clock_gettime(CLOCK_MONOTONIC, &t);
	return t.tv_sec + t.tv_nsec / 1e9;
}

int main(int argc, char **argv)
{
	const char *path = argc > 1 ? argv[1] : "/dev/dri/card0";
	double secs = argc > 2 ? atof(argv[2]) : 5;
	drmModeRes *res;
	drmModeConnector *conn = NULL;
	drmModeModeInfo mode;
	uint32_t crtc;
	int i;

	fd = open(path, O_RDWR | O_CLOEXEC);
	if (fd < 0 || drmSetMaster(fd)) {
		perror(path);
		return 1;
	}
	res = drmModeGetResources(fd);
	for (i = 0; i < res->count_connectors && !conn; i++) {
		conn = drmModeGetConnector(fd, res->connectors[i]);
		if (conn->connection != DRM_MODE_CONNECTED || !conn->count_modes) {
			drmModeFreeConnector(conn);
			conn = NULL;
		}
	}
	if (!conn) {
		fprintf(stderr, "no connected connector\n");
		return 1;
	}
	mode = conn->modes[0];
	for (i = 0; i < conn->count_modes; i++)
		if (conn->modes[i].type & DRM_MODE_TYPE_PREFERRED)
			mode = conn->modes[i];
	if (getenv("KMSGL_HZ"))	/* same size, this refresh rate */
		for (i = 0; i < conn->count_modes; i++)
			if (conn->modes[i].hdisplay == mode.hdisplay && conn->modes[i].vdisplay == mode.vdisplay &&
			    conn->modes[i].vrefresh == (unsigned)atoi(getenv("KMSGL_HZ"))) {
				mode = conn->modes[i];
				break;
			}
	crtc = res->crtcs[0];
	printf("%s: %ux%u@%u\n", path, mode.hdisplay, mode.vdisplay, mode.vrefresh);

	struct gbm_device *gbm = gbm_create_device(fd);
	/* KMSGL_BL=1: let the driver pick a block-linear layout (what compositors negotiate) */
	uint64_t mods[7] = { DRM_FORMAT_MOD_LINEAR };
	unsigned int nmods = 1;

	if (getenv("KMSGL_BL"))
		for (nmods = 0; nmods < 6; nmods++)
			mods[nmods] = DRM_FORMAT_MOD_NVIDIA_BLOCK_LINEAR_2D(0, 1, 2, 0x06, 5 - nmods);
	struct gbm_surface *gs = gbm_surface_create_with_modifiers2(gbm, mode.hdisplay, mode.vdisplay,
								    GBM_FORMAT_XRGB8888, mods, nmods,
								    GBM_BO_USE_SCANOUT | GBM_BO_USE_RENDERING);
	if (!gs) {
		fprintf(stderr, "gbm surface failed\n");
		return 1;
	}
	EGLDisplay dpy = eglGetPlatformDisplay(EGL_PLATFORM_GBM_KHR, gbm, NULL);
	eglInitialize(dpy, NULL, NULL);
	eglBindAPI(EGL_OPENGL_ES_API);
	EGLint cattr[] = { EGL_SURFACE_TYPE, EGL_WINDOW_BIT, EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT, EGL_RED_SIZE, 8,
			   EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 0, EGL_NONE };
	EGLConfig cfgs[256], cfg = NULL;
	EGLint n;
	eglChooseConfig(dpy, cattr, cfgs, 256, &n);
	for (i = 0; i < n && !cfg; i++) {
		EGLint vid;

		eglGetConfigAttrib(dpy, cfgs[i], EGL_NATIVE_VISUAL_ID, &vid);
		if (vid == GBM_FORMAT_XRGB8888)
			cfg = cfgs[i];
	}
	EGLint ctxattr[] = { EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE };
	EGLContext ctx = eglCreateContext(dpy, cfg, EGL_NO_CONTEXT, ctxattr);
	EGLSurface surf = eglCreateWindowSurface(dpy, cfg, (EGLNativeWindowType)gs, NULL);
	if (!cfg || !ctx || surf == EGL_NO_SURFACE || !eglMakeCurrent(dpy, surf, surf, ctx)) {
		fprintf(stderr, "EGL setup failed (0x%x)\n", eglGetError());
		return 1;
	}
	printf("GL_RENDERER %s\n", glGetString(GL_RENDERER));

	struct gbm_bo *prev = NULL, *bo;
	drmEventContext ev = { .version = 2, .page_flip_handler = flip_done };
	double t0 = now(), t;
	int frames = 0;

	glEnable(GL_SCISSOR_TEST);
	while ((t = now() - t0) < secs) {
		int bars = 8, w = mode.hdisplay;

		for (i = 0; i < bars; i++) {	/* moving colour bars */
			float h = fmodf((float)t * 0.5f + i / (float)bars, 1.0f);

			glScissor((int)(fmodf(i * w / (float)bars + t * 400, w)), 0, w / bars + 1, mode.vdisplay);
			glClearColor(fabsf(sinf(h * 6.28f)), fabsf(sinf(h * 6.28f + 2.1f)), fabsf(sinf(h * 6.28f + 4.2f)), 1);
			glClear(GL_COLOR_BUFFER_BIT);
		}
		eglSwapBuffers(dpy, surf);
		bo = gbm_surface_lock_front_buffer(gs);
		if (!prev) {
			if (drmModeSetCrtc(fd, crtc, fb_for(bo), 0, 0, &conn->connector_id, 1, &mode)) {
				perror("SetCrtc");
				return 1;
			}
		} else {
			pending = 1;
			if (drmModePageFlip(fd, crtc, fb_for(bo), DRM_MODE_PAGE_FLIP_EVENT, NULL)) {
				perror("PageFlip");
				return 1;
			}
			while (pending) {
				struct pollfd p = { fd, POLLIN };

				poll(&p, 1, 1000);
				drmHandleEvent(fd, &ev);
			}
			gbm_surface_release_buffer(gs, prev);
		}
		prev = bo;
		frames++;
		if (getenv("KMSGL_VERBOSE") && (int)t != (int)(now() - t0)) {
			static int last;

			printf("  %ds: %d fps\n", (int)(now() - t0), frames - last);
			fflush(stdout);
			last = frames;
		}
	}
	printf("%d frames in %.1f s: %.1f fps\n", frames, t, frames / t);
	return 0;
}
