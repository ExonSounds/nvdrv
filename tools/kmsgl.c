// SPDX-License-Identifier: GPL-2.0
/*
 * GPU-rendered frames shown through KMS, the way a compositor's DRM backend does it:
 * GBM scanout surface, EGL/GLES (Zink on NVK with nvdrv nvk=1), drmModeAddFB2 on the
 * front buffer, page flips. With nvdrv the flips are zero-copy (window points at the
 * buffer in VRAM). Draws moving colour bars for N seconds and reports the flip rate.
 *   cc -O2 -o kmsgl tools/kmsgl.c $(pkg-config --cflags --libs gbm egl glesv2 libdrm)
 *   [KMSGL_HZ=refresh] [KMSGL_BL=1] [KMSGL_ASYNC=1 (tearing flips)] [KMSGL_ALT=1] kmsgl
 *   KMSGL_VRR=1: VRR_ENABLED on the CRTC; KMSGL_FPS=n: frame limiter; flip spacing is reported
 *   KMSGL_ALT: frames alternate dim red / dim blue: even purple with vsync, bands when tearing [/dev/dri/card0] [seconds]
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

/*
 * KMSGL_PROBE=1 (root, nvdrv debugfs): after each flip, poll window 0's ARMed offset
 * (0x690000 user area + 0x800 armed copy + SET_OFFSET 0x260) until it changes and note the
 * raster line (0x616330) at that moment. Vsync'd flips land in vblank; tearing flips anywhere.
 */
static int probe_fd = -1, probe_n, probe_active, probe_miss;
static double probe_us;

static uint32_t mmio(uint32_t reg)
{
	uint32_t v = 0;

	if (pread(probe_fd, &v, 4, reg) != 4)
		return 0;
	return v;
}

static void probe_flip(unsigned int vdisplay)
{
	static uint32_t before;
	struct timespec a, b;
	uint32_t v, line;
	int i;

	clock_gettime(CLOCK_MONOTONIC, &a);
	for (i = 0; i < 200000; i++) {
		v = mmio(0x690a60);
		if (v != before)
			break;
	}
	line = mmio(0x616330) & 0xffff;
	clock_gettime(CLOCK_MONOTONIC, &b);
	if (v == before) {
		probe_miss++;
		return;
	}
	before = v;
	probe_n++;
	probe_active += line < vdisplay;
	probe_us += (b.tv_sec - a.tv_sec) * 1e6 + (b.tv_nsec - a.tv_nsec) / 1e3;
}

/* flip completion times: spacing statistics (VRR test) */
static double flip_last, flip_sum, flip_sq, flip_min = 1e9, flip_max;
static int flip_n;

static void flip_done(int fd, unsigned int seq, unsigned int s, unsigned int us, void *data)
{
	double t = s * 1e3 + us / 1e3;

	if (flip_last) {
		double d = t - flip_last;

		flip_sum += d;
		flip_sq += d * d;
		flip_min = d < flip_min ? d : flip_min;
		flip_max = d > flip_max ? d : flip_max;
		flip_n++;
	}
	flip_last = t;
	pending = 0;
}

static void set_crtc_prop(uint32_t crtc, const char *name, uint64_t val)
{
	drmModeObjectProperties *p = drmModeObjectGetProperties(fd, crtc, DRM_MODE_OBJECT_CRTC);
	uint32_t i;

	for (i = 0; p && i < p->count_props; i++) {
		drmModePropertyRes *r = drmModeGetProperty(fd, p->props[i]);

		if (r && !strcmp(r->name, name) &&
		    drmModeObjectSetProperty(fd, crtc, DRM_MODE_OBJECT_CRTC, r->prop_id, val))
			perror(name);
		drmModeFreeProperty(r);
	}
	drmModeFreeObjectProperties(p);
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
	if (getenv("KMSGL_PROBE"))
		probe_fd = open("/sys/kernel/debug/nvdrv/0000:01:00.0/mmio", O_RDONLY);
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

		if (getenv("KMSGL_ALT")) {	/* tearing test: dim red / dim blue every other frame */
			glScissor(0, 0, w, mode.vdisplay);
			glClearColor(frames & 1 ? 0.25f : 0, 0, frames & 1 ? 0 : 0.25f, 1);
			glClear(GL_COLOR_BUFFER_BIT);
			bars = 0;
		}
		for (i = 0; i < bars; i++) {	/* moving colour bars */
			float h = fmodf((float)t * 0.5f + i / (float)bars, 1.0f);

			glScissor((int)(fmodf(i * w / (float)bars + t * 400, w)), 0, w / bars + 1, mode.vdisplay);
			glClearColor(fabsf(sinf(h * 6.28f)), fabsf(sinf(h * 6.28f + 2.1f)), fabsf(sinf(h * 6.28f + 4.2f)), 1);
			glClear(GL_COLOR_BUFFER_BIT);
		}
		if (getenv("KMSGL_FPS")) {	/* frame limiter: frame n not before t0 + n / fps */
			double due = frames / atof(getenv("KMSGL_FPS"));

			while (now() - t0 < due)
				usleep(200);
		}
		eglSwapBuffers(dpy, surf);
		bo = gbm_surface_lock_front_buffer(gs);
		if (!prev) {
			if (drmModeSetCrtc(fd, crtc, fb_for(bo), 0, 0, &conn->connector_id, 1, &mode)) {
				perror("SetCrtc");
				return 1;
			}
			set_crtc_prop(crtc, "VRR_ENABLED", !!getenv("KMSGL_VRR"));
		} else {
			pending = 1;
			if (drmModePageFlip(fd, crtc, fb_for(bo), DRM_MODE_PAGE_FLIP_EVENT |
					    (getenv("KMSGL_ASYNC") ? DRM_MODE_PAGE_FLIP_ASYNC : 0), NULL)) {
				perror("PageFlip");
				return 1;
			}
			if (probe_fd >= 0)
				probe_flip(mode.vdisplay);
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
	if (flip_n) {
		double mean = flip_sum / flip_n;

		printf("flip spacing: mean %.3f ms, stddev %.3f, min %.3f, max %.3f\n", mean,
		       sqrt(flip_sq / flip_n - mean * mean), flip_min, flip_max);
	}
	if (getenv("KMSGL_VRR"))
		set_crtc_prop(crtc, "VRR_ENABLED", 0);
	if (probe_fd >= 0 && probe_n)
		printf("probe: %d flips seen, %d not seen; %.0f %% took effect during active scanout, %.1f us after the ioctl\n",
		       probe_n, probe_miss, 100.0 * probe_active / probe_n, probe_us / probe_n);
	return 0;
}
