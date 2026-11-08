// SPDX-License-Identifier: GPL-2.0
/* Build: gcc -O1 -o egltest tools/egltest.c -lgbm -lEGL -lGLESv2 ; run: egltest /dev/dri/cardN */
/* GBM + EGL + GLES2 smoke test on a DRM device: which driver does Mesa pick, can it render into a scanout BO? */
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <gbm.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>
int main(int argc, char **argv)
{
	const char *path = argc > 1 ? argv[1] : "/dev/dri/card1";
	int fd = open(path, O_RDWR | O_CLOEXEC);
	if (fd < 0) { perror(path); return 1; }
	struct gbm_device *gbm = gbm_create_device(fd);
	if (!gbm) { fprintf(stderr, "gbm_create_device failed\n"); return 1; }
	printf("gbm backend: %s\n", gbm_device_get_backend_name(gbm));
	PFNEGLGETPLATFORMDISPLAYEXTPROC gpd = (void *)eglGetProcAddress("eglGetPlatformDisplayEXT");
	EGLDisplay dpy = gpd(EGL_PLATFORM_GBM_KHR, gbm, NULL);
	EGLint maj, min;
	if (!eglInitialize(dpy, &maj, &min)) { fprintf(stderr, "eglInitialize failed 0x%x\n", eglGetError()); return 1; }
	printf("EGL %d.%d vendor %s\n", maj, min, eglQueryString(dpy, EGL_VENDOR));
	eglBindAPI(EGL_OPENGL_ES_API);
	{
		EGLConfig all[256]; EGLint na = 0, i;
		eglGetConfigs(dpy, all, 256, &na);
		printf("%d configs, native visuals:", na);
		for (i = 0; i < na && i < 12; i++) { EGLint v; eglGetConfigAttrib(dpy, all[i], EGL_NATIVE_VISUAL_ID, &v); printf(" %.4s", (char *)&v); }
		printf("\n");
		EGLint a[] = { EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE };
		EGLContext c0 = eglCreateContext(dpy, EGL_NO_CONFIG_KHR, EGL_NO_CONTEXT, a);
		if (c0 != EGL_NO_CONTEXT && eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, c0))
			printf("surfaceless: GL_RENDERER %s\n", glGetString(GL_RENDERER));
		else
			printf("surfaceless context failed 0x%x\n", eglGetError());
		eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
	}
	EGLint cattr[] = { EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
			   EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT, EGL_NATIVE_VISUAL_ID, GBM_FORMAT_XRGB8888, EGL_NONE };
	EGLConfig cfgs[256], cfg = NULL; EGLint n = 0, k;
	eglChooseConfig(dpy, cattr, cfgs, 256, &n);
	for (k = 0; k < n; k++) { EGLint v; eglGetConfigAttrib(dpy, cfgs[k], EGL_NATIVE_VISUAL_ID, &v); if (v == GBM_FORMAT_XRGB8888) { cfg = cfgs[k]; break; } }
	if (!cfg) { fprintf(stderr, "no XR24 config among %d\n", n); return 1; }
	EGLint ctxattr[] = { EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE };
	EGLContext ctx = eglCreateContext(dpy, cfg, EGL_NO_CONTEXT, ctxattr);
	struct gbm_surface *gs = gbm_surface_create(gbm, 256, 256, GBM_FORMAT_XRGB8888, GBM_BO_USE_SCANOUT | GBM_BO_USE_RENDERING);
	if (!gs) { fprintf(stderr, "gbm_surface_create failed\n"); return 1; }
	EGLSurface surf = eglCreateWindowSurface(dpy, cfg, (EGLNativeWindowType)gs, NULL);
	if (surf == EGL_NO_SURFACE || !eglMakeCurrent(dpy, surf, surf, ctx)) { fprintf(stderr, "surface/makecurrent failed 0x%x\n", eglGetError()); return 1; }
	printf("GL_RENDERER %s\nGL_VERSION %s\n", glGetString(GL_RENDERER), glGetString(GL_VERSION));
	glClearColor(0.0f, 1.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);
	eglSwapBuffers(dpy, surf);
	struct gbm_bo *bo = gbm_surface_lock_front_buffer(gs);
	uint32_t stride; void *map_data = NULL;
	uint32_t *p = gbm_bo_map(bo, 0, 0, 256, 256, GBM_BO_TRANSFER_READ, &stride, &map_data);
	printf("front bo: handle %u stride %u, pixel[0] 0x%08x, prime fd %d\n", gbm_bo_get_handle(bo).u32, stride,
	       p ? p[0] : 0, gbm_bo_get_fd(bo));
	return 0;
}
