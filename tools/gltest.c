// SPDX-License-Identifier: GPL-2.0
/*
 * GL (Zink on NVK) rendering correctness, off-screen: an RGBA8 texture + depth renderbuffer
 * FBO (dedicated, unshared images: what NVK compresses with uAPI >= 1.4.2), a per-pixel
 * gradient, then a depth-tested overlap pattern; glReadPixels compared with the expected
 * image computed here.
 *   cc -O2 -o gltest tools/gltest.c $(pkg-config --cflags --libs gbm egl glesv2)
 *   gltest [/dev/dri/renderD128]
 */
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <fcntl.h>
#include <gbm.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define W 1024
#define H 1024

static GLuint prog(const char *vs, const char *fs)
{
	GLuint p = glCreateProgram(), s[2];
	const char *src[2] = { vs, fs };
	GLint ok;

	for (int i = 0; i < 2; i++) {
		s[i] = glCreateShader(i ? GL_FRAGMENT_SHADER : GL_VERTEX_SHADER);
		glShaderSource(s[i], 1, &src[i], NULL);
		glCompileShader(s[i]);
		glGetShaderiv(s[i], GL_COMPILE_STATUS, &ok);
		if (!ok) {
			char log[512];

			glGetShaderInfoLog(s[i], sizeof(log), NULL, log);
			fprintf(stderr, "shader: %s\n", log);
			exit(1);
		}
		glAttachShader(p, s[i]);
	}
	glLinkProgram(p);
	return p;
}

static const char *vs =
	"#version 300 es\n"
	"uniform vec4 rect; uniform float z;\n"
	"void main() { vec2 c = vec2(gl_VertexID & 1, gl_VertexID >> 1);\n"
	"  gl_Position = vec4(mix(rect.xy, rect.zw, c), z, 1.0); }\n";
static const char *fs_grad =
	"#version 300 es\nprecision highp float; out vec4 o;\n"
	"void main() { uvec2 p = uvec2(gl_FragCoord.xy);\n"
	"  o = vec4(float(p.x & 255u), float(p.y & 255u), float((p.x ^ p.y) & 255u), 255.0) / 255.0; }\n";
static const char *fs_col =
	"#version 300 es\nprecision highp float; uniform vec4 col; out vec4 o; void main() { o = col; }\n";

static void quad(GLuint p, float x0, float y0, float x1, float y1, float z)
{
	glUseProgram(p);
	glUniform4f(glGetUniformLocation(p, "rect"), x0, y0, x1, y1);
	glUniform1f(glGetUniformLocation(p, "z"), z);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
}

int main(int argc, char **argv)
{
	int fd = open(argc > 1 ? argv[1] : "/dev/dri/renderD128", O_RDWR | O_CLOEXEC);
	struct gbm_device *gbm = gbm_create_device(fd);
	EGLDisplay dpy = eglGetPlatformDisplay(EGL_PLATFORM_GBM_KHR, gbm, NULL);
	EGLint ca[] = { EGL_CONTEXT_MAJOR_VERSION, 3, EGL_NONE };
	EGLContext ctx;
	GLuint tex, rb, fbo, pg, pc;
	unsigned char *px;
	long bad = 0;

	if (fd < 0 || !gbm || !eglInitialize(dpy, NULL, NULL)) {
		fprintf(stderr, "EGL init failed\n");
		return 1;
	}
	eglBindAPI(EGL_OPENGL_ES_API);
	ctx = eglCreateContext(dpy, EGL_NO_CONFIG_KHR, EGL_NO_CONTEXT, ca);
	if (!ctx || !eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx)) {
		fprintf(stderr, "context failed 0x%x\n", eglGetError());
		return 1;
	}
	printf("GL_RENDERER %s\n", glGetString(GL_RENDERER));

	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, W, H);
	glGenRenderbuffers(1, &rb);
	glBindRenderbuffer(GL_RENDERBUFFER, rb);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, W, H);
	glGenFramebuffers(1, &fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, fbo);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rb);
	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
		fprintf(stderr, "FBO incomplete\n");
		return 1;
	}
	glViewport(0, 0, W, H);
	pg = prog(vs, fs_grad);
	pc = prog(vs, fs_col);

	glClearColor(0, 0, 1, 1);
	glClearDepthf(1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);
	/* top half: gradient (z 0.0); bottom half: red at z 0.5, then green behind it on the left
	 * (rejected) and yellow in front of it on the right (accepted) */
	quad(pg, -1, 0, 1, 1, 0.0f);
	glUseProgram(pc);
	glUniform4f(glGetUniformLocation(pc, "col"), 1, 0, 0, 1);
	quad(pc, -1, -1, 1, 0, 0.0f);
	glUniform4f(glGetUniformLocation(pc, "col"), 0, 1, 0, 1);
	quad(pc, -1, -1, 0, 0, 0.4f);
	glUniform4f(glGetUniformLocation(pc, "col"), 1, 1, 0, 1);
	quad(pc, 0, -1, 1, 0, -0.4f);

	px = malloc(W * H * 4);
	glReadPixels(0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, px);
	for (int y = 0; y < H; y++)
		for (int x = 0; x < W; x++) {
			const unsigned char *p = px + (y * W + x) * 4;
			unsigned char e[4];

			if (y >= H / 2) {
				e[0] = x & 255; e[1] = y & 255; e[2] = (x ^ y) & 255; e[3] = 255;
			} else if (x < W / 2) {
				e[0] = 255; e[1] = 0; e[2] = 0; e[3] = 255;
			} else {
				e[0] = 255; e[1] = 255; e[2] = 0; e[3] = 255;
			}
			if (memcmp(p, e, 4)) {
				if (bad < 4)
					printf("  (%d,%d): %02x%02x%02x%02x, expected %02x%02x%02x%02x\n", x, y, p[0], p[1],
					       p[2], p[3], e[0], e[1], e[2], e[3]);
				bad++;
			}
		}
	printf("gltest: %ld of %d pixels wrong -> %s\n", bad, W * H, bad ? "FAIL" : "PASS");
	return bad != 0;
}
