// SPDX-License-Identifier: GPL-2.0
/*
 * Compute preemption latency, Vulkan (NVK on nvdrv). Two processes, two VkDevices (two GPU
 * channels): the spinner fills the GPU with one dispatch whose workgroups each run for
 * about SPIN_MS; the probe submits a trivial dispatch every 10 ms and measures submit ->
 * fence latency, idle and while the spinner runs. Without instruction-level preemption the
 * probe waits for the spinner's workgroups to finish; with CILP it is time-sliced in.
 * With "gfx" as the waves argument the spinner is one full-screen draw with a long fragment
 * shader through GLES (Zink on NVK): graphics preemption (WFI vs GFXP).
 *   glslc -o /tmp/spin.spv tools/preempt_spin.comp ; glslc -o /tmp/probe.spv tools/preempt_probe.comp
 *   cc -O2 -o preempttest tools/preempttest.c -lvulkan -lEGL -lGLESv2 -lgbm
 *   VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nouveau_icd.json preempttest [spin_ms] [dir with .spv] [waves]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#include <vulkan/vulkan.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <fcntl.h>
#include <gbm.h>

#define CHECKVK(x) do { VkResult r_ = (x); if (r_) { fprintf(stderr, "%s:%d %s = %d\n", __FILE__, __LINE__, #x, r_); exit(1); } } while (0)

static double now_ms(void)
{
	struct timespec t;

	clock_gettime(CLOCK_MONOTONIC, &t);
	return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

struct gpu {
	VkInstance inst;
	VkDevice dev;
	VkQueue q;
	VkCommandPool pool;
	VkCommandBuffer cb;
	VkPipeline pipe;
	VkPipelineLayout layout;
	VkDescriptorSet set;
	VkBuffer buf;
	VkDeviceMemory mem;
	VkFence fence;
};

static void *load(const char *path, size_t *len)
{
	FILE *f = fopen(path, "rb");
	void *d;

	if (!f) {
		perror(path);
		exit(1);
	}
	fseek(f, 0, SEEK_END);
	*len = ftell(f);
	rewind(f);
	d = malloc(*len);
	if (fread(d, 1, *len, f) != *len)
		exit(1);
	fclose(f);
	return d;
}

static void gpu_init(struct gpu *g, const char *spv, VkDeviceSize bufsize)
{
	VkApplicationInfo ai = { VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_2 };
	VkInstanceCreateInfo ici = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &ai };
	VkPhysicalDevice pd[4];
	uint32_t npd = 4, qf = 0, nqf = 16, i;
	VkQueueFamilyProperties qfp[16];
	float prio = 1.0f;
	size_t len;
	void *code = load(spv, &len);

	CHECKVK(vkCreateInstance(&ici, NULL, &g->inst));
	CHECKVK(vkEnumeratePhysicalDevices(g->inst, &npd, pd));
	vkGetPhysicalDeviceQueueFamilyProperties(pd[0], &nqf, qfp);
	for (i = 0; i < nqf; i++)
		if (qfp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
			qf = i;
			break;
		}
	VkDeviceQueueCreateInfo qci = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = qf,
					.queueCount = 1, .pQueuePriorities = &prio };
	VkDeviceCreateInfo dci = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci };
	CHECKVK(vkCreateDevice(pd[0], &dci, NULL, &g->dev));
	vkGetDeviceQueue(g->dev, qf, 0, &g->q);

	VkBufferCreateInfo bci = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = bufsize,
				   .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT };
	CHECKVK(vkCreateBuffer(g->dev, &bci, NULL, &g->buf));
	VkMemoryRequirements mr;
	VkPhysicalDeviceMemoryProperties mp;
	vkGetBufferMemoryRequirements(g->dev, g->buf, &mr);
	vkGetPhysicalDeviceMemoryProperties(pd[0], &mp);
	for (i = 0; i < mp.memoryTypeCount; i++)
		if ((mr.memoryTypeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
			break;
	VkMemoryAllocateInfo mai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size, .memoryTypeIndex = i };
	CHECKVK(vkAllocateMemory(g->dev, &mai, NULL, &g->mem));
	CHECKVK(vkBindBufferMemory(g->dev, g->buf, g->mem, 0));

	VkDescriptorSetLayoutBinding b = { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT };
	VkDescriptorSetLayoutCreateInfo dl = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 1,
					       .pBindings = &b };
	VkDescriptorSetLayout dsl;
	CHECKVK(vkCreateDescriptorSetLayout(g->dev, &dl, NULL, &dsl));
	VkPushConstantRange pcr = { VK_SHADER_STAGE_COMPUTE_BIT, 0, 4 };
	VkPipelineLayoutCreateInfo pl = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
					  .pSetLayouts = &dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr };
	CHECKVK(vkCreatePipelineLayout(g->dev, &pl, NULL, &g->layout));
	VkShaderModuleCreateInfo smi = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = len, .pCode = code };
	VkShaderModule sm;
	CHECKVK(vkCreateShaderModule(g->dev, &smi, NULL, &sm));
	VkComputePipelineCreateInfo cpi = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
		.stage = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_COMPUTE_BIT,
			   .module = sm, .pName = "main" }, .layout = g->layout };
	CHECKVK(vkCreateComputePipelines(g->dev, VK_NULL_HANDLE, 1, &cpi, NULL, &g->pipe));

	VkDescriptorPoolSize ps = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 };
	VkDescriptorPoolCreateInfo dpi = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 1, .poolSizeCount = 1,
					   .pPoolSizes = &ps };
	VkDescriptorPool dp;
	CHECKVK(vkCreateDescriptorPool(g->dev, &dpi, NULL, &dp));
	VkDescriptorSetAllocateInfo dsa = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = dp,
					    .descriptorSetCount = 1, .pSetLayouts = &dsl };
	CHECKVK(vkAllocateDescriptorSets(g->dev, &dsa, &g->set));
	VkDescriptorBufferInfo dbi = { g->buf, 0, VK_WHOLE_SIZE };
	VkWriteDescriptorSet w = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = g->set, .descriptorCount = 1,
				   .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &dbi };
	vkUpdateDescriptorSets(g->dev, 1, &w, 0, NULL);

	VkCommandPoolCreateInfo cpci = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
					 .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = qf };
	CHECKVK(vkCreateCommandPool(g->dev, &cpci, NULL, &g->pool));
	VkCommandBufferAllocateInfo cba = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = g->pool,
					    .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1 };
	CHECKVK(vkAllocateCommandBuffers(g->dev, &cba, &g->cb));
	VkFenceCreateInfo fci = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
	CHECKVK(vkCreateFence(g->dev, &fci, NULL, &g->fence));
}

/* One dispatch of groups workgroups with push constant pc; returns the submit -> fence time. */
static double run(struct gpu *g, uint32_t groups, uint32_t pc)
{
	VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
	VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &g->cb };
	double t0;
	VkResult r;

	CHECKVK(vkResetCommandBuffer(g->cb, 0));
	CHECKVK(vkBeginCommandBuffer(g->cb, &bi));
	vkCmdBindPipeline(g->cb, VK_PIPELINE_BIND_POINT_COMPUTE, g->pipe);
	vkCmdBindDescriptorSets(g->cb, VK_PIPELINE_BIND_POINT_COMPUTE, g->layout, 0, 1, &g->set, 0, NULL);
	vkCmdPushConstants(g->cb, g->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &pc);
	vkCmdDispatch(g->cb, groups, 1, 1);
	CHECKVK(vkEndCommandBuffer(g->cb));
	CHECKVK(vkResetFences(g->dev, 1, &g->fence));
	t0 = now_ms();
	CHECKVK(vkQueueSubmit(g->q, 1, &si, g->fence));
	r = vkWaitForFences(g->dev, 1, &g->fence, VK_TRUE, 60ull * 1000000000);
	if (r) {
		fprintf(stderr, "fence wait: %d (device lost?)\n", r);
		return -1;
	}
	return now_ms() - t0;
}

/* ---- graphics spinner: GLES on a GBM render node (Zink on NVK) ---- */
static GLint gfx_iters_loc;

static void gfx_init(void)
{
	static const char *vs = "#version 300 es\nvoid main() { vec2 c = vec2(gl_VertexID & 1, gl_VertexID >> 1);\n"
				"  gl_Position = vec4(c * 2.0 - 1.0, 0.0, 1.0); }\n";
	static const char *fs = "#version 300 es\nprecision highp float; uniform uint iters; out vec4 o;\n"
				"void main() { float x = gl_FragCoord.x * 1e-3;\n"
				"  for (uint i = 0u; i < iters; i++) x = x * 0.9999997 + 0.25;\n  o = vec4(x); }\n";
	int fd = open("/dev/dri/renderD128", O_RDWR | O_CLOEXEC);
	struct gbm_device *gbm = gbm_create_device(fd);
	EGLDisplay dpy = eglGetPlatformDisplay(EGL_PLATFORM_GBM_KHR, gbm, NULL);
	EGLint ca[] = { EGL_CONTEXT_MAJOR_VERSION, 3, EGL_NONE };
	const char *src[2] = { vs, fs };
	GLuint prog, tex, fbo;
	EGLContext ctx;

	if (!eglInitialize(dpy, NULL, NULL))
		exit(1);
	eglBindAPI(EGL_OPENGL_ES_API);
	ctx = eglCreateContext(dpy, EGL_NO_CONFIG_KHR, EGL_NO_CONTEXT, ca);
	if (!ctx || !eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx))
		exit(1);
	prog = glCreateProgram();
	for (int i = 0; i < 2; i++) {
		GLuint sh = glCreateShader(i ? GL_FRAGMENT_SHADER : GL_VERTEX_SHADER);

		glShaderSource(sh, 1, &src[i], NULL);
		glCompileShader(sh);
		glAttachShader(prog, sh);
	}
	glLinkProgram(prog);
	glUseProgram(prog);
	gfx_iters_loc = glGetUniformLocation(prog, "iters");
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, 2048, 2048);
	glGenFramebuffers(1, &fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, fbo);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
	glViewport(0, 0, 2048, 2048);
	printf("graphics spinner: %s\n", glGetString(GL_RENDERER));
}

static double gfx_run(uint32_t iters)
{
	double t0 = now_ms();

	glUniform1ui(gfx_iters_loc, iters);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	glFinish();
	return now_ms() - t0;
}

struct shared {
	volatile double spin_start, spin_end;
	volatile int ready, spin_err;
};

static int cmpd(const void *a, const void *b)
{
	double x = *(const double *)a, y = *(const double *)b;

	return x < y ? -1 : x > y;
}

int main(int argc, char **argv)
{
	double spin_ms = argc > 1 ? atof(argv[1]) : 1500;
	const char *dir = argc > 2 ? argv[2] : "/tmp";
	int waves = argc > 3 ? atoi(argv[3]) : 1;	/* workgroup waves: each runs spin_ms / waves */
	int gfx = argc > 3 && !strcmp(argv[3], "gfx");
	struct shared *sh = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
	char path[512];
	pid_t pid;

	memset((void *)sh, 0, sizeof(*sh));
	pid = fork();
	if (pid == 0) {	/* probe */
		static double lat[4096], t[4096];
		double idle[4096], busy[4096];
		int n = 0, ni = 0, nb = 0, i;
		struct gpu g;

		snprintf(path, sizeof(path), "%s/probe.spv", dir);
		gpu_init(&g, path, 4096);
		run(&g, 1, 0);	/* warm up */
		while (!sh->ready)
			usleep(1000);
		while (n < 4096 && !(sh->spin_end && now_ms() > sh->spin_end + 300)) {
			t[n] = now_ms();
			lat[n] = run(&g, 1, 0);
			n++;
			usleep(10000);
		}
		for (i = 0; i < n; i++) {
			if (lat[i] < 0)
				continue;
			if (t[i] >= sh->spin_start && t[i] < sh->spin_end)
				busy[nb++] = lat[i];
			else if (t[i] < sh->spin_start)
				idle[ni++] = lat[i];
		}
		qsort(idle, ni, sizeof(double), cmpd);
		qsort(busy, nb, sizeof(double), cmpd);
		if (ni)
			printf("probe idle:  %d jobs, median %.2f ms, max %.2f ms\n", ni, idle[ni / 2], idle[ni - 1]);
		if (nb)
			printf("probe busy:  %d jobs, median %.2f ms, max %.2f ms\n", nb, busy[nb / 2], busy[nb - 1]);
		else
			printf("probe busy:  no job completed while the spinner ran\n");
		exit(0);
	}

	if (gfx) {	/* graphics spinner: one draw of about spin_ms */
		uint32_t iters = 64;
		double ms;

		gfx_init();
		ms = gfx_run(iters);
		while (ms < 20) {
			iters *= 4;
			ms = gfx_run(iters);
		}
		iters = (uint32_t)(iters * (spin_ms / ms));
		sh->ready = 1;
		usleep(300000);
		sh->spin_start = now_ms();
		ms = gfx_run(iters);
		sh->spin_end = now_ms();
		printf("graphics spinner: one 2048x2048 draw, %u iterations per pixel, took %.1f ms\n", iters, ms);
		waitpid(pid, NULL, 0);
		return 0;
	}

	/* spinner: calibrate iterations for spin_ms per workgroup, then one full-GPU dispatch */
	struct gpu g;
	uint32_t groups = 38 * 16 * waves, iters = 1u << 16;
	double ms;

	snprintf(path, sizeof(path), "%s/spin.spv", dir);
	gpu_init(&g, path, groups * 64 * 4);
	ms = run(&g, groups, iters);
	while (ms < 20) {
		iters *= 4;
		ms = run(&g, groups, iters);
	}
	iters = (uint32_t)(iters * (spin_ms / ms));
	printf("spinner: %u workgroups (%d waves) x %u iterations (calibrated: %.1f ms per run)\n", groups, waves, iters,
	       spin_ms);
	sh->ready = 1;
	usleep(300000);	/* idle probes first */
	sh->spin_start = now_ms();
	ms = run(&g, groups, iters);
	sh->spin_end = now_ms();
	printf("spinner: dispatch took %.1f ms%s\n", ms, ms < 0 ? " (FAILED)" : "");
	waitpid(pid, NULL, 0);
	return 0;
}
