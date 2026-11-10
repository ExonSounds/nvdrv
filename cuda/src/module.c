// SPDX-License-Identifier: MIT
/*
 * Modules, functions and launches.
 *
 * Images: PTX text (src/ptx.c), or SPIR-V (OpenCL Kernel modules, e.g. CUDA C++ compiled by clang
 * for spirv64; an nvcuda extension). cuModuleLoadData gets no size, so a SPIR-V image must
 * end with a zero word (cuModuleLoad adds it). Functions are compiled on first lookup by
 * src/compile.c (Mesa's spirv_to_nir + NAK) and uploaded to VRAM. NVIDIA cubins are never
 * accepted (undocumented format): CUDA_ERROR_NO_BINARY_FOR_GPU, so programs that also ship PTX
 * can fall back to it.
 *
 * Launch (the way NVK and NAK's hw_runner do it, classes from open-gpu-doc clc7c0.h): constant
 * buffer 0 (block, grid, parameters) and a QMD (V03_00, filled by NAK) are written into the push
 * ring, then INVALIDATE_SHADER_CACHES_NO_WFI, SEND_PCAS_A = QMD >> 8 and SEND_SIGNALING_PCAS2_B =
 * INVALIDATE_COPY_SCHEDULE on the context's compute channel, in stream order.
 */
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <xf86drm.h>
#include <drm/nouveau_drm.h>

#include "internal.h"
#include "compile.h"

#define DRM_NOUVEAU_NVIF 0x07
struct nvif_hdr { uint8_t version, type, pad02[4], owner, route; uint64_t token, object; };
struct nvif_new { uint8_t version, pad01[6], route; uint64_t token, object; uint32_t handle; int32_t oclass; };

#define SUBC_COMPUTE				1
#define C7C0_SET_SHADER_SHARED_MEMORY_WINDOW_A	0x02a0
#define C7C0_SET_SHADER_LOCAL_MEMORY_WINDOW_A	0x07b0
#define C7C0_SET_SHADER_LOCAL_MEMORY_A		0x0790
#define C7C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A 0x02e4
#define C7C0_INVALIDATE_SKED_CACHES		0x0298
#define C7C0_SEND_PCAS_A			0x02b4
#define C7C0_SEND_SIGNALING_PCAS2_B		0x02c0
#define C7C0_INVALIDATE_SHADER_CACHES_NO_WFI	0x1698
#define INVALIDATE_INSTRUCTION_GLOBAL_CONSTANT	(1u << 0 | 1u << 4 | 1u << 12)
#define PCAS_ACTION_INVALIDATE_COPY_SCHEDULE	3
/* shared/local memory windows NVK uses: generic addresses there are not global memory */
#define SHARED_WINDOW				0xfe000000u
#define LOCAL_WINDOW				0xff000000u
#define CODE_PREFETCH_PAD			2048	/* the instruction prefetcher reads past the end */

#define SPIRV_MAGIC				0x07230203u

struct CUfunc_st {
	struct CUfunc_st *next;
	struct CUmod_st *mod;
	char *name;
	struct nvc_kernel_bin *kb;
	struct nvc_bo code;
	uint32_t max_dyn_smem;		/* CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES */
};

struct CUmod_st {
	struct CUmod_st *next;
	struct CUctx_st *ctx;
	uint32_t *words;		/* SPIR-V, or NULL */
	size_t nwords;
	char *ptx;			/* PTX text, or NULL */
	struct CUfunc_st *funcs;
};

static pthread_mutex_t compiler_lock = PTHREAD_MUTEX_INITIALIZER;

static bool is_elf(const void *image)
{
	return !memcmp(image, "\x7f" "ELF", 4);
}

static bool is_spirv(const void *image)
{
	return *(const uint32_t *)image == SPIRV_MAGIC;
}

/* PTX is text: its .version and .target directives come before any code */
static bool is_ptx(const char *s)
{
	const char *v = strstr(s, ".version");

	return v && v - s < 4096 && strstr(s, ".target");
}

/* ---- the compute channel, created with the first launch ---- */

static CUresult gr_init(struct CUctx_st *ctx)
{
	struct drm_nouveau_channel_alloc ca = { .fb_ctxdma_handle = ~0u, .tt_ctxdma_handle = NOUVEAU_FIFO_ENGINE_GR };
	struct { struct nvif_hdr h; struct nvif_new n; } on = {
		.h = { .type = 2, .route = 0xff }, .n = { .handle = 0xc7c00000, .oclass = 0xc7c0 } };
	uint32_t *start, *p;
	CUresult r;

	pthread_mutex_lock(&compiler_lock);
	r = nvc_compiler_init(ctx->dev->gpc_count, ctx->dev->tpc_count) ? CUDA_ERROR_UNKNOWN : CUDA_SUCCESS;
	pthread_mutex_unlock(&compiler_lock);
	if (r)
		return r;
	pthread_mutex_lock(&ctx->lock);
	if (ctx->gr_ready) {
		pthread_mutex_unlock(&ctx->lock);
		return CUDA_SUCCESS;
	}
	if (drmCommandWriteRead(ctx->fd, DRM_NOUVEAU_CHANNEL_ALLOC, &ca, sizeof(ca))) {
		pthread_mutex_unlock(&ctx->lock);
		return CUDA_ERROR_OUT_OF_MEMORY;
	}
	on.h.token = ca.channel;
	if (drmCommandWrite(ctx->fd, DRM_NOUVEAU_NVIF, &on, sizeof(on))) {
		pthread_mutex_unlock(&ctx->lock);
		return CUDA_ERROR_UNKNOWN;
	}
	ctx->gr_chan = ca.channel;
	start = p = nvc_push_begin(ctx, 16);
	p = nvc_mthd1(p, SUBC_COMPUTE, 0x0000, 0xc7c0);	/* SET_OBJECT */
	p = nvc_mthd(p, SUBC_COMPUTE, C7C0_SET_SHADER_SHARED_MEMORY_WINDOW_A, 2, (uint32_t[]){ 0, SHARED_WINDOW });
	p = nvc_mthd(p, SUBC_COMPUTE, C7C0_SET_SHADER_LOCAL_MEMORY_WINDOW_A, 2, (uint32_t[]){ 0, LOCAL_WINDOW });
	p = nvc_mthd1(p, SUBC_COMPUTE, C7C0_INVALIDATE_SKED_CACHES, 0);
	r = nvc_push_submit(ctx, &ctx->null_stream, ctx->gr_chan, start, p);
	if (r == CUDA_SUCCESS)
		ctx->gr_ready = true;
	pthread_mutex_unlock(&ctx->lock);
	return r;
}

/*
 * Local memory for every TPC, sized like NVK (nvk_device.c): per warp align(slm * 32, 0x200),
 * per TPC align(per_warp * 48 warps * 2 SMs, 0x8000), in total align(per_tpc * TPCs, 0x20000).
 * Grows (never shrinks) when a kernel needs more; the context is idled first. Caller holds no
 * lock.
 */
static CUresult slm_ensure(struct CUctx_st *ctx, uint32_t slm_per_thread)
{
	uint64_t per_warp, per_tpc, total;
	uint32_t *start, *p;
	CUresult r;

	if (slm_per_thread <= ctx->slm_per_thread)
		return CUDA_SUCCESS;
	slm_per_thread = (slm_per_thread + 0x1ff) & ~0x1ffu;	/* some headroom against regrowing */
	per_warp = ((uint64_t)slm_per_thread * 32 + 0x1ff) & ~0x1ffull;
	per_tpc = (per_warp * 48 * 2 + 0x7fff) & ~0x7fffull;
	total = (per_tpc * ctx->dev->tpc_count + 0x1ffff) & ~0x1ffffull;
	nvc_ctx_idle(ctx);
	pthread_mutex_lock(&ctx->lock);
	nvc_bo_free(ctx, &ctx->slm);
	if (nvc_bo_new(ctx, &ctx->slm, total, NOUVEAU_GEM_DOMAIN_VRAM, false, 0)) {
		ctx->slm_per_thread = 0;
		pthread_mutex_unlock(&ctx->lock);
		return CUDA_ERROR_OUT_OF_MEMORY;
	}
	start = p = nvc_push_begin(ctx, 8);
	p = nvc_mthd(p, SUBC_COMPUTE, C7C0_SET_SHADER_LOCAL_MEMORY_A, 2,
		     (uint32_t[]){ (uint32_t)(ctx->slm.va >> 32), (uint32_t)ctx->slm.va });
	p = nvc_mthd(p, SUBC_COMPUTE, C7C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A, 3,
		     (uint32_t[]){ (uint32_t)(per_tpc >> 32), (uint32_t)per_tpc, 0xff });
	r = nvc_push_submit(ctx, &ctx->null_stream, ctx->gr_chan, start, p);
	if (r == CUDA_SUCCESS)
		ctx->slm_per_thread = slm_per_thread;
	pthread_mutex_unlock(&ctx->lock);
	return r;
}

/* ---- modules ---- */

CUAPI CUresult cuModuleLoadDataEx(CUmodule *module, const void *image, unsigned int n, CUjit_option *options,
				  void **values)
{
	struct CUctx_st *ctx;
	struct CUmod_st *m;
	const uint32_t *w = image;
	size_t words;
	CUresult r = nvc_ctx_get(&ctx);

	(void)n;
	(void)options;
	(void)values;
	if (r)
		return r;
	if (!module || !image)
		return CUDA_ERROR_INVALID_VALUE;
	if (is_elf(image))
		return CUDA_ERROR_NO_BINARY_FOR_GPU;
	if (!is_spirv(image)) {
		if (!is_ptx(image))
			return CUDA_ERROR_INVALID_IMAGE;
		m = calloc(1, sizeof(*m));
		if (!m || !(m->ptx = strdup(image))) {	/* PTX images are NUL-terminated text */
			free(m);
			return CUDA_ERROR_OUT_OF_MEMORY;
		}
		m->ctx = ctx;
		goto add;
	}
	/* SPIR-V carries no length: the header's bound is not one either, so walk the instructions */
	for (words = 5; ; ) {
		uint32_t len = w[words] >> 16;

		if (!len || (w[words] & 0xffff) == 0)	/* end: a zero word (images are padded) */
			break;
		words += len;
		if (words > (64u << 20) / 4)
			return CUDA_ERROR_INVALID_IMAGE;
	}
	m = calloc(1, sizeof(*m));
	if (!m)
		return CUDA_ERROR_OUT_OF_MEMORY;
	m->ctx = ctx;
	m->nwords = words;
	m->words = malloc(words * 4);
	if (!m->words) {
		free(m);
		return CUDA_ERROR_OUT_OF_MEMORY;
	}
	memcpy(m->words, image, words * 4);
add:
	pthread_mutex_lock(&ctx->lock);
	m->next = ctx->modules;
	ctx->modules = m;
	pthread_mutex_unlock(&ctx->lock);
	*module = m;
	return CUDA_SUCCESS;
}

/* A SPIR-V image of known size (cuModuleLoad, tests): the instruction walk needs a terminator. */
static CUresult load_sized(CUmodule *module, const void *data, size_t size)
{
	uint32_t *buf;
	CUresult r;

	if (size >= 4 && is_elf(data))
		return CUDA_ERROR_NO_BINARY_FOR_GPU;
	if (size < 20 || (is_spirv(data) && size % 4))
		return CUDA_ERROR_INVALID_IMAGE;
	buf = calloc(1, size + 4);	/* a zero word after SPIR-V, a NUL after PTX */
	if (!buf)
		return CUDA_ERROR_OUT_OF_MEMORY;
	memcpy(buf, data, size);
	r = cuModuleLoadDataEx(module, buf, 0, NULL, NULL);
	free(buf);
	return r;
}

CUAPI CUresult cuModuleLoadData(CUmodule *module, const void *image)
{
	return cuModuleLoadDataEx(module, image, 0, NULL, NULL);
}

CUAPI CUresult cuModuleLoad(CUmodule *module, const char *path)
{
	FILE *f;
	long n;
	void *data;
	CUresult r;

	if (!module || !path)
		return CUDA_ERROR_INVALID_VALUE;
	f = fopen(path, "rb");
	if (!f)
		return CUDA_ERROR_FILE_NOT_FOUND;
	if (fseek(f, 0, SEEK_END) || (n = ftell(f)) <= 0) {
		fclose(f);
		return CUDA_ERROR_INVALID_IMAGE;
	}
	rewind(f);
	data = malloc(n);
	if (!data || fread(data, 1, n, f) != (size_t)n) {
		free(data);
		fclose(f);
		return CUDA_ERROR_INVALID_IMAGE;
	}
	fclose(f);
	r = load_sized(module, data, n);
	free(data);
	return r;
}

static void func_free(struct CUctx_st *ctx, struct CUfunc_st *f)
{
	nvc_bo_free(ctx, &f->code);
	nvc_kernel_bin_free(f->kb);
	free(f->name);
	free(f);
}

static void module_free(struct CUmod_st *m)
{
	struct CUfunc_st *f;

	while ((f = m->funcs)) {
		m->funcs = f->next;
		func_free(m->ctx, f);
	}
	free(m->words);
	free(m->ptx);
	free(m);
}

CUAPI CUresult cuModuleUnload(CUmodule module)
{
	struct CUctx_st *ctx;
	struct CUmod_st **pp;

	if (!module)
		return CUDA_ERROR_INVALID_HANDLE;
	ctx = module->ctx;
	nvc_ctx_idle(ctx);	/* launches may still run its code */
	pthread_mutex_lock(&ctx->lock);
	for (pp = &ctx->modules; *pp && *pp != module; pp = &(*pp)->next)
		;
	if (*pp)
		*pp = module->next;
	pthread_mutex_unlock(&ctx->lock);
	module_free(module);
	return CUDA_SUCCESS;
}

void nvc_modules_drop(struct CUctx_st *ctx)
{
	struct CUmod_st *m;

	while ((m = ctx->modules)) {
		ctx->modules = m->next;
		module_free(m);
	}
}

CUAPI CUresult cuModuleGetFunction(CUfunction *func, CUmodule module, const char *name)
{
	struct CUctx_st *ctx;
	struct CUfunc_st *f;
	char err[256] = "";
	CUresult r;

	if (!func || !name)
		return CUDA_ERROR_INVALID_VALUE;
	if (!module)
		return CUDA_ERROR_INVALID_HANDLE;
	ctx = module->ctx;
	pthread_mutex_lock(&ctx->lock);
	for (f = module->funcs; f && strcmp(f->name, name); f = f->next)
		;
	pthread_mutex_unlock(&ctx->lock);
	if (f) {
		*func = f;
		return CUDA_SUCCESS;
	}
	f = calloc(1, sizeof(*f));
	if (!f)
		return CUDA_ERROR_OUT_OF_MEMORY;
	f->mod = module;
	f->name = strdup(name);
	f->max_dyn_smem = 48 << 10;
	pthread_mutex_lock(&compiler_lock);
	r = nvc_compiler_init(ctx->dev->gpc_count, ctx->dev->tpc_count) ? CUDA_ERROR_UNKNOWN : CUDA_SUCCESS;
	if (r == CUDA_SUCCESS) {
		f->kb = module->ptx ? nvc_compile_ptx(module->ptx, name, err, sizeof(err)) :
				      nvc_compile_spirv(module->words, module->nwords, name, err, sizeof(err));
		if (!f->kb)
			r = CUDA_ERROR_NOT_FOUND;
	}
	pthread_mutex_unlock(&compiler_lock);
	if (r == CUDA_SUCCESS) {
		if (nvc_bo_new(ctx, &f->code, f->kb->code_size + CODE_PREFETCH_PAD, NOUVEAU_GEM_DOMAIN_VRAM, false, 0))
			r = CUDA_ERROR_OUT_OF_MEMORY;
		else
			r = nvc_upload(ctx, f->code.va, f->kb->code, f->kb->code_size);
	}
	if (r != CUDA_SUCCESS) {
		if (err[0] && getenv("NVCUDA_DEBUG"))
			fprintf(stderr, "nvcuda: %s: %s\n", name, err);
		func_free(ctx, f);
		return r;
	}
	pthread_mutex_lock(&ctx->lock);
	f->next = module->funcs;
	module->funcs = f;
	pthread_mutex_unlock(&ctx->lock);
	*func = f;
	return CUDA_SUCCESS;
}

CUAPI CUresult cuModuleGetGlobal_v2(CUdeviceptr *dptr, size_t *bytes, CUmodule module, const char *name)
{
	(void)dptr;
	(void)bytes;
	(void)name;
	return module ? CUDA_ERROR_NOT_FOUND : CUDA_ERROR_INVALID_HANDLE;	/* module globals: milestone 4 */
}

/* ---- function attributes and occupancy ---- */

/* Threads per block the register file allows: registers go out per warp in units of 8 per thread. */
static uint32_t max_threads(const struct nvc_kernel_bin *kb)
{
	uint32_t regs = (nvc_kernel_gprs(kb) + 7) & ~7u, warps;

	if (!regs)
		regs = 8;
	warps = 65536 / (regs * 32);
	return warps * 32 < 1024 ? warps * 32 : 1024;
}

CUAPI CUresult cuFuncGetAttribute(int *value, CUfunction_attribute attrib, CUfunction func)
{
	if (!value)
		return CUDA_ERROR_INVALID_VALUE;
	if (!func)
		return CUDA_ERROR_INVALID_HANDLE;
	switch ((int)attrib) {
	case CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK:	*value = max_threads(func->kb); break;
	case CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES:	*value = func->kb->shared_static; break;
	case CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES:	*value = 0; break;
	case CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES:	*value = nvc_kernel_slm(func->kb); break;
	case CU_FUNC_ATTRIBUTE_NUM_REGS:		*value = nvc_kernel_gprs(func->kb); break;
	case CU_FUNC_ATTRIBUTE_PTX_VERSION:		*value = 86; break;
	case CU_FUNC_ATTRIBUTE_BINARY_VERSION:		*value = 86; break;
	case CU_FUNC_ATTRIBUTE_CACHE_MODE_CA:		*value = 1; break;
	case 8 /* MAX_DYNAMIC_SHARED_SIZE_BYTES */:	*value = func->max_dyn_smem; break;
	case 9 /* PREFERRED_SHARED_MEMORY_CARVEOUT */:	*value = -1; break;
	default:					return CUDA_ERROR_INVALID_VALUE;
	}
	return CUDA_SUCCESS;
}

CUAPI CUresult cuFuncSetAttribute(CUfunction func, CUfunction_attribute attrib, int value);
CUAPI CUresult cuFuncSetAttribute(CUfunction func, CUfunction_attribute attrib, int value)
{
	if (!func)
		return CUDA_ERROR_INVALID_HANDLE;
	switch ((int)attrib) {
	case 8 /* MAX_DYNAMIC_SHARED_SIZE_BYTES */:
		if (value < 0 || func->kb->shared_static + (uint32_t)value > (99u << 10))
			return CUDA_ERROR_INVALID_VALUE;
		func->max_dyn_smem = value;
		return CUDA_SUCCESS;
	case 9 /* PREFERRED_SHARED_MEMORY_CARVEOUT: a hint; NAK picks the carveout per launch */:
		return CUDA_SUCCESS;
	default:
		return CUDA_ERROR_INVALID_VALUE;
	}
}

CUAPI CUresult cuFuncSetCacheConfig(CUfunction func, CUfunc_cache config)
{
	(void)config;
	return func ? CUDA_SUCCESS : CUDA_ERROR_INVALID_HANDLE;	/* the carveout is chosen per launch */
}

/* The largest block the kernel can run with, and enough blocks to fill every SM once. */
CUAPI CUresult cuOccupancyMaxPotentialBlockSize(int *min_grid, int *block, CUfunction func, void *size_fn,
						size_t dyn_smem, int limit)
{
	struct CUctx_st *ctx;
	uint32_t b, per_sm;

	(void)size_fn;
	if (!min_grid || !block)
		return CUDA_ERROR_INVALID_VALUE;
	if (!func)
		return CUDA_ERROR_INVALID_HANDLE;
	ctx = func->mod->ctx;
	b = max_threads(func->kb);
	if (limit > 0 && (uint32_t)limit < b)
		b = (uint32_t)limit & ~31u ? (uint32_t)limit & ~31u : (uint32_t)limit;
	per_sm = 1536 / b;
	if (func->kb->shared_static + dyn_smem) {
		uint32_t s = (100u << 10) / (func->kb->shared_static + dyn_smem);

		if (s < per_sm)
			per_sm = s;
	}
	if (!per_sm)
		per_sm = 1;
	*block = b;
	*min_grid = per_sm * ctx->dev->sm_count;
	return CUDA_SUCCESS;
}

/* ---- launch ---- */

CUAPI CUresult cuLaunchKernel(CUfunction f, unsigned int gx, unsigned int gy, unsigned int gz, unsigned int bx,
			      unsigned int by, unsigned int bz, unsigned int shared_bytes, CUstream stream,
			      void **params, void **extra)
{
	const struct nvc_kernel_bin *kb;
	struct CUctx_st *ctx;
	struct CUstream_st *s;
	uint32_t grid[3] = { gx, gy, gz }, block[3] = { bx, by, bz };
	uint32_t cb_size, qmd_size, *cb, *qmd, *start, *p, i;
	uint64_t threads = (uint64_t)bx * by * bz, cb_va, qmd_va;
	char *ring;
	CUresult r;

	if (!f)
		return CUDA_ERROR_INVALID_HANDLE;
	kb = f->kb;
	ctx = f->mod->ctx;
	s = nvc_stream(ctx, stream);
	if (!s)
		return CUDA_ERROR_INVALID_HANDLE;
	if (!gx || !gy || !gz || !bx || !by || !bz || bx > 1024 || by > 1024 || bz > 64 || gy > 65535 ||
	    gz > 65535 || gx > 0x7fffffffu || threads > 1024)
		return CUDA_ERROR_INVALID_VALUE;
	if (threads > max_threads(kb))
		return CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES;
	if (shared_bytes > f->max_dyn_smem)
		return CUDA_ERROR_INVALID_VALUE;
	if (kb->nparams && !params && !extra)
		return CUDA_ERROR_INVALID_VALUE;
	if (!ctx->gr_ready && (r = gr_init(ctx)) != CUDA_SUCCESS)
		return r;
	if (nvc_kernel_slm(kb) && (r = slm_ensure(ctx, nvc_kernel_slm(kb))) != CUDA_SUCCESS)
		return r;

	cb_size = (NVC_CB_PARAMS + kb->param_bytes + 255) & ~255u;
	qmd_size = nvc_qmd_size();
	pthread_mutex_lock(&ctx->lock);
	/* constant buffer and QMD go into the push ring (256-byte aligned), the methods after them */
	ring = (char *)nvc_push_begin(ctx, (256 + cb_size + 256 + qmd_size) / 4 + 32);
	cb = (uint32_t *)(((uintptr_t)ring + 255) & ~(uintptr_t)255);
	qmd = (uint32_t *)((char *)cb + cb_size);
	cb_va = ctx->push.va + ((char *)cb - (char *)ctx->push.map);
	qmd_va = cb_va + cb_size;
	memset(cb, 0, cb_size);
	memcpy(cb + NVC_CB_BLOCK / 4, block, sizeof(block));
	memcpy(cb + NVC_CB_GRID / 4, grid, sizeof(grid));
	if (params) {
		for (i = 0; i < kb->nparams; i++)
			memcpy((char *)cb + NVC_CB_PARAMS + kb->params[i].offset, params[i], kb->params[i].size);
	} else if (extra) {
		void *buf = NULL;
		size_t size = 0;

		for (i = 0; extra[i] != CU_LAUNCH_PARAM_END; i += 2) {
			if (extra[i] == CU_LAUNCH_PARAM_BUFFER_POINTER)
				buf = extra[i + 1];
			else if (extra[i] == CU_LAUNCH_PARAM_BUFFER_SIZE)
				size = *(size_t *)extra[i + 1];
		}
		if (buf)
			memcpy((char *)cb + NVC_CB_PARAMS, buf, size < kb->param_bytes ? size : kb->param_bytes);
	}
	memset(qmd, 0, qmd_size);
	nvc_fill_qmd(kb, f->code.va, cb_va, cb_size, grid, block, shared_bytes, qmd, qmd_size);
	start = p = (uint32_t *)((char *)qmd + ((qmd_size + 63) & ~63u));
	p = nvc_mthd1(p, SUBC_COMPUTE, C7C0_INVALIDATE_SHADER_CACHES_NO_WFI, INVALIDATE_INSTRUCTION_GLOBAL_CONSTANT);
	p = nvc_mthd1(p, SUBC_COMPUTE, C7C0_SEND_PCAS_A, (uint32_t)(qmd_va >> 8));
	p = nvc_mthd1(p, SUBC_COMPUTE, C7C0_SEND_SIGNALING_PCAS2_B, PCAS_ACTION_INVALIDATE_COPY_SCHEDULE);
	r = nvc_push_submit(ctx, s, ctx->gr_chan, start, p);
	pthread_mutex_unlock(&ctx->lock);
	return r;
}
