// SPDX-License-Identifier: MIT
/*
 * Contexts. Each context is its own DRM file on the render node: its own GPU VM (VM_INIT), a copy
 * channel (AMPERE_DMA_COPY_B), a push ring, a staging buffer for pageable host copies and a
 * page of event timestamps. The current context is a per-thread stack (cuCtxPush/PopCurrent).
 */
#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <xf86drm.h>
#include <drm/nouveau_drm.h>

#include "internal.h"

#define DRM_NOUVEAU_NVIF 0x07
struct nvif_hdr { uint8_t version, type, pad02[4], owner, route; uint64_t token, object; };
struct nvif_new { uint8_t version, pad01[6], route; uint64_t token, object; uint32_t handle; int32_t oclass; };

#define CTX_STACK 16
static __thread struct CUctx_st *ctx_stack[CTX_STACK];
static __thread int ctx_depth;
static pthread_mutex_t primary_lock = PTHREAD_MUTEX_INITIALIZER;

struct CUctx_st *nvc_current(void)
{
	return ctx_depth ? ctx_stack[ctx_depth - 1] : NULL;
}

CUresult nvc_ctx_get(struct CUctx_st **ctx)
{
	if (!nvc_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;
	*ctx = nvc_current();
	return *ctx ? CUDA_SUCCESS : CUDA_ERROR_INVALID_CONTEXT;
}

int nvc_bo_new(struct CUctx_st *ctx, struct nvc_bo *bo, uint64_t size, uint32_t domain, bool map, uint64_t va)
{
	uint64_t align = size >= (2u << 20) ? (2u << 20) : 0x10000;
	struct drm_nouveau_gem_new n = {
		.info = { .size = (size + align - 1) & ~(align - 1),
			  .domain = domain | (map ? NOUVEAU_GEM_DOMAIN_MAPPABLE : 0) },
		.align = (uint32_t)align,
	};
	struct drm_nouveau_vm_bind_op op = { .op = DRM_NOUVEAU_VM_BIND_OP_MAP };
	struct drm_nouveau_vm_bind b = { .op_count = 1, .op_ptr = (uintptr_t)&op };
	bool own_va = !va;
	int r;

	memset(bo, 0, sizeof(*bo));
	r = drmCommandWriteRead(ctx->fd, DRM_NOUVEAU_GEM_NEW, &n, sizeof(n));
	if (r)
		return r;
	bo->handle = n.info.handle;
	bo->size = n.info.size;
	if (own_va && !(va = nvc_va_get(bo->size, align))) {
		r = -ENOSPC;
		goto fail;
	}
	if (map) {
		bo->map = mmap(NULL, bo->size, PROT_READ | PROT_WRITE, MAP_SHARED, ctx->fd, n.info.map_handle);
		if (bo->map == MAP_FAILED) {
			bo->map = NULL;
			r = -errno;
			goto fail_va;
		}
	}
	op.handle = bo->handle;
	op.addr = va;
	op.range = bo->size;
	r = drmCommandWriteRead(ctx->fd, DRM_NOUVEAU_VM_BIND, &b, sizeof(b));
	if (r)
		goto fail_map;
	bo->va = va;
	return 0;
fail_map:
	if (bo->map)
		munmap(bo->map, bo->size);
fail_va:
	if (own_va)
		nvc_va_put(va, bo->size);
fail:
	drmCloseBufferHandle(ctx->fd, bo->handle);
	memset(bo, 0, sizeof(*bo));
	return r;
}

/* Unmap and free; the VA goes back to the allocator. */
void nvc_bo_free(struct CUctx_st *ctx, struct nvc_bo *bo)
{
	struct drm_nouveau_vm_bind_op op = { .op = DRM_NOUVEAU_VM_BIND_OP_UNMAP, .addr = bo->va, .range = bo->size };
	struct drm_nouveau_vm_bind b = { .op_count = 1, .op_ptr = (uintptr_t)&op };

	if (!bo->handle)
		return;
	if (bo->va) {
		drmCommandWriteRead(ctx->fd, DRM_NOUVEAU_VM_BIND, &b, sizeof(b));
		nvc_va_put(bo->va, bo->size);
	}
	if (bo->map)
		munmap(bo->map, bo->size);
	drmCloseBufferHandle(ctx->fd, bo->handle);
	memset(bo, 0, sizeof(*bo));
}

static int stream_init(struct CUctx_st *ctx, struct CUstream_st *s, unsigned int flags)
{
	memset(s, 0, sizeof(*s));
	s->ctx = ctx;
	s->flags = flags;
	return drmSyncobjCreate(ctx->fd, 0, &s->tl);
}

CUresult nvc_ctx_init(struct CUctx_st *ctx, CUdevice dev, unsigned int flags)
{
	struct drm_nouveau_vm_init vi = { NVC_KERNEL_VA, NVC_KERNEL_VA_SIZE };
	struct drm_nouveau_channel_alloc ca = { .fb_ctxdma_handle = ~0u, .tt_ctxdma_handle = NOUVEAU_FIFO_ENGINE_CE };
	struct { struct nvif_hdr h; struct nvif_new n; } on = { .h = { .type = 2, .route = 0xff }, .n = { .oclass = 0xc7b5 } };
	uint32_t *p, *start;
	CUresult res;

	memset(ctx, 0, sizeof(*ctx));
	ctx->dev = &nvc_devices[dev];
	ctx->ordinal = dev;
	ctx->flags = flags;
	ctx->limit_stack = 1024;
	ctx->limit_heap = 8u << 20;
	ctx->limit_printf = 1u << 20;
	pthread_mutex_init(&ctx->lock, NULL);
	ctx->fd = open(ctx->dev->path, O_RDWR | O_CLOEXEC);
	if (ctx->fd < 0)
		return CUDA_ERROR_INVALID_DEVICE;
	if (drmCommandWrite(ctx->fd, DRM_NOUVEAU_VM_INIT, &vi, sizeof(vi)) || nvc_nvif_device(ctx->fd) ||
	    drmCommandWriteRead(ctx->fd, DRM_NOUVEAU_CHANNEL_ALLOC, &ca, sizeof(ca))) {
		res = CUDA_ERROR_DEVICE_UNAVAILABLE;
		goto fail;
	}
	ctx->ce_chan = ca.channel;
	on.h.token = ca.channel;
	if (drmCommandWrite(ctx->fd, DRM_NOUVEAU_NVIF, &on, sizeof(on))) {
		res = CUDA_ERROR_DEVICE_UNAVAILABLE;
		goto fail;
	}
	if (nvc_bo_new(ctx, &ctx->push, NVC_PUSH_SIZE, NOUVEAU_GEM_DOMAIN_GART, true, 0) ||
	    nvc_bo_new(ctx, &ctx->staging, NVC_STAGING_SIZE, NOUVEAU_GEM_DOMAIN_GART, true, 0) ||
	    nvc_bo_new(ctx, &ctx->events, NVC_EVENT_SLOTS * 16, NOUVEAU_GEM_DOMAIN_GART, true, 0) ||
	    stream_init(ctx, &ctx->null_stream, 0)) {
		res = CUDA_ERROR_OUT_OF_MEMORY;
		goto fail;
	}
	memset(ctx->events.map, 0, ctx->events.size);
	/* bind the copy class to its subchannel once */
	pthread_mutex_lock(&ctx->lock);
	start = p = nvc_push_begin(ctx, 2);
	p = nvc_mthd1(p, NVC_SUBC_CE, 0x0000, 0xc7b5);
	res = nvc_push_submit(ctx, &ctx->null_stream, ctx->ce_chan, start, p);
	pthread_mutex_unlock(&ctx->lock);
	if (res == CUDA_SUCCESS)
		res = nvc_stream_wait_idle(&ctx->null_stream);
	if (res != CUDA_SUCCESS)
		goto fail;
	return CUDA_SUCCESS;
fail:
	nvc_ctx_fini(ctx);
	return res;
}

void nvc_ctx_fini(struct CUctx_st *ctx)
{
	int i;

	if (ctx->fd < 0)
		return;
	if (ctx->null_stream.tl)
		nvc_ctx_idle(ctx);
	nvc_modules_drop(ctx);
	nvc_alloc_drop_ctx(ctx);	/* allocations the program did not free */
	nvc_bo_free(ctx, &ctx->slm);
	for (i = 0; i < ctx->nstreams; i++) {
		drmSyncobjDestroy(ctx->fd, ctx->streams[i]->tl);
		free(ctx->streams[i]);
	}
	free(ctx->streams);
	if (ctx->null_stream.tl)
		drmSyncobjDestroy(ctx->fd, ctx->null_stream.tl);
	nvc_bo_free(ctx, &ctx->events);
	nvc_bo_free(ctx, &ctx->staging);
	nvc_bo_free(ctx, &ctx->push);
	close(ctx->fd);	/* channels and the VM go with the file */
	ctx->fd = -1;
	pthread_mutex_destroy(&ctx->lock);
}

static void ctx_unstack(struct CUctx_st *ctx)
{
	int i, j;

	for (i = j = 0; i < ctx_depth; i++)
		if (ctx_stack[i] != ctx)
			ctx_stack[j++] = ctx_stack[i];
	ctx_depth = j;
}

CUAPI CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev)
{
	struct CUctx_st *ctx;
	CUresult res;

	if (!nvc_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (!pctx)
		return CUDA_ERROR_INVALID_VALUE;
	if (dev < 0 || dev >= nvc_ndevices)
		return CUDA_ERROR_INVALID_DEVICE;
	if (ctx_depth == CTX_STACK)
		return CUDA_ERROR_OUT_OF_MEMORY;
	ctx = calloc(1, sizeof(*ctx));
	if (!ctx)
		return CUDA_ERROR_OUT_OF_MEMORY;
	res = nvc_ctx_init(ctx, dev, flags);
	if (res != CUDA_SUCCESS) {
		free(ctx);
		return res;
	}
	ctx_stack[ctx_depth++] = ctx;
	*pctx = ctx;
	return CUDA_SUCCESS;
}

CUAPI CUresult cuCtxDestroy_v2(CUcontext ctx)
{
	if (!ctx)
		return CUDA_ERROR_INVALID_VALUE;
	if (ctx == ctx->dev->primary)
		return CUDA_ERROR_INVALID_CONTEXT;	/* primary contexts go through Release */
	ctx_unstack(ctx);
	nvc_ctx_fini(ctx);
	free(ctx);
	return CUDA_SUCCESS;
}

CUAPI CUresult cuCtxSetCurrent(CUcontext ctx)
{
	if (!nvc_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (!ctx) {
		if (ctx_depth)
			ctx_depth--;
		return CUDA_SUCCESS;
	}
	if (ctx_depth)
		ctx_stack[ctx_depth - 1] = ctx;
	else
		ctx_stack[ctx_depth++] = ctx;
	return CUDA_SUCCESS;
}

CUAPI CUresult cuCtxGetCurrent(CUcontext *ctx)
{
	if (!nvc_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (!ctx)
		return CUDA_ERROR_INVALID_VALUE;
	*ctx = nvc_current();
	return CUDA_SUCCESS;
}

CUAPI CUresult cuCtxPushCurrent_v2(CUcontext ctx)
{
	if (!nvc_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (!ctx)
		return CUDA_ERROR_INVALID_CONTEXT;
	if (ctx_depth == CTX_STACK)
		return CUDA_ERROR_OUT_OF_MEMORY;
	ctx_stack[ctx_depth++] = ctx;
	return CUDA_SUCCESS;
}

CUAPI CUresult cuCtxPopCurrent_v2(CUcontext *ctx)
{
	if (!nvc_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (!ctx_depth)
		return CUDA_ERROR_INVALID_CONTEXT;
	ctx_depth--;
	if (ctx)
		*ctx = ctx_stack[ctx_depth];
	return CUDA_SUCCESS;
}

CUAPI CUresult cuCtxGetDevice(CUdevice *device)
{
	struct CUctx_st *ctx;
	CUresult r = nvc_ctx_get(&ctx);

	if (r)
		return r;
	if (!device)
		return CUDA_ERROR_INVALID_VALUE;
	*device = ctx->ordinal;
	return CUDA_SUCCESS;
}

CUAPI CUresult cuCtxSynchronize(void)
{
	struct CUctx_st *ctx;
	CUresult r = nvc_ctx_get(&ctx);

	return r ? r : nvc_ctx_idle(ctx);
}

CUAPI CUresult cuCtxGetApiVersion(CUcontext ctx, unsigned int *version)
{
	if (!version)
		return CUDA_ERROR_INVALID_VALUE;
	(void)ctx;
	*version = 3020;	/* the _v2 context API */
	return CUDA_SUCCESS;
}

CUAPI CUresult cuCtxGetLimit(size_t *value, CUlimit limit)
{
	struct CUctx_st *ctx;
	CUresult r = nvc_ctx_get(&ctx);

	if (r)
		return r;
	if (!value)
		return CUDA_ERROR_INVALID_VALUE;
	switch (limit) {
	case CU_LIMIT_STACK_SIZE:		*value = ctx->limit_stack; break;
	case CU_LIMIT_PRINTF_FIFO_SIZE:		*value = ctx->limit_printf; break;
	case CU_LIMIT_MALLOC_HEAP_SIZE:		*value = ctx->limit_heap; break;
	default:				return CUDA_ERROR_UNSUPPORTED_LIMIT;
	}
	return CUDA_SUCCESS;
}

CUAPI CUresult cuCtxSetLimit(CUlimit limit, size_t value)
{
	struct CUctx_st *ctx;
	CUresult r = nvc_ctx_get(&ctx);

	if (r)
		return r;
	switch (limit) {
	case CU_LIMIT_STACK_SIZE:		ctx->limit_stack = value; break;
	case CU_LIMIT_PRINTF_FIFO_SIZE:		ctx->limit_printf = value; break;
	case CU_LIMIT_MALLOC_HEAP_SIZE:		ctx->limit_heap = value; break;
	default:				return CUDA_ERROR_UNSUPPORTED_LIMIT;
	}
	return CUDA_SUCCESS;
}

CUAPI CUresult cuCtxGetCacheConfig(CUfunc_cache *config)
{
	struct CUctx_st *ctx;
	CUresult r = nvc_ctx_get(&ctx);

	if (r)
		return r;
	if (!config)
		return CUDA_ERROR_INVALID_VALUE;
	*config = ctx->cache_config;
	return CUDA_SUCCESS;
}

CUAPI CUresult cuCtxSetCacheConfig(CUfunc_cache config)
{
	struct CUctx_st *ctx;
	CUresult r = nvc_ctx_get(&ctx);

	if (r)
		return r;
	ctx->cache_config = config;	/* a preference; the L1/shared split is chosen per launch */
	return CUDA_SUCCESS;
}

/* ---- primary contexts: one per device, reference counted ---- */

CUAPI CUresult cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev)
{
	struct nvc_device *d;
	CUresult res = CUDA_SUCCESS;

	if (!nvc_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (!pctx)
		return CUDA_ERROR_INVALID_VALUE;
	if (dev < 0 || dev >= nvc_ndevices)
		return CUDA_ERROR_INVALID_DEVICE;
	d = &nvc_devices[dev];
	pthread_mutex_lock(&primary_lock);
	if (!d->primary) {
		struct CUctx_st *ctx = calloc(1, sizeof(*ctx));

		res = ctx ? nvc_ctx_init(ctx, dev, d->primary_flags) : CUDA_ERROR_OUT_OF_MEMORY;
		if (res == CUDA_SUCCESS)
			d->primary = ctx;
		else
			free(ctx);
	}
	if (res == CUDA_SUCCESS) {
		d->primary_refs++;
		*pctx = d->primary;
	}
	pthread_mutex_unlock(&primary_lock);
	return res;
}

CUAPI CUresult cuDevicePrimaryCtxRelease_v2(CUdevice dev)
{
	struct nvc_device *d;
	struct CUctx_st *gone = NULL;

	if (!nvc_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (dev < 0 || dev >= nvc_ndevices)
		return CUDA_ERROR_INVALID_DEVICE;
	d = &nvc_devices[dev];
	pthread_mutex_lock(&primary_lock);
	if (!d->primary_refs) {
		pthread_mutex_unlock(&primary_lock);
		return CUDA_ERROR_INVALID_CONTEXT;
	}
	if (!--d->primary_refs) {
		gone = d->primary;
		d->primary = NULL;
	}
	pthread_mutex_unlock(&primary_lock);
	if (gone) {
		ctx_unstack(gone);
		nvc_ctx_fini(gone);
		free(gone);
	}
	return CUDA_SUCCESS;
}

CUAPI CUresult cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int *flags, int *active)
{
	if (!nvc_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (dev < 0 || dev >= nvc_ndevices)
		return CUDA_ERROR_INVALID_DEVICE;
	if (!flags || !active)
		return CUDA_ERROR_INVALID_VALUE;
	pthread_mutex_lock(&primary_lock);
	*flags = nvc_devices[dev].primary_flags;
	*active = nvc_devices[dev].primary_refs > 0;
	pthread_mutex_unlock(&primary_lock);
	return CUDA_SUCCESS;
}

CUAPI CUresult cuDevicePrimaryCtxSetFlags_v2(CUdevice dev, unsigned int flags)
{
	if (!nvc_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (dev < 0 || dev >= nvc_ndevices)
		return CUDA_ERROR_INVALID_DEVICE;
	pthread_mutex_lock(&primary_lock);
	nvc_devices[dev].primary_flags = flags;
	pthread_mutex_unlock(&primary_lock);
	return CUDA_SUCCESS;
}

CUAPI CUresult cuDevicePrimaryCtxReset_v2(CUdevice dev)
{
	struct nvc_device *d;
	struct CUctx_st *gone;

	if (!nvc_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (dev < 0 || dev >= nvc_ndevices)
		return CUDA_ERROR_INVALID_DEVICE;
	d = &nvc_devices[dev];
	pthread_mutex_lock(&primary_lock);
	gone = d->primary;
	d->primary = NULL;
	d->primary_refs = 0;
	pthread_mutex_unlock(&primary_lock);
	if (gone) {
		ctx_unstack(gone);
		nvc_ctx_fini(gone);
		free(gone);
	}
	return CUDA_SUCCESS;
}
