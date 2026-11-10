// SPDX-License-Identifier: MIT
/*
 * Device memory (VRAM GEM objects at a unified-address VA), pinned host memory (GART objects
 * mapped into both the process and the GPU VM), and copies/memsets on the copy engine
 * (AMPERE_DMA_COPY_B, method layout from NVIDIA's open-gpu-doc clc7b5.h).
 *
 * Pageable host memory goes through the context's staging buffer in chunks (two halves, so the
 * CPU fills one while the copy engine drains the other). Copies from or to pinned memory go
 * directly, and are truly asynchronous in the *Async variants; with pageable memory the Async
 * variants complete before returning, as CUDA allows.
 */
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <xf86drm.h>
#include <drm/nouveau_drm.h>

#include "internal.h"

#define C7B5_LAUNCH_DMA			0x300
#define C7B5_OFFSET_IN_UPPER		0x400
#define C7B5_OFFSET_OUT_UPPER		0x408
#define C7B5_PITCH_IN			0x410
#define C7B5_LINE_LENGTH_IN		0x418
#define C7B5_LINE_COUNT			0x41c
#define LAUNCH_MULTI_LINE		(1u << 9)
#define C7B5_SET_REMAP_CONST_A		0x700
#define C7B5_SET_REMAP_COMPONENTS	0x708
/* non-pipelined, flush, pitch -> pitch, virtual addresses, no semaphore (EXEC signals) */
#define LAUNCH_COPY			0x186
#define LAUNCH_REMAP			(1u << 10)
#define REMAP_DST_X_CONST_A		0x4
#define CE_CHUNK			(1ull << 30)	/* LINE_LENGTH_IN is 32 bits */

/* ---- allocation ---- */

static CUresult alloc_bo(struct CUctx_st *ctx, enum nvc_alloc_kind kind, size_t bytes, struct nvc_alloc **out)
{
	struct nvc_alloc *a = calloc(1, sizeof(*a));
	int r;

	if (!a)
		return CUDA_ERROR_OUT_OF_MEMORY;
	a->kind = kind;
	a->ctx = ctx;
	r = nvc_bo_new(ctx, &a->bo, bytes, kind == NVC_ALLOC_DEVICE ? NOUVEAU_GEM_DOMAIN_VRAM : NOUVEAU_GEM_DOMAIN_GART,
		       kind == NVC_ALLOC_HOST, 0);
	if (r) {
		free(a);
		return r == -ENOMEM || r == -ENOSPC ? CUDA_ERROR_OUT_OF_MEMORY : CUDA_ERROR_UNKNOWN;
	}
	nvc_alloc_add(a);
	*out = a;
	return CUDA_SUCCESS;
}

/* a device allocation for nvcuda's own use (module globals): copies work on it like on cuMemAlloc memory */
CUresult nvc_mem_alloc_device(struct CUctx_st *ctx, size_t bytes, uint64_t *va)
{
	struct nvc_alloc *a;
	CUresult r = alloc_bo(ctx, NVC_ALLOC_DEVICE, bytes, &a);

	if (r == CUDA_SUCCESS)
		*va = a->bo.va;
	return r;
}

void nvc_mem_free_device(struct CUctx_st *ctx, uint64_t va)
{
	struct nvc_alloc *a = va ? nvc_alloc_remove_va(va) : NULL;

	if (a) {
		nvc_bo_free(ctx, &a->bo);
		free(a);
	}
}

CUAPI CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytes)
{
	struct CUctx_st *ctx;
	struct nvc_alloc *a;
	CUresult r = nvc_ctx_get(&ctx);

	if (r)
		return r;
	if (!dptr || !bytes)
		return CUDA_ERROR_INVALID_VALUE;
	r = alloc_bo(ctx, NVC_ALLOC_DEVICE, bytes, &a);
	if (r == CUDA_SUCCESS)
		*dptr = a->bo.va;
	return r;
}

CUAPI CUresult cuMemFree_v2(CUdeviceptr dptr)
{
	struct nvc_alloc *a;

	if (!nvc_initialized)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (!dptr)
		return CUDA_SUCCESS;
	a = nvc_alloc_remove_va(dptr);
	if (!a)
		return CUDA_ERROR_INVALID_VALUE;
	/* the GPU may still use it: cuMemFree synchronizes the device's work first */
	nvc_ctx_idle(a->ctx);
	nvc_bo_free(a->ctx, &a->bo);
	free(a);
	return CUDA_SUCCESS;
}

CUAPI CUresult cuMemGetAddressRange_v2(CUdeviceptr *base, size_t *size, CUdeviceptr dptr)
{
	struct nvc_alloc *a = nvc_alloc_find_va(dptr, 0);

	if (!a || dptr == a->bo.va + a->bo.size)
		return CUDA_ERROR_NOT_FOUND;
	if (base)
		*base = a->bo.va;
	if (size)
		*size = a->bo.size;
	return CUDA_SUCCESS;
}

CUAPI CUresult cuMemGetInfo_v2(size_t *free_bytes, size_t *total)
{
	struct drm_nouveau_getparam gp = { .param = NOUVEAU_GETPARAM_VRAM_USED };
	struct CUctx_st *ctx;
	CUresult r = nvc_ctx_get(&ctx);
	uint64_t t;

	if (r)
		return r;
	t = ctx->dev->vram_user;
	if (drmCommandWriteRead(ctx->fd, DRM_NOUVEAU_GETPARAM, &gp, sizeof(gp)))
		gp.value = 0;
	if (free_bytes)
		*free_bytes = gp.value < t ? t - gp.value : 0;
	if (total)
		*total = t;
	return CUDA_SUCCESS;
}

CUAPI CUresult cuMemHostAlloc(void **pp, size_t bytes, unsigned int flags)
{
	struct CUctx_st *ctx;
	struct nvc_alloc *a;
	CUresult r = nvc_ctx_get(&ctx);

	if (r)
		return r;
	if (!pp || !bytes || flags & ~0x7u)
		return CUDA_ERROR_INVALID_VALUE;
	r = alloc_bo(ctx, NVC_ALLOC_HOST, bytes, &a);
	if (r == CUDA_SUCCESS)
		*pp = a->bo.map;
	return r;
}

CUAPI CUresult cuMemAllocHost_v2(void **pp, size_t bytes)
{
	return cuMemHostAlloc(pp, bytes, 0);
}

CUAPI CUresult cuMemFreeHost(void *p)
{
	struct nvc_alloc *a = nvc_alloc_remove_host(p);

	if (!a)
		return CUDA_ERROR_INVALID_VALUE;
	nvc_ctx_idle(a->ctx);
	nvc_bo_free(a->ctx, &a->bo);
	free(a);
	return CUDA_SUCCESS;
}

CUAPI CUresult cuMemHostGetDevicePointer_v2(CUdeviceptr *dptr, void *p, unsigned int flags)
{
	struct nvc_alloc *a;

	if (!dptr || flags)
		return CUDA_ERROR_INVALID_VALUE;
	a = nvc_alloc_find_host(p, 0);
	if (!a)
		return CUDA_ERROR_INVALID_VALUE;
	*dptr = a->bo.va + ((uintptr_t)p - (uintptr_t)a->bo.map);
	return CUDA_SUCCESS;
}

/* ---- copy engine ---- */

/* Queue dst <- src (GPU VAs) on stream s. Caller holds ctx->lock. */
static CUresult ce_copy(struct CUctx_st *ctx, struct CUstream_st *s, uint64_t dst, uint64_t src, uint64_t bytes)
{
	while (bytes) {
		uint64_t n = bytes < CE_CHUNK ? bytes : CE_CHUNK;
		uint32_t io[4] = { (uint32_t)(src >> 32), (uint32_t)src, (uint32_t)(dst >> 32), (uint32_t)dst };
		uint32_t *start = nvc_push_begin(ctx, 9), *p = start;
		CUresult r;

		p = nvc_mthd(p, NVC_SUBC_CE, C7B5_OFFSET_IN_UPPER, 4, io);
		p = nvc_mthd1(p, NVC_SUBC_CE, C7B5_LINE_LENGTH_IN, (uint32_t)n);
		p = nvc_mthd1(p, NVC_SUBC_CE, C7B5_LAUNCH_DMA, LAUNCH_COPY);
		r = nvc_push_submit(ctx, s, ctx->ce_chan, start, p);
		if (r)
			return r;
		dst += n;
		src += n;
		bytes -= n;
	}
	return CUDA_SUCCESS;
}

/* lines rows of width bytes: dst (pitch dp) <- src (pitch sp), GPU VAs. Caller holds ctx->lock. */
static CUresult ce_copy2d(struct CUctx_st *ctx, struct CUstream_st *s, uint64_t dst, uint32_t dp, uint64_t src,
			  uint32_t sp, uint32_t width, uint32_t lines)
{
	uint32_t io[4] = { (uint32_t)(src >> 32), (uint32_t)src, (uint32_t)(dst >> 32), (uint32_t)dst };
	uint32_t pitches[2] = { sp, dp };
	uint32_t *start = nvc_push_begin(ctx, 12), *p = start;

	p = nvc_mthd(p, NVC_SUBC_CE, C7B5_OFFSET_IN_UPPER, 4, io);
	p = nvc_mthd(p, NVC_SUBC_CE, C7B5_PITCH_IN, 2, pitches);
	p = nvc_mthd1(p, NVC_SUBC_CE, C7B5_LINE_LENGTH_IN, width);
	p = nvc_mthd1(p, NVC_SUBC_CE, C7B5_LINE_COUNT, lines);
	p = nvc_mthd1(p, NVC_SUBC_CE, C7B5_LAUNCH_DMA, LAUNCH_COPY | LAUNCH_MULTI_LINE);
	return nvc_push_submit(ctx, s, ctx->ce_chan, start, p);
}

/* dst <- value, count elements of 1, 2 or 4 bytes (remap: DST_X = CONST_A). */
static CUresult ce_fill(struct CUctx_st *ctx, struct CUstream_st *s, uint64_t dst, uint32_t value, int esize,
			uint64_t count)
{
	static const uint32_t comp_size[5] = { 0, 0, 1, 2, 3 };

	while (count) {
		uint64_t n = count < CE_CHUNK / 4 ? count : CE_CHUNK / 4;
		uint32_t out[2] = { (uint32_t)(dst >> 32), (uint32_t)dst };
		uint32_t *start = nvc_push_begin(ctx, 10), *p = start;
		CUresult r;

		p = nvc_mthd1(p, NVC_SUBC_CE, C7B5_SET_REMAP_CONST_A, value);
		p = nvc_mthd1(p, NVC_SUBC_CE, C7B5_SET_REMAP_COMPONENTS, REMAP_DST_X_CONST_A | comp_size[esize] << 16);
		p = nvc_mthd(p, NVC_SUBC_CE, C7B5_OFFSET_OUT_UPPER, 2, out);
		p = nvc_mthd1(p, NVC_SUBC_CE, C7B5_LINE_LENGTH_IN, (uint32_t)n);
		p = nvc_mthd1(p, NVC_SUBC_CE, C7B5_LAUNCH_DMA, LAUNCH_COPY | LAUNCH_REMAP);
		r = nvc_push_submit(ctx, s, ctx->ce_chan, start, p);
		if (r)
			return r;
		dst += n * esize;
		count -= n;
	}
	return CUDA_SUCCESS;
}

/* A device pointer range: a device allocation or the GPU side of pinned memory. */
static struct CUctx_st *dev_range(CUdeviceptr p, size_t bytes)
{
	struct nvc_alloc *a = nvc_alloc_find_va(p, bytes);

	return a ? a->ctx : NULL;
}

/* GPU VA of a pinned host range, or 0 for pageable memory */
static uint64_t pinned_va(struct CUctx_st *ctx, const void *p, size_t bytes)
{
	struct nvc_alloc *a = nvc_alloc_find_host(p, bytes);

	return a && a->ctx == ctx ? a->bo.va + ((uintptr_t)p - (uintptr_t)a->bo.map) : 0;
}

/* pageable host memory -> GPU VA through the staging halves; returns with the copy done */
static CUresult htod_staged(struct CUctx_st *ctx, struct CUstream_st *s, uint64_t dst, const void *src, size_t bytes)
{
	uint64_t half = NVC_STAGING_SIZE / 2, done = 0, half_point[2] = { 0, 0 };
	CUresult r = CUDA_SUCCESS;
	int h = 0;

	while (done < bytes && r == CUDA_SUCCESS) {
		uint64_t n = bytes - done < half ? bytes - done : half;

		if (half_point[h])
			nvc_stream_wait_idle(&(struct CUstream_st){ .ctx = ctx, .tl = s->tl, .point = half_point[h] });
		memcpy((char *)ctx->staging.map + h * half, (const char *)src + done, n);
		pthread_mutex_lock(&ctx->lock);
		r = ce_copy(ctx, s, dst + done, ctx->staging.va + h * half, n);
		half_point[h] = s->point;
		pthread_mutex_unlock(&ctx->lock);
		done += n;
		h ^= 1;
	}
	return r ? r : nvc_stream_wait_idle(s);
}

CUresult nvc_upload(struct CUctx_st *ctx, uint64_t va, const void *data, size_t bytes)
{
	return htod_staged(ctx, &ctx->null_stream, va, data, bytes);
}

static CUresult memcpy_htod(CUdeviceptr dst, const void *src, size_t bytes, CUstream stream, bool async)
{
	struct CUctx_st *ctx;
	struct CUstream_st *s;
	uint64_t pin;
	CUresult r;

	if (!bytes)
		return CUDA_SUCCESS;
	if (!src || !(ctx = dev_range(dst, bytes)))
		return CUDA_ERROR_INVALID_VALUE;
	s = nvc_stream(ctx, stream);
	if (!s)
		return CUDA_ERROR_INVALID_HANDLE;
	pin = pinned_va(ctx, src, bytes);
	if (pin) {
		pthread_mutex_lock(&ctx->lock);
		r = ce_copy(ctx, s, dst, pin, bytes);
		pthread_mutex_unlock(&ctx->lock);
		return r || async ? r : nvc_stream_wait_idle(s);
	}
	return htod_staged(ctx, s, dst, src, bytes);
}

static CUresult memcpy_dtoh_(void *dst, CUdeviceptr src, size_t bytes, CUstream stream, bool async);

/*
 * NVCUDA_TRACE=1: small readbacks are printed (the values a program's loop decides on); an
 * asynchronous one is printed when the next synchronisation completes (nvc_trace_synced).
 */
static struct {
	const void *dst;
	size_t bytes;
	uint64_t src;
} trace_pending;

static void trace_values(const void *dst, size_t bytes)
{
	size_t i;

	fprintf(stderr, " =");
	for (i = 0; i + 4 <= bytes; i += 4)
		fprintf(stderr, " %u", ((const uint32_t *)dst)[i / 4]);
	fprintf(stderr, "\n");
}

void nvc_trace_synced(void)
{
	if (!trace_pending.dst)
		return;
	fprintf(stderr, "nvcuda: synced: DtoH %zu bytes from 0x%llx", trace_pending.bytes,
		(unsigned long long)trace_pending.src);
	trace_values(trace_pending.dst, trace_pending.bytes);
	trace_pending.dst = NULL;
}

static CUresult memcpy_dtoh(void *dst, CUdeviceptr src, size_t bytes, CUstream stream, bool async)
{
	CUresult r = memcpy_dtoh_(dst, src, bytes, stream, async);

	if (getenv("NVCUDA_TRACE")) {
		fprintf(stderr, "nvcuda: DtoH %zu bytes from 0x%llx%s -> %d", bytes, (unsigned long long)src,
			async ? " (async)" : "", r);
		if (!r && bytes <= 128 && !async)
			trace_values(dst, bytes);
		else
			fprintf(stderr, "\n");
		if (!r && bytes <= 128 && async) {
			trace_pending.dst = dst;
			trace_pending.bytes = bytes;
			trace_pending.src = src;
		}
	}
	return r;
}

static CUresult memcpy_dtoh_(void *dst, CUdeviceptr src, size_t bytes, CUstream stream, bool async)
{
	struct CUctx_st *ctx;
	struct CUstream_st *s;
	uint64_t pin, half = NVC_STAGING_SIZE / 2, done = 0, pending = 0;
	CUresult r = CUDA_SUCCESS;
	int h = 0;

	if (!bytes)
		return CUDA_SUCCESS;
	if (!dst || !(ctx = dev_range(src, bytes)))
		return CUDA_ERROR_INVALID_VALUE;
	s = nvc_stream(ctx, stream);
	if (!s)
		return CUDA_ERROR_INVALID_HANDLE;
	pin = pinned_va(ctx, dst, bytes);
	if (pin) {
		pthread_mutex_lock(&ctx->lock);
		r = ce_copy(ctx, s, pin, src, bytes);
		pthread_mutex_unlock(&ctx->lock);
		return r || async ? r : nvc_stream_wait_idle(s);
	}
	/* pageable: copy chunk k+1 into one half while the CPU reads chunk k out of the other */
	while (r == CUDA_SUCCESS && (done < bytes || pending)) {
		uint64_t n = bytes - done < half ? bytes - done : half, point = 0;

		if (n) {
			pthread_mutex_lock(&ctx->lock);
			r = ce_copy(ctx, s, ctx->staging.va + h * half, src + done, n);
			point = s->point;
			pthread_mutex_unlock(&ctx->lock);
		}
		if (pending) {		/* the previous chunk, in the other half */
			uint64_t prev = done - pending;

			memcpy((char *)dst + prev, (char *)ctx->staging.map + (h ^ 1) * half, pending);
		}
		pending = 0;
		if (n && r == CUDA_SUCCESS) {
			r = nvc_stream_wait_idle(&(struct CUstream_st){ .ctx = ctx, .tl = s->tl, .point = point });
			done += n;
			pending = n;
			h ^= 1;
		}
	}
	return r;
}

static CUresult memcpy_dtod(CUdeviceptr dst, CUdeviceptr src, size_t bytes, CUstream stream, bool async)
{
	struct CUctx_st *ctx, *ctx2;
	struct CUstream_st *s;
	CUresult r;

	if (!bytes)
		return CUDA_SUCCESS;
	ctx = dev_range(dst, bytes);
	ctx2 = dev_range(src, bytes);
	if (!ctx || ctx != ctx2)
		return CUDA_ERROR_INVALID_VALUE;	/* cross-context copies: not yet */
	s = nvc_stream(ctx, stream);
	if (!s)
		return CUDA_ERROR_INVALID_HANDLE;
	pthread_mutex_lock(&ctx->lock);
	r = ce_copy(ctx, s, dst, src, bytes);
	pthread_mutex_unlock(&ctx->lock);
	return r || async ? r : nvc_stream_wait_idle(s);
}

static CUresult memset_d(CUdeviceptr dst, uint32_t value, int esize, size_t count, CUstream stream, bool async)
{
	struct CUctx_st *ctx;
	struct CUstream_st *s;
	CUresult r;

	if (!count)
		return CUDA_SUCCESS;
	if (dst % esize || !(ctx = dev_range(dst, count * esize)))
		return CUDA_ERROR_INVALID_VALUE;
	s = nvc_stream(ctx, stream);
	if (!s)
		return CUDA_ERROR_INVALID_HANDLE;
	/* bytes and halves as words where possible: 4-byte components are the fast path */
	if (esize < 4 && !(dst & 3) && !((count * esize) & 3)) {
		value = esize == 1 ? (value & 0xff) * 0x01010101u : (value & 0xffff) * 0x00010001u;
		count = count * esize / 4;
		esize = 4;
	}
	pthread_mutex_lock(&ctx->lock);
	r = ce_fill(ctx, s, dst, value, esize, count);
	pthread_mutex_unlock(&ctx->lock);
	return r || async ? r : nvc_stream_wait_idle(s);
}

CUAPI CUresult cuMemcpyHtoD_v2(CUdeviceptr dst, const void *src, size_t bytes)
{
	return memcpy_htod(dst, src, bytes, NULL, false);
}

CUAPI CUresult cuMemcpyDtoH_v2(void *dst, CUdeviceptr src, size_t bytes)
{
	return memcpy_dtoh(dst, src, bytes, NULL, false);
}

CUAPI CUresult cuMemcpyDtoD_v2(CUdeviceptr dst, CUdeviceptr src, size_t bytes)
{
	return memcpy_dtod(dst, src, bytes, NULL, false);
}

CUAPI CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr dst, const void *src, size_t bytes, CUstream stream)
{
	return memcpy_htod(dst, src, bytes, stream, true);
}

CUAPI CUresult cuMemcpyDtoHAsync_v2(void *dst, CUdeviceptr src, size_t bytes, CUstream stream)
{
	return memcpy_dtoh(dst, src, bytes, stream, true);
}

CUAPI CUresult cuMemcpyDtoDAsync_v2(CUdeviceptr dst, CUdeviceptr src, size_t bytes, CUstream stream)
{
	return memcpy_dtod(dst, src, bytes, stream, true);
}

CUAPI CUresult cuMemsetD8_v2(CUdeviceptr dst, unsigned char value, size_t count)
{
	return memset_d(dst, value, 1, count, NULL, false);
}

CUAPI CUresult cuMemsetD16_v2(CUdeviceptr dst, unsigned short value, size_t count)
{
	return memset_d(dst, value, 2, count, NULL, false);
}

CUAPI CUresult cuMemsetD32_v2(CUdeviceptr dst, unsigned int value, size_t count)
{
	return memset_d(dst, value, 4, count, NULL, false);
}

CUAPI CUresult cuMemsetD8Async(CUdeviceptr dst, unsigned char value, size_t count, CUstream stream)
{
	return memset_d(dst, value, 1, count, stream, true);
}

CUAPI CUresult cuMemsetD32Async(CUdeviceptr dst, unsigned int value, size_t count, CUstream stream)
{
	return memset_d(dst, value, 4, count, stream, true);
}

CUAPI CUresult cuMemsetD16Async(CUdeviceptr dst, unsigned short value, size_t count, CUstream stream);
CUAPI CUresult cuMemsetD16Async(CUdeviceptr dst, unsigned short value, size_t count, CUstream stream)
{
	return memset_d(dst, value, 2, count, stream, true);
}

/* ---- 2D copies (CUDA_MEMCPY2D): rows of WidthInBytes, pitched on either side ---- */

static CUresult memcpy2d(const CUDA_MEMCPY2D *m, CUstream stream, bool async)
{
	struct CUctx_st *ctx = NULL;
	struct CUstream_st *s;
	uint64_t dsrc = 0, ddst = 0, half = NVC_STAGING_SIZE / 2;
	const char *hsrc = NULL;
	char *hdst = NULL;
	uint32_t w, h, sp, dp, y, rows;
	CUresult r = CUDA_SUCCESS;
	int hh = 0;

	if (!m)
		return CUDA_ERROR_INVALID_VALUE;
	w = (uint32_t)m->WidthInBytes;
	h = (uint32_t)m->Height;
	sp = (uint32_t)m->srcPitch;
	dp = (uint32_t)m->dstPitch;
	if (!w || !h)
		return CUDA_SUCCESS;
	if (m->srcMemoryType == CU_MEMORYTYPE_ARRAY || m->dstMemoryType == CU_MEMORYTYPE_ARRAY)
		return CUDA_ERROR_NOT_SUPPORTED;
	if (sp < w || dp < w)
		return CUDA_ERROR_INVALID_VALUE;
	/* sides: a device pointer (device or unified memory) or a host pointer (pinned or pageable) */
	if (m->srcMemoryType == CU_MEMORYTYPE_HOST || (m->srcMemoryType == CU_MEMORYTYPE_UNIFIED && m->srcHost &&
						      !nvc_alloc_find_va((uint64_t)(uintptr_t)m->srcHost, 0)))
		hsrc = (const char *)m->srcHost + m->srcY * sp + m->srcXInBytes;
	else
		dsrc = (m->srcMemoryType == CU_MEMORYTYPE_UNIFIED ? (uint64_t)(uintptr_t)m->srcHost : m->srcDevice) +
		       m->srcY * sp + m->srcXInBytes;
	if (m->dstMemoryType == CU_MEMORYTYPE_HOST || (m->dstMemoryType == CU_MEMORYTYPE_UNIFIED && m->dstHost &&
						      !nvc_alloc_find_va((uint64_t)(uintptr_t)m->dstHost, 0)))
		hdst = (char *)m->dstHost + m->dstY * dp + m->dstXInBytes;
	else
		ddst = (m->dstMemoryType == CU_MEMORYTYPE_UNIFIED ? (uint64_t)(uintptr_t)m->dstHost : m->dstDevice) +
		       m->dstY * dp + m->dstXInBytes;
	if (hsrc && hdst) {
		for (y = 0; y < h; y++)
			memcpy(hdst + (size_t)y * dp, hsrc + (size_t)y * sp, w);
		return CUDA_SUCCESS;
	}
	if (dsrc && !(ctx = dev_range(dsrc, (uint64_t)(h - 1) * sp + w)))
		return CUDA_ERROR_INVALID_VALUE;
	if (ddst && !(ctx = dev_range(ddst, (uint64_t)(h - 1) * dp + w)))
		return CUDA_ERROR_INVALID_VALUE;
	s = nvc_stream(ctx, stream);
	if (!s)
		return CUDA_ERROR_INVALID_HANDLE;
	if (hsrc) {
		uint64_t pin = pinned_va(ctx, hsrc, (uint64_t)(h - 1) * sp + w);

		if (pin) {
			pthread_mutex_lock(&ctx->lock);
			r = ce_copy2d(ctx, s, ddst, dp, pin, sp, w, h);
			pthread_mutex_unlock(&ctx->lock);
			return r || async ? r : nvc_stream_wait_idle(s);
		}
		if (w > half)
			return CUDA_ERROR_INVALID_VALUE;	/* rows wider than the staging half */
		rows = (uint32_t)(half / w);
		{
			uint64_t half_point[2] = { 0, 0 };

			for (y = 0; y < h && r == CUDA_SUCCESS; y += rows) {
				uint32_t n = h - y < rows ? h - y : rows, i;
				char *stage = (char *)ctx->staging.map + hh * half;

				if (half_point[hh])
					nvc_stream_wait_idle(&(struct CUstream_st){ .ctx = ctx, .tl = s->tl, .point = half_point[hh] });
				for (i = 0; i < n; i++)
					memcpy(stage + (size_t)i * w, hsrc + (size_t)(y + i) * sp, w);
				pthread_mutex_lock(&ctx->lock);
				r = ce_copy2d(ctx, s, ddst + (uint64_t)y * dp, dp, ctx->staging.va + hh * half, w, w, n);
				half_point[hh] = s->point;
				pthread_mutex_unlock(&ctx->lock);
				hh ^= 1;
			}
		}
		return r ? r : nvc_stream_wait_idle(s);
	}
	if (hdst) {
		uint64_t pin = pinned_va(ctx, hdst, (uint64_t)(h - 1) * dp + w);

		if (pin) {
			pthread_mutex_lock(&ctx->lock);
			r = ce_copy2d(ctx, s, pin, dp, dsrc, sp, w, h);
			pthread_mutex_unlock(&ctx->lock);
			return r || async ? r : nvc_stream_wait_idle(s);
		}
		if (w > half)
			return CUDA_ERROR_INVALID_VALUE;
		rows = (uint32_t)(half / w);
		for (y = 0; y < h && r == CUDA_SUCCESS; y += rows) {
			uint32_t n = h - y < rows ? h - y : rows, i;
			uint64_t point;

			pthread_mutex_lock(&ctx->lock);
			r = ce_copy2d(ctx, s, ctx->staging.va, w, dsrc + (uint64_t)y * sp, sp, w, n);
			point = s->point;
			pthread_mutex_unlock(&ctx->lock);
			if (r)
				break;
			r = nvc_stream_wait_idle(&(struct CUstream_st){ .ctx = ctx, .tl = s->tl, .point = point });
			for (i = 0; i < n; i++)
				memcpy(hdst + (size_t)(y + i) * dp, (char *)ctx->staging.map + (size_t)i * w, w);
		}
		return r;
	}
	pthread_mutex_lock(&ctx->lock);
	r = ce_copy2d(ctx, s, ddst, dp, dsrc, sp, w, h);
	pthread_mutex_unlock(&ctx->lock);
	return r || async ? r : nvc_stream_wait_idle(s);
}

CUAPI CUresult cuMemcpy2D_v2(const CUDA_MEMCPY2D *copy)
{
	return memcpy2d(copy, NULL, false);
}

CUAPI CUresult cuMemcpy2DUnaligned_v2(const CUDA_MEMCPY2D *copy)
{
	return memcpy2d(copy, NULL, false);
}

CUAPI CUresult cuMemcpy2DAsync_v2(const CUDA_MEMCPY2D *copy, CUstream stream)
{
	return memcpy2d(copy, stream, true);
}
