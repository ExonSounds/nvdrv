// SPDX-License-Identifier: MIT
/*
 * Texture objects. Each context has a texture header pool (TIC) and a sampler pool (TSC) in
 * host memory (GART, mapped), bound to its compute channel once (module.c gr_init). A texture
 * object is one entry of each: its handle, the 32-bit value a kernel passes to tex (sampler
 * index << 20 | header index), is the low half of the CUtexObject; the high half is the object's
 * slot + 1. texture.c builds the headers.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <drm/nouveau_drm.h>

#include "internal.h"
#include "compile.h"

#define NVC_TIC_COUNT	4096
#define NVC_TSC_COUNT	1024

static int alloc_bit(uint64_t *map, int n)
{
	for (int i = 0; i < n; i++)
		if (!(map[i / 64] & (1ull << (i % 64)))) {
			map[i / 64] |= 1ull << (i % 64);
			return i;
		}
	return -1;
}

int nvc_tex_pools_init(struct CUctx_st *ctx)
{
	if (nvc_bo_new(ctx, &ctx->tic_pool, NVC_TIC_COUNT * 32, NOUVEAU_GEM_DOMAIN_GART, true, 0) ||
	    nvc_bo_new(ctx, &ctx->tsc_pool, NVC_TSC_COUNT * 32, NOUVEAU_GEM_DOMAIN_GART, true, 0))
		return -1;
	memset(ctx->tic_pool.map, 0, ctx->tic_pool.size);
	memset(ctx->tsc_pool.map, 0, ctx->tsc_pool.size);
	ctx->tic_used[0] = 1;	/* index 0 stays a null header */
	return 0;
}

void nvc_tex_pools_fini(struct CUctx_st *ctx)
{
	free(ctx->texobjs);
	ctx->texobjs = NULL;
	nvc_bo_free(ctx, &ctx->tic_pool);
	nvc_bo_free(ctx, &ctx->tsc_pool);
}

uint32_t nvc_tex_pool_counts(uint32_t *tsc)
{
	*tsc = NVC_TSC_COUNT;
	return NVC_TIC_COUNT;
}

CUAPI CUresult cuTexObjectCreate(CUtexObject *tex, const CUDA_RESOURCE_DESC *res, const CUDA_TEXTURE_DESC *desc,
				 const CUDA_RESOURCE_VIEW_DESC *view)
{
	struct CUctx_st *ctx;
	struct nvc_texobj *t;
	uint32_t tic[8], tsc[8];
	char err[128] = "";
	int slot, i_tic, i_tsc;
	CUresult r = nvc_ctx_get(&ctx);

	if (r)
		return r;
	if (!tex || !res || !desc)
		return CUDA_ERROR_INVALID_VALUE;
	if (view)
		return CUDA_ERROR_NOT_SUPPORTED;	/* resource views reinterpret arrays */
	if (nvc_compiler_init(ctx->dev->gpc_count, ctx->dev->tpc_count))
		return CUDA_ERROR_UNKNOWN;
	if (nvc_tex_headers(res, desc, tic, tsc, err, sizeof(err))) {
		if (getenv("NVCUDA_DEBUG"))
			fprintf(stderr, "nvcuda: cuTexObjectCreate: %s\n", err);
		return strstr(err, "not supported") ? CUDA_ERROR_NOT_SUPPORTED : CUDA_ERROR_INVALID_VALUE;
	}
	pthread_mutex_lock(&ctx->lock);
	for (slot = 0; slot < ctx->ntexobjs && ctx->texobjs[slot].used; slot++)
		;
	if (slot == ctx->ntexobjs) {
		int n = ctx->ntexobjs ? ctx->ntexobjs * 2 : 64;
		struct nvc_texobj *a = realloc(ctx->texobjs, n * sizeof(*a));

		if (!a) {
			pthread_mutex_unlock(&ctx->lock);
			return CUDA_ERROR_OUT_OF_MEMORY;
		}
		memset(a + ctx->ntexobjs, 0, (n - ctx->ntexobjs) * sizeof(*a));
		ctx->texobjs = a;
		ctx->ntexobjs = n;
	}
	i_tic = alloc_bit(ctx->tic_used, NVC_TIC_COUNT);
	i_tsc = alloc_bit(ctx->tsc_used, NVC_TSC_COUNT);
	if (i_tic < 0 || i_tsc < 0) {
		if (i_tic >= 0)
			ctx->tic_used[i_tic / 64] &= ~(1ull << (i_tic % 64));
		pthread_mutex_unlock(&ctx->lock);
		return CUDA_ERROR_OUT_OF_MEMORY;
	}
	memcpy((char *)ctx->tic_pool.map + i_tic * 32, tic, 32);
	memcpy((char *)ctx->tsc_pool.map + i_tsc * 32, tsc, 32);
	t = &ctx->texobjs[slot];
	t->used = true;
	t->tic = i_tic;
	t->tsc = i_tsc;
	t->res = *res;
	t->desc = *desc;
	ctx->tex_dirty = true;	/* the next launch invalidates the header and sampler caches */
	pthread_mutex_unlock(&ctx->lock);
	*tex = (uint64_t)(slot + 1) << 32 | (uint32_t)i_tsc << 20 | (uint32_t)i_tic;
	return CUDA_SUCCESS;
}

static struct nvc_texobj *lookup(struct CUctx_st *ctx, CUtexObject tex)
{
	uint32_t slot = (uint32_t)(tex >> 32);

	if (!slot || slot > (uint32_t)ctx->ntexobjs || !ctx->texobjs[slot - 1].used)
		return NULL;
	return &ctx->texobjs[slot - 1];
}

CUAPI CUresult cuTexObjectDestroy(CUtexObject tex)
{
	struct CUctx_st *ctx;
	struct nvc_texobj *t;
	CUresult r = nvc_ctx_get(&ctx);

	if (r)
		return r;
	pthread_mutex_lock(&ctx->lock);
	t = lookup(ctx, tex);
	if (!t) {
		pthread_mutex_unlock(&ctx->lock);
		return CUDA_ERROR_INVALID_HANDLE;
	}
	ctx->tic_used[t->tic / 64] &= ~(1ull << (t->tic % 64));
	ctx->tsc_used[t->tsc / 64] &= ~(1ull << (t->tsc % 64));
	memset((char *)ctx->tic_pool.map + t->tic * 32, 0, 32);	/* kernels still running: a null texture */
	t->used = false;
	pthread_mutex_unlock(&ctx->lock);
	return CUDA_SUCCESS;
}

CUAPI CUresult cuTexObjectGetResourceDesc(CUDA_RESOURCE_DESC *res, CUtexObject tex)
{
	struct CUctx_st *ctx;
	struct nvc_texobj *t;
	CUresult r = nvc_ctx_get(&ctx);

	if (r)
		return r;
	if (!res)
		return CUDA_ERROR_INVALID_VALUE;
	pthread_mutex_lock(&ctx->lock);
	t = lookup(ctx, tex);
	if (t)
		*res = t->res;
	pthread_mutex_unlock(&ctx->lock);
	return t ? CUDA_SUCCESS : CUDA_ERROR_INVALID_HANDLE;
}

CUAPI CUresult cuTexObjectGetTextureDesc(CUDA_TEXTURE_DESC *desc, CUtexObject tex)
{
	struct CUctx_st *ctx;
	struct nvc_texobj *t;
	CUresult r = nvc_ctx_get(&ctx);

	if (r)
		return r;
	if (!desc)
		return CUDA_ERROR_INVALID_VALUE;
	pthread_mutex_lock(&ctx->lock);
	t = lookup(ctx, tex);
	if (t)
		*desc = t->desc;
	pthread_mutex_unlock(&ctx->lock);
	return t ? CUDA_SUCCESS : CUDA_ERROR_INVALID_HANDLE;
}

/* pitch: 128-byte rows keep every row 256-byte addressable for textures (pitch 32-aligned) */
CUAPI CUresult cuMemAllocPitch_v2(CUdeviceptr *dptr, size_t *pitch, size_t width_bytes, size_t height,
				  unsigned int element_bytes)
{
	size_t p;

	if (!dptr || !pitch || !width_bytes || !height ||
	    (element_bytes != 4 && element_bytes != 8 && element_bytes != 16))
		return CUDA_ERROR_INVALID_VALUE;
	p = (width_bytes + 127) & ~(size_t)127;
	*pitch = p;
	return cuMemAlloc_v2(dptr, p * height);
}
