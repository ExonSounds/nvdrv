// SPDX-License-Identifier: MIT
/*
 * Submission. Methods are written into the context's push ring (GART, mapped in the GPU VM) and
 * submitted with DRM_NOUVEAU_EXEC. Every stream is a timeline syncobj: each submission waits for
 * the stream's previous point (and any events the stream was told to wait for) and signals the
 * next one, so work in a stream runs in order even across channels. Callers hold ctx->lock
 * between nvc_push_begin and nvc_push_submit.
 */
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <xf86drm.h>
#include <drm/nouveau_drm.h>

#include "internal.h"

uint32_t *nvc_mthd(uint32_t *p, int subc, uint32_t mthd, int n, const uint32_t *data)
{
	*p++ = 0x20000000 | (uint32_t)n << 16 | (uint32_t)subc << 13 | mthd >> 2;	/* incrementing */
	while (n--)
		*p++ = *data++;
	return p;
}

struct CUstream_st *nvc_stream(struct CUctx_st *ctx, CUstream s)
{
	if (!s || s == CU_STREAM_LEGACY || s == CU_STREAM_PER_THREAD)
		return &ctx->null_stream;
	return s->ctx == ctx ? s : NULL;
}

uint32_t *nvc_push_begin(struct CUctx_st *ctx, uint32_t dwords)
{
	if (ctx->push_off + dwords * 4 > ctx->push.size) {
		/* wrap: everything submitted from the ring must have been fetched */
		pthread_mutex_unlock(&ctx->lock);
		nvc_ctx_idle(ctx);
		pthread_mutex_lock(&ctx->lock);
		ctx->push_off = 0;
	}
	return (uint32_t *)((char *)ctx->push.map + ctx->push_off);
}

CUresult nvc_push_submit(struct CUctx_st *ctx, struct CUstream_st *s, uint32_t chan, uint32_t *start, uint32_t *end)
{
	struct drm_nouveau_sync waits[9], sig;
	struct drm_nouveau_exec_push ep;
	struct drm_nouveau_exec ex;
	uint32_t off = (uint32_t)((char *)start - (char *)ctx->push.map);
	uint32_t len = (uint32_t)((char *)end - (char *)start);
	int nw = 0, i, r;

	if (s->point)
		waits[nw++] = (struct drm_nouveau_sync){ .flags = DRM_NOUVEAU_SYNC_TIMELINE_SYNCOBJ,
							 .handle = s->tl, .timeline_value = s->point };
	for (i = 0; i < s->nwaits; i++)
		waits[nw++] = (struct drm_nouveau_sync){ .flags = DRM_NOUVEAU_SYNC_TIMELINE_SYNCOBJ,
							 .handle = s->waits[i].tl,
							 .timeline_value = s->waits[i].point };
	sig = (struct drm_nouveau_sync){ .flags = DRM_NOUVEAU_SYNC_TIMELINE_SYNCOBJ, .handle = s->tl,
					 .timeline_value = s->point + 1 };
	ep = (struct drm_nouveau_exec_push){ .va = ctx->push.va + off, .va_len = len };
	ex = (struct drm_nouveau_exec){ .channel = chan, .push_count = 1, .push_ptr = (uintptr_t)&ep,
					.wait_count = nw, .wait_ptr = (uintptr_t)waits,
					.sig_count = 1, .sig_ptr = (uintptr_t)&sig };
	r = drmCommandWriteRead(ctx->fd, DRM_NOUVEAU_EXEC, &ex, sizeof(ex));
	ctx->push_off = (off + len + 63) & ~63u;
	if (r)
		return r == -ENODEV ? CUDA_ERROR_DEVICE_UNAVAILABLE : CUDA_ERROR_LAUNCH_FAILED;
	s->point++;
	s->nwaits = 0;
	return CUDA_SUCCESS;
}

CUresult nvc_stream_wait_idle(struct CUstream_st *s)
{
	uint64_t point = s->point;
	int r;

	if (!point)
		return CUDA_SUCCESS;
	r = drmSyncobjTimelineWait(s->ctx->fd, &s->tl, &point, 1, INT64_MAX, DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL, NULL);
	return r ? CUDA_ERROR_LAUNCH_FAILED : CUDA_SUCCESS;
}

CUresult nvc_ctx_idle(struct CUctx_st *ctx)
{
	CUresult res = nvc_stream_wait_idle(&ctx->null_stream), r;
	int i;

	for (i = 0; i < ctx->nstreams; i++)
		if ((r = nvc_stream_wait_idle(ctx->streams[i])) != CUDA_SUCCESS)
			res = r;
	return res;
}

/* ---- streams ---- */

CUAPI CUresult cuStreamCreate(CUstream *stream, unsigned int flags)
{
	struct CUctx_st *ctx;
	struct CUstream_st *s;
	CUresult r = nvc_ctx_get(&ctx);

	if (r)
		return r;
	if (!stream)
		return CUDA_ERROR_INVALID_VALUE;
	s = calloc(1, sizeof(*s));
	if (!s)
		return CUDA_ERROR_OUT_OF_MEMORY;
	s->ctx = ctx;
	s->flags = flags;
	if (drmSyncobjCreate(ctx->fd, 0, &s->tl)) {
		free(s);
		return CUDA_ERROR_OUT_OF_MEMORY;
	}
	pthread_mutex_lock(&ctx->lock);
	if (ctx->nstreams == ctx->maxstreams) {
		int n = ctx->maxstreams ? ctx->maxstreams * 2 : 8;
		struct CUstream_st **a = realloc(ctx->streams, n * sizeof(*a));

		if (!a) {
			pthread_mutex_unlock(&ctx->lock);
			drmSyncobjDestroy(ctx->fd, s->tl);
			free(s);
			return CUDA_ERROR_OUT_OF_MEMORY;
		}
		ctx->streams = a;
		ctx->maxstreams = n;
	}
	ctx->streams[ctx->nstreams++] = s;
	pthread_mutex_unlock(&ctx->lock);
	*stream = s;
	return CUDA_SUCCESS;
}

CUAPI CUresult cuStreamDestroy_v2(CUstream stream)
{
	struct CUctx_st *ctx;
	int i;

	if (!stream || stream == CU_STREAM_LEGACY || stream == CU_STREAM_PER_THREAD)
		return CUDA_ERROR_INVALID_HANDLE;
	ctx = stream->ctx;
	nvc_stream_wait_idle(stream);	/* the API lets pending work finish */
	pthread_mutex_lock(&ctx->lock);
	for (i = 0; i < ctx->nstreams; i++)
		if (ctx->streams[i] == stream) {
			ctx->streams[i] = ctx->streams[--ctx->nstreams];
			break;
		}
	pthread_mutex_unlock(&ctx->lock);
	drmSyncobjDestroy(ctx->fd, stream->tl);
	free(stream);
	return CUDA_SUCCESS;
}

CUAPI CUresult cuStreamSynchronize(CUstream stream)
{
	struct CUctx_st *ctx;
	struct CUstream_st *s;
	CUresult r = nvc_ctx_get(&ctx);

	if (stream && stream != CU_STREAM_LEGACY && stream != CU_STREAM_PER_THREAD)
		ctx = stream->ctx;
	else if (r)
		return r;
	s = nvc_stream(ctx, stream);
	return s ? nvc_stream_wait_idle(s) : CUDA_ERROR_INVALID_HANDLE;
}

CUAPI CUresult cuStreamQuery(CUstream stream)
{
	struct CUctx_st *ctx;
	struct CUstream_st *s;
	uint64_t point;
	CUresult r = nvc_ctx_get(&ctx);

	if (stream && stream != CU_STREAM_LEGACY && stream != CU_STREAM_PER_THREAD)
		ctx = stream->ctx;
	else if (r)
		return r;
	s = nvc_stream(ctx, stream);
	if (!s)
		return CUDA_ERROR_INVALID_HANDLE;
	point = s->point;
	if (!point)
		return CUDA_SUCCESS;
	return drmSyncobjTimelineWait(ctx->fd, &s->tl, &point, 1, 0, DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL, NULL) ?
	       CUDA_ERROR_NOT_READY : CUDA_SUCCESS;
}

CUAPI CUresult cuStreamWaitEvent(CUstream stream, CUevent event, unsigned int flags)
{
	struct CUctx_st *ctx;
	struct CUstream_st *s;
	CUresult r = nvc_ctx_get(&ctx);

	if (!event)
		return CUDA_ERROR_INVALID_HANDLE;
	if (stream && stream != CU_STREAM_LEGACY && stream != CU_STREAM_PER_THREAD)
		ctx = stream->ctx;
	else if (r)
		return r;
	(void)flags;
	s = nvc_stream(ctx, stream);
	if (!s)
		return CUDA_ERROR_INVALID_HANDLE;
	if (!event->tl)
		return CUDA_SUCCESS;		/* never recorded: nothing to wait for */
	if (event->ctx != ctx)
		return CUDA_ERROR_NOT_SUPPORTED;	/* the syncobj lives in another file */
	pthread_mutex_lock(&ctx->lock);
	if (s->nwaits == (int)(sizeof(s->waits) / sizeof(s->waits[0]))) {
		pthread_mutex_unlock(&ctx->lock);
		nvc_stream_wait_idle(&(struct CUstream_st){ .ctx = ctx, .tl = event->tl, .point = event->point });
		return CUDA_SUCCESS;
	}
	s->waits[s->nwaits].tl = event->tl;
	s->waits[s->nwaits].point = event->point;
	s->nwaits++;
	pthread_mutex_unlock(&ctx->lock);
	return CUDA_SUCCESS;
}

/* ---- events: a point on a stream's timeline, plus a GPU timestamp written by the copy engine ---- */

#define C7B5_SET_SEMAPHORE_A	0x240
#define C7B5_LAUNCH_DMA		0x300
/* no transfer, flush, four-word semaphore release (payload + 64-bit timestamp) */
#define LAUNCH_SEM_TIMESTAMP	(0x0 | 1 << 2 | 2 << 3)

CUAPI CUresult cuEventCreate(CUevent *event, unsigned int flags)
{
	struct CUctx_st *ctx;
	struct CUevent_st *e;
	CUresult r = nvc_ctx_get(&ctx);
	int i;

	if (r)
		return r;
	if (!event)
		return CUDA_ERROR_INVALID_VALUE;
	e = calloc(1, sizeof(*e));
	if (!e)
		return CUDA_ERROR_OUT_OF_MEMORY;
	e->ctx = ctx;
	e->flags = flags;
	e->slot = -1;
	if (!(flags & CU_EVENT_DISABLE_TIMING)) {
		pthread_mutex_lock(&ctx->lock);
		for (i = 0; i < NVC_EVENT_SLOTS; i++)
			if (!(ctx->event_used[i / 64] & (1ull << (i % 64)))) {
				ctx->event_used[i / 64] |= 1ull << (i % 64);
				e->slot = i;
				break;
			}
		pthread_mutex_unlock(&ctx->lock);
		if (e->slot < 0) {
			free(e);
			return CUDA_ERROR_OUT_OF_MEMORY;
		}
	}
	*event = e;
	return CUDA_SUCCESS;
}

CUAPI CUresult cuEventDestroy_v2(CUevent event)
{
	struct CUctx_st *ctx;

	if (!event)
		return CUDA_ERROR_INVALID_HANDLE;
	ctx = event->ctx;
	if (event->slot >= 0) {
		/* the slot may still be written by a pending record */
		if (event->tl)
			nvc_stream_wait_idle(&(struct CUstream_st){ .ctx = ctx, .tl = event->tl, .point = event->point });
		pthread_mutex_lock(&ctx->lock);
		ctx->event_used[event->slot / 64] &= ~(1ull << (event->slot % 64));
		pthread_mutex_unlock(&ctx->lock);
	}
	free(event);
	return CUDA_SUCCESS;
}

CUAPI CUresult cuEventRecord(CUevent event, CUstream stream)
{
	struct CUctx_st *ctx;
	struct CUstream_st *s;
	CUresult res;

	if (!event)
		return CUDA_ERROR_INVALID_HANDLE;
	ctx = event->ctx;
	s = nvc_stream(ctx, stream);
	if (!s)
		return CUDA_ERROR_INVALID_HANDLE;
	pthread_mutex_lock(&ctx->lock);
	if (event->slot >= 0) {
		uint64_t va = ctx->events.va + event->slot * 16;
		uint32_t sem[3] = { (uint32_t)(va >> 32), (uint32_t)va, 1 };
		uint32_t *start = nvc_push_begin(ctx, 8), *p = start;

		p = nvc_mthd(p, NVC_SUBC_CE, C7B5_SET_SEMAPHORE_A, 3, sem);
		p = nvc_mthd1(p, NVC_SUBC_CE, C7B5_LAUNCH_DMA, LAUNCH_SEM_TIMESTAMP);
		res = nvc_push_submit(ctx, s, ctx->ce_chan, start, p);
	} else if (!s->point) {
		/* an empty stream: an empty submission gives the event a point to wait for */
		uint32_t *start = nvc_push_begin(ctx, 2), *p = start;

		p = nvc_mthd1(p, NVC_SUBC_CE, 0x0000, 0xc7b5);
		res = nvc_push_submit(ctx, s, ctx->ce_chan, start, p);
	} else {
		res = CUDA_SUCCESS;
	}
	if (res == CUDA_SUCCESS) {
		event->tl = s->tl;
		event->point = s->point;
	}
	pthread_mutex_unlock(&ctx->lock);
	return res;
}

CUAPI CUresult cuEventSynchronize(CUevent event)
{
	if (!event)
		return CUDA_ERROR_INVALID_HANDLE;
	if (!event->tl)
		return CUDA_SUCCESS;
	return nvc_stream_wait_idle(&(struct CUstream_st){ .ctx = event->ctx, .tl = event->tl, .point = event->point });
}

CUAPI CUresult cuEventQuery(CUevent event)
{
	uint64_t point;

	if (!event)
		return CUDA_ERROR_INVALID_HANDLE;
	if (!event->tl)
		return CUDA_SUCCESS;
	point = event->point;
	return drmSyncobjTimelineWait(event->ctx->fd, &event->tl, &point, 1, 0, DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL,
				      NULL) ? CUDA_ERROR_NOT_READY : CUDA_SUCCESS;
}

CUAPI CUresult cuEventElapsedTime(float *ms, CUevent start, CUevent end)
{
	const volatile uint64_t *t0, *t1;
	CUresult r;

	if (!ms || !start || !end)
		return CUDA_ERROR_INVALID_VALUE;
	if (start->slot < 0 || end->slot < 0)
		return CUDA_ERROR_INVALID_HANDLE;
	if (!start->tl || !end->tl)
		return CUDA_ERROR_INVALID_HANDLE;
	if ((r = cuEventQuery(start)) != CUDA_SUCCESS || (r = cuEventQuery(end)) != CUDA_SUCCESS)
		return r;
	/* four-word semaphore: payload at +0, GPU timestamp (ns) at +8 */
	t0 = (const volatile uint64_t *)((char *)start->ctx->events.map + start->slot * 16 + 8);
	t1 = (const volatile uint64_t *)((char *)end->ctx->events.map + end->slot * 16 + 8);
	*ms = (float)((double)(int64_t)(*t1 - *t0) / 1e6);
	return CUDA_SUCCESS;
}
