// SPDX-License-Identifier: MIT
/*
 * The process-wide device address space (unified addressing) and the allocation records.
 * [NVC_VA_BASE, NVC_VA_END) is reserved in the process (PROT_NONE, no backing) on first use, so
 * no host allocation can ever get an address that is also a device pointer. GPU VAs come from a
 * first-fit free list over that range; every context maps its objects at the VA they got here.
 */
#include <stdlib.h>
#include <sys/mman.h>

#include "internal.h"

struct range {
	struct range *next;
	uint64_t start, end;
};

static pthread_mutex_t va_lock = PTHREAD_MUTEX_INITIALIZER;
static struct range *free_list;
static bool va_ready;
static struct nvc_alloc *allocs;

static void va_init(void)
{
	void *want = (void *)(uintptr_t)NVC_VA_BASE;
	void *got;

	free_list = calloc(1, sizeof(*free_list));
	free_list->start = NVC_VA_BASE;
	free_list->end = NVC_VA_END;
	/* best effort: without it, a device pointer could in theory equal a host pointer */
	got = mmap(want, NVC_VA_END - NVC_VA_BASE, PROT_NONE,
		   MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED_NOREPLACE, -1, 0);
	if (got != MAP_FAILED && got != want)
		munmap(got, NVC_VA_END - NVC_VA_BASE);
	va_ready = true;
}

uint64_t nvc_va_get(uint64_t size, uint64_t align)
{
	struct range **pp, *r;
	uint64_t va = 0;

	pthread_mutex_lock(&va_lock);
	if (!va_ready)
		va_init();
	for (pp = &free_list; (r = *pp); pp = &r->next) {
		uint64_t s = (r->start + align - 1) & ~(align - 1);

		if (s + size > r->end || s + size < s)
			continue;
		va = s;
		if (s > r->start) {		/* keep the alignment gap */
			struct range *gap = malloc(sizeof(*gap));

			if (!gap) {
				va = 0;
				break;
			}
			gap->start = r->start;
			gap->end = s;
			gap->next = r;
			*pp = gap;
			pp = &gap->next;
		}
		r->start = s + size;
		if (r->start == r->end) {
			*pp = r->next;
			free(r);
		}
		break;
	}
	pthread_mutex_unlock(&va_lock);
	return va;
}

void nvc_va_put(uint64_t va, uint64_t size)
{
	struct range **pp, *r, *n;

	pthread_mutex_lock(&va_lock);
	for (pp = &free_list; (r = *pp) && r->start < va; pp = &r->next)
		;
	n = malloc(sizeof(*n));
	if (n) {
		n->start = va;
		n->end = va + size;
		n->next = r;
		*pp = n;
		if (r && n->end == r->start) {		/* merge with the next range */
			n->end = r->end;
			n->next = r->next;
			free(r);
		}
		/* and with the previous one */
		for (r = free_list; r && r->next != n; r = r->next)
			;
		if (r && r->end == n->start) {
			r->end = n->end;
			r->next = n->next;
			free(n);
		}
	}
	pthread_mutex_unlock(&va_lock);
}

void nvc_alloc_add(struct nvc_alloc *a)
{
	pthread_mutex_lock(&va_lock);
	a->next = allocs;
	allocs = a;
	pthread_mutex_unlock(&va_lock);
}

static struct nvc_alloc *remove_if(bool (*match)(struct nvc_alloc *, const void *), const void *key)
{
	struct nvc_alloc **pp, *a;

	pthread_mutex_lock(&va_lock);
	for (pp = &allocs; (a = *pp); pp = &a->next)
		if (match(a, key)) {
			*pp = a->next;
			break;
		}
	pthread_mutex_unlock(&va_lock);
	return a;
}

static bool match_va(struct nvc_alloc *a, const void *key)
{
	return a->kind == NVC_ALLOC_DEVICE && a->bo.va == *(const uint64_t *)key;
}

static bool match_host(struct nvc_alloc *a, const void *key)
{
	return a->kind == NVC_ALLOC_HOST && a->bo.map == key;
}

struct nvc_alloc *nvc_alloc_remove_va(uint64_t va)
{
	return remove_if(match_va, &va);
}

struct nvc_alloc *nvc_alloc_remove_host(void *p)
{
	return remove_if(match_host, p);
}

/* Device pointers: device allocations and the GPU side of pinned host allocations. */
struct nvc_alloc *nvc_alloc_find_va(uint64_t va, uint64_t len)
{
	struct nvc_alloc *a;

	pthread_mutex_lock(&va_lock);
	for (a = allocs; a; a = a->next)
		if (va >= a->bo.va && va - a->bo.va <= a->bo.size && len <= a->bo.size - (va - a->bo.va))
			break;
	pthread_mutex_unlock(&va_lock);
	return a;
}

struct nvc_alloc *nvc_alloc_find_host(const void *p, uint64_t len)
{
	uintptr_t x = (uintptr_t)p;
	struct nvc_alloc *a;

	pthread_mutex_lock(&va_lock);
	for (a = allocs; a; a = a->next) {
		uintptr_t m = (uintptr_t)a->bo.map;

		if (a->kind == NVC_ALLOC_HOST && x >= m && x - m <= a->bo.size && len <= a->bo.size - (x - m))
			break;
	}
	pthread_mutex_unlock(&va_lock);
	return a;
}

/* A context going away frees what it still owns. */
void nvc_alloc_drop_ctx(struct CUctx_st *ctx)
{
	struct nvc_alloc **pp, *a, *gone = NULL;

	pthread_mutex_lock(&va_lock);
	for (pp = &allocs; (a = *pp);) {
		if (a->ctx == ctx) {
			*pp = a->next;
			a->next = gone;
			gone = a;
		} else {
			pp = &a->next;
		}
	}
	pthread_mutex_unlock(&va_lock);
	while ((a = gone)) {
		gone = a->next;
		nvc_bo_free(ctx, &a->bo);
		free(a);
	}
}
