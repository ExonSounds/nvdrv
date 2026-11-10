/* SPDX-License-Identifier: MIT */
#ifndef NVCUDA_INTERNAL_H
#define NVCUDA_INTERNAL_H

#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>

#include "cuda.h"

#define CUAPI __attribute__((visibility("default")))

#define NVC_MAX_DEVICES		4
/*
 * Unified addressing: every device allocation gets a GPU VA from one process-wide range, and the
 * same range is reserved (PROT_NONE) in the process, so a device pointer never equals a host
 * pointer. Below the uAPI's kernel-managed range (VM_INIT at 512 GiB).
 */
#define NVC_VA_BASE		0x0000002000000000ull	/* 128 GiB */
#define NVC_VA_END		0x0000007f00000000ull	/* < 512 GiB */
#define NVC_KERNEL_VA		(1ull << 39)
#define NVC_KERNEL_VA_SIZE	(1ull << 39)

#define NVC_PUSH_SIZE		(1u << 20)	/* push ring per context (GART) */
#define NVC_STAGING_SIZE	(8u << 20)	/* bounce buffer for pageable host copies, two halves */
#define NVC_EVENT_SLOTS		1024		/* 16-byte four-word semaphores (payload + timestamp) */

struct nvc_device {
	char path[64];			/* render node */
	char name[64];
	uint16_t chipset;
	uint64_t vram, vram_user;
	uint32_t sm_count, gpc_count, tpc_count;
	uint16_t pci_domain;
	uint8_t pci_bus, pci_dev, pci_func;
	uint16_t pci_device_id;
	struct CUctx_st *primary;
	int primary_refs;
	unsigned int primary_flags;
};

/* a GEM object mapped into the context's GPU VM (and maybe the CPU) */
struct nvc_bo {
	uint32_t handle;
	uint64_t size;
	uint64_t va;
	void *map;
};

/* a CUDA stream: one timeline syncobj, points in submission order */
struct CUstream_st {
	struct CUctx_st *ctx;
	uint32_t tl;
	uint64_t point;			/* last point submitted */
	struct { uint32_t tl; uint64_t point; } waits[8];	/* cuStreamWaitEvent, for the next submission */
	int nwaits;
	unsigned int flags;
};

struct CUevent_st {
	struct CUctx_st *ctx;
	unsigned int flags;
	uint32_t tl;			/* recorded at (tl, point); 0 = never recorded */
	uint64_t point;
	int slot;			/* timestamp slot, -1 with CU_EVENT_DISABLE_TIMING */
};

struct CUctx_st {
	struct nvc_device *dev;
	CUdevice ordinal;
	unsigned int flags;
	int fd;
	pthread_mutex_t lock;
	uint32_t ce_chan;		/* copy channel (AMPERE_DMA_COPY_B) */
	uint32_t gr_chan;		/* compute channel (AMPERE_COMPUTE_B), created on first launch */
	bool gr_ready;
	struct nvc_bo slm;		/* shader local memory for all TPCs */
	uint32_t slm_per_thread;
	struct CUmod_st *modules;
	struct nvc_bo push;		/* push ring */
	uint32_t push_off;
	struct nvc_bo staging;		/* pageable copies */
	struct nvc_bo events;		/* event timestamps */
	uint64_t event_used[NVC_EVENT_SLOTS / 64];
	struct CUstream_st null_stream;	/* stream 0 / legacy */
	struct CUstream_st **streams;	/* user streams, for cuCtxSynchronize */
	int nstreams, maxstreams;
	size_t limit_stack, limit_heap, limit_printf;
	CUfunc_cache cache_config;
};

/* device.c */
extern struct nvc_device nvc_devices[NVC_MAX_DEVICES];
extern int nvc_ndevices;
extern bool nvc_initialized;
int nvc_nvif_device(int fd);

/* context.c */
struct CUctx_st *nvc_current(void);
CUresult nvc_ctx_get(struct CUctx_st **ctx);	/* current context or CUDA_ERROR_INVALID_CONTEXT */
CUresult nvc_ctx_init(struct CUctx_st *ctx, CUdevice dev, unsigned int flags);
void nvc_ctx_fini(struct CUctx_st *ctx);
int nvc_bo_new(struct CUctx_st *ctx, struct nvc_bo *bo, uint64_t size, uint32_t domain, bool map, uint64_t va);
void nvc_bo_free(struct CUctx_st *ctx, struct nvc_bo *bo);

/* va.c: the process-wide device VA allocator and allocation records */
enum nvc_alloc_kind { NVC_ALLOC_DEVICE, NVC_ALLOC_HOST };
struct nvc_alloc {
	struct nvc_alloc *next;
	enum nvc_alloc_kind kind;
	struct CUctx_st *ctx;
	struct nvc_bo bo;		/* bo.va = device pointer; bo.map = host pointer (HOST) */
};
uint64_t nvc_va_get(uint64_t size, uint64_t align);
void nvc_va_put(uint64_t va, uint64_t size);
void nvc_alloc_add(struct nvc_alloc *a);
struct nvc_alloc *nvc_alloc_remove_va(uint64_t va);
struct nvc_alloc *nvc_alloc_remove_host(void *p);
struct nvc_alloc *nvc_alloc_find_va(uint64_t va, uint64_t len);	/* [va, va+len) inside one allocation */
struct nvc_alloc *nvc_alloc_find_host(const void *p, uint64_t len);
void nvc_alloc_drop_ctx(struct CUctx_st *ctx);

/* submit.c */
struct CUstream_st *nvc_stream(struct CUctx_st *ctx, CUstream s);
uint32_t *nvc_push_begin(struct CUctx_st *ctx, uint32_t dwords);
CUresult nvc_push_submit(struct CUctx_st *ctx, struct CUstream_st *s, uint32_t chan, uint32_t *start, uint32_t *end);
CUresult nvc_stream_wait_idle(struct CUstream_st *s);
CUresult nvc_ctx_idle(struct CUctx_st *ctx);
uint32_t *nvc_mthd(uint32_t *p, int subc, uint32_t mthd, int n, const uint32_t *data);

/* memory.c: pageable host -> GPU VA, synchronous (code upload) */
CUresult nvc_upload(struct CUctx_st *ctx, uint64_t va, const void *data, size_t bytes);
CUresult nvc_mem_alloc_device(struct CUctx_st *ctx, size_t bytes, uint64_t *va);
void nvc_mem_free_device(struct CUctx_st *ctx, uint64_t va);

/* module.c */
void nvc_modules_drop(struct CUctx_st *ctx);

/* copy engine helpers (memory.c) */
#define NVC_SUBC_CE	4

static inline uint32_t *nvc_mthd1(uint32_t *p, int subc, uint32_t mthd, uint32_t v)
{
	return nvc_mthd(p, subc, mthd, 1, &v);
}

#endif
