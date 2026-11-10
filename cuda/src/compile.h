/* SPDX-License-Identifier: MIT */
/* The kernel compiler (compile.c): kept free of Mesa headers so the rest of nvcuda stays plain C. */
#ifndef NVCUDA_COMPILE_H
#define NVCUDA_COMPILE_H

#include <stddef.h>
#include <stdint.h>

/* nvcuda's constant buffer 0 */
#define NVC_CB_BLOCK	0x00	/* uint32 x 3: block size */
#define NVC_CB_GRID	0x0c	/* uint32 x 3: grid size */
#define NVC_CB_PARAMS	0x20	/* kernel parameters */
#define NVC_CB_SIZE	0x1000
#define NVC_MAX_PARAMS	128
/*
 * Generic memory accesses are branch-free: shared, local and global are all accessed and the
 * right result selected. The inactive accesses go to a dummy slot: the first NVC_DUMMY_BYTES of
 * shared memory and of each thread's scratch, and a dummy page every context maps at this VA.
 */
#define NVC_DUMMY_VA	0x1f00000000ull
#define NVC_DUMMY_BYTES	64

struct nvc_kernel_bin {
	void *info;			/* struct nak_shader_info, owned by compile.c */
	void *code;
	uint32_t code_size;
	uint32_t shared_static;		/* static shared memory the launch allocates */
	uint32_t shared_hidden;		/* of which nvcuda's own (dummy slots), not the kernel's */
	uint32_t max_threads;		/* .maxntid / .reqntid, 0 = none */
	bool used_regmem;		/* PTX: registers were kept in scratch */
	uint32_t nparams, param_bytes;
	struct { uint32_t offset, size; } params[NVC_MAX_PARAMS];
};

int nvc_compiler_init(unsigned int gpcs, unsigned int tpcs);
struct nvc_kernel_bin *nvc_compile_spirv(const uint32_t *words, size_t nwords, const char *entry, char *err,
					 size_t errlen);
struct nvc_kernel_bin *nvc_compile_ptx(const char *ptx, const char *entry, uint64_t globals_va, char *err, size_t errlen);

/* module-scope .global/.const variables of a PTX module (ptx.c): layout and initial contents */
struct nvc_global {
	char name[128];
	uint32_t offset, size;
};
/* texture.c: TIC and TSC for a CUDA texture object (32 bytes each) */
struct CUDA_RESOURCE_DESC_st;
struct CUDA_TEXTURE_DESC_st;
int nvc_tex_headers(const struct CUDA_RESOURCE_DESC_st *res, const struct CUDA_TEXTURE_DESC_st *desc, uint32_t tic[8],
		    uint32_t tsc[8], char *err, size_t errlen);
const struct nv_device_info *nvc_compiler_devinfo(void);

int nvc_ptx_globals(const char *ptx, struct nvc_global **globals, uint32_t *count, void **image, uint32_t *size,
		    char *err, size_t errlen);
void nvc_kernel_bin_free(struct nvc_kernel_bin *kb);
uint32_t nvc_qmd_size(void);
void nvc_fill_qmd(const struct nvc_kernel_bin *kb, uint64_t code_va, uint64_t cb0_va, uint32_t cb0_size,
		  const uint32_t grid[3], const uint32_t block[3], uint32_t dyn_smem, void *qmd, size_t qmd_size);
/* per-thread local memory (bytes) the kernel needs, and its register / barrier counts */
uint32_t nvc_kernel_slm(const struct nvc_kernel_bin *kb);
uint32_t nvc_kernel_gprs(const struct nvc_kernel_bin *kb);

#endif
