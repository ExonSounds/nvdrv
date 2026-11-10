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

struct nvc_kernel_bin {
	void *info;			/* struct nak_shader_info, owned by compile.c */
	void *code;
	uint32_t code_size;
	uint32_t shared_static;		/* bytes of __shared__ the kernel declares */
	uint32_t nparams, param_bytes;
	struct { uint32_t offset, size; } params[NVC_MAX_PARAMS];
};

int nvc_compiler_init(unsigned int gpcs, unsigned int tpcs);
struct nvc_kernel_bin *nvc_compile_spirv(const uint32_t *words, size_t nwords, const char *entry, char *err,
					 size_t errlen);
void nvc_kernel_bin_free(struct nvc_kernel_bin *kb);
uint32_t nvc_qmd_size(void);
void nvc_fill_qmd(const struct nvc_kernel_bin *kb, uint64_t code_va, uint64_t cb0_va, uint32_t cb0_size,
		  const uint32_t grid[3], const uint32_t block[3], uint32_t dyn_smem, void *qmd, size_t qmd_size);
/* per-thread local memory (bytes) the kernel needs, and its register / barrier counts */
uint32_t nvc_kernel_slm(const struct nvc_kernel_bin *kb);
uint32_t nvc_kernel_gprs(const struct nvc_kernel_bin *kb);

#endif
