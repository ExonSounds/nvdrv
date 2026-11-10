/* SPDX-License-Identifier: MIT */
/* PTX front end (ptx.c); for compile.c only (needs Mesa's NIR headers). */
#ifndef NVCUDA_PTX_H
#define NVCUDA_PTX_H

struct nvc_kernel_bin;

/*
 * Build NIR for the PTX .entry @entry: stage KERNEL, unstructured, registers as function_temp
 * variables, memory already as explicit intrinsics. Fills kb->params / nparams / param_bytes.
 */
/* how to build: -1 = decide from the kernel (and NVCUDA_OUTLINE / NVCUDA_REGMEM), 0 / 1 = force */
struct nvc_ptx_opts {
	int outline;
	int regmem;
};

nir_shader *nvc_ptx_build(const char *src, const char *entry, const nir_shader_compiler_options *opts,
			  struct nvc_kernel_bin *kb, uint64_t globals_va, const struct nvc_ptx_opts *po, char *err,
			  size_t errlen);

#endif
