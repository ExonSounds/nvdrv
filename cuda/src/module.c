// SPDX-License-Identifier: MIT
/*
 * Modules and launches. Milestone 2: PTX -> NIR -> NAK (Mesa's compiler) -> QMD launch on
 * AMPERE_COMPUTE_B. Until then loading a module fails cleanly. NVIDIA cubins are never accepted
 * (undocumented format): a cubin image gets CUDA_ERROR_NO_BINARY_FOR_GPU, so programs that also
 * ship PTX can fall back to it.
 */
#include <string.h>

#include "internal.h"

static bool is_elf(const void *image)
{
	return !memcmp(image, "\x7f" "ELF", 4);
}

CUAPI CUresult cuModuleLoadDataEx(CUmodule *module, const void *image, unsigned int n, CUjit_option *options,
				  void **values)
{
	struct CUctx_st *ctx;
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
	return CUDA_ERROR_NOT_SUPPORTED;	/* PTX: milestone 2 */
}

CUAPI CUresult cuModuleLoadData(CUmodule *module, const void *image)
{
	return cuModuleLoadDataEx(module, image, 0, NULL, NULL);
}

CUAPI CUresult cuModuleLoad(CUmodule *module, const char *path)
{
	if (!module || !path)
		return CUDA_ERROR_INVALID_VALUE;
	return CUDA_ERROR_NOT_SUPPORTED;
}

CUAPI CUresult cuModuleUnload(CUmodule module)
{
	return module ? CUDA_ERROR_INVALID_HANDLE : CUDA_ERROR_INVALID_VALUE;
}

CUAPI CUresult cuModuleGetFunction(CUfunction *func, CUmodule module, const char *name)
{
	(void)func;
	(void)name;
	return module ? CUDA_ERROR_INVALID_HANDLE : CUDA_ERROR_INVALID_VALUE;
}

CUAPI CUresult cuModuleGetGlobal_v2(CUdeviceptr *dptr, size_t *bytes, CUmodule module, const char *name)
{
	(void)dptr;
	(void)bytes;
	(void)name;
	return module ? CUDA_ERROR_INVALID_HANDLE : CUDA_ERROR_INVALID_VALUE;
}

CUAPI CUresult cuFuncGetAttribute(int *value, CUfunction_attribute attrib, CUfunction func)
{
	(void)value;
	(void)attrib;
	return func ? CUDA_ERROR_INVALID_HANDLE : CUDA_ERROR_INVALID_VALUE;
}

CUAPI CUresult cuFuncSetCacheConfig(CUfunction func, CUfunc_cache config)
{
	(void)config;
	return func ? CUDA_ERROR_INVALID_HANDLE : CUDA_ERROR_INVALID_VALUE;
}

CUAPI CUresult cuLaunchKernel(CUfunction f, unsigned int gx, unsigned int gy, unsigned int gz, unsigned int bx,
			      unsigned int by, unsigned int bz, unsigned int shared_bytes, CUstream stream,
			      void **params, void **extra)
{
	(void)gx; (void)gy; (void)gz; (void)bx; (void)by; (void)bz;
	(void)shared_bytes; (void)stream; (void)params; (void)extra;
	return f ? CUDA_ERROR_INVALID_HANDLE : CUDA_ERROR_INVALID_VALUE;
}

CUAPI CUresult cuOccupancyMaxPotentialBlockSize(int *min_grid, int *block, CUfunction func, void *size_fn,
						size_t dyn_smem, int limit)
{
	(void)min_grid; (void)block; (void)size_fn; (void)dyn_smem; (void)limit;
	return func ? CUDA_ERROR_INVALID_HANDLE : CUDA_ERROR_INVALID_VALUE;
}
