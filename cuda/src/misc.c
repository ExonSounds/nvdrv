// SPDX-License-Identifier: MIT
/* Small entry points: unversioned names some loaders look up, stream queries, GL interop. */
#include "internal.h"

/* cuew looks these up by their plain names; same ABI as the _v2 versions */
#undef cuDevicePrimaryCtxRelease
#undef cuDevicePrimaryCtxReset
#undef cuDevicePrimaryCtxSetFlags

CUAPI CUresult cuDevicePrimaryCtxRelease(CUdevice dev);
CUAPI CUresult cuDevicePrimaryCtxReset(CUdevice dev);
CUAPI CUresult cuDevicePrimaryCtxSetFlags(CUdevice dev, unsigned int flags);

CUAPI CUresult cuDevicePrimaryCtxRelease(CUdevice dev)
{
	return cuDevicePrimaryCtxRelease_v2(dev);
}

CUAPI CUresult cuDevicePrimaryCtxReset(CUdevice dev)
{
	return cuDevicePrimaryCtxReset_v2(dev);
}

CUAPI CUresult cuDevicePrimaryCtxSetFlags(CUdevice dev, unsigned int flags)
{
	return cuDevicePrimaryCtxSetFlags_v2(dev, flags);
}

CUAPI CUresult cuMemsetD16Async(CUdeviceptr dst, unsigned short value, size_t count, CUstream stream);
CUAPI CUresult cuStreamGetCtx(CUstream stream, CUcontext *ctx);
CUAPI CUresult cuStreamGetFlags(CUstream stream, unsigned int *flags);
CUAPI CUresult cuGLGetDevices_v2(unsigned int *count, CUdevice *devices, unsigned int max, int list);

CUAPI CUresult cuStreamGetCtx(CUstream stream, CUcontext *ctx)
{
	if (!ctx)
		return CUDA_ERROR_INVALID_VALUE;
	if (stream && stream != CU_STREAM_LEGACY && stream != CU_STREAM_PER_THREAD) {
		*ctx = stream->ctx;
		return CUDA_SUCCESS;
	}
	return cuCtxGetCurrent(ctx);
}

CUAPI CUresult cuStreamGetFlags(CUstream stream, unsigned int *flags)
{
	if (!flags)
		return CUDA_ERROR_INVALID_VALUE;
	*flags = stream && stream != CU_STREAM_LEGACY && stream != CU_STREAM_PER_THREAD ? stream->flags : 0;
	return CUDA_SUCCESS;
}

/* No OpenGL interop: no CUDA device drives the current GL context. */
CUAPI CUresult cuGLGetDevices_v2(unsigned int *count, CUdevice *devices, unsigned int max, int list)
{
	(void)devices;
	(void)max;
	(void)list;
	if (count)
		*count = 0;
	return CUDA_ERROR_NO_DEVICE;
}
