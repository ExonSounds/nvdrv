/* SPDX-License-Identifier: MIT */
/*
 * nvcuda: an open implementation of the CUDA Driver API on nvdrv (nouveau uAPI).
 *
 * Written from NVIDIA's public CUDA Driver API reference; no NVIDIA header is used. Types are
 * ABI-compatible with programs built against the official header: the same sizes, enum values
 * (cuda_enums.h) and versioned entry points (cuMemAlloc -> cuMemAlloc_v2 and so on; the library
 * exports both names).
 */
#ifndef NVDRV_CUDA_H
#define NVDRV_CUDA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CUDA_VERSION 13000	/* reported API level; Blender requires >= 13000 */

#include "cuda_enums.h"

typedef int CUdevice;
typedef unsigned long long CUdeviceptr;
typedef struct CUctx_st *CUcontext;
typedef struct CUmod_st *CUmodule;
typedef struct CUfunc_st *CUfunction;
typedef struct CUstream_st *CUstream;
typedef struct CUevent_st *CUevent;

typedef struct CUuuid_st {
	char bytes[16];
} CUuuid;

#define CU_STREAM_LEGACY	((CUstream)0x1)
#define CU_STREAM_PER_THREAD	((CUstream)0x2)

#define CU_MEMHOSTALLOC_PORTABLE	0x01
#define CU_MEMHOSTALLOC_DEVICEMAP	0x02
#define CU_MEMHOSTALLOC_WRITECOMBINED	0x04

#define CU_LAUNCH_PARAM_END		((void *)0x00)
#define CU_LAUNCH_PARAM_BUFFER_POINTER	((void *)0x01)
#define CU_LAUNCH_PARAM_BUFFER_SIZE	((void *)0x02)

/* Versioned entry points, as the official header maps them. */
#define cuDeviceTotalMem	cuDeviceTotalMem_v2
#define cuCtxCreate		cuCtxCreate_v2
#define cuCtxDestroy		cuCtxDestroy_v2
#define cuCtxPushCurrent	cuCtxPushCurrent_v2
#define cuCtxPopCurrent		cuCtxPopCurrent_v2
#define cuDevicePrimaryCtxRelease	cuDevicePrimaryCtxRelease_v2
#define cuDevicePrimaryCtxReset	cuDevicePrimaryCtxReset_v2
#define cuDevicePrimaryCtxSetFlags	cuDevicePrimaryCtxSetFlags_v2
#define cuMemGetInfo		cuMemGetInfo_v2
#define cuMemAlloc		cuMemAlloc_v2
#define cuMemFree		cuMemFree_v2
#define cuMemAllocHost		cuMemAllocHost_v2
#define cuMemHostGetDevicePointer	cuMemHostGetDevicePointer_v2
#define cuMemGetAddressRange	cuMemGetAddressRange_v2
#define cuMemcpyHtoD		cuMemcpyHtoD_v2
#define cuMemcpyDtoH		cuMemcpyDtoH_v2
#define cuMemcpyDtoD		cuMemcpyDtoD_v2
#define cuMemcpyHtoDAsync	cuMemcpyHtoDAsync_v2
#define cuMemcpyDtoHAsync	cuMemcpyDtoHAsync_v2
#define cuMemcpyDtoDAsync	cuMemcpyDtoDAsync_v2
#define cuMemsetD8		cuMemsetD8_v2
#define cuMemsetD16		cuMemsetD16_v2
#define cuMemsetD32		cuMemsetD32_v2
#define cuStreamDestroy		cuStreamDestroy_v2
#define cuEventDestroy		cuEventDestroy_v2
#define cuModuleGetGlobal	cuModuleGetGlobal_v2

/* Errors */
CUresult cuGetErrorString(CUresult error, const char **str);
CUresult cuGetErrorName(CUresult error, const char **str);

/* Initialisation, version, devices */
CUresult cuInit(unsigned int flags);
CUresult cuDriverGetVersion(int *version);
CUresult cuDeviceGetCount(int *count);
CUresult cuDeviceGet(CUdevice *device, int ordinal);
CUresult cuDeviceGetName(char *name, int len, CUdevice dev);
CUresult cuDeviceGetUuid(CUuuid *uuid, CUdevice dev);
CUresult cuDeviceTotalMem_v2(size_t *bytes, CUdevice dev);
CUresult cuDeviceGetAttribute(int *value, CUdevice_attribute attrib, CUdevice dev);
CUresult cuDeviceComputeCapability(int *major, int *minor, CUdevice dev);
CUresult cuDeviceGetPCIBusId(char *busid, int len, CUdevice dev);

/* Contexts */
CUresult cuCtxCreate_v2(CUcontext *ctx, unsigned int flags, CUdevice dev);
CUresult cuCtxDestroy_v2(CUcontext ctx);
CUresult cuCtxSetCurrent(CUcontext ctx);
CUresult cuCtxGetCurrent(CUcontext *ctx);
CUresult cuCtxPushCurrent_v2(CUcontext ctx);
CUresult cuCtxPopCurrent_v2(CUcontext *ctx);
CUresult cuCtxGetDevice(CUdevice *device);
CUresult cuCtxSynchronize(void);
CUresult cuCtxGetApiVersion(CUcontext ctx, unsigned int *version);
CUresult cuCtxGetLimit(size_t *value, CUlimit limit);
CUresult cuCtxSetLimit(CUlimit limit, size_t value);
CUresult cuCtxGetCacheConfig(CUfunc_cache *config);
CUresult cuCtxSetCacheConfig(CUfunc_cache config);
CUresult cuDevicePrimaryCtxRetain(CUcontext *ctx, CUdevice dev);
CUresult cuDevicePrimaryCtxRelease_v2(CUdevice dev);
CUresult cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int *flags, int *active);
CUresult cuDevicePrimaryCtxSetFlags_v2(CUdevice dev, unsigned int flags);
CUresult cuDevicePrimaryCtxReset_v2(CUdevice dev);

/* Memory */
CUresult cuMemGetInfo_v2(size_t *free, size_t *total);
CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytes);
CUresult cuMemFree_v2(CUdeviceptr dptr);
CUresult cuMemGetAddressRange_v2(CUdeviceptr *base, size_t *size, CUdeviceptr dptr);
CUresult cuMemAllocHost_v2(void **pp, size_t bytes);
CUresult cuMemHostAlloc(void **pp, size_t bytes, unsigned int flags);
CUresult cuMemFreeHost(void *p);
CUresult cuMemHostGetDevicePointer_v2(CUdeviceptr *dptr, void *p, unsigned int flags);
CUresult cuMemcpyHtoD_v2(CUdeviceptr dst, const void *src, size_t bytes);
CUresult cuMemcpyDtoH_v2(void *dst, CUdeviceptr src, size_t bytes);
CUresult cuMemcpyDtoD_v2(CUdeviceptr dst, CUdeviceptr src, size_t bytes);
CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr dst, const void *src, size_t bytes, CUstream stream);
CUresult cuMemcpyDtoHAsync_v2(void *dst, CUdeviceptr src, size_t bytes, CUstream stream);
CUresult cuMemcpyDtoDAsync_v2(CUdeviceptr dst, CUdeviceptr src, size_t bytes, CUstream stream);
CUresult cuMemsetD8_v2(CUdeviceptr dst, unsigned char value, size_t count);
CUresult cuMemsetD16_v2(CUdeviceptr dst, unsigned short value, size_t count);
CUresult cuMemsetD32_v2(CUdeviceptr dst, unsigned int value, size_t count);
CUresult cuMemsetD8Async(CUdeviceptr dst, unsigned char value, size_t count, CUstream stream);
CUresult cuMemsetD32Async(CUdeviceptr dst, unsigned int value, size_t count, CUstream stream);

/* Streams and events */
CUresult cuStreamCreate(CUstream *stream, unsigned int flags);
CUresult cuStreamDestroy_v2(CUstream stream);
CUresult cuStreamSynchronize(CUstream stream);
CUresult cuStreamQuery(CUstream stream);
CUresult cuStreamWaitEvent(CUstream stream, CUevent event, unsigned int flags);
CUresult cuEventCreate(CUevent *event, unsigned int flags);
CUresult cuEventDestroy_v2(CUevent event);
CUresult cuEventRecord(CUevent event, CUstream stream);
CUresult cuEventSynchronize(CUevent event);
CUresult cuEventQuery(CUevent event);
CUresult cuEventElapsedTime(float *ms, CUevent start, CUevent end);

/* Textures: objects on linear or pitched device memory (CUDA_RESOURCE_DESC / CUDA_TEXTURE_DESC) */
typedef unsigned long long CUtexObject;
typedef struct CUarray_st *CUarray;
typedef struct CUmipmappedArray_st *CUmipmappedArray;

typedef enum CUarray_format_enum {
	CU_AD_FORMAT_UNSIGNED_INT8 = 0x01,
	CU_AD_FORMAT_UNSIGNED_INT16 = 0x02,
	CU_AD_FORMAT_UNSIGNED_INT32 = 0x03,
	CU_AD_FORMAT_SIGNED_INT8 = 0x08,
	CU_AD_FORMAT_SIGNED_INT16 = 0x09,
	CU_AD_FORMAT_SIGNED_INT32 = 0x0a,
	CU_AD_FORMAT_HALF = 0x10,
	CU_AD_FORMAT_FLOAT = 0x20,
} CUarray_format;

typedef enum CUaddress_mode_enum {
	CU_TR_ADDRESS_MODE_WRAP = 0,
	CU_TR_ADDRESS_MODE_CLAMP = 1,
	CU_TR_ADDRESS_MODE_MIRROR = 2,
	CU_TR_ADDRESS_MODE_BORDER = 3,
} CUaddress_mode;

typedef enum CUfilter_mode_enum {
	CU_TR_FILTER_MODE_POINT = 0,
	CU_TR_FILTER_MODE_LINEAR = 1,
} CUfilter_mode;

typedef enum CUresourcetype_enum {
	CU_RESOURCE_TYPE_ARRAY = 0x00,
	CU_RESOURCE_TYPE_MIPMAPPED_ARRAY = 0x01,
	CU_RESOURCE_TYPE_LINEAR = 0x02,
	CU_RESOURCE_TYPE_PITCH2D = 0x03,
} CUresourcetype;

#define CU_TRSF_READ_AS_INTEGER		0x01
#define CU_TRSF_NORMALIZED_COORDINATES	0x02
#define CU_TRSF_SRGB			0x10

typedef struct CUDA_RESOURCE_DESC_st {
	CUresourcetype resType;
	union {
		struct { CUarray hArray; } array;
		struct { CUmipmappedArray hMipmappedArray; } mipmap;
		struct { CUdeviceptr devPtr; CUarray_format format; unsigned int numChannels; size_t sizeInBytes; } linear;
		struct { CUdeviceptr devPtr; CUarray_format format; unsigned int numChannels; size_t width;
			 size_t height; size_t pitchInBytes; } pitch2D;
		struct { int reserved[32]; } reserved;
	} res;
	unsigned int flags;
} CUDA_RESOURCE_DESC;

typedef struct CUDA_TEXTURE_DESC_st {
	CUaddress_mode addressMode[3];
	CUfilter_mode filterMode;
	unsigned int flags;
	unsigned int maxAnisotropy;
	CUfilter_mode mipmapFilterMode;
	float mipmapLevelBias;
	float minMipmapLevelClamp;
	float maxMipmapLevelClamp;
	float borderColor[4];
	int reserved[12];
} CUDA_TEXTURE_DESC;

typedef struct CUDA_RESOURCE_VIEW_DESC_st {
	int format;
	size_t width, height, depth;
	unsigned int firstMipmapLevel, lastMipmapLevel, firstLayer, lastLayer;
	unsigned int reserved[16];
} CUDA_RESOURCE_VIEW_DESC;

CUresult cuTexObjectCreate(CUtexObject *tex, const CUDA_RESOURCE_DESC *res, const CUDA_TEXTURE_DESC *desc,
			   const CUDA_RESOURCE_VIEW_DESC *view);
CUresult cuTexObjectDestroy(CUtexObject tex);
CUresult cuTexObjectGetResourceDesc(CUDA_RESOURCE_DESC *res, CUtexObject tex);
CUresult cuTexObjectGetTextureDesc(CUDA_TEXTURE_DESC *desc, CUtexObject tex);

/* 2D copies */
typedef struct CUDA_MEMCPY2D_st {
	size_t srcXInBytes, srcY;
	CUmemorytype srcMemoryType;
	const void *srcHost;
	CUdeviceptr srcDevice;
	CUarray srcArray;
	size_t srcPitch;
	size_t dstXInBytes, dstY;
	CUmemorytype dstMemoryType;
	void *dstHost;
	CUdeviceptr dstDevice;
	CUarray dstArray;
	size_t dstPitch;
	size_t WidthInBytes, Height;
} CUDA_MEMCPY2D;

#define cuMemcpy2D		cuMemcpy2D_v2
#define cuMemcpy2DUnaligned	cuMemcpy2DUnaligned_v2
#define cuMemcpy2DAsync		cuMemcpy2DAsync_v2
#define cuMemAllocPitch		cuMemAllocPitch_v2
CUresult cuMemcpy2D_v2(const CUDA_MEMCPY2D *copy);
CUresult cuMemcpy2DUnaligned_v2(const CUDA_MEMCPY2D *copy);
CUresult cuMemcpy2DAsync_v2(const CUDA_MEMCPY2D *copy, CUstream stream);
CUresult cuMemAllocPitch_v2(CUdeviceptr *dptr, size_t *pitch, size_t width_bytes, size_t height,
			    unsigned int element_bytes);

/* Modules and launches (milestone 2: PTX through an open compiler) */
CUresult cuModuleLoadData(CUmodule *module, const void *image);
CUresult cuModuleLoadDataEx(CUmodule *module, const void *image, unsigned int n, CUjit_option *options,
			    void **values);
CUresult cuModuleLoad(CUmodule *module, const char *path);
CUresult cuModuleUnload(CUmodule module);
CUresult cuModuleGetFunction(CUfunction *func, CUmodule module, const char *name);
CUresult cuModuleGetGlobal_v2(CUdeviceptr *dptr, size_t *bytes, CUmodule module, const char *name);
CUresult cuFuncGetAttribute(int *value, CUfunction_attribute attrib, CUfunction func);
CUresult cuFuncSetCacheConfig(CUfunction func, CUfunc_cache config);
CUresult cuLaunchKernel(CUfunction f, unsigned int gx, unsigned int gy, unsigned int gz, unsigned int bx,
			unsigned int by, unsigned int bz, unsigned int shared_bytes, CUstream stream, void **params,
			void **extra);

#ifdef __cplusplus
}
#endif

#endif
