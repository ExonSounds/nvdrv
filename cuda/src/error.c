// SPDX-License-Identifier: MIT
/* cuGetErrorName / cuGetErrorString (names generated from cuda_enums.h) */
#include <stddef.h>
#include "internal.h"

static const struct { CUresult code; const char *name, *text; } errors[] = {
	{ CUDA_SUCCESS, "CUDA_SUCCESS", "no error" },
	{ CUDA_ERROR_INVALID_VALUE, "CUDA_ERROR_INVALID_VALUE", "invalid argument" },
	{ CUDA_ERROR_OUT_OF_MEMORY, "CUDA_ERROR_OUT_OF_MEMORY", "out of memory" },
	{ CUDA_ERROR_NOT_INITIALIZED, "CUDA_ERROR_NOT_INITIALIZED", "initialization error" },
	{ CUDA_ERROR_DEINITIALIZED, "CUDA_ERROR_DEINITIALIZED", "deinitialized" },
	{ CUDA_ERROR_PROFILER_DISABLED, "CUDA_ERROR_PROFILER_DISABLED", "profiler disabled" },
	{ CUDA_ERROR_PROFILER_NOT_INITIALIZED, "CUDA_ERROR_PROFILER_NOT_INITIALIZED", "profiler not initialized" },
	{ CUDA_ERROR_PROFILER_ALREADY_STARTED, "CUDA_ERROR_PROFILER_ALREADY_STARTED", "profiler already started" },
	{ CUDA_ERROR_PROFILER_ALREADY_STOPPED, "CUDA_ERROR_PROFILER_ALREADY_STOPPED", "profiler already stopped" },
	{ CUDA_ERROR_DEVICE_UNAVAILABLE, "CUDA_ERROR_DEVICE_UNAVAILABLE", "CUDA-capable device(s) is/are busy or unavailable" },
	{ CUDA_ERROR_NO_DEVICE, "CUDA_ERROR_NO_DEVICE", "no CUDA-capable device is detected" },
	{ CUDA_ERROR_INVALID_DEVICE, "CUDA_ERROR_INVALID_DEVICE", "invalid device ordinal" },
	{ CUDA_ERROR_INVALID_IMAGE, "CUDA_ERROR_INVALID_IMAGE", "device kernel image is invalid" },
	{ CUDA_ERROR_INVALID_CONTEXT, "CUDA_ERROR_INVALID_CONTEXT", "invalid device context" },
	{ CUDA_ERROR_CONTEXT_ALREADY_CURRENT, "CUDA_ERROR_CONTEXT_ALREADY_CURRENT", "context already current" },
	{ CUDA_ERROR_MAP_FAILED, "CUDA_ERROR_MAP_FAILED", "map failed" },
	{ CUDA_ERROR_UNMAP_FAILED, "CUDA_ERROR_UNMAP_FAILED", "unmap failed" },
	{ CUDA_ERROR_ARRAY_IS_MAPPED, "CUDA_ERROR_ARRAY_IS_MAPPED", "array is mapped" },
	{ CUDA_ERROR_ALREADY_MAPPED, "CUDA_ERROR_ALREADY_MAPPED", "already mapped" },
	{ CUDA_ERROR_NO_BINARY_FOR_GPU, "CUDA_ERROR_NO_BINARY_FOR_GPU", "no kernel image is available for execution on the device" },
	{ CUDA_ERROR_ALREADY_ACQUIRED, "CUDA_ERROR_ALREADY_ACQUIRED", "already acquired" },
	{ CUDA_ERROR_NOT_MAPPED, "CUDA_ERROR_NOT_MAPPED", "not mapped" },
	{ CUDA_ERROR_NOT_MAPPED_AS_ARRAY, "CUDA_ERROR_NOT_MAPPED_AS_ARRAY", "not mapped as array" },
	{ CUDA_ERROR_NOT_MAPPED_AS_POINTER, "CUDA_ERROR_NOT_MAPPED_AS_POINTER", "not mapped as pointer" },
	{ CUDA_ERROR_ECC_UNCORRECTABLE, "CUDA_ERROR_ECC_UNCORRECTABLE", "ecc uncorrectable" },
	{ CUDA_ERROR_UNSUPPORTED_LIMIT, "CUDA_ERROR_UNSUPPORTED_LIMIT", "unsupported limit" },
	{ CUDA_ERROR_CONTEXT_ALREADY_IN_USE, "CUDA_ERROR_CONTEXT_ALREADY_IN_USE", "context already in use" },
	{ CUDA_ERROR_PEER_ACCESS_UNSUPPORTED, "CUDA_ERROR_PEER_ACCESS_UNSUPPORTED", "peer access unsupported" },
	{ CUDA_ERROR_INVALID_PTX, "CUDA_ERROR_INVALID_PTX", "invalid ptx" },
	{ CUDA_ERROR_INVALID_GRAPHICS_CONTEXT, "CUDA_ERROR_INVALID_GRAPHICS_CONTEXT", "invalid graphics context" },
	{ CUDA_ERROR_NVLINK_UNCORRECTABLE, "CUDA_ERROR_NVLINK_UNCORRECTABLE", "nvlink uncorrectable" },
	{ CUDA_ERROR_JIT_COMPILER_NOT_FOUND, "CUDA_ERROR_JIT_COMPILER_NOT_FOUND", "jit compiler not found" },
	{ CUDA_ERROR_UNSUPPORTED_PTX_VERSION, "CUDA_ERROR_UNSUPPORTED_PTX_VERSION", "unsupported ptx version" },
	{ CUDA_ERROR_INVALID_SOURCE, "CUDA_ERROR_INVALID_SOURCE", "invalid source" },
	{ CUDA_ERROR_FILE_NOT_FOUND, "CUDA_ERROR_FILE_NOT_FOUND", "file not found" },
	{ CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND, "CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND", "shared object symbol not found" },
	{ CUDA_ERROR_SHARED_OBJECT_INIT_FAILED, "CUDA_ERROR_SHARED_OBJECT_INIT_FAILED", "shared object init failed" },
	{ CUDA_ERROR_OPERATING_SYSTEM, "CUDA_ERROR_OPERATING_SYSTEM", "operating system" },
	{ CUDA_ERROR_INVALID_HANDLE, "CUDA_ERROR_INVALID_HANDLE", "invalid resource handle" },
	{ CUDA_ERROR_NOT_FOUND, "CUDA_ERROR_NOT_FOUND", "named symbol not found" },
	{ CUDA_ERROR_NOT_READY, "CUDA_ERROR_NOT_READY", "device not ready" },
	{ CUDA_ERROR_ILLEGAL_ADDRESS, "CUDA_ERROR_ILLEGAL_ADDRESS", "illegal address" },
	{ CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES, "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES", "launch out of resources" },
	{ CUDA_ERROR_LAUNCH_TIMEOUT, "CUDA_ERROR_LAUNCH_TIMEOUT", "launch timeout" },
	{ CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING, "CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING", "launch incompatible texturing" },
	{ CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED, "CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED", "peer access already enabled" },
	{ CUDA_ERROR_PEER_ACCESS_NOT_ENABLED, "CUDA_ERROR_PEER_ACCESS_NOT_ENABLED", "peer access not enabled" },
	{ CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE, "CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE", "primary context active" },
	{ CUDA_ERROR_CONTEXT_IS_DESTROYED, "CUDA_ERROR_CONTEXT_IS_DESTROYED", "context is destroyed" },
	{ CUDA_ERROR_ASSERT, "CUDA_ERROR_ASSERT", "assert" },
	{ CUDA_ERROR_TOO_MANY_PEERS, "CUDA_ERROR_TOO_MANY_PEERS", "too many peers" },
	{ CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED, "CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED", "host memory already registered" },
	{ CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED, "CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED", "host memory not registered" },
	{ CUDA_ERROR_HARDWARE_STACK_ERROR, "CUDA_ERROR_HARDWARE_STACK_ERROR", "hardware stack error" },
	{ CUDA_ERROR_ILLEGAL_INSTRUCTION, "CUDA_ERROR_ILLEGAL_INSTRUCTION", "illegal instruction" },
	{ CUDA_ERROR_MISALIGNED_ADDRESS, "CUDA_ERROR_MISALIGNED_ADDRESS", "misaligned address" },
	{ CUDA_ERROR_INVALID_ADDRESS_SPACE, "CUDA_ERROR_INVALID_ADDRESS_SPACE", "invalid address space" },
	{ CUDA_ERROR_INVALID_PC, "CUDA_ERROR_INVALID_PC", "invalid pc" },
	{ CUDA_ERROR_LAUNCH_FAILED, "CUDA_ERROR_LAUNCH_FAILED", "unspecified launch failure" },
	{ CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE, "CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE", "cooperative launch too large" },
	{ CUDA_ERROR_NOT_PERMITTED, "CUDA_ERROR_NOT_PERMITTED", "not permitted" },
	{ CUDA_ERROR_NOT_SUPPORTED, "CUDA_ERROR_NOT_SUPPORTED", "operation not supported" },
	{ CUDA_ERROR_UNKNOWN, "CUDA_ERROR_UNKNOWN", "unknown error" },
};

static int find(CUresult e)
{
	for (size_t i = 0; i < sizeof(errors) / sizeof(errors[0]); i++)
		if (errors[i].code == e)
			return (int)i;
	return -1;
}

CUAPI CUresult cuGetErrorName(CUresult error, const char **str)
{
	int i = find(error);

	if (!str)
		return CUDA_ERROR_INVALID_VALUE;
	*str = i < 0 ? NULL : errors[i].name;
	return i < 0 ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
}

CUAPI CUresult cuGetErrorString(CUresult error, const char **str)
{
	int i = find(error);

	if (!str)
		return CUDA_ERROR_INVALID_VALUE;
	*str = i < 0 ? NULL : errors[i].text;
	return i < 0 ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
}
