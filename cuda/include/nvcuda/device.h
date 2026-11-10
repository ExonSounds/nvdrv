/* SPDX-License-Identifier: MIT */
/*
 * Device-side CUDA definitions for compiling CUDA C++ with clang to SPIR-V (no NVIDIA SDK):
 *   clang -x cuda --cuda-device-only -nocudainc -nocudalib --offload=spirv64-unknown-unknown \
 *         -include nvcuda/device.h -O2 -S -emit-llvm k.cu -o k.ll
 *   llc -mtriple=spirv64-unknown-unknown -filetype=obj k.ll -o k.spv
 * threadIdx and friends map to the SPIR-V compute builtins; __syncthreads to OpControlBarrier
 * (Workgroup scope, workgroup memory semantics), which the LLVM SPIR-V backend recognises by
 * its __spirv_ name.
 */
#ifndef NVCUDA_DEVICE_H
#define NVCUDA_DEVICE_H

#define __global__	__attribute__((global))
#define __device__	__attribute__((device))
#define __host__	__attribute__((host))
#define __shared__	__attribute__((shared))
#define __constant__	__attribute__((constant))
#define __forceinline__	__inline__ __attribute__((always_inline))

struct nvcuda_dim3 { unsigned int x, y, z; };

#define NVCUDA_SV(name, fn)								\
	struct nvcuda_##name##_t {							\
		__device__ unsigned int __get(int i) const { return (unsigned int)fn(i); } \
		__declspec(property(get = __get_x)) unsigned int x;			\
		__declspec(property(get = __get_y)) unsigned int y;			\
		__declspec(property(get = __get_z)) unsigned int z;			\
		__device__ unsigned int __get_x() const { return __get(0); }		\
		__device__ unsigned int __get_y() const { return __get(1); }		\
		__device__ unsigned int __get_z() const { return __get(2); }		\
	};										\
	extern const __device__ __attribute__((weak)) nvcuda_##name##_t name;

NVCUDA_SV(threadIdx, __builtin_spirv_local_invocation_id)
NVCUDA_SV(blockIdx, __builtin_spirv_workgroup_id)
NVCUDA_SV(blockDim, __builtin_spirv_workgroup_size)
NVCUDA_SV(gridDim, __builtin_spirv_num_workgroups)

#define warpSize 32

__device__ void __spirv_ControlBarrier(unsigned int exec, unsigned int mem, unsigned int sem);
/* Scope Workgroup = 2; semantics AcquireRelease (0x8) | WorkgroupMemory (0x100) */
static __device__ __forceinline__ void __syncthreads(void)
{
	__spirv_ControlBarrier(2, 2, 0x108);
}

#endif
