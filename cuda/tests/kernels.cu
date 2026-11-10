// SPDX-License-Identifier: MIT
/* Test kernels for nvcuda (CUDA C++ -> clang -> SPIR-V -> NAK). Built by the Makefile. */

extern "C" __global__ void saxpy(float a, const float *x, float *y, unsigned int n)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n)
		y[i] = a * x[i] + y[i];
}

/* each block reverses its 256 elements through shared memory */
extern "C" __global__ void reverse256(unsigned int *d)
{
	__shared__ unsigned int t[256];
	unsigned int base = blockIdx.x * 256;

	t[threadIdx.x] = d[base + threadIdx.x];
	__syncthreads();
	d[base + threadIdx.x] = t[255 - threadIdx.x];
}

/* 2D grid and block: out[y][x] = x * 65536 + y */
extern "C" __global__ void coords(unsigned int *out, unsigned int width, unsigned int height)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height)
		out[y * width + x] = x * 65536 + y;
}

/* a per-thread array indexed at run time: lives in local memory */
extern "C" __global__ void local_array(unsigned int *out, unsigned int k)
{
	unsigned int a[64];
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	for (unsigned int j = 0; j < 64; j++)
		a[j] = i * j;
	out[i] = a[(i + k) & 63];
}

/* global atomics: every thread adds its index */
extern "C" __global__ void atomic_sum(unsigned long long *sum, unsigned int n)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n)
		__atomic_fetch_add(sum, (unsigned long long)i, __ATOMIC_RELAXED);
}
