// SPDX-License-Identifier: MIT
/*
 * nvcuda milestone 2 test: kernels written in CUDA C++, compiled by clang to SPIR-V
 * (build/kernels.spv), loaded with cuModuleLoad, compiled by NAK and launched.
 *   make && LD_LIBRARY_PATH=build build/kerneltest
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cuda.h"

static int fails, total;

#define CK(call) do {										\
	CUresult r_ = (call);									\
	if (r_ != CUDA_SUCCESS) {								\
		const char *n_ = NULL;								\
		cuGetErrorName(r_, &n_);							\
		printf("FAIL %s:%d %s -> %d %s\n", __FILE__, __LINE__, #call, r_, n_ ? n_ : "?"); \
		exit(1);									\
	}											\
} while (0)

#define TEST(cond, ...) do { int c_ = (cond); total++; fails += !c_;				\
	printf("  %-4s ", c_ ? "ok" : "FAIL"); printf(__VA_ARGS__); printf("\n"); } while (0)

int main(int argc, char **argv)
{
	const char *spv = argc > 1 ? argv[1] : "build/kernels.spv";
	CUdevice dev;
	CUcontext ctx;
	CUmodule mod;
	CUfunction saxpy, rev, coords, local, asum;
	int regs, maxt, lmem, smem;

	CK(cuInit(0));
	CK(cuDeviceGet(&dev, 0));
	CK(cuCtxCreate(&ctx, 0, dev));
	CK(cuModuleLoad(&mod, spv));
	CK(cuModuleGetFunction(&saxpy, mod, "saxpy"));
	CK(cuModuleGetFunction(&rev, mod, "reverse256"));
	CK(cuModuleGetFunction(&coords, mod, "coords"));
	CK(cuModuleGetFunction(&local, mod, "local_array"));
	CK(cuModuleGetFunction(&asum, mod, "atomic_sum"));
	CK(cuFuncGetAttribute(&regs, CU_FUNC_ATTRIBUTE_NUM_REGS, saxpy));
	CK(cuFuncGetAttribute(&maxt, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, saxpy));
	printf("saxpy: %d registers, up to %d threads per block\n", regs, maxt);
	CK(cuFuncGetAttribute(&smem, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, rev));
	CK(cuFuncGetAttribute(&lmem, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, local));
	printf("reverse256: %d bytes shared; local_array: %d bytes local per thread\n", smem, lmem);
	TEST(smem == 1024, "static shared memory size");
	TEST(cuModuleGetFunction(&saxpy, mod, "no_such_kernel") == CUDA_ERROR_NOT_FOUND, "unknown kernel name");
	CK(cuModuleGetFunction(&saxpy, mod, "saxpy"));

	/* saxpy over 16M floats */
	const unsigned int N = 16u << 20;
	float *x = malloc(N * 4), *y = malloc(N * 4), a = 2.5f;
	CUdeviceptr dx, dy;
	for (unsigned int i = 0; i < N; i++) {
		x[i] = (float)(i % 1000);
		y[i] = (float)(i % 77);
	}
	CK(cuMemAlloc(&dx, N * 4));
	CK(cuMemAlloc(&dy, N * 4));
	CK(cuMemcpyHtoD(dx, x, N * 4));
	CK(cuMemcpyHtoD(dy, y, N * 4));
	{
		void *args[] = { &a, &dx, &dy, (void *)&N };
		CUevent e0, e1;
		float ms;

		CK(cuEventCreate(&e0, 0));
		CK(cuEventCreate(&e1, 0));
		CK(cuEventRecord(e0, NULL));
		CK(cuLaunchKernel(saxpy, N / 256, 1, 1, 256, 1, 1, 0, NULL, args, NULL));
		CK(cuEventRecord(e1, NULL));
		CK(cuCtxSynchronize());
		CK(cuEventElapsedTime(&ms, e0, e1));
		CK(cuMemcpyDtoH(y, dy, N * 4));
		unsigned int bad = 0;
		for (unsigned int i = 0; i < N; i++)
			if (y[i] != a * (float)(i % 1000) + (float)(i % 77))
				bad++;
		TEST(!bad, "saxpy, 16M elements, 65536 blocks of 256 (%u wrong): %.3f ms with launch latency", bad, ms);
		/* throughput: 20 launches back to back on one stream */
		CK(cuEventRecord(e0, NULL));
		for (int k = 0; k < 20; k++)
			CK(cuLaunchKernel(saxpy, N / 256, 1, 1, 256, 1, 1, 0, NULL, args, NULL));
		CK(cuEventRecord(e1, NULL));
		CK(cuEventSynchronize(e1));
		CK(cuEventElapsedTime(&ms, e0, e1));
		printf("       20 x saxpy: %.3f ms each, %.1f GB/s\n", ms / 20, 20 * 3.0 * N * 4 / (ms / 1e3) / 1e9);
		for (unsigned int i = 0; i < N; i++)
			y[i] = (float)(i % 77);
		CK(cuMemcpyHtoD(dy, y, N * 4));
	}

	/* shared memory and __syncthreads */
	{
		const unsigned int B = 4096, M = B * 256;
		unsigned int *h = malloc(M * 4);
		CUdeviceptr d;
		for (unsigned int i = 0; i < M; i++)
			h[i] = i;
		CK(cuMemAlloc(&d, M * 4));
		CK(cuMemcpyHtoD(d, h, M * 4));
		void *args[] = { &d };
		CK(cuLaunchKernel(rev, B, 1, 1, 256, 1, 1, 0, NULL, args, NULL));
		CK(cuMemcpyDtoH(h, d, M * 4));
		unsigned int bad = 0;
		for (unsigned int i = 0; i < M; i++)
			if (h[i] != (i & ~255u) + 255 - (i & 255))
				bad++;
		TEST(!bad, "reverse256: 4096 blocks reversed through shared memory (%u wrong)", bad);
		CK(cuMemFree(d));
		free(h);
	}

	/* 2D grid */
	{
		const unsigned int W = 1000, H = 700;
		unsigned int *h = malloc(W * H * 4);
		CUdeviceptr d;
		CK(cuMemAlloc(&d, W * H * 4));
		CK(cuMemsetD32(d, 0, W * H));
		void *args[] = { &d, (void *)&W, (void *)&H };
		CK(cuLaunchKernel(coords, (W + 15) / 16, (H + 15) / 16, 1, 16, 16, 1, 0, NULL, args, NULL));
		CK(cuMemcpyDtoH(h, d, W * H * 4));
		unsigned int bad = 0;
		for (unsigned int yy = 0; yy < H; yy++)
			for (unsigned int xx = 0; xx < W; xx++)
				if (h[yy * W + xx] != xx * 65536 + yy)
					bad++;
		TEST(!bad, "coords: 2D grid 63x44 of 16x16 blocks (%u wrong)", bad);
		CK(cuMemFree(d));
		free(h);
	}

	/* local memory */
	{
		const unsigned int T = 1 << 20, k = 13;
		unsigned int *h = malloc(T * 4);
		CUdeviceptr d;
		CK(cuMemAlloc(&d, T * 4));
		void *args[] = { &d, (void *)&k };
		CK(cuLaunchKernel(local, T / 128, 1, 1, 128, 1, 1, 0, NULL, args, NULL));
		CK(cuMemcpyDtoH(h, d, T * 4));
		unsigned int bad = 0;
		for (unsigned int i = 0; i < T; i++)
			if (h[i] != i * ((i + k) & 63))
				bad++;
		TEST(!bad, "local_array: per-thread array in local memory, 1M threads (%u wrong)", bad);
		CK(cuMemFree(d));
		free(h);
	}

	/* atomics */
	{
		const unsigned int T = 1 << 20;
		unsigned long long sum = 0;
		CUdeviceptr d;
		CK(cuMemAlloc(&d, 8));
		CK(cuMemsetD32(d, 0, 2));
		void *args[] = { &d, (void *)&T };
		CK(cuLaunchKernel(asum, T / 256, 1, 1, 256, 1, 1, 0, NULL, args, NULL));
		CK(cuMemcpyDtoH(&sum, d, 8));
		TEST(sum == (unsigned long long)T * (T - 1) / 2, "atomic_sum: 64-bit atomic add from 1M threads = %llu",
		     sum);
		CK(cuMemFree(d));
	}

	/* streams: a kernel on one stream, a copy on another that waits for it */
	{
		CUstream s1, s2;
		CUevent done;
		float *h2 = malloc(N * 4);
		void *args[] = { &a, &dx, &dy, (void *)&N };
		CK(cuStreamCreate(&s1, 0));
		CK(cuStreamCreate(&s2, 0));
		CK(cuEventCreate(&done, CU_EVENT_DISABLE_TIMING));
		CK(cuMemcpyHtoD(dy, y, N * 4));
		CK(cuLaunchKernel(saxpy, N / 256, 1, 1, 256, 1, 1, 0, s1, args, NULL));
		CK(cuEventRecord(done, s1));
		CK(cuStreamWaitEvent(s2, done, 0));
		CK(cuMemcpyDtoHAsync(h2, dy, N * 4, s2));
		CK(cuStreamSynchronize(s2));
		unsigned int bad = 0;
		for (unsigned int i = 0; i < N; i++)
			if (h2[i] != a * (float)(i % 1000) + y[i])
				bad++;
		TEST(!bad, "a copy on stream 2 waits for the kernel on stream 1 (%u wrong)", bad);
		CK(cuStreamDestroy(s1));
		CK(cuStreamDestroy(s2));
		free(h2);
	}

	/* launch errors */
	{
		void *args[] = { &a, &dx, &dy, (void *)&N };
		TEST(cuLaunchKernel(saxpy, 1, 1, 1, 2048, 1, 1, 0, NULL, args, NULL) == CUDA_ERROR_INVALID_VALUE,
		     "2048 threads per block refused");
		TEST(cuLaunchKernel(saxpy, 1, 1, 1, 32, 1, 1, 0, NULL, NULL, NULL) == CUDA_ERROR_INVALID_VALUE,
		     "missing parameters refused");
	}

	CK(cuModuleUnload(mod));
	CK(cuMemFree(dx));
	CK(cuMemFree(dy));
	CK(cuCtxDestroy(ctx));
	printf("%d/%d passed\n", total - fails, total);
	return fails != 0;
}
