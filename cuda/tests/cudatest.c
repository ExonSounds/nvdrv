// SPDX-License-Identifier: MIT
/*
 * nvcuda milestone 1 test: devices, contexts, memory, copies, memsets, pinned memory, streams,
 * events, error handling. Uses only the CUDA Driver API.
 *   make && LD_LIBRARY_PATH=build build/cudatest
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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

static double now(void)
{
	struct timespec t;

	clock_gettime(CLOCK_MONOTONIC, &t);
	return t.tv_sec + t.tv_nsec * 1e-9;
}

static int check_u32(const uint32_t *p, size_t n, uint32_t (*f)(size_t))
{
	for (size_t i = 0; i < n; i++)
		if (p[i] != f(i)) {
			printf("    mismatch at %zu: 0x%08x != 0x%08x\n", i, p[i], f(i));
			return 0;
		}
	return 1;
}

static uint32_t pat(size_t i) { return (uint32_t)(i * 2654435761u) ^ 0x5a5a1234; }
static uint32_t pat2(size_t i) { return (uint32_t)i * 7 + 3; }

int main(void)
{
	char name[128], bus[32];
	int ver, n, major, minor, sms, v;
	size_t mem, fr, tot, fr2;
	CUdevice dev;
	CUcontext ctx;
	CUuuid uuid;

	CK(cuInit(0));
	CK(cuDriverGetVersion(&ver));
	CK(cuDeviceGetCount(&n));
	CK(cuDeviceGet(&dev, 0));
	CK(cuDeviceGetName(name, sizeof(name), dev));
	CK(cuDeviceTotalMem(&mem, dev));
	CK(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev));
	CK(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev));
	CK(cuDeviceGetAttribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));
	CK(cuDeviceGetPCIBusId(bus, sizeof(bus), dev));
	CK(cuDeviceGetUuid(&uuid, dev));
	printf("driver API %d, %d device(s): \"%s\" (%s), %zu MiB, compute %d.%d, %d SMs\n", ver, n, name, bus,
	       mem >> 20, major, minor, sms);
	TEST(ver >= 13000 && n >= 1 && major == 8 && minor == 6 && sms > 0, "device query");
	TEST(cuDeviceGetAttribute(&v, 0, dev) == CUDA_ERROR_INVALID_VALUE, "unknown attribute rejected");

	CK(cuCtxCreate(&ctx, 0, dev));
	CK(cuMemGetInfo(&fr, &tot));
	printf("memory: %zu MiB free of %zu MiB\n", fr >> 20, tot >> 20);

	/* pageable copies, larger than the staging buffer */
	const size_t N = 64u << 20, W = N / 4;
	uint32_t *h = malloc(N), *h2 = malloc(N);
	CUdeviceptr d, d2;
	for (size_t i = 0; i < W; i++)
		h[i] = pat(i);
	CK(cuMemAlloc(&d, N));
	CK(cuMemAlloc(&d2, N));
	printf("device pointers 0x%llx, 0x%llx\n", d, d2);
	CK(cuMemGetInfo(&fr2, &tot));
	TEST(fr - fr2 >= 2 * N, "cuMemGetInfo sees 128 MiB allocated (%zu MiB less free)", (fr - fr2) >> 20);
	double t0 = now();
	CK(cuMemcpyHtoD(d, h, N));
	double t1 = now();
	memset(h2, 0, N);
	CK(cuMemcpyDtoH(h2, d, N));
	double t2 = now();
	TEST(check_u32(h2, W, pat), "64 MiB pageable HtoD + DtoH round trip (%.2f / %.2f GB/s)", N / (t1 - t0) / 1e9,
	     N / (t2 - t1) / 1e9);

	CK(cuMemcpyDtoD(d2, d, N));
	memset(h2, 0, N);
	CK(cuMemcpyDtoH(h2, d2, N));
	TEST(check_u32(h2, W, pat), "64 MiB device-to-device copy");

	/* unaligned pieces */
	CK(cuMemcpyHtoD(d + 1, (char *)h + 5, 1000003));
	memset(h2, 0, 1000003);
	CK(cuMemcpyDtoH(h2, d + 1, 1000003));
	TEST(!memcmp(h2, (char *)h + 5, 1000003), "unaligned copy (offset 1, 1000003 bytes)");

	/* memsets */
	CK(cuMemsetD32(d, 0xdeadbeef, W));
	CK(cuMemcpyDtoH(h2, d, N));
	{
		size_t i;
		for (i = 0; i < W && h2[i] == 0xdeadbeef; i++)
			;
		TEST(i == W, "cuMemsetD32 64 MiB");
	}
	CK(cuMemsetD8(d + 3, 0x7e, 4093));
	CK(cuMemcpyDtoH(h2, d, 8192));
	{
		const unsigned char *b = (const unsigned char *)h2;
		int ok = b[0] == 0xef && b[1] == 0xbe && b[2] == 0xad;
		for (int i = 3; i < 3 + 4093; i++)
			ok &= b[i] == 0x7e;
		ok &= b[4096] == 0xef;
		TEST(ok, "cuMemsetD8 at an odd offset and length (byte components)");
	}
	CK(cuMemsetD16(d, 0x1234, 8));
	CK(cuMemcpyDtoH(h2, d, 16));
	TEST(h2[0] == 0x12341234 && h2[3] == 0x12341234, "cuMemsetD16");

	/* pinned host memory */
	uint32_t *ph;
	CUdeviceptr pd;
	CK(cuMemHostAlloc((void **)&ph, N, CU_MEMHOSTALLOC_DEVICEMAP));
	CK(cuMemHostGetDevicePointer(&pd, ph, 0));
	for (size_t i = 0; i < W; i++)
		ph[i] = pat2(i);
	t0 = now();
	CK(cuMemcpyHtoD(d, ph, N));
	t1 = now();
	memset(ph, 0, N);
	CK(cuMemcpyDtoH(ph, d, N));
	t2 = now();
	TEST(check_u32(ph, W, pat2), "64 MiB pinned HtoD + DtoH (%.2f / %.2f GB/s)", N / (t1 - t0) / 1e9,
	     N / (t2 - t1) / 1e9);
	CK(cuMemsetD32(pd, 0x01020304, 16));
	TEST(ph[0] == 0x01020304 && ph[15] == 0x01020304 && ph[16] == pat2(16),
	     "pinned memory written through its device pointer 0x%llx", pd);

	/* streams and events */
	CUstream s1, s2;
	CUevent e0, e1, e2;
	float ms;
	CK(cuStreamCreate(&s1, 0));
	CK(cuStreamCreate(&s2, 0));
	CK(cuEventCreate(&e0, 0));
	CK(cuEventCreate(&e1, 0));
	CK(cuEventCreate(&e2, CU_EVENT_DISABLE_TIMING));
	for (size_t i = 0; i < W; i++)
		ph[i] = pat(i);
	CK(cuEventRecord(e0, s1));
	for (int k = 0; k < 8; k++)
		CK(cuMemcpyHtoDAsync(d, ph, N, s1));
	CK(cuEventRecord(e1, s1));
	TEST(cuStreamQuery(s1) == CUDA_ERROR_NOT_READY || cuStreamQuery(s1) == CUDA_SUCCESS, "cuStreamQuery");
	/* s2 waits for s1's copies, then copies d -> d2 */
	CK(cuStreamWaitEvent(s2, e1, 0));
	CK(cuMemcpyDtoDAsync(d2, d, N, s2));
	CK(cuEventRecord(e2, s2));
	CK(cuEventSynchronize(e2));
	CK(cuEventElapsedTime(&ms, e0, e1));
	TEST(ms > 0 && ms < 10000, "8 x 64 MiB async pinned HtoD timed by events: %.3f ms (%.2f GB/s)", ms,
	     8.0 * N / (ms / 1e3) / 1e9);
	CK(cuMemcpyDtoH(h2, d2, N));
	TEST(check_u32(h2, W, pat), "cuStreamWaitEvent orders stream 2 after stream 1");
	CK(cuStreamSynchronize(s1));
	TEST(cuStreamQuery(s1) == CUDA_SUCCESS && cuEventQuery(e1) == CUDA_SUCCESS, "stream and event idle");

	/* errors and modules */
	const char *es;
	CUmodule mod;
	CK(cuGetErrorString(CUDA_ERROR_OUT_OF_MEMORY, &es));
	TEST(!strcmp(es, "out of memory"), "cuGetErrorString");
	TEST(cuModuleLoadData(&mod, "\x7f" "ELF\2\1\1") == CUDA_ERROR_NO_BINARY_FOR_GPU, "cubin images refused");
	CUdeviceptr big;
	TEST(cuMemAlloc(&big, (size_t)64 << 30) == CUDA_ERROR_OUT_OF_MEMORY, "64 GiB allocation fails cleanly");
	TEST(cuMemcpyHtoD(0x1234, h, 16) == CUDA_ERROR_INVALID_VALUE, "copy to an invalid device pointer refused");

	CK(cuEventDestroy(e0));
	CK(cuEventDestroy(e1));
	CK(cuEventDestroy(e2));
	CK(cuStreamDestroy(s1));
	CK(cuStreamDestroy(s2));
	CK(cuMemFreeHost(ph));
	CK(cuMemFree(d));
	CK(cuMemFree(d2));
	CK(cuMemGetInfo(&fr2, &tot));
	TEST(fr2 + (1u << 20) >= fr, "memory returned after free (%zu MiB free)", fr2 >> 20);
	CK(cuCtxDestroy(ctx));

	/* primary context, as runtime-style programs use it */
	CUcontext p1, p2;
	unsigned int flags;
	int active;
	CK(cuDevicePrimaryCtxRetain(&p1, dev));
	CK(cuDevicePrimaryCtxRetain(&p2, dev));
	CK(cuDevicePrimaryCtxGetState(dev, &flags, &active));
	CK(cuCtxPushCurrent(p1));
	CK(cuMemAlloc(&d, 4096));
	CK(cuMemsetD32(d, 42, 1024));
	CK(cuMemcpyDtoH(h2, d, 4096));
	CK(cuCtxPopCurrent(NULL));
	CK(cuDevicePrimaryCtxRelease(dev));
	CK(cuDevicePrimaryCtxRelease(dev));
	TEST(p1 == p2 && active && h2[1023] == 42, "primary context retained twice, used, released");

	printf("%d/%d passed\n", total - fails, total);
	return fails != 0;
}
