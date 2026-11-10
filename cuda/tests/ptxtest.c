// SPDX-License-Identifier: MIT
/*
 * PTX front end coverage test: runs tests/ptxops.cu (compiled to PTX by clang) and checks every
 * result against a host reference.
 *   make && LD_LIBRARY_PATH=build build/ptxtest
 */
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cuda.h"

static int fails, total;
static CUmodule mod;

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

static CUfunction fn(const char *name)
{
	CUfunction f;

	CK(cuModuleGetFunction(&f, mod, name));
	return f;
}

static CUdeviceptr dalloc(size_t n, const void *init)
{
	CUdeviceptr d;

	CK(cuMemAlloc(&d, n));
	if (init)
		CK(cuMemcpyHtoD(d, init, n));
	else
		CK(cuMemsetD8(d, 0, n));
	return d;
}

static uint64_t lcg(uint64_t *s)
{
	*s = *s * 6364136223846793005ull + 1442695040888963407ull;
	return *s ^ (*s >> 29);
}

static int close_f(double a, double b, double rel)
{
	return fabs(a - b) <= rel * fmax(fabs(b), 1e-30);
}

int main(int argc, char **argv)
{
	const char *path = argc > 1 ? argv[1] : "build/ptxops.ptx";
	CUdevice dev;
	CUcontext ctx;
	const int N = 4096;
	uint64_t seed = 12345;

	CK(cuInit(0));
	CK(cuDeviceGet(&dev, 0));
	CK(cuCtxCreate(&ctx, 0, dev));
	CK(cuModuleLoad(&mod, path));

	/* float math */
	{
		float *in = malloc(N * 4), *out = malloc(N * 8 * 4);
		int bad[8] = { 0 }, n = N;
		for (int i = 0; i < N; i++)
			in[i] = i * 0.37f + 0.5f;
		CUdeviceptr di = dalloc(N * 4, in), dout = dalloc(N * 32, NULL);
		void *args[] = { &dout, &di, &n };
		CK(cuLaunchKernel(fn("mathf"), N / 128, 1, 1, 128, 1, 1, 0, NULL, args, NULL));
		CK(cuMemcpyDtoH(out, dout, N * 32));
		for (int i = 0; i < N; i++) {
			float x = in[i], *o = out + i * 8;
			bad[0] += !close_f(o[0], sqrtf(x), 1e-6);
			bad[1] += !close_f(o[1], x / 3.0f, 2e-7);
			bad[2] += o[2] != fmaf(x, 1.5f, -2.0f);
			bad[3] += o[3] != fminf(x, 10.0f);
			bad[4] += o[4] != fmaxf(x, 10.0f);
			bad[5] += o[5] != (float)(int)(x * -7.3f);
			bad[6] += !close_f(o[6], exp2f(x * 0.01f), 2e-6);
			bad[7] += o[7] != rintf(x * 0.37f);
		}
		TEST(!bad[0] && !bad[1] && !bad[2], "f32 sqrt.rn / div.rn / fma.rn (%d %d %d wrong)", bad[0], bad[1], bad[2]);
		TEST(!bad[3] && !bad[4], "f32 min / max");
		TEST(!bad[5] && !bad[7], "cvt.rzi.s32.f32, cvt.rn.f32.s32, cvt.rni.f32.f32 (%d %d wrong)", bad[5], bad[7]);
		TEST(!bad[6], "ex2.approx.f32 (%d wrong)", bad[6]);
		free(in);
		free(out);
	}

	/* double math */
	{
		double *in = malloc(N * 8), *out = malloc(N * 32);
		int bad[4] = { 0 }, n = N;
		for (int i = 0; i < N; i++)
			in[i] = i * 1.37 + 0.25;
		CUdeviceptr di = dalloc(N * 8, in), dout = dalloc(N * 32, NULL);
		void *args[] = { &dout, &di, &n };
		CK(cuLaunchKernel(fn("mathd"), N / 128, 1, 1, 128, 1, 1, 0, NULL, args, NULL));
		CK(cuMemcpyDtoH(out, dout, N * 32));
		for (int i = 0; i < N; i++) {
			double x = in[i], *o = out + i * 4;
			bad[0] += !close_f(o[0], sqrt(x), 1e-15);
			bad[1] += !close_f(o[1], x / 7.0, 1e-15);
			bad[2] += !close_f(o[2], fma(x, x, -1.0), 1e-15);
			bad[3] += o[3] != (double)(long long)(x * -3.7);
		}
		TEST(!bad[0] && !bad[1] && !bad[2] && !bad[3], "f64 sqrt / div / fma / cvt.rzi.s64.f64 (%d %d %d %d wrong)",
		     bad[0], bad[1], bad[2], bad[3]);
		free(in);
		free(out);
	}

	/* 64-bit integers */
	{
		uint64_t *in = malloc(N * 8), *out = malloc(N * 64);
		int bad = 0, n = N;
		for (int i = 0; i < N; i++)
			in[i] = lcg(&seed) >> (i % 40);
		CUdeviceptr di = dalloc(N * 8, in), dout = dalloc(N * 64, NULL);
		void *args[] = { &dout, &di, &n };
		CK(cuLaunchKernel(fn("int64ops"), N / 128, 1, 1, 128, 1, 1, 0, NULL, args, NULL));
		CK(cuMemcpyDtoH(out, dout, N * 64));
		for (int i = 0; i < N; i++) {
			uint64_t a = in[i], b = in[i ^ 1] | 1, *o = out + i * 8;
			long long sa = (long long)a, sb = (long long)(b >> 1) - 12345;
			uint64_t ref[8] = { a * b, a / b, a % b, (a >> 13) ^ (a << 7), (uint64_t)(sa / (sb ? sb : 1)),
					    (uint64_t)(sa >> 9), (uint64_t)__builtin_popcountll(a),
					    a ? (uint64_t)__builtin_clzll(a) : 64 };
			for (int k = 0; k < 8; k++)
				if (o[k] != ref[k]) {
					if (bad < 3)
						printf("    int64ops[%d][%d]: 0x%llx != 0x%llx\n", i, k,
						       (unsigned long long)o[k], (unsigned long long)ref[k]);
					bad++;
				}
		}
		TEST(!bad, "64-bit mul / div / rem / shifts / signed div / popc / clz (%d wrong)", bad);
		free(in);
		free(out);
	}

	/* 32-bit bit operations */
	{
		uint32_t *in = malloc(N * 4), *out = malloc(N * 48);
		int bad = 0, n = N;
		for (int i = 0; i < N; i++)
			in[i] = (uint32_t)lcg(&seed) >> (i % 20);
		CUdeviceptr di = dalloc(N * 4, in), dout = dalloc(N * 48, NULL);
		void *args[] = { &dout, &di, &n };
		CK(cuLaunchKernel(fn("bits32"), N / 128, 1, 1, 128, 1, 1, 0, NULL, args, NULL));
		CK(cuMemcpyDtoH(out, dout, N * 48));
		for (int i = 0; i < N; i++) {
			uint32_t x = in[i], y = in[(i + 1) % N], *o = out + i * 12, r = 0;
			int sx = (int)x;
			for (int k = 0; k < 32; k++)
				r |= ((x >> k) & 1) << (31 - k);
			uint32_t ref[12] = { (uint32_t)__builtin_popcount(x), x ? (uint32_t)__builtin_clz(x) : 32, r,
					     (x >> 5) & 0x3ff, (uint32_t)(sx >> 3), (uint32_t)(sx / -7), (uint32_t)(sx % 13),
					     (uint32_t)(sx < (int)y ? sx : (int)y), x > y ? x : y,
					     (uint32_t)(((uint64_t)x * y) >> 32),
					     (x & 0xff) | (y & 0xff) << 8 | ((x >> 8) & 0xff) << 16 | ((y >> 8) & 0xff) << 24,
					     (x & 1) ? x * 3 + 1 : x / 2 };
			for (int k = 0; k < 12; k++)
				if (o[k] != ref[k]) {
					if (bad < 3)
						printf("    bits32[%d][%d]: 0x%x != 0x%x\n", i, k, o[k], ref[k]);
					bad++;
				}
		}
		TEST(!bad, "popc / clz / brev / bfe / shr.s / signed div+rem / min / max / mul.hi / prmt / selp (%d wrong)",
		     bad);
		free(in);
		free(out);
	}

	/* warp shuffles and votes */
	{
		uint32_t *out = calloc(1024 + 2048, 4);
		int bad = 0;
		CUdeviceptr dout = dalloc((1024 + 2048) * 4, NULL);
		void *args[] = { &dout };
		CK(cuLaunchKernel(fn("warp"), 4, 1, 1, 256, 1, 1, 0, NULL, args, NULL));
		CK(cuMemcpyDtoH(out, dout, (1024 + 2048) * 4));
		for (uint32_t w = 0; w < 32; w++) {
			uint32_t ballot = 0;
			for (uint32_t l = 0; l < 32; l++)
				ballot |= ((32 * w + l) % 3 == 0) << l;
			bad += out[w * 4] != 32 * 32 * w + 496;
			bad += out[w * 4 + 1] != 32 * w + 3;
			bad += out[w * 4 + 2] != ballot;
		}
		TEST(!bad, "shfl.sync.down reduction, shfl.sync.idx broadcast, vote.sync.ballot (%d wrong)", bad);
		bad = 0;
		for (uint32_t t = 0; t < 1024; t++) {
			bad += out[1024 + t * 2] != ((t & 31) ? t - 1 : t);
			bad += out[1024 + t * 2 + 1] != (t ^ 5);
		}
		TEST(!bad, "shfl.sync.up (lane 0 keeps its value), shfl.sync.bfly (%d wrong)", bad);
		free(out);
	}

	/* atomics */
	{
		const int B = 64;
		uint32_t *out = malloc(B * 4);
		float fs = 0;
		int im = -1000000, bad = 0;
		CUdeviceptr dout = dalloc(B * 4, NULL), dfs = dalloc(4, &fs), dim = dalloc(4, &im);
		void *args[] = { &dout, &dfs, &dim };
		CK(cuLaunchKernel(fn("atomics"), B, 1, 1, 256, 1, 1, 0, NULL, args, NULL));
		CK(cuMemcpyDtoH(out, dout, B * 4));
		CK(cuMemcpyDtoH(&fs, dfs, 4));
		CK(cuMemcpyDtoH(&im, dim, 4));
		for (int i = 0; i < B; i++)
			bad += out[i] != 256 * 257 / 2;
		TEST(!bad, "atom.shared.add per block (%d wrong)", bad);
		TEST(fs == B * 256 * 0.5f, "float add through atom.global.cas: %.1f (want %.1f)", fs, B * 256 * 0.5);
		TEST(im == (B - 1) * 1000 + 255 - 5000, "atom.global.max.s32: %d", im);
		free(out);
	}

	/* dynamic shared memory */
	{
		const int T = 128, B = 16;
		float *out = malloc(T * B * 4);
		int bad = 0;
		CUdeviceptr dout = dalloc(T * B * 4, NULL);
		void *args[] = { &dout };
		CK(cuLaunchKernel(fn("dynshared"), B, 1, 1, T, 1, 1, T * 4, NULL, args, NULL));
		CK(cuMemcpyDtoH(out, dout, T * B * 4));
		for (int b = 0; b < B; b++)
			for (int t = 0; t < T; t++)
				bad += out[b * T + t] != (float)(T - 1 - t) + 1000.0f + (t & 31);
		TEST(!bad, "extern __shared__ after a static array (%d wrong)", bad);
		free(out);
	}

	/* generic addressing */
	{
		const int B = 4;
		uint32_t *buf = malloc(B * 256 * 4);
		int bad = 0, pick = 0;
		for (int i = 0; i < B * 256; i++)
			buf[i] = 0xdead0000 + i;
		CUdeviceptr d = dalloc(B * 256 * 4, buf);
		void *args[] = { &d, &pick };
		CK(cuLaunchKernel(fn("generic"), B, 1, 1, 256, 1, 1, 0, NULL, args, NULL));
		uint32_t *out = malloc(B * 256 * 4);
		CK(cuMemcpyDtoH(out, d, B * 256 * 4));
		for (int b = 0; b < B; b++)
			for (int t = 0; t < 256; t++) {
				uint32_t want = (t & 1) ? (uint32_t)(t ^ 1) * 3 + 7 : buf[b * 256 + (t ^ 1)] + 7;
				bad += out[b * 256 + t] != want;
			}
		TEST(!bad, "generic ld through cvta.shared and global pointers (%d wrong)", bad);
		free(buf);
		free(out);
	}

	/* data-dependent loops */
	{
		uint32_t *out = malloc(N * 4);
		int bad = 0, n = N;
		CUdeviceptr d = dalloc(N * 4, NULL);
		void *args[] = { &d, &n };
		CK(cuLaunchKernel(fn("collatz"), N / 128, 1, 1, 128, 1, 1, 0, NULL, args, NULL));
		CK(cuMemcpyDtoH(out, d, N * 4));
		for (int i = 0; i < N; i++) {
			unsigned long long x = i + 1;
			uint32_t s = 0;
			while (x != 1) {
				x = (x & 1) ? 3 * x + 1 : x >> 1;
				s++;
			}
			bad += out[i] != s;
		}
		TEST(!bad, "Collatz loop, 4096 threads (%d wrong)", bad);
		free(out);
	}

	CK(cuModuleUnload(mod));
	CK(cuCtxDestroy(ctx));
	printf("%d/%d passed\n", total - fails, total);
	return fails != 0;
}
