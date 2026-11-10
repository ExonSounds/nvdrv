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

	/* rotations (funnel shifts) */
	{
		uint32_t *in = malloc(N * 4), *out = malloc(N * 16);
		int bad = 0, n = N;
		for (int i = 0; i < N; i++)
			in[i] = (uint32_t)lcg(&seed) >> (i % 20);
		CUdeviceptr di = dalloc(N * 4, in), dout = dalloc(N * 16, NULL);
		void *args[] = { &dout, &di, &n };
		CK(cuLaunchKernel(fn("rotates"), N / 128, 1, 1, 128, 1, 1, 0, NULL, args, NULL));
		CK(cuMemcpyDtoH(out, dout, N * 16));
		for (int i = 0; i < N; i++) {
			uint32_t x = in[i], y = in[(i + 3) % N], s = y & 31;
#define ROL(v, k) ((k) ? ((v) << (k)) | ((v) >> (32 - (k))) : (v))
#define ROR(v, k) ((k) ? ((v) >> (k)) | ((v) << (32 - (k))) : (v))
			uint32_t ref[4] = { ROL(x, s), ROR(x, s), ROL(x, 13), ROR(x ^ y, 7) + ROL(y, 23) };
			for (int k = 0; k < 4; k++)
				if (out[i * 4 + k] != ref[k]) {
					if (bad < 3)
						printf("    rotates[%d][%d]: 0x%x != 0x%x\n", i, k, out[i * 4 + k], ref[k]);
					bad++;
				}
		}
		TEST(!bad, "shf.l.wrap / shf.r.wrap rotations (%d wrong)", bad);
		free(in);
		free(out);
	}

	/* 8- and 16-bit memory */
	{
		uint16_t *u16 = malloc(N * 2), *gu16 = malloc(N * 2);
		uint8_t *u8 = malloc(N), *gu8 = malloc(N);
		int16_t *s16 = malloc(N * 2), *gs16 = malloc(N * 2);
		uint32_t *in = malloc(N * 4);
		int bad = 0, n = N;
		for (int i = 0; i < N; i++) {
			in[i] = (uint32_t)lcg(&seed);
			u16[i] = (uint16_t)(in[i] >> 3);
			u8[i] = (uint8_t)(in[i] >> 11);
			s16[i] = (int16_t)(in[i] >> 7);
		}
		CUdeviceptr d16 = dalloc(N * 2, u16), d8 = dalloc(N, u8), ds = dalloc(N * 2, s16), di = dalloc(N * 4, in);
		void *args[] = { &d16, &d8, &ds, &di, &n };
		CK(cuLaunchKernel(fn("small_types"), N / 128, 1, 1, 128, 1, 1, 0, NULL, args, NULL));
		CK(cuMemcpyDtoH(gu16, d16, N * 2));
		CK(cuMemcpyDtoH(gu8, d8, N));
		CK(cuMemcpyDtoH(gs16, ds, N * 2));
		for (int i = 0; i < N; i++) {
			uint16_t a = u16[i];
			uint8_t b = u8[i];
			int16_t c = s16[i];
			a = (uint16_t)(a + 1);
			if (in[i] & 1)
				a = (uint16_t)(a * 3);
			b = (uint8_t)(b | (in[i] & 0x50));
			b = (uint8_t)(b - 1);
			c = (int16_t)(c - (int16_t)(in[i] & 0xff));
			if (c < -100)
				c = (int16_t)(c / 3);
			c = (int16_t)(c + (int16_t)(a >> 3));
			if (gu16[i] != a || gu8[i] != b || gs16[i] != c) {
				if (bad < 3)
					printf("    small_types[%d]: %u/%u %u/%u %d/%d\n", i, gu16[i], a, gu8[i], b, gs16[i], c);
				bad++;
			}
		}
		TEST(!bad, "8/16-bit loads, stores and arithmetic (%d wrong)", bad);
		free(u16); free(gu16); free(u8); free(gu8); free(s16); free(gs16); free(in);
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

	/* device function calls and module-scope variables */
	{
		const int n = 4096;
		float *out = malloc(n * 4);
		uint32_t *iout = malloc(n * 4), sc = 7;
		unsigned long long hits = 0;
		int tbl[16] = { 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3 };
		float coef[8] = { 1.0f, 0.5f, 0.25f, 0.125f, 2.0f, 4.0f, 8.0f, 16.0f };
		CUdeviceptr dout = dalloc(n * 4, NULL), diout = dalloc(n * 4, NULL), gscale, ghits, gtable;
		size_t sz;
		int bad = 0, badi = 0;
		void *args[] = { &dout, &diout, (void *)&n };

		CK(cuModuleGetGlobal(&gscale, &sz, mod, "scale"));
		TEST(sz == 4, "cuModuleGetGlobal: __constant__ scale is %zu bytes", sz);
		CK(cuModuleGetGlobal(&ghits, &sz, mod, "hits"));
		CK(cuModuleGetGlobal(&gtable, &sz, mod, "table"));
		TEST(sz == 64 && cuModuleGetGlobal(&gtable, &sz, mod, "nope") == CUDA_ERROR_NOT_FOUND,
		     "cuModuleGetGlobal: table is 64 bytes, unknown names fail");
		CK(cuMemcpyHtoD(gscale, &sc, 4));
		CK(cuMemsetD32(ghits, 0, 2));
		CK(cuLaunchKernel(fn("calls"), n / 128, 1, 1, 128, 1, 1, 0, NULL, args, NULL));
		CK(cuMemcpyDtoH(out, dout, n * 4));
		CK(cuMemcpyDtoH(iout, diout, n * 4));
		CK(cuMemcpyDtoH(&hits, ghits, 8));
		for (int i = 0; i < n; i++) {
			float a[3] = { (float)i, 1.0f, 2.0f }, b[3] = { 0.5f, (float)(i & 3), -1.0f }, s = coef[i & 7], d1 = 0, d2 = 0;
			uint32_t acc = 0, want;
			for (int k = 0; k < 3; k++) {
				d1 += a[k] * s * b[k];
				d2 += b[k] * (a[k] * 0.5f);
			}
			if (!close_f(out[i], d1 + d2 * (float)sc, 1e-5))
				bad++;
			want = 0;
			for (uint32_t k = 0; k < (uint32_t)(i & 7); k++) {
				acc += tbl[(i + k) & 15];
				if (acc > 20) {
					want = acc * 2;
					goto done;
				}
			}
			want = acc;
done:
			badi += iout[i] != want;
		}
		TEST(!bad, "device functions: struct arguments and returns, __constant__ initializer and host-set value (%d wrong)",
		     bad);
		TEST(!badi, "callee with a loop and an early return, reading a __device__ table (%d wrong)", badi);
		TEST(hits == (n + 96) / 97, "__device__ variable read back through cuModuleGetGlobal: %llu", hits);
		free(out);
		free(iout);
	}

	/* 2D copies with pitches, cuMemAllocPitch */
	{
		const int W = 100, H = 37;
		size_t pitch;
		CUdeviceptr d;
		float *src = malloc(W * H * 4), *dst = calloc(W * H, 4);
		CUDA_MEMCPY2D cp = { 0 };
		int bad = 0;

		for (int i = 0; i < W * H; i++)
			src[i] = i * 0.5f;
		CK(cuMemAllocPitch(&d, &pitch, W * 4, H, 4));
		TEST(pitch >= W * 4 && pitch % 32 == 0, "cuMemAllocPitch: pitch %zu for %d-byte rows", pitch, W * 4);
		cp.srcMemoryType = CU_MEMORYTYPE_HOST;
		cp.srcHost = src;
		cp.srcPitch = W * 4;
		cp.dstMemoryType = CU_MEMORYTYPE_DEVICE;
		cp.dstDevice = d;
		cp.dstPitch = pitch;
		cp.WidthInBytes = W * 4;
		cp.Height = H;
		CK(cuMemcpy2DUnaligned(&cp));
		cp.srcMemoryType = CU_MEMORYTYPE_DEVICE;
		cp.srcDevice = d;
		cp.srcPitch = pitch;
		cp.dstMemoryType = CU_MEMORYTYPE_HOST;
		cp.dstHost = dst;
		cp.dstPitch = W * 4;
		CK(cuMemcpy2D(&cp));
		for (int i = 0; i < W * H; i++)
			bad += dst[i] != src[i];
		TEST(!bad, "cuMemcpy2D host -> pitched device -> host, %dx%d floats (%d wrong)", W, H, bad);
		/* a sub-rectangle: columns 10..29 of rows 5..14, into a pinned host buffer */
		float *pin;
		CK(cuMemHostAlloc((void **)&pin, 20 * 10 * 4, 0));
		memset(pin, 0, 20 * 10 * 4);
		cp.srcXInBytes = 10 * 4;
		cp.srcY = 5;
		cp.dstHost = pin;
		cp.dstPitch = 20 * 4;
		cp.WidthInBytes = 20 * 4;
		cp.Height = 10;
		CK(cuMemcpy2D(&cp));
		bad = 0;
		for (int y = 0; y < 10; y++)
			for (int x = 0; x < 20; x++)
				bad += pin[y * 20 + x] != src[(y + 5) * W + x + 10];
		TEST(!bad, "cuMemcpy2D sub-rectangle into pinned memory (%d wrong)", bad);
		CK(cuMemFreeHost(pin));
		/* keep d for the texture tests */

		/* textures on that pitched image: R32F, point and linear filtering, clamping */
		CUDA_RESOURCE_DESC rd = { 0 };
		CUDA_TEXTURE_DESC td = { 0 };
		CUtexObject tex;
		float *out = malloc(W * H * 16);
		rd.resType = CU_RESOURCE_TYPE_PITCH2D;
		rd.res.pitch2D.devPtr = d;
		rd.res.pitch2D.format = CU_AD_FORMAT_FLOAT;
		rd.res.pitch2D.numChannels = 1;
		rd.res.pitch2D.width = W;
		rd.res.pitch2D.height = H;
		rd.res.pitch2D.pitchInBytes = pitch;
		td.addressMode[0] = td.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP;
		td.filterMode = CU_TR_FILTER_MODE_POINT;
		CK(cuTexObjectCreate(&tex, &rd, &td, NULL));
		CUdeviceptr dout = dalloc(W * H * 16, NULL);
		float x0 = 0.5f, y0 = 0.5f, one = 1.0f;
		int w = W, h = H;
		void *args[] = { &dout, &tex, &x0, &y0, &one, &one, &w, &h };
		CK(cuLaunchKernel(fn("tex2d"), (W * H + 127) / 128, 1, 1, 128, 1, 1, 0, NULL, args, NULL));
		CK(cuMemcpyDtoH(out, dout, W * H * 16));
		bad = 0;
		for (int i = 0; i < W * H; i++)
			bad += out[i * 4] != src[i] || out[i * 4 + 1] != 0.0f || out[i * 4 + 3] != 1.0f;
		TEST(!bad, "tex.2d point sampling at texel centres, R32F pitch2D (%d wrong)", bad);
		CK(cuTexObjectDestroy(tex));
		td.filterMode = CU_TR_FILTER_MODE_LINEAR;
		CK(cuTexObjectCreate(&tex, &rd, &td, NULL));
		x0 = 1.0f;	/* between texels: the mean of two neighbours */
		CK(cuLaunchKernel(fn("tex2d"), (W * H + 127) / 128, 1, 1, 128, 1, 1, 0, NULL, args, NULL));
		CK(cuMemcpyDtoH(out, dout, W * H * 16));
		bad = 0;
		for (int y = 0; y < H; y++)
			for (int x = 0; x < W; x++) {
				float want = x + 1 < W ? 0.5f * (src[y * W + x] + src[y * W + x + 1]) : src[y * W + x];

				bad += !close_f(out[(y * W + x) * 4], want, 1e-5);
			}
		TEST(!bad, "tex.2d linear filtering between texels, clamp at the right edge (%d wrong)", bad);
		CK(cuTexObjectDestroy(tex));
		CK(cuMemFree(d));
		free(src);
		free(dst);

		/* uchar4, normalized reads and normalized coordinates */
		const int W2 = 64, H2 = 16;
		unsigned char *px = malloc(W2 * H2 * 4);
		for (int i = 0; i < W2 * H2 * 4; i++)
			px[i] = (unsigned char)(i * 7);
		CK(cuMemAllocPitch(&d, &pitch, W2 * 4, H2, 4));
		cp = (CUDA_MEMCPY2D){ .srcMemoryType = CU_MEMORYTYPE_HOST, .srcHost = px, .srcPitch = W2 * 4,
				      .dstMemoryType = CU_MEMORYTYPE_DEVICE, .dstDevice = d, .dstPitch = pitch,
				      .WidthInBytes = W2 * 4, .Height = H2 };
		CK(cuMemcpy2D(&cp));
		rd.res.pitch2D.devPtr = d;
		rd.res.pitch2D.format = CU_AD_FORMAT_UNSIGNED_INT8;
		rd.res.pitch2D.numChannels = 4;
		rd.res.pitch2D.width = W2;
		rd.res.pitch2D.height = H2;
		rd.res.pitch2D.pitchInBytes = pitch;
		td.filterMode = CU_TR_FILTER_MODE_POINT;
		td.flags = CU_TRSF_NORMALIZED_COORDINATES;
		CK(cuTexObjectCreate(&tex, &rd, &td, NULL));
		x0 = 0.5f / W2;
		y0 = 0.5f / H2;
		float dx = 1.0f / W2, dy = 1.0f / H2;
		w = W2;
		h = H2;
		void *args2[] = { &dout, &tex, &x0, &y0, &dx, &dy, &w, &h };
		CK(cuLaunchKernel(fn("tex2d"), (W2 * H2 + 127) / 128, 1, 1, 128, 1, 1, 0, NULL, args2, NULL));
		CK(cuMemcpyDtoH(out, dout, W2 * H2 * 16));
		bad = 0;
		for (int i = 0; i < W2 * H2 * 4; i++)
			bad += !close_f(out[i], px[i] / 255.0f, 1e-6);
		TEST(!bad, "tex.2d uchar4 as normalized floats, normalized coordinates (%d wrong)", bad);
		CK(cuTexObjectDestroy(tex));
		CK(cuMemFree(d));
		free(px);

		/* 1-D linear memory: tex1Dfetch */
		const int N1 = 5000;
		float *lin = malloc(N1 * 8);
		for (int i = 0; i < N1; i++) {
			lin[i * 2] = i * 0.25f;
			lin[i * 2 + 1] = (float)(i % 7);
		}
		CUdeviceptr dl = dalloc(N1 * 8, lin);
		rd = (CUDA_RESOURCE_DESC){ .resType = CU_RESOURCE_TYPE_LINEAR };
		rd.res.linear.devPtr = dl;
		rd.res.linear.format = CU_AD_FORMAT_FLOAT;
		rd.res.linear.numChannels = 2;
		rd.res.linear.sizeInBytes = N1 * 8;
		td = (CUDA_TEXTURE_DESC){ 0 };
		CK(cuTexObjectCreate(&tex, &rd, &td, NULL));
		int n1 = N1;
		void *args3[] = { &dout, &tex, &n1 };
		CK(cuLaunchKernel(fn("tex1dfetch"), (N1 + 127) / 128, 1, 1, 128, 1, 1, 0, NULL, args3, NULL));
		CK(cuMemcpyDtoH(out, dout, N1 * 4));
		bad = 0;
		for (int i = 0; i < N1; i++)
			bad += out[i] != i * 0.25f + (i % 7) * 1000.0f;
		TEST(!bad, "tex.1d fetch from linear float2 memory (%d wrong)", bad);
		CK(cuTexObjectDestroy(tex));
		TEST(cuTexObjectDestroy(tex) == CUDA_ERROR_INVALID_HANDLE, "destroyed texture objects are invalid");
		CK(cuMemFree(dl));
		CK(cuMemFree(dout));
		free(lin);
		free(out);
	}

	CK(cuModuleUnload(mod));
	CK(cuCtxDestroy(ctx));
	printf("%d/%d passed\n", total - fails, total);
	return fails != 0;
}
