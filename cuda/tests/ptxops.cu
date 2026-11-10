// SPDX-License-Identifier: MIT
/*
 * Instruction coverage for nvcuda's PTX front end. Compiled by clang's NVPTX back end only
 * (uses NVVM builtins). Each kernel writes results the host checks (tests/ptxtest.c).
 */
#define LANES 0xffffffffu

extern "C" __global__ void mathf(float *out, const float *in, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n)
		return;
	float x = in[i];
	out[i * 8 + 0] = __builtin_sqrtf(x);
	out[i * 8 + 1] = x / 3.0f;
	out[i * 8 + 2] = __builtin_fmaf(x, 1.5f, -2.0f);
	out[i * 8 + 3] = __builtin_fminf(x, 10.0f);
	out[i * 8 + 4] = __builtin_fmaxf(x, 10.0f);
	out[i * 8 + 5] = (float)(int)(x * -7.3f);
	out[i * 8 + 6] = __builtin_exp2f(x * 0.01f);
	out[i * 8 + 7] = __builtin_rintf(x * 0.37f);
}

extern "C" __global__ void mathd(double *out, const double *in, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n)
		return;
	double x = in[i];
	out[i * 4 + 0] = __builtin_sqrt(x);
	out[i * 4 + 1] = x / 7.0;
	out[i * 4 + 2] = __builtin_fma(x, x, -1.0);
	out[i * 4 + 3] = (double)(long long)(x * -3.7);
}

extern "C" __global__ void int64ops(unsigned long long *out, const unsigned long long *in, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n)
		return;
	unsigned long long a = in[i], b = in[i ^ 1] | 1;
	long long sa = (long long)a, sb = (long long)(b >> 1) - 12345;
	out[i * 8 + 0] = a * b;
	out[i * 8 + 1] = a / b;
	out[i * 8 + 2] = a % b;
	out[i * 8 + 3] = (a >> 13) ^ (a << 7);
	out[i * 8 + 4] = (unsigned long long)(sa / (sb ? sb : 1));
	out[i * 8 + 5] = (unsigned long long)(sa >> 9);
	out[i * 8 + 6] = __builtin_popcountll(a);
	out[i * 8 + 7] = a ? __builtin_clzll(a) : 64;
}

extern "C" __global__ void bits32(unsigned *out, const unsigned *in, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n)
		return;
	unsigned x = in[i], y = in[(i + 1) % n];
	int sx = (int)x;
	out[i * 12 + 0] = __builtin_popcount(x);
	out[i * 12 + 1] = x ? __builtin_clz(x) : 32;
	out[i * 12 + 2] = __builtin_bitreverse32(x);
	out[i * 12 + 3] = (x >> 5) & 0x3ff;
	out[i * 12 + 4] = (unsigned)(sx >> 3);
	out[i * 12 + 5] = (unsigned)(sx / -7);
	out[i * 12 + 6] = (unsigned)(sx % 13);
	out[i * 12 + 7] = (unsigned)(sx < (int)y ? sx : (int)y);
	out[i * 12 + 8] = x > y ? x : y;
	out[i * 12 + 9] = __nvvm_mulhi_ui(x, y);
	out[i * 12 + 10] = __nvvm_prmt(x, y, 0x5140);
	out[i * 12 + 11] = (x & 1) ? x * 3 + 1 : x / 2;
}

/* per-warp sum with shfl.down, broadcast with shfl.idx, ballot, shfl.up and shfl.bfly */
extern "C" __global__ void warp(unsigned *out)
{
	unsigned t = blockIdx.x * blockDim.x + threadIdx.x, lane = threadIdx.x & 31, v = t;
	for (int d = 16; d; d >>= 1)
		v += __nvvm_shfl_sync_down_i32(LANES, v, d, 0x1f);
	unsigned w = t / 32;
	unsigned bcast = __nvvm_shfl_sync_idx_i32(LANES, t, 3, 0x1f);
	unsigned ballot = __nvvm_vote_ballot_sync(LANES, (t % 3) == 0);
	unsigned up = __nvvm_shfl_sync_up_i32(LANES, t, 1, 0);
	unsigned bfly = __nvvm_shfl_sync_bfly_i32(LANES, t, 5, 0x1f);
	if (lane == 0) {
		out[w * 4 + 0] = v;
		out[w * 4 + 1] = bcast;
		out[w * 4 + 2] = ballot;
	}
	out[1024 + t * 2 + 0] = up;
	out[1024 + t * 2 + 1] = bfly;
}

/* shared-memory atomics; a CAS loop on global memory; atomic max */
extern "C" __global__ void atomics(unsigned *out, float *fsum, int *imax)
{
	__shared__ unsigned count;
	if (threadIdx.x == 0)
		count = 0;
	__syncthreads();
	__atomic_fetch_add(&count, threadIdx.x + 1, __ATOMIC_RELAXED);
	__syncthreads();
	if (threadIdx.x == 0)
		out[blockIdx.x] = count;
	/* float add through compare-and-swap */
	unsigned *p = (unsigned *)fsum, old = *(volatile unsigned *)p, assumed;
	do {
		assumed = old;
		float f = __builtin_bit_cast(float, assumed) + 0.5f;
		old = assumed;
		__atomic_compare_exchange_n(p, &old, __builtin_bit_cast(unsigned, f), false, __ATOMIC_RELAXED,
					    __ATOMIC_RELAXED);
	} while (old != assumed);
	__atomic_fetch_max(imax, (int)(blockIdx.x * 1000 + threadIdx.x) - 5000, __ATOMIC_RELAXED);
}

/* dynamic shared memory after a static array */
extern "C" __global__ void dynshared(float *out)
{
	__shared__ float stat[32];
	extern __shared__ float dyn[];
	unsigned t = threadIdx.x;
	if (t < 32)
		stat[t] = 1000.0f + t;
	dyn[t] = (float)t;
	__syncthreads();
	out[blockIdx.x * blockDim.x + t] = dyn[blockDim.x - 1 - t] + stat[t & 31];
}

/* a pointer chosen at run time between shared and global memory: generic loads and stores */
extern "C" __global__ void generic(unsigned *out, int pick)
{
	__shared__ unsigned sh[256];
	unsigned t = threadIdx.x;
	sh[t] = t * 3;
	__syncthreads();
	unsigned *volatile pv = ((t + pick) & 1) ? sh : out + blockIdx.x * 256;
	unsigned *p = pv;
	unsigned v = p[t ^ 1];
	__syncthreads();
	out[blockIdx.x * 256 + t] = v + 7;
}

/* data-dependent loop: Collatz steps */
extern "C" __global__ void collatz(unsigned *out, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n)
		return;
	unsigned long long x = i + 1;
	unsigned steps = 0;
	while (x != 1) {
		x = (x & 1) ? 3 * x + 1 : x >> 1;
		steps++;
	}
	out[i] = steps;
}

/* ---- device functions (not inlined: real PTX calls with .param passing) and module variables ---- */

struct vec3 { float x, y, z; };

__constant__ float coef[8] = { 1.0f, 0.5f, 0.25f, 0.125f, 2.0f, 4.0f, 8.0f, 16.0f };
__constant__ unsigned int scale;			/* set by the host through cuModuleGetGlobal */
__device__ unsigned long long hits;			/* read back by the host */
__device__ int table[16] = { 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3 };

__device__ __attribute__((noinline)) float dot3(vec3 a, vec3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __attribute__((noinline)) vec3 scale3(vec3 a, float s)
{
	vec3 r = { a.x * s, a.y * s, a.z * s };
	return r;
}

__device__ __attribute__((noinline)) unsigned int weird(unsigned int i)
{
	/* a loop and a conditional return inside a callee */
	unsigned int acc = 0;
	for (unsigned int k = 0; k < (i & 7); k++) {
		acc += table[(i + k) & 15];
		if (acc > 20)
			return acc * 2;
	}
	return acc;
}

extern "C" __global__ void calls(float *out, unsigned int *iout, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n)
		return;
	vec3 a = { (float)i, 1.0f, 2.0f }, b = { 0.5f, (float)(i & 3), -1.0f };
	vec3 c = scale3(a, coef[i & 7]);
	out[i] = dot3(c, b) + dot3(b, scale3(a, 0.5f)) * (float)scale;
	iout[i] = weird(i);
	if (i % 97 == 0)
		__atomic_fetch_add(&hits, 1ull, __ATOMIC_RELAXED);
}

/* ---- textures: inline PTX, the forms nvcc emits for tex2D / tex1Dfetch on texture objects ---- */

extern "C" __global__ void tex2d(float *out, unsigned long long tex, float x0, float y0, float dx, float dy,
				 int w, int h)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= w * h)
		return;
	float x = x0 + (i % w) * dx, y = y0 + (i / w) * dy, r, g, b, a;
	asm volatile("tex.2d.v4.f32.f32 {%0, %1, %2, %3}, [%4, {%5, %6}];"
		     : "=f"(r), "=f"(g), "=f"(b), "=f"(a) : "l"(tex), "f"(x), "f"(y));
	out[i * 4 + 0] = r;
	out[i * 4 + 1] = g;
	out[i * 4 + 2] = b;
	out[i * 4 + 3] = a;
}

extern "C" __global__ void tex1dfetch(float *out, unsigned long long tex, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n)
		return;
	float r, g, b, a;
	asm volatile("tex.1d.v4.f32.s32 {%0, %1, %2, %3}, [%4, {%5}];"
		     : "=f"(r), "=f"(g), "=f"(b), "=f"(a) : "l"(tex), "r"(i));
	out[i] = r + g * 1000.0f;
}
