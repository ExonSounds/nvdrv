// SPDX-License-Identifier: MIT
/* Runs tests/stress.cu through nvcuda in the given mode and compares with the host computation. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cuda.h"
#include "stress_kernel.h"

int main(int argc, char **argv)
{
	const char *ptx = argc > 1 ? argv[1] : "build/stress.ptx";
	const unsigned int N = 1u << 16;
	unsigned int *ref = malloc(N * 16), *got = malloc(N * 16), i, bad = 0, firstbad = ~0u;
	CUdevice dev;
	CUcontext ctx;
	CUmodule mod;
	CUfunction f;
	CUdeviceptr d;
	CUresult r;

	for (i = 0; i < N; i++)
		stress_path(i, ref + i * 4);
	if ((r = cuInit(0)) || (r = cuDeviceGet(&dev, 0)) || (r = cuCtxCreate(&ctx, 0, dev)) ||
	    (r = cuModuleLoad(&mod, ptx)) || (r = cuModuleGetFunction(&f, mod, "stress")) || (r = cuMemAlloc(&d, N * 16))) {
		printf("setup failed: %d\n", r);
		return 1;
	}
	{
		void *args[] = { &d, (void *)&N };

		if ((r = cuLaunchKernel(f, N / 128, 1, 1, 128, 1, 1, 0, NULL, args, NULL)) || (r = cuMemcpyDtoH(got, d, N * 16))) {
			printf("launch failed: %d\n", r);
			return 1;
		}
	}
	for (i = 0; i < N * 4; i++)
		if (got[i] != ref[i]) {
			if (firstbad == ~0u)
				firstbad = i;
			bad++;
		}
	if (bad) {
		i = firstbad / 4;
		printf("FAIL: %u of %u values differ; thread %u: got %u %u %u %u, want %u %u %u %u\n", bad, N * 4, i,
		       got[i * 4], got[i * 4 + 1], got[i * 4 + 2], got[i * 4 + 3], ref[i * 4], ref[i * 4 + 1],
		       ref[i * 4 + 2], ref[i * 4 + 3]);
		return 1;
	}
	printf("ok: %u threads match the host (OUTLINE=%s REGMEM=%s)\n", N, getenv("NVCUDA_OUTLINE") ?: "auto",
	       getenv("NVCUDA_REGMEM") ?: "auto");
	return 0;
}

/* second test: compaction (built as a separate main through STRESS_COMPACTION) */
