// SPDX-License-Identifier: MIT
/* Cycles-style stream compaction through nvcuda: the set of selected ids must match the host. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cuda.h"

static int cmpu(const void *a, const void *b)
{
	unsigned int x = *(const unsigned int *)a, y = *(const unsigned int *)b;

	return x < y ? -1 : x > y;
}

int main(int argc, char **argv)
{
	const char *ptx = argc > 1 ? argv[1] : "build/stress.ptx";
	const unsigned int N = 100003, salt = 0x9e37;
	unsigned int *ref = malloc(N * 4), *got = malloc(N * 4), nref = 0, ngot = 0, i, bad = 0;
	CUdevice dev;
	CUcontext ctx;
	CUmodule mod;
	CUfunction f;
	CUdeviceptr didx, dcount;
	CUresult r;

	for (i = 0; i < N; i++)
		if ((((i * 2654435761u) ^ salt) >> 7) % 3 != 0)
			ref[nref++] = i;
	if ((r = cuInit(0)) || (r = cuDeviceGet(&dev, 0)) || (r = cuCtxCreate(&ctx, 0, dev)) ||
	    (r = cuModuleLoad(&mod, ptx)) || (r = cuModuleGetFunction(&f, mod, "compaction")) ||
	    (r = cuMemAlloc(&didx, N * 4)) || (r = cuMemAlloc(&dcount, 4)) || (r = cuMemsetD32(dcount, 0, 1)) ||
	    (r = cuMemsetD32(didx, 0xffffffff, N))) {
		printf("setup failed: %d\n", r);
		return 1;
	}
	{
		void *args[] = { &didx, &dcount, (void *)&N, (void *)&salt };

		if ((r = cuLaunchKernel(f, (N + 511) / 512, 1, 1, 512, 1, 1, 0, NULL, args, NULL)) ||
		    (r = cuMemcpyDtoH(&ngot, dcount, 4)) || (r = cuMemcpyDtoH(got, didx, N * 4))) {
			printf("launch failed: %d\n", r);
			return 1;
		}
	}
	qsort(got, ngot < N ? ngot : N, 4, cmpu);
	for (i = 0; i < nref && i < ngot; i++)
		bad += got[i] != ref[i];
	if (ngot != nref || bad) {
		printf("FAIL: count %u (want %u), %u ids differ\n", ngot, nref, bad);
		return 1;
	}
	printf("ok: compaction of %u ids matches the host (OUTLINE=%s REGMEM=%s)\n", nref,
	       getenv("NVCUDA_OUTLINE") ?: "auto", getenv("NVCUDA_REGMEM") ?: "auto");
	return 0;
}
