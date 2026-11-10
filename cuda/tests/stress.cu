// SPDX-License-Identifier: MIT
#include "stress_kernel.h"

extern "C" __global__ void stress(unsigned int *out, unsigned int n)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < n)
		stress_path(tid, out + tid * 4);
}

/*
 * Cycles-style stream compaction (gpu_parallel_active_index_array): every thread with a true
 * predicate gets a unique slot from a warp ballot + popcount, a per-warp shared prefix sum and
 * one block-level atomic. out_idx collects the thread ids, out_count the total.
 */
extern "C" __global__ void compaction(unsigned int *out_idx, unsigned int *out_count, unsigned int n, unsigned int salt)
{
	__shared__ unsigned int warp_offset[16];
	__shared__ unsigned int block_base;
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
	unsigned int active = tid < n && (((tid * 2654435761u) ^ salt) >> 7) % 3 != 0;
	unsigned int ballot = __nvvm_vote_ballot_sync(0xffffffffu, active);
	unsigned int lanemask_lt = (1u << lane) - 1u;

	if (lane == 0)
		warp_offset[warp] = __builtin_popcount(ballot);
	__syncthreads();
	if (threadIdx.x == 0) {
		unsigned int sum = 0, w;

		for (w = 0; w < blockDim.x / 32; w++) {
			unsigned int c = warp_offset[w];

			warp_offset[w] = sum;
			sum += c;
		}
		block_base = __atomic_fetch_add(out_count, sum, __ATOMIC_RELAXED);
	}
	__syncthreads();
	if (active)
		out_idx[block_base + warp_offset[warp] + __builtin_popcount(ballot & lanemask_lt)] = tid;
}
