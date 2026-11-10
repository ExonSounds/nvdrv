// SPDX-License-Identifier: MIT
/* Offline check of nvcuda's compiler: nakc file.spv entry...  (prints NAK's result per kernel) */
#include <stdio.h>
#include <stdlib.h>

#include "../src/compile.h"

int main(int argc, char **argv)
{
	FILE *f;
	long n;
	uint32_t *w;
	int i, fails = 0;

	if (argc < 3) {
		fprintf(stderr, "usage: %s file.spv entry...\n", argv[0]);
		return 2;
	}
	f = fopen(argv[1], "rb");
	if (!f || fseek(f, 0, SEEK_END) || (n = ftell(f)) <= 0)
		return 1;
	rewind(f);
	w = malloc(n);
	if (fread(w, 1, n, f) != (size_t)n)
		return 1;
	fclose(f);
	if (nvc_compiler_init(6, 19))
		return 1;
	for (i = 2; i < argc; i++) {
		char err[256] = "";
		struct nvc_kernel_bin *kb = nvc_compile_spirv(w, n / 4, argv[i], err, sizeof(err));

		if (!kb) {
			printf("%s: FAIL %s\n", argv[i], err);
			fails++;
			continue;
		}
		printf("%s: %u bytes of code, %u GPRs, %u B local/thread, %u B shared, %u params (%u bytes):", argv[i],
		       kb->code_size, nvc_kernel_gprs(kb), nvc_kernel_slm(kb), kb->shared_static, kb->nparams,
		       kb->param_bytes);
		for (unsigned int p = 0; p < kb->nparams; p++)
			printf(" %u@%u", kb->params[p].size, kb->params[p].offset);
		printf("\n");
		nvc_kernel_bin_free(kb);
	}
	return fails != 0;
}
