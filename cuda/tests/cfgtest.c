// SPDX-License-Identifier: MIT
/*
 * Structurizer probe: builds an unstructured NIR shader from a CFG description ("cfg N: goto M",
 * "cfg N: goto_if A B", "cfg end E", as NVCUDA_DUMP_CFG=1 prints it) and runs nir_lower_goto_ifs.
 * Built against an asserts-on Mesa (make cfgtest MESA_MK=mesa/mesa-debug.mk) it aborts on the shapes
 * the structurizer cannot do; tests/reduce_cfg.py shrinks a failing CFG to a minimal one.
 * Exit 0 = structurized, 3 = a return without the exit marker (an unroutable branch).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nir.h"
#include "nir_builder.h"

int main(int argc, char **argv)
{
	static const nir_shader_compiler_options opts = { 0 };
	FILE *f = fopen(argv[1], "r");
	char line[256];
	int n = 0, end = -1, i;
	int (*succ)[2] = calloc(100000, sizeof(*succ));
	nir_block **blocks;

	for (i = 0; i < 100000; i++) succ[i][0] = succ[i][1] = -1;
	while (fgets(line, sizeof(line), f)) {
		int a, b, c;
		if (sscanf(line, "cfg %d: goto_if %d %d", &a, &b, &c) == 3) { succ[a][0] = b; succ[a][1] = c; if (a >= n) n = a + 1; }
		else if (sscanf(line, "cfg %d: goto %d", &a, &b) == 2) { succ[a][0] = b; if (a >= n) n = a + 1; }
		else if (sscanf(line, "cfg end %d", &a) == 1) end = a;
	}
	glsl_type_singleton_init_or_ref();
	nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_KERNEL, &opts, "cfg");
	nir_function_impl *impl = b.impl;
	impl->structured = false;
	blocks = calloc(n + 1, sizeof(*blocks));
	blocks[0] = nir_start_block(impl);
	for (i = 1; i < n; i++) {
		blocks[i] = nir_block_create(impl);
		exec_list_push_tail(&impl->body, &blocks[i]->cf_node.node);
		blocks[i]->cf_node.parent = &impl->cf_node;
	}
	blocks[end < 0 ? n : end] = impl->end_block;
	for (i = 0; i < n; i++) {
		b.cursor = nir_after_block(blocks[i]);
		if (succ[i][0] < 0) continue;
		nir_def *v = nir_imm_int(&b, i);	/* an instruction per block, so the block is not empty */
		nir_store_global(&b, v, nir_imm_int64(&b, 0x1000 + 4 * i), .write_mask = 1, .align_mul = 4);
		if (succ[i][1] >= 0) {
			nir_def *cond = nir_ine_imm(&b, nir_load_local_invocation_index(&b), i);
			nir_goto_if(&b, blocks[succ[i][0]], cond, blocks[succ[i][1]]);
		} else {
			if (blocks[succ[i][0]] == impl->end_block)
				nir_nop(&b);	/* the exit marker */
			nir_goto(&b, blocks[succ[i][0]]);
		}
	}
	nir_sort_unstructured_blocks(impl);
	nir_validate_shader(b.shader, "cfgtest input");
	nir_lower_goto_ifs(b.shader);
	nir_validate_shader(b.shader, "structured");
	/* every return must follow the marker */
	nir_foreach_block(block, impl) {
		nir_foreach_instr(instr, block) {
			if (instr->type == nir_instr_type_jump && nir_instr_as_jump(instr)->type == nir_jump_return) {
				nir_instr *prev = nir_instr_prev(instr);
				if (!prev || prev->type != nir_instr_type_intrinsic ||
				    nir_instr_as_intrinsic(prev)->intrinsic != nir_intrinsic_nop) {
					printf("unroutable branch (return without marker)\n");
					return 3;
				}
			}
		}
	}
	printf("ok\n");
	return 0;
}
