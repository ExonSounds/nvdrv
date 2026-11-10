// SPDX-License-Identifier: MIT
/*
 * Kernel compiler: SPIR-V (OpenCL "Kernel" modules, e.g. from clang's CUDA -> spirv64 path)
 * -> NIR (Mesa's spirv_to_nir) -> NAK (Mesa's NVIDIA compiler) -> sm_86 machine code.
 * Mesa is MIT; nvcuda links Mesa's static libraries (cuda/mesa/build-nak.sh).
 *
 * Lowering follows Mesa's OpenCL users (rusticl, panfrost and kosmickrisp's CLC):
 *  - every pointer is 62-bit generic: CUDA code is generic by default and clang's SPIR-V output
 *    reaches kernel arguments through integer casts, so the address space is only known at run
 *    time. NAK sees load_global / load_shared / load_scratch behind the mode checks.
 *  - kernel arguments (uniform variables) become load_kernel_input, then reads of nvcuda's
 *    constant buffer 0 (ldc_nv):
 *      0x00 block size (x, y, z)   0x0c grid size (x, y, z)   0x20.. the kernel's parameters
 *    in natural C layout, as cuLaunchKernel packs them.
 *  - workgroup size and count are read from the same buffer (the block size is per launch).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nir.h"
#include "util/sparse_bitset.h"
#include "nir_builder.h"
#include "compiler/glsl_types.h"
#include "spirv/nir_spirv.h"
#include "spirv/spirv_info.h"
#include "nak.h"
#include "nv_device_info.h"

#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include "compile.h"
#include "ptx.h"

/* ---- on-disk cache of compiled kernels: $XDG_CACHE_HOME/nvcuda (NVCUDA_CACHE=0 disables) ---- */

#define CACHE_MAGIC	"NVCUDAK2"
#define CACHE_VERSION	"nvcuda-2026-11-10c"	/* bump when the compiler or the ABI changes */

struct cache_hdr {
	char magic[8];
	uint32_t info_size, code_size, shared_static, shared_hidden, max_threads, nparams, param_bytes, pad;
};

static uint64_t fnv(uint64_t h, const void *data, size_t len)
{
	const uint8_t *p = data;

	while (len--)
		h = (h ^ *p++) * 0x100000001b3ull;
	return h;
}

static bool cache_path(uint64_t key, char *path, size_t len, bool create)
{
	const char *e = getenv("NVCUDA_CACHE"), *x = getenv("XDG_CACHE_HOME"), *home = getenv("HOME");
	char dir[512];

	if (e && e[0] == '0')
		return false;
	if (x && x[0])
		snprintf(dir, sizeof(dir), "%s/nvcuda", x);
	else if (home && home[0])
		snprintf(dir, sizeof(dir), "%s/.cache/nvcuda", home);
	else
		return false;
	if (create && mkdir(dir, 0700) && errno != EEXIST)
		return false;
	snprintf(path, len, "%s/%016llx.bin", dir, (unsigned long long)key);
	return true;
}

static struct nvc_kernel_bin *cache_load(uint64_t key)
{
	char path[600];
	struct cache_hdr h;
	struct nvc_kernel_bin *kb = NULL;
	FILE *f;

	if (!cache_path(key, path, sizeof(path), false) || !(f = fopen(path, "rb")))
		return NULL;
	if (fread(&h, sizeof(h), 1, f) != 1 || memcmp(h.magic, CACHE_MAGIC, 8) ||
	    h.info_size != sizeof(struct nak_shader_info) || h.nparams > NVC_MAX_PARAMS)
		goto out;
	kb = calloc(1, sizeof(*kb));
	if (!kb)
		goto out;
	kb->info = malloc(h.info_size);
	kb->code = malloc(h.code_size);
	if (!kb->info || !kb->code || fread(kb->info, h.info_size, 1, f) != 1 ||
	    fread(kb->params, sizeof(kb->params[0]), h.nparams, f) != h.nparams ||
	    (h.code_size && fread(kb->code, h.code_size, 1, f) != 1)) {
		nvc_kernel_bin_free(kb);
		kb = NULL;
		goto out;
	}
	kb->code_size = h.code_size;
	kb->shared_static = h.shared_static;
	kb->shared_hidden = h.shared_hidden;
	kb->max_threads = h.max_threads;
	kb->nparams = h.nparams;
	kb->param_bytes = h.param_bytes;
out:
	fclose(f);
	return kb;
}

static void cache_store(uint64_t key, const struct nvc_kernel_bin *kb)
{
	char path[600], tmp[620];
	struct cache_hdr h = { CACHE_MAGIC, sizeof(struct nak_shader_info), kb->code_size, kb->shared_static,
			       kb->shared_hidden, kb->max_threads, kb->nparams, kb->param_bytes, 0 };
	FILE *f;

	if (!cache_path(key, path, sizeof(path), true))
		return;
	snprintf(tmp, sizeof(tmp), "%s.%d", path, getpid());
	f = fopen(tmp, "wb");
	if (!f)
		return;
	if (fwrite(&h, sizeof(h), 1, f) == 1 && fwrite(kb->info, h.info_size, 1, f) == 1 &&
	    fwrite(kb->params, sizeof(kb->params[0]), kb->nparams, f) == kb->nparams &&
	    (!kb->code_size || fwrite(kb->code, kb->code_size, 1, f) == 1) && !fclose(f)) {
		if (rename(tmp, path))
			unlink(tmp);
	} else {
		fclose(f);
		unlink(tmp);
	}
}

static struct nak_compiler *nak;
static struct nv_device_info devinfo;

static const struct spirv_capabilities spirv_caps = {
	.Addresses = true,
	.Float16 = true,
	.Float16Buffer = true,
	.Float64 = true,
	.GenericPointer = true,
	.Groups = true,
	.Int8 = true,
	.Int16 = true,
	.Int64 = true,
	.Int64Atomics = true,
	.Kernel = true,
	.Linkage = true,
};

static const struct spirv_to_nir_options spirv_options = {
	.environment = NIR_SPIRV_OPENCL,
	.capabilities = &spirv_caps,
	.shared_addr_format = nir_address_format_62bit_generic,
	.global_addr_format = nir_address_format_62bit_generic,
	.temp_addr_format = nir_address_format_62bit_generic,
	.constant_addr_format = nir_address_format_64bit_global,
};

int nvc_compiler_init(unsigned int gpcs, unsigned int tpcs)
{
	if (nak)
		return 0;
	glsl_type_singleton_init_or_ref();
	memset(&devinfo, 0, sizeof(devinfo));
	devinfo.type = NV_DEVICE_TYPE_DIS;
	devinfo.chipset = 0x174;
	devinfo.sm = 86;
	devinfo.gpc_count = gpcs;
	devinfo.tpc_count = tpcs;
	devinfo.mp_per_tpc = 2;
	devinfo.max_warps_per_mp = 48;
	devinfo.max_blocks_per_mp = 16;
	devinfo.cls_copy = 0xc7b5;
	devinfo.cls_compute = 0xc7c0;
	devinfo.cls_eng3d = 0xc797;
	devinfo.cls_gpfifo = 0xc56f;
	/* compute capability 8.6: 0, 8, 16, 32, 64 or 100 KiB of shared memory per SM */
	static const uint16_t smem[] = { 0, 8, 16, 32, 64, 100 };
	memcpy(devinfo.sm_smem_sizes_kB, smem, sizeof(smem));
	devinfo.sm_smem_size_count = 6;
	devinfo.max_smem_per_wg_kB = 100;
	nak = nak_compiler_create(&devinfo);
	return nak ? 0 : -1;
}

const struct nv_device_info *nvc_compiler_devinfo(void)
{
	return &devinfo;
}

static void optimize(nir_shader *nir)
{
	bool progress;

	do {
		progress = false;
		NIR_PASS(progress, nir, nir_split_var_copies);
		NIR_PASS(progress, nir, nir_lower_var_copies);
		NIR_PASS(progress, nir, nir_lower_vars_to_ssa);
		NIR_PASS(progress, nir, nir_opt_copy_prop);
		NIR_PASS(progress, nir, nir_opt_copy_prop_vars);
		NIR_PASS(progress, nir, nir_opt_dead_write_vars);
		NIR_PASS(progress, nir, nir_opt_deref);
		NIR_PASS(progress, nir, nir_opt_remove_phis);
		NIR_PASS(progress, nir, nir_opt_dce);
		NIR_PASS(progress, nir, nir_opt_dead_cf);
		NIR_PASS(progress, nir, nir_opt_cse);
		NIR_PASS(progress, nir, nir_opt_constant_folding);
		NIR_PASS(progress, nir, nir_opt_algebraic);
		NIR_PASS(progress, nir, nir_opt_undef);
		NIR_PASS(progress, nir, nir_opt_if, nir_opt_if_optimize_phi_true_false);
		{
			/* small branches become selects: fewer blocks (NAK's memory grows with blocks x values) */
			nir_opt_peephole_select_options sel = { .limit = 16, .indirect_load_ok = true,
								.expensive_alu_ok = true };

			NIR_PASS(progress, nir, nir_opt_peephole_select, &sel);
		}
	} while (progress);
}

/*
 * Atomics through generic pointers: NIR can only lower them for global and shared memory (an
 * atomic on a thread's private memory has no meaning), so narrow such derefs with a cast.
 */
static bool narrow_generic_atomics(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
	const nir_variable_mode priv = nir_var_function_temp | nir_var_shader_temp;
	nir_deref_instr *deref, *cast;

	if (intr->intrinsic != nir_intrinsic_deref_atomic && intr->intrinsic != nir_intrinsic_deref_atomic_swap)
		return false;
	deref = nir_src_as_deref(intr->src[0]);
	if (!(deref->modes & priv) || !(deref->modes & ~priv))
		return false;
	b->cursor = nir_before_instr(&intr->instr);
	cast = nir_build_deref_cast(b, &deref->def, deref->modes & ~priv, deref->type, 0);
	nir_src_rewrite(&intr->src[0], &cast->def);
	return true;
}

/* kernel inputs and launch dimensions -> constant buffer 0 */
static bool lower_cbuf0(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
	unsigned int base;
	nir_def *off, *val;

	switch (intr->intrinsic) {
	case nir_intrinsic_load_kernel_input:
		base = NVC_CB_PARAMS + nir_intrinsic_base(intr);
		off = intr->src[0].ssa;
		break;
	case nir_intrinsic_load_workgroup_size:
		base = NVC_CB_BLOCK;
		off = nir_imm_int(b, 0);
		break;
	case nir_intrinsic_load_num_workgroups:
		base = NVC_CB_GRID;
		off = nir_imm_int(b, 0);
		break;
	default:
		return false;
	}
	b->cursor = nir_before_instr(&intr->instr);
	off = nir_iadd_imm(b, nir_u2u32(b, off), base);
	val = nir_ldc_nv(b, intr->def.num_components, intr->def.bit_size, nir_imm_int(b, 0), off,
			 .align_mul = intr->def.bit_size / 8, .align_offset = 0);
	nir_def_replace(&intr->def, val);
	return true;
}

static void set_err(char *err, size_t len, const char *what)
{
	if (err && len)
		snprintf(err, len, "%s", what);
}

/*
 * Shared by both front ends: an explicit-I/O kernel in NIR (stage KERNEL) -> cbuf 0 -> NAK.
 * Takes ownership of nir and kb (freed on failure).
 */
static struct nvc_kernel_bin *finish(nir_shader *nir, struct nvc_kernel_bin *kb, char *err, size_t errlen)
{
	struct nak_shader_bin *bin;

	/* from here NAK's view: a compute shader with its inputs in cbuf 0 */
	NIR_PASS(_, nir, nir_shader_intrinsics_pass, lower_cbuf0, nir_metadata_control_flow, NULL);
	nir->info.stage = MESA_SHADER_COMPUTE;
	nir->info.workgroup_size_variable = true;
	/*
	 * NAK derives the register limit from the workgroup size: 1x1x1 leaves it at 255, as CUDA does
	 * without __launch_bounds__; cuLaunchKernel checks registers x threads against the SM.
	 */
	nir->info.workgroup_size[0] = kb->max_threads ? kb->max_threads : 1;	/* .maxntid: nvcc's cap too */
	nir->info.workgroup_size[1] = nir->info.workgroup_size[2] = 1;
	nak_preprocess_nir(nir, nak);
	NIR_PASS(_, nir, nir_lower_int64);
	optimize(nir);
	nak_lower_nir_before_linking(nir, nak);
	nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
	kb->shared_static = nir->info.shared_size;
	if (kb->shared_hidden > kb->shared_static)
		kb->shared_hidden = kb->shared_static;
	if (getenv("NVCUDA_DUMP_NIR"))
		nir_print_shader(nir, stderr);

	bin = nak_compile_shader(nir, getenv("NVCUDA_DUMP_ASM") != NULL, nak, 0, NULL, false);
	if (!bin) {
		set_err(err, errlen, "NAK compile failed");
		free(kb);
		ralloc_free(nir);
		return NULL;
	}
	if (bin->asm_str && getenv("NVCUDA_DUMP_ASM"))
		fprintf(stderr, "%s\n", bin->asm_str);
	kb->info = malloc(sizeof(bin->info));
	if (kb->info)
		memcpy(kb->info, &bin->info, sizeof(bin->info));
	kb->code_size = bin->code_size;
	kb->code = malloc(bin->code_size);
	if (kb->code)
		memcpy(kb->code, bin->code, bin->code_size);
	nak_shader_bin_destroy(bin);
	ralloc_free(nir);
	if (!kb->code || !kb->info) {
		nvc_kernel_bin_free(kb);
		return NULL;
	}
	return kb;
}

static struct nvc_kernel_bin *compile_spirv(const uint32_t *words, size_t nwords, const char *entry, char *err,
					    size_t errlen)
{
	const nir_shader_compiler_options *nir_opts = nak_nir_options(nak);
	struct nvc_kernel_bin *kb = NULL;
	nir_shader *nir;
	unsigned int n = 0;

	nir = spirv_to_nir(words, nwords, NULL, MESA_SHADER_KERNEL, entry, &spirv_options, nir_opts);
	if (!nir) {
		set_err(err, errlen, "spirv_to_nir failed (entry point not found or unsupported SPIR-V)");
		return NULL;
	}
	nir_validate_shader(nir, "after spirv_to_nir");

	/* one function: the entry point, with everything inlined */
	NIR_PASS(_, nir, nir_lower_variable_initializers, nir_var_function_temp);
	NIR_PASS(_, nir, nir_lower_returns);
	NIR_PASS(_, nir, nir_inline_functions);
	nir_remove_non_entrypoints(nir);
	NIR_PASS(_, nir, nir_opt_deref);
	NIR_PASS(_, nir, nir_lower_variable_initializers, ~nir_var_function_temp);
	NIR_PASS(_, nir, nir_remove_dead_variables, nir_var_shader_in | nir_var_shader_out | nir_var_system_value,
		 NULL);
	optimize(nir);

	NIR_PASS(_, nir, nir_remove_dead_variables, nir_var_mem_shared | nir_var_function_temp, NULL);
	NIR_PASS(_, nir, nir_lower_system_values);
	{
		nir_lower_compute_system_values_options cs = { 0 };

		NIR_PASS(_, nir, nir_lower_compute_system_values, &cs);
	}

	/* kernel parameters: natural C layout, in argument order */
	NIR_PASS(_, nir, nir_lower_vars_to_explicit_types,
		 nir_var_uniform | nir_var_mem_shared | nir_var_function_temp | nir_var_shader_temp |
		 nir_var_mem_global | nir_var_mem_generic | nir_var_mem_constant,
		 glsl_get_cl_type_size_align);
	kb = calloc(1, sizeof(*kb));
	if (!kb)
		goto fail;
	nir_foreach_variable_with_modes(var, nir, nir_var_uniform) {
		unsigned int size, align;

		if (n == NVC_MAX_PARAMS) {
			set_err(err, errlen, "too many kernel parameters");
			goto fail;
		}
		glsl_get_cl_type_size_align(var->type, &size, &align);
		kb->params[n].offset = var->data.driver_location;
		kb->params[n].size = size;
		if (var->data.driver_location + size > kb->param_bytes)
			kb->param_bytes = var->data.driver_location + size;
		n++;
	}
	kb->nparams = n;
	if (NVC_CB_PARAMS + kb->param_bytes > NVC_CB_SIZE) {
		set_err(err, errlen, "kernel parameters larger than 4 KiB");
		goto fail;
	}

	NIR_PASS(_, nir, nir_lower_memcpy);
	NIR_PASS(_, nir, nir_shader_intrinsics_pass, narrow_generic_atomics, nir_metadata_control_flow, NULL);
	NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_uniform, nir_address_format_32bit_offset_as_64bit);
	NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_constant, nir_address_format_64bit_global);
	NIR_PASS(_, nir, nir_lower_explicit_io,
		 nir_var_mem_global | nir_var_mem_shared | nir_var_function_temp | nir_var_shader_temp |
		 nir_var_mem_generic, nir_address_format_62bit_generic);
	optimize(nir);

	return finish(nir, kb, err, errlen);
fail:
	free(kb);
	ralloc_free(nir);
	return NULL;
}

void nvc_kernel_bin_free(struct nvc_kernel_bin *kb)
{
	if (kb) {
		free(kb->code);
		free(kb->info);
		free(kb);
	}
}

/* Fill a QMD (V03_00 on GA10x) for a launch: NAK's own filler with the per-launch block size. */
void nvc_fill_qmd(const struct nvc_kernel_bin *kb, uint64_t code_va, uint64_t cb0_va, uint32_t cb0_size,
		  const uint32_t grid[3], const uint32_t block[3], uint32_t dyn_smem, void *qmd, size_t qmd_size)
{
	struct nak_shader_info info = *(const struct nak_shader_info *)kb->info;
	struct nak_qmd_info qi = { 0 };

	info.cs.local_size[0] = block[0];
	info.cs.local_size[1] = block[1];
	info.cs.local_size[2] = block[2];
	qi.addr = code_va;
	qi.smem_size = kb->shared_static + dyn_smem;
	qi.global_size[0] = grid[0];
	qi.global_size[1] = grid[1];
	qi.global_size[2] = grid[2];
	qi.num_cbufs = 1;
	qi.cbufs[0].index = 0;
	qi.cbufs[0].size = cb0_size;
	qi.cbufs[0].addr = cb0_va;
	nak_fill_qmd(&devinfo, &info, &qi, qmd, qmd_size);
}

uint32_t nvc_qmd_size(void)
{
	return nak_qmd_size_B(&devinfo);
}

uint32_t nvc_kernel_slm(const struct nvc_kernel_bin *kb)
{
	return ((const struct nak_shader_info *)kb->info)->slm_size;
}

uint32_t nvc_kernel_gprs(const struct nvc_kernel_bin *kb)
{
	return ((const struct nak_shader_info *)kb->info)->num_gprs;
}

/*
 * NAK reads two tables its Vulkan driver normally provides. nvcuda has no sample state (compute
 * only) and does not lower printf yet; printf would use the reserved 8 bytes at cbuf 0 + 0x18.
 */
const struct nak_constant_offset_info nak_const_offsets_base = {
	.printf_cb = 0,
	.printf_buffer_offset = 0x18,
};
const struct nak_constant_offset_info nak_const_offsets_turing_graphics = {
	.printf_cb = 0,
	.printf_buffer_offset = 0x18,
};

/* NVCUDA_STATS=1: instruction and block counts per stage (compile-time problems on big kernels) */
static void stats(nir_shader *nir, const char *stage)
{
	nir_function_impl *impl = nir_shader_get_entrypoint(nir);
	unsigned int blocks = 0, instrs = 0;

	if (!getenv("NVCUDA_STATS"))
		return;
	nir_foreach_block(block, impl) {
		blocks++;
		nir_foreach_instr(instr, block)
			instrs++;
	}
	fprintf(stderr, "nvcuda: %s: %s: %u blocks, %u instructions, scratch %u, shared %u\n", nir->info.name, stage,
		blocks, instrs, nir->scratch_size, nir->info.shared_size);
	if (getenv("NVCUDA_LIVESTATS") && !strcmp(stage, "optimized")) {
		unsigned int maxl = 0, nd = impl->ssa_alloc;
		uint64_t sum = 0;

		nir_metadata_require(impl, nir_metadata_live_defs);
		nir_foreach_block(block, impl) {
			unsigned int n = u_sparse_bitset_count(&block->live_in);

			sum += n;
			if (n > maxl)
				maxl = n;
		}
		fprintf(stderr, "nvcuda: %s: live defs per block: max %u, mean %llu (of %u defs)\n", nir->info.name, maxl,
			(unsigned long long)(blocks ? sum / blocks : 0), nd);
		/* which defs are live in most blocks? */
		{
			uint32_t *cnt = calloc(nd, 4);
			nir_def **defs = calloc(nd, sizeof(*defs));
			unsigned int i, k;

			nir_foreach_block(block, impl) {
				U_SPARSE_BITSET_FOREACH_SET(&block->live_in, d)
					cnt[d]++;
				nir_foreach_instr(instr, block) {
					nir_def *def = nir_instr_def(instr);

					if (def && def->index < nd)
						defs[def->index] = def;
				}
			}
			for (k = 0; k < 25; k++) {
				unsigned int best = 0;

				for (i = 1; i < nd; i++)
					if (cnt[i] > cnt[best] && !(defs[i] && nir_def_instr(defs[i])->type == nir_instr_type_load_const))
						best = i;
				if (!cnt[best])
					break;
				fprintf(stderr, "  live in %u blocks: ", cnt[best]);
				if (defs[best])
					nir_print_instr(nir_def_instr(defs[best]), stderr);
				fprintf(stderr, "\n");
				cnt[best] = 0;
			}
			/* a histogram: how many defs are live in more than N blocks */
			{
				unsigned int h[6] = { 0 };

				nir_foreach_block(block, impl) {
					U_SPARSE_BITSET_FOREACH_SET(&block->live_in, d)
						cnt[d]++;
				}
				for (i = 0; i < nd; i++) {
					if (defs[i] && nir_def_instr(defs[i])->type == nir_instr_type_load_const)
						continue;
					if (cnt[i] > 10000) h[0]++;
					else if (cnt[i] > 1000) h[1]++;
					else if (cnt[i] > 100) h[2]++;
					else if (cnt[i] > 10) h[3]++;
					else if (cnt[i] > 0) h[4]++;
				}
				fprintf(stderr, "  non-constant defs live in >10000 blocks: %u, >1000: %u, >100: %u, >10: %u, >0: %u\n", h[0], h[1],
					h[2], h[3], h[4]);
			}
			free(cnt);
			free(defs);
		}
	}
}

/* PTX text -> NIR (ptx.c) -> the same tail as SPIR-V */
static struct nvc_kernel_bin *compile_ptx_once(const char *ptx, const char *entry,
					       const struct nvc_ptx_opts *po, char *err, size_t errlen)
{
	struct nvc_kernel_bin *kb = calloc(1, sizeof(*kb));
	nir_shader *nir;

	if (!kb)
		return NULL;
	nir = nvc_ptx_build(ptx, entry, nak_nir_options(nak), kb, po, err, errlen);
	if (!nir) {
		free(kb);
		return NULL;
	}
	if (NVC_CB_PARAMS + kb->param_bytes > NVC_CB_SIZE) {
		set_err(err, errlen, "kernel parameters larger than 4 KiB");
		free(kb);
		ralloc_free(nir);
		return NULL;
	}
	kb->shared_hidden = NVC_DUMMY_BYTES;
	nir_validate_shader(nir, "after the PTX front end");
	stats(nir, "front end");
	if (getenv("NVCUDA_FRONTEND_ONLY")) {
		set_err(err, errlen, "stopped after the front end (NVCUDA_FRONTEND_ONLY)");
		free(kb);
		ralloc_free(nir);
		return NULL;
	}
	if (!getenv("NVCUDA_NO_STRUCTURIZE"))	/* experiment: NAK's from_nir takes goto/goto_if itself */
		NIR_PASS(_, nir, nir_lower_goto_ifs);		/* PTX's branches -> structured control flow */
	stats(nir, "structured");
	if (getenv("NVCUDA_STATS")) {	/* any goto left means a block the structurizer did not reach */
		unsigned int left = 0, nulls = 0;

		nir_foreach_block(block, nir_shader_get_entrypoint(nir)) {
			nir_instr *last = nir_block_last_instr(block);

			if (last && last->type == nir_instr_type_jump) {
				nir_jump_instr *j = nir_instr_as_jump(last);

				if (j->type == nir_jump_goto || j->type == nir_jump_goto_if) {
					left++;
					nulls += !j->target;
				}
			}
		}
		if (left)
			fprintf(stderr, "nvcuda: %s: %u gotos left after structurizing (%u without target)\n", nir->info.name,
				left, nulls);
	}
	NIR_PASS(_, nir, nir_opt_dead_cf);
	NIR_PASS(_, nir, nir_lower_vars_to_ssa);	/* PTX registers -> SSA */
	stats(nir, "ssa");
	NIR_PASS(_, nir, nir_lower_system_values);
	{
		nir_lower_compute_system_values_options cs = { 0 };

		NIR_PASS(_, nir, nir_lower_compute_system_values, &cs);
	}
	optimize(nir);
	stats(nir, "optimized");
	if (getenv("NVCUDA_NO_NAK")) {
		set_err(err, errlen, "stopped before NAK (NVCUDA_NO_NAK)");
		free(kb);
		ralloc_free(nir);
		return NULL;
	}
	return finish(nir, kb, err, errlen);
}

/*
 * Compile in a forked child and read the result back through a pipe. Mesa's structurizer
 * crashes on some shapes of huge kernels (nondeterministically: its pointer sets follow ASLR),
 * and a JIT inside an application must not take the process down. NVCUDA_NO_FORK=1 compiles
 * in-process (debugging).
 */
static bool write_all(int fd, const void *p, size_t n)
{
	while (n) {
		ssize_t w = write(fd, p, n);

		if (w <= 0)
			return false;
		p = (const char *)p + w;
		n -= w;
	}
	return true;
}

static bool read_all(int fd, void *p, size_t n)
{
	while (n) {
		ssize_t r = read(fd, p, n);

		if (r <= 0)
			return false;
		p = (char *)p + r;
		n -= r;
	}
	return true;
}

static struct nvc_kernel_bin *compile_ptx_forked(const char *ptx, const char *entry,
						 const struct nvc_ptx_opts *po, char *err, size_t errlen)
{
	struct nvc_kernel_bin *kb = NULL;
	struct cache_hdr h;
	int fds[2], status;
	pid_t pid;

	if (getenv("NVCUDA_NO_FORK"))
		return compile_ptx_once(ptx, entry, po, err, errlen);
	if (pipe(fds))
		return compile_ptx_once(ptx, entry, po, err, errlen);
	pid = fork();
	if (pid < 0) {
		close(fds[0]);
		close(fds[1]);
		return compile_ptx_once(ptx, entry, po, err, errlen);
	}
	if (pid == 0) {	/* child: compile, write header + info + params + code (+ the error text on failure) */
		char cerr[256] = "";
		struct nvc_kernel_bin *ckb;

		close(fds[0]);
		signal(SIGPIPE, SIG_IGN);
		ckb = compile_ptx_once(ptx, entry, po, cerr, sizeof(cerr));
		if (!ckb) {
			memset(&h, 0, sizeof(h));
			memcpy(h.magic, "NVCFAIL!", 8);
			write_all(fds[1], &h, sizeof(h));
			write_all(fds[1], cerr, sizeof(cerr));
			_exit(1);
		}
		h = (struct cache_hdr){ CACHE_MAGIC, sizeof(struct nak_shader_info), ckb->code_size, ckb->shared_static,
					ckb->shared_hidden, ckb->max_threads, ckb->nparams, ckb->param_bytes,
					ckb->used_regmem };
		write_all(fds[1], &h, sizeof(h));
		write_all(fds[1], ckb->info, h.info_size);
		write_all(fds[1], ckb->params, sizeof(ckb->params[0]) * ckb->nparams);
		write_all(fds[1], ckb->code, ckb->code_size);
		_exit(0);
	}
	close(fds[1]);
	if (!read_all(fds[0], &h, sizeof(h))) {
		set_err(err, errlen, "the compiler crashed");
	} else if (!memcmp(h.magic, "NVCFAIL!", 8)) {
		char cerr[256];

		if (read_all(fds[0], cerr, sizeof(cerr)))
			set_err(err, errlen, cerr);
	} else if (!memcmp(h.magic, CACHE_MAGIC, 8) && h.info_size == sizeof(struct nak_shader_info) &&
		   h.nparams <= NVC_MAX_PARAMS) {
		kb = calloc(1, sizeof(*kb));
		if (kb) {
			kb->info = malloc(h.info_size);
			kb->code = malloc(h.code_size ? h.code_size : 1);
			if (!kb->info || !kb->code || !read_all(fds[0], kb->info, h.info_size) ||
			    !read_all(fds[0], kb->params, sizeof(kb->params[0]) * h.nparams) ||
			    (h.code_size && !read_all(fds[0], kb->code, h.code_size))) {
				nvc_kernel_bin_free(kb);
				kb = NULL;
				set_err(err, errlen, "the compiler crashed while writing its result");
			} else {
				kb->code_size = h.code_size;
				kb->shared_static = h.shared_static;
				kb->shared_hidden = h.shared_hidden;
				kb->max_threads = h.max_threads;
				kb->nparams = h.nparams;
				kb->param_bytes = h.param_bytes;
				kb->used_regmem = h.pad;
			}
		}
	} else {
		set_err(err, errlen, "garbled compiler result");
	}
	close(fds[0]);
	while (waitpid(pid, &status, 0) < 0 && errno == EINTR)
		;
	return kb;
}

/*
 * NAK does not survive every shape the structurizer makes of a huge kernel. Try the default
 * build first; if it fails, registers in scratch (a different shape), then no outlining.
 */
static struct nvc_kernel_bin *compile_ptx(const char *ptx, const char *entry, char *err, size_t errlen)
{
	static const struct nvc_ptx_opts tries[] = { { -1, -1 }, { -1, 1 }, { 0, -1 }, { 0, 1 } };
	struct nvc_kernel_bin *kb = NULL;
	unsigned int i;

	for (i = 0; i < sizeof(tries) / sizeof(tries[0]) && !kb; i++) {
		if (i && getenv("NVCUDA_DEBUG"))
			fprintf(stderr, "nvcuda: %s: retrying (outline %d, regmem %d): %s\n", entry, tries[i].outline,
				tries[i].regmem, err);
		if (i && strstr(err, "ptx line"))
			break;	/* the PTX itself is the problem */
		kb = compile_ptx_forked(ptx, entry, &tries[i], err, errlen);
		if (!kb && strstr(err, "crashed"))	/* the crash follows the address layout: once more as is */
			kb = compile_ptx_forked(ptx, entry, &tries[i], err, errlen);
	}
	return kb;
}

struct nvc_kernel_bin *nvc_compile_spirv(const uint32_t *words, size_t nwords, const char *entry, char *err,
					 size_t errlen)
{
	uint64_t key = 0xcbf29ce484222325ull;
	struct nvc_kernel_bin *kb;

	key = fnv(key, CACHE_VERSION, strlen(CACHE_VERSION));
	key = fnv(key, "spirv", 5);
	key = fnv(key, entry, strlen(entry) + 1);
	key = fnv(key, words, nwords * 4);
	kb = cache_load(key);
	if (kb)
		return kb;
	kb = compile_spirv(words, nwords, entry, err, errlen);
	if (kb)
		cache_store(key, kb);
	return kb;
}

struct nvc_kernel_bin *nvc_compile_ptx(const char *ptx, const char *entry, char *err, size_t errlen)
{
	uint64_t key = 0xcbf29ce484222325ull;
	struct nvc_kernel_bin *kb;

	/* the globals address is an immediate in the code; the mode overrides change the code */
	key = fnv(key, CACHE_VERSION, strlen(CACHE_VERSION));
	key = fnv(key, "ptx", 3);
	{
		const char *o = getenv("NVCUDA_OUTLINE"), *m = getenv("NVCUDA_REGMEM");

		key = fnv(key, o ? o : "-", o ? strlen(o) : 1);
		key = fnv(key, m ? m : "-", m ? strlen(m) : 1);
	}
	key = fnv(key, entry, strlen(entry) + 1);
	key = fnv(key, ptx, strlen(ptx));
	kb = cache_load(key);
	if (kb)
		return kb;
	kb = compile_ptx(ptx, entry, err, errlen);
	if (kb)
		cache_store(key, kb);
	return kb;
}
