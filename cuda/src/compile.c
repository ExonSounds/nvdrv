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
#include "nir_builder.h"
#include "compiler/glsl_types.h"
#include "spirv/nir_spirv.h"
#include "spirv/spirv_info.h"
#include "nak.h"
#include "nv_device_info.h"

#include "compile.h"

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

struct nvc_kernel_bin *nvc_compile_spirv(const uint32_t *words, size_t nwords, const char *entry, char *err,
					 size_t errlen)
{
	const nir_shader_compiler_options *nir_opts = nak_nir_options(nak);
	struct nvc_kernel_bin *kb = NULL;
	struct nak_shader_bin *bin;
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

	/* from here NAK's view: a compute shader with its inputs in cbuf 0 */
	NIR_PASS(_, nir, nir_shader_intrinsics_pass, lower_cbuf0, nir_metadata_control_flow, NULL);
	nir->info.stage = MESA_SHADER_COMPUTE;
	nir->info.workgroup_size_variable = true;
	/*
	 * NAK derives the register limit from the workgroup size: 1x1x1 leaves it at 255, as CUDA does
	 * without __launch_bounds__; cuLaunchKernel checks registers x threads against the SM.
	 */
	nir->info.workgroup_size[0] = nir->info.workgroup_size[1] = nir->info.workgroup_size[2] = 1;
	nak_preprocess_nir(nir, nak);
	NIR_PASS(_, nir, nir_lower_int64);
	optimize(nir);
	nak_lower_nir_before_linking(nir, nak);
	nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
	kb->shared_static = nir->info.shared_size;
	if (getenv("NVCUDA_DUMP_NIR"))
		nir_print_shader(nir, stderr);

	bin = nak_compile_shader(nir, getenv("NVCUDA_DUMP_ASM") != NULL, nak, 0, NULL, false);
	if (!bin) {
		set_err(err, errlen, "NAK compile failed");
		goto fail;
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
