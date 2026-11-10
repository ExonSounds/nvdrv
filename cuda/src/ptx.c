// SPDX-License-Identifier: MIT
/*
 * PTX front end: PTX assembly text (NVIDIA's public "Parallel Thread Execution ISA" specification)
 * -> NIR, for nvcuda's compiler (compile.c), which takes it from there to NAK.
 *
 * One .entry is built per call. The NIR is produced already in "explicit I/O" form:
 *  - every PTX register is a function_temp variable (nir_lower_vars_to_ssa makes SSA of it);
 *  - PTX's arbitrary branches become an unstructured CFG (nir_goto / nir_goto_if, structured
 *    later by nir_lower_goto_ifs, as spirv_to_nir does for OpenCL kernels);
 *  - ld/st/atom name their state space in PTX, so they map straight onto load_global,
 *    load_shared, load_scratch and load_kernel_input. Generic addresses (no state space) use
 *    NVIDIA's model: shared and local memory appear in windows of the generic address space
 *    (nvcuda programs them at 0xfe000000 and 0xff000000); cvta converts, and a generic access
 *    checks the window at run time;
 *  - predicated instructions get a block of their own;
 *  - device functions (.func) are inlined at each call: the callee's registers, labels and
 *    variables live in a frame of their own, its .param arguments are the caller's .param
 *    variables, and .param/.local variables are scratch memory allocated like a stack (freed
 *    at the end of the PTX scope that declared them, and when a callee returns);
 *  - module-scope .global/.const variables live in one buffer per module (nvc_ptx_globals lays
 *    it out and fills the initializers); ld.const/ld.global of them are global loads from it.
 * Kernel parameters are laid out in declaration order with their alignment, as cuLaunchKernel
 * packs them (load_kernel_input offsets).
 *
 * Unsupported constructs fail with the PTX line; nothing is guessed silently.
 */
#include <ctype.h>
#include <math.h>
#include <setjmp.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nir.h"
#include "nir_builder.h"
#include "compiler/glsl_types.h"
#include "util/hash_table.h"
#include "util/ralloc.h"

#include "compile.h"
#include "ptx.h"

#define SHARED_WINDOW	0xfe000000ull
#define LOCAL_WINDOW	0xff000000ull
#define WINDOW_SIZE	0x01000000ull

/* ---- lexer ---- */

enum tok { T_EOF, T_ID, T_INT, T_FLT, T_HEXF, T_STR, T_P };

struct lex {
	const char *src, *p;
	int line;
	enum tok t;
	char s[256];		/* identifier / directive / string text */
	uint64_t u;		/* integer value, or hex-float bits */
	double d;		/* decimal float value */
	int hexf_bits;		/* 32 (0f) or 64 (0d) */
	int pc;			/* punctuation char */
};

#define MAX_DEPTH	32	/* nested inlined calls */

/* one function being emitted: the entry, or an inlined callee */
struct frame {
	struct hash_table *regs, *labels, *syms;
	nir_block *ret;		/* where ret goes (the impl's end block for the entry) */
};

/* a .func of the module */
struct pfunc {
	const char *hdr;	/* just after ".func" */
	int line;
	bool has_body;
};

struct ctx {
	struct lex lx;
	jmp_buf fail;
	char *err;
	size_t errlen;
	void *mem;
	nir_shader *nir;
	nir_function_impl *impl;
	nir_builder b;
	struct hash_table *modsyms;	/* module-scope variables */
	struct hash_table *funcs;	/* name -> struct pfunc */
	struct frame frames[MAX_DEPTH + 1];
	int depth;
	struct nvc_kernel_bin *kb;
	bool terminated;	/* current block ended with a jump */
	uint32_t shared_size;
	uint32_t local_top, local_max;	/* scratch: stack top and high-water mark */
	bool dyn_shared;
	/* module globals (.global/.const): layout and initializers */
	struct nvc_global *globals;
	uint32_t nglobals, maxglobals, globals_size;
	uint8_t *globals_image;
	uint64_t globals_va;
};

static void inline_call(struct ctx *c, char rets[][256], int nrets, const char *fname, char args[][256], int nargs);
static void parse_params(struct ctx *c);

static void __attribute__((noreturn, format(printf, 2, 3))) fail(struct ctx *c, const char *fmt, ...)
{
	va_list ap;
	int n = snprintf(c->err, c->errlen, "ptx line %d: ", c->lx.line);

	va_start(ap, fmt);
	if (n >= 0 && (size_t)n < c->errlen)
		vsnprintf(c->err + n, c->errlen - n, fmt, ap);
	va_end(ap);
	longjmp(c->fail, 1);
}

static bool idchar(int ch)
{
	return isalnum(ch) || ch == '_' || ch == '$' || ch == '%' || ch == '.';
}

static void next(struct ctx *c)
{
	struct lex *l = &c->lx;
	const char *p = l->p;
	size_t n = 0;

	for (;;) {
		while (*p && isspace((unsigned char)*p)) {
			if (*p == '\n')
				l->line++;
			p++;
		}
		if (p[0] == '/' && p[1] == '/') {
			while (*p && *p != '\n')
				p++;
			continue;
		}
		if (p[0] == '/' && p[1] == '*') {
			for (p += 2; *p && !(p[0] == '*' && p[1] == '/'); p++)
				if (*p == '\n')
					l->line++;
			if (*p)
				p += 2;
			continue;
		}
		break;
	}
	if (!*p) {
		l->t = T_EOF;
		l->p = p;
		return;
	}
	if (isdigit((unsigned char)*p)) {
		if (p[0] == '0' && (p[1] == 'x' || p[1] == 'X')) {
			l->u = strtoull(p + 2, (char **)&p, 16);
			l->t = T_INT;
		} else if (p[0] == '0' && (p[1] == 'f' || p[1] == 'F' || p[1] == 'd' || p[1] == 'D') &&
			   isxdigit((unsigned char)p[2])) {
			l->hexf_bits = (p[1] == 'f' || p[1] == 'F') ? 32 : 64;
			l->u = strtoull(p + 2, (char **)&p, 16);
			l->t = T_HEXF;
		} else {
			const char *q = p;
			bool flt = false;

			while (isdigit((unsigned char)*q))
				q++;
			if (*q == '.' || *q == 'e' || *q == 'E')
				flt = true;
			if (flt) {
				l->d = strtod(p, (char **)&p);
				l->t = T_FLT;
			} else {
				l->u = strtoull(p, (char **)&p, 10);
				l->t = T_INT;
			}
		}
		if (*p == 'U')
			p++;
		l->p = p;
		return;
	}
	if (idchar((unsigned char)*p)) {
		while (*p && (idchar((unsigned char)*p) || (p[0] == ':' && p[1] == ':'))) {
			if (p[0] == ':') {
				if (n + 2 < sizeof(l->s)) {
					l->s[n++] = ':';
					l->s[n++] = ':';
				}
				p += 2;
				continue;
			}
			if (n + 1 < sizeof(l->s))
				l->s[n++] = *p;
			p++;
		}
		l->s[n] = 0;
		l->t = T_ID;
		l->p = p;
		return;
	}
	if (*p == '"') {
		for (p++; *p && *p != '"'; p++)
			if (n + 1 < sizeof(l->s))
				l->s[n++] = *p;
		l->s[n] = 0;
		if (*p)
			p++;
		l->t = T_STR;
		l->p = p;
		return;
	}
	l->t = T_P;
	l->pc = *p++;
	l->p = p;
}

static bool is_p(struct ctx *c, int ch)
{
	return c->lx.t == T_P && c->lx.pc == ch;
}

static bool accept(struct ctx *c, int ch)
{
	if (!is_p(c, ch))
		return false;
	next(c);
	return true;
}

static void expect(struct ctx *c, int ch)
{
	if (!accept(c, ch))
		fail(c, "expected '%c'", ch);
}

static bool is_id(struct ctx *c, const char *s)
{
	return c->lx.t == T_ID && !strcmp(c->lx.s, s);
}

static const char *expect_id(struct ctx *c)
{
	static __thread char buf[256];

	if (c->lx.t != T_ID)
		fail(c, "expected a name");
	snprintf(buf, sizeof(buf), "%s", c->lx.s);
	next(c);
	return buf;
}

static uint64_t expect_int(struct ctx *c)
{
	uint64_t v;

	if (c->lx.t != T_INT)
		fail(c, "expected a number");
	v = c->lx.u;
	next(c);
	return v;
}

/* skip to just after the matching '}' (cursor on '{') */
static void skip_braces(struct ctx *c)
{
	int depth = 0;

	do {
		if (c->lx.t == T_EOF)
			fail(c, "unterminated block");
		if (is_p(c, '{'))
			depth++;
		else if (is_p(c, '}'))
			depth--;
		next(c);
	} while (depth);
}

/* ---- types ---- */

enum tcls { TC_B, TC_U, TC_S, TC_F, TC_PRED };

struct ptype {
	enum tcls cls;
	unsigned bits;
};

static bool parse_type(const char *s, struct ptype *t)
{
	if (!strcmp(s, "pred")) {
		*t = (struct ptype){ TC_PRED, 1 };
		return true;
	}
	if ((s[0] == 'b' || s[0] == 'u' || s[0] == 's' || s[0] == 'f') && isdigit((unsigned char)s[1])) {
		unsigned bits = atoi(s + 1);
		char *end;

		strtoul(s + 1, &end, 10);
		if (*end || (bits != 8 && bits != 16 && bits != 32 && bits != 64))
			return false;
		if (s[0] == 'f' && bits == 8)
			return false;
		t->bits = bits;
		t->cls = s[0] == 'b' ? TC_B : s[0] == 'u' ? TC_U : s[0] == 's' ? TC_S : TC_F;
		return true;
	}
	return false;
}

static unsigned reg_bits(struct ptype t)
{
	return t.cls == TC_PRED ? 1 : t.bits;
}

/* ---- symbols: registers, labels, variables ---- */

struct preg {
	nir_variable *var;
	unsigned bits;		/* 1 for predicates */
};

enum space { SP_NONE, SP_PARAM, SP_SHARED, SP_LOCAL, SP_GLOBAL, SP_CONST };

struct psym {
	enum space space;
	uint32_t offset, size;
};

static struct frame *frame(struct ctx *c)
{
	return &c->frames[c->depth];
}

static struct hash_table *new_table(struct ctx *c)
{
	return _mesa_hash_table_create(c->mem, _mesa_hash_string, _mesa_key_string_equal);
}

static struct preg *find_reg(struct ctx *c, const char *name)
{
	struct hash_entry *e = _mesa_hash_table_search(frame(c)->regs, name);

	return e ? e->data : NULL;
}

static void declare_reg(struct ctx *c, const char *name, struct ptype t)
{
	struct preg *r;
	const struct glsl_type *gt = t.cls == TC_PRED ? glsl_bool_type() : glsl_uintN_t_type(t.bits < 16 ? 16 : t.bits);
	unsigned bits = t.cls == TC_PRED ? 1 : (t.bits < 16 ? 16 : t.bits);	/* registers are at least 16 bits */

	r = find_reg(c, name);
	if (r) {	/* nvcc redeclares temp_param_reg in every call scope */
		if (r->bits != bits)
			fail(c, "register %s redeclared with another size", name);
		return;
	}
	r = ralloc(c->mem, struct preg);
	r->bits = bits;
	r->var = nir_local_variable_create(c->impl, gt, name);
	_mesa_hash_table_insert(frame(c)->regs, ralloc_strdup(c->mem, name), r);
}

/* the frame's own variables first, then the module's */
static struct psym *find_sym(struct ctx *c, const char *name)
{
	struct hash_entry *e = _mesa_hash_table_search(frame(c)->syms, name);

	if (!e)
		e = _mesa_hash_table_search(c->modsyms, name);
	return e ? e->data : NULL;
}

static struct psym *add_sym_to(struct ctx *c, struct hash_table *t, const char *name, enum space space,
			       uint32_t offset, uint32_t size)
{
	struct psym *s = ralloc(c->mem, struct psym);

	s->space = space;
	s->offset = offset;
	s->size = size;
	_mesa_hash_table_insert(t, ralloc_strdup(c->mem, name), s);
	return s;
}

static struct psym *add_sym(struct ctx *c, const char *name, enum space space, uint32_t offset, uint32_t size)
{
	return add_sym_to(c, frame(c)->syms, name, space, offset, size);
}

/* scratch memory, allocated like a stack (parse_body frees at the end of the scope) */
static uint32_t alloc_local(struct ctx *c, uint32_t size, uint32_t align)
{
	uint32_t off = (c->local_top + align - 1) / align * align;

	c->local_top = off + size;
	if (c->local_top > c->local_max)
		c->local_max = c->local_top;
	return off;
}

static uint32_t align_up(uint32_t v, uint32_t a)
{
	return a ? (v + a - 1) / a * a : v;
}

/* ---- blocks ---- */

static nir_block *new_block(struct ctx *c)
{
	nir_block *n = nir_block_create(c->impl);

	exec_list_push_tail(&c->impl->body, &n->cf_node.node);
	n->cf_node.parent = &c->impl->cf_node;
	return n;
}

static nir_block *label_block(struct ctx *c, const char *name)
{
	struct hash_entry *e = _mesa_hash_table_search(frame(c)->labels, name);
	nir_block *blk;

	if (e)
		return e->data;
	blk = new_block(c);
	_mesa_hash_table_insert(frame(c)->labels, ralloc_strdup(c->mem, name), blk);
	return blk;
}

static void switch_to(struct ctx *c, nir_block *blk)
{
	c->b.cursor = nir_after_block(blk);
	c->terminated = false;
}

/* ---- operands ---- */

enum okind { O_REG, O_IMM, O_FIMM, O_SPECIAL, O_SYM, O_MEM, O_VEC, O_SINK };

struct opnd {
	enum okind kind;
	char name[256];		/* register, special register or symbol */
	uint64_t u;		/* integer immediate, hex-float bits, memory offset */
	int hexf_bits;		/* O_FIMM from 0f/0d: 32/64; 0 = decimal (d) */
	double d;
	bool neg;		/* !%p */
	/* O_MEM: base is a register (name) or a symbol (sym) or absolute (none), plus u */
	bool mem_reg, mem_sym;
	int nvec;
	struct opnd *vec;
};

static void parse_operand(struct ctx *c, struct opnd *o);

static void parse_imm_or_name(struct ctx *c, struct opnd *o)
{
	bool neg = accept(c, '-');

	memset(o, 0, sizeof(*o));
	if (c->lx.t == T_INT) {
		o->kind = O_IMM;
		o->u = neg ? -c->lx.u : c->lx.u;
		next(c);
	} else if (c->lx.t == T_HEXF) {
		o->kind = O_FIMM;
		o->hexf_bits = c->lx.hexf_bits;
		o->u = c->lx.u;
		if (neg)
			o->u ^= o->hexf_bits == 32 ? 0x80000000ull : 0x8000000000000000ull;
		next(c);
	} else if (c->lx.t == T_FLT) {
		o->kind = O_FIMM;
		o->d = neg ? -c->lx.d : c->lx.d;
		next(c);
	} else if (c->lx.t == T_ID) {
		snprintf(o->name, sizeof(o->name), "%s", c->lx.s);
		next(c);
		if (find_reg(c, o->name))
			o->kind = O_REG;
		else if (!strcmp(o->name, "WARP_SZ")) {	/* PTX's predefined constant */
			o->kind = O_IMM;
			o->u = 32;
		} else if (o->name[0] == '%')
			o->kind = O_SPECIAL;
		else if (!strcmp(o->name, "_"))
			o->kind = O_SINK;
		else
			o->kind = O_SYM;
		if (neg)
			fail(c, "negated name");
	} else {
		fail(c, "bad operand");
	}
}

static void parse_operand(struct ctx *c, struct opnd *o)
{
	if (accept(c, '[')) {
		struct opnd base;

		memset(o, 0, sizeof(*o));
		o->kind = O_MEM;
		parse_imm_or_name(c, &base);
		if (base.kind == O_REG) {
			o->mem_reg = true;
			snprintf(o->name, sizeof(o->name), "%s", base.name);
		} else if (base.kind == O_SYM) {
			o->mem_sym = true;
			snprintf(o->name, sizeof(o->name), "%s", base.name);
		} else if (base.kind == O_IMM) {
			o->u = base.u;
		} else {
			fail(c, "bad address");
		}
		while (is_p(c, '+') || is_p(c, '-')) {
			bool minus = is_p(c, '-');

			next(c);
			if (accept(c, '-'))	/* nvcc writes [%rd+-24] */
				minus = !minus;
			if (c->lx.t != T_INT)
				fail(c, "bad address offset");
			o->u += minus ? -c->lx.u : c->lx.u;
			next(c);
		}
		expect(c, ']');
		return;
	}
	if (accept(c, '{')) {
		struct opnd tmp[8];
		int n = 0;

		do {
			if (n == 8)
				fail(c, "vector too long");
			parse_operand(c, &tmp[n++]);
		} while (accept(c, ','));
		expect(c, '}');
		memset(o, 0, sizeof(*o));
		o->kind = O_VEC;
		o->nvec = n;
		o->vec = ralloc_array(c->mem, struct opnd, n);
		memcpy(o->vec, tmp, n * sizeof(*tmp));
		return;
	}
	if (accept(c, '!')) {
		parse_imm_or_name(c, o);
		o->neg = true;
		return;
	}
	parse_imm_or_name(c, o);
	/* symbol+offset as a value (address arithmetic on a variable) */
	if (o->kind == O_SYM && (is_p(c, '+'))) {
		next(c);
		o->u = expect_int(c);
	}
}

/* ---- reading and writing values ---- */

static nir_def *convert_int(nir_builder *b, nir_def *v, unsigned bits, bool sign)
{
	if (v->bit_size == bits)
		return v;
	if (v->bit_size == 1)
		return nir_b2iN(b, v, bits);
	if (v->bit_size > bits)
		return nir_u2uN(b, v, bits);
	return sign ? nir_i2iN(b, v, bits) : nir_u2uN(b, v, bits);
}

static nir_def *special(struct ctx *c, const char *name)
{
	nir_builder *b = &c->b;
	static const char comp[] = "xyz";
	const char *dot = strrchr(name, '.');
	int i = dot && dot[1] && strchr(comp, dot[1]) && !dot[2] ? strchr(comp, dot[1]) - comp : -1;

	if (i >= 0) {
		size_t len = dot - name;

		if (!strncmp(name, "%tid", len))
			return nir_channel(b, nir_load_local_invocation_id(b), i);
		if (!strncmp(name, "%ntid", len))
			return nir_channel(b, nir_load_workgroup_size(b), i);
		if (!strncmp(name, "%ctaid", len))
			return nir_channel(b, nir_load_workgroup_id(b), i);
		if (!strncmp(name, "%nctaid", len))
			return nir_channel(b, nir_load_num_workgroups(b), i);
	}
	if (!strcmp(name, "%laneid"))
		return nir_load_subgroup_invocation(b);
	if (!strcmp(name, "%warpid"))
		return nir_ushr_imm(b, nir_load_local_invocation_index(b), 5);
	if (!strcmp(name, "%clock"))
		return nir_unpack_64_2x32_split_x(b, nir_shader_clock(b, .memory_scope = SCOPE_SUBGROUP));
	if (!strcmp(name, "%clock64") || !strcmp(name, "%globaltimer"))
		return nir_shader_clock(b, .memory_scope = SCOPE_SUBGROUP);
	if (!strcmp(name, "%lanemask_eq"))
		return nir_ishl(b, nir_imm_int(b, 1), nir_load_subgroup_invocation(b));
	if (!strcmp(name, "%lanemask_lt"))
		return nir_iadd_imm(b, nir_ishl(b, nir_imm_int(b, 1), nir_load_subgroup_invocation(b)), -1);
	if (!strcmp(name, "%lanemask_le"))
		return nir_iadd_imm(b, nir_ishl(b, nir_imm_int(b, 2), nir_load_subgroup_invocation(b)), -1);
	if (!strcmp(name, "%lanemask_gt"))
		return nir_inot(b, nir_iadd_imm(b, nir_ishl(b, nir_imm_int(b, 2), nir_load_subgroup_invocation(b)), -1));
	if (!strcmp(name, "%lanemask_ge"))
		return nir_inot(b, nir_iadd_imm(b, nir_ishl(b, nir_imm_int(b, 1), nir_load_subgroup_invocation(b)), -1));
	fail(c, "unknown special register %s", name);
}

/* the address a variable name stands for (mov %r, sym): an offset in its own state space */
static nir_def *sym_value(struct ctx *c, const struct opnd *o, unsigned bits)
{
	struct psym *s = find_sym(c, o->name);

	if (!s)
		fail(c, "unknown name %s", o->name);
	if (s->space == SP_SHARED || s->space == SP_LOCAL || s->space == SP_PARAM)
		return nir_imm_intN_t(&c->b, s->offset + o->u, bits);
	/* .global / .const: an address in the module's globals buffer */
	return nir_imm_intN_t(&c->b, c->globals_va + s->offset + o->u, bits);
}

static nir_def *rd(struct ctx *c, const struct opnd *o, struct ptype t)
{
	nir_builder *b = &c->b;
	unsigned bits = reg_bits(t);
	nir_def *v;

	switch (o->kind) {
	case O_REG: {
		struct preg *r = find_reg(c, o->name);

		v = nir_load_var(b, r->var);
		if (t.cls == TC_PRED) {
			if (v->bit_size != 1)
				v = nir_ine_imm(b, v, 0);
			return o->neg ? nir_inot(b, v) : v;
		}
		if (v->bit_size == 1)
			v = nir_b2iN(b, v, bits);
		return convert_int(b, v, bits, t.cls == TC_S);
	}
	case O_IMM:
		if (t.cls == TC_PRED)
			return nir_imm_bool(b, o->u != 0);
		if (t.cls == TC_F)	/* an integer literal where a float is expected */
			return nir_imm_floatN_t(b, (double)(int64_t)o->u, bits);
		return nir_imm_intN_t(b, o->u, bits);
	case O_FIMM:
		if (o->hexf_bits == bits)
			return nir_imm_intN_t(b, o->u, bits);
		if (o->hexf_bits == 32) {
			union { uint32_t u; float f; } x = { .u = (uint32_t)o->u };

			return nir_imm_floatN_t(b, x.f, bits);
		}
		if (o->hexf_bits == 64) {
			union { uint64_t u; double d; } x = { .u = o->u };

			return nir_imm_floatN_t(b, x.d, bits);
		}
		return nir_imm_floatN_t(b, o->d, bits);
	case O_SPECIAL:
		v = special(c, o->name);
		return convert_int(b, v, bits, false);
	case O_SYM:
		return sym_value(c, o, bits);
	default:
		fail(c, "operand cannot be read");
	}
}

static void wr(struct ctx *c, const struct opnd *o, nir_def *v, struct ptype t)
{
	nir_builder *b = &c->b;
	struct preg *r;

	if (o->kind == O_SINK)
		return;
	if (o->kind != O_REG)
		fail(c, "destination is not a register");
	r = find_reg(c, o->name);
	if (r->bits == 1) {
		if (v->bit_size != 1)
			v = nir_ine_imm(b, v, 0);
	} else {
		if (v->bit_size == 1)
			v = nir_b2iN(b, v, r->bits);
		v = convert_int(b, v, r->bits, t.cls == TC_S);
	}
	nir_store_var(b, r->var, v, 1);
}

/* ---- memory ---- */

/*
 * Where an address operand points: the state space and the address in that space's form
 * (64-bit for global and generic, an offset for shared/local/param). A named variable decides
 * by itself; a register-based ld.param inside a callee is scratch, and .const is global memory.
 */
static enum space resolve_mem(struct ctx *c, const struct opnd *m, enum space isp, nir_def **addr)
{
	nir_builder *b = &c->b;
	nir_def *base = NULL;

	if (m->mem_sym) {
		struct psym *s = find_sym(c, m->name);

		if (!s)
			fail(c, "unknown name %s", m->name);
		if (s->space == SP_GLOBAL || s->space == SP_CONST) {
			*addr = nir_imm_int64(b, c->globals_va + s->offset + m->u);
			return SP_GLOBAL;
		}
		*addr = nir_imm_int(b, s->offset + (uint32_t)m->u);
		return s->space;
	}
	if (isp == SP_PARAM && c->depth)
		isp = SP_LOCAL;
	if (isp == SP_CONST)
		isp = SP_GLOBAL;
	if (m->mem_reg)
		base = nir_load_var(b, find_reg(c, m->name)->var);
	if (isp == SP_GLOBAL || isp == SP_NONE)
		*addr = base ? nir_iadd_imm(b, nir_u2u64(b, base), m->u) : nir_imm_int64(b, m->u);
	else
		*addr = base ? nir_iadd_imm(b, nir_u2u32(b, base), (uint32_t)m->u) : nir_imm_int(b, (uint32_t)m->u);
	return isp;
}

enum maccess { M_LOAD, M_STORE, M_ATOMIC, M_CAS };

struct mop {
	enum maccess kind;
	unsigned comps, bits;
	nir_def *data, *data2;		/* store value / atomic operand / cas new value */
	nir_atomic_op aop;
	unsigned align;
};

/* one access in a known state space; returns the loaded value (loads, atomics) */
static nir_def *access_space(struct ctx *c, enum space sp, nir_def *addr, const struct mop *m)
{
	nir_builder *b = &c->b;
	unsigned al = m->align;

	switch (sp) {
	case SP_GLOBAL:
		addr = nir_u2u64(b, addr);
		switch (m->kind) {
		case M_LOAD:
			return nir_load_global(b, m->comps, m->bits, addr, .align_mul = al);
		case M_STORE:
			nir_store_global(b, m->data, addr, .write_mask = BITFIELD_MASK(m->comps), .align_mul = al);
			return NULL;
		case M_ATOMIC:
			return nir_global_atomic(b, m->bits, addr, m->data, .atomic_op = m->aop);
		case M_CAS:
			return nir_global_atomic_swap(b, m->bits, addr, m->data, m->data2, .atomic_op = m->aop);
		}
		break;
	case SP_SHARED:
		addr = nir_u2u32(b, addr);
		switch (m->kind) {
		case M_LOAD:
			return nir_load_shared(b, m->comps, m->bits, addr, .align_mul = al);
		case M_STORE:
			nir_store_shared(b, m->data, addr, .write_mask = BITFIELD_MASK(m->comps), .align_mul = al);
			return NULL;
		case M_ATOMIC:
			return nir_shared_atomic(b, m->bits, addr, m->data, .atomic_op = m->aop);
		case M_CAS:
			return nir_shared_atomic_swap(b, m->bits, addr, m->data, m->data2, .atomic_op = m->aop);
		}
		break;
	case SP_LOCAL:
		addr = nir_u2u32(b, addr);
		switch (m->kind) {
		case M_LOAD:
			return nir_load_scratch(b, m->comps, m->bits, addr, .align_mul = al);
		case M_STORE:
			nir_store_scratch(b, m->data, addr, .write_mask = BITFIELD_MASK(m->comps), .align_mul = al);
			return NULL;
		default:
			fail(c, "atomics on local memory");
		}
		break;
	case SP_PARAM:
		if (m->kind != M_LOAD)
			fail(c, "only loads from kernel parameters");
		addr = nir_u2u32(b, addr);
		return nir_load_kernel_input(b, m->comps, m->bits, addr, .base = 0,
					     .range = NVC_CB_SIZE - NVC_CB_PARAMS, .align_mul = al);
	default:
		break;
	}
	fail(c, "unsupported state space");
}

/*
 * A generic address: shared window, local window, or global. Three blocks and a join, the
 * loaded value passes through a variable.
 */
static nir_def *access_generic(struct ctx *c, nir_def *addr, const struct mop *m)
{
	nir_builder *b = &c->b;
	nir_block *sh = new_block(c), *rest = new_block(c), *lo = new_block(c), *gl = new_block(c);
	nir_block *join = new_block(c);
	nir_variable *res = NULL;
	nir_def *v, *win = nir_ushr_imm(b, addr, 24);
	bool has_result = m->kind != M_STORE;
	bool local_ok = m->kind == M_LOAD || m->kind == M_STORE;

	if (has_result)
		res = nir_local_variable_create(c->impl, m->comps == 1 ? glsl_uintN_t_type(m->bits) :
						glsl_vector_type(glsl_get_base_type(glsl_uintN_t_type(m->bits)),
								 m->comps), "generic");
	nir_goto_if(b, sh, nir_ieq_imm(b, win, SHARED_WINDOW >> 24), rest);
	b->cursor = nir_after_block(sh);
	v = access_space(c, SP_SHARED, nir_iand_imm(b, addr, WINDOW_SIZE - 1), m);
	if (res)
		nir_store_var(b, res, v, BITFIELD_MASK(m->comps));
	nir_goto(b, join);
	b->cursor = nir_after_block(rest);
	nir_goto_if(b, lo, nir_ieq_imm(b, win, LOCAL_WINDOW >> 24), gl);
	b->cursor = nir_after_block(lo);
	if (local_ok) {
		v = access_space(c, SP_LOCAL, nir_iand_imm(b, addr, WINDOW_SIZE - 1), m);
		if (res)
			nir_store_var(b, res, v, BITFIELD_MASK(m->comps));
	}
	nir_goto(b, join);
	b->cursor = nir_after_block(gl);
	v = access_space(c, SP_GLOBAL, addr, m);
	if (res)
		nir_store_var(b, res, v, BITFIELD_MASK(m->comps));
	nir_goto(b, join);
	switch_to(c, join);
	return res ? nir_load_var(b, res) : NULL;
}

static enum space space_of(const char *s)
{
	if (!strcmp(s, "global"))
		return SP_GLOBAL;
	if (!strcmp(s, "shared") || !strcmp(s, "shared::cta"))
		return SP_SHARED;
	if (!strcmp(s, "local"))
		return SP_LOCAL;
	if (!strcmp(s, "param") || !strcmp(s, "param::entry"))
		return SP_PARAM;
	if (!strcmp(s, "const"))
		return SP_CONST;
	return SP_NONE;
}

/* ---- instructions ---- */

struct insn {
	char op[32];
	char mods[16][24];
	int nmods;
	struct ptype types[4];
	int ntypes;
	enum space space;
	int vec;
	struct opnd ops[8];
	int nops;
};

static bool has_mod(const struct insn *in, const char *m)
{
	for (int i = 0; i < in->nmods; i++)
		if (!strcmp(in->mods[i], m))
			return true;
	return false;
}

static nir_def *shift(nir_builder *b, const char *op, nir_def *a, nir_def *n, bool sign)
{
	/* PTX clamps the shift amount to the width; NIR masks it */
	nir_def *big = nir_uge_imm(b, n, a->bit_size), *r;

	if (!strcmp(op, "shl")) {
		r = nir_ishl(b, a, n);
		return nir_bcsel(b, big, nir_imm_intN_t(b, 0, a->bit_size), r);
	}
	if (sign) {
		r = nir_ishr(b, a, n);
		return nir_bcsel(b, big, nir_ishr_imm(b, a, a->bit_size - 1), r);
	}
	r = nir_ushr(b, a, n);
	return nir_bcsel(b, big, nir_imm_intN_t(b, 0, a->bit_size), r);
}

static nir_def *compare(struct ctx *c, const char *cmp, struct ptype t, nir_def *a, nir_def *bb)
{
	nir_builder *b = &c->b;

	if (t.cls == TC_F) {
		if (!strcmp(cmp, "eq"))	return nir_feq(b, a, bb);
		if (!strcmp(cmp, "ne"))	return nir_iand(b, nir_fneu(b, a, bb), nir_iand(b, nir_feq(b, a, a), nir_feq(b, bb, bb)));
		if (!strcmp(cmp, "lt"))	return nir_flt(b, a, bb);
		if (!strcmp(cmp, "le"))	return nir_fge(b, bb, a);
		if (!strcmp(cmp, "gt"))	return nir_flt(b, bb, a);
		if (!strcmp(cmp, "ge"))	return nir_fge(b, a, bb);
		if (!strcmp(cmp, "equ")) return nir_ior(b, nir_feq(b, a, bb), nir_ior(b, nir_fneu(b, a, a), nir_fneu(b, bb, bb)));
		if (!strcmp(cmp, "neu")) return nir_fneu(b, a, bb);
		if (!strcmp(cmp, "ltu")) return nir_inot(b, nir_fge(b, a, bb));
		if (!strcmp(cmp, "leu")) return nir_inot(b, nir_flt(b, bb, a));
		if (!strcmp(cmp, "gtu")) return nir_inot(b, nir_fge(b, bb, a));
		if (!strcmp(cmp, "geu")) return nir_inot(b, nir_flt(b, a, bb));
		if (!strcmp(cmp, "num")) return nir_iand(b, nir_feq(b, a, a), nir_feq(b, bb, bb));
		if (!strcmp(cmp, "nan")) return nir_ior(b, nir_fneu(b, a, a), nir_fneu(b, bb, bb));
	} else {
		bool s = t.cls == TC_S;

		if (!strcmp(cmp, "eq"))	return nir_ieq(b, a, bb);
		if (!strcmp(cmp, "ne"))	return nir_ine(b, a, bb);
		if (!strcmp(cmp, "lt"))	return s ? nir_ilt(b, a, bb) : nir_ult(b, a, bb);
		if (!strcmp(cmp, "le"))	return s ? nir_ige(b, bb, a) : nir_uge(b, bb, a);
		if (!strcmp(cmp, "gt"))	return s ? nir_ilt(b, bb, a) : nir_ult(b, bb, a);
		if (!strcmp(cmp, "ge"))	return s ? nir_ige(b, a, bb) : nir_uge(b, a, bb);
		if (!strcmp(cmp, "lo"))	return nir_ult(b, a, bb);
		if (!strcmp(cmp, "ls"))	return nir_uge(b, bb, a);
		if (!strcmp(cmp, "hi"))	return nir_ult(b, bb, a);
		if (!strcmp(cmp, "hs"))	return nir_uge(b, a, bb);
	}
	fail(c, "unsupported comparison %s", cmp);
}

static const char *cmp_mod(const struct insn *in)
{
	static const char *const cmps[] = { "eq", "ne", "lt", "le", "gt", "ge", "lo", "ls", "hi", "hs", "equ", "neu",
					    "ltu", "leu", "gtu", "geu", "num", "nan" };

	for (int i = 0; i < in->nmods; i++)
		for (unsigned j = 0; j < sizeof(cmps) / sizeof(cmps[0]); j++)
			if (!strcmp(in->mods[i], cmps[j]))
				return cmps[j];
	return NULL;
}

/* cvt: integer/float conversions with PTX rounding modifiers */
static nir_def *cvt(struct ctx *c, const struct insn *in, nir_def *v, struct ptype d, struct ptype s)
{
	nir_builder *b = &c->b;
	bool rzi = has_mod(in, "rzi"), rni = has_mod(in, "rni"), rmi = has_mod(in, "rmi"), rpi = has_mod(in, "rpi");
	nir_def *r;

	if (s.cls == TC_F && d.cls == TC_F) {
		r = v;
		if (s.bits != d.bits)
			r = nir_f2fN(b, v, d.bits);
		if (rzi)
			r = nir_ftrunc(b, r);
		else if (rni)
			r = nir_fround_even(b, r);
		else if (rmi)
			r = nir_ffloor(b, r);
		else if (rpi)
			r = nir_fceil(b, r);
	} else if (s.cls == TC_F) {
		if (rni)
			v = nir_fround_even(b, v);
		else if (rmi)
			v = nir_ffloor(b, v);
		else if (rpi)
			v = nir_fceil(b, v);
		/* PTX saturates out-of-range values (and NaN -> 0) */
		r = d.cls == TC_S ? nir_f2iN(b, v, d.bits < 32 ? 32 : d.bits) : nir_f2uN(b, v, d.bits < 32 ? 32 : d.bits);
		if (d.bits < 32)
			r = nir_u2uN(b, r, d.bits);
	} else if (d.cls == TC_F) {
		if (s.bits < 32)
			v = s.cls == TC_S ? nir_i2i32(b, v) : nir_u2u32(b, v);
		r = s.cls == TC_S ? nir_i2fN(b, v, d.bits) : nir_u2fN(b, v, d.bits);
	} else {
		r = convert_int(b, v, d.bits, s.cls == TC_S);
	}
	if (has_mod(in, "sat") && d.cls == TC_F)
		r = nir_fsat(b, r);
	return r;
}

static nir_atomic_op atomic_op(struct ctx *c, const struct insn *in, struct ptype t)
{
	static const struct { const char *m; nir_atomic_op u, s, f; } ops[] = {
		{ "add", nir_atomic_op_iadd, nir_atomic_op_iadd, nir_atomic_op_fadd },
		{ "min", nir_atomic_op_umin, nir_atomic_op_imin, nir_atomic_op_fmin },
		{ "max", nir_atomic_op_umax, nir_atomic_op_imax, nir_atomic_op_fmax },
		{ "and", nir_atomic_op_iand, nir_atomic_op_iand, 0 },
		{ "or", nir_atomic_op_ior, nir_atomic_op_ior, 0 },
		{ "xor", nir_atomic_op_ixor, nir_atomic_op_ixor, 0 },
		{ "exch", nir_atomic_op_xchg, nir_atomic_op_xchg, nir_atomic_op_xchg },
		{ "inc", nir_atomic_op_inc_wrap, nir_atomic_op_inc_wrap, 0 },
		{ "dec", nir_atomic_op_dec_wrap, nir_atomic_op_dec_wrap, 0 },
		{ "cas", nir_atomic_op_cmpxchg, nir_atomic_op_cmpxchg, 0 },
	};

	for (unsigned i = 0; i < sizeof(ops) / sizeof(ops[0]); i++)
		if (has_mod(in, ops[i].m))
			return t.cls == TC_F ? ops[i].f : t.cls == TC_S ? ops[i].s : ops[i].u;
	fail(c, "unsupported atomic operation");
}

static void emit_barrier(nir_builder *b, mesa_scope exec, mesa_scope mem)
{
	nir_barrier(b, .execution_scope = exec, .memory_scope = mem, .memory_semantics = NIR_MEMORY_ACQ_REL,
		    .memory_modes = nir_var_mem_shared | nir_var_mem_global);
}

static void emit_insn(struct ctx *c, struct insn *in)
{
	nir_builder *b = &c->b;
	const char *op = in->op;
	struct ptype t = in->ntypes ? in->types[0] : (struct ptype){ TC_B, 32 };
	struct opnd *o = in->ops;
	nir_def *a, *bb, *cc, *r = NULL;

	/* --- data movement --- */
	if (!strcmp(op, "mov")) {
		if (o[1].kind == O_VEC) {	/* mov.b64 %rd, {%r1, %r2}: pack */
			nir_def *lo = rd(c, &o[1].vec[0], (struct ptype){ TC_B, t.bits / 2 });
			nir_def *hi = rd(c, &o[1].vec[1], (struct ptype){ TC_B, t.bits / 2 });

			wr(c, &o[0], t.bits == 64 ? nir_pack_64_2x32_split(b, lo, hi) :
					nir_pack_32_2x16_split(b, lo, hi), t);
		} else if (o[0].kind == O_VEC) {	/* unpack */
			nir_def *v = rd(c, &o[1], t);

			wr(c, &o[0].vec[0], t.bits == 64 ? nir_unpack_64_2x32_split_x(b, v) :
					       nir_unpack_32_2x16_split_x(b, v), (struct ptype){ TC_B, t.bits / 2 });
			wr(c, &o[0].vec[1], t.bits == 64 ? nir_unpack_64_2x32_split_y(b, v) :
					       nir_unpack_32_2x16_split_y(b, v), (struct ptype){ TC_B, t.bits / 2 });
		} else {
			wr(c, &o[0], rd(c, &o[1], t), t);
		}
		return;
	}
	if (!strcmp(op, "cvta")) {
		/* cvta.space.size d, a (to generic) / cvta.to.space.size d, a (from generic) */
		bool to = has_mod(in, "to");
		enum space sp = SP_NONE;

		for (int i = 0; i < in->nmods; i++)
			if (space_of(in->mods[i]) != SP_NONE)
				sp = space_of(in->mods[i]);
		a = rd(c, &o[1], t);
		if (sp == SP_GLOBAL || sp == SP_CONST || sp == SP_PARAM)
			r = a;
		else if (sp == SP_SHARED || sp == SP_LOCAL)
			r = to ? nir_iand_imm(b, a, WINDOW_SIZE - 1) :
				 nir_iadd_imm(b, a, sp == SP_SHARED ? SHARED_WINDOW : LOCAL_WINDOW);
		else
			fail(c, "cvta without a state space");
		wr(c, &o[0], r, t);
		return;
	}
	if (!strcmp(op, "cvt")) {
		struct ptype d = in->types[0], s = in->ntypes > 1 ? in->types[1] : in->types[0];

		wr(c, &o[0], cvt(c, in, rd(c, &o[1], s), d, s), d);
		return;
	}
	if (!strcmp(op, "ld") || !strcmp(op, "st") || !strcmp(op, "ldu")) {
		bool store = op[0] == 's';
		struct opnd *mem = store ? &o[0] : &o[1], *val = store ? &o[1] : &o[0];
		struct mop m = { .kind = store ? M_STORE : M_LOAD, .comps = in->vec ? in->vec : 1, .bits = t.bits,
				 .align = t.bits / 8 * (in->vec ? in->vec : 1) };
		nir_def *addr;
		enum space sp;

		if (mem->kind != O_MEM)
			fail(c, "%s without an address", op);
		if (store) {
			if (in->vec) {
				nir_def *comps[4];

				for (int i = 0; i < in->vec; i++)
					comps[i] = rd(c, &val->vec[i], t);
				m.data = nir_vec(b, comps, in->vec);
			} else {
				m.data = rd(c, val, t);
			}
		}
		sp = resolve_mem(c, mem, in->space, &addr);
		r = sp == SP_NONE ? access_generic(c, addr, &m) : access_space(c, sp, addr, &m);
		if (!store) {
			if (in->vec)
				for (int i = 0; i < in->vec; i++)
					wr(c, &val->vec[i], nir_channel(b, r, i), t);
			else
				wr(c, val, r, t);
		}
		return;
	}
	if (!strcmp(op, "atom") || !strcmp(op, "red")) {
		bool red = op[0] == 'r';
		struct opnd *mem = red ? &o[0] : &o[1], *src = red ? &o[1] : &o[2];
		struct mop m = { .kind = M_ATOMIC, .comps = 1, .bits = t.bits, .aop = atomic_op(c, in, t) };
		nir_def *addr;
		enum space sp = resolve_mem(c, mem, in->space, &addr);

		if (has_mod(in, "release") || has_mod(in, "acq_rel"))
			emit_barrier(b, SCOPE_NONE, SCOPE_DEVICE);
		m.data = rd(c, src, t);
		if (m.aop == nir_atomic_op_cmpxchg) {
			m.kind = M_CAS;
			m.data2 = rd(c, &o[3], t);
		}
		r = sp == SP_NONE ? access_generic(c, addr, &m) : access_space(c, sp, addr, &m);
		if (has_mod(in, "acquire") || has_mod(in, "acq_rel"))
			emit_barrier(b, SCOPE_NONE, SCOPE_DEVICE);
		if (!red)
			wr(c, &o[0], r, t);
		return;
	}

	/* --- control --- */
	if (!strcmp(op, "bar") || !strcmp(op, "barrier")) {
		if (has_mod(in, "warp")) {
			emit_barrier(b, SCOPE_SUBGROUP, SCOPE_SUBGROUP);
			return;
		}
		emit_barrier(b, SCOPE_WORKGROUP, SCOPE_WORKGROUP);
		return;
	}
	if (!strcmp(op, "membar") || !strcmp(op, "fence")) {
		mesa_scope s = has_mod(in, "cta") ? SCOPE_WORKGROUP : SCOPE_DEVICE;

		emit_barrier(b, SCOPE_NONE, s);
		return;
	}

	/* --- warp-level --- */
	if (!strcmp(op, "shfl")) {
		/* shfl.sync.mode.b32 d[|p], a, b, c, membermask (PTX ISA: lane selection) */
		/* d|p, a, b, c[, membermask]: the predicate output is optional */
		int k = in->nops > (has_mod(in, "sync") ? 5 : 4) ? 1 : 0;
		nir_def *lane = nir_load_subgroup_invocation(b), *j, *maxl, *minl, *seg, *pval;
		nir_def *bv = nir_iand_imm(b, rd(c, &o[2 + k], (struct ptype){ TC_B, 32 }), 0x1f);
		nir_def *cv = rd(c, &o[3 + k], (struct ptype){ TC_B, 32 });

		a = rd(c, &o[1 + k], (struct ptype){ TC_B, 32 });
		seg = nir_iand_imm(b, nir_ushr_imm(b, cv, 8), 0x1f);
		maxl = nir_ior(b, nir_iand(b, lane, seg), nir_iand(b, nir_iand_imm(b, cv, 0x1f), nir_inot(b, seg)));
		minl = nir_iand(b, lane, seg);
		if (has_mod(in, "up")) {
			j = nir_isub(b, lane, bv);
			pval = nir_ige(b, j, maxl);
		} else if (has_mod(in, "down")) {
			j = nir_iadd(b, lane, bv);
			pval = nir_uge(b, maxl, j);
		} else if (has_mod(in, "bfly")) {
			j = nir_ixor(b, lane, bv);
			pval = nir_uge(b, maxl, j);
		} else if (has_mod(in, "idx")) {
			j = nir_ior(b, minl, nir_iand(b, bv, nir_inot(b, seg)));
			pval = nir_uge(b, maxl, j);
		} else {
			fail(c, "shfl mode");
		}
		j = nir_bcsel(b, pval, j, lane);
		wr(c, &o[0], nir_shuffle(b, a, j), (struct ptype){ TC_B, 32 });
		if (k)
			wr(c, &o[1], pval, (struct ptype){ TC_PRED, 1 });
		return;
	}
	if (!strcmp(op, "vote")) {
		if (has_mod(in, "ballot")) {
			wr(c, &o[0], nir_ballot(b, 1, 32, rd(c, &o[1], (struct ptype){ TC_PRED, 1 })), t);
			return;
		}
		a = rd(c, &o[1], (struct ptype){ TC_PRED, 1 });
		if (has_mod(in, "all"))
			r = nir_vote_all(b, 1, a);
		else if (has_mod(in, "any"))
			r = nir_vote_any(b, 1, a);
		else if (has_mod(in, "uni"))
			r = nir_vote_ieq(b, 1, a);
		else
			fail(c, "vote mode");
		wr(c, &o[0], r, (struct ptype){ TC_PRED, 1 });
		return;
	}
	if (!strcmp(op, "activemask")) {
		wr(c, &o[0], nir_ballot(b, 1, 32, nir_imm_true(b)), (struct ptype){ TC_B, 32 });
		return;
	}

	/* --- predicates and selection --- */
	if (!strcmp(op, "setp")) {
		const char *cmp = cmp_mod(in);
		bool boolop = has_mod(in, "and") || has_mod(in, "or") || has_mod(in, "xor");
		nir_def *res;
		/* setp.cmp[.bool].t p[|q], a, b[, c]: "p|q" arrives as two operands */
		int ai = in->nops == (boolop ? 5 : 4) ? 2 : 1;

		if (!cmp)
			fail(c, "setp without a comparison");
		res = compare(c, cmp, t, rd(c, &o[ai], t), rd(c, &o[ai + 1], t));
		if (has_mod(in, "and"))
			res = nir_iand(b, res, rd(c, &o[ai + 2], (struct ptype){ TC_PRED, 1 }));
		else if (has_mod(in, "or"))
			res = nir_ior(b, res, rd(c, &o[ai + 2], (struct ptype){ TC_PRED, 1 }));
		else if (has_mod(in, "xor"))
			res = nir_ixor(b, res, rd(c, &o[ai + 2], (struct ptype){ TC_PRED, 1 }));
		wr(c, &o[0], res, (struct ptype){ TC_PRED, 1 });
		if (ai == 2)
			wr(c, &o[1], nir_inot(b, res), (struct ptype){ TC_PRED, 1 });
		return;
	}
	if (!strcmp(op, "selp")) {
		wr(c, &o[0], nir_bcsel(b, rd(c, &o[3], (struct ptype){ TC_PRED, 1 }), rd(c, &o[1], t),
					  rd(c, &o[2], t)), t);
		return;
	}

	/* --- arithmetic --- */
	if (in->nops >= 2)
		a = rd(c, &o[1], t);
	else
		a = NULL;
	bb = in->nops >= 3 ? rd(c, &o[2], t) : NULL;
	cc = in->nops >= 4 ? rd(c, &o[3], t) : NULL;

	if (t.cls == TC_PRED) {
		if (!strcmp(op, "and"))		r = nir_iand(b, a, bb);
		else if (!strcmp(op, "or"))	r = nir_ior(b, a, bb);
		else if (!strcmp(op, "xor"))	r = nir_ixor(b, a, bb);
		else if (!strcmp(op, "not"))	r = nir_inot(b, a);
		else fail(c, "unsupported predicate instruction %s", op);
		wr(c, &o[0], r, t);
		return;
	}
	if (t.cls == TC_F) {
		if (!strcmp(op, "add"))		r = nir_fadd(b, a, bb);
		else if (!strcmp(op, "sub"))	r = nir_fsub(b, a, bb);
		else if (!strcmp(op, "mul"))	r = nir_fmul(b, a, bb);
		else if (!strcmp(op, "fma") || !strcmp(op, "mad")) r = nir_ffma(b, a, bb, cc);
		else if (!strcmp(op, "div"))	r = nir_fdiv(b, a, bb);
		else if (!strcmp(op, "neg"))	r = nir_fneg(b, a);
		else if (!strcmp(op, "abs"))	r = nir_fabs(b, a);
		else if (!strcmp(op, "min"))	r = nir_fmin(b, a, bb);
		else if (!strcmp(op, "max"))	r = nir_fmax(b, a, bb);
		else if (!strcmp(op, "sqrt"))	r = nir_fsqrt(b, a);
		else if (!strcmp(op, "rsqrt"))	r = nir_frsq(b, a);
		else if (!strcmp(op, "rcp"))	r = nir_frcp(b, a);
		else if (!strcmp(op, "sin"))	r = nir_fsin(b, a);
		else if (!strcmp(op, "cos"))	r = nir_fcos(b, a);
		else if (!strcmp(op, "ex2"))	r = nir_fexp2(b, a);
		else if (!strcmp(op, "lg2"))	r = nir_flog2(b, a);
		else if (!strcmp(op, "copysign"))	/* magnitude of b, sign of a */
			r = nir_bcsel(b, nir_flt(b, a, nir_imm_floatN_t(b, 0.0, t.bits)), nir_fneg(b, nir_fabs(b, bb)), nir_fabs(b, bb));
		else if (!strcmp(op, "tanh")) {
			nir_def *e = nir_fexp2(b, nir_fmul_imm(b, a, 2.0 * 1.4426950408889634));

			r = nir_fdiv(b, nir_fadd_imm(b, e, -1.0), nir_fadd_imm(b, e, 1.0));
		} else if (!strcmp(op, "testp")) {
			if (has_mod(in, "nan"))		r = nir_fneu(b, a, a);
			else if (has_mod(in, "finite"))	r = nir_flt(b, nir_fabs(b, a), nir_imm_floatN_t(b, INFINITY, t.bits));
			else if (has_mod(in, "infinite")) r = nir_feq(b, nir_fabs(b, a), nir_imm_floatN_t(b, INFINITY, t.bits));
			else if (has_mod(in, "number"))	r = nir_feq(b, a, a);
			else fail(c, "testp mode");
			wr(c, &o[0], r, (struct ptype){ TC_PRED, 1 });
			return;
		} else fail(c, "unsupported float instruction %s", op);
		if (has_mod(in, "sat"))
			r = nir_fsat(b, r);
		wr(c, &o[0], r, t);
		return;
	}

	/* integer / bit */
	{
		bool s = t.cls == TC_S;

		if (!strcmp(op, "add"))
			r = has_mod(in, "sat") ? nir_iadd_sat(b, a, bb) : nir_iadd(b, a, bb);
		else if (!strcmp(op, "sub"))
			r = has_mod(in, "sat") ? nir_isub_sat(b, a, bb) : nir_isub(b, a, bb);
		else if (!strcmp(op, "mul") || !strcmp(op, "mad")) {
			bool mad = op[1] == 'a';

			if (has_mod(in, "wide")) {
				unsigned w = t.bits * 2;
				nir_def *aw = s ? nir_i2iN(b, a, w) : nir_u2uN(b, a, w);
				nir_def *bw = s ? nir_i2iN(b, bb, w) : nir_u2uN(b, bb, w);

				r = nir_imul(b, aw, bw);
				if (mad)
					r = nir_iadd(b, r, rd(c, &o[3], (struct ptype){ t.cls, w }));
				wr(c, &o[0], r, (struct ptype){ t.cls, w });
				return;
			}
			r = has_mod(in, "hi") ? (s ? nir_imul_high(b, a, bb) : nir_umul_high(b, a, bb)) : nir_imul(b, a, bb);
			if (mad)
				r = nir_iadd(b, r, cc);
		} else if (!strcmp(op, "mul24") || !strcmp(op, "mad24")) {
			r = s ? nir_imul(b, nir_ibfe_imm(b, a, 0, 24), nir_ibfe_imm(b, bb, 0, 24)) :
				nir_imul(b, nir_iand_imm(b, a, 0xffffff), nir_iand_imm(b, bb, 0xffffff));
			if (op[1] == 'a')
				r = nir_iadd(b, r, cc);
		} else if (!strcmp(op, "div"))
			r = s ? nir_idiv(b, a, bb) : nir_udiv(b, a, bb);
		else if (!strcmp(op, "rem"))
			r = s ? nir_irem(b, a, bb) : nir_umod(b, a, bb);
		else if (!strcmp(op, "abs"))
			r = nir_iabs(b, a);
		else if (!strcmp(op, "neg"))
			r = nir_ineg(b, a);
		else if (!strcmp(op, "min"))
			r = s ? nir_imin(b, a, bb) : nir_umin(b, a, bb);
		else if (!strcmp(op, "max"))
			r = s ? nir_imax(b, a, bb) : nir_umax(b, a, bb);
		else if (!strcmp(op, "and"))
			r = nir_iand(b, a, bb);
		else if (!strcmp(op, "or"))
			r = nir_ior(b, a, bb);
		else if (!strcmp(op, "xor"))
			r = nir_ixor(b, a, bb);
		else if (!strcmp(op, "not"))
			r = nir_inot(b, a);
		else if (!strcmp(op, "cnot"))
			r = nir_b2iN(b, nir_ieq_imm(b, a, 0), t.bits);
		else if (!strcmp(op, "shl") || !strcmp(op, "shr"))
			r = shift(b, op, a, rd(c, &o[2], (struct ptype){ TC_U, 32 }), s);
		else if (!strcmp(op, "popc")) {
			wr(c, &o[0], nir_bit_count(b, a), (struct ptype){ TC_U, 32 });
			return;
		} else if (!strcmp(op, "clz")) {
			nir_def *msb = nir_ufind_msb(b, a);	/* -1 for zero */

			wr(c, &o[0], nir_isub(b, nir_imm_int(b, t.bits - 1), msb), (struct ptype){ TC_U, 32 });
			return;
		} else if (!strcmp(op, "brev")) {
			r = t.bits == 64 ? nir_pack_64_2x32_split(b, nir_bitfield_reverse(b, nir_unpack_64_2x32_split_y(b, a)),
								  nir_bitfield_reverse(b, nir_unpack_64_2x32_split_x(b, a))) :
					   nir_bitfield_reverse(b, a);
		} else if (!strcmp(op, "bfe")) {
			nir_def *pos = rd(c, &o[2], (struct ptype){ TC_U, 32 }), *len = rd(c, &o[3], (struct ptype){ TC_U, 32 });

			if (t.bits != 32)
				fail(c, "bfe.%u", t.bits);
			r = s ? nir_ibitfield_extract(b, a, pos, len) : nir_ubitfield_extract(b, a, pos, len);
			r = nir_bcsel(b, nir_ieq_imm(b, len, 0), nir_imm_int(b, 0), r);
		} else if (!strcmp(op, "bfi")) {
			/* bfi d, a (insert), b (base), pos, len: len bits of a at bit pos of b */
			nir_def *pos = rd(c, &o[3], (struct ptype){ TC_U, 32 }), *len = rd(c, &o[4], (struct ptype){ TC_U, 32 });

			if (t.bits == 32) {
				r = nir_bitfield_insert(b, bb, a, pos, len);
			} else {
				nir_def *one = nir_imm_intN_t(b, 1, t.bits), *mask, *p64 = nir_u2uN(b, pos, t.bits);

				mask = nir_iadd_imm(b, nir_ishl(b, one, nir_u2uN(b, len, t.bits)), -1);
				mask = nir_bcsel(b, nir_uge_imm(b, len, t.bits), nir_imm_intN_t(b, ~0ull, t.bits), mask);
				mask = nir_ishl(b, mask, p64);
				mask = nir_bcsel(b, nir_uge_imm(b, pos, t.bits), nir_imm_intN_t(b, 0, t.bits), mask);
				r = nir_ior(b, nir_iand(b, bb, nir_inot(b, mask)), nir_iand(b, nir_ishl(b, a, p64), mask));
			}
		} else if (!strcmp(op, "shf")) {
			/* funnel shift of {b:a} by c: shf.l d = high word of (b:a) << n, shf.r d = low word of >> n */
			nir_def *n = rd(c, &o[3], (struct ptype){ TC_U, 32 }), *w = nir_imm_int(b, 32), *rest;

			n = has_mod(in, "clamp") ? nir_umin(b, n, w) : nir_iand_imm(b, n, 31);
			if (t.bits != 32)
				fail(c, "shf.%u", t.bits);
			if (has_mod(in, "l")) {
				rest = nir_ior(b, nir_ishl(b, bb, n), nir_ushr(b, a, nir_isub(b, w, n)));
				r = nir_bcsel(b, nir_ieq_imm(b, n, 0), bb, nir_bcsel(b, nir_ieq_imm(b, n, 32), a, rest));
			} else {
				rest = nir_ior(b, nir_ushr(b, a, n), nir_ishl(b, bb, nir_isub(b, w, n)));
				r = nir_bcsel(b, nir_ieq_imm(b, n, 0), a, nir_bcsel(b, nir_ieq_imm(b, n, 32), bb, rest));
			}
		} else if (!strcmp(op, "prmt")) {
			/* default mode: each selector nibble picks one of the 8 bytes of {b, a}, bit 3 = replicate sign */
			nir_def *sel = rd(c, &o[3], (struct ptype){ TC_U, 32 });
			nir_def *both = nir_pack_64_2x32_split(b, a, bb);

			if (in->nmods > 1 && strcmp(in->mods[0], "b32"))
				fail(c, "prmt mode");
			r = nir_imm_int(b, 0);
			for (int i = 0; i < 4; i++) {
				nir_def *n = nir_iand_imm(b, nir_ushr_imm(b, sel, 4 * i), 0xf);
				nir_def *byte = nir_u2u32(b, nir_iand_imm(b, nir_ushr(b, both, nir_imul_imm(b, nir_iand_imm(b, n, 7), 8)), 0xff));
				nir_def *sgn = nir_bcsel(b, nir_ine_imm(b, nir_iand_imm(b, byte, 0x80), 0), nir_imm_int(b, 0xff), nir_imm_int(b, 0));

				byte = nir_bcsel(b, nir_ine_imm(b, nir_iand_imm(b, n, 8), 0), sgn, byte);
				r = nir_ior(b, r, nir_ishl_imm(b, byte, 8 * i));
			}
		} else if (!strcmp(op, "bfind")) {
			nir_def *msb = s ? nir_ifind_msb(b, a) : nir_ufind_msb(b, a);

			if (has_mod(in, "shiftamt"))
				msb = nir_bcsel(b, nir_ilt_imm(b, msb, 0), msb, nir_isub(b, nir_imm_int(b, t.bits - 1), msb));
			wr(c, &o[0], msb, (struct ptype){ TC_U, 32 });
			return;
		} else
			fail(c, "unsupported instruction %s.%s", op, in->nmods ? in->mods[0] : "");
		wr(c, &o[0], r, t);
	}
}

/* parse one instruction (cursor on the opcode) into *in */
static void parse_insn(struct ctx *c, struct insn *in)
{
	char opc[256], *save, *tok;

	memset(in, 0, sizeof(*in));
	snprintf(opc, sizeof(opc), "%s", c->lx.s);
	next(c);
	tok = strtok_r(opc, ".", &save);
	snprintf(in->op, sizeof(in->op), "%s", tok);
	while ((tok = strtok_r(NULL, ".", &save))) {
		struct ptype t;

		if (parse_type(tok, &t) && in->ntypes < 4) {
			in->types[in->ntypes++] = t;
		} else if (!strcmp(tok, "v2") || !strcmp(tok, "v4") || !strcmp(tok, "v8")) {
			in->vec = tok[1] - '0';
		} else {
			if (space_of(tok) != SP_NONE && in->space == SP_NONE)
				in->space = space_of(tok);
			if (in->nmods < 16)
				snprintf(in->mods[in->nmods++], sizeof(in->mods[0]), "%s", tok);
		}
	}
	/* cvta names its space without being a memory access */
	if (!strcmp(in->op, "cvta") || !strcmp(in->op, "isspacep"))
		in->space = SP_NONE;
	if (!strcmp(in->op, "ld") || !strcmp(in->op, "st") || !strcmp(in->op, "ldu") || !strcmp(in->op, "atom") ||
	    !strcmp(in->op, "red")) {
		if (in->space == SP_NONE && has_mod(in, "generic"))
			in->space = SP_NONE;
	} else if (strcmp(in->op, "cvta")) {
		in->space = SP_NONE;
	}
	while (!is_p(c, ';')) {
		if (in->nops == 8)
			fail(c, "too many operands");
		parse_operand(c, &in->ops[in->nops++]);
		if (!accept(c, ',') && !accept(c, '|'))
			break;
	}
	expect(c, ';');
}

/* a label, a directive, or one (possibly predicated) instruction */
static void emit_statement(struct ctx *c)
{
	struct insn in;
	struct opnd pred;
	bool has_pred = false;
	nir_block *then = NULL, *join = NULL;

	if (accept(c, '@')) {
		parse_operand(c, &pred);
		has_pred = true;
	}
	if (c->lx.t != T_ID)
		fail(c, "expected an instruction");
	if (!strncmp(c->lx.s, "tex.", 4) || !strncmp(c->lx.s, "tld4.", 5) || !strncmp(c->lx.s, "suld.", 5) ||
	    !strncmp(c->lx.s, "sust.", 5) || !strncmp(c->lx.s, "txq.", 4) || !strncmp(c->lx.s, "suq.", 4))
		fail(c, "textures and surfaces (%s) are not supported yet", c->lx.s);
	if (!strcmp(c->lx.s, "call") || !strncmp(c->lx.s, "call.", 5)) {
		/* call[.uni] [(ret, ...),] name[, (arg, ...)]; -- inlined */
		char rets[8][256], args[32][256], fname[256];
		int nrets = 0, nargs = 0;

		next(c);
		if (accept(c, '(')) {
			do {
				if (nrets == 8)
					fail(c, "too many return values");
				snprintf(rets[nrets++], 256, "%s", expect_id(c));
			} while (accept(c, ','));
			expect(c, ')');
			expect(c, ',');
		}
		snprintf(fname, sizeof(fname), "%s", expect_id(c));
		if (accept(c, ',')) {
			expect(c, '(');
			if (!accept(c, ')')) {
				do {
					if (nargs == 32)
						fail(c, "too many arguments");
					snprintf(args[nargs++], 256, "%s", expect_id(c));
				} while (accept(c, ','));
				expect(c, ')');
			}
			if (accept(c, ','))
				fail(c, "indirect call (prototype)");
		}
		expect(c, ';');
		if (c->terminated)
			return;
		if (has_pred) {
			then = new_block(c);
			join = new_block(c);
			nir_goto_if(&c->b, then, rd(c, &pred, (struct ptype){ TC_PRED, 1 }), join);
			switch_to(c, then);
		}
		inline_call(c, rets, nrets, fname, args, nargs);
		if (has_pred) {
			nir_goto(&c->b, join);
			switch_to(c, join);
		}
		return;
	}
	if (c->terminated) {	/* unreachable code after a jump, up to the next label */
		parse_insn(c, &in);
		return;
	}
	parse_insn(c, &in);
	if (!strcmp(in.op, "bra") || !strcmp(in.op, "brx")) {
		nir_block *target, *fall;

		if (in.nops != 1 || in.ops[0].kind != O_SYM)
			fail(c, "bra needs a label");
		target = label_block(c, in.ops[0].name);
		if (has_pred) {
			fall = new_block(c);
			nir_goto_if(&c->b, target, rd(c, &pred, (struct ptype){ TC_PRED, 1 }), fall);
			switch_to(c, fall);
		} else {
			nir_goto(&c->b, target);
			c->terminated = true;
		}
		return;
	}
	if (!strcmp(in.op, "ret") || !strcmp(in.op, "exit") || !strcmp(in.op, "trap")) {
		nir_block *target = !strcmp(in.op, "ret") ? frame(c)->ret : c->impl->end_block;

		if (!strcmp(in.op, "trap") && !has_pred)
			fail(c, "trap");
		if (has_pred) {
			nir_block *fall = new_block(c);

			nir_goto_if(&c->b, target, rd(c, &pred, (struct ptype){ TC_PRED, 1 }), fall);
			switch_to(c, fall);
		} else {
			nir_goto(&c->b, target);
			c->terminated = true;
		}
		return;
	}
	if (has_pred) {
		then = new_block(c);
		join = new_block(c);
		nir_goto_if(&c->b, then, rd(c, &pred, (struct ptype){ TC_PRED, 1 }), join);
		switch_to(c, then);
	}
	emit_insn(c, &in);
	if (has_pred) {
		nir_goto(&c->b, join);
		switch_to(c, join);
	}
}

/* .reg .t %r<N>, %x, ...; */
static void parse_reg_decl(struct ctx *c)
{
	struct ptype t;
	const char *ts;

	next(c);	/* .reg */
	if (is_id(c, ".v2") || is_id(c, ".v4"))
		fail(c, "vector registers");
	ts = expect_id(c);
	if (ts[0] != '.' || !parse_type(ts + 1, &t))
		fail(c, "register type %s", ts);
	do {
		char name[256];

		snprintf(name, sizeof(name), "%s", expect_id(c));
		if (accept(c, '<')) {
			uint64_t n = expect_int(c);

			expect(c, '>');
			for (uint64_t i = 0; i < n; i++) {
				char full[300];

				snprintf(full, sizeof(full), "%s%u", name, (unsigned)i);
				declare_reg(c, full, t);
			}
		} else {
			declare_reg(c, name, t);
		}
	} while (accept(c, ','));
	expect(c, ';');
}

/* [.align N] .type name[N]... in a state space; returns the variable's size and alignment */
static void parse_var(struct ctx *c, char *name, size_t len, uint32_t *size, uint32_t *align, bool *unsized,
		      struct ptype *elem)
{
	struct ptype t = { TC_B, 8 };
	uint32_t al = 0, count = 1;

	bool ptr = false;

	*unsized = false;
	while (c->lx.t == T_ID && c->lx.s[0] == '.') {
		if (is_id(c, ".align")) {
			next(c);
			if (ptr)	/* .ptr [.space] .align N: the pointee's alignment */
				expect_int(c);
			else
				al = expect_int(c);
		} else if (is_id(c, ".ptr")) {
			ptr = true;
			next(c);
		} else if (is_id(c, ".global") || is_id(c, ".shared") || is_id(c, ".local") || is_id(c, ".const")) {
			next(c);
		} else if (is_id(c, ".v2") || is_id(c, ".v4")) {
			count *= c->lx.s[2] - '0';
			next(c);
		} else if (parse_type(c->lx.s + 1, &t)) {
			next(c);
		} else {
			fail(c, "unexpected %s", c->lx.s);
		}
	}
	snprintf(name, len, "%s", expect_id(c));
	while (accept(c, '[')) {
		if (accept(c, ']')) {
			*unsized = true;
			continue;
		}
		count *= expect_int(c);
		expect(c, ']');
	}
	*size = count * (t.cls == TC_PRED ? 1 : t.bits / 8);
	*align = al ? al : (t.cls == TC_PRED ? 1 : t.bits / 8);
	if (elem)
		*elem = t;
}

static void parse_body(struct ctx *c);
static void inline_call(struct ctx *c, char rets[][256], int nrets, const char *fname, char args[][256], int nargs);

static void parse_body(struct ctx *c)
{
	uint32_t scope_top[64];
	int scope = 0;

	expect(c, '{');
	for (;;) {
		if (c->lx.t == T_EOF)
			fail(c, "unterminated function");
		if (accept(c, '}')) {
			if (!scope)
				break;
			c->local_top = scope_top[--scope];	/* the scope's .param/.local variables die */
			continue;
		}
		if (accept(c, '{')) {	/* nested scope (nvcc: one per call); names stay visible */
			if (scope == 64)
				fail(c, "scopes nested too deep");
			scope_top[scope++] = c->local_top;
			continue;
		}
		if (c->lx.t == T_ID && c->lx.s[0] == '.') {
			if (is_id(c, ".reg")) {
				parse_reg_decl(c);
			} else if (is_id(c, ".local") || is_id(c, ".shared") || is_id(c, ".param")) {
				bool shared = is_id(c, ".shared"), unsized;
				char name[256];
				uint32_t size, al;

				next(c);
				parse_var(c, name, sizeof(name), &size, &al, &unsized, NULL);
				expect(c, ';');
				if (shared && _mesa_hash_table_search(c->modsyms, name)) {
					/* allocated by prescan_shared */
				} else if (shared) {
					c->shared_size = align_up(c->shared_size, al);
					add_sym_to(c, c->modsyms, name, SP_SHARED, c->shared_size, size);
					c->shared_size += size;
				} else {	/* .local, and a call's .param variables: scratch */
					add_sym(c, name, SP_LOCAL, alloc_local(c, size, al), size);
				}
			} else if (is_id(c, ".loc")) {	/* debug line info: the rest of the line, no ';' */
				int line = c->lx.line;

				while (c->lx.t != T_EOF && c->lx.line == line)
					next(c);
			} else if (is_id(c, ".pragma")) {
				while (!is_p(c, ';') && c->lx.t != T_EOF)
					next(c);
				expect(c, ';');
			} else {
				fail(c, "unsupported directive %s", c->lx.s);
			}
			continue;
		}
		/* label? */
		if (c->lx.t == T_ID && c->lx.s[0] != '@') {
			const char *save_p = c->lx.p;
			int save_line = c->lx.line;
			char name[256];

			snprintf(name, sizeof(name), "%s", c->lx.s);
			next(c);
			if (accept(c, ':')) {
				nir_block *blk = label_block(c, name);

				if (!c->terminated)
					nir_goto(&c->b, blk);
				switch_to(c, blk);
				continue;
			}
			/* not a label: rewind to the opcode */
			c->lx.p = save_p;
			c->lx.line = save_line;
			c->lx.t = T_ID;
			snprintf(c->lx.s, sizeof(c->lx.s), "%s", name);
		}
		emit_statement(c);
	}
	if (!c->terminated)
		nir_goto(&c->b, frame(c)->ret);
}

/*
 * Inline a call: the callee's header binds its .param names to the caller's variables (return
 * values first, then arguments), then its body is emitted in a frame of its own with ret going
 * to a fresh block. The lexer resumes after the call afterwards.
 */
static void inline_call(struct ctx *c, char rets[][256], int nrets, const char *fname, char args[][256], int nargs)
{
	struct hash_entry *e = _mesa_hash_table_search(c->funcs, fname);
	struct pfunc *f = e ? e->data : NULL;
	struct lex saved = c->lx;
	struct frame *fr;
	struct psym *bind[40];
	uint32_t saved_top = c->local_top;
	nir_block *ret;
	int i, n = 0;

	if (!f)
		fail(c, "call of unknown function %s%s", fname, find_reg(c, fname) ? " (indirect calls)" : "");
	if (!f->has_body)
		fail(c, "call of %s, which has no body in this module", fname);
	if (c->depth == MAX_DEPTH)
		fail(c, "calls nested deeper than %d (recursion?)", MAX_DEPTH);
	for (i = 0; i < nrets; i++) {
		bind[n] = find_sym(c, rets[i]);
		if (!bind[n] || bind[n]->space != SP_LOCAL)
			fail(c, "return variable %s is not a .param", rets[i]);
		n++;
	}
	for (i = 0; i < nargs; i++) {
		bind[n] = find_sym(c, args[i]);
		if (!bind[n] || bind[n]->space != SP_LOCAL)
			fail(c, "argument %s is not a .param variable (registers as arguments are not supported)", args[i]);
		n++;
	}
	ret = new_block(c);
	c->depth++;
	fr = frame(c);
	fr->regs = new_table(c);
	fr->labels = new_table(c);
	fr->syms = new_table(c);
	fr->ret = ret;
	c->lx.p = f->hdr;
	c->lx.line = f->line;
	next(c);
	/* ( .param ret, ... ) name ( .param arg, ... ) */
	i = 0;
	if (accept(c, '(')) {
		do {
			char name[256];
			uint32_t size, al;
			bool unsized;

			if (!is_id(c, ".param"))
				fail(c, "expected .param");
			next(c);
			parse_var(c, name, sizeof(name), &size, &al, &unsized, NULL);
			if (i >= nrets)
				fail(c, "%s returns more values than the call takes", fname);
			_mesa_hash_table_insert(fr->syms, ralloc_strdup(c->mem, name), bind[i++]);
		} while (accept(c, ','));
		expect(c, ')');
	}
	if (i != nrets)
		fail(c, "%s: %d return values, call takes %d", fname, i, nrets);
	expect_id(c);
	i = 0;
	if (accept(c, '(') && !accept(c, ')')) {
		do {
			char name[256];
			uint32_t size, al;
			bool unsized;

			if (!is_id(c, ".param"))
				fail(c, "expected .param");
			next(c);
			parse_var(c, name, sizeof(name), &size, &al, &unsized, NULL);
			if (i >= nargs)
				fail(c, "%s takes more arguments than the call passes", fname);
			_mesa_hash_table_insert(fr->syms, ralloc_strdup(c->mem, name), bind[nrets + i++]);
		} while (accept(c, ','));
		expect(c, ')');
	}
	if (i != nargs)
		fail(c, "%s: %d parameters, call passes %d", fname, i, nargs);
	while (!is_p(c, '{') && c->lx.t != T_EOF)	/* .noreturn, .maxnreg, ... */
		next(c);
	parse_body(c);
	c->depth--;
	c->local_top = saved_top;
	c->lx = saved;
	switch_to(c, ret);
}

/*
 * Static .shared variables of a body (cursor on its '{'), so dynamic (extern) shared memory can
 * start after all of them. Every function's body is scanned, whether or not it is called.
 */
static void prescan_shared(struct ctx *c)
{
	int depth = 0;

	do {
		if (c->lx.t == T_EOF)
			fail(c, "unterminated function");
		if (is_p(c, '{')) {
			depth++;
		} else if (is_p(c, '}')) {
			depth--;
		} else if (is_id(c, ".shared")) {
			char name[256];
			uint32_t size, al;
			bool unsized;

			next(c);
			parse_var(c, name, sizeof(name), &size, &al, &unsized, NULL);
			if (!_mesa_hash_table_search(c->modsyms, name)) {
				c->shared_size = align_up(c->shared_size, al);
				add_sym_to(c, c->modsyms, name, SP_SHARED, c->shared_size, size);
				c->shared_size += size;
			}
			continue;
		}
		next(c);
	} while (depth);
}

/* ---- module globals (.global / .const) with initializers ---- */

static void init_write(struct ctx *c, uint8_t *dst, uint32_t size, uint32_t *pos, struct ptype t, uint64_t bits)
{
	uint32_t n = t.bits / 8;

	if (*pos + n > size)
		fail(c, "initializer longer than the variable");
	memcpy(dst + *pos, &bits, n);	/* little-endian host and GPU */
	*pos += n;
}

/* = { ... } or = value; nested braces are flattened; elements take the variable's element type */
static void parse_init(struct ctx *c, uint8_t *dst, uint32_t size, struct ptype t)
{
	uint32_t pos = 0;
	int depth = 0;

	do {
		if (accept(c, '{')) {
			depth++;
			continue;
		}
		if (is_p(c, '}')) {
			next(c);
			if (--depth < 0)
				fail(c, "unbalanced initializer");
			if (!depth)
				break;
			accept(c, ',');
			continue;
		}
		if (accept(c, ',')) {
			continue;
		}
		if (c->lx.t == T_INT) {
			init_write(c, dst, size, &pos, t, c->lx.u);
			next(c);
		} else if (c->lx.t == T_HEXF) {
			init_write(c, dst, size, &pos, t, c->lx.u);
			next(c);
		} else if (c->lx.t == T_FLT) {
			uint64_t bits;

			if (t.bits == 64) {
				double d = c->lx.d;

				memcpy(&bits, &d, 8);
			} else {
				float f = (float)c->lx.d;
				uint32_t u;

				memcpy(&u, &f, 4);
				bits = u;
			}
			init_write(c, dst, size, &pos, t, bits);
			next(c);
		} else if (accept(c, '-')) {
			if (c->lx.t == T_INT)
				init_write(c, dst, size, &pos, t, -c->lx.u);
			else
				fail(c, "negative float initializers");
			next(c);
		} else {
			fail(c, "unsupported initializer element (addresses of other variables?)");
		}
	} while (depth);
}

/* a module-scope .global/.const declaration (cursor after the space keyword): layout + initializer */
static void module_global(struct ctx *c, enum space sp)
{
	char name[256];
	uint32_t size, al, off;
	bool unsized;
	struct ptype t;

	parse_var(c, name, sizeof(name), &size, &al, &unsized, &t);
	if (unsized)
		fail(c, "%s: unsized module variable", name);
	if (al < 16)
		al = 16;
	off = align_up(c->globals_size, al);
	if (c->nglobals == c->maxglobals) {
		c->maxglobals = c->maxglobals ? c->maxglobals * 2 : 64;
		c->globals = realloc(c->globals, c->maxglobals * sizeof(*c->globals));
		if (!c->globals)
			fail(c, "out of memory");
	}
	snprintf(c->globals[c->nglobals].name, sizeof(c->globals[0].name), "%s", name);
	c->globals[c->nglobals].offset = off;
	c->globals[c->nglobals].size = size;
	c->nglobals++;
	c->globals_size = off + size;
	c->globals_image = realloc(c->globals_image, c->globals_size);
	if (!c->globals_image)
		fail(c, "out of memory");
	memset(c->globals_image + off, 0, size);
	add_sym_to(c, c->modsyms, name, sp, off, size);
	if (accept(c, '='))
		parse_init(c, c->globals_image + off, size, t);
	while (!is_p(c, ';') && c->lx.t != T_EOF)
		next(c);
	expect(c, ';');
}

/*
 * The module scope: functions (catalogued), module variables (laid out), the entry (its
 * parameters, .maxntid and body position). Every body is prescanned for .shared.
 */
static void scan_module(struct ctx *c, const char *entry, const char **body, int *body_line, bool *found)
{
	*found = false;
	next(c);
	while (c->lx.t != T_EOF) {
		if (c->lx.t != T_ID) {
			next(c);
			continue;
		}
		if (is_id(c, ".entry")) {
			bool want;

			next(c);
			want = entry && !strcmp(c->lx.s, entry);
			expect_id(c);
			if (want) {
				*found = true;
				parse_params(c);
				/* performance directives until the body */
				while (!is_p(c, '{') && c->lx.t != T_EOF) {
					if (is_id(c, ".maxntid") || is_id(c, ".reqntid")) {
						uint64_t prod = 1;

						next(c);
						do {
							prod *= expect_int(c);
						} while (accept(c, ','));
						c->kb->max_threads = prod > 1024 ? 1024 : (uint32_t)prod;
						continue;
					}
					next(c);
				}
				*body = c->lx.p - 1;
				*body_line = c->lx.line;
				prescan_shared(c);
			} else {
				while (!is_p(c, '{') && !is_p(c, ';') && c->lx.t != T_EOF)
					next(c);
				if (is_p(c, '{'))
					prescan_shared(c);
			}
			continue;
		}
		if (is_id(c, ".func")) {
			struct pfunc *f;
			struct hash_entry *e;
			const char *hdr = c->lx.p;
			int line = c->lx.line, depth = 0;
			char name[256];

			next(c);
			if (accept(c, '(')) {		/* return values */
				depth = 1;
				while (depth && c->lx.t != T_EOF) {
					if (is_p(c, '('))
						depth++;
					else if (is_p(c, ')'))
						depth--;
					next(c);
				}
			}
			snprintf(name, sizeof(name), "%s", expect_id(c));
			while (!is_p(c, '{') && !is_p(c, ';') && c->lx.t != T_EOF)
				next(c);
			e = _mesa_hash_table_search(c->funcs, name);
			f = e ? e->data : NULL;
			if (is_p(c, '{')) {	/* a definition replaces a declaration */
				if (!f) {
					f = ralloc(c->mem, struct pfunc);
					_mesa_hash_table_insert(c->funcs, ralloc_strdup(c->mem, name), f);
				}
				f->hdr = hdr;
				f->line = line;
				f->has_body = true;
				prescan_shared(c);
			} else if (!f) {
				f = ralloc(c->mem, struct pfunc);
				f->hdr = hdr;
				f->line = line;
				f->has_body = false;
				_mesa_hash_table_insert(c->funcs, ralloc_strdup(c->mem, name), f);
			}
			continue;
		}
		if (is_id(c, ".shared")) {
			char name[256];
			uint32_t size, al;
			bool unsized;

			next(c);
			parse_var(c, name, sizeof(name), &size, &al, &unsized, NULL);
			if (unsized) {	/* .extern .shared x[]: dynamic shared memory, after the static part */
				c->dyn_shared = true;
				add_sym_to(c, c->modsyms, name, SP_SHARED, ~0u, 0);
			} else {
				c->shared_size = align_up(c->shared_size, al);
				add_sym_to(c, c->modsyms, name, SP_SHARED, c->shared_size, size);
				c->shared_size += size;
			}
			while (!is_p(c, ';') && c->lx.t != T_EOF)
				next(c);
			continue;
		}
		if (is_id(c, ".global") || is_id(c, ".const")) {
			enum space sp = is_id(c, ".global") ? SP_GLOBAL : SP_CONST;

			next(c);
			module_global(c, sp);
			continue;
		}
		if (is_id(c, ".file")) {
			int line = c->lx.line;

			while (c->lx.t != T_EOF && c->lx.line == line)
				next(c);
			continue;
		}
		if (is_id(c, ".section")) {
			while (!is_p(c, '{') && c->lx.t != T_EOF)
				next(c);
			skip_braces(c);
			continue;
		}
		next(c);
	}
	if (c->dyn_shared) {	/* extern shared arrays start after the static ones, 16-byte aligned */
		c->shared_size = align_up(c->shared_size, 16);
		hash_table_foreach(c->modsyms, e) {
			struct psym *s = e->data;

			if (s->space == SP_SHARED && s->offset == ~0u)
				s->offset = c->shared_size;
		}
	}
}

static struct ctx *ctx_new(const char *src, char *err, size_t errlen)
{
	struct ctx *c = calloc(1, sizeof(*c));

	if (!c)
		return NULL;
	c->mem = ralloc_context(NULL);
	c->err = err;
	c->errlen = errlen;
	c->lx.src = c->lx.p = src;
	c->lx.line = 1;
	c->modsyms = new_table(c);
	c->funcs = new_table(c);
	c->frames[0].regs = new_table(c);
	c->frames[0].labels = new_table(c);
	c->frames[0].syms = new_table(c);
	return c;
}

static void ctx_free(struct ctx *c)
{
	free(c->globals);
	free(c->globals_image);
	ralloc_free(c->mem);
	free(c);
}

int nvc_ptx_globals(const char *ptx, struct nvc_global **globals, uint32_t *count, void **image, uint32_t *size,
		    char *err, size_t errlen)
{
	struct ctx *c = ctx_new(ptx, err, errlen);
	struct nvc_kernel_bin kb = { 0 };
	const char *body;
	int line;
	bool found;

	if (!c)
		return -1;
	c->kb = &kb;
	if (setjmp(c->fail)) {
		ctx_free(c);
		return -1;
	}
	scan_module(c, NULL, &body, &line, &found);
	*globals = c->globals;
	*count = c->nglobals;
	*image = c->globals_image;
	*size = c->globals_size;
	c->globals = NULL;
	c->globals_image = NULL;
	ctx_free(c);
	return 0;
}

/* .entry name ( .param ... ) — the parameter list, laid out for cuLaunchKernel */
static void parse_params(struct ctx *c)
{
	uint32_t off = 0;

	if (!accept(c, '('))
		return;
	if (accept(c, ')'))
		return;
	do {
		char name[256];
		uint32_t size, al;
		bool unsized;

		if (!is_id(c, ".param"))
			fail(c, "expected .param");
		next(c);
		parse_var(c, name, sizeof(name), &size, &al, &unsized, NULL);
		if (c->kb->nparams == NVC_MAX_PARAMS)
			fail(c, "too many parameters");
		off = align_up(off, al);
		c->kb->params[c->kb->nparams].offset = off;
		c->kb->params[c->kb->nparams].size = size;
		c->kb->nparams++;
		add_sym(c, name, SP_PARAM, off, size);
		off += size;
	} while (accept(c, ','));
	expect(c, ')');
	c->kb->param_bytes = off;
}

nir_shader *nvc_ptx_build(const char *src, const char *entry, const nir_shader_compiler_options *opts,
			  struct nvc_kernel_bin *kb, uint64_t globals_va, char *err, size_t errlen)
{
	struct ctx *c = ctx_new(src, err, errlen);
	nir_shader *nir = NULL;
	nir_function *f;
	const char *body = NULL;
	int body_line = 0;
	bool found;

	if (!c)
		return NULL;
	c->kb = kb;
	c->globals_va = globals_va;
	if (setjmp(c->fail)) {
		if (nir)
			ralloc_free(nir);
		nir = NULL;
		goto out;
	}
	scan_module(c, entry, &body, &body_line, &found);
	if (!found) {
		snprintf(err, errlen, "no .entry %s in the PTX", entry);
		goto out;
	}

	nir = nir_shader_create(NULL, MESA_SHADER_KERNEL, opts);
	nir->info.name = ralloc_strdup(nir, entry);
	f = nir_function_create(nir, entry);
	f->is_entrypoint = true;
	c->impl = nir_function_impl_create(f);
	c->impl->structured = false;
	c->nir = nir;
	c->b = nir_builder_at(nir_after_block(nir_start_block(c->impl)));
	c->frames[0].ret = c->impl->end_block;
	c->lx.p = body;
	c->lx.line = body_line;
	next(c);
	parse_body(c);
	nir->info.shared_size = c->shared_size;
	nir->scratch_size = c->local_max;
	nir->info.workgroup_size_variable = true;
out:
	ctx_free(c);
	return nir;
}
