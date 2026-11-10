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
	const char *tok;	/* start of the current token */
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
	uint32_t regbase;	/* memory mode: this frame's register area in scratch */
	nir_block *ret;		/* where ret goes (the impl's end block for the entry) */
	struct pfunc *fn;	/* whose body this frame emits */
	bool boundary;		/* an outlined body or the entry: callers below are not part of it */
	size_t call_pos;	/* position in fn's body of the call being inlined from this frame */
};

/* a .func of the module */
struct pfunc {
	const char *hdr;	/* just after ".func" */
	int line;
	bool has_body;
	const char *body_end;
	uint32_t instances;	/* inlined copies emitted (NVCUDA_CALLSTATS) */
	const char *name;
	/* static call graph (prescan) and the outlining decision (decide_outlining) */
	const char **calls;
	int ncalls, maxcalls;
	uint32_t inst;		/* how many copies full inlining would make */
	bool reachable, visiting, visited, outlined;
	/*
	 * Outlined: the body is emitted once, at the first call. Parameters and return values live
	 * at fixed scratch offsets; a call copies its arguments in, sets retid to its site number
	 * and jumps to entry; ret jumps to the dispatch, which jumps back to the site.
	 */
	bool emitted;
	struct psym **params, **rets;
	int nparams, nrets;
	nir_block *entry_blk, *ret_blk;
	nir_variable *retid;
	nir_block **sites;
	int nsites, maxsites;
	/* register liveness (compute_liveness), for caller-save at outlined calls */
	bool liv_done;
	struct hash_table *regidx;	/* register name -> index */
	int nregs;
	size_t *callpos;		/* its call statements, and the registers live after each */
	uint64_t *call_live;
	uint64_t *crossblock;		/* registers live into some block */
	uint32_t *slot_off;		/* memory mode: scratch slot of each cross-block register (by index) */
	uint8_t *regsize;		/* bytes per register index (0: not declared) */
	uint32_t maxsizes;
	uint32_t regs_bytes;		/* size of the register area a frame of this function needs */
	int ncallpos;
	uint32_t save_bytes, save_off;	/* this function's save area (one per function) */
	bool save_alloc;
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
	struct pfunc entry_fn;		/* the entry's call list */
	struct pfunc *cur_fn;		/* whose body prescan_shared is scanning */
	struct pfunc **outlined;	/* for the dispatch blocks at the end */
	int noutlined, maxoutlined;
	bool static_mode;		/* allocations are permanent (an outlined body's own) */
	int static_depth;		/* the outlined body's frame; nested inlined frames use the stack */
	size_t stmt_pos;		/* position of the statement being emitted in its body */
	bool regmem;			/* registers crossing blocks live in scratch (big kernels) */
	uint32_t cache_gen;		/* bumped at every block switch */
	struct preg **dirty;		/* registers written in the current block */
	int ndirty, maxdirty;
	uint64_t est_bytes;		/* PTX text the kernel inlines (decide_outlining) */
	struct nvc_ptx_opts po;
	uint32_t floor;			/* scratch below this is permanent */
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
static uint32_t alloc_local(struct ctx *c, uint32_t size, uint32_t align);
static void compute_liveness(struct ctx *c, struct pfunc *f);
struct pfunc;
struct psym;
static void call_outlined(struct ctx *c, struct pfunc *f, struct psym **bind, int nrets, int nargs);

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
	l->tok = p;
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
	nir_variable *var;	/* SSA mode: the register is a variable (vars_to_ssa) */
	unsigned bits;		/* 1 for predicates */
	/* memory mode (regmem): registers that cross blocks live in scratch; within a block the
	 * value is cached in SSA and dirty values are stored at the block's jump */
	bool in_mem;
	uint32_t off;
	nir_def *cached;
	uint32_t cache_gen;
	bool dirty;
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
	r = rzalloc(c->mem, struct preg);
	r->bits = bits;
	if (c->regmem) {
		struct pfunc *f = frame(c)->fn;
		struct hash_entry *e = f && f->regidx ? _mesa_hash_table_search(f->regidx, name) : NULL;

		if (e) {
			uint32_t idx = (uint32_t)(uintptr_t)e->data;

			if (f->crossblock[idx / 64] & (1ull << (idx % 64)) && f->slot_off[idx] != ~0u) {
				r->in_mem = true;
				r->off = frame(c)->regbase + f->slot_off[idx];
			}
		}
		/* registers used within single blocks only: SSA through the block cache, never stored */
		r->var = NULL;
		r->cache_gen = ~0u;
	} else {
		r->var = nir_local_variable_create(c->impl, gt, name);
	}
	_mesa_hash_table_insert(frame(c)->regs, ralloc_strdup(c->mem, name), r);
}

/* memory mode: the register's current value in this block */
static nir_def *reg_get(struct ctx *c, struct preg *r)
{
	nir_builder *b = &c->b;
	unsigned bits = r->bits == 1 ? 8 : r->bits;
	nir_def *v;

	if (r->cache_gen == c->cache_gen && r->cached)
		return r->cached;
	if (!r->in_mem)	/* read before any write in this block: undefined in PTX too */
		v = nir_imm_intN_t(b, 0, bits);
	else
		v = nir_load_scratch(b, 1, bits, nir_imm_int(b, r->off), .align_mul = bits / 8);
	if (r->bits == 1)
		v = nir_ine_imm(b, v, 0);
	r->cached = v;
	r->cache_gen = c->cache_gen;
	r->dirty = false;
	return v;
}

static void reg_set(struct ctx *c, struct preg *r, nir_def *v)
{
	if (r->cache_gen != c->cache_gen || !r->dirty) {
		if (c->ndirty == c->maxdirty) {
			c->maxdirty = c->maxdirty ? c->maxdirty * 2 : 256;
			c->dirty = reralloc(c->mem, c->dirty, struct preg *, c->maxdirty);
		}
		c->dirty[c->ndirty++] = r;
	}
	r->cached = v;
	r->cache_gen = c->cache_gen;
	r->dirty = true;
}

/* store the block's dirty registers (before its jump) and start a new block cache */
static void flush_regs(struct ctx *c)
{
	nir_builder *b = &c->b;
	int i;

	for (i = 0; i < c->ndirty; i++) {
		struct preg *r = c->dirty[i];

		if (r->cache_gen == c->cache_gen && r->dirty && r->in_mem) {
			nir_def *v = r->cached;

			if (r->bits == 1)
				v = nir_b2i8(b, v);
			nir_store_scratch(b, v, nir_imm_int(b, r->off), .write_mask = 1, .align_mul = v->bit_size / 8);
		}
		r->dirty = false;
	}
	c->ndirty = 0;
}

static void new_block_cache(struct ctx *c)
{
	c->cache_gen++;
	c->ndirty = 0;
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
	if (c->static_mode && c->depth == c->static_depth)
		c->floor = c->local_top;
	return off;
}

static uint32_t pop_local(struct ctx *c, uint32_t saved)
{
	return saved > c->floor ? saved : c->floor;
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

static void new_block_cache(struct ctx *c);
static void flush_regs(struct ctx *c);

static void switch_to(struct ctx *c, nir_block *blk)
{
	c->b.cursor = nir_after_block(blk);
	c->terminated = false;
	new_block_cache(c);
}

static void jump(struct ctx *c, nir_block *target)
{
	flush_regs(c);
	nir_goto(&c->b, target);
}

static void jump_if(struct ctx *c, nir_block *target, nir_def *cond, nir_block *fall)
{
	flush_regs(c);
	nir_goto_if(&c->b, target, cond, fall);
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
		if (accept(c, ',')) {	/* tex: [texobj, {coords}] */
			struct opnd co;

			parse_operand(c, &co);
			if (co.kind == O_VEC) {
				o->vec = co.vec;
				o->nvec = co.nvec;
			} else {
				o->vec = ralloc(c->mem, struct opnd);
				*o->vec = co;
				o->nvec = 1;
			}
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

		v = c->regmem ? reg_get(c, r) : nir_load_var(b, r->var);
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
	if (c->regmem)
		reg_set(c, r, v);
	else
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
		base = c->regmem ? reg_get(c, find_reg(c, m->name)) : nir_load_var(b, find_reg(c, m->name)->var);
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
 * A generic address: shared window, local window, or global. Loads and stores are branch-free:
 * all three spaces are accessed, the inactive ones at their dummy slot (NVC_DUMMY_*), and the
 * result selected. That keeps the block count flat: the shading kernels of a path tracer inline
 * thousands of these, and the compiler's memory use grows with blocks x values. Atomics branch.
 */
static nir_def *access_generic(struct ctx *c, nir_def *addr, const struct mop *m)
{
	nir_builder *b = &c->b;
	nir_def *hi = nir_ushr_imm(b, addr, 24), *off = nir_iand_imm(b, nir_u2u32(b, addr), WINDOW_SIZE - 1);
	nir_def *is_sh = nir_ieq_imm(b, hi, SHARED_WINDOW >> 24), *is_lo = nir_ieq_imm(b, hi, LOCAL_WINDOW >> 24);
	nir_def *sh_off, *lo_off, *g_addr, *vs, *vl, *vg;

	if (m->kind == M_ATOMIC || m->kind == M_CAS) {
		nir_block *sh = new_block(c), *gl = new_block(c), *join = new_block(c);
		nir_variable *res = nir_local_variable_create(c->impl, glsl_uintN_t_type(m->bits), "generic_atomic");

		jump_if(c, sh, is_sh, gl);
		b->cursor = nir_after_block(sh);
		nir_store_var(b, res, access_space(c, SP_SHARED, off, m), 1);
		jump(c, join);
		b->cursor = nir_after_block(gl);
		nir_store_var(b, res, access_space(c, SP_GLOBAL, addr, m), 1);
		jump(c, join);
		switch_to(c, join);
		return nir_load_var(b, res);
	}
	sh_off = nir_bcsel(b, is_sh, off, nir_imm_int(b, 0));
	lo_off = nir_bcsel(b, is_lo, off, nir_imm_int(b, 0));
	g_addr = nir_bcsel(b, nir_ior(b, is_sh, is_lo), nir_imm_int64(b, NVC_DUMMY_VA), addr);
	vs = access_space(c, SP_SHARED, sh_off, m);
	vl = access_space(c, SP_LOCAL, lo_off, m);
	vg = access_space(c, SP_GLOBAL, g_addr, m);
	if (m->kind == M_STORE)
		return NULL;
	return nir_bcsel(b, is_sh, vs, nir_bcsel(b, is_lo, vl, vg));
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

	/* --- textures: tex.geom.v4.dtype.ctype {d0..d3}, [texobj, {coords}] --- */
	if (!strcmp(op, "tex")) {
		struct opnd *mem = &o[1];
		struct ptype dt = in->types[0], ct = in->ntypes > 1 ? in->types[1] : (struct ptype){ TC_F, 32 };
		nir_tex_instr *tex;
		nir_def *handle, *coord, *comps[4], *lod;
		enum glsl_sampler_dim dim;
		unsigned int ncoord, i;
		bool fetch = ct.cls != TC_F;

		if (mem->kind != O_MEM || !mem->mem_reg || !mem->vec || o[0].kind != O_VEC || o[0].nvec != 4)
			fail(c, "tex needs [texobj, {coords}] and a 4-vector destination");
		if (has_mod(in, "level") || has_mod(in, "grad") || has_mod(in, "a1d") || has_mod(in, "a2d") ||
		    has_mod(in, "cube") || has_mod(in, "acube"))
			fail(c, "tex: explicit LOD, gradients and array/cube textures are not supported yet");
		if (has_mod(in, "1d")) {
			dim = GLSL_SAMPLER_DIM_BUF;	/* CUDA 1-D textures are linear memory: integer fetch */
			ncoord = 1;
			fetch = true;
		} else if (has_mod(in, "2d")) {
			dim = GLSL_SAMPLER_DIM_2D;
			ncoord = 2;
		} else if (has_mod(in, "3d")) {
			dim = GLSL_SAMPLER_DIM_3D;
			ncoord = 3;
		} else {
			fail(c, "tex geometry");
		}
		if (mem->nvec < (int)ncoord)
			fail(c, "tex: %u coordinates needed", ncoord);
		if (getenv("NVCUDA_NO_TEX")) {	/* diagnosis: a texture read as zero */
			for (i = 0; i < 4; i++)
				wr(c, &o[0].vec[i], nir_imm_int(b, 0), dt);
			return;
		}
		handle = nir_u2u32(b, c->regmem ? reg_get(c, find_reg(c, mem->name)) : nir_load_var(b, find_reg(c, mem->name)->var));
		for (i = 0; i < ncoord; i++) {
			comps[i] = rd(c, &mem->vec[i], ct);
			if (fetch && ct.cls == TC_F)
				comps[i] = nir_f2i32(b, nir_ffloor(b, comps[i]));
		}
		coord = nir_vec(b, comps, ncoord);
		tex = nir_tex_instr_create(c->nir, 4);
		tex->op = fetch ? nir_texop_txf : nir_texop_txl;
		tex->sampler_dim = dim;
		tex->dest_type = dt.cls == TC_F ? nir_type_float32 : dt.cls == TC_S ? nir_type_int32 : nir_type_uint32;
		tex->coord_components = ncoord;
		tex->is_array = false;
		lod = fetch ? nir_imm_int(b, 0) : nir_imm_float(b, 0.0f);
		tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_texture_handle, handle);
		tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_sampler_handle, handle);
		tex->src[2] = nir_tex_src_for_ssa(nir_tex_src_coord, coord);
		tex->src[3] = nir_tex_src_for_ssa(nir_tex_src_lod, lod);
		nir_def_init(&tex->instr, &tex->def, 4, 32);
		nir_builder_instr_insert(b, &tex->instr);
		for (i = 0; i < 4; i++)
			wr(c, &o[0].vec[i], nir_channel(b, &tex->def, i), dt);
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

	c->stmt_pos = frame(c)->fn && frame(c)->fn->hdr ? (size_t)(c->lx.tok - frame(c)->fn->hdr) : 0;
	if (accept(c, '@')) {
		parse_operand(c, &pred);
		has_pred = true;
	}
	if (c->lx.t != T_ID)
		fail(c, "expected an instruction");
	if (!strncmp(c->lx.s, "tld4.", 5) || !strncmp(c->lx.s, "suld.", 5) || !strncmp(c->lx.s, "sust.", 5) ||
	    !strncmp(c->lx.s, "txq.", 4) || !strncmp(c->lx.s, "suq.", 4))
		fail(c, "%s is not supported yet (surfaces, gathers, queries)", c->lx.s);
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
			jump_if(c, then, rd(c, &pred, (struct ptype){ TC_PRED, 1 }), join);
			switch_to(c, then);
		}
		inline_call(c, rets, nrets, fname, args, nargs);
		if (has_pred) {
			jump(c, join);
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
			jump_if(c, target, rd(c, &pred, (struct ptype){ TC_PRED, 1 }), fall);
			switch_to(c, fall);
		} else {
			jump(c, target);
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

			jump_if(c, target, rd(c, &pred, (struct ptype){ TC_PRED, 1 }), fall);
			switch_to(c, fall);
		} else {
			jump(c, target);
			c->terminated = true;
		}
		return;
	}
	if (has_pred) {
		then = new_block(c);
		join = new_block(c);
		jump_if(c, then, rd(c, &pred, (struct ptype){ TC_PRED, 1 }), join);
		switch_to(c, then);
	}
	emit_insn(c, &in);
	if (has_pred) {
		jump(c, join);
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
			c->local_top = pop_local(c, scope_top[--scope]);	/* the scope's .param/.local variables die */
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
			const char *save_p = c->lx.p, *save_tok = c->lx.tok;
			int save_line = c->lx.line;
			char name[256];

			snprintf(name, sizeof(name), "%s", c->lx.s);
			next(c);
			if (accept(c, ':')) {
				nir_block *blk = label_block(c, name);

				if (!c->terminated)
					jump(c, blk);
				switch_to(c, blk);
				continue;
			}
			/* not a label: rewind to the opcode */
			c->lx.p = save_p;
			c->lx.tok = save_tok;
			c->lx.line = save_line;
			c->lx.t = T_ID;
			snprintf(c->lx.s, sizeof(c->lx.s), "%s", name);
		}
		emit_statement(c);
	}
	if (!c->terminated)
		jump(c, frame(c)->ret);
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
	if (f->outlined) {
		call_outlined(c, f, bind, nrets, nargs);
		return;
	}
	ret = new_block(c);
	f->instances++;
	frame(c)->call_pos = c->stmt_pos;
	if (c->regmem)
		compute_liveness(c, f);
	c->depth++;
	fr = frame(c);
	fr->regs = new_table(c);
	fr->labels = new_table(c);
	fr->syms = new_table(c);
	fr->ret = ret;
	fr->fn = f;
	fr->boundary = false;
	fr->regbase = c->regmem && f->regs_bytes ? alloc_local(c, f->regs_bytes, 8) : 0;
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
	c->local_top = pop_local(c, saved_top);
	c->lx = saved;
	switch_to(c, ret);
}

/* ---- outlined functions ---- */

/* scratch -> scratch copy of a .param value (32-bit words, then bytes) */
static void copy_scratch(struct ctx *c, uint32_t dst, uint32_t src, uint32_t size)
{
	nir_builder *b = &c->b;
	uint32_t off = 0;

	for (; off + 4 <= size; off += 4) {
		nir_def *v = nir_load_scratch(b, 1, 32, nir_imm_int(b, src + off), .align_mul = 4);

		nir_store_scratch(b, v, nir_imm_int(b, dst + off), .write_mask = 1, .align_mul = 4);
	}
	for (; off < size; off++) {
		nir_def *v = nir_load_scratch(b, 1, 8, nir_imm_int(b, src + off), .align_mul = 1);

		nir_store_scratch(b, v, nir_imm_int(b, dst + off), .write_mask = 1, .align_mul = 1);
	}
}

/* Parse the header once: fixed scratch slots for the return values and parameters. */
static void outline_prepare(struct ctx *c, struct pfunc *f)
{
	struct lex saved = c->lx;
	bool saved_static = c->static_mode;
	char name[256];
	uint32_t size, al;
	bool unsized;

	if (f->params)
		return;
	c->static_mode = true;
	{
		int sd = c->static_depth;

		c->static_depth = c->depth;
		c->local_top = c->local_top > c->floor ? c->local_top : c->floor;
		(void)sd;
	}
	c->lx.p = f->hdr;
	c->lx.line = f->line;
	next(c);
	f->rets = ralloc_array(c->mem, struct psym *, 8);
	f->params = ralloc_array(c->mem, struct psym *, 40);
	if (accept(c, '(')) {
		do {
			if (!is_id(c, ".param") || f->nrets == 8)
				fail(c, "%s: return values", f->name);
			next(c);
			parse_var(c, name, sizeof(name), &size, &al, &unsized, NULL);
			f->rets[f->nrets++] = add_sym_to(c, c->modsyms, "", SP_LOCAL, alloc_local(c, size, al), size);
		} while (accept(c, ','));
		expect(c, ')');
	}
	expect_id(c);
	if (accept(c, '(') && !accept(c, ')')) {
		do {
			if (!is_id(c, ".param") || f->nparams == 40)
				fail(c, "%s: parameters", f->name);
			next(c);
			parse_var(c, name, sizeof(name), &size, &al, &unsized, NULL);
			f->params[f->nparams++] = add_sym_to(c, c->modsyms, "", SP_LOCAL, alloc_local(c, size, al), size);
		} while (accept(c, ','));
		expect(c, ')');
	}
	f->retid = nir_local_variable_create(c->impl, glsl_uint_type(), "retid");
	f->entry_blk = new_block(c);
	f->ret_blk = new_block(c);
	c->floor = c->local_top;	/* the slots stay */
	c->static_mode = saved_static;
	c->lx = saved;
	if (c->noutlined == c->maxoutlined) {
		c->maxoutlined = c->maxoutlined ? c->maxoutlined * 2 : 32;
		c->outlined = reralloc(c->mem, c->outlined, struct pfunc *, c->maxoutlined);
	}
	c->outlined[c->noutlined++] = f;
}

/* Emit the body once, in its own frame, with its .param names bound to the fixed slots. */
static void outline_emit(struct ctx *c, struct pfunc *f)
{
	struct lex saved = c->lx;
	nir_cursor saved_cursor = c->b.cursor;
	bool saved_term = c->terminated, saved_static = c->static_mode;
	uint32_t saved_top = c->local_top;
	int saved_sdepth = c->static_depth;
	struct frame *fr;
	char name[256];
	uint32_t size, al;
	bool unsized;
	int i;

	if (c->depth == MAX_DEPTH)
		fail(c, "calls nested deeper than %d (recursion?)", MAX_DEPTH);
	f->emitted = true;
	c->static_mode = true;
	c->static_depth = c->depth + 1;	/* the body's frame; its inlined callees stack above it */
	c->local_top = c->local_top > c->floor ? c->local_top : c->floor;
	frame(c)->call_pos = c->stmt_pos;
	if (c->regmem)
		compute_liveness(c, f);
	c->depth++;
	fr = frame(c);
	fr->regs = new_table(c);
	fr->labels = new_table(c);
	fr->syms = new_table(c);
	fr->ret = f->ret_blk;
	fr->fn = f;
	fr->boundary = true;
	fr->regbase = c->regmem && f->regs_bytes ? alloc_local(c, f->regs_bytes, 8) : 0;
	c->lx.p = f->hdr;
	c->lx.line = f->line;
	next(c);
	i = 0;
	if (accept(c, '(')) {
		do {
			next(c);	/* .param */
			parse_var(c, name, sizeof(name), &size, &al, &unsized, NULL);
			_mesa_hash_table_insert(fr->syms, ralloc_strdup(c->mem, name), f->rets[i++]);
		} while (accept(c, ','));
		expect(c, ')');
	}
	expect_id(c);
	i = 0;
	if (accept(c, '(') && !accept(c, ')')) {
		do {
			next(c);
			parse_var(c, name, sizeof(name), &size, &al, &unsized, NULL);
			_mesa_hash_table_insert(fr->syms, ralloc_strdup(c->mem, name), f->params[i++]);
		} while (accept(c, ','));
		expect(c, ')');
	}
	while (!is_p(c, '{') && c->lx.t != T_EOF)
		next(c);
	switch_to(c, f->entry_blk);
	parse_body(c);
	c->depth--;
	/* everything the body and its inlined callees used stays reserved (runs at any stack depth) */
	c->floor = c->local_max;
	c->static_mode = saved_static;
	c->static_depth = saved_sdepth;
	c->local_top = pop_local(c, saved_top);
	c->lx = saved;
	c->b.cursor = saved_cursor;
	c->terminated = saved_term;
}

/*
 * Register liveness of a function body: statements are classified (registers read and written,
 * labels, branches, calls, returns), split into basic blocks, and a backward dataflow gives the
 * registers live after each call statement. Registers are the '%' tokens; an instruction's
 * destinations are the tokens before its first top-level comma (none for st, red, bar, ...).
 */
enum skind { S_OTHER, S_LABEL, S_BRA, S_RET, S_CALL };

struct pstmt {
	uint8_t kind;
	bool pred;
	bool split;		/* the emitter puts a block boundary before and after it (predicated non-branch, atomics) */
	size_t pos;
	int label;		/* S_LABEL: its index; S_BRA: the target's */
	uint32_t *rd, *wr;
	uint16_t nrd, nwr;
};

struct pblock {
	int first, last;	/* statements */
	int succ[2], nsucc;
};

static bool no_dest(const char *op)
{
	static const char *const ops[] = { "st", "red", "bar", "barrier", "bra", "brx", "call", "ret", "exit", "trap",
					   "membar", "fence", "prefetch", "prefetchu", "discard", "applypriority",
					   "setmaxnreg" };

	for (unsigned i = 0; i < sizeof(ops) / sizeof(ops[0]); i++)
		if (!strcmp(op, ops[i]))
			return true;
	return false;
}

static uint32_t name_index(struct ctx *c, struct hash_table *t, const char *name)
{
	struct hash_entry *e = _mesa_hash_table_search(t, name);

	if (e)
		return (uint32_t)(uintptr_t)e->data;
	_mesa_hash_table_insert(t, ralloc_strdup(c->mem, name), (void *)(uintptr_t)t->entries);
	return t->entries - 1;
}

#define BITS_WORDS(n)	(((n) + 63) / 64)

/* register name -> size in bytes (from its .reg declaration), kept as the index table's twin */
static void reg_size_set(struct ctx *c, struct pfunc *f, const char *name, unsigned bytes)
{
	uint32_t idx = name_index(c, f->regidx, name);

	if (idx >= f->maxsizes) {
		uint32_t n = f->maxsizes ? f->maxsizes * 2 : 1024;

		while (n <= idx)
			n *= 2;
		f->regsize = reralloc(c->mem, f->regsize, uint8_t, n);
		memset(f->regsize + f->maxsizes, 0, n - f->maxsizes);
		f->maxsizes = n;
	}
	f->regsize[idx] = bytes;
}

static void compute_liveness(struct ctx *c, struct pfunc *f)
{
	struct lex saved = c->lx;
	struct hash_table *labels = new_table(c);
	struct pstmt *st = NULL;
	struct pblock *bl = NULL;
	int nst = 0, maxst = 0, nbl = 0, i, depth = 0, nregs, nw;
	uint64_t *use, *def, *in, *out, *live;
	int *label_block;

	if (f->liv_done)
		return;
	f->liv_done = true;
	f->regidx = new_table(c);
	c->lx.p = f->hdr;
	c->lx.line = f->line;
	next(c);
	while (!is_p(c, '{') && c->lx.t != T_EOF)
		next(c);
	/* 1. statements */
	do {
		uint32_t rd[64], wr[16];
		int nrd = 0, nwr = 0, brace = 0;
		struct pstmt *sp;
		char op[64] = "";
		bool dest_zone;

		if (c->lx.t == T_EOF)
			fail(c, "unterminated function");
		if (is_p(c, '{')) {
			depth++;
			next(c);
			continue;
		}
		if (is_p(c, '}')) {
			depth--;
			next(c);
			continue;
		}
		if (c->lx.t == T_ID && c->lx.s[0] == '.') {	/* a directive: no register reads */
			if (is_id(c, ".loc")) {
				int line = c->lx.line;

				while (c->lx.t != T_EOF && c->lx.line == line)
					next(c);
			} else if (is_id(c, ".reg")) {	/* register sizes, for the scratch slots */
				struct ptype t = { TC_B, 32 };

				next(c);
				if (c->lx.t == T_ID && c->lx.s[0] == '.')
					parse_type(c->lx.s + 1, &t);
				while (!is_p(c, ';') && c->lx.t != T_EOF) {
					if (c->lx.t == T_ID && c->lx.s[0] == '%') {
						char base[256];
						unsigned bytes = t.cls == TC_PRED ? 1 : (t.bits < 16 ? 16 : t.bits) / 8;

						snprintf(base, sizeof(base), "%s", c->lx.s);
						next(c);
						if (accept(c, '<')) {
							uint64_t n = expect_int(c), k;

							expect(c, '>');
							for (k = 0; k < n; k++) {
								char full[300];

								snprintf(full, sizeof(full), "%s%u", base, (unsigned)k);
								reg_size_set(c, f, full, bytes);
							}
						} else {
							reg_size_set(c, f, base, bytes);
						}
						continue;
					}
					next(c);
				}
				accept(c, ';');
			} else {
				while (!is_p(c, ';') && !is_p(c, '}') && c->lx.t != T_EOF)
					next(c);
				accept(c, ';');
			}
			continue;
		}
		if (nst == maxst) {
			maxst = maxst ? maxst * 2 : 256;
			st = reralloc(c->mem, st, struct pstmt, maxst);
		}
		sp = &st[nst];
		memset(sp, 0, sizeof(*sp));
		sp->pos = c->lx.tok - f->hdr;
		if (accept(c, '@')) {
			accept(c, '!');
			if (c->lx.t == T_ID)
				rd[nrd++] = name_index(c, f->regidx, c->lx.s);
			next(c);
			sp->pred = true;
		}
		if (c->lx.t != T_ID) {
			next(c);
			continue;
		}
		snprintf(op, sizeof(op), "%.63s", c->lx.s);
		next(c);
		if (accept(c, ':')) {
			sp->kind = S_LABEL;
			sp->label = name_index(c, labels, op);
			nst++;
			continue;
		}
		{
			char *dot = strchr(op, '.');

			if (dot)
				*dot = 0;
		}
		if (!strcmp(op, "bra")) {
			sp->kind = S_BRA;
			if (c->lx.t == T_ID)
				sp->label = name_index(c, labels, c->lx.s);
		} else if (!strcmp(op, "ret") || !strcmp(op, "exit")) {
			sp->kind = S_RET;
		} else if (!strcmp(op, "call")) {
			sp->kind = S_CALL;
		}
		sp->split = (sp->pred && sp->kind != S_BRA) || !strcmp(op, "atom") || !strcmp(op, "red");
		dest_zone = !no_dest(op);
		while (!is_p(c, ';') && c->lx.t != T_EOF) {
			if (is_p(c, '{') || is_p(c, '['))
				brace++;
			else if (is_p(c, '}') || is_p(c, ']'))
				brace--;
			else if (is_p(c, ',') && brace == 0)
				dest_zone = false;
			else if (c->lx.t == T_ID && c->lx.s[0] == '%') {
				uint32_t idx = name_index(c, f->regidx, c->lx.s);

				if (dest_zone && nwr < 16)
					wr[nwr++] = idx;
				else if (!dest_zone && nrd < 64)
					rd[nrd++] = idx;
			}
			next(c);
		}
		accept(c, ';');
		if (nrd) {
			sp->rd = ralloc_array(c->mem, uint32_t, nrd);
			memcpy(sp->rd, rd, nrd * 4);
			sp->nrd = nrd;
		}
		if (nwr) {
			sp->wr = ralloc_array(c->mem, uint32_t, nwr);
			memcpy(sp->wr, wr, nwr * 4);
			sp->nwr = nwr;
		}
		nst++;
	} while (depth);
	c->lx = saved;
	nregs = f->regidx->entries;
	nw = BITS_WORDS(nregs);

	/* 2. blocks: a new one at every label and after every branch or return */
	bl = ralloc_array(c->mem, struct pblock, nst + 1);
	label_block = ralloc_array(c->mem, int, labels->entries + 1);
	for (i = 0; i <= (int)labels->entries; i++)
		label_block[i] = -1;
	for (i = 0; i < nst; i++) {
		/* the emitter's block boundaries: labels, after branches/returns/calls, around split statements */
		bool start = i == 0 || st[i].kind == S_LABEL || st[i - 1].kind == S_BRA || st[i - 1].kind == S_RET ||
			     st[i - 1].kind == S_CALL || st[i - 1].split || st[i].split;

		if (start) {
			bl[nbl].first = i;
			bl[nbl].nsucc = 0;
			if (nbl)
				bl[nbl - 1].last = i - 1;
			nbl++;
		}
		if (st[i].kind == S_LABEL)
			label_block[st[i].label] = nbl - 1;
	}
	if (nbl)
		bl[nbl - 1].last = nst - 1;
	for (i = 0; i < nbl; i++) {
		struct pstmt *last = &st[bl[i].last];

		if (last->kind == S_RET)
			continue;
		if (last->kind == S_BRA) {
			if (label_block[last->label] >= 0)
				bl[i].succ[bl[i].nsucc++] = label_block[last->label];
			if (!last->pred)
				continue;
		}
		if (i + 1 < nbl)
			bl[i].succ[bl[i].nsucc++] = i + 1;
	}

	/* 3. use/def per block, then live-in/out to a fixpoint */
	use = rzalloc_array(c->mem, uint64_t, (size_t)nbl * nw);
	def = rzalloc_array(c->mem, uint64_t, (size_t)nbl * nw);
	in = rzalloc_array(c->mem, uint64_t, (size_t)nbl * nw);
	out = rzalloc_array(c->mem, uint64_t, (size_t)nbl * nw);
	for (i = 0; i < nbl; i++) {
		uint64_t *u = use + (size_t)i * nw, *d = def + (size_t)i * nw;
		int k, j;

		for (k = bl[i].first; k <= bl[i].last; k++) {
			for (j = 0; j < st[k].nrd; j++) {
				uint32_t r = st[k].rd[j];

				if (!(d[r / 64] & (1ull << (r % 64))))
					u[r / 64] |= 1ull << (r % 64);
			}
			for (j = 0; j < st[k].nwr; j++) {
				uint32_t r = st[k].wr[j];

				d[r / 64] |= 1ull << (r % 64);
			}
		}
	}
	for (bool changed = true; changed;) {
		changed = false;
		for (i = nbl - 1; i >= 0; i--) {
			uint64_t *o = out + (size_t)i * nw, *n = in + (size_t)i * nw;
			int k, w;

			for (k = 0; k < bl[i].nsucc; k++) {
				uint64_t *si = in + (size_t)bl[i].succ[k] * nw;

				for (w = 0; w < nw; w++)
					o[w] |= si[w];
			}
			for (w = 0; w < nw; w++) {
				uint64_t v = use[(size_t)i * nw + w] | (o[w] & ~def[(size_t)i * nw + w]);

				if (v != n[w]) {
					n[w] = v;
					changed = true;
				}
			}
		}
	}

	/* registers live into any block cross blocks (memory mode keeps them in scratch) */
	f->crossblock = rzalloc_array(c->mem, uint64_t, nw);
	for (i = 0; i < nbl; i++) {
		int w;

		for (w = 0; w < nw; w++)
			f->crossblock[w] |= in[(size_t)i * nw + w];
	}

	/*
	 * Scratch slots for them: a register's live range is the block range [lo, hi] over the
	 * blocks where it is live-in, used or defined; registers with disjoint ranges share a
	 * slot (greedy by lo, per size class).
	 */
	{
		int *lo = ralloc_array(c->mem, int, nregs + 1), *hi = ralloc_array(c->mem, int, nregs + 1);
		uint32_t *order = ralloc_array(c->mem, uint32_t, nregs + 1);
		struct { uint32_t off; int busy_until; } *slots = NULL;
		int nslots = 0, maxslots = 0, r, k, j, no = 0;
		uint32_t top = 0;

		f->slot_off = ralloc_array(c->mem, uint32_t, nregs + 1);
		for (r = 0; r < nregs; r++) {
			lo[r] = nbl;
			hi[r] = -1;
			f->slot_off[r] = ~0u;
		}
		for (i = 0; i < nbl; i++) {
			uint64_t *n = in + (size_t)i * nw, *u = use + (size_t)i * nw, *d = def + (size_t)i * nw;
			int w;

			for (w = 0; w < nw; w++) {
				uint64_t bits = n[w] | u[w] | d[w];

				while (bits) {
					int bit = __builtin_ctzll(bits);

					r = w * 64 + bit;
					bits &= bits - 1;
					if (r >= nregs)
						break;
					if (i < lo[r])
						lo[r] = i;
					if (i > hi[r])
						hi[r] = i;
				}
			}
		}
		/* sort cross-block registers by lo (counting sort over blocks) */
		{
			int *cnt = rzalloc_array(c->mem, int, nbl + 2);

			for (r = 0; r < nregs; r++)
				if ((f->crossblock[r / 64] & (1ull << (r % 64))) && hi[r] >= 0)
					cnt[lo[r] + 1]++;
			for (i = 0; i < nbl; i++)
				cnt[i + 1] += cnt[i];
			for (r = 0; r < nregs; r++)
				if ((f->crossblock[r / 64] & (1ull << (r % 64))) && hi[r] >= 0)
					order[cnt[lo[r]]++] = r, no++;
			for (k = 0; k < no; k++) {
				unsigned bytes;

				r = order[k];
				bytes = r < (int)f->maxsizes && f->regsize[r] ? f->regsize[r] : 8;
				for (j = 0; j < nslots; j++)
					if (slots[j].busy_until < lo[r] && (slots[j].off % bytes) == 0 &&
					    (j + 1 == nslots || slots[j + 1].off - slots[j].off >= bytes))
						break;
				if (j == nslots) {
					if (nslots == maxslots) {
						maxslots = maxslots ? maxslots * 2 : 256;
						slots = reralloc(c->mem, slots, __typeof__(*slots), maxslots);
					}
					top = (top + bytes - 1) / bytes * bytes;
					slots[nslots].off = top;
					top += bytes;
					nslots++;
				}
				slots[j].busy_until = hi[r];
				f->slot_off[r] = slots[j].off;
			}
		}
		f->regs_bytes = (top + 7) & ~7u;
		if (getenv("NVCUDA_REGSTATS"))
			fprintf(stderr, "regstats %u blocks %u regs %u cross %u slots %u bytes %s\n", nbl, nregs, no, nslots,
				f->regs_bytes, f->name ? f->name : "entry");
	}

	/* 4. live after each call: walk its block backwards from live-out */
	f->ncallpos = 0;
	for (i = 0; i < nst; i++)
		f->ncallpos += st[i].kind == S_CALL;
	f->callpos = ralloc_array(c->mem, size_t, f->ncallpos + 1);
	f->call_live = rzalloc_array(c->mem, uint64_t, (size_t)(f->ncallpos + 1) * nw);
	f->nregs = nregs;
	live = ralloc_array(c->mem, uint64_t, nw);
	{
		int ci = 0;

		for (i = 0; i < nbl; i++) {
			int k, j, w;

			memcpy(live, out + (size_t)i * nw, nw * 8);
			for (k = bl[i].last; k >= bl[i].first; k--) {
				if (st[k].kind == S_CALL) {	/* calls are numbered in text order: fill by position */
					f->callpos[ci] = st[k].pos;
					memcpy(f->call_live + (size_t)ci * nw, live, nw * 8);
					ci++;
				}
				for (j = 0; j < st[k].nwr; j++)
					live[st[k].wr[j] / 64] &= ~(1ull << (st[k].wr[j] % 64));
				for (j = 0; j < st[k].nrd; j++)
					live[st[k].rd[j] / 64] |= 1ull << (st[k].rd[j] % 64);
			}
			(void)w;
		}
		f->ncallpos = ci;
	}
}

/* is register @name live after the call statement at @pos in f's body? */
static bool reg_live_after(struct pfunc *f, const char *name, size_t pos)
{
	struct hash_entry *e = _mesa_hash_table_search(f->regidx, name);
	uint32_t r;
	int i;

	if (!e)
		return false;
	r = (uint32_t)(uintptr_t)e->data;
	for (i = 0; i < f->ncallpos; i++)
		if (f->callpos[i] == pos)
			return f->call_live[(size_t)i * BITS_WORDS(f->nregs) + r / 64] & (1ull << (r % 64));
	if (getenv("NVCUDA_DEBUG")) {
		fprintf(stderr, "nvcuda: liveness: no call at %zu in %s (%d calls:", pos, f->name ? f->name : "entry", f->ncallpos);
		for (i = 0; i < f->ncallpos && i < 8; i++)
			fprintf(stderr, " %zu", f->callpos[i]);
		fprintf(stderr, ")\n");
	}
	return true;	/* not a call position we know: keep it (conservative) */
}

struct saved_reg {
	nir_variable *var;
	uint32_t off;
	unsigned bits;
};

/*
 * Caller-save around an outlined call: the registers of every frame down to the nearest
 * boundary that are live after the call go to their function's save area (one area per
 * function: a function runs one call at a time, and no recursion). Keeps callers' values out
 * of the shared body, where the union over all sites would be live.
 */
static struct saved_reg *caller_save(struct ctx *c, int *count)
{
	nir_builder *b = &c->b;
	struct saved_reg *list = NULL;
	int n = 0, max = 0, k;

	for (k = c->depth; k >= 0; k--) {
		struct frame *fr = &c->frames[k];
		struct pfunc *f = fr->fn;
		size_t pos = k == c->depth ? c->stmt_pos : fr->call_pos;
		uint32_t off = 0;

		if (!f || !f->hdr)
			break;
		compute_liveness(c, f);
		hash_table_foreach(fr->regs, e) {
			struct preg *r = e->data;
			unsigned bytes = r->bits == 1 ? 1 : r->bits / 8;

			if (!reg_live_after(f, e->key, pos))
				continue;
			off = (off + bytes - 1) / bytes * bytes;
			if (n == max) {
				max = max ? max * 2 : 64;
				list = reralloc(c->mem, list, struct saved_reg, max);
			}
			list[n].var = r->var;
			list[n].off = off;
			list[n].bits = r->bits;
			n++;
			off += bytes;
		}
		if (!f->save_alloc) {	/* size the area over every call in the body, plus slack */
			int i;

			for (i = 0; i < f->ncallpos; i++) {
				uint32_t o = 0;

				hash_table_foreach(fr->regs, e2) {
					struct preg *r = e2->data;
					unsigned bytes = r->bits == 1 ? 1 : r->bits / 8;

					if (!reg_live_after(f, e2->key, f->callpos[i]))
						continue;
					o = (o + bytes - 1) / bytes * bytes + bytes;
				}
				if (o > f->save_bytes)
					f->save_bytes = o;
			}
			f->save_bytes += 64;
		}
		if (off > f->save_bytes)
			fail(c, "%s: save area too small (%u > %u)", f->name ? f->name : "entry", off, f->save_bytes);
		if (fr->boundary)
			break;
	}
	/* allocate the areas (permanent) and emit the stores */
	for (k = 0; k < n; k++) {
		struct frame *fr;
		struct pfunc *f = NULL;
		nir_def *v;
		int d;

		/* find the frame this variable belongs to: walk again (cheap enough) */
		for (d = c->depth; d >= 0; d--) {
			fr = &c->frames[d];
			if (fr->regs && _mesa_hash_table_search(fr->regs, list[k].var->name) &&
			    ((struct preg *)_mesa_hash_table_search(fr->regs, list[k].var->name)->data)->var == list[k].var) {
				f = fr->fn;
				break;
			}
			if (fr->boundary)
				break;
		}
		if (!f)
			fail(c, "caller-save: frame not found");
		if (!f->save_alloc) {
			c->local_top = c->local_top > c->floor ? c->local_top : c->floor;
			f->save_off = alloc_local(c, f->save_bytes ? f->save_bytes : 1, 8);
			c->floor = c->local_top;
			f->save_alloc = true;
		}
		list[k].off += f->save_off;
		v = nir_load_var(b, list[k].var);
		if (list[k].bits == 1)
			v = nir_b2i8(b, v);
		nir_store_scratch(b, v, nir_imm_int(b, list[k].off), .write_mask = 1, .align_mul = v->bit_size / 8);
	}
	*count = n;
	return list;
}

static void caller_restore(struct ctx *c, struct saved_reg *list, int n)
{
	nir_builder *b = &c->b;
	int k;

	for (k = 0; k < n; k++) {
		unsigned bits = list[k].bits == 1 ? 8 : list[k].bits;
		nir_def *v = nir_load_scratch(b, 1, bits, nir_imm_int(b, list[k].off), .align_mul = bits / 8);

		if (list[k].bits == 1)
			v = nir_ine_imm(b, v, 0);
		nir_store_var(b, list[k].var, v, 1);
	}
}

/* A call of an outlined function: arguments in, retid, jump; return values out at the site. */
static void call_outlined(struct ctx *c, struct pfunc *f, struct psym **bind, int nrets, int nargs)
{
	nir_builder *b = &c->b;
	nir_block *cont;
	int i;

	struct saved_reg *saved;
	int nsaved;

	outline_prepare(c, f);
	if (nrets != f->nrets || nargs != f->nparams)
		fail(c, "%s: %d/%d return values, %d/%d arguments", f->name, nrets, f->nrets, nargs, f->nparams);
	if (!f->emitted) {
		size_t pos = c->stmt_pos;	/* the body's statements overwrite it */

		flush_regs(c);			/* the body's blocks start a new cache */
		outline_emit(c, f);
		c->stmt_pos = pos;
	}
	saved = c->regmem ? NULL : caller_save(c, &nsaved);
	if (c->regmem)
		nsaved = 0;
	for (i = 0; i < nargs; i++)
		copy_scratch(c, f->params[i]->offset, bind[nrets + i]->offset,
			     bind[nrets + i]->size < f->params[i]->size ? bind[nrets + i]->size : f->params[i]->size);
	cont = new_block(c);
	if (f->nsites == f->maxsites) {
		f->maxsites = f->maxsites ? f->maxsites * 2 : 16;
		f->sites = reralloc(c->mem, f->sites, nir_block *, f->maxsites);
	}
	f->sites[f->nsites] = cont;
	nir_store_var(b, f->retid, nir_imm_int(b, f->nsites), 1);
	f->nsites++;
	jump(c, f->entry_blk);
	switch_to(c, cont);
	caller_restore(c, saved, nsaved);
	for (i = 0; i < nrets; i++)
		copy_scratch(c, bind[i]->offset, f->rets[i]->offset,
			     bind[i]->size < f->rets[i]->size ? bind[i]->size : f->rets[i]->size);
}

/*
 * Blocks nothing jumps to (labels never branched to, outlined bodies whose calls were all in
 * dead code) would keep a goto the structurizer never visits, and NAK then finds a jump with no
 * target. Walk the gotos from the start block and unlink the rest.
 */
static void remove_unreachable_blocks(struct ctx *c)
{
	nir_function_impl *impl = c->impl;
	unsigned int n = 0, i, top = 0;
	nir_block **stack, **blocks;
	bool *seen;

	nir_foreach_block_unstructured(block, impl)
		n++;
	blocks = calloc(n, sizeof(*blocks));
	stack = calloc(n + 1, sizeof(*stack));
	seen = calloc(n, sizeof(*seen));
	i = 0;
	nir_foreach_block_unstructured(block, impl) {
		block->index = i;
		blocks[i++] = block;
	}
	seen[nir_start_block(impl)->index] = true;
	stack[top++] = nir_start_block(impl);
	while (top) {
		nir_block *b = stack[--top];
		nir_instr *last = nir_block_last_instr(b);
		nir_jump_instr *j = last && last->type == nir_instr_type_jump ? nir_instr_as_jump(last) : NULL;
		nir_block *t[2] = { NULL, NULL };
		int k;

		if (j && j->type == nir_jump_goto) {
			t[0] = j->target;
		} else if (j && j->type == nir_jump_goto_if) {
			t[0] = j->target;
			t[1] = j->else_target;
		}
		for (k = 0; k < 2; k++)
			if (t[k] && !seen[t[k]->index]) {
				seen[t[k]->index] = true;
				stack[top++] = t[k];
			}
	}
	for (i = 0; i < n; i++) {
		if (seen[i] || blocks[i] == impl->end_block)
			continue;
		nir_foreach_instr_safe(instr, blocks[i])
			nir_instr_remove(instr);
		exec_node_remove(&blocks[i]->cf_node.node);
	}
	free(blocks);
	free(stack);
	free(seen);
}

/* After everything: each outlined function's ret dispatches on retid. */
static void emit_dispatch(struct ctx *c)
{
	nir_builder *b = &c->b;
	int i, k;

	for (i = 0; i < c->noutlined; i++) {
		struct pfunc *f = c->outlined[i];

		if (!f->emitted)
			continue;
		switch_to(c, f->ret_blk);
		for (k = 0; k < f->nsites; k++) {
			nir_block *next = new_block(c);

			jump_if(c, f->sites[k], nir_ieq_imm(b, nir_load_var(b, f->retid), k), next);
			switch_to(c, next);
		}
		jump(c, c->impl->end_block);	/* not reached */
	}
}

/*
 * Which functions to emit once instead of inlining: those full inlining would copy often or
 * large (a path tracer's noise functions: 39 KB of PTX x 595 copies). Static instance counts
 * over the call graph in topological order; an outlined function counts as one copy for its
 * callees. NVCUDA_OUTLINE=0 inlines everything, =all outlines every called function.
 */
static void topo_visit(struct ctx *c, struct pfunc *f, struct pfunc **order, int *n)
{
	int i;

	if (f->visited)
		return;
	if (f->visiting)
		fail(c, "recursion through %s", f->name ? f->name : "the entry");
	f->visiting = true;
	for (i = 0; i < f->ncalls; i++) {
		struct hash_entry *e = _mesa_hash_table_search(c->funcs, f->calls[i]);

		if (e)
			topo_visit(c, e->data, order, n);
	}
	f->visiting = false;
	f->visited = true;
	order[(*n)++] = f;	/* callees first */
}

static void decide_outlining(struct ctx *c)
{
	const char *env = getenv("NVCUDA_OUTLINE");
	struct pfunc **order = ralloc_array(c->mem, struct pfunc *, c->funcs->entries + 1);
	int n = 0, i, k;
	bool none = c->po.outline == 0 || (c->po.outline < 0 && env && !strcmp(env, "0"));
	bool all = c->po.outline < 0 && env && !strcmp(env, "all");
	topo_visit(c, &c->entry_fn, order, &n);
	c->entry_fn.inst = 1;
	for (i = n - 1; i >= 0; i--) {	/* callers first */
		struct pfunc *f = order[i];
		uint32_t eff;

		if (f != &c->entry_fn && f->inst) {
			size_t size = f->body_end ? (size_t)(f->body_end - f->hdr) : 0;

			f->outlined = !none && (all || (f->inst >= 2 && ((uint64_t)f->inst * size >= 256 * 1024 ||
									(f->inst >= 8 && size >= 8 * 1024))));
		}
		eff = f->outlined ? 1 : f->inst;
		if (f != &c->entry_fn && f->body_end)
			c->est_bytes += (uint64_t)eff * (size_t)(f->body_end - f->hdr);
		for (k = 0; k < f->ncalls; k++) {
			struct hash_entry *e = _mesa_hash_table_search(c->funcs, f->calls[k]);

			if (e)
				((struct pfunc *)e->data)->inst += eff;
		}
	}
}

/*
 * Static .shared variables of a body (cursor on its '{'), so dynamic (extern) shared memory can
 * start after all of them. Every function's body is scanned, whether or not it is called.
 */
static void add_call(struct ctx *c, struct pfunc *f, const char *callee)
{
	if (!f)
		return;
	if (f->ncalls == f->maxcalls) {
		f->maxcalls = f->maxcalls ? f->maxcalls * 2 : 16;
		f->calls = reralloc(c->mem, f->calls, const char *, f->maxcalls);
	}
	f->calls[f->ncalls++] = ralloc_strdup(c->mem, callee);
}

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
		} else if (c->lx.t == T_ID && (!strcmp(c->lx.s, "call") || !strncmp(c->lx.s, "call.", 5))) {
			/* call[.uni] [(rets),] name ...: the static call graph */
			next(c);
			if (accept(c, '(')) {
				while (!is_p(c, ')') && c->lx.t != T_EOF)
					next(c);
				expect(c, ')');
				accept(c, ',');
			}
			if (c->lx.t == T_ID)
				add_call(c, c->cur_fn, c->lx.s);
			continue;
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
	snprintf(c->globals[c->nglobals].name, sizeof(c->globals[0].name), "%.127s", name);
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
				c->cur_fn = &c->entry_fn;
				prescan_shared(c);
				c->cur_fn = NULL;
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
					f = rzalloc(c->mem, struct pfunc);
					_mesa_hash_table_insert(c->funcs, ralloc_strdup(c->mem, name), f);
				}
				f->hdr = hdr;
				f->line = line;
				f->has_body = true;
				f->name = ralloc_strdup(c->mem, name);
				f->ncalls = 0;	/* a declaration may have been scanned first */
				c->cur_fn = f;
				prescan_shared(c);
				c->cur_fn = NULL;
				f->body_end = c->lx.p;
			} else if (!f) {
				f = rzalloc(c->mem, struct pfunc);
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
	c->shared_size = NVC_DUMMY_BYTES;	/* dummy slots for generic accesses (access_generic) */
	c->local_top = c->local_max = NVC_DUMMY_BYTES;
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
			  struct nvc_kernel_bin *kb, uint64_t globals_va, const struct nvc_ptx_opts *po, char *err,
			  size_t errlen)
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
	c->po = *po;
	kb->used_regmem = false;
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
	c->frames[0].fn = &c->entry_fn;
	c->frames[0].boundary = true;
	c->entry_fn.hdr = body;
	c->entry_fn.line = body_line;
	decide_outlining(c);
	{
		/*
		 * Memory mode for big kernels: NAK's spiller cannot handle ~1000 values live across
		 * tens of thousands of blocks; 8 MB of inlined PTX is about where that starts.
		 */
		const char *rm = getenv("NVCUDA_REGMEM");

		if (c->po.regmem >= 0)
			c->regmem = c->po.regmem;
		else
			c->regmem = rm ? rm[0] != '0' : c->est_bytes > (4u << 20);
		if (c->regmem) {
			compute_liveness(c, &c->entry_fn);
			c->frames[0].regbase = c->entry_fn.regs_bytes ? alloc_local(c, c->entry_fn.regs_bytes, 8) : 0;
		}
		kb->used_regmem = c->regmem;
		if (getenv("NVCUDA_STATS"))
			fprintf(stderr, "nvcuda: %s: inlines %llu KB of PTX, registers in %s\n", entry,
				(unsigned long long)(c->est_bytes >> 10), c->regmem ? "scratch" : "SSA");
	}
	c->lx.p = body;
	c->lx.line = body_line;
	next(c);
	parse_body(c);
	emit_dispatch(c);
	remove_unreachable_blocks(c);
	nir->info.shared_size = c->shared_size;
	nir->scratch_size = c->local_max;
	nir->info.workgroup_size_variable = true;
	if (getenv("NVCUDA_CALLSTATS")) {	/* inlining: instances x body size per function */
		hash_table_foreach(c->funcs, e) {
			struct pfunc *f = e->data;

			if (f->instances || f->emitted)
				fprintf(stderr, "callstats %u %zu %s%s\n", f->instances,
					f->body_end ? (size_t)(f->body_end - f->hdr) : 0, f->name,
					f->emitted ? " (outlined)" : "");
		}
	}
out:
	ctx_free(c);
	return nir;
}
