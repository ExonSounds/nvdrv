/* SPDX-License-Identifier: MIT */
/*
 * A path-tracer-shaped integer computation, compiled both as a CUDA kernel (clang -> PTX) and as
 * host code for an exact reference. Exercises what Cycles' shading kernels do: a bounce loop,
 * a state struct in local memory passed by pointer through non-inlined calls (generic
 * addressing), nested calls with struct returns, early exits, a switch over "materials", many
 * values live across calls.
 */
#ifndef STRESS_KERNEL_H
#define STRESS_KERNEL_H

#ifdef __CUDA__
#define DEV __device__
#define NOINL __attribute__((noinline))
#else
#define DEV
#define NOINL __attribute__((noinline))
#endif

typedef unsigned int u32;

struct state {
	int pos[3], dir[3];
	u32 rng, weight, depth, material, flags;
	u32 history[8];
};

struct hit {
	int t;
	u32 material;
	int normal[3];
};

static DEV inline u32 xorshift(u32 *s)
{
	u32 x = *s;

	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	*s = x;
	return x;
}

static DEV NOINL struct hit intersect(const struct state *st)
{
	struct hit h;
	int k, best = 0x7fffffff;

	h.material = 0;
	h.normal[0] = h.normal[1] = h.normal[2] = 0;
	for (k = 0; k < 6; k++) {
		int plane = (k & 1) ? 1000 : -1000, axis = k >> 1, d = st->dir[axis], p = st->pos[axis];
		int t;

		if (d == 0)
			continue;
		t = (plane - p) * 16 / d;
		if (t > 0 && t < best) {
			best = t;
			h.material = (u32)(k + 1 + (st->depth & 1));
			h.normal[0] = h.normal[1] = h.normal[2] = 0;
			h.normal[axis] = (k & 1) ? -1 : 1;
		}
	}
	h.t = best == 0x7fffffff ? -1 : best;
	return h;
}

static DEV NOINL u32 bsdf_diffuse(struct state *st, const struct hit *h)
{
	u32 r = xorshift(&st->rng);

	st->dir[0] = (int)(r & 63) - 32 + h->normal[0] * 40;
	st->dir[1] = (int)((r >> 6) & 63) - 32 + h->normal[1] * 40;
	st->dir[2] = (int)((r >> 12) & 63) - 32 + h->normal[2] * 40;
	return 200 + (r >> 24);
}

static DEV NOINL u32 bsdf_glossy(struct state *st, const struct hit *h)
{
	int k, dn = 0;

	for (k = 0; k < 3; k++)
		dn += st->dir[k] * h->normal[k];
	for (k = 0; k < 3; k++)
		st->dir[k] = st->dir[k] - 2 * dn * h->normal[k] / 1;
	if (xorshift(&st->rng) & 1)
		st->flags |= 2;
	return 240;
}

static DEV NOINL u32 bsdf_transparent(struct state *st, const struct hit *h)
{
	(void)h;
	st->flags |= 4;
	return 255;
}

static DEV NOINL int shade(struct state *st, const struct hit *h)
{
	u32 albedo;
	int k;

	if (h->t < 0) {		/* background: terminate */
		st->flags |= 1;
		return 0;
	}
	for (k = 0; k < 3; k++)
		st->pos[k] += st->dir[k] * h->t / 16;
	switch (h->material % 3) {
	case 0:
		albedo = bsdf_diffuse(st, h);
		break;
	case 1:
		albedo = bsdf_glossy(st, h);
		break;
	default:
		albedo = bsdf_transparent(st, h);
		break;
	}
	st->weight = st->weight * albedo / 256;
	st->history[st->depth & 7] = (u32)h->t ^ albedo;
	st->material = h->material;
	if (st->weight < 16)
		return 0;	/* russian roulette would go here */
	if (st->depth >= 11)
		return 0;
	st->depth++;
	return 1;
}

/* one thread's work; out[0..3] */
static DEV inline void stress_path(u32 tid, u32 *out)
{
	struct state st;
	int k, bounces = 0;
	u32 acc = 0;

	for (k = 0; k < 3; k++) {
		st.pos[k] = (int)((tid * (7 + k)) % 500) - 250;
		st.dir[k] = (int)((tid * (13 + 5 * k)) % 97) - 48;
	}
	if (!st.dir[0] && !st.dir[1] && !st.dir[2])
		st.dir[2] = 1;
	st.rng = tid * 2654435761u + 1;
	st.weight = 65536;
	st.depth = 0;
	st.material = 0;
	st.flags = 0;
	for (k = 0; k < 8; k++)
		st.history[k] = 0;
	for (;;) {
		struct hit h = intersect(&st);

		if (!shade(&st, &h))
			break;
		bounces++;
		acc += h.t * (bounces + 1);
	}
	for (k = 0; k < 8; k++)
		acc = acc * 31 + st.history[k];
	out[0] = acc;
	out[1] = st.weight | st.flags << 24;
	out[2] = (u32)bounces | st.depth << 8 | st.material << 16;
	out[3] = (u32)st.pos[0] ^ (u32)st.pos[1] * 3 ^ (u32)st.pos[2] * 5 ^ st.rng;
}

#endif
