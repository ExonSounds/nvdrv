// SPDX-License-Identifier: GPL-2.0
/*
 * VRAM pool: one large range carved from the free framebuffer region on first use and
 * managed with drm_mm, so GEM objects, client page tables and channel contexts can be
 * freed again (the carve-out allocator in evo.c only grows).
 */
#include "nvdrv.h"

#define NV_MEM_KEEP	(256ULL << 20)	/* left to the carve-out for nvdrv's own objects */

static int pool_init(struct nvdrv_dev *nv)
{
	struct nv_mem *m = &nv->mem;
	struct nv_evo *e = &nv->evo;
	u64 dummy, size;
	int ret;

	if (m->ready)
		return 0;
	ret = nv_vram_alloc(nv, SZ_4K, SZ_4K, &dummy);	/* initialises the carve-out bounds */
	if (ret)
		return ret;
	size = ALIGN_DOWN(e->vram_next - e->vram_bottom, SZ_2M);
	if (size <= NV_MEM_KEEP + SZ_64M)
		return -ENOSPC;
	size -= NV_MEM_KEEP;
	ret = nv_vram_alloc(nv, size, SZ_2M, &m->base);
	if (ret)
		return ret;
	m->size = size;
	drm_mm_init(&m->mm, m->base, size);
	m->ready = true;
	dev_info(&nv->pdev->dev, "mem: VRAM pool 0x%llx..0x%llx (%llu MiB)\n", m->base, m->base + size - 1,
		 size >> 20);
	return 0;
}

int nv_mem_get(struct nvdrv_dev *nv, struct drm_mm_node *n, u64 size, u64 align)
{
	struct nv_mem *m = &nv->mem;
	int ret;

	mutex_lock(&m->lock);
	ret = pool_init(nv);
	if (!ret) {
		memset(n, 0, sizeof(*n));
		ret = drm_mm_insert_node_generic(&m->mm, n, ALIGN(size, SZ_4K), max_t(u64, align, SZ_4K), 0,
						 DRM_MM_INSERT_BEST);
	}
	if (!ret)
		m->used += n->size;
	mutex_unlock(&m->lock);
	return ret;
}

void nv_mem_put(struct nvdrv_dev *nv, struct drm_mm_node *n)
{
	struct nv_mem *m = &nv->mem;

	if (!drm_mm_node_allocated(n))
		return;
	mutex_lock(&m->lock);
	m->used -= n->size;
	drm_mm_remove_node(n);
	mutex_unlock(&m->lock);
}

/* With the GPU going away; objects still holding pool memory then leak it. */
void nv_mem_fini(struct nvdrv_dev *nv)
{
	struct nv_mem *m = &nv->mem;

	mutex_lock(&m->lock);
	if (m->ready && drm_mm_clean(&m->mm))
		drm_mm_takedown(&m->mm);
	else if (m->ready)
		dev_warn(&nv->pdev->dev, "mem: %llu KiB of pool VRAM still allocated at teardown\n", m->used >> 10);
	m->ready = false;
	m->used = 0;
	mutex_unlock(&m->lock);
}
