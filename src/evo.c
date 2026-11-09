// SPDX-License-Identifier: GPL-2.0
/*
 * Display engine channels ("EVO"/NVDisplay, class C67x on GA10x) on top of GSP-RM.
 *
 * In GSP-client mode CPU-RM owns several display pieces itself, and nvdrv has to do
 * them in its place (references: open-gpu-kernel-modules 570.144):
 *
 *   display instance memory   inst_mem/disp_inst_mem.c (+ arch/v03): 64 KiB in VRAM,
 *                             8 KiB hash table + context-DMA objects, given to GSP-RM
 *                             with NV2080_CTRL_CMD_INTERNAL_DISPLAY_WRITE_INST_MEM
 *   channel push buffers      disp_channel.c kdispSetPushBufferParamsToPhysical:
 *                             NV2080_CTRL_CMD_INTERNAL_DISPLAY_CHANNEL_PUSHBUFFER,
 *                             then the channel class is allocated on GSP-RM
 *   channel control (PUT/GET) BAR0 NV_UDISP_FE_CHN_ASSY_BASEADR(chn), kdispMapDispChannel
 *
 * VRAM: CPU-RM carves its reserved memory from the top of the last FB region GSP-RM did
 * not reserve (memmgrInitReservedMemory_GM107); nvdrv does the same with a bump
 * allocator, and touches VRAM through the 1 MiB BAR0 PRAMIN window.
 */
#include <linux/delay.h>
#include <linux/dma-mapping.h>
#include <linux/iopoll.h>
#include <linux/lockdep.h>
#include <linux/seq_file.h>
#include <linux/slab.h>

#include "nvdrv.h"
#include "nv_regs.h"

/* dev_bus (ga100 manual): BAR0 window onto VRAM at NV_PRAMIN */
#define NV_PBUS_BAR0_WINDOW		0x00001700
#define   BAR0_WINDOW_BASE_SHIFT	16
#define NV_PRAMIN_BASE			0x00700000

/* disp v03_00 dev_disp.h / ga102 display manual */
#define NV_UDISP_HASH_SIZE		0x2000
#define NV_UDISP_INSTMEM_SIZE		0x10000
#define DISP_INST_MEM_ALIGN		0x10000
#define NV_UDISP_FE_CHN_ASSY_BASEADR(chn) ((chn) ? 0x00690000 + ((chn) - 1) * 0x1000 : 0x00680000)
#define NV_PDISP_FE_CHNSTATUS_CORE	0x00610630
#define NV_PDISP_FE_CHNCTL_CORE		0x006104e0
#define NV_PDISP_FE_EXCEPT(chn)		(0x00611020 + (chn) * 12)
#define NV_PDISP_FE_EXCEPTARG(chn)	(0x00611024 + (chn) * 12)
#define NV_PDISP_CHN_NUM_CORE		0
#define NV_PDISP_CHN_NUM_WIN(i)		(1 + (i))

/* classes */
#define NVC670_DISPLAY			0xc670
#define NVC67D_CORE_CHANNEL_DMA		0xc67d
#define NVC67E_WINDOW_CHANNEL_DMA	0xc67e

#define NVDRV_H_DISP			0xd000c670
#define NVDRV_H_CORE			0xd000c67d
#define NVDRV_H_WIN(i)			(0xd000e000 + (i))
#define NVDRV_H_PB_CTXDMA		0xd000f000	/* never resolved by GSP-RM */

/* internal controls on GSP-RM's internal subdevice */
#define NV2080_CTRL_CMD_INTERNAL_DISPLAY_WRITE_INST_MEM		0x20800a49
#define NV2080_CTRL_CMD_INTERNAL_DISPLAY_CHANNEL_PUSHBUFFER	0x20800a58

#define ADDR_SYSMEM			1
#define ADDR_FBMEM			2
#define NV_MEMORY_WRITECOMBINED		2
#define PHYS_PCI_COHERENT		3	/* PBTARGETAPERTURE */
#define PB_SIZE_4KB			0

struct nv2080_write_inst_mem {
	__le64 phys_addr, size;
	__le32 addr_space, cpu_cache_attr;
};
static_assert(sizeof(struct nv2080_write_inst_mem) == 24);

struct nv2080_channel_pushbuffer {
	__le32 address_space, pad0;
	__le64 physical_addr, limit;
	__le32 cache_snoop, hclass, channel_instance;
	u8 valid, pad1[3];
	__le32 pb_target_aperture, channel_pb_size, sub_device_id, pad2;
};
static_assert(sizeof(struct nv2080_channel_pushbuffer) == 56);
static_assert(offsetof(struct nv2080_channel_pushbuffer, pb_target_aperture) == 40);

struct nv50vaio_channeldma_alloc {
	__le32 channel_instance, h_object_buffer, h_object_notify, offset;
	__le64 p_control;
	__le32 flags, channel_pb_size, sub_device_id, pad;
};
static_assert(sizeof(struct nv50vaio_channeldma_alloc) == 40);

/* ---- VRAM ---------------------------------------------------------------- */

/* Top-down bump allocator below the end of the last unreserved FB region. */
int nv_vram_alloc(struct nvdrv_dev *nv, u64 size, u64 align, u64 *out)
{
	struct nv_evo *e = &nv->evo;
	const struct nv_gsp_static_info *si = &nv->gsp_info;
	int i;

	if (!e->vram_top) {
		for (i = si->num_fb_regions - 1; i >= 0; i--) {
			const struct nv_gsp_fb_region *r = &si->fb_regions[i];

			if (!r->reserved && !r->is_protected) {
				e->vram_bottom = r->base;
				e->vram_top = e->vram_next = r->limit + 1;
				break;
			}
		}
		if (!e->vram_top)
			return -ENOSPC;
	}
	if (e->vram_next < size || ALIGN_DOWN(e->vram_next - size, align) < e->vram_bottom)
		return -ENOSPC;
	e->vram_next = ALIGN_DOWN(e->vram_next - size, align);
	*out = e->vram_next;
	return 0;
}

/* Point the PRAMIN window at the 64K block holding off; returns the BAR0 address. */
static void __iomem *pramin(struct nvdrv_dev *nv, u64 off, u32 *avail)
{
	u64 base = off & ~0xffffULL;

	if (nv->evo.pramin_base != base) {
		nv_wr32(nv, NV_PBUS_BAR0_WINDOW, base >> BAR0_WINDOW_BASE_SHIFT);	/* TARGET = VID_MEM */
		(void)nv_rd32(nv, NV_PBUS_BAR0_WINDOW);
		nv->evo.pramin_base = base;
	}
	*avail = NV_PRAMIN_SIZE - (off - base);
	return nv->bar0 + NV_PRAMIN_BASE + (off - base);
}

void nv_vram_wr32(struct nvdrv_dev *nv, u64 off, u32 val)
{
	u32 avail;

	iowrite32(val, pramin(nv, off, &avail));
}

u32 nv_vram_rd32(struct nvdrv_dev *nv, u64 off)
{
	u32 avail;

	return ioread32(pramin(nv, off, &avail));
}

/* Copy @len bytes (4-byte multiple) from @src into VRAM at @off through the PRAMIN window. */
void nv_vram_write(struct nvdrv_dev *nv, u64 off, const void *src, u64 len)
{
	while (len) {
		u32 avail, n;
		void __iomem *p = pramin(nv, off, &avail);

		n = min_t(u64, len, avail);
		memcpy_toio(p, src, n);
		off += n;
		src += n;
		len -= n;
	}
}

/* Fill [off, off + len) with 32-bit words from fn(index) (or a constant if fn is NULL). */
void nv_vram_fill(struct nvdrv_dev *nv, u64 off, u64 len, u32 val,
		  u32 (*fn)(void *ctx, u64 i), void *ctx)
{
	u64 i = 0;

	while (len) {
		u32 avail, n, k;
		void __iomem *p = pramin(nv, off, &avail);

		n = min_t(u64, len, avail);
		for (k = 0; k < n; k += 4, i++)
			iowrite32(fn ? fn(ctx, i) : val, p + k);
		off += n;
		len -= n;
		cond_resched();
	}
	(void)nv_rd32(nv, NV_PBUS_BAR0_WINDOW);	/* flush posted writes */
}

/* ---- display instance memory ------------------------------------------- */

static int evo_instmem_init(struct nvdrv_dev *nv)
{
	struct nv_evo *e = &nv->evo;
	struct nv2080_write_inst_mem p = {};
	int ret;

	if (!e->instmem) {	/* kept across suspend: only re-announced and cleared */
		ret = nv_vram_alloc(nv, NV_UDISP_INSTMEM_SIZE, DISP_INST_MEM_ALIGN, &e->instmem);
		if (ret)
			return ret;
	}
	nv_vram_fill(nv, e->instmem, NV_UDISP_INSTMEM_SIZE, 0, NULL, NULL);
	/* instance 0x200 (16-byte units) is the reserved NULL instance; ctxdmas follow */
	e->inst_next = (NV_UDISP_HASH_SIZE >> 4) + 2;

	p.phys_addr = cpu_to_le64(e->instmem);
	p.size = cpu_to_le64(NV_UDISP_INSTMEM_SIZE);
	p.addr_space = cpu_to_le32(ADDR_FBMEM);
	p.cpu_cache_attr = cpu_to_le32(NV_MEMORY_WRITECOMBINED);
	ret = nv_rm_control(nv, nv->gsp_info.h_internal_client, nv->gsp_info.h_internal_subdevice,
			    NV2080_CTRL_CMD_INTERNAL_DISPLAY_WRITE_INST_MEM, &p, sizeof(p));
	if (!ret)
		dev_info(&nv->pdev->dev, "evo: display instance memory at VRAM 0x%llx\n", e->instmem);
	return ret;
}

/* instmemHashFunc_v03_00 */
static u32 evo_hash(u32 h_client, u32 h_ctxdma, u32 chn)
{
	return ((h_ctxdma >> 0) & 0x3ff) ^ ((h_ctxdma >> 10) & 0x3ff) ^ ((h_ctxdma >> 20) & 0x3ff) ^
	       (((h_client & 0xff) << 2) | (h_ctxdma >> 30)) ^
	       (((chn & 0xf) << 6) | ((h_client >> 8) & 0x3f)) ^ ((chn >> 4) & 0x7);
}

/*
 * instmemBindContextDma: write a context DMA object (instmemCommitContextDma_v03_00)
 * for a pitch-linear VRAM range and enter (handle -> object) into the hash table for
 * one channel. Returns the handle the channel methods refer to.
 */
int nv_evo_ctxdma(struct nvdrv_dev *nv, u32 handle, u64 base, u64 size, u32 chn, u32 flags)
{
	bool sysmem = flags & NV_CTXDMA_SYSMEM;
	struct nv_evo *e = &nv->evo;
	u32 h_client = nv->rm.h_client, inst, entry, i, n = NV_UDISP_HASH_SIZE / 8;
	u64 obj, limit = base + size - 1;

	if (e->inst_next + 2 > NV_UDISP_INSTMEM_SIZE >> 4)
		return -ENOSPC;
	inst = e->inst_next;
	e->inst_next += 2;
	obj = e->instmem + (inst << 4);
	/* TARGET_NODE: 1 = PHYSICAL_NVM (VRAM), 3 = PHYSICAL_PCI_COHERENT (sysmem bus address) */
	/* KIND (bit 20, nouveau gv100_dmaobj_new): the surface is block linear */
	nv_vram_wr32(nv, obj + 0, (sysmem ? 0x3 : 0x1) | BIT(2) /* READ_AND_WRITE */ |
		     ((flags & NV_CTXDMA_BLOCK_LINEAR) ? BIT(20) : 0));
	nv_vram_wr32(nv, obj + 4, lower_32_bits(base >> 8));
	nv_vram_wr32(nv, obj + 8, upper_32_bits(base >> 8) & 0x7f);
	nv_vram_wr32(nv, obj + 12, lower_32_bits(limit >> 8));
	nv_vram_wr32(nv, obj + 16, upper_32_bits(limit >> 8) & 0x7f);

	entry = evo_hash(h_client, handle, chn);
	for (i = 0; i < n; i++, entry = (entry + 1) & (n - 1))
		if (!nv_vram_rd32(nv, e->instmem + entry * 8 + 4))
			break;
	if (i == n)
		return -ENOSPC;
	nv_vram_wr32(nv, e->instmem + entry * 8, handle);
	/* NV_UDISP_HASH_TBL: CLIENT_ID 13:0, INSTANCE (32-byte units) 24:14, CHN 31:25 */
	nv_vram_wr32(nv, e->instmem + entry * 8 + 4,
		     (h_client & 0x3fff) | ((inst >> 1) & 0x7ff) << 14 | (chn & 0x7f) << 25);
	(void)nv_rd32(nv, NV_PBUS_BAR0_WINDOW);
	dev_info(&nv->pdev->dev, "evo: ctxdma 0x%08x -> %s 0x%llx+0x%llx (inst 0x%x, hash %u, chn %u)\n",
		 handle, sysmem ? "sysmem" : "VRAM", base, size, inst, entry, chn);
	return 0;
}

/* ---- channels ------------------------------------------------------------ */

static int evo_channel_alloc(struct nvdrv_dev *nv, struct nv_evo_chan *c, u32 cls, u32 handle,
			     u32 instance, u32 chn)
{
	struct nv2080_channel_pushbuffer pb = {};
	struct nv50vaio_channeldma_alloc a = {};
	int ret;

	c->pb.size = SZ_4K;
	c->pb.cpu = dma_alloc_coherent(&nv->pdev->dev, c->pb.size, &c->pb.dma, GFP_KERNEL);
	if (!c->pb.cpu)
		return -ENOMEM;
	memset(c->pb.cpu, 0, c->pb.size);
	c->chn = chn;
	c->user = nv->bar0 + NV_UDISP_FE_CHN_ASSY_BASEADR(chn);
	c->put = 0;

	pb.address_space = cpu_to_le32(ADDR_SYSMEM);
	pb.physical_addr = cpu_to_le64(c->pb.dma);
	pb.limit = cpu_to_le64(c->pb.size - 1);
	pb.cache_snoop = cpu_to_le32(1);
	pb.hclass = cpu_to_le32(cls);
	pb.channel_instance = cpu_to_le32(instance);
	pb.valid = 1;
	pb.pb_target_aperture = cpu_to_le32(PHYS_PCI_COHERENT);
	pb.channel_pb_size = cpu_to_le32(PB_SIZE_4KB);
	ret = nv_rm_control(nv, nv->gsp_info.h_internal_client, nv->gsp_info.h_internal_subdevice,
			    NV2080_CTRL_CMD_INTERNAL_DISPLAY_CHANNEL_PUSHBUFFER, &pb, sizeof(pb));
	if (ret)
		return ret;

	a.channel_instance = cpu_to_le32(instance);
	a.h_object_buffer = cpu_to_le32(NVDRV_H_PB_CTXDMA);
	a.channel_pb_size = cpu_to_le32(PB_SIZE_4KB);
	ret = nv_rm_alloc(nv, nv->rm.h_client, nv->evo.h_disp, handle, cls, &a, sizeof(a));
	if (ret)
		return ret;
	c->handle = handle;
	dev_info(&nv->pdev->dev, "evo: channel 0x%04x/%u (chn %u) allocated, push buffer %pad, PUT/GET 0x%x/0x%x\n",
		 cls, instance, chn, &c->pb.dma, ioread32(c->user), ioread32(c->user + 4));
	return 0;
}

/*
 * PIO channel (cursor immediate, class 0xc67a): no push buffer. GSP-RM is told so through
 * CHANNEL_PUSHBUFFER with valid = 0 (nouveau r535_curs_init), then the class is allocated
 * with NV50VAIO_CHANNELPIO_ALLOCATION_PARAMETERS. Methods are BAR0 writes to its user area.
 */
int nv_evo_pio_alloc(struct nvdrv_dev *nv, u32 cls, u32 handle, u32 instance)
{
	struct nv2080_channel_pushbuffer pb = {};
	struct { __le32 channel_instance, h_object_notify; __le64 p_control; } a = {};
	int ret;

	pb.hclass = cpu_to_le32(cls);
	pb.channel_instance = cpu_to_le32(instance);
	ret = nv_rm_control(nv, nv->gsp_info.h_internal_client, nv->gsp_info.h_internal_subdevice,
			    NV2080_CTRL_CMD_INTERNAL_DISPLAY_CHANNEL_PUSHBUFFER, &pb, sizeof(pb));
	if (ret)
		return ret;
	a.channel_instance = cpu_to_le32(instance);
	ret = nv_rm_alloc(nv, nv->rm.h_client, nv->evo.h_disp, handle, cls, &a, sizeof(a));
	dev_info(&nv->pdev->dev, "evo: PIO channel 0x%04x/%u: %d\n", cls, instance, ret);
	return ret;
}

static void evo_channel_free_mem(struct nvdrv_dev *nv, struct nv_evo_chan *c)
{
	if (c->pb.cpu)
		dma_free_coherent(&nv->pdev->dev, c->pb.size, c->pb.cpu, c->pb.dma);
	memset(c, 0, sizeof(*c));
}

/* Append one method with n data words. */
void nv_evo_mthd(struct nv_evo_chan *c, u32 mthd, const u32 *data, u32 n)
{
	u32 *pb = c->pb.cpu, i;

	if (c->put / 4 + 1 + n >= (c->pb.size - 48) / 4) {
		c->overflow = true;
		return;
	}
	pb[c->put / 4] = n << 18 | (mthd & 0xfffc);
	for (i = 0; i < n; i++)
		pb[c->put / 4 + 1 + i] = data[i];
	c->put += 4 * (1 + n);
}

/*
 * Make room for a batch of need bytes: if it would not fit before the end of the
 * push buffer, JUMP back to the start (opcode 1, offset 0) once the channel is idle.
 */
int nv_evo_begin(struct nvdrv_dev *nv, struct nv_evo_chan *c, u32 need)
{
	u32 *pb = c->pb.cpu, get;
	int ret;

	c->overflow = false;
	c->begin_put = c->put;
	if (c->put + need < c->pb.size - 48)
		return 0;
	ret = readl_poll_timeout(c->user + 4, get, get == c->put, 10, 2000000);
	if (ret)
		return ret;
	pb[c->put / 4] = 0x20000000;
	wmb();
	iowrite32(0, c->user);
	c->put = c->begin_put = 0;
	return readl_poll_timeout(c->user + 4, get, get == 0, 10, 2000000);
}

/* Publish PUT. */
int nv_evo_kick_nowait(struct nvdrv_dev *nv, struct nv_evo_chan *c)
{
	if (c->overflow) {
		/*
		 * Drop the whole batch: a partly written one must not advance PUT, or the
		 * software PUT runs ahead of the hardware and every later wrap times out.
		 */
		dev_warn_ratelimited(&nv->pdev->dev, "evo: chn %u: methods overflowed the reservation, batch dropped\n",
				     c->chn);
		c->put = c->begin_put;
		return -E2BIG;
	}
	wmb();
	iowrite32(c->put, c->user);
	return 0;
}

/* Wait until the channel has fetched everything (GET == PUT). */
int nv_evo_wait(struct nvdrv_dev *nv, struct nv_evo_chan *c)
{
	u32 get;
	int ret;

	ret = readl_poll_timeout(c->user + 4, get, get == c->put, 10, 2000000);
	if (ret)
		dev_err(&nv->pdev->dev, "evo: chn %u stuck: PUT 0x%x GET 0x%x, except 0x%08x/0x%08x, core chnstatus 0x%08x\n",
			c->chn, c->put, get, nv_rd32(nv, NV_PDISP_FE_EXCEPT(c->chn)),
			nv_rd32(nv, NV_PDISP_FE_EXCEPTARG(c->chn)), nv_rd32(nv, NV_PDISP_FE_CHNSTATUS_CORE));
	return ret;
}

int nv_evo_kick(struct nvdrv_dev *nv, struct nv_evo_chan *c)
{
	int ret = nv_evo_kick_nowait(nv, c);

	return ret ?: nv_evo_wait(nv, c);
}

int nv_evo_init(struct nvdrv_dev *nv)
{
	struct nv_evo *e = &nv->evo;
	int ret;

	lockdep_assert_held(&nv->gsp_lock);
	if (!nv->gsp_running || !nv->rm.h_device)
		return -ENODEV;
	if (e->core.handle)
		return 0;
	if (!e->instmem_ready) {
		ret = evo_instmem_init(nv);
		if (ret)
			return ret;
		e->instmem_ready = true;
	}
	if (!e->h_disp) {
		ret = nv_rm_alloc(nv, nv->rm.h_client, nv->rm.h_device, NVDRV_H_DISP, NVC670_DISPLAY,
				  NULL, 0);
		if (ret)
			return ret;
		e->h_disp = NVDRV_H_DISP;
	}
	ret = evo_channel_alloc(nv, &e->core, NVC67D_CORE_CHANNEL_DMA, NVDRV_H_CORE, 0,
				NV_PDISP_CHN_NUM_CORE);
	if (ret)
		evo_channel_free_mem(nv, &e->core);
	return ret;
}

int nv_evo_window_init(struct nvdrv_dev *nv, u32 win)
{
	struct nv_evo *e = &nv->evo;
	int ret;

	if (win >= ARRAY_SIZE(e->win))
		return -EINVAL;
	lockdep_assert_held(&e->win_lock);	/* called from the modeset, which holds it */
	ret = 0;
	if (!e->win[win].handle) {
		ret = evo_channel_alloc(nv, &e->win[win], NVC67E_WINDOW_CHANNEL_DMA, NVDRV_H_WIN(win), win,
					NV_PDISP_CHN_NUM_WIN(win));
		if (ret)
			evo_channel_free_mem(nv, &e->win[win]);
	}
	return ret;
}

/*
 * Suspend (nvkms frees its display channels too): the display object goes, and its
 * channels with it; VRAM allocations and BAR1 mappings stay. After resume nv_evo_init
 * announces and clears the instance memory again and everything is set up anew.
 * Caller holds gsp_lock; the head is already off.
 */
#define NV5070_CTRL_CMD_SET_RMFREE_FLAGS	0x50700117

void nv_evo_suspend(struct nvdrv_dev *nv)
{
	struct nv_evo *e = &nv->evo;
	u32 i;

	lockdep_assert_held(&nv->gsp_lock);
	mutex_lock(&e->win_lock);
	if (e->h_disp && nv->gsp_running) {
		/* nvkms nvFreeCoreChannelEvo order: cursors, windows, core, then the display */
		ktime_t t0 = ktime_get();

		u32 h[6], k = 0;

		if (e->curs_ready)
			h[k++] = 0xd000c67a;	/* NVDRV_H_CURS (modeset.c) */
		for (i = 0; i < ARRAY_SIZE(e->win); i++)
			if (e->win[i].handle)
				h[k++] = e->win[i].handle;
		if (e->core.handle)
			h[k++] = e->core.handle;
		h[k++] = e->h_disp;
		for (i = 0; i < k; i++) {
			ktime_t t = ktime_get();
			int r;

			/*
			 * nvkms RmFreeEvoChannel with skipConsoleRestore (its suspend path): RM frees
			 * the core and window channels without shutting the hardware down, which
			 * otherwise runs into RM's 4 s timeout per channel. One-shot flag, per free.
			 */
			if (h[i] != e->h_disp && h[i] != 0xd000c67a) {
				struct { __le32 subdevice, flags; } rf = { 0, cpu_to_le32(1) };	/* PRESERVE_HW */

				r = nv_rm_control(nv, nv->rm.h_client, e->h_disp, NV5070_CTRL_CMD_SET_RMFREE_FLAGS,
						  &rf, sizeof(rf));
				if (r)
					dev_warn(&nv->pdev->dev, "evo: RMFREE_FLAGS: %d\n", r);
			}
			r = nv_rm_free(nv, nv->rm.h_client, h[i]);

			dev_info(&nv->pdev->dev, "evo: free 0x%08x: %d (%lld ms)\n", h[i], r,
				 ktime_ms_delta(ktime_get(), t));
		}
		dev_info(&nv->pdev->dev, "evo: display freed for suspend (%lld ms)\n",
			 ktime_ms_delta(ktime_get(), t0));
	}
	evo_channel_free_mem(nv, &e->core);
	for (i = 0; i < ARRAY_SIZE(e->win); i++)
		evo_channel_free_mem(nv, &e->win[i]);
	e->h_disp = 0;
	e->instmem_ready = false;
	e->active = e->win_mapped = e->fb_bound = e->pool_ctxdma = false;
	e->curs_ready = e->curs_pool_ctxdma = e->lut_ready = e->composed = false;
	e->vrr_active = e->vrr_msa = false;
	e->curs_user = NULL;
	e->curs_ctl = e->curs_ctxdma = 0;
	e->curs_offset = 0;
	e->scanout_cur = 0;
	e->scanout_bh = -1;
	e->pramin_base = ~0ULL;
	mutex_unlock(&e->win_lock);
}

/*
 * The display objects die with the RM client (or the FLR); only host memory is
 * freed here. Called after nv_rm_fini_objects / GPU reset.
 */
void nv_evo_fini(struct nvdrv_dev *nv)
{
	struct nv_evo *e = &nv->evo;
	u32 i;

	mutex_lock(&e->win_lock);
	evo_channel_free_mem(nv, &e->core);
	for (i = 0; i < ARRAY_SIZE(e->win); i++)
		evo_channel_free_mem(nv, &e->win[i]);
	if (e->fb_cpu)
		iounmap(e->fb_cpu);
	if (e->curs_cpu)
		iounmap(e->curs_cpu);
	if (e->lut_cpu)
		iounmap(e->lut_cpu);
	if (e->sysfb.cpu)
		dma_free_coherent(&nv->pdev->dev, e->sysfb.size, e->sysfb.cpu, e->sysfb.dma);
	/* everything but the locks, which live as long as nvdrv_dev */
	memset((char *)e + offsetofend(struct nv_evo, curs_lock), 0,
	       sizeof(*e) - offsetofend(struct nv_evo, curs_lock));
	mutex_unlock(&e->win_lock);
}

/* Caller holds gsp_lock. */
void nv_evo_show(struct nvdrv_dev *nv, struct seq_file *m)
{
	struct nv_evo *e = &nv->evo;
	u32 i;

	seq_puts(m, "\n[evo]\n");
	if (!e->instmem) {
		seq_puts(m, "not initialized\n");
		return;
	}
	seq_printf(m, "vram carve-out:  0x%llx - 0x%llx used, region 0x%llx - 0x%llx\n", e->vram_next,
		   e->vram_top, e->vram_bottom, e->vram_top);
	seq_printf(m, "instmem:         0x%llx, next inst 0x%x\n", e->instmem, e->inst_next);
	seq_printf(m, "core:            handle 0x%08x chnctl 0x%08x chnstatus 0x%08x PUT/GET 0x%x/0x%x except 0x%08x/0x%08x\n",
		   e->core.handle, nv_rd32(nv, NV_PDISP_FE_CHNCTL_CORE), nv_rd32(nv, NV_PDISP_FE_CHNSTATUS_CORE),
		   e->core.handle ? ioread32(e->core.user) : 0, e->core.handle ? ioread32(e->core.user + 4) : 0,
		   nv_rd32(nv, NV_PDISP_FE_EXCEPT(0)), nv_rd32(nv, NV_PDISP_FE_EXCEPTARG(0)));
	for (i = 0; i < ARRAY_SIZE(e->win); i++)
		if (e->win[i].handle)
			seq_printf(m, "window %u:        handle 0x%08x PUT/GET 0x%x/0x%x except 0x%08x/0x%08x\n", i,
				   e->win[i].handle, ioread32(e->win[i].user), ioread32(e->win[i].user + 4),
				   nv_rd32(nv, NV_PDISP_FE_EXCEPT(e->win[i].chn)),
				   nv_rd32(nv, NV_PDISP_FE_EXCEPTARG(e->win[i].chn)));
}
