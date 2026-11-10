// SPDX-License-Identifier: GPL-2.0
/*
 * Video engines: NVDEC (decode, NVC7B0_VIDEO_DECODER) and NVENC (encode, NVC7B7_VIDEO_ENCODER).
 *
 * A video channel is set up the way nouveau's r535 code does it (MIT, r535/fifo.c r535_flcn_ctor
 * and r535/nvdec.c, nvenc.c):
 *   1. the engine's runlist and engine descriptor come from the device info table
 *   2. the context buffer size comes from GPU_GET_CONSTRUCTED_FALCON_INFO (by descriptor; r570
 *      nouveau r570_fifo_ectx_size)
 *   3. a GPFIFO channel is created on that runlist; the context buffer goes in VRAM and is
 *      mapped into the channel's VM (not privileged: the engine faults on a privileged PTE)
 *   4. GPU_PROMOTE_CTX in the single-buffer form ({virtAddress, size}, no entries)
 *   5. the class object is allocated with {size, prohibitMultipleInstances, engineInstance}
 *
 * No userspace drives these engines yet: NVK has no Vulkan Video queue, and Mesa has no VA-API
 * driver for Ampere. So debugfs video_test checks that the engine's microcode runs a channel's
 * methods: SET_OBJECT, then a semaphore release executed by the engine itself (SEMAPHORE_A..D,
 * same offsets in clc7b0.h and clc7b7.h). Each test builds its own RM client and VM and frees
 * them afterwards.
 */
#include <linux/iopoll.h>

#include "nvdrv.h"

#define NV2080_CTRL_CMD_GPU_GET_CONSTRUCTED_FALCON_INFO		0x208001b0	/* r570; r535: INTERNAL 0x20800a42 */
#define NV2080_CTRL_CMD_GPU_PROMOTE_CTX				0x2080012b

#define VID_SEMAPHORE_A		0x240	/* A: address 39:32, B: address 31:0, C: payload */
#define VID_SEMAPHORE_D		0x304	/* 0: STRUCTURE_SIZE ONE, OPERATION RELEASE, 32-bit, flush */

#define H_VIDEO_CLIENT		0xc1d00010
#define H_VIDEO_CHAN		0xd000c300
#define H_VIDEO_OBJ		0xd000c301
#define VIDEO_VA		(1ULL << 32)	/* below 1 TiB: SEMAPHORE_A holds VA bits 39:32 only */

struct nv_falcon_info {
	__le32 eng_desc, ctx_attr, ctx_size, addr_spaces, reg_base;
};

struct nv_falcon_info_params {
	__le32 count;
	struct nv_falcon_info t[64];	/* NV2080_CTRL_CMD_INTERNAL_MAX_CONSTRUCTED_FALCONS */
};
static_assert(sizeof(struct nv_falcon_info_params) == 1284);

/* NV2080_CTRL_GPU_PROMOTE_CTX_PARAMS (same layout as the GR promote in fifo.c) */
struct nv_video_promote {
	__le32 engine_type, h_client, chid, h_chan_client, h_object, h_virt_memory;
	__le64 virt_address, size;
	__le32 entry_count, pad;
	u8 entries[16 * 32];
};
static_assert(sizeof(struct nv_video_promote) == 560);

/* NV_BSP_ALLOCATION_PARAMETERS / NV_MSENC_ALLOCATION_PARAMETERS */
struct nv_video_alloc {
	__le32 size, prohibit_multiple, engine_instance;
};

static const struct nv_video_engine {
	const char *name;
	u32 rm_type;	/* rm_engine_type.h, the device info table */
	u32 nv2080;	/* NV2080_ENGINE_TYPE_*, channel allocation and promote */
	u32 cls;
} video_engines[] = {
	{ "NVDEC0", 0x1d, 0x13, 0xc7b0 },	/* RM_ENGINE_TYPE_NVDEC0, NV2080_ENGINE_TYPE_NVDEC0 */
	{ "NVENC0", 0x25, 0x1b, 0xc7b7 },	/* RM_ENGINE_TYPE_NVENC0, NV2080_ENGINE_TYPE_NVENC0 */
};

static int video_engine_test(struct nvdrv_dev *nv, const struct nv_video_engine *e,
			     const struct nv_falcon_info_params *fi)
{
	struct device *dev = &nv->pdev->dev;
	struct nv_video_promote *pc = NULL;
	struct drm_mm_node ctx = {};
	struct nv_chan ch = {};
	struct nv_vmm vm;
	u32 runlist, rl_base, desc, size = 0, seq, got = 0, i;
	u64 va = 0;
	int chid, ret;

	ret = nv_fifo_engine_find(nv, e->rm_type, &runlist, &rl_base, &desc);
	if (ret) {
		dev_info(dev, "video: %s not in the device info table\n", e->name);
		return ret;
	}
	for (i = 0; i < le32_to_cpu(fi->count) && i < ARRAY_SIZE(fi->t); i++)
		if (le32_to_cpu(fi->t[i].eng_desc) == desc)
			size = le32_to_cpu(fi->t[i].ctx_size);
	if (!size) {
		dev_warn(dev, "video: %s (desc 0x%x): no context buffer size\n", e->name, desc);
		return -ENODEV;
	}

	ret = nv_vmm_client_init(nv, &vm, H_VIDEO_CLIENT, VIDEO_VA);
	if (ret)
		return ret;
	chid = nv_chid_get(nv);
	if (chid < 0) {
		ret = chid;
		goto out_vm;
	}
	ret = nv_chan_create(nv, &ch, &vm, H_VIDEO_CHAN, chid, e->nv2080, runlist, rl_base);
	if (ret) {
		if (!ch.chid)
			nv_chid_put(nv, chid);
		goto out_chan;
	}
	ret = nv_mem_get(nv, &ctx, size, SZ_4K);
	if (!ret) {
		nv_vram_fill(nv, ctx.start, size, 0, NULL, NULL);
		ret = nv_vmm_map_vram_flags(nv, &vm, ctx.start, size, SZ_4K, 0, &va);
	}
	if (!ret)
		ret = nv_vmm_flush(nv, &vm);
	pc = kzalloc(sizeof(*pc), GFP_KERNEL);
	if (!ret && !pc)
		ret = -ENOMEM;
	if (!ret) {
		pc->engine_type = cpu_to_le32(e->nv2080);
		pc->h_client = cpu_to_le32(vm.h_client);
		pc->chid = cpu_to_le32(ch.chid);
		pc->h_chan_client = cpu_to_le32(vm.h_client);
		pc->h_object = cpu_to_le32(ch.handle);
		pc->virt_address = cpu_to_le64(va);
		pc->size = cpu_to_le64(size);
		ret = nv_rm_control(nv, vm.h_client, vm.h_subdevice, NV2080_CTRL_CMD_GPU_PROMOTE_CTX,
				    pc, sizeof(*pc));
		if (ret)
			dev_warn(dev, "video: %s context promote: %d\n", e->name, ret);
	}
	if (!ret) {
		struct nv_video_alloc a = { cpu_to_le32(sizeof(a)), 0, 0 };

		ret = nv_rm_alloc(nv, vm.h_client, ch.handle, H_VIDEO_OBJ, e->cls, &a, sizeof(a));
		if (ret)
			dev_warn(dev, "video: %s class 0x%04x: %d\n", e->name, e->cls, ret);
	}
	if (!ret) {
		u32 sem[3];

		seq = 0x51de0000 | e->rm_type;
		sem[0] = upper_32_bits(ch.sem_va);
		sem[1] = lower_32_bits(ch.sem_va);
		sem[2] = seq;
		WRITE_ONCE(*(u32 *)ch.sem.cpu, 0);
		nv_push1(&ch, 0, 0, e->cls);			/* SET_OBJECT, subchannel 0 */
		nv_push(&ch, 0, VID_SEMAPHORE_A, sem, 3);
		nv_push1(&ch, 0, VID_SEMAPHORE_D, 0);		/* release by the engine */
		nv_kick(nv, &ch);
		ret = read_poll_timeout(READ_ONCE, got, got == seq, 50, 2000000, false, *(u32 *)ch.sem.cpu);
	}
	dev_info(dev, "video: %s: runlist %u, desc 0x%x, context %u KiB at va 0x%llx, class 0x%04x on channel %u: semaphore 0x%08x -> %s\n",
		 e->name, runlist, desc, size >> 10, va, e->cls, ch.chid, got, ret ? "FAIL" : "PASS");
	kfree(pc);
out_chan:
	nv_chan_destroy(nv, &ch);	/* the class object goes with the channel */
	nv_mem_put(nv, &ctx);
out_vm:
	nv_vmm_client_fini(nv, &vm);
	return ret;
}

int nv_video_test(struct nvdrv_dev *nv)
{
	struct nv_gsp_static_info *si = &nv->gsp_info;
	struct nv_falcon_info_params *fi;
	int ret, r, i;

	fi = kzalloc(sizeof(*fi), GFP_KERNEL);
	if (!fi)
		return -ENOMEM;
	mutex_lock(&nv->gsp_lock);
	if (!nv->gsp_running || READ_ONCE(nv->uapi.lost)) {
		ret = -ENODEV;
		goto out;
	}
	ret = nv_rm_control(nv, si->h_internal_client, si->h_internal_subdevice,
			    NV2080_CTRL_CMD_GPU_GET_CONSTRUCTED_FALCON_INFO, fi, sizeof(*fi));
	if (ret)
		goto out;
	for (i = 0; i < le32_to_cpu(fi->count) && i < ARRAY_SIZE(fi->t); i++)
		dev_dbg(&nv->pdev->dev, "video: falcon desc 0x%x, context %u bytes\n",
			le32_to_cpu(fi->t[i].eng_desc), le32_to_cpu(fi->t[i].ctx_size));
	for (i = 0; i < ARRAY_SIZE(video_engines); i++) {
		r = video_engine_test(nv, &video_engines[i], fi);
		ret = ret ?: r;
	}
out:
	mutex_unlock(&nv->gsp_lock);
	kfree(fi);
	return ret;
}
