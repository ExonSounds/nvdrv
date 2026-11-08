// SPDX-License-Identifier: GPL-2.0
/*
 * GSP boot arguments and RPC queue memory (Phase 5/6, host side).
 *
 * Mirrors open-gpu-kernel-modules 570.144:
 *   kgspAllocBootArgs_TU102          LibOS init-args page, GSP_ARGUMENTS_CACHED page
 *   _kgspInitLibosLoggingStructures  per-task log buffers (+ LOGKRNL on GA10x/AD10x)
 *   kgspSetupLibosInitArgs           LibosMemoryRegionInitArgument[]: logs, then RMARGS
 *   GspMsgQueuesInit / msgqTxCreate  shared memory: PTE page, cmd queue, status queue
 *   kgspPopulateGspRmInitArgs        GSP_ARGUMENTS_CACHED.messageQueueInitArguments
 *
 * Host memory only; nothing is written to the GPU here.
 */
#include <linux/dma-mapping.h>
#include <linux/stddef.h>

#include "nvdrv.h"

/* LibosMemoryRegionInitArgument (uproc/os/common/include/libos_init_args.h) */
struct libos_region_arg {
	__le64 id8;
	__le64 pa;
	__le64 size;
	u8 kind;
	u8 loc;
	u8 pad[6];
};
static_assert(sizeof(struct libos_region_arg) == 32);

#define LIBOS_REGION_CONTIGUOUS	1
#define LIBOS_LOC_SYSMEM	1
#define LIBOS_INIT_ARGS_SIZE	SZ_4K	/* LIBOS_INIT_ARGUMENTS_SIZE */

/* GSP_ARGUMENTS_CACHED (inc/kernel/gpu/gsp/gsp_init_args.h), natural alignment */
struct gsp_args_cached {
	__le64 mq_shared_mem_phys_addr;		/* MESSAGE_QUEUE_INIT_ARGUMENTS */
	__le32 mq_page_table_entry_count;
	__le32 pad0;
	__le64 mq_cmd_queue_offset;		/* NvLength */
	__le64 mq_stat_queue_offset;
	__le32 sr_old_level;			/* GSP_SR_INIT_ARGUMENTS */
	__le32 sr_flags;
	u8 sr_in_pm_transition;
	u8 pad1[3];
	__le32 gpu_instance;
	u8 dmem_stack;
	u8 pad2[7];
	__le64 profiler_pa;
	__le64 profiler_size;
};
static_assert(sizeof(struct gsp_args_cached) == 72);
static_assert(offsetof(struct gsp_args_cached, sr_old_level) == 32);
static_assert(offsetof(struct gsp_args_cached, gpu_instance) == 44);
static_assert(offsetof(struct gsp_args_cached, profiler_pa) == 56);

/* msgqTxHeader / msgqRxHeader (common/shared/msgq/inc/msgq/msgq_priv.h) */
struct msgq_tx_header {
	__le32 version;
	__le32 size;
	__le32 msg_size;
	__le32 msg_count;
	__le32 write_ptr;
	__le32 flags;
	__le32 rx_hdr_off;
	__le32 entry_off;
};
static_assert(sizeof(struct msgq_tx_header) == 32);

#define MSGQ_VERSION		0
#define MSGQ_FLAGS_SWAP_RX	1
#define GSP_MSGQ_ELEMENT_SIZE	SZ_4K	/* GSP_MSG_QUEUE_ELEMENT_SIZE_MIN */
#define GSP_MSGQ_HEADER_ALIGN	4	/* 2^4 */
#define GSP_MSGQ_ELEMENT_ALIGN	12	/* 2^12 */
#define GSP_CMDQ_SIZE		SZ_256K	/* silicon default */
#define GSP_STATQ_SIZE		SZ_256K

/* Release build sizes; LOGKRNL is added for GA10x/AD10x (LOGIDX_SIZE = 5 there). */
static const struct {
	const char *id;
	u32 size;
} nv_libos_logs[NV_LIBOS_NUM_LOGS] = {
	{ "LOGINIT", SZ_64K },	/* must be first: early init logging */
	{ "LOGINTR", SZ_64K },
	{ "LOGRM",   SZ_64K },
	{ "LOGMNOC", SZ_64K },
	{ "LOGKRNL", SZ_64K },
};

/* _kgspGenerateInitArgId: up to 8 chars, first char most significant. */
static u64 libos_id8(const char *name)
{
	u64 id = 0;
	int i;

	for (i = 0; i < 8 && name[i]; i++)
		id = (id << 8) | (u8)name[i];
	return id;
}

static int dma_buf_alloc(struct nvdrv_dev *nv, struct nv_dma_buf *b, size_t size)
{
	b->cpu = dma_alloc_coherent(&nv->pdev->dev, size, &b->dma, GFP_KERNEL);
	if (!b->cpu)
		return -ENOMEM;
	b->size = size;
	memset(b->cpu, 0, size);
	return 0;
}

static void dma_buf_free(struct nvdrv_dev *nv, struct nv_dma_buf *b)
{
	if (b->cpu)
		dma_free_coherent(&nv->pdev->dev, b->size, b->cpu, b->dma);
	memset(b, 0, sizeof(*b));
}

/* _setupLogBufferBaremetal: word 0 = put pointer, words 1.. = bus address of each page. */
static int libos_log_alloc(struct nvdrv_dev *nv, struct nv_dma_buf *b, u32 size)
{
	__le64 *w;
	u32 i;
	int ret;

	ret = dma_buf_alloc(nv, b, size);
	if (ret)
		return ret;
	w = b->cpu;
	for (i = 0; i < size / SZ_4K; i++)
		w[1 + i] = cpu_to_le64(b->dma + (u64)i * SZ_4K);
	return 0;
}

/* msgqTxCreate for the RM command queue; the status queue is GSP's to create. */
static void msgq_format(struct nv_libos *l)
{
	struct msgq_tx_header *tx;
	u32 rx_off, entry_off;

	memset(l->shm.cpu + l->pt_size, 0, l->cmdq_size + l->statq_size);
	rx_off = ALIGN(sizeof(*tx), 1 << GSP_MSGQ_HEADER_ALIGN);
	entry_off = ALIGN(rx_off + sizeof(__le32), 1 << GSP_MSGQ_ELEMENT_ALIGN);
	tx = l->shm.cpu + l->pt_size;
	tx->version = cpu_to_le32(MSGQ_VERSION);
	tx->size = cpu_to_le32(l->cmdq_size);
	tx->msg_size = cpu_to_le32(GSP_MSGQ_ELEMENT_SIZE);
	tx->msg_count = cpu_to_le32((l->cmdq_size - entry_off) / GSP_MSGQ_ELEMENT_SIZE);
	tx->write_ptr = 0;
	tx->flags = cpu_to_le32(MSGQ_FLAGS_SWAP_RX);
	tx->rx_hdr_off = cpu_to_le32(rx_off);
	tx->entry_off = cpu_to_le32(entry_off);
}

/* Fresh queues and empty logs for a new boot attempt (PTEs and args unchanged). */
void nv_libos_reset(struct nvdrv_dev *nv)
{
	struct nv_libos *l = &nv->libos;
	int i;

	msgq_format(l);
	for (i = 0; i < NV_LIBOS_NUM_LOGS; i++) {
		u32 ptes = l->logs[i].size / SZ_4K;

		*(__le64 *)l->logs[i].cpu = 0;	/* put pointer */
		memset(l->logs[i].cpu + 8 * (1 + ptes), 0, l->logs[i].size - 8 * (1 + ptes));
	}
	wmb();
}

/* GspMsgQueuesInit + _getMsgQueueParams. */
static int msgq_alloc(struct nvdrv_dev *nv, struct nv_libos *l)
{
	u32 queue_pages, i;
	__le64 *pt;
	int ret;

	l->cmdq_size = GSP_CMDQ_SIZE;
	l->statq_size = GSP_STATQ_SIZE;
	queue_pages = (l->cmdq_size + l->statq_size) >> 12;
	l->pte_count = queue_pages + DIV_ROUND_UP(queue_pages * 8, SZ_4K);	/* + PTE pages */
	l->pt_size = ALIGN(l->pte_count * 8, SZ_4K);

	ret = dma_buf_alloc(nv, &l->shm, l->pt_size + l->cmdq_size + l->statq_size);
	if (ret)
		return ret;
	if (l->shm.size >> 12 != l->pte_count)
		return -EINVAL;	/* PTEs must cover the whole block, PTE page included */

	/* One coherent allocation = one contiguous bus range, so PTE i = base + i * 4K. */
	pt = l->shm.cpu;
	for (i = 0; i < l->pte_count; i++)
		pt[i] = cpu_to_le64(l->shm.dma + (u64)i * SZ_4K);

	msgq_format(l);
	return 0;
}

int nv_libos_init(struct nvdrv_dev *nv)
{
	struct device *dev = &nv->pdev->dev;
	struct nv_libos *l = &nv->libos;
	struct libos_region_arg *args;
	struct gsp_args_cached *ga;
	int i, ret;

	for (i = 0; i < NV_LIBOS_NUM_LOGS; i++) {
		ret = libos_log_alloc(nv, &l->logs[i], nv_libos_logs[i].size);
		if (ret)
			goto err;
	}
	/* kmemsysInitFlushSysmemBuffer_GA100: one page, 256-byte aligned (page aligned here) */
	ret = dma_buf_alloc(nv, &l->flush, SZ_4K);
	/* _kgspAllocNotifyOpSharedSurface */
	if (!ret)
		ret = dma_buf_alloc(nv, &l->notify_op, SZ_4K);
	if (!ret)
		ret = dma_buf_alloc(nv, &l->args, LIBOS_INIT_ARGS_SIZE);
	if (!ret)
		ret = dma_buf_alloc(nv, &l->rmargs, SZ_4K);
	if (!ret)
		ret = msgq_alloc(nv, l);
	if (ret)
		goto err;

	/* kgspPopulateGspRmInitArgs, normal boot */
	ga = l->rmargs.cpu;
	ga->mq_shared_mem_phys_addr = cpu_to_le64(l->shm.dma);	/* PTE[0] */
	ga->mq_page_table_entry_count = cpu_to_le32(l->pte_count);
	ga->mq_cmd_queue_offset = cpu_to_le64(l->pt_size);
	ga->mq_stat_queue_offset = cpu_to_le64(l->pt_size + l->cmdq_size);
	ga->dmem_stack = 1;	/* RM_GSP_STACK_PLACEMENT default */
	ga->gpu_instance = 0;

	/* kgspSetupLibosInitArgs: log regions in order, then RMARGS */
	args = l->args.cpu;
	for (i = 0; i < NV_LIBOS_NUM_LOGS; i++) {
		args[i].id8 = cpu_to_le64(libos_id8(nv_libos_logs[i].id));
		args[i].pa = cpu_to_le64(l->logs[i].dma);
		args[i].size = cpu_to_le64(l->logs[i].size);
		args[i].kind = LIBOS_REGION_CONTIGUOUS;
		args[i].loc = LIBOS_LOC_SYSMEM;
	}
	args[i].id8 = cpu_to_le64(libos_id8("RMARGS"));
	args[i].pa = cpu_to_le64(l->rmargs.dma);
	args[i].size = cpu_to_le64(l->rmargs.size);
	args[i].kind = LIBOS_REGION_CONTIGUOUS;
	args[i].loc = LIBOS_LOC_SYSMEM;
	wmb();

	dev_info(dev, "libos: init args %pad (%d regions), rmargs %pad, queues %pad (%u PTEs, cmdq 0x%x @0x%x, statq 0x%x @0x%x)\n",
		 &l->args.dma, NV_LIBOS_NUM_LOGS + 1, &l->rmargs.dma, &l->shm.dma, l->pte_count,
		 l->cmdq_size, l->pt_size, l->statq_size, l->pt_size + l->cmdq_size);
	return 0;
err:
	nv_libos_fini(nv);
	return ret;
}

void nv_libos_fini(struct nvdrv_dev *nv)
{
	struct nv_libos *l = &nv->libos;
	int i;

	dma_buf_free(nv, &l->shm);
	dma_buf_free(nv, &l->notify_op);
	dma_buf_free(nv, &l->flush);
	dma_buf_free(nv, &l->rmargs);
	dma_buf_free(nv, &l->args);
	for (i = 0; i < NV_LIBOS_NUM_LOGS; i++)
		dma_buf_free(nv, &l->logs[i]);
}
