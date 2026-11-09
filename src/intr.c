// SPDX-License-Identifier: GPL-2.0
/*
 * Interrupts under GSP-RM: GSP-RM owns the stall interrupts; the CPU driver gets the
 * nonstall (fence) vectors of the engines and the GSP's own vector (message queue
 * notification). Vectors come from NV2080_CTRL_CMD_INTERNAL_INTR_GET_KERNEL_TABLE; the
 * CPU interrupt tree is programmed as nouveau does (nvkm/subdev/vfn/tu102.c, r535 gsp.c):
 *
 *   base 0xb80000 (NV_VIRTUAL_FUNCTION):
 *     +0x1000 + 4 * leaf  leaf status, write 1 to clear      (vector v: leaf v / 32, bit v % 32)
 *     +0x1200 + 4 * leaf  leaf enable set
 *     +0x1400 + 4 * leaf  leaf enable clear
 *     +0x1600             top status (bit leaf / 2)
 *     +0x1608 / +0x1610   top enable set / clear (0xf: subtrees 0-3, leaves 0-7)
 *
 * The GSP vector's source is the GSP falcon: IRQSTAT 0x110008 bit 6 (SWGEN0) means the
 * message queue has data; it is cleared through IRQSCLR 0x110004 and the falcon interrupt
 * is retriggered through 0x1103e8 (nouveau ga100_flcn_intr_retrigger).
 */
#include <linux/interrupt.h>
#include <linux/pci.h>
#include <linux/slab.h>
#include <linux/workqueue.h>

#include "nvdrv.h"

#define VFN				0x00b80000
#define VFN_LEAF(l)			(VFN + 0x1000 + (l) * 4)
#define VFN_LEAF_EN_SET(l)		(VFN + 0x1200 + (l) * 4)
#define VFN_LEAF_EN_CLR(l)		(VFN + 0x1400 + (l) * 4)
#define VFN_TOP				(VFN + 0x1600)
#define VFN_TOP_EN_SET			(VFN + 0x1608)
#define VFN_TOP_EN_CLR			(VFN + 0x1610)
#define VFN_LEAVES			8

#define GSP_FALCON			0x00110000
#define FALCON_IRQSCLR			0x004
#define FALCON_IRQSTAT			0x008
#define FALCON_RISCV_IRQMASK		(0x1000 + 0x528)
#define FALCON_INTR_RETRIGGER		0x3e8
#define FALCON_IRQ_SWGEN0		BIT(6)

#define NV2080_CTRL_CMD_INTERNAL_INTR_GET_KERNEL_TABLE	0x20800a5c
#define INTR_TABLE_MAX			128
#define MC_ENGINE_IDX_CE0		15	/* r570 nvrm/engine.h */
#define MC_ENGINE_IDX_GSP		50
#define MC_ENGINE_IDX_GR0		84
#define NV_VECTOR_NONE			0xffffffff

struct nv_intr_table {			/* NV2080_CTRL_INTERNAL_INTR_GET_KERNEL_TABLE_PARAMS */
	__le32 len;
	struct { __le16 engine_idx, pad; __le32 pmc_mask, vec_stall, vec_nonstall; } e[INTR_TABLE_MAX];
	u8 subtree_map[14];
	u8 pad[2];
};
static_assert(offsetof(struct nv_intr_table, subtree_map) == 2052);

static void leaf_enable(struct nvdrv_dev *nv, u32 vec)
{
	struct nv_intr *in = &nv->intr;

	if (vec == NV_VECTOR_NONE || vec / 32 >= VFN_LEAVES)
		return;
	in->enabled[vec / 32] |= BIT(vec % 32);
}

static void msg_work(struct work_struct *w)
{
	struct nvdrv_dev *nv = container_of(w, struct nvdrv_dev, intr.msg_work);
	int n;

	mutex_lock(&nv->gsp_lock);
	n = nv->gsp_running ? nv_rpc_drain_events(nv) : 0;
	mutex_unlock(&nv->gsp_lock);
	if (n > 0)
		atomic_add(n, &nv->intr.gsp_events);
}

/*
 * kgspHealthCheck while idle: RPC waits notice a dead GSP themselves, but with interrupts
 * nothing else would. Report once and fail every channel (device lost); recovery needs a
 * reload (an in-place reset would also have to rebuild the display state).
 */
#define GSP_HEALTH_PERIOD	(5 * HZ)
static void health_work(struct work_struct *w)
{
	struct nvdrv_dev *nv = container_of(to_delayed_work(w), struct nvdrv_dev, intr.health_work);
	struct nv_intr *in = &nv->intr;

	if (!READ_ONCE(in->active))
		return;
	if (!in->gsp_dead && READ_ONCE(nv->gsp_running) && !nv_gsp_riscv_active(nv)) {
		in->gsp_dead = true;
		mutex_lock(&nv->gsp_lock);
		nv_gsp_dump_state(nv, "GSP RISC-V stopped while running (health check)");
		mutex_unlock(&nv->gsp_lock);
		nv_uapi_device_lost(nv, "GSP RISC-V stopped (health check)");
	}
	schedule_delayed_work(&in->health_work, GSP_HEALTH_PERIOD);
}

static irqreturn_t nv_intr_handler(int irq, void *data)
{
	struct nvdrv_dev *nv = data;
	struct nv_intr *in = &nv->intr;
	u32 top, leaf, stat, mine, stray;
	bool fence = false, gsp = false;

	top = nv_rd32(nv, VFN_TOP);
	if (!top || top == 0xffffffff)
		return IRQ_NONE;
	nv_wr32(nv, VFN_TOP_EN_CLR, 0xf);	/* unarm */
	for (leaf = 0; leaf < VFN_LEAVES; leaf++) {
		if (!(top & BIT(leaf / 2)))
			continue;
		stat = nv_rd32(nv, VFN_LEAF(leaf));
		if (!stat)
			continue;
		mine = stat & in->enabled[leaf];
		stray = stat & ~in->enabled[leaf];
		if (mine)
			nv_wr32(nv, VFN_LEAF(leaf), mine);	/* acknowledge */
		if (stray) {	/* someone else's source enabled in our tree: keep it from storming */
			nv_wr32(nv, VFN_LEAF_EN_CLR(leaf), stray);
			nv_wr32(nv, VFN_LEAF(leaf), stray);
			dev_warn_ratelimited(&nv->pdev->dev, "intr: masked foreign leaf %u bits 0x%08x\n", leaf, stray);
		}
		if (mine & in->fence_mask[leaf])
			fence = true;
		if (in->gsp_vec != NV_VECTOR_NONE && leaf == in->gsp_vec / 32 && (mine & BIT(in->gsp_vec % 32)))
			gsp = true;
	}
	if (gsp) {
		u32 irq_stat = nv_rd32(nv, GSP_FALCON + FALCON_IRQSTAT) &
			       nv_rd32(nv, GSP_FALCON + FALCON_RISCV_IRQMASK);

		if (irq_stat & FALCON_IRQ_SWGEN0) {
			nv_wr32(nv, GSP_FALCON + FALCON_IRQSCLR, FALCON_IRQ_SWGEN0);
			schedule_work(&in->msg_work);
		}
		if (irq_stat & ~FALCON_IRQ_SWGEN0)
			nv_wr32(nv, GSP_FALCON + FALCON_IRQSCLR, irq_stat & ~FALCON_IRQ_SWGEN0);
		nv_wr32(nv, GSP_FALCON + FALCON_INTR_RETRIGGER, 1);
	}
	if (fence) {
		atomic_inc(&in->fence_irqs);
		nv_uapi_fence_irq(nv);
	}
	nv_wr32(nv, VFN_TOP_EN_SET, 0xf);	/* rearm */
	return IRQ_HANDLED;
}

/* After GSP-RM boot (internal client from the static info). Failure leaves polling in charge. */
int nv_intr_init(struct nvdrv_dev *nv)
{
	struct nv_gsp_static_info *si = &nv->gsp_info;
	struct nv_intr *in = &nv->intr;
	struct nv_intr_table *t;
	u32 i, n, leaf;
	int ret;

	lockdep_assert_held(&nv->gsp_lock);
	if (in->irq > 0)
		return 0;
	memset(in->enabled, 0, sizeof(in->enabled));
	memset(in->fence_mask, 0, sizeof(in->fence_mask));
	in->gsp_vec = NV_VECTOR_NONE;
	t = kzalloc(sizeof(*t), GFP_KERNEL);
	if (!t)
		return -ENOMEM;
	ret = nv_rm_control(nv, si->h_internal_client, si->h_internal_subdevice,
			    NV2080_CTRL_CMD_INTERNAL_INTR_GET_KERNEL_TABLE, t, sizeof(*t));
	if (ret)
		goto out;
	n = min_t(u32, le32_to_cpu(t->len), INTR_TABLE_MAX);
	for (i = 0; i < n; i++) {
		u32 idx = le16_to_cpu(t->e[i].engine_idx), ns = le32_to_cpu(t->e[i].vec_nonstall);
		u32 st = le32_to_cpu(t->e[i].vec_stall);

		dev_dbg(&nv->pdev->dev, "intr: %2u: engine %3u stall 0x%08x nonstall 0x%08x\n", i, idx, st, ns);
		if (idx == MC_ENGINE_IDX_GSP)
			in->gsp_vec = st;
		if ((idx == MC_ENGINE_IDX_GR0 || (idx >= MC_ENGINE_IDX_CE0 && idx < MC_ENGINE_IDX_CE0 + 10)) &&
		    ns != NV_VECTOR_NONE && ns / 32 < VFN_LEAVES) {
			leaf_enable(nv, ns);
			in->fence_mask[ns / 32] |= BIT(ns % 32);
			if (idx == MC_ENGINE_IDX_GR0)
				in->gr_vec = ns;
		}
	}
	leaf_enable(nv, in->gsp_vec);

	ret = pci_alloc_irq_vectors(nv->pdev, 1, 1, PCI_IRQ_MSI | PCI_IRQ_INTX);
	if (ret < 0)
		goto out;
	ret = pci_irq_vector(nv->pdev, 0);
	if (ret < 0)
		goto out_vec;
	INIT_WORK(&in->msg_work, msg_work);
	INIT_DELAYED_WORK(&in->health_work, health_work);
	in->gsp_dead = false;
	in->irq = ret;
	ret = request_irq(in->irq, nv_intr_handler, IRQF_SHARED, "nvdrv", nv);
	if (ret) {
		in->irq = 0;
		goto out_vec;
	}
	for (leaf = 0; leaf < VFN_LEAVES; leaf++) {
		if (!in->enabled[leaf])
			continue;
		nv_wr32(nv, VFN_LEAF(leaf), in->enabled[leaf]);		/* drop stale pending bits */
		nv_wr32(nv, VFN_LEAF_EN_SET(leaf), in->enabled[leaf]);
	}
	nv_wr32(nv, GSP_FALCON + FALCON_IRQSCLR, FALCON_IRQ_SWGEN0);
	nv_wr32(nv, VFN_TOP_EN_SET, 0xf);
	in->active = true;
	schedule_delayed_work(&in->health_work, GSP_HEALTH_PERIOD);
	dev_info(&nv->pdev->dev, "intr: %s irq %d, %u table entries; GSP vector %d, GR nonstall %d, leaves %*ph\n",
		 nv->pdev->msi_enabled ? "MSI" : "INTx", in->irq, n, (int)in->gsp_vec, (int)in->gr_vec,
		 (int)sizeof(in->enabled), in->enabled);
	kfree(t);
	return 0;
out_vec:
	pci_free_irq_vectors(nv->pdev);
out:
	dev_warn(&nv->pdev->dev, "intr: not enabled (%d), fences stay polled\n", ret);
	kfree(t);
	return ret;
}

/* Before the GPU reset: our leaves off, handler gone. */
void nv_intr_fini(struct nvdrv_dev *nv)
{
	struct nv_intr *in = &nv->intr;
	u32 leaf;

	if (in->irq <= 0)
		return;
	in->active = false;
	for (leaf = 0; leaf < VFN_LEAVES; leaf++)
		if (in->enabled[leaf])
			nv_wr32(nv, VFN_LEAF_EN_CLR(leaf), in->enabled[leaf]);
	free_irq(in->irq, nv);
	cancel_work_sync(&in->msg_work);
	cancel_delayed_work_sync(&in->health_work);
	pci_free_irq_vectors(nv->pdev);
	in->irq = 0;
}

void nv_intr_show(struct nvdrv_dev *nv, struct seq_file *m)
{
	struct nv_intr *in = &nv->intr;

	seq_printf(m, "irq %d (%s), active %d, GSP vector %d, GR nonstall %d\n", in->irq,
		   nv->pdev->msi_enabled ? "MSI" : "INTx", in->active, (int)in->gsp_vec, (int)in->gr_vec);
	seq_printf(m, "fence interrupts %d, GSP events drained by interrupt %d, GSP health %s\n",
		   atomic_read(&in->fence_irqs), atomic_read(&in->gsp_events),
		   in->gsp_dead ? "DEAD" : in->active ? "ok (checked every 5 s)" : "-");
	seq_printf(m, "enabled leaves %*ph, top 0x%08x\n", (int)sizeof(in->enabled), in->enabled,
		   in->irq > 0 ? nv_rd32(nv, VFN_TOP) : 0);
}
