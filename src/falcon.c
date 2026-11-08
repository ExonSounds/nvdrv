// SPDX-License-Identifier: GPL-2.0
/*
 * Falcon engine core (Phase 3) and FWSEC-FRTS execution (Phase 5, step 1).
 *
 * Port of the GA102+ path in open-gpu-kernel-modules 570.144:
 *   kernel_falcon_tu102.c / kernel_falcon_ga102.c   reset, core select, start, halt
 *   kernel_gsp_falcon_ga102.c                       kgspExecuteHsFalcon_GA102 (DMA + BROM)
 *   kernel_gsp_frts_tu102.c                         kgspExecuteFwsec_TU102 (result checks)
 *
 * THIS FILE WRITES GPU REGISTERS AND PERFORMS DMA. Nothing here runs at probe;
 * it is reached only through the debugfs "run_fwsec_frts" trigger, and only
 * when the module was loaded with allow_hw_writes=1.
 */
#include <linux/bitfield.h>
#include <linux/delay.h>
#include <linux/dma-mapping.h>
#include <linux/iopoll.h>

#include "nvdrv.h"
#include "nv_regs.h"

/* Falcon v4 registers (ampere/ga102/dev_falcon_v4*.h), relative to falcon base */
#define FLCN_IRQSTAT			0x008
#define FLCN_MAILBOX0			0x040
#define FLCN_MAILBOX1			0x044
#define FLCN_RM				0x084
#define FLCN_HWCFG2			0x0f4
#define   FLCN_HWCFG2_RISCV		BIT(10)
#define   FLCN_HWCFG2_MEM_SCRUBBING	BIT(12)		/* 0 = done */
#define   FLCN_HWCFG2_RESET_READY	BIT(31)
#define FLCN_CPUCTL			0x100
#define   FLCN_CPUCTL_STARTCPU		BIT(1)
#define   FLCN_CPUCTL_HALTED		BIT(4)
#define   FLCN_CPUCTL_ALIAS_EN		BIT(6)
#define FLCN_BOOTVEC			0x104
#define FLCN_DMACTL			0x10c
#define FLCN_DMATRFBASE			0x110
#define FLCN_DMATRFMOFFS		0x114
#define FLCN_DMATRFCMD			0x118
#define   FLCN_DMATRFCMD_FULL		BIT(0)
#define   FLCN_DMATRFCMD_IDLE		BIT(1)
#define   FLCN_DMATRFCMD_SEC		GENMASK(3, 2)
#define   FLCN_DMATRFCMD_IMEM		BIT(4)
#define   FLCN_DMATRFCMD_WRITE		BIT(5)
#define   FLCN_DMATRFCMD_SIZE		GENMASK(10, 8)
#define   FLCN_DMATRFCMD_SIZE_256B	6
#define   FLCN_DMATRFCMD_CTXDMA		GENMASK(14, 12)
#define FLCN_DMATRFFBOFFS		0x11c
#define FLCN_DMATRFBASE1		0x128
#define FLCN_CPUCTL_ALIAS		0x130
#define FLCN_BLK			256

/* FBIF (ampere/ga102/dev_fbif_v4.h), relative to fbif base */
#define FBIF_TRANSCFG(i)		(0x000 + (i) * 4)
#define   FBIF_TRANSCFG_TARGET		GENMASK(1, 0)
#define   FBIF_TRANSCFG_TARGET_COHERENT_SYSMEM 1
#define   FBIF_TRANSCFG_MEM_TYPE_PHYSICAL BIT(2)
#define FBIF_CTL			0x024
#define   FBIF_CTL_ALLOW_PHYS_NO_CTX	BIT(7)

/* Falcon2 / RISC-V (dev_falcon_second_pri.h, dev_riscv_pri.h), relative to riscv base */
#define FLCN2_MOD_SEL			0x180
#define   FLCN2_MOD_SEL_RSA3K		1
#define FLCN2_BROM_CURR_UCODE_ID	0x198
#define FLCN2_BROM_ENGIDMASK		0x19c
#define FLCN2_BROM_PARAADDR(i)		(0x210 + (i) * 4)
#define RISCV_CPUCTL			0x388
#define   RISCV_CPUCTL_ACTIVE_STAT	BIT(7)
#define RISCV_BCR_CTRL			0x668
#define   RISCV_BCR_CTRL_VALID		BIT(0)
#define   RISCV_BCR_CTRL_CORE_SELECT_RISCV BIT(4)
#define   RISCV_BCR_CTRL_BRFETCH	BIT(8)
#define FLCN_OS				0x080

#define RESET_PROPAGATION_READS		10

struct nv_falcon {
	const char *name;
	u32 base;		/* NV_PGSP */
	u32 riscv_base;		/* NV_FALCON2_GSP_BASE */
	u32 fbif_base;		/* NV_PGSP_FBIF_BASE */
	u32 engine_reg;		/* NV_PGSP_FALCON_ENGINE, bit 0 = reset */
};

/* GA102 bases: NV_PGSP / NV_FALCON2_GSP_BASE / NV_PGSP_FBIF_BASE / NV_PGSP_FALCON_ENGINE */
static const struct nv_falcon nv_gsp_falcon = {
	.name = "gsp",
	.base = 0x110000,
	.riscv_base = 0x111000,
	.fbif_base = 0x110600,
	.engine_reg = 0x1103c0,
};

/* GA102 bases: NV_PSEC / NV_FALCON2_SEC_BASE / NV_PSEC_FBIF_BASE / NV_PSEC_FALCON_ENGINE */
static const struct nv_falcon nv_sec2_falcon = {
	.name = "sec2",
	.base = 0x840000,
	.riscv_base = 0x841000,
	.fbif_base = 0x840600,
	.engine_reg = 0x8403c0,
};

static const struct nv_falcon *nv_falcon_get(enum nv_falcon_id id)
{
	return id == NV_FALCON_SEC2 ? &nv_sec2_falcon : &nv_gsp_falcon;
}

static bool allow_hw_writes;
module_param(allow_hw_writes, bool, 0444);
MODULE_PARM_DESC(allow_hw_writes, "Permit debugfs triggers that write GPU registers / start falcons (default: off)");

bool nv_hw_writes_allowed(void)
{
	return allow_hw_writes;
}

static u32 flcn_rd(struct nvdrv_dev *nv, const struct nv_falcon *f, u32 r)
{
	return nv_rd32(nv, f->base + r);
}

static void flcn_wr(struct nvdrv_dev *nv, const struct nv_falcon *f, u32 r, u32 v)
{
	nv_wr32(nv, f->base + r, v);
}

static u32 riscv_rd(struct nvdrv_dev *nv, const struct nv_falcon *f, u32 r)
{
	return nv_rd32(nv, f->riscv_base + r);
}

static void riscv_wr(struct nvdrv_dev *nv, const struct nv_falcon *f, u32 r, u32 v)
{
	nv_wr32(nv, f->riscv_base + r, v);
}

/* Poll a falcon-relative register until (val & mask) == want. */
static int flcn_poll(struct nvdrv_dev *nv, const struct nv_falcon *f, u32 reg,
		     u32 mask, u32 want, u32 timeout_us, const char *what)
{
	u32 val;
	int ret = readl_poll_timeout(nv->bar0 + f->base + reg, val, (val & mask) == want,
				     1, timeout_us);

	if (ret)
		dev_err(&nv->pdev->dev, "%s falcon: timeout waiting for %s (reg 0x%x = 0x%08x)\n",
			f->name, what, f->base + reg, val);
	return ret;
}

/* ---- reset ------------------------------------------------------------ */

static int falcon_reset(struct nvdrv_dev *nv, const struct nv_falcon *f)
{
	u32 bcr;
	int i, ret;

	/* kflcnPreResetWait_GA102: RESET_READY may never assert (bug 3419321); 150us, no error. */
	readl_poll_timeout(nv->bar0 + f->base + FLCN_HWCFG2, bcr,
			   bcr & FLCN_HWCFG2_RESET_READY, 1, 150);

	/* kgspResetHw_TU102 */
	nv_wr32(nv, f->engine_reg, nv_rd32(nv, f->engine_reg) | BIT(0));
	for (i = 0; i < RESET_PROPAGATION_READS; i++)
		nv_rd32(nv, f->engine_reg);
	nv_wr32(nv, f->engine_reg, nv_rd32(nv, f->engine_reg) & ~BIT(0));
	for (i = 0; i < RESET_PROPAGATION_READS; i++)
		nv_rd32(nv, f->engine_reg);

	/* kflcnWaitForResetToFinish_GA102 */
	ret = flcn_poll(nv, f, FLCN_HWCFG2, FLCN_HWCFG2_MEM_SCRUBBING, 0, 1000000, "mem scrubbing");
	if (ret)
		return ret;

	/* kflcnSwitchToFalcon_GA102 */
	if (flcn_rd(nv, f, FLCN_HWCFG2) & FLCN_HWCFG2_RISCV) {
		bcr = riscv_rd(nv, f, RISCV_BCR_CTRL);
		if (bcr & RISCV_BCR_CTRL_CORE_SELECT_RISCV) {
			riscv_wr(nv, f, RISCV_BCR_CTRL, 0);	/* CORE_SELECT_FALCON */
			ret = readl_poll_timeout(nv->bar0 + f->riscv_base + RISCV_BCR_CTRL, bcr,
						 bcr & RISCV_BCR_CTRL_VALID, 1, 1000000);
			if (ret) {
				dev_err(&nv->pdev->dev, "%s falcon: core switch to falcon failed (BCR 0x%08x)\n",
					f->name, bcr);
				return ret;
			}
		}
	}

	flcn_wr(nv, f, FLCN_RM, nv->boot0);	/* pGpu->chipId0 */
	return 0;
}

/* ---- HS ucode execution ------------------------------------------------ */

static int falcon_dma(struct nvdrv_dev *nv, const struct nv_falcon *f, u32 dest,
		      u32 mem_off, dma_addr_t src, u32 size, u32 cmd)
{
	u32 done;
	int ret;

	ret = flcn_poll(nv, f, FLCN_DMATRFCMD, FLCN_DMATRFCMD_FULL, 0, 1000000, "DMA queue");
	if (ret)
		return ret;
	flcn_wr(nv, f, FLCN_DMATRFBASE, lower_32_bits(src >> 8));
	flcn_wr(nv, f, FLCN_DMATRFBASE1, upper_32_bits(src >> 8) & 0x1ff);

	for (done = 0; done < size; done += FLCN_BLK) {
		ret = flcn_poll(nv, f, FLCN_DMATRFCMD, FLCN_DMATRFCMD_FULL, 0, 1000000, "DMA queue");
		if (ret)
			return ret;
		flcn_wr(nv, f, FLCN_DMATRFMOFFS, dest + done);
		flcn_wr(nv, f, FLCN_DMATRFFBOFFS, mem_off + done);
		flcn_wr(nv, f, FLCN_DMATRFCMD, cmd);
	}
	/* GA10x has no TCM tagging: wait for idle before the next transfer. */
	return flcn_poll(nv, f, FLCN_DMATRFCMD, FLCN_DMATRFCMD_IDLE, FLCN_DMATRFCMD_IDLE,
			 1000000, "DMA idle");
}

/*
 * kgspExecuteHsFalcon_GA102 for "boot from HS" ucode (FWSEC, Booter) whose
 * image sits at @img in coherent system memory. Caller resets the falcon first.
 */
static int falcon_run_hs(struct nvdrv_dev *nv, const struct nv_falcon *f,
			 const struct nv_hs_ucode *u, dma_addr_t img, u32 *mbox0, u32 *mbox1)
{
	u32 cmd, v;
	int ret;

	if ((u->imem_size | u->dmem_size) & (FLCN_BLK - 1))
		return -EINVAL;

	/* kflcnDisableCtxReq_TU102 */
	nv_wr32(nv, f->fbif_base + FBIF_CTL,
		nv_rd32(nv, f->fbif_base + FBIF_CTL) | FBIF_CTL_ALLOW_PHYS_NO_CTX);
	flcn_wr(nv, f, FLCN_DMACTL, 0);

	/* ctxdma 0: physical, coherent sysmem */
	v = nv_rd32(nv, f->fbif_base + FBIF_TRANSCFG(0));
	v &= ~(FBIF_TRANSCFG_TARGET | FBIF_TRANSCFG_MEM_TYPE_PHYSICAL);
	v |= FIELD_PREP(FBIF_TRANSCFG_TARGET, FBIF_TRANSCFG_TARGET_COHERENT_SYSMEM) |
	     FBIF_TRANSCFG_MEM_TYPE_PHYSICAL;
	nv_wr32(nv, f->fbif_base + FBIF_TRANSCFG(0), v);

	cmd = FIELD_PREP(FLCN_DMATRFCMD_SIZE, FLCN_DMATRFCMD_SIZE_256B) |
	      FIELD_PREP(FLCN_DMATRFCMD_CTXDMA, 0);

	/* IMEM, secure */
	ret = falcon_dma(nv, f, u->imem_pa, u->imem_va, img + u->code_offset - u->imem_va,
			 u->imem_size, cmd | FLCN_DMATRFCMD_IMEM | FIELD_PREP(FLCN_DMATRFCMD_SEC, 1));
	if (ret)
		return ret;

	/* DMEM (desc v3: no DMEM VA, so mem offset 0 and no DMTAG) */
	ret = falcon_dma(nv, f, u->dmem_pa, 0, img + u->dmem_offset, u->dmem_size, cmd);
	if (ret)
		return ret;

	/* BROM: where the signature is, which engine, which ucode, RSA-3K */
	riscv_wr(nv, f, FLCN2_BROM_PARAADDR(0), u->pkc_data_offset);
	riscv_wr(nv, f, FLCN2_BROM_ENGIDMASK, u->engine_id_mask);
	riscv_wr(nv, f, FLCN2_BROM_CURR_UCODE_ID, u->ucode_id);
	riscv_wr(nv, f, FLCN2_MOD_SEL, FLCN2_MOD_SEL_RSA3K);

	flcn_wr(nv, f, FLCN_BOOTVEC, u->imem_va);
	if (mbox0)
		flcn_wr(nv, f, FLCN_MAILBOX0, *mbox0);
	if (mbox1)
		flcn_wr(nv, f, FLCN_MAILBOX1, *mbox1);

	/* kflcnStartCpu_TU102 */
	if (flcn_rd(nv, f, FLCN_CPUCTL) & FLCN_CPUCTL_ALIAS_EN)
		flcn_wr(nv, f, FLCN_CPUCTL_ALIAS, FLCN_CPUCTL_STARTCPU);
	else
		flcn_wr(nv, f, FLCN_CPUCTL, FLCN_CPUCTL_STARTCPU);

	ret = flcn_poll(nv, f, FLCN_CPUCTL, FLCN_CPUCTL_HALTED, FLCN_CPUCTL_HALTED,
			4000000, "halt");

	if (mbox0)
		*mbox0 = flcn_rd(nv, f, FLCN_MAILBOX0);
	if (mbox1)
		*mbox1 = flcn_rd(nv, f, FLCN_MAILBOX1);
	return ret;
}

/* ---- generic HS execution -------------------------------------------- */

/*
 * Reset @id, copy @u into coherent sysmem, run it as HS ucode and wait for
 * halt (kflcnReset + kgspExecuteHsFalcon_GA102). Mailboxes in/out.
 */
int nv_falcon_hs_exec(struct nvdrv_dev *nv, enum nv_falcon_id id,
		      const struct nv_hs_ucode *u, u32 *mbox0, u32 *mbox1)
{
	const struct nv_falcon *f = nv_falcon_get(id);
	struct device *dev = &nv->pdev->dev;
	dma_addr_t img_dma;
	void *img;
	int ret;

	if (!allow_hw_writes)
		return -EPERM;
	pci_set_master(nv->pdev);

	img = dma_alloc_coherent(dev, u->image_size, &img_dma, GFP_KERNEL);
	if (!img)
		return -ENOMEM;
	memcpy(img, u->image, u->image_size);

	ret = falcon_reset(nv, f);
	if (!ret)
		ret = falcon_run_hs(nv, f, u, img_dma, mbox0, mbox1);
	dev_info(dev, "%s falcon: HS ucode %u %s, mailbox0 0x%08x mailbox1 0x%08x\n", f->name,
		 u->ucode_id, ret ? "did NOT halt" : "halted", mbox0 ? *mbox0 : 0, mbox1 ? *mbox1 : 0);
	dma_free_coherent(dev, u->image_size, img, img_dma);
	return ret;
}

/* ---- GSP RISC-V helpers (kgspBootstrap_TU102 + sequencer core ops) ---- */

/* kflcnResetIntoRiscv_GA102: reset, then select the RISC-V core with BR fetch. */
int nv_gsp_reset_into_riscv(struct nvdrv_dev *nv)
{
	const struct nv_falcon *f = &nv_gsp_falcon;
	u32 v;
	int i, ret;

	readl_poll_timeout(nv->bar0 + f->base + FLCN_HWCFG2, v, v & FLCN_HWCFG2_RESET_READY, 1, 150);
	nv_wr32(nv, f->engine_reg, nv_rd32(nv, f->engine_reg) | BIT(0));
	for (i = 0; i < RESET_PROPAGATION_READS; i++)
		nv_rd32(nv, f->engine_reg);
	nv_wr32(nv, f->engine_reg, nv_rd32(nv, f->engine_reg) & ~BIT(0));
	for (i = 0; i < RESET_PROPAGATION_READS; i++)
		nv_rd32(nv, f->engine_reg);
	ret = flcn_poll(nv, f, FLCN_HWCFG2, FLCN_HWCFG2_MEM_SCRUBBING, 0, 1000000, "mem scrubbing");
	if (ret)
		return ret;
	riscv_wr(nv, f, RISCV_BCR_CTRL, RISCV_BCR_CTRL_CORE_SELECT_RISCV |
		 RISCV_BCR_CTRL_VALID | RISCV_BCR_CTRL_BRFETCH);
	return 0;
}

void nv_falcon_write_mailboxes(struct nvdrv_dev *nv, enum nv_falcon_id id, u32 m0, u32 m1)
{
	flcn_wr(nv, nv_falcon_get(id), FLCN_MAILBOX0, m0);
	flcn_wr(nv, nv_falcon_get(id), FLCN_MAILBOX1, m1);
}

void nv_falcon_read_mailboxes(struct nvdrv_dev *nv, enum nv_falcon_id id, u32 *m0, u32 *m1)
{
	*m0 = flcn_rd(nv, nv_falcon_get(id), FLCN_MAILBOX0);
	*m1 = flcn_rd(nv, nv_falcon_get(id), FLCN_MAILBOX1);
}

void nv_gsp_write_os(struct nvdrv_dev *nv, u32 val)
{
	flcn_wr(nv, &nv_gsp_falcon, FLCN_OS, val);
}

bool nv_gsp_riscv_active(struct nvdrv_dev *nv)
{
	return riscv_rd(nv, &nv_gsp_falcon, RISCV_CPUCTL) & RISCV_CPUCTL_ACTIVE_STAT;
}

/* Sequencer CORE_RESET: kflcnReset + kflcnDisableCtxReq on the GSP falcon. */
int nv_gsp_core_reset(struct nvdrv_dev *nv)
{
	const struct nv_falcon *f = &nv_gsp_falcon;
	int ret = falcon_reset(nv, f);

	if (ret)
		return ret;
	nv_wr32(nv, f->fbif_base + FBIF_CTL,
		nv_rd32(nv, f->fbif_base + FBIF_CTL) | FBIF_CTL_ALLOW_PHYS_NO_CTX);
	flcn_wr(nv, f, FLCN_DMACTL, 0);
	return 0;
}

void nv_gsp_core_start(struct nvdrv_dev *nv)
{
	nv_falcon_start(nv, NV_FALCON_GSP);
}

/* kflcnStartCpu_TU102 */
void nv_falcon_start(struct nvdrv_dev *nv, enum nv_falcon_id id)
{
	const struct nv_falcon *f = nv_falcon_get(id);

	if (flcn_rd(nv, f, FLCN_CPUCTL) & FLCN_CPUCTL_ALIAS_EN)
		flcn_wr(nv, f, FLCN_CPUCTL_ALIAS, FLCN_CPUCTL_STARTCPU);
	else
		flcn_wr(nv, f, FLCN_CPUCTL, FLCN_CPUCTL_STARTCPU);
}

int nv_gsp_core_wait_halt(struct nvdrv_dev *nv)
{
	return flcn_poll(nv, &nv_gsp_falcon, FLCN_CPUCTL, FLCN_CPUCTL_HALTED,
			 FLCN_CPUCTL_HALTED, 4000000, "halt");
}

/* ---- FWSEC-FRTS -------------------------------------------------------- */

int nv_fwsec_run_frts(struct nvdrv_dev *nv)
{
	struct device *dev = &nv->pdev->dev;
	struct nv_fwsec *fw = &nv->fwsec;
	struct nv_hs_ucode u = {
		.image = nv->fwsec_frts, .image_size = fw->image_size,
		.code_offset = 0, .imem_size = fw->imem_size,
		.imem_pa = fw->imem_pa, .imem_va = fw->imem_va,
		.dmem_offset = fw->dmem_offset, .dmem_size = fw->dmem_size, .dmem_pa = fw->dmem_pa,
		.pkc_data_offset = fw->pkc_data_offset,
		.engine_id_mask = fw->engine_id_mask, .ucode_id = fw->ucode_id,
	};
	u32 mbox0 = 0, mbox1 = 0, err;
	int ret;

	if (!allow_hw_writes)
		return -EPERM;
	if (!nv->fwsec_frts || !nv->fb.frts_offset)
		return -ENODATA;

	/* _kgspBootGspRm: never run with WPR2 already up. */
	if (nv_wpr2_hi(nv)) {
		dev_err(dev, "fwsec-frts: WPR2 already up (0x%llx-0x%llx); GPU needs a reset\n",
			nv_wpr2_lo(nv), nv_wpr2_hi(nv));
		return -EBUSY;
	}

	ret = nv_falcon_hs_exec(nv, NV_FALCON_GSP, &u, &mbox0, &mbox1);
	if (ret)
		return ret;

	/* kgspExecuteFwsec_TU102 result checks */
	err = FIELD_GET(NV_VBIOS_FWSECLIC_FRTS_ERR_CODE,
			nv_rd32(nv, NV_PBUS_VBIOS_SCRATCH(NV_VBIOS_FWSECLIC_SCRATCH_INDEX_0E)));
	if (err) {
		dev_err(dev, "fwsec-frts: FRTS error code 0x%x\n", err);
		return -EIO;
	}
	if (!nv_wpr2_hi(nv)) {
		dev_err(dev, "fwsec-frts: no WPR2 after FWSEC\n");
		return -EIO;
	}
	if (nv_wpr2_lo(nv) != nv->fb.frts_offset) {
		dev_err(dev, "fwsec-frts: WPR2 at 0x%llx, expected 0x%llx\n",
			nv_wpr2_lo(nv), nv->fb.frts_offset);
		return -EIO;
	}
	dev_info(dev, "fwsec-frts: SUCCESS, WPR2 0x%llx-0x%llx\n", nv_wpr2_lo(nv), nv_wpr2_hi(nv));
	return 0;
}

/* Read-only snapshot of the GSP and SEC2 falcons for debugfs. */
void nv_falcon_dump(struct nvdrv_dev *nv, struct seq_file *m)
{
	const struct nv_falcon *fs[] = { &nv_gsp_falcon, &nv_sec2_falcon };
	int i;

	for (i = 0; i < ARRAY_SIZE(fs); i++) {
		const struct nv_falcon *f = fs[i];

		seq_printf(m, "falcon %s @0x%x:\n", f->name, f->base);
		seq_printf(m, "  engine (reset):  0x%08x\n", nv_rd32(nv, f->engine_reg));
		seq_printf(m, "  hwcfg2:          0x%08x\n", flcn_rd(nv, f, FLCN_HWCFG2));
		seq_printf(m, "  cpuctl:          0x%08x\n", flcn_rd(nv, f, FLCN_CPUCTL));
		seq_printf(m, "  irqstat:         0x%08x\n", flcn_rd(nv, f, FLCN_IRQSTAT));
		seq_printf(m, "  mailbox0/1:      0x%08x 0x%08x\n", flcn_rd(nv, f, FLCN_MAILBOX0),
			   flcn_rd(nv, f, FLCN_MAILBOX1));
		seq_printf(m, "  os:              0x%08x\n", flcn_rd(nv, f, FLCN_OS));
		seq_printf(m, "  riscv bcr_ctrl:  0x%08x\n", riscv_rd(nv, f, RISCV_BCR_CTRL));
		seq_printf(m, "  riscv cpuctl:    0x%08x\n", riscv_rd(nv, f, RISCV_CPUCTL));
		seq_printf(m, "  fbif transcfg0:  0x%08x  ctl 0x%08x\n",
			   nv_rd32(nv, f->fbif_base + FBIF_TRANSCFG(0)), nv_rd32(nv, f->fbif_base + FBIF_CTL));
	}
}
