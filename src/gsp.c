// SPDX-License-Identifier: GPL-2.0
/*
 * GSP-RM boot (Phase 5/6): kgspBootstrap_TU102 for GA10x, 570.144.
 *
 *   1. queue GSP_SET_SYSTEM_INFO + SET_REGISTRY      (kgspQueueAsyncInitRpcs)
 *   2. FWSEC-FRTS on the GSP falcon                   (WPR2 = FRTS)
 *   3. reset GSP into RISC-V, BCR with BR fetch       (kflcnResetIntoRiscv_GA102)
 *   4. GSP MAILBOX0/1 = LibOS init-args bus address   (kgspProgramLibosBootArgsAddr_TU102)
 *   5. Booter-Load on SEC2, MAILBOX0/1 = WPR meta     (kgspExecuteBooterLoad_TU102)
 *   6. GSP FALCON_OS = bootloader appVersion; RISC-V must be active
 *   7. link the status queue, then service messages until GSP_INIT_DONE
 *      (running CPU sequencer requests on the way)   (kgspWaitForRmInitDone)
 *   8. GET_GSP_STATIC_INFO, the first synchronous RPC (kgspInitRm)
 *   9. allocate nvdrv's root client -> device -> subdevice (rm.c)
 *
 * Writes GPU registers; reached only from a debugfs trigger with allow_hw_writes=1.
 *
 * Unload (kgspUnloadRm_IMPL + kgspTeardown_TU102, normal mode):
 *   1. UNLOADING_GUEST_DRIVER RPC, then MAILBOX0 = PROCESSOR_SUSPENDED
 *   2. FWSEC-SB on the GSP falcon (puts back the pre-OS apps)
 *   3. Booter-Unload on SEC2 (mailboxes 0xff), WPR2 must be gone afterwards
 * Any failure falls back to an FLR.
 */
#include <linux/delay.h>
#include <linux/iopoll.h>
#include <linux/jiffies.h>
#include <linux/pci.h>
#include <linux/unaligned.h>

#include "nvdrv.h"
#include "nv_regs.h"

#define STATQ_LINK_TIMEOUT_MS	4000	/* GspStatusQueueInit */
#define INIT_DONE_TIMEOUT_MS	10000
#define SUSPEND_TIMEOUT_MS	5000

static bool clean_unload = true;
module_param(clean_unload, bool, 0644);
MODULE_PARM_DESC(clean_unload, "Shut GSP-RM down as NVIDIA's driver does at unbind (default on); off: FLR");

/* GSP_SEQ_BUF_OPCODE (arch/nvalloc/common/inc/rmgspseq.h) */
enum {
	SEQ_REG_WRITE, SEQ_REG_MODIFY, SEQ_REG_POLL, SEQ_DELAY_US, SEQ_REG_STORE,
	SEQ_CORE_RESET, SEQ_CORE_START, SEQ_CORE_WAIT_FOR_HALT, SEQ_CORE_RESUME,
};
static const u8 seq_payload_dwords[] = { 2, 3, 5, 1, 2, 0, 0, 0, 0 };

static const char *rpc_event_name(u32 fn)
{
	switch (fn) {
	case NV_RPC_EVENT_GSP_INIT_DONE:	return "GSP_INIT_DONE";
	case NV_RPC_EVENT_RUN_CPU_SEQUENCER:	return "RUN_CPU_SEQUENCER";
	case 0x1003:				return "POST_EVENT";
	case 0x1004:				return "RC_TRIGGERED";
	case NV_RPC_EVENT_OS_ERROR_LOG:		return "OS_ERROR_LOG";
	case 0x100c:				return "UCODE_LIBOS_PRINT";
	case 0x100f:				return "PERF_BRIDGELESS_INFO_UPDATE";
	case 0x101c:				return "GSP_LOCKDOWN_NOTICE";
	case 0x1020:				return "GSP_POST_NOCAT_RECORD";
	case 0x1005:				return "MMU_FAULT_QUEUED";
	default:				return "?";
	}
}

/*
 * kgspExecuteSequencerCommand_GA102 CORE_RESUME: put GSP back into RISC-V mode,
 * re-arm the LibOS args, and restart SEC2 (Booter is still resident there) so
 * it reloads GSP-RM; done when BSI_SECURE_SCRATCH_14.BOOT_STAGE_3_HANDOFF is set.
 */
static int gsp_core_resume(struct nvdrv_dev *nv)
{
	struct device *dev = &nv->pdev->dev;
	u32 v, m0, m1;
	int ret;

	ret = nv_gsp_reset_into_riscv(nv);
	if (ret)
		return ret;
	nv_falcon_write_mailboxes(nv, NV_FALCON_GSP, lower_32_bits(nv->libos.args.dma),
				  upper_32_bits(nv->libos.args.dma));
	dev_info(dev, "sequencer: CORE_RESUME, starting SEC2 to reload GSP-RM\n");
	nv_falcon_start(nv, NV_FALCON_SEC2);
	ret = readl_poll_timeout(nv->bar0 + NV_PGC6_BSI_SECURE_SCRATCH_14, v,
				 v & NV_PGC6_BSI_SS14_BOOT_STAGE_3_HANDOFF, 10, 4000000);
	nv_falcon_read_mailboxes(nv, NV_FALCON_SEC2, &m0, &m1);
	if (ret || m0) {
		dev_err(dev, "sequencer: SEC2 did not resume GSP-RM (scratch14 0x%08x, SEC2 mbox0 0x%x)\n",
			v, m0);
		return -ETIMEDOUT;
	}
	nv_gsp_write_os(nv, nv->gsp_fw.bl_desc.appVersion);
	if (!nv_gsp_riscv_active(nv)) {
		dev_err(dev, "sequencer: GSP RISC-V not active after resume\n");
		return -EIO;
	}
	dev_info(dev, "sequencer: GSP-RM resumed\n");
	return 0;
}

static bool reg_ok(struct nvdrv_dev *nv, u32 addr)
{
	return !(addr & 3) && addr < nv->bar0_len;
}

/* kgspExecuteSequencerBuffer: rpc_run_cpu_sequencer_v17_00 */
static int gsp_run_sequencer(struct nvdrv_dev *nv, void *payload, u32 len)
{
	struct device *dev = &nv->pdev->dev;
	__le32 *p = payload;
	u32 size_dw, end, i = 0, *cmd, *save;
	int ret = 0, ops = 0;

	if (len < 40)
		return -EINVAL;
	size_dw = le32_to_cpu(p[0]);
	end = le32_to_cpu(p[1]);
	save = (u32 *)&p[2];			/* regSaveArea[8] */
	cmd = (u32 *)&p[10];			/* commandBuffer[] */
	if (!size_dw || end >= size_dw || 40 + (u64)end * 4 > len)
		return -EINVAL;

	while (i < end && !ret) {
		u32 op = le32_to_cpu(cmd[i++]), a, m, v, t, val;

		if (op >= ARRAY_SIZE(seq_payload_dwords) || i + seq_payload_dwords[op] > end)
			return -EINVAL;
		ops++;
		switch (op) {
		case SEQ_REG_WRITE:
			a = le32_to_cpu(cmd[i]); v = le32_to_cpu(cmd[i + 1]);
			if (!reg_ok(nv, a))
				return -EINVAL;
			nv_wr32(nv, a, v);
			break;
		case SEQ_REG_MODIFY:
			a = le32_to_cpu(cmd[i]); m = le32_to_cpu(cmd[i + 1]); v = le32_to_cpu(cmd[i + 2]);
			if (!reg_ok(nv, a))
				return -EINVAL;
			nv_wr32(nv, a, (nv_rd32(nv, a) & ~m) | v);
			break;
		case SEQ_REG_POLL:
			a = le32_to_cpu(cmd[i]); m = le32_to_cpu(cmd[i + 1]);
			v = le32_to_cpu(cmd[i + 2]); t = le32_to_cpu(cmd[i + 3]);
			if (!reg_ok(nv, a))
				return -EINVAL;
			ret = readl_poll_timeout(nv->bar0 + a, val, (val & m) == v, 1, t ?: 4000000);
			if (ret)
				dev_err(dev, "sequencer: poll 0x%x timed out (0x%08x), err code 0x%x\n",
					a, val, le32_to_cpu(cmd[i + 4]));
			break;
		case SEQ_DELAY_US:
			fsleep(le32_to_cpu(cmd[i]));
			break;
		case SEQ_REG_STORE:
			a = le32_to_cpu(cmd[i]); v = le32_to_cpu(cmd[i + 1]);
			if (!reg_ok(nv, a) || v >= 8)
				return -EINVAL;
			save[v] = cpu_to_le32(nv_rd32(nv, a));
			break;
		case SEQ_CORE_RESET:
			ret = nv_gsp_core_reset(nv);
			break;
		case SEQ_CORE_START:
			nv_gsp_core_start(nv);
			break;
		case SEQ_CORE_WAIT_FOR_HALT:
			ret = nv_gsp_core_wait_halt(nv);
			break;
		case SEQ_CORE_RESUME:
			ret = gsp_core_resume(nv);
			break;
		default:
			dev_err(dev, "sequencer: opcode %u not supported\n", op);
			return -EOPNOTSUPP;
		}
		i += seq_payload_dwords[op];
	}
	dev_info(dev, "sequencer: %d ops executed%s\n", ops, ret ? " (FAILED)" : "");
	return ret;
}

void nv_gsp_dump_state(struct nvdrv_dev *nv, const char *why)
{
	u32 m0, m1, s0, s1;
	int i;

	nv_falcon_read_mailboxes(nv, NV_FALCON_GSP, &m0, &m1);
	nv_falcon_read_mailboxes(nv, NV_FALCON_SEC2, &s0, &s1);
	dev_err(&nv->pdev->dev, "gsp: %s: riscv %s, GSP mbox 0x%08x/0x%08x, SEC2 mbox 0x%08x/0x%08x, WPR2 0x%llx-0x%llx\n",
		why, nv_gsp_riscv_active(nv) ? "active" : "inactive", m0, m1, s0, s1,
		nv_wpr2_lo(nv), nv_wpr2_hi(nv));
	for (i = 0; i < NV_LIBOS_NUM_LOGS; i++)
		dev_err(&nv->pdev->dev, "gsp: log %d put pointer %llu\n", i,
			le64_to_cpu(*(__le64 *)nv->libos.logs[i].cpu));
}

/*
 * A message from GSP that nobody is waiting for (_kgspProcessRpcEvent). Returns
 * nonzero only for failures that make GSP-RM unusable (a failed sequencer run).
 */
int nv_gsp_handle_event(struct nvdrv_dev *nv, u32 fn, struct nv_rpc_header *h, void *payload, u32 len)
{
	struct device *dev = &nv->pdev->dev;

	dev_info(dev, "gsp: <- %s (0x%x), %u bytes, result 0x%x\n", rpc_event_name(fn), fn,
		 len, le32_to_cpu(h->rpc_result));
	switch (fn) {
	case NV_RPC_EVENT_RUN_CPU_SEQUENCER:
		return gsp_run_sequencer(nv, payload, len);
	case 0x1003:	/* POST_EVENT: rpc_post_event_v17_00, event data at 29 */
		if (len >= 29) {
			const u8 *p = payload;
			u32 size = get_unaligned_le32(p + 24);

			if (29 + size <= len)
				nv_disp_event(nv, get_unaligned_le32(p + 4), p + 29, size);
		}
		break;
	case 0x1020:	/* GSP_POST_NOCAT_RECORD: NV2080CtrlNocatJournalInsertRecord */
		if (len >= 180) {
			static const char *const types[] = { "unknown", "bugcheck", "engine", "tdr", "rc",
							     "assert" };
			const u8 *r = payload;
			u8 type = r[16];

			nv->gsp_nocat++;
			dev_info_ratelimited(dev, "gsp nocat: %s, source \"%.*s\", engine \"%.*s\", error 0x%llx, bugcheck 0x%x\n",
					     type < ARRAY_SIZE(types) ? types[type] : "?",
					     (int)strnlen(r + 24, 65), r + 24, (int)strnlen(r + 104, 65), r + 104,
					     get_unaligned_le64(r + 96), get_unaligned_le32(r + 20));
		}
		break;
	case 0x1004:	/* RC_TRIGGERED: rpc_rc_triggered_v17_02 */
		if (len >= 48) {
			const __le32 *w = payload;

			dev_err(dev, "gsp: RC on engine 0x%x chid %u: exceptLevel %u exceptType %u (robust-channel code) scope %u, MMU fault addr 0x%08x%08x type %u\n",
				le32_to_cpu(w[0]), le32_to_cpu(w[1]), le32_to_cpu(w[3]), le32_to_cpu(w[4]),
				le32_to_cpu(w[5]), le32_to_cpu(w[8]), le32_to_cpu(w[7]), le32_to_cpu(w[9]));
		}
		break;
	case NV_RPC_EVENT_OS_ERROR_LOG:
		/* rpc_os_error_log_v17_00: exceptType, runlistId, chid, errString[0x100] */
		if (len >= 12 + 1)
			dev_err(dev, "gsp: error log: %.*s\n", (int)min(len - 12, 0x100u),
				(char *)payload + 12);
		break;
	default:
		break;
	}
	return 0;
}

static int gsp_wait_init_done(struct nvdrv_dev *nv)
{
	struct nv_rpc_header *h;
	void *payload;
	u32 len, result;
	int ret;

	ret = nv_rpc_wait(nv, NV_RPC_EVENT_GSP_INIT_DONE, INIT_DONE_TIMEOUT_MS, &h, &payload, &len);
	if (ret)
		return ret;
	result = le32_to_cpu(h->rpc_result);
	dev_info(&nv->pdev->dev, "gsp: <- GSP_INIT_DONE, result 0x%x\n", result);
	return result ? -EIO : 0;
}

static int gsp_boot_locked(struct nvdrv_dev *nv)
{
	struct device *dev = &nv->pdev->dev;
	struct nv_gsp_fw *g = &nv->gsp_fw;
	u32 m0, m1;
	unsigned long deadline;
	int ret;

	if (!nv_hw_writes_allowed())
		return -EPERM;
	if (!nv->wpr_meta || !nv->libos.shm.cpu || !nv->fwsec_frts || !g->booter_load.image)
		return -ENODATA;
	if (nv->gsp_touched) {
		dev_err(dev, "gsp: already booted (or attempted) on this bind; reset first\n");
		return -EBUSY;
	}
	if (nv_wpr2_hi(nv)) {
		dev_err(dev, "gsp: WPR2 already up; run scripts/reset.sh first\n");
		return -EBUSY;
	}

	/* 1. fresh queues/logs, then the async init RPCs GSP-RM reads on startup */
	nv_libos_reset(nv);
	ret = nv_rpc_init(nv);
	if (!ret)
		ret = nv_rpc_queue_preboot(nv);
	if (ret) {
		dev_err(dev, "gsp: queueing pre-boot RPCs failed: %d\n", ret);
		return ret;
	}
	dev_info(dev, "gsp: queued GSP_SET_SYSTEM_INFO + SET_REGISTRY (cmdq writePtr %u)\n",
		 nv->rpc.tx_write);

	/*
	 * Sysmem flush buffer first (kmemsysInitFlushSysmemBuffer_GA100): without it
	 * the FB hub's flush writes go to bus address 0.
	 */
	nv->gsp_touched = true;
	pci_set_master(nv->pdev);
	nv_wr32(nv, NV_PFB_NISO_FLUSH_SYSMEM_ADDR_HI,
		upper_32_bits(nv->libos.flush.dma >> NV_PFB_NISO_FLUSH_SYSMEM_ADDR_SHIFT) & 0x7f);
	nv_wr32(nv, NV_PFB_NISO_FLUSH_SYSMEM_ADDR,
		lower_32_bits(nv->libos.flush.dma >> NV_PFB_NISO_FLUSH_SYSMEM_ADDR_SHIFT));
	dev_info(dev, "gsp: sysmem flush buffer %pad programmed\n", &nv->libos.flush.dma);

	/* 2. FWSEC-FRTS */
	ret = nv_fwsec_run_frts(nv);
	if (ret)
		return ret;

	/* 3 + 4. GSP into RISC-V mode, LibOS args in its mailboxes */
	ret = nv_gsp_reset_into_riscv(nv);
	if (ret)
		return ret;
	nv_falcon_write_mailboxes(nv, NV_FALCON_GSP, lower_32_bits(nv->libos.args.dma),
				  upper_32_bits(nv->libos.args.dma));

	/* 5. Booter-Load on SEC2 with the WPR meta address */
	m0 = lower_32_bits(nv->wpr_meta_dma);
	m1 = upper_32_bits(nv->wpr_meta_dma);
	dev_info(dev, "gsp: running Booter-Load on SEC2 (WPR meta %pad)\n", &nv->wpr_meta_dma);
	ret = nv_falcon_hs_exec(nv, NV_FALCON_SEC2, &g->booter_load, &m0, &m1);
	if (ret)
		return ret;
	if (m0) {
		nv_gsp_dump_state(nv, "Booter-Load failed");
		return -EIO;
	}

	/* 6. FALCON_OS, then the RISC-V core must be running */
	nv_gsp_write_os(nv, g->bl_desc.appVersion);
	if (!nv_gsp_riscv_active(nv)) {
		nv_gsp_dump_state(nv, "GSP RISC-V not active after Booter");
		return -EIO;
	}
	dev_info(dev, "gsp: Booter OK, GSP RISC-V running, WPR2 0x%llx-0x%llx\n",
		 nv_wpr2_lo(nv), nv_wpr2_hi(nv));

	/* 7. link the status queue (GSP-RM creates its TX header), wait for INIT_DONE */
	deadline = jiffies + msecs_to_jiffies(STATQ_LINK_TIMEOUT_MS);
	while ((ret = nv_rpc_link_statq(nv)) == -EAGAIN) {
		if (time_after(jiffies, deadline) || !nv_gsp_riscv_active(nv)) {
			nv_gsp_dump_state(nv, "status queue never appeared");
			return -ETIMEDOUT;
		}
		usleep_range(1000, 2000);
	}
	if (ret)
		return ret;
	dev_info(dev, "gsp: status queue linked (%u entries); cmdq read by GSP up to %u\n",
		 nv->rpc.rx_count, le32_to_cpu(*(__le32 *)(nv->libos.shm.cpu + nv->libos.pt_size +
							   nv->libos.cmdq_size + nv->rpc.rx_rxhdr_off)));

	ret = gsp_wait_init_done(nv);
	if (ret)
		return ret;
	nv->gsp_running = true;
	dev_info(dev, "gsp: GSP-RM " NV_GSP_FW_VERSION " initialized\n");

	/* 8. first synchronous RPC: what GSP-RM found out about the GPU (kgspInitRm) */
	ret = nv_rm_get_static_info(nv);
	if (ret)
		return ret;

	/* 9. our own client/device/subdevice inside GSP-RM (Phase 7) */
	ret = nv_rm_init_objects(nv);
	if (!ret)
		nv_intr_init(nv);	/* optional: without it fences are polled */
	return ret;
}

int nv_gsp_boot(struct nvdrv_dev *nv)
{
	int ret;

	mutex_lock(&nv->gsp_lock);
	ret = gsp_boot_locked(nv);
	mutex_unlock(&nv->gsp_lock);
	return ret;
}

/* rpc_unloading_guest_driver_v1F_07 */
struct nv_rpc_unloading_guest_driver {
	u8 in_pm_transition;
	u8 gc6_entering;
	u8 pad[2];
	__le32 new_level;	/* NV2080_CTRL_GPU_SET_POWER_STATE_GPU_LEVEL_0 */
};
static_assert(sizeof(struct nv_rpc_unloading_guest_driver) == 8);

/*
 * Normal unload. Returns 0 only if GSP-RM stopped, FWSEC-SB ran and Booter-Unload
 * took WPR2 down; then nothing on the GPU references our memory any more.
 */
static int gsp_unload_locked(struct nvdrv_dev *nv)
{
	struct device *dev = &nv->pdev->dev;
	struct nv_rpc_unloading_guest_driver req = {};
	u32 m0 = 0xff, m1 = 0xff;
	ktime_t t0 = ktime_get();
	int ret;

	if (!nv->gsp_running || nv->intr.gsp_dead || !nv_gsp_riscv_active(nv))
		return -ENODEV;
	if (!nv->fwsec_sb || !nv->gsp_fw.booter_unload.image)
		return -ENODATA;

	ret = nv_rpc_call(nv, NV_RPC_UNLOADING_GUEST_DRIVER, &req, sizeof(req), &req, 0);
	nv->gsp_running = false;	/* no RPC from here on, whatever happens */
	if (ret) {
		dev_err(dev, "gsp: UNLOADING_GUEST_DRIVER failed: %d\n", ret);
		return ret;
	}
	ret = nv_gsp_wait_suspended(nv, SUSPEND_TIMEOUT_MS);
	if (ret)
		return ret;

	ret = nv_fwsec_run_sb(nv);
	if (ret) {
		dev_err(dev, "gsp: FWSEC-SB failed: %d\n", ret);
		return ret;
	}

	if (!nv_wpr2_hi(nv)) {	/* kgspExecuteBooterUnloadIfNeeded: nothing to tear down */
		dev_warn(dev, "gsp: WPR2 already down before Booter-Unload\n");
		return 0;
	}
	ret = nv_falcon_hs_exec(nv, NV_FALCON_SEC2, &nv->gsp_fw.booter_unload, &m0, &m1);
	if (ret)
		return ret;
	if (m0) {
		dev_err(dev, "gsp: Booter-Unload failed, mailbox0 0x%x\n", m0);
		return -EIO;
	}
	if (nv_wpr2_hi(nv)) {
		dev_err(dev, "gsp: WPR2 still up after Booter-Unload (0x%llx-0x%llx)\n",
			nv_wpr2_lo(nv), nv_wpr2_hi(nv));
		return -EIO;
	}
	dev_info(dev, "gsp: GSP-RM unloaded, WPR2 down (%lld ms)\n",
		 ktime_ms_delta(ktime_get(), t0));
	return 0;
}

/*
 * Called from .remove (device lock held): once Booter has run, GSP may keep
 * DMAing into our queues and logs. Shut GSP-RM down properly (the GPU then
 * goes back to its post-VBIOS state, ready for any driver); if that is off or
 * fails, stop it with an FLR before that memory is freed.
 */
void nv_gsp_fini(struct nvdrv_dev *nv)
{
	int ret = -EPERM;

	if (!nv->gsp_touched)
		return;
	nv_intr_fini(nv);		/* interrupt handler and health check off first */
	mutex_lock(&nv->gsp_lock);
	nv_rm_fini_objects(nv);		/* frees our client: channels and display objects go with it */
	if (clean_unload && nv_hw_writes_allowed())
		ret = gsp_unload_locked(nv);
	nv->gsp_running = false;
	mutex_unlock(&nv->gsp_lock);
	if (ret) {
		dev_info(&nv->pdev->dev, "gsp: resetting GPU (FLR) before freeing GSP-shared memory (%s %d)\n",
			 clean_unload ? "clean unload failed:" : "clean_unload=0", ret);
		if (pci_reset_function_locked(nv->pdev))
			dev_err(&nv->pdev->dev, "gsp: FLR failed; GSP may still access freed memory\n");
	}
	pci_clear_master(nv->pdev);	/* belt and braces: no DMA into what is freed below */
	nv_evo_fini(nv);	/* push buffers: only after the unload/FLR stopped the display engine */
	nv_fifo_fini(nv);	/* GPFIFO/push buffers: likewise */
	nv_uapi_fini(nv);
	nv_mem_fini(nv);
	nv->gsp_touched = nv->gsp_running = false;
	nv->gsp_info.valid = false;
}
