// SPDX-License-Identifier: GPL-2.0
/*
 * GSP RPC transport (Phase 6) over the queues set up in libos.c.
 *
 * Mirrors open-gpu-kernel-modules 570.144:
 *   message_queue_priv.h   GSP_MSG_QUEUE_ELEMENT, _checkSum32
 *   message_queue_cpu.c    GspMsgQueueSendCommand / ReceiveStatus / GspStatusQueueInit
 *   common/shared/msgq     msgqTx*, msgqRx* (both sides set MSGQ_FLAGS_SWAP_RX)
 *   rpc_common.c, rpc.c    rpcWriteCommonHeader, GSP_SET_SYSTEM_INFO, SET_REGISTRY,
 *                          _issueRpcAndWait
 *   kernel_gsp.c           _kgspRpcSendMessage (doorbell), _kgspRpcRecvPoll / _kgspRpcDrainEvents
 *
 * Read pointers are "swapped": our read position in the status queue is stored
 * in the RX header inside the *command* queue, and GSP's read position in the
 * command queue is stored in the RX header inside the *status* queue.
 */
#include <linux/delay.h>
#include <linux/jiffies.h>
#include <linux/lockdep.h>
#include <linux/ioport.h>
#include <linux/pci.h>
#include <linux/sched.h>
#include <linux/slab.h>
#include <linux/stddef.h>
#include <linux/unaligned.h>
#include <linux/vgaarb.h>

#include "nvdrv.h"
#include "nv_regs.h"

#define MSGQ_ELEM_SIZE		SZ_4K
#define MSGQ_ELEM_MAX		16		/* GSP_MSG_QUEUE_ELEMENT_SIZE_MAX / MIN */
#define MSGQ_TX_WRITE_PTR	16		/* msgqTxHeader.writePtr */
#define MSGQ_VERSION		0

#define RPC_HEADER_VERSION	0x03000000	/* MAJOR 3, MINOR 0 */
#define RPC_SIGNATURE		0x43505256	/* "VRPC" */
#define RPC_RESULT_PENDING	0xffffffff

struct gsp_msg_element {
	u8 auth_tag[16];
	u8 aad[16];
	__le32 checksum;
	__le32 seq;
	__le32 elem_count;
	__le32 pad;
	struct nv_rpc_header rpc;
	u8 data[];
};
static_assert(offsetof(struct gsp_msg_element, rpc) == 48);
static_assert(sizeof(struct nv_rpc_header) == 32);

#define ELEM_HDR_SIZE		offsetof(struct gsp_msg_element, rpc)

/* ---- queue geometry ---------------------------------------------------- */

static u8 *cmdq(struct nv_libos *l)  { return l->shm.cpu + l->pt_size; }
static u8 *statq(struct nv_libos *l) { return l->shm.cpu + l->pt_size + l->cmdq_size; }

static u32 rd32(const void *p) { return le32_to_cpu(READ_ONCE(*(const __le32 *)p)); }
static void wr32(void *p, u32 v) { WRITE_ONCE(*(__le32 *)p, cpu_to_le32(v)); }

/* _checkSum32: XOR of 64-bit words, folded to 32 bits. len is padded to 8. */
static u32 checksum32(const void *data, u32 len)
{
	const u64 *p = data, *end = data + ALIGN(len, 8);
	u64 sum = 0;

	while (p < end)
		sum ^= le64_to_cpu(*p++);
	return lower_32_bits(sum) ^ upper_32_bits(sum);
}

int nv_rpc_init(struct nvdrv_dev *nv)
{
	struct nv_rpc *r = &nv->rpc;
	struct nv_libos *l = &nv->libos;

	if (!l->shm.cpu)
		return -ENODATA;
	if (!r->stage) {
		r->stage = kvzalloc(MSGQ_ELEM_SIZE * MSGQ_ELEM_MAX, GFP_KERNEL);
		if (!r->stage)
			return -ENOMEM;
	}
	r->tx_count = rd32(cmdq(l) + 12);		/* msgCount */
	r->tx_entry_off = rd32(cmdq(l) + 28);		/* entryOff */
	r->tx_rxhdr_off = rd32(cmdq(l) + 24);		/* rxHdrOff */
	r->tx_write = 0;
	r->tx_seq = 0;
	r->rx_linked = false;
	r->rx_read = 0;
	r->rx_seq = 0;
	return 0;
}

void nv_rpc_fini(struct nvdrv_dev *nv)
{
	kvfree(nv->rpc.stage);
	nv->rpc.stage = NULL;
}

/* ---- send (GspMsgQueueSendCommand + msgqTxSubmitBuffers) --------------- */

static u32 tx_free(struct nvdrv_dev *nv)
{
	struct nv_rpc *r = &nv->rpc;
	struct nv_libos *l = &nv->libos;
	u32 rp;

	/* msgqTxGetFreeSpace: before rx link it reads our own RX header. */
	rp = r->rx_linked ? rd32(statq(l) + r->rx_rxhdr_off) : rd32(cmdq(l) + r->tx_rxhdr_off);
	if (rp >= r->tx_count)
		return 0;
	return (rp + r->tx_count - r->tx_write - 1) % r->tx_count;
}

int nv_rpc_send(struct nvdrv_dev *nv, u32 function, const void *payload, u32 len)
{
	struct nv_rpc *r = &nv->rpc;
	struct nv_libos *l = &nv->libos;
	struct gsp_msg_element *e = r->stage;
	u32 msg_len = ELEM_HDR_SIZE + sizeof(e->rpc) + len, n, i;

	if (!e || msg_len > MSGQ_ELEM_SIZE * MSGQ_ELEM_MAX)
		return -EINVAL;
	n = DIV_ROUND_UP(msg_len, MSGQ_ELEM_SIZE);
	if (n > tx_free(nv))
		return -EBUSY;

	memset(e, 0, n * MSGQ_ELEM_SIZE);
	/* rpcWriteCommonHeader */
	e->rpc.header_version = cpu_to_le32(RPC_HEADER_VERSION);
	e->rpc.signature = cpu_to_le32(RPC_SIGNATURE);
	e->rpc.length = cpu_to_le32(sizeof(e->rpc) + len);
	e->rpc.function = cpu_to_le32(function);
	e->rpc.rpc_result = cpu_to_le32(RPC_RESULT_PENDING);
	e->rpc.rpc_result_private = cpu_to_le32(RPC_RESULT_PENDING);
	memcpy(e->data, payload, len);

	e->seq = cpu_to_le32(r->tx_seq);
	e->elem_count = cpu_to_le32(n);
	e->checksum = 0;
	e->checksum = cpu_to_le32(checksum32(e, msg_len));

	/* One element at a time: the ring may wrap in the middle of a message. */
	for (i = 0; i < n; i++) {
		u32 slot = (r->tx_write + i) % r->tx_count;

		memcpy(cmdq(l) + r->tx_entry_off + slot * MSGQ_ELEM_SIZE,
		       (u8 *)e + i * MSGQ_ELEM_SIZE, MSGQ_ELEM_SIZE);
	}
	wmb();	/* portAtomicMemoryFenceStore before publishing writePtr */
	r->tx_write = (r->tx_write + n) % r->tx_count;
	wr32(cmdq(l) + MSGQ_TX_WRITE_PTR, r->tx_write);
	wmb();
	r->tx_seq++;
	return 0;
}

/* ---- receive (GspStatusQueueInit, GspMsgQueueReceiveStatus) ------------ */

/* msgqRxLink: validate GSP's TX header in the status queue. 0 = linked, -EAGAIN = not yet. */
int nv_rpc_link_statq(struct nvdrv_dev *nv)
{
	struct nv_rpc *r = &nv->rpc;
	struct nv_libos *l = &nv->libos;
	const u8 *h = statq(l);
	u32 size, msg_size, msg_count, version, rx_off, entry_off, flags;

	rmb();
	version = rd32(h + 0);
	size = rd32(h + 4);
	msg_size = rd32(h + 8);
	msg_count = rd32(h + 12);
	flags = rd32(h + 20);
	rx_off = rd32(h + 24);
	entry_off = rd32(h + 28);

	if (size != l->statq_size || msg_size != MSGQ_ELEM_SIZE || version != MSGQ_VERSION)
		return -EAGAIN;
	if (rx_off < 32 || entry_off < r->tx_rxhdr_off + 4 ||
	    msg_count != (size - entry_off) / msg_size)
		return -EAGAIN;
	if (!(flags & 1))	/* we rely on swapped read pointers (SWAP_RX on both sides) */
		return -EPROTO;

	r->rx_count = msg_count;
	r->rx_entry_off = entry_off;
	r->rx_rxhdr_off = rx_off;
	r->rx_read = 0;
	wr32(cmdq(l) + r->tx_rxhdr_off, 0);	/* pReadOutgoing = our RX header */
	wmb();
	r->rx_linked = true;
	return 0;
}

/*
 * Copy the next complete message into nv->rpc.stage. Returns 0, -EAGAIN if
 * nothing is pending, or -EIO on a corrupt message (which is consumed).
 */
int nv_rpc_recv(struct nvdrv_dev *nv, struct nv_rpc_header **hdr, void **payload, u32 *len)
{
	struct nv_rpc *r = &nv->rpc;
	struct nv_libos *l = &nv->libos;
	struct gsp_msg_element *e = r->stage;
	u32 wp, avail, n, i, rpc_len;
	int ret = 0;

	if (!r->rx_linked)
		return -ENOTCONN;
	rmb();
	wp = rd32(statq(l) + MSGQ_TX_WRITE_PTR);
	if (wp >= r->rx_count)
		return -EIO;
	avail = (wp + r->rx_count - r->rx_read) % r->rx_count;
	if (!avail)
		return -EAGAIN;
	rmb();

	memcpy(e, statq(l) + r->rx_entry_off + r->rx_read * MSGQ_ELEM_SIZE, MSGQ_ELEM_SIZE);
	n = le32_to_cpu(e->elem_count);
	if (!n || n > MSGQ_ELEM_MAX || n > avail)
		return n > avail && n <= MSGQ_ELEM_MAX ? -EAGAIN : -EIO;
	for (i = 1; i < n; i++) {
		u32 slot = (r->rx_read + i) % r->rx_count;

		memcpy((u8 *)e + i * MSGQ_ELEM_SIZE,
		       statq(l) + r->rx_entry_off + slot * MSGQ_ELEM_SIZE, MSGQ_ELEM_SIZE);
	}

	rpc_len = le32_to_cpu(e->rpc.length);
	if (rpc_len < sizeof(e->rpc) || ELEM_HDR_SIZE + rpc_len > n * MSGQ_ELEM_SIZE) {
		dev_err(&nv->pdev->dev, "rpc: bad message length %u\n", rpc_len);
		ret = -EIO;
	} else if (checksum32(e, ELEM_HDR_SIZE + rpc_len)) {
		dev_err(&nv->pdev->dev, "rpc: bad checksum (seq %u)\n", le32_to_cpu(e->seq));
		ret = -EIO;
	} else if (le32_to_cpu(e->seq) != r->rx_seq) {
		dev_err(&nv->pdev->dev, "rpc: sequence %u, expected %u\n", le32_to_cpu(e->seq), r->rx_seq);
		ret = -EIO;
	}

	/* msgqRxMarkConsumed: our read pointer lives in the command queue's RX header. */
	r->rx_read = (r->rx_read + n) % r->rx_count;
	mb();
	wr32(cmdq(l) + r->tx_rxhdr_off, r->rx_read);
	wmb();
	r->rx_seq = le32_to_cpu(e->seq) + 1;
	if (ret)
		return ret;

	*hdr = &e->rpc;
	*payload = e->data;
	*len = rpc_len - sizeof(e->rpc);
	return 0;
}

/* ---- synchronous calls (_issueRpcAndWait) ------------------------------- */

/*
 * _kgspRpcRecvPoll: service GSP's messages until one with function == fn arrives.
 * Everything else (async events, sequencer requests) is handled on the way. The
 * matching message is left in nv->rpc.stage. Replies are matched on function only,
 * like RM: there is at most one synchronous RPC outstanding (gsp_lock).
 */
int nv_rpc_wait(struct nvdrv_dev *nv, u32 fn, unsigned int timeout_ms,
		struct nv_rpc_header **hdr, void **payload, u32 *len)
{
	unsigned long deadline = jiffies + msecs_to_jiffies(timeout_ms);
	unsigned int delay_us = 20;	/* backs off to 1 ms: quick replies, cheap long waits */
	bool timed_out;
	int ret;

	for (;;) {
		/* Sample the deadline first so a reply that lands meanwhile still counts. */
		timed_out = time_after(jiffies, deadline);
		ret = nv_rpc_recv(nv, hdr, payload, len);
		if (ret == -EAGAIN) {
			if (!nv_gsp_riscv_active(nv)) {	/* kgspHealthCheck */
				nv_gsp_dump_state(nv, "GSP RISC-V stopped");
				nv_uapi_device_lost(nv, "GSP RISC-V stopped during an RPC");
				return -EIO;
			}
			if (timed_out) {
				dev_err(&nv->pdev->dev, "rpc: timeout waiting for function 0x%x\n", fn);
				nv_gsp_dump_state(nv, "rpc timeout");
				return -ETIMEDOUT;
			}
			usleep_range(delay_us, delay_us * 2);
			delay_us = min(delay_us * 2, 1000u);
			continue;
		}
		if (ret)
			return ret;
		if (le32_to_cpu((*hdr)->function) == fn)
			return 0;
		ret = nv_gsp_handle_event(nv, le32_to_cpu((*hdr)->function), *hdr, *payload, *len);
		if (ret)
			return ret;
	}
}

/* 1.5 x RM's default GPU timeout (4 s), as _kgspRpcRecvPoll allows. */
#define RPC_CALL_TIMEOUT_MS	6000

/*
 * Send fn with req as payload, ring the doorbell, wait for the reply and copy its
 * first reply_len payload bytes out. Returns -EREMOTEIO when GSP-RM reports a
 * failure in rpc_result (kept in nv->rpc.last_result). The reply is copied out
 * even then, because RM reads the RM-level status from it (rpcRmApiAlloc_GSP).
 */
/*
 * Handle every pending GSP message as an asynchronous event (GSP message-queue interrupt).
 * Caller holds gsp_lock, so no synchronous RPC is waiting for a reply meanwhile.
 */
int nv_rpc_drain_events(struct nvdrv_dev *nv)
{
	struct nv_rpc_header *hdr;
	void *payload;
	u32 len;
	int ret, n = 0;

	lockdep_assert_held(&nv->gsp_lock);
	while (!(ret = nv_rpc_recv(nv, &hdr, &payload, &len))) {
		ret = nv_gsp_handle_event(nv, le32_to_cpu(hdr->function), hdr, payload, len);
		n++;
		if (ret)
			return ret;
	}
	return ret == -EAGAIN ? n : ret;
}

/*
 * RPCs larger than one queue message (16 elements): _issueRpcAndWaitLarge / nouveau
 * r535_gsp_rpc_push. The first message carries the function and as much payload as fits,
 * the rest follows in CONTINUATION_RECORD messages; then the reply is awaited once (a long
 * reply comes back the same way; its continuation records are drained as events).
 */
#define NV_RPC_CONTINUATION_RECORD	71

int nv_rpc_call_large(struct nvdrv_dev *nv, u32 fn, const void *req, u32 req_len)
{
	const u32 max = MSGQ_ELEM_SIZE * MSGQ_ELEM_MAX - ELEM_HDR_SIZE - sizeof(struct nv_rpc_header);
	struct nv_rpc_header *h;
	unsigned long until;
	u32 off = 0, len;
	void *p;
	int ret;

	lockdep_assert_held(&nv->gsp_lock);
	if (req_len <= max)
		return nv_rpc_call(nv, fn, req, req_len, NULL, 0);
	if (!nv->gsp_running)
		return -ENODEV;
	while (off < req_len) {
		u32 n = min(req_len - off, max);

		until = jiffies + msecs_to_jiffies(2000);
		while ((ret = nv_rpc_send(nv, off ? NV_RPC_CONTINUATION_RECORD : fn, req + off, n)) == -EBUSY) {
			nv_wr32(nv, NV_PGSP_QUEUE_HEAD(NV_PGSP_RPC_QUEUE_IDX), 0);	/* let GSP drain */
			if (time_after(jiffies, until))
				break;
			usleep_range(100, 200);
		}
		if (ret)
			return ret;
		off += n;
	}
	nv_wr32(nv, NV_PGSP_QUEUE_HEAD(NV_PGSP_RPC_QUEUE_IDX), 0);
	ret = nv_rpc_wait(nv, fn, RPC_CALL_TIMEOUT_MS, &h, &p, &len);
	if (ret)
		return ret;
	nv->rpc.last_result = le32_to_cpu(h->rpc_result);
	return nv->rpc.last_result ? -EREMOTEIO : 0;
}

static int rpc_call(struct nvdrv_dev *nv, u32 fn, const void *req, u32 req_len,
		    void *reply, u32 reply_len, unsigned int timeout_ms);

/* For RPCs that do a lot of work on GSP (the suspend unload saves its VRAM first). */
int nv_rpc_call_timeout(struct nvdrv_dev *nv, u32 fn, const void *req, u32 req_len, unsigned int timeout_ms)
{
	return rpc_call(nv, fn, req, req_len, NULL, 0, timeout_ms);
}

int nv_rpc_call(struct nvdrv_dev *nv, u32 fn, const void *req, u32 req_len,
		void *reply, u32 reply_len)
{
	return rpc_call(nv, fn, req, req_len, reply, reply_len, RPC_CALL_TIMEOUT_MS);
}

static int rpc_call(struct nvdrv_dev *nv, u32 fn, const void *req, u32 req_len,
		    void *reply, u32 reply_len, unsigned int timeout_ms)
{
	struct device *dev = &nv->pdev->dev;
	struct nv_rpc_header *h;
	void *p;
	u32 len, result;
	int ret;

	lockdep_assert_held(&nv->gsp_lock);
	if (!nv->gsp_running)
		return -ENODEV;

	ret = nv_rpc_send(nv, fn, req, req_len);
	if (ret) {
		dev_err(dev, "rpc: sending function %u failed: %d\n", fn, ret);
		return ret;
	}
	/* Register write: doorbell. GSP-RM ignores the value (RM writes 0). */
	nv_wr32(nv, NV_PGSP_QUEUE_HEAD(NV_PGSP_RPC_QUEUE_IDX), 0);

	ret = nv_rpc_wait(nv, fn, timeout_ms, &h, &p, &len);
	if (ret)
		return ret;
	result = le32_to_cpu(h->rpc_result);
	nv->rpc.last_result = result;
	nv->rpc.last_result_private = le32_to_cpu(h->rpc_result_private);
	if (reply)
		memcpy(reply, p, min(len, reply_len));
	if (result) {
		dev_dbg(dev, "rpc: function %u: result 0x%x private 0x%x\n", fn, result,
			nv->rpc.last_result_private);
		return -EREMOTEIO;
	}
	if (len < reply_len) {
		dev_err(dev, "rpc: function %u reply has %u bytes, expected %u\n", fn, len, reply_len);
		return -EPROTO;
	}
	return 0;
}

/* ---- pre-boot RPCs (kgspQueueAsyncInitRpcs) ----------------------------- */

/*
 * GspSystemInfo (inc/kernel/gpu/gsp/gsp_static_config.h, 570.144), 928 bytes.
 * Offsets taken from compiling NVIDIA's header (see tools/gsp_sysinfo_probe.sh).
 */
struct gsp_system_info {
	__le64 gpu_phys_addr;
	__le64 gpu_phys_fb_addr;
	__le64 gpu_phys_inst_addr;
	__le64 gpu_phys_io_addr;
	__le64 nv_domain_bus_device_func;
	__le64 sim_access_buf_phys_addr;
	__le64 notify_op_shared_surface_phys_addr;
	__le64 pcie_atomics_op_mask;
	__le64 console_mem_size;
	__le64 max_user_va;
	__le32 pci_config_mirror_base;
	__le32 pci_config_mirror_size;
	__le32 pci_device_id;
	__le32 pci_sub_device_id;
	__le32 pci_revision_id;
	__le32 pcie_atomics_cpl_device_cap_mask;
	u8 oor_arch;
	u8 pad0[7];
	__le64 cl_pdb_properties;
	__le32 chipset;
	u8 gpu_behind_bridge;
	u8 flr_supported;
	u8 b64b_bar0_supported;
	u8 mnoc_available;
	__le32 chipset_l1ss_enable;
	u8 upstream_l0s_unsupported;
	u8 upstream_l1_unsupported;
	u8 upstream_l1_por_supported;
	u8 upstream_l1_por_mobile_only;
	u8 system_has_mux;
	u8 upstream_address_valid;
	u8 fhb_bus_info[10];		/* BUSINFO */
	u8 chipset_id_info[10];		/* BUSINFO */
	u8 pad1[2];
	u8 acpi_method_data[676];	/* ACPI_METHOD_DATA, bValid = 0 */
	__le32 hypervisor_type;
	u8 is_passthru;
	u8 pad2[7];
	__le64 sys_timer_offset_ns;
	u8 gsp_vf_info[40];		/* GSP_VF_INFO */
	u8 is_primary;
	u8 is_grid_build;
	u8 pad3[2];
	__le32 pcie_link_cap;		/* GSP_PCIE_CONFIG_REG.linkCap */
	__le32 grid_build_csp;
	u8 preserve_video_memory_allocations;
	u8 tdr_event_supported;
	u8 feature_stretch_vblank_capable;
	u8 enable_dynamic_granularity_page_arrays;
	u8 clock_boost_supported;
	u8 route_disp_intrs_to_cpu;
	u8 pad4[6];
	__le64 host_page_size;
} __packed;
static_assert(sizeof(struct gsp_system_info) == 928);
static_assert(offsetof(struct gsp_system_info, pci_config_mirror_base) == 80);
static_assert(offsetof(struct gsp_system_info, oor_arch) == 104);
static_assert(offsetof(struct gsp_system_info, cl_pdb_properties) == 112);
static_assert(offsetof(struct gsp_system_info, chipset_l1ss_enable) == 128);
static_assert(offsetof(struct gsp_system_info, fhb_bus_info) == 138);
static_assert(offsetof(struct gsp_system_info, acpi_method_data) == 160);
static_assert(offsetof(struct gsp_system_info, hypervisor_type) == 836);
static_assert(offsetof(struct gsp_system_info, sys_timer_offset_ns) == 848);
static_assert(offsetof(struct gsp_system_info, gsp_vf_info) == 856);
static_assert(offsetof(struct gsp_system_info, is_primary) == 896);
static_assert(offsetof(struct gsp_system_info, pcie_link_cap) == 900);
static_assert(offsetof(struct gsp_system_info, preserve_video_memory_allocations) == 908);
static_assert(offsetof(struct gsp_system_info, host_page_size) == 920);

#define OS_HYPERVISOR_UNKNOWN	4	/* bare metal: hypervisorGetHypervisorType() */
#define NV_PCFG_BASE		0x88000	/* DEVICE_BASE(NV_PCFG): kbifGetPciConfigSpacePriMirror_GM107 */
#define NV_PCFG_SIZE		0x1000

/* osinit.c: console size = the boot framebuffer inside BAR1, 64K aligned. */
static u64 boot_console_size(struct pci_dev *pdev)
{
	struct resource *bar1 = &pdev->resource[1], *c;
	u64 size = 0;

	for (c = bar1->child; c; c = c->sibling)
		if (c->start == bar1->start)
			size = resource_size(c);
	if (!size && pdev == vga_default_device())
		size = 0x40000;	/* legacy VGA text console */
	return ALIGN(size, SZ_64K);
}

static int rpc_set_system_info(struct nvdrv_dev *nv)
{
	struct pci_dev *pdev = nv->pdev;
	struct gsp_system_info *si;
	u32 id, subsys, rev, lnkcap = 0, devcap = 0;
	int ret;

	si = kzalloc(sizeof(*si), GFP_KERNEL);
	if (!si)
		return -ENOMEM;

	pci_read_config_dword(pdev, PCI_VENDOR_ID, &id);		/* NV_XVE_ID */
	pci_read_config_dword(pdev, PCI_SUBSYSTEM_VENDOR_ID, &subsys);	/* NV_XVE_SUBSYSTEM */
	pci_read_config_dword(pdev, PCI_CLASS_REVISION, &rev);		/* NV_XVE_REV_ID */
	pcie_capability_read_dword(pdev, PCI_EXP_LNKCAP, &lnkcap);
	pcie_capability_read_dword(pdev, PCI_EXP_DEVCAP, &devcap);

	/* osinit.c attach args: BAR0 regs, BAR1 FB, BAR3 instance memory; no I/O BAR */
	si->gpu_phys_addr = cpu_to_le64(pci_resource_start(pdev, 0));
	si->gpu_phys_fb_addr = cpu_to_le64(pci_resource_start(pdev, 1));
	si->gpu_phys_inst_addr = cpu_to_le64(pci_resource_start(pdev, 3));
	si->nv_domain_bus_device_func = cpu_to_le64(((u64)pci_domain_nr(pdev->bus) << 32) |
						    (pdev->bus->number << 8) | PCI_SLOT(pdev->devfn));
	si->notify_op_shared_surface_phys_addr = cpu_to_le64(nv->libos.notify_op.dma);
	si->console_mem_size = cpu_to_le64(boot_console_size(pdev));
	si->max_user_va = cpu_to_le64(TASK_SIZE);
	si->pci_config_mirror_base = cpu_to_le32(NV_PCFG_BASE);
	si->pci_config_mirror_size = cpu_to_le32(NV_PCFG_SIZE);
	si->pci_device_id = cpu_to_le32(id);
	si->pci_sub_device_id = cpu_to_le32(subsys);
	si->pci_revision_id = cpu_to_le32(rev & 0xff);	/* class code masked off */
	si->oor_arch = 0;				/* OOR_ARCH_X86_64 */
	si->flr_supported = !!(devcap & PCI_EXP_DEVCAP_FLR);
	si->b64b_bar0_supported = !!(pci_resource_flags(pdev, 0) & IORESOURCE_MEM_64);
	si->hypervisor_type = cpu_to_le32(OS_HYPERVISOR_UNKNOWN);
	si->is_primary = pdev == vga_default_device();
	si->pcie_link_cap = cpu_to_le32(lnkcap);
	si->host_page_size = cpu_to_le64(PAGE_SIZE);
	/*
	 * Left 0 on purpose: sim access buffer (silicon), chipset/CL properties (GSP then treats PCIe config as
	 * not accessible and skips ASPM/link tuning), ACPI data (bValid = 0), VF
	 * info (no SR-IOV), atomics masks, MNOC, clock boost (FSP-only).
	 */

	ret = nv_rpc_send(nv, NV_RPC_GSP_SET_SYSTEM_INFO, si, sizeof(*si));
	kfree(si);
	return ret;
}

/*
 * RmPackageRegistry: {u32 size, u32 numEntries, entries[], strings}. Entries are
 * {u32 nameOffset, u8 type, u32 data, u32 length} with natural alignment (16 bytes).
 * These are the per-GPU defaults the Linux driver always sets
 * (initUnixSpecificRegistry); an empty table would make RM's SET_REGISTRY fail.
 */
static const struct {
	const char *name;
	u32 value;
} nv_registry[] = {
	{ "RMSecBusResetEnable", 1 },
	{ "RMForcePcieConfigSave", 1 },
};

static int rpc_set_registry(struct nvdrv_dev *nv)
{
	u32 n = ARRAY_SIZE(nv_registry), size = 8 + 16 * n, off, i;
	u8 *t;
	int ret;

	for (i = 0; i < n; i++)
		size += strlen(nv_registry[i].name) + 1;
	t = kzalloc(size, GFP_KERNEL);
	if (!t)
		return -ENOMEM;
	put_unaligned_le32(size, t);
	put_unaligned_le32(n, t + 4);
	off = 8 + 16 * n;
	for (i = 0; i < n; i++) {
		u8 *ent = t + 8 + 16 * i;
		u32 slen = strlen(nv_registry[i].name) + 1;

		put_unaligned_le32(off, ent + 0);	/* nameOffset */
		ent[4] = 1;				/* REGISTRY_TABLE_ENTRY_TYPE_DWORD */
		put_unaligned_le32(nv_registry[i].value, ent + 8);
		put_unaligned_le32(4, ent + 12);	/* length */
		memcpy(t + off, nv_registry[i].name, slen);
		off += slen;
	}
	ret = nv_rpc_send(nv, NV_RPC_SET_REGISTRY, t, size);
	kfree(t);
	return ret;
}

int nv_rpc_queue_preboot(struct nvdrv_dev *nv)
{
	int ret = rpc_set_system_info(nv);

	if (!ret)
		ret = rpc_set_registry(nv);
	return ret;
}
