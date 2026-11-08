#!/usr/bin/env python3
"""Cross-check for src/libos.c: decode the LibOS init args, GSP_ARGUMENTS_CACHED,
the RPC queue page table and the command-queue header, and verify them against
RM 570.144's rules (kgspSetupLibosInitArgs, kgspPopulateGspRmInitArgs,
_getMsgQueueParams, msgqTxCreate) and against each other.

  python3 -I tools/libos.py DEBUGFS_DIR
"""
import re
import struct
import sys

LOGS = ["LOGINIT", "LOGINTR", "LOGRM", "LOGMNOC", "LOGKRNL"]   # GA10x: LOGIDX_SIZE = 5
CMDQ = STATQ = 0x40000


def id8_str(v):
    return v.to_bytes(8, "big").lstrip(b"\0").decode()


def main(d):
    rd = lambda n: open(f"{d}/{n}", "rb").read()
    info = open(f"{d}/info").read()
    addr = lambda pat: int(re.search(pat + r"\s+(0x[0-9a-f]+)", info).group(1), 16)
    args, ga, pt, cq = rd("libos_args.bin"), rd("gsp_args.bin"), rd("msgq_pt.bin"), rd("cmdq_hdr.bin")
    errs = []
    chk = lambda cond, msg: cond or errs.append(msg)

    # Expected queue geometry, derived independently
    qpages = (CMDQ + STATQ) >> 12
    ptes = qpages + -(-qpages * 8 // 4096)
    pt_size = -(-ptes * 8 // 4096) * 4096
    shm = addr("queue shm:")

    # LibOS init args: logs in order, then RMARGS, then zeros
    print("LibOS init args:")
    names = LOGS + ["RMARGS"]
    for i in range(128):
        id8, pa, size, kind, loc = struct.unpack_from("<QQQBB", args, i * 32)
        if i >= len(names):
            chk(args[i * 32:(i + 1) * 32] == bytes(32), f"arg {i} not zero")
            continue
        name = id8_str(id8)
        print(f"  [{i}] {name:8s} pa 0x{pa:x} size 0x{size:x} kind {kind} loc {loc}")
        chk(name == names[i], f"arg {i}: id {name!r}, expected {names[i]!r}")
        chk((kind, loc) == (1, 1), f"arg {i}: kind/loc {kind}/{loc}, expected contiguous/sysmem")
        if i < len(LOGS):
            chk(pa == addr(f"log {i}:") and size == 0x10000, f"{name}: pa/size mismatch with driver")
        else:
            chk(pa == addr("rm args:") and size == 0x1000, "RMARGS does not point at GSP_ARGUMENTS_CACHED")

    # Log buffers (_setupLogBufferBaremetal): word 0 put pointer, then one PTE per 4K page
    for i, name in enumerate(LOGS):
        hdr = rd(f"log{i}.bin")
        base = addr(f"log {i}:")
        words = struct.unpack_from("<18Q", hdr, 0)
        chk(words[0] == 0, f"{name}: put pointer not 0")
        chk(list(words[1:17]) == [base + p * 4096 for p in range(16)], f"{name}: PTEs wrong")
        chk(words[17] == 0 and hdr[18 * 8:] == bytes(len(hdr) - 18 * 8), f"{name}: data after PTEs")
    print(f"log buffers: {len(LOGS)} x 16 PTEs verified")

    # GSP_ARGUMENTS_CACHED
    smpa, ptec, cqo, sqo = struct.unpack_from("<QI4xQQ", ga, 0)
    old, flags, inpm = struct.unpack_from("<IIB", ga, 32)
    inst, dmem = struct.unpack_from("<IB", ga, 44)
    ppa, psz = struct.unpack_from("<QQ", ga, 56)
    print(f"GSP_ARGUMENTS_CACHED: shm 0x{smpa:x}, {ptec} PTEs, cmdq @0x{cqo:x}, statq @0x{sqo:x}, "
          f"sr {old}/{flags}/{inpm}, gpu {inst}, dmemStack {dmem}, profiler 0x{ppa:x}/{psz}")
    chk(smpa == shm and smpa == struct.unpack_from("<Q", pt, 0)[0], "sharedMemPhysAddr != PTE[0]")
    chk((ptec, cqo, sqo) == (ptes, pt_size, pt_size + CMDQ), "queue geometry mismatch")
    chk((old, flags, inpm, inst, dmem, ppa, psz) == (0, 0, 0, 0, 1, 0, 0), "SR/instance/stack/profiler")
    chk(ga[72:] == bytes(len(ga) - 72), "GSP args page not zero after struct")

    # Page table: every page of the block, itself included
    entries = struct.unpack_from(f"<{ptes}Q", pt, 0)
    chk(all(e == shm + i * 4096 for i, e in enumerate(entries)), "PTEs are not base + i*4K")
    chk(pt[ptes * 8:] == bytes(len(pt) - ptes * 8), "PTE page not zero after entries")
    print(f"PTEs: {ptes} entries, 0x{entries[0]:x} .. 0x{entries[-1]:x}")

    # Command queue msgqTxHeader (msgqTxCreate with GSP's parameters)
    rx_off = -(-32 // 16) * 16
    entry_off = -(-(rx_off + 4) // 4096) * 4096
    got = struct.unpack_from("<8I", cq, 0)
    # writePtr: 0 on a fresh queue, 2 once the pre-boot RPCs are queued (tests/phase6_rpc.sh)
    exp = (0, CMDQ, 0x1000, (CMDQ - entry_off) // 0x1000, got[4] if got[4] in (0, 2) else 0,
           1, rx_off, entry_off)
    print("cmdq header: version {} size 0x{:x} msgSize 0x{:x} msgCount {} writePtr {} flags {} "
          "rxHdrOff 0x{:x} entryOff 0x{:x}".format(*got))
    chk(got == exp, f"cmdq header {got} != {exp}")
    chk(cq[32:] == bytes(len(cq) - 32), "cmdq header page not zero after header (readPtr must be 0)")

    if errs:
        sys.exit("FAIL:\n  " + "\n  ".join(errs))
    print("PASS: LibOS args, GSP args, queue PTEs and cmdq header are consistent with RM's rules")


if __name__ == "__main__":
    main(sys.argv[1])
