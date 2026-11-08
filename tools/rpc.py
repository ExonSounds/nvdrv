#!/usr/bin/env python3
"""Cross-check for src/rpc.c: decode the pre-boot RPCs in the command queue and verify
framing (msgq + GSP_MSG_QUEUE_ELEMENT + rpc_message_header_v03_00), checksums, and the
GspSystemInfo / registry payloads against values read independently from sysfs.

  python3 -I tools/rpc.py DEBUGFS_DIR PCI_BDF
"""
import re
import struct
import sys

SYSINFO_SIZE = 928


def checksum_fold(b):
    b = b + bytes(-len(b) % 8)
    s = 0
    for (w,) in struct.iter_unpack("<Q", b):
        s ^= w
    return (s >> 32) ^ (s & 0xFFFFFFFF)


def main(d, bdf):
    q = open(f"{d}/cmdq.bin", "rb").read()
    errs = []
    chk = lambda c, m: c or errs.append(m)
    ver, size, msz, cnt, wp, flags, rxo, eo = struct.unpack_from("<8I", q, 0)
    print(f"cmdq: writePtr {wp}, msgCount {cnt}, entryOff 0x{eo:x}")
    chk(wp == 2, f"writePtr {wp}, expected 2 messages")

    sysfs = f"/sys/bus/pci/devices/{bdf}"
    res = [int(l.split()[0], 16) for l in open(f"{sysfs}/resource")]
    cfg = open(f"{sysfs}/config", "rb").read(64)
    ident, subsys = struct.unpack_from("<I", cfg, 0)[0], struct.unpack_from("<I", cfg, 0x2c)[0]
    rev = cfg[8]
    dom, bus, dev = int(bdf[0:4], 16), int(bdf[5:7], 16), int(bdf[8:10], 16)
    iomem = open("/proc/iomem").read()
    m = re.search(r"([0-9a-f]+)-([0-9a-f]+) : BOOTFB", iomem)
    console = 0
    if m and int(m.group(1), 16) == res[1]:
        console = int(m.group(2), 16) - int(m.group(1), 16) + 1
        console = -(-console // 0x10000) * 0x10000

    for i in range(wp):
        e = q[eo + i * 0x1000:eo + (i + 1) * 0x1000]
        csum, seq, nelem = struct.unpack_from("<III", e, 32)
        hv, sig, length, fn, res_, resp, sequence, spare = struct.unpack_from("<8I", e, 48)
        msg = e[:48 + length]
        print(f"[{i}] seq {seq} elems {nelem} csum 0x{csum:08x} | ver 0x{hv:08x} sig {struct.pack('<I', sig)!r} "
              f"len {length} fn {fn} result 0x{res_:x}/0x{resp:x}")
        chk(seq == i and nelem == 1, f"[{i}] seq/elemCount")
        chk(checksum_fold(msg) == 0, f"[{i}] checksum does not fold to 0")
        chk(hv == 0x03000000 and sig == 0x43505256 and res_ == resp == 0xFFFFFFFF, f"[{i}] header")
        chk(e[:32] == bytes(32) and e[44:48] == bytes(4), f"[{i}] auth/aad/pad not zero")
        p = e[80:48 + length]
        if i == 0:
            chk(fn == 72 and len(p) == SYSINFO_SIZE, "[0] not GSP_SET_SYSTEM_INFO/928")
            g = lambda fmt, off: struct.unpack_from(fmt, p, off)[0]
            want = {
                "gpuPhysAddr": (g("<Q", 0), res[0]), "gpuPhysFbAddr": (g("<Q", 8), res[1]),
                "gpuPhysInstAddr": (g("<Q", 16), res[3]), "gpuPhysIoAddr": (g("<Q", 24), 0),
                "nvDomainBusDeviceFunc": (g("<Q", 32), (dom << 32) | (bus << 8) | dev),
                "consoleMemSize": (g("<Q", 64), console),
                "pciConfigMirrorBase": (g("<I", 80), 0x88000), "pciConfigMirrorSize": (g("<I", 84), 0x1000),
                "PCIDeviceID": (g("<I", 88), ident), "PCISubDeviceID": (g("<I", 92), subsys),
                "PCIRevisionID": (g("<I", 96), rev), "oorArch": (p[104], 0),
                "hypervisorType": (g("<I", 836), 4), "hostPageSize": (g("<Q", 920), 4096),
                "bIsPrimary": (p[896], int(open(f"{sysfs}/boot_vga").read())),
            }
            for k, (got, exp) in want.items():
                ok = got == exp
                print(f"    {k:22s} 0x{got:x}{'' if ok else f'   EXPECTED 0x{exp:x}'}")
                chk(ok, f"sysinfo {k}")
            mva = g("<Q", 72)
            print(f"    {'maxUserVa':22s} 0x{mva:x}  linkCap 0x{g('<I', 900):08x}  flr {p[125]}  64bBar0 {p[126]}")
            chk(mva in (0x7ffffffff000, 0xfffffffffff000), "maxUserVa is not an x86-64 TASK_SIZE")
            chk(p[160:836] == bytes(676), "acpiMethodData should be all zero")
        else:
            chk(fn == 73, "[1] not SET_REGISTRY")
            tsize, n = struct.unpack_from("<II", p, 0)
            chk(tsize == len(p), f"registry size {tsize} != payload {len(p)}")
            for j in range(n):
                no, ty, data, ln = struct.unpack_from("<IB3xII", p, 8 + 16 * j)
                name = p[no:p.index(b"\0", no)].decode()
                print(f"    registry: {name} = {data} (type {ty}, len {ln})")
                chk(ty == 1 and ln == 4, f"registry entry {name}")

    if errs:
        sys.exit("FAIL:\n  " + "\n  ".join(errs))
    print("PASS: pre-boot RPCs are well-formed and match independently read values")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
