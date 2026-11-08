#!/usr/bin/env python3
"""Cross-check for src/fb.c + src/wpr.c: recompute the GSP FB layout independently and
verify a GspFwWprMeta blob field by field.

  python3 -I tools/layout.py wpr_meta.bin FWDIR VER FB_SIZE VGA_WS_OFFSET \\
      RADIX3_ADDR BOOTLOADER_ADDR SIGNATURE_ADDR

Inputs are hardware facts (FB size, VGA workspace) and sysmem bus addresses
(only the kernel knows those); everything else is derived here from the
firmware files and the rules in kgspCalculateFbLayout_TU102 (570.144).
"""
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fw  # noqa: E402

MB, GB = 1 << 20, 1 << 30
FIELDS = ("magic", "revision", "sysmemAddrOfRadix3Elf", "sizeOfRadix3Elf", "sysmemAddrOfBootloader",
          "sizeOfBootloader", "bootloaderCodeOffset", "bootloaderDataOffset",
          "bootloaderManifestOffset", "sysmemAddrOfSignature", "sizeOfSignature", "gspFwRsvdStart",
          "nonWprHeapOffset", "nonWprHeapSize", "gspFwWprStart", "gspFwHeapOffset", "gspFwHeapSize",
          "gspFwOffset", "bootBinOffset", "frtsOffset", "frtsSize", "gspFwWprEnd", "fbSize",
          "vgaWorkspaceOffset", "vgaWorkspaceSize", "bootCount")


def down(x, a): return x // a * a
def up(x, a): return (x + a - 1) // a * a


def expected(fb_size, vga, bl_size, elf_size, desc, sig_size, addrs):
    e = {"magic": 0xDC3AAE21371A60B3, "revision": 1, "fbSize": fb_size,
         "vgaWorkspaceOffset": vga, "vgaWorkspaceSize": fb_size - vga}
    e["gspFwWprEnd"] = down(vga, 0x20000)          # normal boot: no WPR end margin; MMU lock is GA100-only
    e["frtsSize"] = MB
    e["frtsOffset"] = e["gspFwWprEnd"] - MB
    e["sizeOfBootloader"] = bl_size
    e["bootBinOffset"] = down(e["frtsOffset"] - bl_size, 0x1000)
    e["sizeOfRadix3Elf"] = elf_size
    e["gspFwOffset"] = down(e["bootBinOffset"] - elf_size, 0x10000)
    non_wpr, meta = up(MB, MB), up(256, MB)
    pre, post = meta + non_wpr, up(fb_size - e["gspFwOffset"], MB)
    heap = 22 * MB + 8 * MB + up((96 << 10) * (up(fb_size, GB) // GB), MB) + up((48 << 10) * 2048, MB)
    heap = min(max(heap, 88 * MB), ((256 * MB - (pre + post)) >> 20) << 20)
    e["gspFwHeapOffset"] = down(e["gspFwOffset"] - heap, MB)
    e["gspFwHeapSize"] = down(e["gspFwOffset"] - e["gspFwHeapOffset"], MB)
    e["gspFwWprStart"] = e["gspFwHeapOffset"] - meta
    e["nonWprHeapSize"] = non_wpr
    e["nonWprHeapOffset"] = e["gspFwWprStart"] - non_wpr
    e["gspFwRsvdStart"] = e["nonWprHeapOffset"]
    e["bootloaderCodeOffset"] = desc["monitorCodeOffset"]
    e["bootloaderDataOffset"] = desc["monitorDataOffset"]
    e["bootloaderManifestOffset"] = desc["manifestOffset"]
    e["sizeOfSignature"] = up(sig_size, 256)
    e["sysmemAddrOfRadix3Elf"], e["sysmemAddrOfBootloader"], e["sysmemAddrOfSignature"] = addrs
    e["bootCount"] = 0
    return e


def main(meta_path, fwdir, ver, fb_size, vga, *addrs):
    blob = open(meta_path, "rb").read()
    assert len(blob) == 256, f"meta is {len(blob)} bytes"
    got = dict(zip(FIELDS, struct.unpack_from("<26Q", blob, 0)))
    tail = struct.unpack_from("<4I Q I I B B 2s I Q", blob, 208)
    desc, img = fw.bootloader(fw.load(f"{fwdir}/bootloader-{ver}.bin"))
    g = fw.load(f"{fwdir}/gsp-{ver}.bin")
    secs = fw.elf_sections(g)
    exp = expected(int(fb_size, 0), int(vga, 0), len(img), secs[".fwimage"][1], desc,
                   secs[".fwsignature_ga10x"][1], [int(a, 0) for a in addrs])

    bad = 0
    for k in FIELDS:
        ok = got[k] == exp[k]
        bad += not ok
        print(f"  {k:26s} 0x{got[k]:012x}{'' if ok else f'   EXPECTED 0x{exp[k]:x}'}")
    rpc_pad, (crash_addr, crash_size, ls_pad, vf, flags, pad, pmu, verified) = tail[:4], tail[4:]
    tail_ok = (rpc_pad == (0, 0, 0, 0) and crash_addr == 0 and crash_size == 0 and ls_pad == 0
               and vf == 0 and flags == 1 and pad == b"\0\0" and pmu == 0 and verified == 0)
    print(f"  tail: crashcat 0x{crash_addr:x}/{crash_size}, vf {vf}, flags 0x{flags:x}, pmu {pmu}, "
          f"verified 0x{verified:x}{'' if tail_ok else '   EXPECTED all 0 except flags=1 (CLOCK_BOOST)'}")
    bad += not tail_ok

    # Structural invariants: ordered, non-overlapping, inside the pre-scrubbed top 256 MiB
    regions = [("non-WPR heap", got["nonWprHeapOffset"], got["nonWprHeapSize"]),
               ("WPR meta", got["gspFwWprStart"], MB),
               ("GSP heap", got["gspFwHeapOffset"], got["gspFwHeapSize"]),
               ("GSP ELF", got["gspFwOffset"], got["sizeOfRadix3Elf"]),
               ("boot bin", got["bootBinOffset"], got["sizeOfBootloader"]),
               ("FRTS", got["frtsOffset"], got["frtsSize"]),
               ("VGA ws", got["vgaWorkspaceOffset"], got["vgaWorkspaceSize"])]
    for (n1, o1, s1), (n2, o2, _) in zip(regions, regions[1:]):
        if o1 + s1 > o2:
            print(f"  OVERLAP: {n1} ends 0x{o1 + s1:x} > {n2} at 0x{o2:x}")
            bad += 1
    if regions[-1][1] + regions[-1][2] != got["fbSize"] or got["gspFwRsvdStart"] < got["fbSize"] - 256 * MB:
        print("  BAD: layout does not end at fbSize or leaves the pre-scrubbed region")
        bad += 1
    print(f"  GSP reservation: {(got['fbSize'] - got['gspFwRsvdStart']) >> 20} MiB, heap {got['gspFwHeapSize'] >> 20} MiB")
    if bad:
        sys.exit(f"FAIL: {bad} mismatches")
    print("PASS: WPR meta matches independent layout computation")


if __name__ == "__main__":
    main(*sys.argv[1:])
