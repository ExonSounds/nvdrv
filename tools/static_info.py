#!/usr/bin/env python3
"""Cross-check for src/rm.c: decode the raw GET_GSP_STATIC_INFO reply (gsp_static_info.bin)
with field offsets taken from compiling NVIDIA's header (tools/gsp_static_info_probe.sh output,
not the constants in rm.c), compare with the driver's own decode in `info`, and check the
values against what nvdrv measured independently (VRAM size, GSP carve-out, chip name).

  tools/gsp_static_info_probe.sh > probe.txt
  python3 -I tools/static_info.py probe.txt DEBUGFS_DIR
"""
import re
import struct
import sys

# NV2080_ENGINE_TYPE_* (class/cl2080_notification.h), the numbering engineCaps uses
NV2080_ENGINES = {0x01: "GR", **{0x09 + i: f"CE{i}" for i in range(10)}, 0x13: "NVDEC0",
                  0x1B: "NVENC0", 0x22: "SW", 0x26: "SEC2", 0x33: "OFA"}


def parse_probe(path):
    top, sub = {}, {}
    size = None
    section = None
    for line in open(path):
        if m := re.match(r"sizeof GspStaticConfigInfo (\d+)", line):
            size = int(m.group(1))
        elif m := re.match(r"(\w+): sizeof (\d+)(.*)", line):
            section = m.group(1)
            sub[section] = {"sizeof": int(m.group(2))}
            for k, v in re.findall(r"(\w+) @(\d+)", m.group(3)):
                sub[section][k] = int(v)
            if e := re.search(r"entry (\d+)", m.group(3)):
                sub[section]["entry"] = int(e.group(1))
        elif m := re.match(r"\s+region\.(\w+)\s+(\d+)", line):
            sub["FB_REGION_INFO"][m.group(1)] = int(m.group(2))
        elif m := re.match(r"(\w+)\s+(\d+)\s+(\d+)$", line):
            top[m.group(1)] = (int(m.group(2)), int(m.group(3)))
    return size, top, sub


def info_value(info, key):
    m = re.search(rf"^{re.escape(key)}:\s+(.*)$", info, re.M)
    return m.group(1).strip() if m else None


def main(probe, d):
    size, top, sub = parse_probe(probe)
    b = open(f"{d}/gsp_static_info.bin", "rb").read()
    info = open(f"{d}/info").read()
    errs = []
    chk = lambda c, msg: c or errs.append(msg)

    chk(size == len(b), f"blob is {len(b)} bytes, header says {size}")
    chk(any(b), "blob is all zeros: GET_GSP_STATIC_INFO never answered")
    u32 = lambda name, extra=0: struct.unpack_from("<I", b, top[name][0] + extra)[0]
    u64 = lambda name, extra=0: struct.unpack_from("<Q", b, top[name][0] + extra)[0]
    cstr = lambda off, n: b[off:off + n].split(b"\0")[0].decode("ascii", "replace")

    name = cstr(top["gpuNameString"][0], top["gpuNameString"][1])
    short = cstr(top["gpuShortNameString"][0], top["gpuShortNameString"][1])
    fb_len = u64("fb_length")
    gid_len = u32("gidInfo", sub["GID_INFO"]["length"])
    g = b[top["gidInfo"][0] + sub["GID_INFO"]["data"]:][:16]
    uuid = f"GPU-{g[0:4].hex()}-{g[4:6].hex()}-{g[6:8].hex()}-{g[8:10].hex()}-{g[10:16].hex()}"
    caps = [u32("engineCaps", 4 * i) for i in range(top["engineCaps"][1] // 4)]
    engines = [NV2080_ENGINES.get(i, f"#{i}") for i in range(32 * len(caps)) if caps[i // 32] >> (i % 32) & 1]
    hc, hd, hs = u32("hInternalClient"), u32("hInternalDevice"), u32("hInternalSubdevice")

    fr = sub["FB_REGION_INFO"]
    fr_off = top["fbRegionInfoParams"][0]
    nreg = struct.unpack_from("<I", b, fr_off)[0]
    regions = []
    for i in range(min(nreg, 16)):
        e = fr_off + fr["fbRegion"] + i * fr["entry"]
        base, limit, rsvd = struct.unpack_from("<QQQ", b, e)
        regions.append((base, limit, rsvd, b[e + fr["bProtected"]]))

    print(f"name {name!r} ({short}), {uuid} (gid length {gid_len})")
    print(f"fb_length {fb_len >> 20} MiB, {nreg} FB regions, engines {' '.join(engines)}")
    print(f"internal client/device/subdevice 0x{hc:08x}/0x{hd:08x}/0x{hs:08x}")
    for i, (base, limit, rsvd, prot) in enumerate(regions):
        print(f"  region {i}: 0x{base:x} - 0x{limit:x} reserved 0x{rsvd:x}{' protected' if prot else ''}")

    # 1. the driver's decode (rm.c offsets) agrees with the header-derived one
    chk(info_value(info, "name") == name, "driver name differs")
    chk(info_value(info, "short name") == short, "driver short name differs")
    chk(info_value(info, "uuid") == uuid, f"driver uuid {info_value(info, 'uuid')} differs")
    chk((info_value(info, "fb") or "").startswith(f"{fb_len >> 20} MiB"), "driver fb_length differs")
    chk(info_value(info, "internal hdls") == f"client 0x{hc:08x} device 0x{hd:08x} subdevice 0x{hs:08x}",
        "driver internal handles differ")
    eng = info_value(info, "engine caps") or ""
    chk(eng.split(":", 1)[-1].split() == engines, f"driver engine list differs: {eng}")
    for i, (base, limit, rsvd, _) in enumerate(regions):
        chk((info_value(info, f"fb region {i}") or "").startswith(f"0x{base:x} - 0x{limit:x}, reserved 0x{rsvd:x}"),
            f"driver fb region {i} differs")

    # 2. against what nvdrv measured before GSP existed
    chip = info_value(info, "chip").split()[0]
    chk(short.startswith(chip), f"GSP short name {short} is not chip {chip}")
    lmr = int(info_value(info, "vram (lmr)").split()[0])
    chk(fb_len >> 20 == lmr, f"GSP fb_length {fb_len >> 20} MiB != LOCAL_MEMORY_RANGE {lmr} MiB")
    m = re.search(r"^gsp reserved:\s+0x([0-9a-f]+) - 0x([0-9a-f]+)", info, re.M)
    rsvd_lo, rsvd_hi = int(m.group(1), 16), int(m.group(2), 16)
    chk(any(base == rsvd_lo and limit == rsvd_hi - 1 and rsvd == rsvd_hi - rsvd_lo for base, limit, rsvd, _ in regions),
        f"no FB region covers our GSP carve-out 0x{rsvd_lo:x}-0x{rsvd_hi:x} as fully reserved")
    chk(regions and regions[-1][1] == rsvd_hi - 1, "FB regions do not end at the top of VRAM")
    chk(all(regions[i][1] + 1 == regions[i + 1][0] for i in range(len(regions) - 1)), "FB regions not contiguous")
    chk(gid_len >= 16 and any(g), "no GPU UUID")
    chk("GR" in engines and any(e.startswith("CE") for e in engines), "no GR or copy engine reported")
    chk(hc and hd and hs, "internal RM handles missing")

    for e in errs:
        print("FAIL:", e)
    print("PASS" if not errs else f"{len(errs)} check(s) failed")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
