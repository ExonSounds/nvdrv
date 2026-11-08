#!/usr/bin/env python3
"""Verify the kernel's FRTS-patched FWSEC image against an independent parse of the VBIOS.

  sudo cat /sys/kernel/debug/nvdrv/0000:01:00.0/vbios          > /tmp/vbios.bin
  sudo cat /sys/kernel/debug/nvdrv/0000:01:00.0/fwsec_frts.bin > /tmp/fwsec_frts.bin
  python3 -I tests/check_fwsec_frts.py vbios.bin fwsec_frts.bin FRTS_OFFSET FUSE_VERSION
"""
import os
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import vbios as vb  # noqa: E402

rom, img = open(sys.argv[1], "rb").read(), open(sys.argv[2], "rb").read()
frts, fuse = int(sys.argv[3], 0), int(sys.argv[4], 0)

images = vb.walk_images(rom)
exp = vb.expansion_rom_offset(images)
toks = vb.parse_bit(rom, vb.find_bit(rom, len(rom)))
_, _, desc_at = vb.find_fwsec(rom, toks, exp)
d = vb.parse_desc_v3(rom, desc_at)
orig = rom[d["image_offset"]:d["image_offset"] + d["image_size"]]
assert len(img) == len(orig), f"size {len(img):#x} != {len(orig):#x}"

dmem = d["imem_load_size"]
_, _, mapper = vb.find_dmem_mapper(orig[dmem:], d["interface_offset"])
cmd_in = struct.unpack_from("<I", orig, dmem + mapper + 8)[0]

# Expected patches
exp_img = bytearray(orig)
struct.pack_into("<I", exp_img, dmem + mapper + 0x2c, 0x15)
cmd = struct.pack("<IIQII", 1, 24, 0, 0, 2) + struct.pack("<IIIII", 1, 20, frts >> 12, 0x100, 2) + b"\0" * 4
exp_img[dmem + cmd_in:dmem + cmd_in + 48] = cmd
vers = d["signature_versions"]
assert vers & (1 << fuse), "no signature for this fuse version"
idx = bin(vers & ((1 << fuse) - 1)).count("1")
sig = rom[d["sigs_offset"] + idx * 384:d["sigs_offset"] + (idx + 1) * 384]
exp_img[dmem + d["pkc_data_offset"]:dmem + d["pkc_data_offset"] + 384] = sig

diff = [i for i in range(len(img)) if img[i] != orig[i]]
regions = []
for i in diff:
    if regions and i <= regions[-1][1] + 4:
        regions[-1][1] = i
    else:
        regions.append([i, i])
print("changed regions (DMEM-relative):",
      ", ".join(f"0x{a - dmem:x}-0x{b - dmem:x}" for a, b in regions))
print(f"expected: init_cmd @0x{mapper + 0x2c:x}, cmd_in @0x{cmd_in:x}+48, sig[{idx}] @0x{d['pkc_data_offset']:x}+384")
if bytes(exp_img) == img:
    print("PASS: kernel image matches independent reconstruction byte-for-byte")
else:
    bad = [i for i in range(len(img)) if img[i] != exp_img[i]]
    sys.exit(f"FAIL: {len(bad)} bytes differ, first at image 0x{bad[0]:x}")
