#!/usr/bin/env python3
"""Phase 5 prototype / cross-check for src/fw.c: decode the GSP boot firmware files.

  python3 -I tools/fw.py /lib/firmware/nvidia/ga104/gsp 570.144 [SEC2_FUSE_VERSION]

Files are the linux-firmware packaging of the bindata archives that RM embeds
(open-gpu-kernel-modules 570.144: kernel_gsp_booter.c, kernel_gsp.c):
  booter_load / booter_unload  bin header -> HS v2 header -> {image, header, sigs, patch_*}
  bootloader                   bin header -> RM_RISCV_UCODE_DESC + image
  gsp                          RISC-V ELF: .fwimage, .fwversion, .fwsignature_<family>
"""
import os
import struct
import subprocess
import sys
import zlib

BIN_MAGIC = 0x10DE
RISCV_DESC_FIELDS = (
    "version", "bootloaderOffset", "bootloaderSize", "bootloaderParamOffset",
    "bootloaderParamSize", "riscvElfOffset", "riscvElfSize", "appVersion",
    "manifestOffset", "manifestSize", "monitorDataOffset", "monitorDataSize",
    "monitorCodeOffset", "monitorCodeSize", "bIsMonitorEnabled", "swbromCodeOffset",
    "swbromCodeSize", "swbromDataOffset", "swbromDataSize", "fbReservedSize", "bSignedAsCode")
BOOTER_HDR_FIELDS = ("osCodeOffset", "osCodeSize", "osDataOffset", "osDataSize", "numApps",
                     "appCodeOffset", "appCodeSize", "appDataOffset", "appDataSize")


def load(path):
    if os.path.exists(path):
        return open(path, "rb").read()
    return subprocess.run(["zstd", "-dqc", path + ".zst"], check=True, capture_output=True).stdout


def u32(d, o):
    return struct.unpack_from("<I", d, o)[0]


def bin_hdr(d):
    magic, ver, size, hdr_off, data_off, data_size = struct.unpack_from("<6I", d, 0)
    assert magic == BIN_MAGIC and ver == 1, "bad bin header"   # size field is not the file length
    assert data_off + data_size <= len(d)
    return hdr_off, data_off, data_size


def booter(d, fuse_ver):
    hdr_off, data_off, data_size = bin_hdr(d)
    (sig_off, sig_size, patch_loc_off, patch_sig_off, meta_off, meta_size,
     num_sigs_off, header_off, header_size) = struct.unpack_from("<9I", d, hdr_off)
    assert meta_size == 12 and header_size == 36
    hdr = dict(zip(BOOTER_HDR_FIELDS, struct.unpack_from("<9I", d, header_off)))
    patch_loc, patch_sig = u32(d, patch_loc_off), u32(d, patch_sig_off)
    fuse_meta, engine_id, ucode_id = struct.unpack_from("<3I", d, meta_off)
    num_sigs = u32(d, num_sigs_off)
    assert hdr["numApps"] == 1 and patch_sig == 0 and num_sigs and sig_size % num_sigs == 0
    sig_size_each = sig_size // num_sigs
    image = bytearray(d[data_off:data_off + data_size])
    r = dict(hdr, data_off=data_off, data_size=data_size, sig_off=sig_off, sig_total=sig_size,
             num_sigs=num_sigs, sig_size=sig_size_each, patch_loc=patch_loc,
             meta_fuse=fuse_meta, engine_id=engine_id, ucode_id=ucode_id,
             hs_sig_dmem=patch_loc - hdr["osDataOffset"])
    if fuse_ver is not None:
        # s_patchBooterUcodeSignature: index counts down from the newest signature
        idx = num_sigs - 1 - fuse_ver if num_sigs > 1 else 0
        assert 0 <= idx < num_sigs, f"no signature for fuse version {fuse_ver}"
        image[patch_loc:patch_loc + sig_size_each] = d[sig_off + idx * sig_size_each:
                                                       sig_off + (idx + 1) * sig_size_each]
        r["sig_index"] = idx
    r["image"] = bytes(image)
    return r


def bootloader(d):
    hdr_off, data_off, data_size = bin_hdr(d)
    desc = dict(zip(RISCV_DESC_FIELDS, struct.unpack_from("<21I", d, hdr_off)))
    assert data_off - hdr_off >= 84
    return desc, d[data_off:data_off + data_size]


def elf_sections(d):
    assert d[:4] == b"\x7fELF" and d[4] == 2 and d[5] == 1
    shoff, = struct.unpack_from("<Q", d, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", d, 0x3A)
    assert shentsize == 64
    def sh(i):
        return struct.unpack_from("<IIQQQQIIQQ", d, shoff + i * 64)
    stroff = sh(shstrndx)[4]
    secs = {}
    for i in range(shnum):
        name, _, _, _, off, size = sh(i)[:6]
        n = d[stroff + name:d.index(b"\0", stroff + name)].decode()
        secs[n] = (off, size)
    return secs


def radix3_shape(size):
    pages = [0, 0, 0, (size + 4095) >> 12]
    for i in (2, 1, 0):
        pages[i] = ((pages[i + 1] - 1) >> 9) + 1
    assert pages[0] == 1
    return pages


def main(fwdir, ver, fuse):
    fuse = int(fuse, 0) if fuse is not None else None
    for name in ("booter_load", "booter_unload"):
        b = booter(load(f"{fwdir}/{name}-{ver}.bin"), fuse)
        print(f"{name}: image 0x{b['data_size']:x} @file 0x{b['data_off']:x}, crc32 {zlib.crc32(b['image']):08x}")
        print(f"  os code 0x{b['osCodeOffset']:x}+0x{b['osCodeSize']:x}  os data 0x{b['osDataOffset']:x}+0x{b['osDataSize']:x}"
              f"  app code 0x{b['appCodeOffset']:x}+0x{b['appCodeSize']:x}  app data 0x{b['appDataOffset']:x}+0x{b['appDataSize']:x}")
        print(f"  sigs {b['num_sigs']}x0x{b['sig_size']:x}, patch_loc 0x{b['patch_loc']:x} (DMEM 0x{b['hs_sig_dmem']:x}),"
              f" ucode id {b['ucode_id']}, engine mask 0x{b['engine_id']:x}, meta fuse {b['meta_fuse']}"
              + (f", fuse {fuse} -> sig {b['sig_index']}" if fuse is not None else ""))

    desc, img = bootloader(load(f"{fwdir}/bootloader-{ver}.bin"))
    print(f"bootloader: image 0x{len(img):x}, crc32 {zlib.crc32(img):08x}")
    print("  " + ", ".join(f"{k} 0x{v:x}" for k, v in desc.items() if v))

    g = load(f"{fwdir}/gsp-{ver}.bin")
    secs = elf_sections(g)
    fv_off, fv_size = secs[".fwversion"]
    fwver = g[fv_off:fv_off + fv_size]
    assert fwver == ver.encode() + b"\0", f"fwversion {fwver!r}"
    io, isz = secs[".fwimage"]
    so, ssz = secs[".fwsignature_ga10x"]
    print(f"gsp: .fwversion {fwver[:-1].decode()}, .fwimage 0x{isz:x} crc32 {zlib.crc32(g[io:io + isz]):08x},"
          f" .fwsignature_ga10x 0x{ssz:x} crc32 {zlib.crc32(g[so:so + ssz]):08x}")
    print(f"  radix3 pages per level: {radix3_shape(isz)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
