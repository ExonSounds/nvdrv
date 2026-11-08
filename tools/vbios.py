#!/usr/bin/env python3
"""Phase 2 prototype / cross-check for src/vbios.c: parse an NVIDIA VBIOS dump.

  sudo cat /sys/kernel/debug/nvdrv/0000:01:00.0/vbios > vbios.bin
  python3 -I tools/vbios.py vbios.bin

Rules follow open-gpu-kernel-modules 570.144
(kernel_gsp_vbios_tu102.c, kernel_gsp_fwsec.c, kernel_gsp_frts_tu102.c).
"""
import struct
import sys

ROM_SIGS = (0xAA55, 0x4E56, 0xBB77)            # standard, "VN", NV2
DATA_SIGS = (b"PCIR", b"NPDS", b"RGIS")
CODE_TYPES = {0x00: "x86 base", 0x03: "EFI", 0xE0: "VBIOS ext (FWSEC)"}
APP_IDS = {0x05: "FIRMWARE_SEC_LIC", 0x45: "FWSEC_DBG", 0x85: "FWSEC_PROD"}
DESC_V3_SIZE = 44
IFACE_ID_DMEMMAPPER = 4
DMEM_MAPPER_V3 = ("signature", "version_size", "cmd_in_buffer_offset", "cmd_in_buffer_size",
                  "cmd_out_buffer_offset", "cmd_out_buffer_size", "nvf_img_data_buffer_offset",
                  "nvf_img_data_buffer_size", "printfBufferHdr", "ucode_build_time_stamp",
                  "ucode_signature", "init_cmd", "ucode_feature", "ucode_cmd_mask0",
                  "ucode_cmd_mask1", "multiTgtTbl")


def u8(d, o): return d[o]
def u16(d, o): return struct.unpack_from("<H", d, o)[0]
def u32(d, o): return struct.unpack_from("<I", d, o)[0]


def walk_images(d):
    """s_locateExpansionRoms(): follow the chain until the last-image flag
    (taken from the NPDE extension when it is new enough to carry one)."""
    off, images = 0, []
    while True:
        if u16(d, off) not in ROM_SIGS:
            raise ValueError(f"bad ROM signature at 0x{off:x}")
        pcir = off + u16(d, off + 0x18)
        if d[pcir:pcir + 4] not in DATA_SIGS:
            raise ValueError(f"bad PCI data signature at 0x{pcir:x}")
        img = {"offset": off, "sig": d[pcir:pcir + 4].decode(),
               "vendor": u16(d, pcir + 4), "device": u16(d, pcir + 6),
               "code_type": u8(d, pcir + 0x14),
               "last": bool(u8(d, pcir + 0x15) & 0x80), "npde": False}
        img_len = sub_len = u16(d, pcir + 0x10)
        ext = (pcir + u16(d, pcir + 0x0A) + 0xF) & ~0xF
        if d[ext:ext + 4] == b"NPDE" and u16(d, ext + 4) in (0x100, 0x101):
            img["npde"] = True
            ext_len = u16(d, ext + 6)
            sub_len = u16(d, ext + 8)
            if 0x0A + 1 <= ext_len:
                img["last"] = bool(u8(d, ext + 0x0A) & 0x80)
            elif sub_len < img_len:
                img["last"] = False
        img["length"] = sub_len * 512
        images.append(img)
        if img["last"]:
            return images
        off += img["length"]


def expansion_rom_offset(images):
    """Pointers in the falcon tables assume the first 0xE0 image follows the
    base image directly; this is the gap (here: the EFI image) to add back."""
    ext = next((i["offset"] for i in images if i["code_type"] == 0xE0), 0)
    base = next((i["length"] for i in images if i["code_type"] == 0x00), 0)
    return ext - base if ext and base else 0


def find_bit(d, size):
    at = 0
    while True:
        at = d.find(b"\xff\xb8BIT\x00", at, size)
        if at < 0:
            return None
        if sum(d[at:at + d[at + 8]]) & 0xFF == 0:      # header checksum
            return at
        at += 1


def parse_bit(d, at):
    hdr_size, tok_size, tok_count = d[at + 8], d[at + 9], d[at + 10]
    fmt = "<BBHI" if tok_size >= 8 else "<BBHH"
    toks = []
    for i in range(tok_count):
        tid, ver, size, ptr = struct.unpack_from(fmt, d, at + hdr_size + i * tok_size)
        toks.append({"id": tid, "ver": ver, "size": size, "ptr": ptr})
    return toks


def find_fwsec(d, toks, exp_off, debug=False):
    want = 0x45 if debug else 0x85
    for t in toks:
        if t["id"] != 0x70 or t["ver"] != 2 or t["size"] < 4:
            continue
        table = exp_off + u32(d, t["ptr"])
        ver, hsz, esz, cnt = d[table:table + 4]
        if ver != 1 or hsz < 6 or esz < 6:
            continue
        entries = []
        for i in range(cnt):
            app, tgt, ptr = struct.unpack_from("<BBI", d, table + hsz + i * esz)
            entries.append((app, tgt, ptr))
        for app, tgt, ptr in entries:
            if app in (0x05, want):
                return table, entries, exp_off + ptr
    return None, [], None


def parse_desc_v3(d, at):
    hdr = u32(d, at)
    if not hdr & 1 or (hdr >> 8) & 0xFF != 3:
        raise ValueError(f"desc @0x{at:x}: hdr 0x{hdr:08x} is not a v3 descriptor")
    names = ("hdr", "stored_size", "pkc_data_offset", "interface_offset", "imem_phys_base",
             "imem_load_size", "imem_virt_base", "dmem_phys_base", "dmem_load_size")
    f = dict(zip(names, struct.unpack_from("<9I", d, at)))
    (f["engine_id_mask"], f["ucode_id"], f["signature_count"],
     f["signature_versions"]) = struct.unpack_from("<HBBH", d, at + 36)
    f["desc_size"] = hdr >> 16
    f["image_offset"] = at + f["desc_size"]
    f["image_size"] = (f["stored_size"] + 255) & ~255
    f["sigs_offset"] = at + DESC_V3_SIZE
    return f


def find_dmem_mapper(dmem, iface_off):
    ver, hsz, esz, cnt = dmem[iface_off:iface_off + 4]
    entries = [struct.unpack_from("<II", dmem, iface_off + 4 + i * 8) for i in range(cnt)]
    mapper = next((off for eid, off in entries if eid == IFACE_ID_DMEMMAPPER), None)
    return (ver, hsz, esz, cnt), entries, mapper


def main(path):
    d = open(path, "rb").read()
    images = walk_images(d)
    rom_size = images[-1]["offset"] + images[-1]["length"]
    exp_off = expansion_rom_offset(images)
    print(f"{len(images)} ROM images, total 0x{rom_size:x}, expansionRomOffset 0x{exp_off:x}")
    for i, im in enumerate(images):
        print(f"  [{i}] @0x{im['offset']:06x} len 0x{im['length']:06x} {im['sig']} "
              f"{im['vendor']:04x}:{im['device']:04x} type 0x{im['code_type']:02x} "
              f"({CODE_TYPES.get(im['code_type'], '?')}){' NPDE' if im['npde'] else ''}"
              f"{' LAST' if im['last'] else ''}")

    bit = find_bit(d, rom_size)
    toks = parse_bit(d, bit)
    print(f"\nBIT @0x{bit:x} (checksum ok), {len(toks)} tokens: "
          + " ".join(chr(t['id']) if 0x20 < t['id'] < 0x7F else f"0x{t['id']:02x}" for t in toks))

    table, entries, desc_at = find_fwsec(d, toks, exp_off)
    print(f"\nFalcon ucode table @0x{table:x}:")
    for app, tgt, ptr in entries:
        if app or ptr:
            print(f"  app 0x{app:02x} target 0x{tgt:02x} desc 0x{exp_off + ptr:06x} {APP_IDS.get(app, '')}")

    desc = parse_desc_v3(d, desc_at)
    print(f"\nFWSEC_PROD desc v3 @0x{desc_at:x}:")
    for k in ("desc_size", "stored_size", "image_size", "image_offset", "pkc_data_offset",
              "interface_offset", "imem_load_size", "dmem_load_size", "engine_id_mask",
              "ucode_id", "signature_count", "signature_versions"):
        print(f"  {k:20s} 0x{desc[k]:x}")

    img = d[desc["image_offset"]:desc["image_offset"] + desc["image_size"]]
    dmem = img[desc["imem_load_size"]:desc["imem_load_size"] + desc["dmem_load_size"]]
    (ver, hsz, esz, cnt), ientries, mapper = find_dmem_mapper(dmem, desc["interface_offset"])
    print(f"\nInterface table @DMEM 0x{desc['interface_offset']:x}: v{ver} hdr {hsz} entry {esz} count {cnt}")
    for eid, off in ientries:
        print(f"  id 0x{eid:x} -> DMEM 0x{off:x}{'  (DMEMMAPPER)' if eid == IFACE_ID_DMEMMAPPER else ''}")
    m = dict(zip(DMEM_MAPPER_V3, struct.unpack_from("<16I", dmem, mapper)))
    print(f"\nDMEM mapper @DMEM 0x{mapper:x}: signature {struct.pack('<I', m['signature'])!r}")
    for k in DMEM_MAPPER_V3[1:]:
        print(f"  {k:28s} 0x{m[k]:x}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "vbios.bin")
