// SPDX-License-Identifier: GPL-2.0
/*
 * VBIOS access and FWSEC extraction (Phase 2).
 *
 * Follows the procedure in open-gpu-kernel-modules 570.144:
 *   kernel_gsp_vbios_tu102.c  - ROM image chain / expansion ROM offset
 *   kernel_gsp_fwsec.c        - BIT -> falcon ucode table -> desc v3
 *   kernel_gsp_frts_tu102.c   - DMEM mapper patching for the FRTS command
 *
 * Everything here reads registers and builds buffers in system memory only.
 */
#include <linux/bitops.h>
#include <linux/overflow.h>
#include <linux/slab.h>
#include <linux/string.h>
#include <linux/unaligned.h>

#include "nvdrv.h"
#include "nv_regs.h"

/* PCI expansion ROM (platform/pci_exp_table.h) */
#define ROM_SIG_PCI		0xaa55
#define ROM_SIG_NV		0x4e56	/* "VN" */
#define ROM_SIG_NV2		0xbb77
#define ROM_PCI_DATA_PTR	0x18
#define PCIR_SIG_PCI		0x52494350	/* "PCIR" */
#define PCIR_SIG_NV		0x5344504e	/* "NPDS" */
#define PCIR_SIG_NV2		0x53494752	/* "RGIS" */
#define PCIR_LEN		0x0a
#define PCIR_IMAGE_LEN		0x10
#define PCIR_CODE_TYPE		0x14
#define PCIR_LAST_IMAGE		0x15
#define NPDE_SIG		0x4544504e	/* "NPDE" */
#define NPDE_REV		0x04
#define NPDE_LEN		0x06
#define NPDE_SUBIMAGE_LEN	0x08
#define NPDE_LAST_IMAGE		0x0a
#define PCI_LAST_IMAGE		BIT(7)
#define ROM_BLOCK		512
#define CODE_TYPE_BASE		0x00
#define CODE_TYPE_EXT		0xe0

/* BIT */
#define BIT_HEADER_ID		0xb8ff
#define BIT_HEADER_SIG		0x00544942	/* "BIT\0" */
#define BIT_TOKEN_FALCON_DATA	0x70

/* Falcon ucode table */
#define APPID_FIRMWARE_SEC_LIC	0x05
#define APPID_FWSEC_DBG		0x45
#define APPID_FWSEC_PROD	0x85
#define DESC_V3_SIZE		44

/* FWSEC application interface (DMEM) */
#define IFACE_ID_DMEMMAPPER	0x4
#define DMAP_SIG		0x50414d44	/* "DMAP" */
#define DMAP_SIZE		64		/* FALCON_APPLICATION_INTERFACE_DMEM_MAPPER_V3 */
#define DMAP_CMD_IN_OFFSET	0x08
#define DMAP_CMD_IN_SIZE	0x0c
#define DMAP_INIT_CMD		0x2c
#define DMAP_CMD_FRTS		0x15

#define FRTS_SIZE_4K		0x100		/* 1 MiB */
#define FRTS_MEDIA_FB		2
#define READ_VBIOS_FLAGS	2

struct fwseclic_read_vbios_desc {
	__le32 version;
	__le32 size;
	__le64 gfw_image_offset;
	__le32 gfw_image_size;
	__le32 flags;
};

struct fwseclic_frts_region_desc {
	__le32 version;
	__le32 size;
	__le32 frts_region_offset_4k;
	__le32 frts_region_size;
	__le32 frts_region_media_type;
};

/* Matches RM's sizeof(FWSECLIC_FRTS_CMD) == 48: 24 + 20 + 4 bytes tail padding. */
struct fwseclic_frts_cmd {
	struct fwseclic_read_vbios_desc read_vbios;
	struct fwseclic_frts_region_desc frts_region;
	__le32 pad;
};
static_assert(sizeof(struct fwseclic_frts_cmd) == 48);

/* ---- PROM access ------------------------------------------------------ */

/* Unaligned 1/2/4-byte read from PROM via aligned dword reads. */
static int prom_rd(struct nvdrv_dev *nv, u32 off, unsigned int n, u32 *val)
{
	u32 aligned = off & ~3u, shift = (off & 3) * 8;
	u64 w;

	if (off > NV_PROM_SIZE - 8)
		return -ERANGE;
	w = nv_rd32(nv, NV_PROM_DATA + aligned);
	if ((off & 3) + n > 4)
		w |= (u64)nv_rd32(nv, NV_PROM_DATA + aligned + 4) << 32;
	w >>= shift;
	*val = n == 4 ? (u32)w : (u32)w & (BIT(n * 8) - 1);
	return 0;
}

#define PROM_RD(nv, off, n, out)				\
	({ int __r = prom_rd(nv, off, n, out); if (__r) return __r; })

/*
 * Walk the ROM image chain in PROM to find the total size and the expansion
 * ROM offset (s_locateExpansionRoms()).
 */
static int prom_walk(struct nvdrv_dev *nv, u32 *size, u32 *exp_rom_offset)
{
	struct device *dev = &nv->pdev->dev;
	u32 cur = 0, blk_off = 0, blk_size = 0, ext_off = 0, base_size = 0;
	int n;

	for (n = 0; n < 32; n++) {
		u32 sig, ptr, pcir, psig, len, img_len, sub_len, ext, esig, rev, elen, type, last;

		PROM_RD(nv, cur, 2, &sig);
		if (sig != ROM_SIG_PCI && sig != ROM_SIG_NV && sig != ROM_SIG_NV2) {
			if (cur == 0)
				dev_err(dev, "vbios: no ROM signature at 0 (IFR header not supported yet)\n");
			else
				dev_err(dev, "vbios: bad ROM signature %04x at 0x%x\n", sig, cur);
			return -EINVAL;
		}
		PROM_RD(nv, cur + ROM_PCI_DATA_PTR, 2, &ptr);
		pcir = cur + ptr;
		PROM_RD(nv, pcir, 4, &psig);
		if (psig != PCIR_SIG_PCI && psig != PCIR_SIG_NV && psig != PCIR_SIG_NV2) {
			dev_err(dev, "vbios: bad PCI data signature %08x at 0x%x\n", psig, pcir);
			return -EINVAL;
		}
		PROM_RD(nv, pcir + PCIR_LAST_IMAGE, 1, &last);
		PROM_RD(nv, pcir + PCIR_IMAGE_LEN, 2, &img_len);
		PROM_RD(nv, pcir + PCIR_LEN, 2, &len);
		PROM_RD(nv, pcir + PCIR_CODE_TYPE, 1, &type);
		last &= PCI_LAST_IMAGE;
		sub_len = img_len;

		ext = ALIGN(pcir + len, 16);
		PROM_RD(nv, ext, 4, &esig);
		if (esig == NPDE_SIG) {
			PROM_RD(nv, ext + NPDE_REV, 2, &rev);
			if (rev == 0x100 || rev == 0x101) {
				PROM_RD(nv, ext + NPDE_LEN, 2, &elen);
				PROM_RD(nv, ext + NPDE_SUBIMAGE_LEN, 2, &sub_len);
				if (NPDE_LAST_IMAGE + 1 <= elen) {
					PROM_RD(nv, ext + NPDE_LAST_IMAGE, 1, &last);
					last &= PCI_LAST_IMAGE;
				} else if (sub_len < img_len) {
					last = 0;
				}
			}
		}

		blk_off = cur;
		blk_size = sub_len * ROM_BLOCK;
		if (!ext_off && type == CODE_TYPE_EXT)
			ext_off = blk_off;
		else if (!base_size && type == CODE_TYPE_BASE)
			base_size = blk_size;

		if (last)
			break;
		if (!blk_size)
			return -EINVAL;
		cur += blk_size;
	}
	if (n == 32 || blk_off + blk_size > NV_PROM_SIZE)
		return -EINVAL;

	*size = blk_off + blk_size;
	*exp_rom_offset = (ext_off && base_size) ? ext_off - base_size : 0;
	return 0;
}

/* ---- bounds-checked image accessors ----------------------------------- */

static bool in_img(u32 img_size, u32 off, u32 len)
{
	u32 end;

	return !check_add_overflow(off, len, &end) && end <= img_size;
}

static int img_rd(const struct nv_vbios *v, u32 off, unsigned int n, u32 *val)
{
	if (!in_img(v->size, off, n))
		return -ERANGE;
	switch (n) {
	case 1: *val = v->data[off]; break;
	case 2: *val = get_unaligned_le16(v->data + off); break;
	default: *val = get_unaligned_le32(v->data + off); break;
	}
	return 0;
}

#define IMG_RD(v, off, n, out)					\
	({ int __r = img_rd(v, off, n, out); if (__r) return __r; })

static int find_bit(struct nv_vbios *v)
{
	u32 addr;

	for (addr = 0; addr + 12 < v->size; addr++) {
		u32 id, sig, hsize, i;
		u8 sum = 0;

		IMG_RD(v, addr, 2, &id);
		if (id != BIT_HEADER_ID)
			continue;
		IMG_RD(v, addr + 2, 4, &sig);
		if (sig != BIT_HEADER_SIG)
			continue;
		IMG_RD(v, addr + 8, 1, &hsize);
		if (!in_img(v->size, addr, hsize))
			continue;
		for (i = 0; i < hsize; i++)
			sum += v->data[addr + i];
		if (sum == 0) {
			v->bit_offset = addr;
			return 0;
		}
	}
	return -ENOENT;
}

int nv_vbios_load(struct nvdrv_dev *nv)
{
	struct nv_vbios *v = &nv->vbios;
	u32 size, exp, i;
	int ret;

	ret = prom_walk(nv, &size, &exp);
	if (ret)
		return ret;

	v->data = kvmalloc(size, GFP_KERNEL);
	if (!v->data)
		return -ENOMEM;
	for (i = 0; i < size; i += 4)
		put_unaligned_le32(nv_rd32(nv, NV_PROM_DATA + i), v->data + i);
	v->size = size;
	v->exp_rom_offset = exp;

	ret = find_bit(v);
	if (ret) {
		dev_err(&nv->pdev->dev, "vbios: BIT header not found\n");
		return ret;
	}
	dev_info(&nv->pdev->dev, "vbios: 0x%x bytes, BIT at 0x%x, expansion ROM offset 0x%x\n",
		 v->size, v->bit_offset, v->exp_rom_offset);
	return 0;
}

/* ---- FWSEC ------------------------------------------------------------ */

u32 nv_fuse_ucode_version(struct nvdrv_dev *nv, u8 ucode_id)
{
	u32 index = (u32)ucode_id - 1, val;

	if (index >= 16)
		return 0;
	val = nv_rd32(nv, NV_FUSE_OPT_FPF_GSP_UCODE1_VERSION + 4 * index);
	return val ? fls(val) : 0;	/* HIGHESTBITIDX + 1 */
}

/* BIT 'p' -> falcon ucode table -> descriptor offset of the wanted FWSEC. */
static int find_fwsec_desc(struct nv_vbios *v, bool debug, u32 *desc_off)
{
	u32 want = debug ? APPID_FWSEC_DBG : APPID_FWSEC_PROD;
	u32 hsize, tsize, tcount, t;

	IMG_RD(v, v->bit_offset + 8, 1, &hsize);
	IMG_RD(v, v->bit_offset + 9, 1, &tsize);
	IMG_RD(v, v->bit_offset + 10, 1, &tcount);
	if (tsize < 6)
		return -EINVAL;

	for (t = 0; t < tcount; t++) {
		u32 tok = v->bit_offset + hsize + t * tsize;
		u32 id, ver, dsize, dptr, tbl_ptr, table, tver, thsz, tesz, tcnt, e;

		IMG_RD(v, tok, 1, &id);
		IMG_RD(v, tok + 1, 1, &ver);
		IMG_RD(v, tok + 2, 2, &dsize);
		IMG_RD(v, tok + 4, tsize >= 8 ? 4 : 2, &dptr);
		if (id != BIT_TOKEN_FALCON_DATA || ver != 2 || dsize < 4)
			continue;

		IMG_RD(v, dptr, 4, &tbl_ptr);
		table = v->exp_rom_offset + tbl_ptr;
		IMG_RD(v, table, 1, &tver);
		IMG_RD(v, table + 1, 1, &thsz);
		IMG_RD(v, table + 2, 1, &tesz);
		IMG_RD(v, table + 3, 1, &tcnt);
		if (tver != 1 || thsz < 6 || tesz < 6)
			continue;

		for (e = 0; e < tcnt; e++) {
			u32 ent = table + thsz + e * tesz, app, ptr;

			IMG_RD(v, ent, 1, &app);
			IMG_RD(v, ent + 2, 4, &ptr);
			if (app == APPID_FIRMWARE_SEC_LIC || app == want) {
				*desc_off = v->exp_rom_offset + ptr;
				return 0;
			}
		}
	}
	return -ENOENT;
}

/* Locate the DMEM mapper via the application interface table in DMEM. */
static int find_dmem_mapper(struct nv_fwsec *fw)
{
	const u8 *dmem = fw->image + fw->dmem_offset;
	u32 off = fw->interface_offset, hsz, esz, cnt, i;

	if (!in_img(fw->dmem_size, off, 4))
		return -ERANGE;
	hsz = dmem[off + 1];
	esz = dmem[off + 2];
	cnt = dmem[off + 3];
	if (hsz != 4 || esz != 8 || cnt < 2)
		return -EINVAL;

	for (i = 0; i < cnt; i++) {
		u32 ent = off + hsz + i * esz, id, moff;

		if (!in_img(fw->dmem_size, ent, 8))
			return -ERANGE;
		id = get_unaligned_le32(dmem + ent);
		moff = get_unaligned_le32(dmem + ent + 4);
		if (id != IFACE_ID_DMEMMAPPER)
			continue;
		if (!in_img(fw->dmem_size, moff, DMAP_SIZE))
			return -ERANGE;
		if (get_unaligned_le32(dmem + moff) != DMAP_SIG)
			return -EINVAL;
		fw->dmem_mapper = moff;
		return 0;
	}
	return -ENOENT;
}

int nv_fwsec_extract(struct nvdrv_dev *nv)
{
	struct device *dev = &nv->pdev->dev;
	struct nv_vbios *v = &nv->vbios;
	struct nv_fwsec *fw = &nv->fwsec;
	u32 desc, hdr, dsize, stored, img_off, f[9], x;
	int ret, i;

	fw->debug = !(nv_rd32(nv, NV_FUSE_OPT_SECURE_GSP_DEBUG_DIS) &
		      NV_FUSE_OPT_SECURE_GSP_DEBUG_DIS_YES);

	ret = find_fwsec_desc(v, fw->debug, &desc);
	if (ret) {
		dev_err(dev, "fwsec: no %s entry in falcon ucode table\n",
			fw->debug ? "FWSEC_DBG" : "FWSEC_PROD");
		return ret;
	}
	IMG_RD(v, desc, 4, &hdr);
	dsize = hdr >> 16;
	if (!(hdr & 1) || ((hdr >> 8) & 0xff) != 3 || dsize < DESC_V3_SIZE) {
		dev_err(dev, "fwsec: unsupported descriptor hdr 0x%08x at 0x%x\n", hdr, desc);
		return -EINVAL;
	}
	for (i = 0; i < 9; i++)
		IMG_RD(v, desc + 4 * i, 4, &f[i]);
	stored = f[1];
	fw->desc_offset = desc;
	fw->pkc_data_offset = f[2];
	fw->interface_offset = f[3];
	fw->imem_pa = f[4];
	fw->imem_size = f[5];
	fw->imem_va = f[6];
	fw->dmem_pa = f[7];
	fw->dmem_size = f[8];
	IMG_RD(v, desc + 36, 2, &x); fw->engine_id_mask = x;
	IMG_RD(v, desc + 38, 1, &x); fw->ucode_id = x;
	IMG_RD(v, desc + 39, 1, &x); fw->sig_count = x;
	IMG_RD(v, desc + 40, 2, &x); fw->sig_versions = x;

	/* Sanity checks from s_vbiosFillFlcnUcodeFromDescV3(). */
	fw->image_size = ALIGN(stored, 256);
	fw->dmem_offset = fw->imem_size;
	img_off = desc + dsize;
	if (!in_img(v->size, img_off, fw->image_size) ||
	    fw->imem_size > fw->image_size ||
	    fw->dmem_offset >= fw->image_size ||
	    !in_img(fw->image_size, fw->dmem_offset, fw->dmem_size) ||
	    !in_img(fw->dmem_size, fw->pkc_data_offset, NV_FWSEC_SIG_SIZE)) {
		dev_err(dev, "fwsec: descriptor fields out of bounds\n");
		return -EINVAL;
	}

	fw->sigs_size = dsize - DESC_V3_SIZE;
	if (fw->sigs_size < fw->sig_count * NV_FWSEC_SIG_SIZE) {
		dev_err(dev, "fwsec: %u signatures do not fit in 0x%x bytes\n",
			fw->sig_count, fw->sigs_size);
		return -EINVAL;
	}
	fw->sigs = kmemdup(v->data + desc + DESC_V3_SIZE, fw->sigs_size, GFP_KERNEL);
	fw->image = kvmemdup(v->data + img_off, fw->image_size, GFP_KERNEL);
	if (!fw->sigs || !fw->image)
		return -ENOMEM;

	ret = find_dmem_mapper(fw);
	if (ret) {
		dev_err(dev, "fwsec: DMEM mapper not found (%d)\n", ret);
		return ret;
	}

	dev_info(dev, "fwsec: %s desc v3 at 0x%x, IMEM 0x%x DMEM 0x%x, ucode id %u, sigs 0x%x/%u, DMAP at DMEM 0x%x\n",
		 fw->debug ? "DBG" : "PROD", desc, fw->imem_size, fw->dmem_size,
		 fw->ucode_id, fw->sig_versions, fw->sig_count, fw->dmem_mapper);
	return 0;
}

/*
 * Build a copy of the FWSEC image patched to run the FRTS command
 * (s_prepareForFwsec_TU102()): signature for this chip's fuse version into
 * DMEM at pkc_data_offset, init_cmd = FRTS, FRTS_CMD into cmd_in_buffer.
 */
int nv_fwsec_prepare_frts(struct nvdrv_dev *nv, u64 frts_offset, u8 **out)
{
	struct nv_fwsec *fw = &nv->fwsec;
	struct fwseclic_frts_cmd cmd = {};
	u32 fuse_ver, ver_bit, sig_idx, sig_off, cmd_in, cmd_in_size;
	u8 *img, *dmem, *dmap;

	if (!fw->image || !frts_offset)
		return -EINVAL;

	fuse_ver = nv_fuse_ucode_version(nv, fw->ucode_id);
	if (fuse_ver >= 16)
		return -EINVAL;
	ver_bit = BIT(fuse_ver);
	if (!(ver_bit & fw->sig_versions)) {
		dev_err(&nv->pdev->dev, "fwsec: no signature for fuse version %u (have 0x%x)\n",
			fuse_ver, fw->sig_versions);
		return -ENOENT;
	}
	sig_idx = hweight32(fw->sig_versions & (ver_bit - 1));
	sig_off = sig_idx * NV_FWSEC_SIG_SIZE;
	if (!in_img(fw->sigs_size, sig_off, NV_FWSEC_SIG_SIZE))
		return -ERANGE;

	img = kvmemdup(fw->image, fw->image_size, GFP_KERNEL);
	if (!img)
		return -ENOMEM;
	dmem = img + fw->dmem_offset;
	dmap = dmem + fw->dmem_mapper;

	cmd.read_vbios.version = cpu_to_le32(1);
	cmd.read_vbios.size = cpu_to_le32(sizeof(cmd.read_vbios));
	cmd.read_vbios.flags = cpu_to_le32(READ_VBIOS_FLAGS);
	cmd.frts_region.version = cpu_to_le32(1);
	cmd.frts_region.size = cpu_to_le32(sizeof(cmd.frts_region));
	cmd.frts_region.frts_region_offset_4k = cpu_to_le32(frts_offset >> 12);
	cmd.frts_region.frts_region_size = cpu_to_le32(FRTS_SIZE_4K);
	cmd.frts_region.frts_region_media_type = cpu_to_le32(FRTS_MEDIA_FB);

	cmd_in = get_unaligned_le32(dmap + DMAP_CMD_IN_OFFSET);
	cmd_in_size = get_unaligned_le32(dmap + DMAP_CMD_IN_SIZE);
	if (cmd_in_size < sizeof(cmd) || !in_img(fw->dmem_size, cmd_in, sizeof(cmd))) {
		kvfree(img);
		return -ERANGE;
	}

	put_unaligned_le32(DMAP_CMD_FRTS, dmap + DMAP_INIT_CMD);
	memcpy(dmem + cmd_in, &cmd, sizeof(cmd));
	memcpy(dmem + fw->pkc_data_offset, fw->sigs + sig_off, NV_FWSEC_SIG_SIZE);

	dev_info(&nv->pdev->dev, "fwsec: FRTS image prepared (fuse ver %u -> sig %u, FRTS @0x%llx)\n",
		 fuse_ver, sig_idx, frts_offset);
	*out = img;
	return 0;
}

void nv_vbios_fini(struct nvdrv_dev *nv)
{
	kvfree(nv->fwsec_frts);
	kvfree(nv->fwsec.image);
	kfree(nv->fwsec.sigs);
	kvfree(nv->vbios.data);
}
