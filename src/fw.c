// SPDX-License-Identifier: GPL-2.0
/*
 * GSP boot firmware loading (Phase 5, host side). GPU access: fuse reads only.
 *
 * linux-firmware ships RM's bindata archives as files (decompressed from .zst
 * by the firmware loader):
 *   booter_{load,unload}-VER.bin  bin header -> HS v2 header -> image/header/sigs/patch
 *                                 (s_allocateUcodeFromBinArchive, kernel_gsp_booter.c)
 *   bootloader-VER.bin            bin header -> RM_RISCV_UCODE_DESC + image
 *                                 (kgspPrepareBootBinaryImage, kernel_gsp.c)
 *   gsp-VER.bin                   ELF: .fwversion, .fwimage, .fwsignature_<family>
 *                                 (_kgspPrepareGspRmBinaryImage, kgspCreateRadix3)
 */
#include <linux/crc32.h>
#include <linux/dma-mapping.h>
#include <linux/firmware.h>
#include <linux/overflow.h>
#include <linux/scatterlist.h>
#include <linux/slab.h>
#include <linux/string.h>
#include <linux/unaligned.h>
#include <linux/vmalloc.h>

#include "nvdrv.h"
#include "nv_regs.h"

#define BIN_MAGIC		0x10de
#define RADIX3_PAGE_SHIFT	12	/* LIBOS_MEMORY_REGION_RADIX_PAGE_LOG2 */
#define RADIX3_PAGE_SIZE	BIT(RADIX3_PAGE_SHIFT)
#define RADIX3_ENTRIES_SHIFT	(RADIX3_PAGE_SHIFT - 3)

static_assert(PAGE_SIZE == RADIX3_PAGE_SIZE);

MODULE_FIRMWARE("nvidia/ga104/gsp/booter_load-" NV_GSP_FW_VERSION ".bin");
MODULE_FIRMWARE("nvidia/ga104/gsp/booter_unload-" NV_GSP_FW_VERSION ".bin");
MODULE_FIRMWARE("nvidia/ga104/gsp/bootloader-" NV_GSP_FW_VERSION ".bin");
MODULE_FIRMWARE("nvidia/ga104/gsp/gsp-" NV_GSP_FW_VERSION ".bin");

static bool fw_in(size_t total, u64 off, u64 len)
{
	u64 end;

	return !check_add_overflow(off, len, &end) && end <= total;
}

static int fw_request(struct nvdrv_dev *nv, const char *what, const struct firmware **fw)
{
	char path[96], chip[8];
	int ret;

	strscpy(chip, nv->name);
	string_lower(chip, chip);
	snprintf(path, sizeof(path), "nvidia/%s/gsp/%s-%s.bin", chip, what, NV_GSP_FW_VERSION);
	ret = request_firmware(fw, path, &nv->pdev->dev);
	if (ret)
		dev_err(&nv->pdev->dev, "fw: cannot load %s: %d\n", path, ret);
	return ret;
}

/* Common container: {u32 magic, version, size, header_offset, data_offset, data_size}. */
static int fw_bin_hdr(const struct firmware *fw, u32 *hdr_off, u32 *data_off, u32 *data_size)
{
	if (fw->size < 24 || get_unaligned_le32(fw->data) != BIN_MAGIC ||
	    get_unaligned_le32(fw->data + 4) != 1)
		return -EINVAL;
	*hdr_off = get_unaligned_le32(fw->data + 12);
	*data_off = get_unaligned_le32(fw->data + 16);
	*data_size = get_unaligned_le32(fw->data + 20);
	/* The size field (+8) is not the file length; bound the data range instead. */
	if (!fw_in(fw->size, *data_off, *data_size) || *hdr_off >= fw->size)
		return -EINVAL;
	return 0;
}

static u32 sec2_fuse_ucode_version(struct nvdrv_dev *nv, u32 ucode_id)
{
	u32 index = ucode_id - 1, val;

	if (index >= 16)
		return 0;
	val = nv_rd32(nv, NV_FUSE_OPT_FPF_SEC2_UCODE1_VERSION + 4 * index);
	return val ? fls(val) : 0;
}

/* ---- Booter ------------------------------------------------------------ */

static int fw_load_booter(struct nvdrv_dev *nv, const char *what, struct nv_hs_ucode *u)
{
	struct device *dev = &nv->pdev->dev;
	const struct firmware *fw;
	u32 hdr_off, data_off, data_size, h[9], hdr[9], meta[3];
	u32 patch_loc, patch_sig, num_sigs, sig_size, fuse, idx;
	int ret, i;

	ret = fw_request(nv, what, &fw);
	if (ret)
		return ret;
	ret = -EINVAL;
	if (fw_bin_hdr(fw, &hdr_off, &data_off, &data_size) || !fw_in(fw->size, hdr_off, sizeof(h)))
		goto bad;

	/* HS v2 header: offsets of sig, patch_loc, patch_sig, meta, num_sigs, header. */
	for (i = 0; i < 9; i++)
		h[i] = get_unaligned_le32(fw->data + hdr_off + 4 * i);
	/* h: 0 sig_off 1 sig_size 2 patch_loc 3 patch_sig 4 meta_off 5 meta_size
	 *    6 num_sigs 7 header_off 8 header_size */
	if (h[5] != sizeof(meta) || h[8] != sizeof(hdr) || !fw_in(fw->size, h[0], h[1]) ||
	    !fw_in(fw->size, h[2], 4) || !fw_in(fw->size, h[3], 4) || !fw_in(fw->size, h[4], h[5]) ||
	    !fw_in(fw->size, h[6], 4) || !fw_in(fw->size, h[7], h[8]))
		goto bad;

	for (i = 0; i < 9; i++)
		hdr[i] = get_unaligned_le32(fw->data + h[7] + 4 * i);
	for (i = 0; i < 3; i++)
		meta[i] = get_unaligned_le32(fw->data + h[4] + 4 * i);
	patch_loc = get_unaligned_le32(fw->data + h[2]);
	patch_sig = get_unaligned_le32(fw->data + h[3]);
	num_sigs = get_unaligned_le32(fw->data + h[6]);

	/* hdr: 0 osCodeOff 1 osCodeSize 2 osDataOff 3 osDataSize 4 numApps 5 appCodeOff 6 appCodeSize */
	if (hdr[4] != 1 || patch_sig != 0 || !num_sigs || h[1] % num_sigs)
		goto bad;
	sig_size = h[1] / num_sigs;
	if (!fw_in(data_size, patch_loc, sig_size) || patch_loc < hdr[2] ||
	    !fw_in(data_size, hdr[5], hdr[6]) || !fw_in(data_size, hdr[2], hdr[3]))
		goto bad;

	/* s_patchBooterUcodeSignature: newest signature first, indexed by SEC2 fuse. */
	fuse = sec2_fuse_ucode_version(nv, meta[2]);
	idx = 0;
	if (num_sigs > 1) {
		if (fuse > num_sigs - 1) {
			dev_err(dev, "fw: %s has no signature for SEC2 fuse version %u\n", what, fuse);
			goto bad;
		}
		idx = num_sigs - 1 - fuse;
	}

	u->image = kvmemdup(fw->data + data_off, data_size, GFP_KERNEL);
	if (!u->image) {
		ret = -ENOMEM;
		goto out;
	}
	memcpy(u->image + patch_loc, fw->data + h[0] + idx * sig_size, sig_size);

	/* kgsp HS "boot from HS" fields for Booter */
	u->image_size = data_size;
	u->code_offset = hdr[5];
	u->imem_size = hdr[6];
	u->imem_pa = 0;
	u->imem_va = hdr[5];
	u->dmem_offset = hdr[2];
	u->dmem_size = hdr[3];
	u->dmem_pa = 0;
	u->pkc_data_offset = patch_loc - hdr[2];
	u->engine_id_mask = meta[1];
	u->ucode_id = meta[2];

	dev_info(dev, "fw: %s: image 0x%x crc %08x, IMEM 0x%x@0x%x DMEM 0x%x@0x%x, ucode %u, SEC2 fuse %u -> sig %u/%u\n",
		 what, data_size, ~crc32_le(~0, u->image, data_size), u->imem_size, u->code_offset,
		 u->dmem_size, u->dmem_offset, u->ucode_id, fuse, idx, num_sigs);
	ret = 0;
	goto out;
bad:
	dev_err(dev, "fw: %s: malformed container\n", what);
out:
	release_firmware(fw);
	return ret;
}

/* ---- RISC-V bootloader ------------------------------------------------- */

static int fw_load_bootloader(struct nvdrv_dev *nv)
{
	struct device *dev = &nv->pdev->dev;
	struct nv_gsp_fw *g = &nv->gsp_fw;
	const struct firmware *fw;
	u32 hdr_off, data_off, data_size, *d = (u32 *)&g->bl_desc;
	int ret, i;

	ret = fw_request(nv, "bootloader", &fw);
	if (ret)
		return ret;
	ret = -EINVAL;
	if (fw_bin_hdr(fw, &hdr_off, &data_off, &data_size) ||
	    !fw_in(fw->size, hdr_off, sizeof(g->bl_desc)))
		goto out;
	for (i = 0; i < sizeof(g->bl_desc) / 4; i++)
		d[i] = get_unaligned_le32(fw->data + hdr_off + 4 * i);
	if (!fw_in(data_size, g->bl_desc.monitorCodeOffset, g->bl_desc.monitorCodeSize) ||
	    !fw_in(data_size, g->bl_desc.monitorDataOffset, g->bl_desc.monitorDataSize) ||
	    !fw_in(data_size, g->bl_desc.manifestOffset, g->bl_desc.manifestSize))
		goto out;

	/* kgspPrepareBootBinaryImage: sysmem, 4K aligned allocation, size = image size. */
	g->bl_size = data_size;
	g->bl_alloc = ALIGN(data_size, SZ_4K);
	g->bl = dma_alloc_coherent(dev, g->bl_alloc, &g->bl_dma, GFP_KERNEL);
	if (!g->bl) {
		ret = -ENOMEM;
		goto out;
	}
	memcpy(g->bl, fw->data + data_off, data_size);
	dev_info(dev, "fw: bootloader: desc v%u, image 0x%x crc %08x at %pad, monitor code 0x%x+0x%x data 0x%x+0x%x manifest 0x%x+0x%x\n",
		 g->bl_desc.version, data_size, ~crc32_le(~0, g->bl, data_size), &g->bl_dma,
		 g->bl_desc.monitorCodeOffset, g->bl_desc.monitorCodeSize,
		 g->bl_desc.monitorDataOffset, g->bl_desc.monitorDataSize,
		 g->bl_desc.manifestOffset, g->bl_desc.manifestSize);
	ret = 0;
out:
	if (ret == -EINVAL)
		dev_err(dev, "fw: bootloader: malformed container\n");
	release_firmware(fw);
	return ret;
}

/* ---- GSP ELF + radix-3 -------------------------------------------------- */

/* _kgspFwContainerGetSection: exact-name lookup in a little-endian ELF64. */
static int elf_section(const struct firmware *fw, const char *name, const u8 **data, u64 *size)
{
	const u8 *d = fw->data;
	u64 shoff, stroff, strsize, len = strlen(name);
	u16 shentsize, shnum, shstrndx;
	int i;

	if (fw->size < 64 || memcmp(d, "\x7f" "ELF", 4) || d[4] != 2 || d[5] != 1)
		return -EINVAL;
	shoff = get_unaligned_le64(d + 0x28);
	shentsize = get_unaligned_le16(d + 0x3a);
	shnum = get_unaligned_le16(d + 0x3c);
	shstrndx = get_unaligned_le16(d + 0x3e);
	if (shentsize != 64 || shstrndx >= shnum || !fw_in(fw->size, shoff, (u64)shnum * 64))
		return -EINVAL;
	stroff = get_unaligned_le64(d + shoff + shstrndx * 64 + 0x18);
	strsize = get_unaligned_le64(d + shoff + shstrndx * 64 + 0x20);
	if (!strsize || !fw_in(fw->size, stroff, strsize))
		return -EINVAL;

	for (i = 0; i < shnum; i++) {
		const u8 *sh = d + shoff + i * 64;
		u32 nm = get_unaligned_le32(sh);
		u64 off = get_unaligned_le64(sh + 0x18), sz = get_unaligned_le64(sh + 0x20);

		if (nm >= strsize || !fw_in(fw->size, off, sz))
			return -EINVAL;
		if (len < strsize - nm && !memcmp(d + stroff + nm, name, len) &&
		    d[stroff + nm + len] == '\0') {
			*data = d + off;
			*size = sz;
			return 0;
		}
	}
	return -ENOENT;
}

static void radix3_free(struct nvdrv_dev *nv, struct nv_radix3 *r)
{
	u32 i;

	if (r->mapped)
		dma_unmap_sgtable(&nv->pdev->dev, &r->sgt, DMA_TO_DEVICE, 0);
	if (r->sgt.sgl)
		sg_free_table(&r->sgt);
	if (r->vaddr)
		vunmap(r->vaddr);
	if (r->pages)
		for (i = 0; i < r->npages; i++)
			if (r->pages[i])
				__free_page(r->pages[i]);
	kvfree(r->pages);
	kvfree(r->dma);
	memset(r, 0, sizeof(*r));
}

/*
 * kgspCreateRadix3: one buffer of 4K pages laid out [L0][L1][L2][data].
 * Each Ln page holds the u64 bus addresses of the L(n+1) pages, 512 per page;
 * L0 is a single page whose bus address is what GSP gets (sysmemAddrOfRadix3Elf).
 * Pages are allocated individually, so no large physically contiguous block
 * is needed with or without an IOMMU.
 */
static int radix3_build(struct nvdrv_dev *nv, struct nv_radix3 *r, const u8 *data, u64 size)
{
	struct device *dev = &nv->pdev->dev;
	struct sg_dma_page_iter it;
	u32 i, n, off;
	u64 *pt;
	int ret;

	r->lvl_pages[3] = DIV_ROUND_UP(size, RADIX3_PAGE_SIZE);
	for (i = 3; i > 0; i--)
		r->lvl_pages[i - 1] = ((r->lvl_pages[i] - 1) >> RADIX3_ENTRIES_SHIFT) + 1;
	if (r->lvl_pages[0] != 1)
		return -E2BIG;
	for (i = 0, off = 0; i < 4; i++) {
		r->lvl_first[i] = off;
		off += r->lvl_pages[i];
	}
	r->npages = off;
	r->size = size;

	r->pages = kvcalloc(r->npages, sizeof(*r->pages), GFP_KERNEL);
	r->dma = kvcalloc(r->npages, sizeof(*r->dma), GFP_KERNEL);
	if (!r->pages || !r->dma) {
		ret = -ENOMEM;
		goto err;
	}
	for (i = 0; i < r->npages; i++) {
		r->pages[i] = alloc_page(GFP_KERNEL | __GFP_ZERO);
		if (!r->pages[i]) {
			ret = -ENOMEM;
			goto err;
		}
	}
	r->vaddr = vmap(r->pages, r->npages, VM_MAP, PAGE_KERNEL);
	if (!r->vaddr) {
		ret = -ENOMEM;
		goto err;
	}
	ret = sg_alloc_table_from_pages(&r->sgt, r->pages, r->npages, 0,
					(size_t)r->npages << PAGE_SHIFT, GFP_KERNEL);
	if (ret)
		goto err;
	ret = dma_map_sgtable(dev, &r->sgt, DMA_TO_DEVICE, 0);
	if (ret)
		goto err;
	r->mapped = true;

	n = 0;
	for_each_sgtable_dma_page(&r->sgt, &it, 0) {
		if (n < r->npages)
			r->dma[n] = sg_page_iter_dma_address(&it);
		n++;
	}
	if (n != r->npages) {
		ret = -EIO;
		goto err;
	}

	/* Data first, then every level's entries point at the next level's pages. */
	memcpy(r->vaddr + ((size_t)r->lvl_first[3] << PAGE_SHIFT), data, size);
	pt = r->vaddr;
	for (i = 0; i < 3; i++) {
		u64 *ents = pt + ((size_t)r->lvl_first[i] << RADIX3_ENTRIES_SHIFT);
		u32 j;

		for (j = 0; j < r->lvl_pages[i + 1]; j++)
			ents[j] = cpu_to_le64(r->dma[r->lvl_first[i + 1] + j]);
	}
	dma_sync_sgtable_for_device(dev, &r->sgt, DMA_TO_DEVICE);
	r->root = r->dma[0];
	return 0;
err:
	radix3_free(nv, r);
	return ret;
}

/* Walk the table the way GSP would (bus address -> page) and compare with the source. */
static int radix3_verify(struct nvdrv_dev *nv, struct nv_radix3 *r, const u8 *data)
{
	const u64 *pt = r->vaddr;
	u32 lvl, j;

	for (lvl = 0; lvl < 3; lvl++) {
		const u64 *ents = pt + ((size_t)r->lvl_first[lvl] << RADIX3_ENTRIES_SHIFT);
		u32 cap = r->lvl_pages[lvl] << RADIX3_ENTRIES_SHIFT;

		for (j = 0; j < cap; j++) {
			u64 want = j < r->lvl_pages[lvl + 1] ? r->dma[r->lvl_first[lvl + 1] + j] : 0;

			if (le64_to_cpu(ents[j]) != want)
				return -EIO;
		}
	}
	if (memcmp(r->vaddr + ((size_t)r->lvl_first[3] << PAGE_SHIFT), data, r->size))
		return -EIO;
	return 0;
}

static int fw_load_gsp(struct nvdrv_dev *nv)
{
	struct device *dev = &nv->pdev->dev;
	struct nv_gsp_fw *g = &nv->gsp_fw;
	const struct firmware *fw;
	const u8 *ver, *img, *sig;
	u64 ver_size, img_size, sig_size;
	int ret;

	ret = fw_request(nv, "gsp", &fw);
	if (ret)
		return ret;

	/* _kgspFwContainerVerifyVersion: exact match including the terminator. */
	ret = elf_section(fw, ".fwversion", &ver, &ver_size);
	if (ret || ver_size != sizeof(NV_GSP_FW_VERSION) ||
	    memcmp(ver, NV_GSP_FW_VERSION, sizeof(NV_GSP_FW_VERSION))) {
		dev_err(dev, "fw: gsp: .fwversion missing or not " NV_GSP_FW_VERSION "\n");
		ret = -EINVAL;
		goto out;
	}
	ret = elf_section(fw, ".fwimage", &img, &img_size);
	if (!ret)
		ret = elf_section(fw, ".fwsignature_ga10x", &sig, &sig_size);
	if (ret) {
		dev_err(dev, "fw: gsp: missing ELF section: %d\n", ret);
		goto out;
	}

	g->sig_size = sig_size;
	g->sig = dma_alloc_coherent(dev, sig_size, &g->sig_dma, GFP_KERNEL);
	if (!g->sig) {
		ret = -ENOMEM;
		goto out;
	}
	memcpy(g->sig, sig, sig_size);

	ret = radix3_build(nv, &g->radix3, img, img_size);
	if (!ret)
		ret = radix3_verify(nv, &g->radix3, img);
	if (ret) {
		dev_err(dev, "fw: gsp: radix3 build/verify failed: %d\n", ret);
		goto out;
	}
	g->img_crc = ~crc32_le(~0, img, img_size);
	g->sig_crc = ~crc32_le(~0, sig, sig_size);
	dev_info(dev, "fw: gsp " NV_GSP_FW_VERSION ": image 0x%llx crc %08x, sig 0x%x crc %08x, radix3 %u/%u/%u/%u pages, root %pad (verified)\n",
		 img_size, g->img_crc, g->sig_size, g->sig_crc,
		 g->radix3.lvl_pages[0], g->radix3.lvl_pages[1], g->radix3.lvl_pages[2],
		 g->radix3.lvl_pages[3], &g->radix3.root);
out:
	release_firmware(fw);
	return ret;
}

/* ---- entry points ------------------------------------------------------- */

int nv_fw_load(struct nvdrv_dev *nv)
{
	int ret;

	ret = fw_load_booter(nv, "booter_load", &nv->gsp_fw.booter_load);
	if (!ret)
		ret = fw_load_booter(nv, "booter_unload", &nv->gsp_fw.booter_unload);
	if (!ret)
		ret = fw_load_bootloader(nv);
	if (!ret)
		ret = fw_load_gsp(nv);
	return ret;
}

void nv_fw_fini(struct nvdrv_dev *nv)
{
	struct nv_gsp_fw *g = &nv->gsp_fw;
	struct device *dev = &nv->pdev->dev;

	radix3_free(nv, &g->radix3);
	if (g->sig)
		dma_free_coherent(dev, g->sig_size, g->sig, g->sig_dma);
	if (g->bl)
		dma_free_coherent(dev, g->bl_alloc, g->bl, g->bl_dma);
	kvfree(g->booter_load.image);
	kvfree(g->booter_unload.image);
	memset(g, 0, sizeof(*g));
}
