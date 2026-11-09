// SPDX-License-Identifier: GPL-2.0
/*
 * nvdrv - from-scratch NVIDIA GPU driver (Ampere GA10x first).
 *
 * Binds only via driver_override. Current state: read-only.
 *   Phase 1: chip identification, BAR0 debug access
 *   Phase 2: VBIOS copy, FWSEC extraction, FRTS image prepared in memory
 *   Phase 4 (partial): FB layout down to the FRTS region
 *
 * debugfs (/sys/kernel/debug/nvdrv/<bdf>/):
 *   info            chip, boot status, fuses, WPR2, FB layout, FWSEC summary
 *   mmio            pread() at offset N = BAR0 dword N (4-byte aligned)
 *   prom            raw PROM window (1 MiB)
 *   vbios           VBIOS image as copied at probe (through the last ROM image)
 *   fwsec_frts.bin  FWSEC image patched for FRTS (what Phase 5 will upload)
 *   falcon          GSP falcon register snapshot
 *   run_fwsec_frts  (only with allow_hw_writes=1) write "1" to run FWSEC-FRTS
 *   run_gsp_boot    (only with allow_hw_writes=1) write "1" to boot GSP-RM, then
 *                   fetch GET_GSP_STATIC_INFO
 *   gsp_static_info.bin  raw GspStaticConfigInfo reply (zeros until GSP-RM booted)
 *   rm_selftest     (only with allow_hw_writes=1) write "1": RM error paths + alloc/free
 *
 * Probe is read-only: no register writes, no DMA, BAR1 untouched (the boot
 * framebuffer lives there). Writes happen only through explicit triggers.
 */
#include <linux/aperture.h>
#include <linux/bitfield.h>
#include <linux/debugfs.h>
#include <linux/iopoll.h>
#include <linux/module.h>
#include <linux/seq_file.h>
#include <linux/slab.h>
#include <linux/uaccess.h>

#include "nvdrv.h"
#include "nv_regs.h"

static struct dentry *nvdrv_dbg_root;

static const struct {
	u16 chipset;
	const char *name;
} nv_chips[] = {
	{ 0x162, "TU102" }, { 0x164, "TU104" }, { 0x166, "TU106" },
	{ 0x167, "TU117" }, { 0x168, "TU116" },
	{ 0x170, "GA100" }, { 0x172, "GA102" }, { 0x173, "GA103" },
	{ 0x174, "GA104" }, { 0x176, "GA106" }, { 0x177, "GA107" },
	{ 0x192, "AD102" }, { 0x193, "AD103" }, { 0x194, "AD104" },
	{ 0x196, "AD106" }, { 0x197, "AD107" },
};

static const char *nv_chip_name(u16 chipset)
{
	for (int i = 0; i < ARRAY_SIZE(nv_chips); i++)
		if (nv_chips[i].chipset == chipset)
			return nv_chips[i].name;
	return NULL;
}

static int nv_identify(struct nvdrv_dev *nv)
{
	struct device *dev = &nv->pdev->dev;

	nv->boot0 = nv_rd32(nv, NV_PMC_BOOT_0);
	if (nv->boot0 == NV_DEAD) {
		dev_err(dev, "BOOT_0 reads 0xffffffff: device not responding\n");
		return -ENODEV;
	}
	nv->boot42 = nv_rd32(nv, NV_PMC_BOOT_42);

	/* Prefer BOOT_42 (authoritative on Ampere+), fall back to BOOT_0. */
	if (nv->boot42 && nv->boot42 != NV_DEAD) {
		nv->chipset = FIELD_GET(NV_PMC_BOOT_42_CHIPSET, nv->boot42);
		nv->rev = FIELD_GET(NV_PMC_BOOT_42_MAJOR_REVISION, nv->boot42) << 4 |
			  FIELD_GET(NV_PMC_BOOT_42_MINOR_REVISION, nv->boot42);
	} else {
		nv->chipset = FIELD_GET(NV_PMC_BOOT_0_CHIPSET, nv->boot0);
		nv->rev = FIELD_GET(NV_PMC_BOOT_0_MAJOR_REVISION, nv->boot0) << 4 |
			  FIELD_GET(NV_PMC_BOOT_0_MINOR_REVISION, nv->boot0);
	}

	nv->name = nv_chip_name(nv->chipset);
	if (!nv->name) {
		dev_err(dev, "unsupported chipset 0x%03x (BOOT_0=%08x BOOT_42=%08x)\n",
			nv->chipset, nv->boot0, nv->boot42);
		return -ENODEV;
	}
	return 0;
}

/* Only valid once devinit (part of GFW boot) has configured the memory controller. */
static void nv_vram_probe(struct nvdrv_dev *nv)
{
	u32 lmr = nv_rd32(nv, NV_PFB_PRI_MMU_LOCAL_MEMORY_RANGE);

	nv->vram_lmr = (u64)FIELD_GET(NV_PFB_LMR_LOWER_MAG, lmr)
		       << (FIELD_GET(NV_PFB_LMR_LOWER_SCALE, lmr) + 20);
	if (lmr & NV_PFB_LMR_ECC_MODE)
		nv->vram_lmr = nv->vram_lmr / 4 * 3;
	dev_info(&nv->pdev->dev, "VRAM %llu MiB\n", nv->vram_lmr >> 20);
}

/* ---- debugfs ---------------------------------------------------------- */

static int info_show(struct seq_file *m, void *unused)
{
	struct nvdrv_dev *nv = m->private;
	struct nv_fwsec *fw = &nv->fwsec;
	struct nv_fb_layout *fb = &nv->fb;
	u32 gfw, scratch0e;
	bool done = nv_gfw_boot_completed(nv, &gfw);

	scratch0e = nv_rd32(nv, NV_PBUS_VBIOS_SCRATCH(NV_VBIOS_FWSECLIC_SCRATCH_INDEX_0E));

	seq_printf(m, "chip:            %s (chipset 0x%03x rev %02x)\n", nv->name, nv->chipset, nv->rev);
	seq_printf(m, "boot0/boot42:    0x%08x / 0x%08x\n", nv->boot0, nv->boot42);
	seq_printf(m, "bar0/1/3:        %pR %pR %pR\n", &nv->pdev->resource[0],
		   &nv->pdev->resource[1], &nv->pdev->resource[3]);
	seq_printf(m, "vram (lmr):      %llu MiB\n", nv->vram_lmr >> 20);
	seq_printf(m, "gfw boot:        progress 0x%02x%s\n", gfw, done ? " (completed)" : "");
	seq_printf(m, "frts err (0x0e): 0x%04lx\n", FIELD_GET(NV_VBIOS_FWSECLIC_FRTS_ERR_CODE, scratch0e));
	seq_printf(m, "wpr2:            0x%llx - 0x%llx%s\n", nv_wpr2_lo(nv), nv_wpr2_hi(nv),
		   nv_wpr2_hi(nv) ? "" : " (not set up)");

	seq_puts(m, "\n[vbios]\n");
	seq_printf(m, "size:            0x%x\n", nv->vbios.size);
	seq_printf(m, "bit:             0x%x\n", nv->vbios.bit_offset);
	seq_printf(m, "exp rom offset:  0x%x\n", nv->vbios.exp_rom_offset);

	seq_puts(m, "\n[fwsec]\n");
	if (fw->image) {
		seq_printf(m, "variant:         %s\n", fw->debug ? "FWSEC_DBG" : "FWSEC_PROD");
		seq_printf(m, "desc:            0x%x\n", fw->desc_offset);
		seq_printf(m, "image:           0x%x bytes (IMEM 0x%x @pa 0x%x va 0x%x, DMEM 0x%x @pa 0x%x)\n",
			   fw->image_size, fw->imem_size, fw->imem_pa, fw->imem_va,
			   fw->dmem_size, fw->dmem_pa);
		seq_printf(m, "ucode id:        %u (fuse version %u)\n", fw->ucode_id,
			   nv_fuse_ucode_version(nv, fw->ucode_id));
		seq_printf(m, "signatures:      %u, versions 0x%x, patch at DMEM 0x%x\n",
			   fw->sig_count, fw->sig_versions, fw->pkc_data_offset);
		seq_printf(m, "engine mask:     0x%x\n", fw->engine_id_mask);
		seq_printf(m, "interface/DMAP:  DMEM 0x%x / 0x%x\n", fw->interface_offset, fw->dmem_mapper);
		seq_printf(m, "frts image:      %s\n", nv->fwsec_frts ? "prepared" : "not prepared");
	} else {
		seq_puts(m, "not extracted\n");
	}

	seq_puts(m, "\n[gsp firmware " NV_GSP_FW_VERSION "]\n");
	if (nv->gsp_fw.radix3.root) {
		struct nv_gsp_fw *g = &nv->gsp_fw;
		const struct nv_hs_ucode *b = &g->booter_load;

		seq_printf(m, "booter_load:     0x%x bytes, IMEM 0x%x@0x%x DMEM 0x%x@0x%x, sig at DMEM 0x%x, ucode %u, engines 0x%x\n",
			   b->image_size, b->imem_size, b->code_offset, b->dmem_size, b->dmem_offset,
			   b->pkc_data_offset, b->ucode_id, b->engine_id_mask);
		b = &g->booter_unload;
		seq_printf(m, "booter_unload:   0x%x bytes, IMEM 0x%x@0x%x DMEM 0x%x@0x%x, sig at DMEM 0x%x, ucode %u, engines 0x%x\n",
			   b->image_size, b->imem_size, b->code_offset, b->dmem_size, b->dmem_offset,
			   b->pkc_data_offset, b->ucode_id, b->engine_id_mask);
		seq_printf(m, "bootloader:      0x%x bytes at %pad (desc v%u, fbReserved 0x%x)\n",
			   g->bl_size, &g->bl_dma, g->bl_desc.version, g->bl_desc.fbReservedSize);
		seq_printf(m, "gsp image:       0x%llx bytes crc %08x, radix3 root %pad (%u/%u/%u/%u pages)\n",
			   g->radix3.size, g->img_crc, &g->radix3.root, g->radix3.lvl_pages[0],
			   g->radix3.lvl_pages[1], g->radix3.lvl_pages[2], g->radix3.lvl_pages[3]);
		seq_printf(m, "gsp signature:   0x%x bytes crc %08x at %pad\n", g->sig_size, g->sig_crc, &g->sig_dma);
	} else {
		seq_puts(m, "not loaded\n");
	}

	seq_printf(m, "gsp:             %s\n", nv->gsp_running ? "RUNNING (INIT_DONE received)" :
		   nv->gsp_touched ? "boot attempted" : "not booted");
	seq_printf(m, "gsp nocat:       %u diagnostic records\n", nv->gsp_nocat);

	seq_puts(m, "\n[libos / rpc queues]\n");
	if (nv->libos.shm.cpu) {
		struct nv_libos *l = &nv->libos;
		int i;

		seq_printf(m, "init args:       %pad\n", &l->args.dma);
		seq_printf(m, "rm args:         %pad\n", &l->rmargs.dma);
		for (i = 0; i < NV_LIBOS_NUM_LOGS; i++)
			seq_printf(m, "log %d:           %pad + 0x%zx\n", i, &l->logs[i].dma, l->logs[i].size);
		seq_printf(m, "queue shm:       %pad + 0x%zx (%u PTEs)\n", &l->shm.dma, l->shm.size, l->pte_count);
	} else {
		seq_puts(m, "not set up\n");
	}

	seq_puts(m, "\n[fb layout]\n");
	if (fb->fb_size) {
		seq_printf(m, "fb size:         0x%llx\n", fb->fb_size);
		seq_printf(m, "vga workspace:   0x%llx + 0x%llx%s\n", fb->vga_ws_offset, fb->vga_ws_size,
			   fb->vga_ws_from_hw ? "" : " (default)");
		seq_printf(m, "mmu lock:        0x%llx - 0x%llx (%s)\n", fb->mmu_lock_lo,
			   fb->mmu_lock_hi, fb->mmu_lock_valid ? "used" : "ignored on this chip");
		seq_printf(m, "wpr end:         0x%llx\n", fb->wpr_end);
		seq_printf(m, "frts:            0x%llx + 0x%llx\n", fb->frts_offset, fb->frts_size);
		if (fb->rsvd_start) {
			seq_printf(m, "boot bin:        0x%llx + 0x%llx\n", fb->boot_bin_offset, fb->boot_bin_size);
			seq_printf(m, "gsp elf:         0x%llx + 0x%llx\n", fb->gsp_fw_offset, fb->gsp_fw_size);
			seq_printf(m, "gsp heap:        0x%llx + 0x%llx\n", fb->heap_offset, fb->heap_size);
			seq_printf(m, "wpr start:       0x%llx (meta)\n", fb->wpr_start);
			seq_printf(m, "non-wpr heap:    0x%llx + 0x%llx\n", fb->non_wpr_heap_offset, fb->non_wpr_heap_size);
			seq_printf(m, "gsp reserved:    0x%llx - 0x%llx (%llu MiB)\n", fb->rsvd_start, fb->fb_size,
				   (fb->fb_size - fb->rsvd_start) >> 20);
		}
		if (nv->wpr_meta)
			seq_printf(m, "wpr meta (sys):  %pad\n", &nv->wpr_meta_dma);
	} else {
		seq_puts(m, "not computed\n");
	}
	mutex_lock(&nv->gsp_lock);
	nv_rm_static_info_show(nv, m);
	nv_rm_objects_show(nv, m);
	nv_disp_show(nv, m);
	nv_evo_show(nv, m);
	mutex_unlock(&nv->gsp_lock);
	return 0;
}
DEFINE_SHOW_ATTRIBUTE(info);

/* Read-only window onto BAR0; offset and length must be dword aligned. */
static ssize_t mmio_read(struct file *f, char __user *buf, size_t count, loff_t *ppos)
{
	struct nvdrv_dev *nv = f->private_data;
	loff_t pos = *ppos;
	size_t done = 0;

	if (pos < 0 || (pos & 3) || (count & 3))
		return -EINVAL;
	if (pos >= nv->bar0_len)
		return 0;
	count = min_t(size_t, count, nv->bar0_len - pos);
	count = min_t(size_t, count, PAGE_SIZE);

	while (done < count) {
		u32 v = nv_rd32(nv, pos + done);

		if (copy_to_user(buf + done, &v, 4))
			return done ? done : -EFAULT;
		done += 4;
	}
	*ppos = pos + done;
	return done;
}

static const struct file_operations mmio_fops = {
	.owner = THIS_MODULE,
	.open = simple_open,
	.read = mmio_read,
	.llseek = default_llseek,
};

/* Byte-granular read of the raw PROM window, assembled from dword reads. */
static ssize_t prom_read(struct file *f, char __user *buf, size_t count, loff_t *ppos)
{
	struct nvdrv_dev *nv = f->private_data;
	loff_t pos = *ppos;
	size_t done = 0;

	if (pos < 0)
		return -EINVAL;
	if (pos >= NV_PROM_SIZE)
		return 0;
	count = min_t(size_t, count, NV_PROM_SIZE - pos);
	count = min_t(size_t, count, SZ_64K);

	while (done < count) {
		u32 off = pos + done;
		__le32 w = cpu_to_le32(nv_rd32(nv, NV_PROM_DATA + (off & ~3u)));
		unsigned int skip = off & 3;
		size_t n = min_t(size_t, 4 - skip, count - done);

		if (copy_to_user(buf + done, (u8 *)&w + skip, n))
			return done ? done : -EFAULT;
		done += n;
	}
	*ppos = pos + done;
	return done;
}

static const struct file_operations prom_fops = {
	.owner = THIS_MODULE,
	.open = simple_open,
	.read = prom_read,
	.llseek = default_llseek,
};

static int falcon_show(struct seq_file *m, void *unused)
{
	nv_falcon_dump(m->private, m);
	return 0;
}
DEFINE_SHOW_ATTRIBUTE(falcon);

static ssize_t run_fwsec_frts_write(struct file *f, const char __user *buf,
				    size_t count, loff_t *ppos)
{
	struct nvdrv_dev *nv = f->private_data;
	bool go;
	int ret;

	ret = kstrtobool_from_user(buf, count, &go);
	if (ret)
		return ret;
	if (!go)
		return count;
	ret = nv_fwsec_run_frts(nv);
	return ret ? ret : count;
}

static const struct file_operations run_fwsec_frts_fops = {
	.owner = THIS_MODULE,
	.open = simple_open,
	.write = run_fwsec_frts_write,
};

static ssize_t run_gsp_boot_write(struct file *f, const char __user *buf,
				  size_t count, loff_t *ppos)
{
	struct nvdrv_dev *nv = f->private_data;
	bool go;
	int ret;

	ret = kstrtobool_from_user(buf, count, &go);
	if (ret)
		return ret;
	if (!go)
		return count;
	ret = nv_gsp_boot(nv);
	return ret ? ret : count;
}

static const struct file_operations run_gsp_boot_fops = {
	.owner = THIS_MODULE,
	.open = simple_open,
	.write = run_gsp_boot_write,
};

static ssize_t rm_selftest_write(struct file *f, const char __user *buf,
				 size_t count, loff_t *ppos)
{
	bool go;
	int ret = kstrtobool_from_user(buf, count, &go);

	if (ret)
		return ret;
	if (go)
		ret = nv_rm_selftest(f->private_data);
	return ret ? ret : count;
}

static const struct file_operations rm_selftest_fops = {
	.owner = THIS_MODULE,
	.open = simple_open,
	.write = rm_selftest_write,
};

static ssize_t disp_probe_write(struct file *f, const char __user *buf,
				size_t count, loff_t *ppos)
{
	bool go;
	int ret = kstrtobool_from_user(buf, count, &go);

	if (ret)
		return ret;
	if (go)
		ret = nv_disp_probe(f->private_data);
	return ret ? ret : count;
}

static const struct file_operations disp_probe_fops = {
	.owner = THIS_MODULE,
	.open = simple_open,
	.write = disp_probe_write,
};

static ssize_t disp_link_write(struct file *f, const char __user *buf,
			       size_t count, loff_t *ppos)
{
	bool go;
	int ret = kstrtobool_from_user(buf, count, &go);

	if (ret)
		return ret;
	if (go)
		ret = nv_disp_link(f->private_data);
	return ret ? ret : count;
}

static const struct file_operations disp_link_fops = {
	.owner = THIS_MODULE,
	.open = simple_open,
	.write = disp_link_write,
};

static ssize_t disp_modeset_write(struct file *f, const char __user *buf,
				  size_t count, loff_t *ppos)
{
	u32 flags;
	int ret = kstrtou32_from_user(buf, count, 0, &flags);

	if (ret)
		return ret;
	ret = nv_disp_modeset(f->private_data, flags);
	return ret ? ret : count;
}

static const struct file_operations disp_modeset_fops = {
	.owner = THIS_MODULE,
	.open = simple_open,
	.write = disp_modeset_write,
};

static ssize_t disp_flip_write(struct file *f, const char __user *buf,
			       size_t count, loff_t *ppos)
{
	u32 flags;
	int ret = kstrtou32_from_user(buf, count, 0, &flags);

	if (ret)
		return ret;
	ret = nv_disp_flip(f->private_data, flags);
	return ret ? ret : count;
}

static const struct file_operations disp_flip_fops = {
	.owner = THIS_MODULE,
	.open = simple_open,
	.write = disp_flip_write,
};

/*
 * vram: pread() at offset N = VRAM byte N, through the PRAMIN window (moves
 * NV_PBUS_BAR0_WINDOW, so only with allow_hw_writes=1; serialized by gsp_lock).
 */
static ssize_t vram_read(struct file *f, char __user *buf, size_t count, loff_t *ppos)
{
	struct nvdrv_dev *nv = f->private_data;
	loff_t pos = *ppos;
	size_t done = 0;

	if (pos < 0 || (pos & 3) || (count & 3))
		return -EINVAL;
	if (pos >= nv->fb.fb_size)
		return 0;
	count = min_t(size_t, count, min_t(u64, nv->fb.fb_size - pos, SZ_64K));
	mutex_lock(&nv->gsp_lock);
	while (done < count) {
		__le32 w = cpu_to_le32(nv_vram_rd32(nv, pos + done));

		if (copy_to_user(buf + done, &w, 4))
			break;
		done += 4;
	}
	mutex_unlock(&nv->gsp_lock);
	if (!done)
		return -EFAULT;
	*ppos = pos + done;
	return done;
}

static const struct file_operations vram_fops = {
	.owner = THIS_MODULE,
	.open = simple_open,
	.read = vram_read,
	.llseek = default_llseek,
};

static ssize_t bar1_test_write(struct file *f, const char __user *buf,
			       size_t count, loff_t *ppos)
{
	bool go;
	int ret = kstrtobool_from_user(buf, count, &go);

	if (ret)
		return ret;
	if (go)
		ret = nv_bar1_test(f->private_data);
	return ret ? ret : count;
}

static const struct file_operations bar1_test_fops = {
	.owner = THIS_MODULE,
	.open = simple_open,
	.write = bar1_test_write,
};

static ssize_t drm_register_write(struct file *f, const char __user *buf,
				  size_t count, loff_t *ppos)
{
	bool go;
	int ret = kstrtobool_from_user(buf, count, &go);

	if (ret)
		return ret;
	if (go)
		ret = nv_drm_register(f->private_data);
	return ret ? ret : count;
}

static const struct file_operations drm_register_fops = {
	.owner = THIS_MODULE,
	.open = simple_open,
	.write = drm_register_write,
};

static int dp_status_show(struct seq_file *m, void *unused)
{
	nv_disp_dp_status_show(m->private, m);
	return 0;
}
DEFINE_SHOW_ATTRIBUTE(dp_status);

static ssize_t disp_retrain_write(struct file *f, const char __user *buf,
				  size_t count, loff_t *ppos)
{
	bool go;
	int ret = kstrtobool_from_user(buf, count, &go);

	if (ret)
		return ret;
	if (go)
		ret = nv_disp_retrain(f->private_data);
	return ret ? ret : count;
}

static const struct file_operations disp_retrain_fops = {
	.owner = THIS_MODULE,
	.open = simple_open,
	.write = disp_retrain_write,
};

static ssize_t disp_up_write(struct file *f, const char __user *buf,
			     size_t count, loff_t *ppos)
{
	bool go;
	int ret = kstrtobool_from_user(buf, count, &go);

	if (ret)
		return ret;
	if (go)
		ret = nv_disp_bringup(f->private_data);
	return ret ? ret : count;
}

static const struct file_operations disp_up_fops = {
	.owner = THIS_MODULE,
	.open = simple_open,
	.write = disp_up_write,
};

static int drm_stats_show(struct seq_file *m, void *unused)
{
	nv_drm_stats_show(m->private, m);
	return 0;
}
DEFINE_SHOW_ATTRIBUTE(drm_stats);

static int i2c_show(struct seq_file *m, void *unused)
{
	nv_i2c_experiment_show(m);
	return 0;
}

static int i2c_open(struct inode *inode, struct file *f)
{
	return single_open(f, i2c_show, inode->i_private);
}

static ssize_t i2c_write(struct file *f, const char __user *buf, size_t count, loff_t *ppos)
{
	struct nvdrv_dev *nv = file_inode(f)->i_private;
	char cmd[64];
	int ret;

	if (count >= sizeof(cmd) || copy_from_user(cmd, buf, count))
		return -EINVAL;
	cmd[count] = 0;
	ret = nv_i2c_experiment(nv, strim(cmd));
	return ret ? ret : count;
}

static const struct file_operations i2c_fops = {
	.owner = THIS_MODULE,
	.open = i2c_open,
	.read = seq_read,
	.write = i2c_write,
	.llseek = seq_lseek,
	.release = single_release,
};

static int mem_show(struct seq_file *m, void *unused)
{
	nv_uapi_mem_show(m->private, m);
	return 0;
}
DEFINE_SHOW_ATTRIBUTE(mem);

static int intr_show(struct seq_file *m, void *unused)
{
	nv_intr_show(m->private, m);
	return 0;
}
DEFINE_SHOW_ATTRIBUTE(intr);

static int sensors_show(struct seq_file *m, void *unused)
{
	nv_sensors_show(m->private, m);
	return 0;
}
DEFINE_SHOW_ATTRIBUTE(sensors);

static int perf_show(struct seq_file *m, void *unused)
{
	nv_rm_perf_show(m->private, m);
	return 0;
}

static int perf_open(struct inode *inode, struct file *f)
{
	return single_open(f, perf_show, inode->i_private);
}

/* write "max" to boost to the highest pstate until cleared, "clear" to drop it */
static ssize_t perf_write(struct file *f, const char __user *buf, size_t count, loff_t *ppos)
{
	struct nvdrv_dev *nv = ((struct seq_file *)f->private_data)->private;
	char cmd[16] = {};
	u32 secs;
	int ret;

	if (copy_from_user(cmd, buf, min(count, sizeof(cmd) - 1)))
		return -EFAULT;
	if (!strncmp(cmd, "max", 3))
		ret = nv_rm_perf_boost(nv, NV_PERF_BOOST_FOREVER);
	else if (!strncmp(cmd, "clear", 5))
		ret = nv_rm_perf_boost(nv, 0);
	else if (!kstrtou32(strim(cmd), 0, &secs))
		ret = nv_rm_perf_boost(nv, secs);	/* "<n>": boost for n seconds */
	else
		ret = -EINVAL;
	return ret ? ret : count;
}

static const struct file_operations perf_fops = {
	.owner = THIS_MODULE,
	.open = perf_open,
	.read = seq_read,
	.write = perf_write,
	.llseek = seq_lseek,
	.release = single_release,
};

static ssize_t bar1_bench_write(struct file *f, const char __user *buf, size_t count, loff_t *ppos)
{
	int ret = nv_bar1_bench(f->private_data);

	return ret ? ret : count;
}

static const struct file_operations bar1_bench_fops = {
	.owner = THIS_MODULE,
	.open = simple_open,
	.write = bar1_bench_write,
};

static int chan_status_show(struct seq_file *m, void *unused)
{
	nv_fifo_show(m->private, m);
	return 0;
}
DEFINE_SHOW_ATTRIBUTE(chan_status);

static ssize_t chan_kick_write(struct file *f, const char __user *buf, size_t count, loff_t *ppos)
{
	u32 mode;
	int ret = kstrtou32_from_user(buf, count, 0, &mode);

	if (!ret)
		ret = nv_chan_kick(f->private_data, mode);
	return ret ? ret : count;
}

static const struct file_operations chan_kick_fops = {
	.owner = THIS_MODULE,
	.open = simple_open,
	.write = chan_kick_write,
};

static ssize_t ce_test_write(struct file *f, const char __user *buf, size_t count, loff_t *ppos)
{
	int ret = nv_ce_test(f->private_data);

	return ret ? ret : count;
}

static const struct file_operations ce_test_fops = {
	.owner = THIS_MODULE,
	.open = simple_open,
	.write = ce_test_write,
};

static ssize_t gr_test_write(struct file *f, const char __user *buf, size_t count, loff_t *ppos)
{
	int ret = nv_gr_test(f->private_data);

	return ret ? ret : count;
}

static ssize_t ce_fault_write(struct file *f, const char __user *buf, size_t count, loff_t *ppos)
{
	int ret = nv_ce_fault_test(f->private_data);

	return ret ?: count;
}

static int preempt_show(struct seq_file *m, void *v)
{
	nv_uapi_preempt_show(m->private, m);
	return 0;
}
DEFINE_SHOW_ATTRIBUTE(preempt);

/* debugfs gpu_lost: run the device-lost path (as if GSP had died); reload nvdrv afterwards */
static ssize_t gpu_lost_write(struct file *f, const char __user *buf, size_t count, loff_t *ppos)
{
	nv_uapi_device_lost(f->private_data, "simulated through debugfs");
	return count;
}

/* pm_test: "1" suspend + resume, "2" with an FLR in between (src/pm.c nv_pm_test) */
static ssize_t pm_test_write(struct file *f, const char __user *buf, size_t count, loff_t *ppos)
{
	u32 mode;
	int ret = kstrtou32_from_user(buf, count, 0, &mode);

	if (ret)
		return ret;
	ret = nv_pm_test(f->private_data, mode);
	return ret ? ret : count;
}

static const struct file_operations pm_test_fops = {
	.owner = THIS_MODULE,
	.open = simple_open,
	.write = pm_test_write,
};

static const struct file_operations gpu_lost_fops = {
	.owner = THIS_MODULE,
	.open = simple_open,
	.write = gpu_lost_write,
};

static const struct file_operations ce_fault_fops = {
	.owner = THIS_MODULE,
	.open = simple_open,
	.write = ce_fault_write,
};

static const struct file_operations gr_test_fops = {
	.owner = THIS_MODULE,
	.open = simple_open,
	.write = gr_test_write,
};

static void nv_debugfs_init(struct nvdrv_dev *nv)
{
	nv->dbg = debugfs_create_dir(pci_name(nv->pdev), nvdrv_dbg_root);
	debugfs_create_file("info", 0400, nv->dbg, nv, &info_fops);
	debugfs_create_file_size("mmio", 0400, nv->dbg, nv, &mmio_fops, nv->bar0_len);
	debugfs_create_file_size("prom", 0400, nv->dbg, nv, &prom_fops, NV_PROM_SIZE);
	debugfs_create_file("falcon", 0400, nv->dbg, nv, &falcon_fops);
	if (nv_hw_writes_allowed())
		debugfs_create_file("run_fwsec_frts", 0200, nv->dbg, nv, &run_fwsec_frts_fops);
	if (nv_hw_writes_allowed())
		debugfs_create_file("run_gsp_boot", 0200, nv->dbg, nv, &run_gsp_boot_fops);
	if (nv_hw_writes_allowed())
		debugfs_create_file("rm_selftest", 0200, nv->dbg, nv, &rm_selftest_fops);
	if (nv_hw_writes_allowed())
		debugfs_create_file("disp_probe", 0200, nv->dbg, nv, &disp_probe_fops);
	if (nv_hw_writes_allowed())
		debugfs_create_file("disp_link", 0200, nv->dbg, nv, &disp_link_fops);
	if (nv_hw_writes_allowed())
		debugfs_create_file("disp_modeset", 0200, nv->dbg, nv, &disp_modeset_fops);
	if (nv_hw_writes_allowed())
		debugfs_create_file("disp_flip", 0200, nv->dbg, nv, &disp_flip_fops);
	if (nv_hw_writes_allowed())
		debugfs_create_file("vram", 0400, nv->dbg, nv, &vram_fops);
	if (nv_hw_writes_allowed())
		debugfs_create_file("bar1_test", 0200, nv->dbg, nv, &bar1_test_fops);
	if (nv_hw_writes_allowed())
		debugfs_create_file("drm_register", 0200, nv->dbg, nv, &drm_register_fops);
	if (nv_hw_writes_allowed())
		debugfs_create_file("dp_status", 0400, nv->dbg, nv, &dp_status_fops);
	if (nv_hw_writes_allowed())
		debugfs_create_file("disp_retrain", 0200, nv->dbg, nv, &disp_retrain_fops);
	if (nv_hw_writes_allowed())
		debugfs_create_file("disp_up", 0200, nv->dbg, nv, &disp_up_fops);
	if (nv_hw_writes_allowed()) {
		debugfs_create_file("perf", 0600, nv->dbg, nv, &perf_fops);
		debugfs_create_file("sensors", 0400, nv->dbg, nv, &sensors_fops);
		debugfs_create_file("intr", 0400, nv->dbg, nv, &intr_fops);
		debugfs_create_file("mem", 0400, nv->dbg, nv, &mem_fops);
		debugfs_create_file("i2c", 0600, nv->dbg, nv, &i2c_fops);	/* experiment, read-only bus access */
		debugfs_create_file("bar1_bench", 0200, nv->dbg, nv, &bar1_bench_fops);
		debugfs_create_file("drm_stats", 0400, nv->dbg, nv, &drm_stats_fops);
		debugfs_create_file("ce_test", 0200, nv->dbg, nv, &ce_test_fops);
		debugfs_create_file("gr_test", 0200, nv->dbg, nv, &gr_test_fops);
		debugfs_create_file("ce_fault", 0200, nv->dbg, nv, &ce_fault_fops);
		debugfs_create_file("gpu_lost", 0200, nv->dbg, nv, &gpu_lost_fops);
		debugfs_create_file("pm_test", 0200, nv->dbg, nv, &pm_test_fops);
		debugfs_create_file("preempt", 0400, nv->dbg, nv, &preempt_fops);
		debugfs_create_file("chan_status", 0400, nv->dbg, nv, &chan_status_fops);
		debugfs_create_file("chan_kick", 0200, nv->dbg, nv, &chan_kick_fops);
	}

	if (nv->vbios.data) {
		nv->vbios_blob.data = nv->vbios.data;
		nv->vbios_blob.size = nv->vbios.size;
		debugfs_create_blob("vbios", 0400, nv->dbg, &nv->vbios_blob);
	}
	nv->gsp_info_blob.data = nv->gsp_info.raw;	/* zeros until GET_GSP_STATIC_INFO */
	nv->gsp_info_blob.size = sizeof(nv->gsp_info.raw);
	debugfs_create_blob("gsp_static_info.bin", 0400, nv->dbg, &nv->gsp_info_blob);
	if (nv->wpr_meta) {
		nv->wpr_meta_blob.data = nv->wpr_meta;
		nv->wpr_meta_blob.size = sizeof(*nv->wpr_meta);
		debugfs_create_blob("wpr_meta.bin", 0400, nv->dbg, &nv->wpr_meta_blob);
	}
	if (nv->libos.shm.cpu) {
		static const char * const names[] = { "libos_args.bin", "gsp_args.bin",
						       "msgq_pt.bin", "cmdq_hdr.bin" };
		struct nv_libos *l = &nv->libos;
		void *src[] = { l->args.cpu, l->rmargs.cpu, l->shm.cpu, l->shm.cpu + l->pt_size };
		int i;

		for (i = 0; i < 4; i++) {
			nv->libos_blobs[i].data = src[i];
			nv->libos_blobs[i].size = SZ_4K;
			debugfs_create_blob(names[i], 0400, nv->dbg, &nv->libos_blobs[i]);
		}
		nv->cmdq_blob.data = l->shm.cpu + l->pt_size;
		nv->cmdq_blob.size = SZ_16K;
		debugfs_create_blob("cmdq.bin", 0400, nv->dbg, &nv->cmdq_blob);
		for (i = 0; i < NV_LIBOS_NUM_LOGS; i++) {
			char name[16];

			snprintf(name, sizeof(name), "log%d.bin", i);
			nv->libos_blobs[4 + i].data = l->logs[i].cpu;
			nv->libos_blobs[4 + i].size = l->logs[i].size;
			debugfs_create_blob(name, 0400, nv->dbg, &nv->libos_blobs[4 + i]);
		}
	}
	if (nv->gsp_fw.booter_load.image) {
		nv->booter_load_blob.data = nv->gsp_fw.booter_load.image;
		nv->booter_load_blob.size = nv->gsp_fw.booter_load.image_size;
		debugfs_create_blob("booter_load.bin", 0400, nv->dbg, &nv->booter_load_blob);
	}
	if (nv->fwsec_frts) {
		nv->fwsec_frts_blob.data = nv->fwsec_frts;
		nv->fwsec_frts_blob.size = nv->fwsec.image_size;
		debugfs_create_blob("fwsec_frts.bin", 0400, nv->dbg, &nv->fwsec_frts_blob);
	}
}

/* ---- PCI driver ------------------------------------------------------- */

/*
 * Everything after identification is best effort: failures are logged and
 * the device stays bound so its state can be inspected through debugfs.
 */
static void nv_init_readonly(struct nvdrv_dev *nv)
{
	struct device *dev = &nv->pdev->dev;
	bool done;
	u32 gfw;
	int ret;

	/*
	 * gpuWaitForGfwBootComplete_TU102: after power-on or FLR the VBIOS's own
	 * boot firmware (devinit, FWSEC) runs for ~300 ms; RM allows 50 ms + 2 s.
	 */
	if (read_poll_timeout(nv_gfw_boot_completed, done, done, 1000, 2050000, false, nv, &gfw)) {
		dev_err(dev, "GFW boot did not complete (progress 0x%02x)\n", gfw);
		return;
	}
	nv_vram_probe(nv);

	ret = nv_vbios_load(nv);
	if (ret) {
		dev_err(dev, "vbios load failed: %d\n", ret);
		return;
	}
	ret = nv_fwsec_extract(nv);
	if (ret)
		dev_err(dev, "fwsec extraction failed: %d\n", ret);

	ret = nv_fb_layout_calc(nv);
	if (ret) {
		dev_err(dev, "fb layout failed: %d\n", ret);
		return;
	}
	if (nv->fwsec.image) {
		ret = nv_fwsec_prepare_frts(nv, nv->fb.frts_offset, &nv->fwsec_frts);
		if (ret)
			dev_err(dev, "fwsec FRTS preparation failed: %d\n", ret);
		ret = nv_fwsec_prepare_sb(nv, &nv->fwsec_sb);
		if (ret)	/* not fatal: unload then falls back to an FLR */
			dev_err(dev, "fwsec SB preparation failed: %d\n", ret);
	}

	ret = nv_fw_load(nv);
	if (ret) {
		dev_err(dev, "GSP firmware load failed: %d\n", ret);
		return;
	}
	ret = nv_wpr_meta_build(nv);
	if (ret) {
		dev_err(dev, "WPR meta build failed: %d\n", ret);
		return;
	}
	ret = nv_libos_init(nv);
	if (ret) {
		dev_err(dev, "LibOS args / queue setup failed: %d\n", ret);
		return;
	}
	/* Host memory only: the async init RPCs GSP-RM reads at startup (inspectable early). */
	ret = nv_rpc_init(nv);
	if (!ret)
		ret = nv_rpc_queue_preboot(nv);
	if (ret)
		dev_err(dev, "pre-boot RPC queueing failed: %d\n", ret);
}

static int rebar = -1;
module_param(rebar, int, 0444);
MODULE_PARM_DESC(rebar, "Resize BAR1 at probe: -1 = largest supported (default), 0 = leave as is, N = N MiB");

/*
 * Resizable BAR (as amdgpu_device_resize_fb_bar): grow BAR1 so the CPU can reach all of
 * VRAM, when the firmware left it small. At probe, before GSP-RM or anything else uses
 * BAR1; the firmware framebuffer (simpledrm/efifb) lives in BAR1 and is removed first,
 * its claim on the old range would block the release. BAR0 sits in a different (32-bit)
 * bridge window and is excluded. Failures leave the BAR as it was (the PCI core restores).
 */
static void nv_rebar(struct pci_dev *pdev)
{
	struct device *dev = &pdev->dev;
	u64 cur = pci_resource_len(pdev, 1), want;
	int max = pci_rebar_get_max_size(pdev, 1), size, ret;
	u16 cmd;

	if (!rebar || max < 0 || !nv_hw_writes_allowed())
		return;
	size = rebar > 0 ? pci_rebar_bytes_to_size((u64)rebar << 20) : max;
	if (size > max || !pci_rebar_size_supported(pdev, 1, size)) {
		dev_warn(dev, "rebar: %d MiB not supported (largest %llu MiB)\n", rebar,
			 (u64)pci_rebar_size_to_bytes(max) >> 20);
		return;
	}
	want = pci_rebar_size_to_bytes(size);
	if (want == cur)
		return;
	dev_info(dev, "rebar: BAR1 %llu MiB -> %llu MiB\n", cur >> 20, want >> 20);
	aperture_remove_conflicting_pci_devices(pdev, DRV_NAME);	/* boot framebuffer is in BAR1 */
	pci_read_config_word(pdev, PCI_COMMAND, &cmd);
	pci_write_config_word(pdev, PCI_COMMAND, cmd & ~PCI_COMMAND_MEMORY);
	ret = pci_resize_resource(pdev, 1, size, BIT(0));
	pci_assign_unassigned_bus_resources(pdev->bus);
	pci_write_config_word(pdev, PCI_COMMAND, cmd);
	if (ret)
		dev_warn(dev, "rebar: resize failed (%d%s), BAR1 stays %llu MiB\n", ret,
			 ret == -ENOSPC ? ": not enough PCI address space" : "", pci_resource_len(pdev, 1) >> 20);
	else
		dev_info(dev, "rebar: BAR1 now %llu MiB at 0x%llx\n", pci_resource_len(pdev, 1) >> 20,
			 (u64)pci_resource_start(pdev, 1));
}

static void nv_release(struct kref *ref)
{
	struct nvdrv_dev *nv = container_of(ref, struct nvdrv_dev, ref);

	mutex_destroy(&nv->gsp_lock);
	mutex_destroy(&nv->mem.lock);
	mutex_destroy(&nv->evo.win_lock);
	pci_dev_put(nv->pdev);
	kfree(nv);
}

void nv_put(struct nvdrv_dev *nv)
{
	kref_put(&nv->ref, nv_release);
}

static void nv_put_action(void *nv)
{
	nv_put(nv);
}

static int nvdrv_probe(struct pci_dev *pdev, const struct pci_device_id *id)
{
	struct nvdrv_dev *nv;
	int ret;

	if (pdev->vendor != PCI_VENDOR_ID_NVIDIA)
		return -ENODEV;

	/*
	 * Not devm: DRM files (and the GEM objects and channels they own) can outlive the
	 * binding, and their teardown still takes nvdrv's locks. One reference for the
	 * binding (dropped by devres at unbind), one taken by the DRM device.
	 */
	nv = kzalloc(sizeof(*nv), GFP_KERNEL);
	if (!nv)
		return -ENOMEM;
	kref_init(&nv->ref);
	nv->pdev = pci_dev_get(pdev);	/* teardown after unbind still frees DMA memory against it */
	mutex_init(&nv->gsp_lock);
	mutex_init(&nv->mem.lock);
	mutex_init(&nv->evo.win_lock);
	mutex_init(&nv->evo.curs_lock);
	INIT_WORK(&nv->disp.hpd_work, nv_disp_hpd_work);
	nv_uapi_init(nv);
	ret = devm_add_action_or_reset(&pdev->dev, nv_put_action, nv);
	if (ret)
		return ret;
	pci_set_drvdata(pdev, nv);

	ret = pcim_enable_device(pdev);
	if (ret)
		return ret;
	nv_rebar(pdev);	/* before anything maps or uses BAR1 */

	/* BAR0 only. BAR1 holds the firmware framebuffer used by simpledrm. */
	nv->bar0 = pcim_iomap_region(pdev, 0, DRV_NAME);
	if (IS_ERR(nv->bar0))
		return PTR_ERR(nv->bar0);
	nv->bar0_len = pci_resource_len(pdev, 0);

	/* GA10x falcons/GSP address 47 bits (DMATRFBASE1 holds bits 40..48). */
	ret = dma_set_mask_and_coherent(&pdev->dev, DMA_BIT_MASK(47));
	if (ret)
		return ret;

	ret = nv_identify(nv);
	if (ret)
		return ret;
	dev_info(&pdev->dev, "%s (chipset 0x%03x rev %02x)\n", nv->name, nv->chipset, nv->rev);

	nv_init_readonly(nv);
	nv_hwmon_init(nv);	/* temperature: BAR0 only */
	nv_debugfs_init(nv);
	return 0;
}

static void nvdrv_remove(struct pci_dev *pdev)
{
	struct nvdrv_dev *nv = pci_get_drvdata(pdev);

	debugfs_remove_recursive(nv->dbg);
	cancel_work_sync(&nv->disp.hpd_work);
	nv_drm_unregister(nv);	/* before nv_gsp_fini: no BAR1 access after the unload/FLR */
	nv_gsp_fini(nv);	/* GSP-RM unload (or FLR) first: it DMAs into what follows */
	nv_rpc_fini(nv);
	nv_libos_fini(nv);
	nv_wpr_meta_fini(nv);
	nv_fw_fini(nv);
	nv_vbios_fini(nv);
	dev_info(&pdev->dev, "unbound\n");
	/* pcim/devm release BAR0 mapping, region and enable count. */
}

/*
 * Intentionally empty: no MODULE_DEVICE_TABLE, no automatic matching.
 * Bind explicitly with driver_override (scripts/bind.sh).
 */
static const struct pci_device_id nvdrv_ids[] = {
	{ }
};

/*
 * System sleep (S3): GSP-RM keeps its state across suspend (src/pm.c), the display goes
 * off and comes back through the DRM helpers. Hibernation is not supported yet (the SR
 * data would have to survive in the image): refused while GSP-RM runs.
 */
static int nvdrv_pm_suspend(struct device *dev)
{
	struct nvdrv_dev *nv = dev_get_drvdata(dev);
	int ret;

	if (!nv->gsp_running)
		return nv->gsp_touched ? -EBUSY : 0;
	ret = nv_drm_suspend(nv);
	if (ret)
		return ret;
	ret = nv_pm_suspend(nv);
	if (ret)
		nv_drm_resume(nv);
	return ret;
}

static int nvdrv_pm_resume(struct device *dev)
{
	struct nvdrv_dev *nv = dev_get_drvdata(dev);
	int ret;

	if (!nv->pm.suspended)
		return 0;
	ret = nv_pm_resume(nv);
	nv_drm_resume(nv);
	return ret;
}

static int nvdrv_pm_freeze(struct device *dev)
{
	struct nvdrv_dev *nv = dev_get_drvdata(dev);

	if (!nv->gsp_touched)
		return 0;
	dev_warn(dev, "hibernation is not supported while GSP-RM runs\n");
	return -EBUSY;
}

static int nvdrv_pm_nop(struct device *dev)
{
	return 0;
}

static const struct dev_pm_ops nvdrv_pm_ops = {
	.suspend = nvdrv_pm_suspend,
	.resume = nvdrv_pm_resume,
	.freeze = nvdrv_pm_freeze,
	.thaw = nvdrv_pm_nop,
	.poweroff = nvdrv_pm_freeze,
	.restore = nvdrv_pm_nop,
};

static struct pci_driver nvdrv_driver = {
	.name = DRV_NAME,
	.id_table = nvdrv_ids,
	.probe = nvdrv_probe,
	.remove = nvdrv_remove,
	.driver.pm = pm_sleep_ptr(&nvdrv_pm_ops),
};

static int __init nvdrv_init(void)
{
	int ret;

	nvdrv_dbg_root = debugfs_create_dir(DRV_NAME, NULL);
	ret = pci_register_driver(&nvdrv_driver);
	if (ret)
		debugfs_remove_recursive(nvdrv_dbg_root);
	return ret;
}

static void __exit nvdrv_exit(void)
{
	pci_unregister_driver(&nvdrv_driver);
	debugfs_remove_recursive(nvdrv_dbg_root);
}

module_init(nvdrv_init);
module_exit(nvdrv_exit);

MODULE_DESCRIPTION("From-scratch NVIDIA GPU driver (experimental)");
MODULE_LICENSE("GPL");
