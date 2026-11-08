// SPDX-License-Identifier: GPL-2.0
/*
 * Sensors: GPU temperature (hwmon + debugfs), utilisation and clocks (debugfs).
 *
 * Sources, all public:
 * - temperature: the internal sensor register 0x020460 that nouveau reads
 *   (nvkm/subdev/therm/gp100.c, MIT): degrees C in bits 16:3 as a fixed-point value
 *   with 8 fractional bits. On GA104 bits 31:30 are set and nouveau's Pascal "valid"
 *   bit 29 is clear; the value was checked against load (35 C idle -> 54 C under
 *   vkmark -> falling afterwards), so nvdrv takes it when either marker is set.
 * - utilisation: NV2080_CTRL_CMD_PERF_GET_GPUMON_PERFMON_UTIL_SAMPLES_V2 (open SDK,
 *   ctrl2080perf.h): 72 samples per engine, busy time in units of 0.01 %.
 * - clocks: NV2080_CTRL_CMD_PERF_GET_LEVEL_INFO_V2 (open SDK): per clock domain current,
 *   default, min and max frequency (kHz) of a perf level. The clock domain IDs are not
 *   in the open SDK, so domains are listed by their ID. GA104 reports 5 levels and two
 *   domains: 0x10 rises to 7001 MHz at the top level (the GDDR6 memory clock), 0x1 stays
 *   near 420 MHz; the graphics (GPC) clock is not among them.
 * Fan speed and board power have no published interface (their RM controls are not
 * in the open SDK) and are not reported.
 */
#include <linux/hwmon.h>
#include <linux/seq_file.h>
#include <linux/slab.h>

#include "nvdrv.h"

#define NV_THERM_SENSOR			0x00020460
#define NV_THERM_SENSOR_TEMP		0x0001fff8	/* C, 8 fractional bits */
#define NV_THERM_SENSOR_MARKERS		(BIT(31) | BIT(30) | BIT(29))

#define NV2080_CTRL_CMD_PERF_GET_GPUMON_PERFMON_UTIL_SAMPLES_V2	0x20802096
#define NV2080_CTRL_CMD_PERF_GET_LEVEL_INFO_V2			0x2080200b
#define GPUMON_SAMPLE_TYPE_PERFMON_UTIL		2
#define GPUMON_SAMPLE_COUNT			72

struct nv_util_engine {			/* NV2080_CTRL_PERF_GPUMON_ENGINE_UTIL_SAMPLE */
	__le32 util, vgpu_scale, proc_id, sub_proc_id;
	u8 rest[112];
};
static_assert(sizeof(struct nv_util_engine) == 128);

struct nv_util_sample {			/* NV2080_CTRL_PERF_GPUMON_PERFMON_UTIL_SAMPLE */
	__le64 timestamp;
	struct nv_util_engine fb, gr, nvenc, nvdec, nvjpg, nvofa;
};
static_assert(sizeof(struct nv_util_sample) == 776);

struct nv_util_samples {		/* ..._UTIL_SAMPLES_V2_PARAMS */
	u8 type, pad[3];
	__le32 buf_size, count, tracker;
	struct nv_util_sample samples[GPUMON_SAMPLE_COUNT];
};
static_assert(offsetof(struct nv_util_samples, samples) == 16);
static_assert(sizeof(struct nv_util_samples) == 16 + 55872);

struct nv_clk_info {			/* NV2080_CTRL_PERF_GET_CLK_INFO */
	__le32 flags, domain, cur, def, min, max;
};

struct nv_level_info {			/* NV2080_CTRL_PERF_GET_LEVEL_INFO_V2_PARAMS */
	__le32 level, flags;
	struct nv_clk_info clk[32];
	__le32 clk_count;
};
static_assert(sizeof(struct nv_level_info) == 780);

/* GPU temperature in millidegrees C; works whenever BAR0 is mapped (no GSP needed). */
int nv_sensor_temp(struct nvdrv_dev *nv, long *mc)
{
	u32 v = nv_rd32(nv, NV_THERM_SENSOR);

	if (!(v & NV_THERM_SENSOR_MARKERS) || v == 0xffffffff)
		return -ENODATA;
	*mc = (long)(v & NV_THERM_SENSOR_TEMP) * 1000 / 256;
	return 0;
}

/* Latest utilisation sample (0.01 % units). Caller holds gsp_lock. */
static int util_latest(struct nvdrv_dev *nv, struct nv_util_sample *out)
{
	struct nv_util_samples *p;
	u64 best = 0;
	int i, ret;

	p = kvzalloc(sizeof(*p), GFP_KERNEL);
	if (!p)
		return -ENOMEM;
	p->type = GPUMON_SAMPLE_TYPE_PERFMON_UTIL;
	p->buf_size = cpu_to_le32(sizeof(p->samples));
	ret = nv_rm_control(nv, nv->rm.h_client, nv->rm.h_subdevice,
			    NV2080_CTRL_CMD_PERF_GET_GPUMON_PERFMON_UTIL_SAMPLES_V2, p, sizeof(*p));
	if (!ret) {
		ret = -ENODATA;
		for (i = 0; i < GPUMON_SAMPLE_COUNT; i++) {
			u64 ts = le64_to_cpu(p->samples[i].timestamp);

			if (ts > best) {
				best = ts;
				*out = p->samples[i];
				ret = 0;
			}
		}
	}
	kvfree(p);
	return ret;
}

/*
 * Clocks of perf level `level`. RM rejects the whole query if any listed domain does not
 * exist on the chip, so the valid domain bits are found once (one query per bit) and
 * cached. Caller holds gsp_lock.
 */
static u32 clk_domains;		/* valid clock-domain bits (0 = not probed yet) */

static int level_query(struct nvdrv_dev *nv, u32 level, u32 mask, struct nv_level_info *li)
{
	u32 n = 0, i;

	memset(li, 0, sizeof(*li));
	li->level = cpu_to_le32(level);
	for (i = 0; i < 32; i++)
		if (mask & BIT(i))
			li->clk[n++].domain = cpu_to_le32(BIT(i));
	li->clk_count = cpu_to_le32(n);
	return nv_rm_control(nv, nv->rm.h_client, nv->rm.h_subdevice, NV2080_CTRL_CMD_PERF_GET_LEVEL_INFO_V2,
			     li, sizeof(*li));
}

static int level_clocks(struct nvdrv_dev *nv, u32 level, struct nv_level_info *li)
{
	u32 i;

	if (!clk_domains) {
		for (i = 0; i < 32; i++)
			if (!level_query(nv, 0, BIT(i), li))
				clk_domains |= BIT(i);
		if (!clk_domains)
			return -ENODEV;
	}
	return level_query(nv, level, clk_domains, li);
}

/* debugfs sensors */
void nv_sensors_show(struct nvdrv_dev *nv, struct seq_file *m)
{
	struct nv_util_sample s;
	struct nv_level_info *li;
	long mc;
	int ret, i;

	if (!nv_sensor_temp(nv, &mc))
		seq_printf(m, "temperature: %ld.%02ld C (raw 0x%08x)\n", mc / 1000, mc % 1000 / 10,
			   nv_rd32(nv, NV_THERM_SENSOR));
	else
		seq_printf(m, "temperature: unavailable (raw 0x%08x)\n", nv_rd32(nv, NV_THERM_SENSOR));
	if (!nv->rm.h_subdevice) {
		seq_puts(m, "(GSP-RM not running: no utilisation or clocks)\n");
		return;
	}
	nv_rm_perf_show(nv, m);

	mutex_lock(&nv->gsp_lock);
	ret = util_latest(nv, &s);
	if (!ret)
		seq_printf(m, "utilisation: graphics %u.%02u %%, memory %u.%02u %%, nvenc %u.%02u %%, nvdec %u.%02u %%\n",
			   le32_to_cpu(s.gr.util) / 100, le32_to_cpu(s.gr.util) % 100,
			   le32_to_cpu(s.fb.util) / 100, le32_to_cpu(s.fb.util) % 100,
			   le32_to_cpu(s.nvenc.util) / 100, le32_to_cpu(s.nvenc.util) % 100,
			   le32_to_cpu(s.nvdec.util) / 100, le32_to_cpu(s.nvdec.util) % 100);
	else
		seq_printf(m, "utilisation: unavailable (%d)\n", ret);

	li = kzalloc(sizeof(*li), GFP_KERNEL);
	for (i = 0; li && i < 16; i++) {	/* perf levels 0.. until RM says no */
		u32 level = i, j;

		ret = level_clocks(nv, level, li);
		if (ret) {
			if (!i)
				seq_printf(m, "perf level clocks: unavailable (%d)\n", ret);
			break;
		}
		seq_printf(m, "level %u clocks (kHz: current / default / min / max):\n", level);
		for (j = 0; j < le32_to_cpu(li->clk_count) && j < 32; j++) {
			const struct nv_clk_info *c = &li->clk[j];

			seq_printf(m, "  domain 0x%08x: %7u / %7u / %7u / %7u\n", le32_to_cpu(c->domain), le32_to_cpu(c->cur),
				   le32_to_cpu(c->def), le32_to_cpu(c->min), le32_to_cpu(c->max));
		}
	}
	kfree(li);
	mutex_unlock(&nv->gsp_lock);
}

/* ---- hwmon: temp1 ---------------------------------------------------------- */

static umode_t hwmon_visible(const void *data, enum hwmon_sensor_types type, u32 attr, int channel)
{
	return type == hwmon_temp && (attr == hwmon_temp_input || attr == hwmon_temp_label) ? 0444 : 0;
}

static int hwmon_read(struct device *dev, enum hwmon_sensor_types type, u32 attr, int channel, long *val)
{
	struct nvdrv_dev *nv = dev_get_drvdata(dev);

	if (type != hwmon_temp || attr != hwmon_temp_input)
		return -EOPNOTSUPP;
	return nv_sensor_temp(nv, val);
}

static int hwmon_read_string(struct device *dev, enum hwmon_sensor_types type, u32 attr, int channel,
			     const char **str)
{
	*str = "GPU";
	return 0;
}

static const struct hwmon_ops nv_hwmon_ops = {
	.is_visible = hwmon_visible,
	.read = hwmon_read,
	.read_string = hwmon_read_string,
};

static const struct hwmon_channel_info *const nv_hwmon_info[] = {
	HWMON_CHANNEL_INFO(temp, HWMON_T_INPUT | HWMON_T_LABEL),
	NULL
};

static const struct hwmon_chip_info nv_hwmon_chip = {
	.ops = &nv_hwmon_ops,
	.info = nv_hwmon_info,
};

/* At probe: the sensor needs only BAR0. Lives until unbind (devm). */
void nv_hwmon_init(struct nvdrv_dev *nv)
{
	struct device *hw;

	hw = devm_hwmon_device_register_with_info(&nv->pdev->dev, "nvdrv", nv, &nv_hwmon_chip, NULL);
	if (IS_ERR(hw))
		dev_warn(&nv->pdev->dev, "hwmon: registration failed: %ld\n", PTR_ERR(hw));
}
