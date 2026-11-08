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
 * - fan speed: the VBIOS DCB GPIO table (open-gpu-doc DCB 4.x; parsed as nouveau's
 *   nvkm/subdev/bios/gpio.c does) names the FAN_SENSE (0x3d) line; its input level is bit 14
 *   of 0x021200 + 4 * line (nouveau nvkm/subdev/gpio/ga102.c). RPM is timed over four level
 *   changes, one rotation (nouveau nvkm/subdev/therm/fan.c). Passive reads only; GSP-RM
 *   keeps driving the fan. Checked against load: ~1290 RPM up to ~50 C, rising to
 *   1800 RPM at 58 C.
 * Board power and the live graphics clock have no published interface (the RM controls
 * are stubs in the open SDK; GSP-RM's NV2080_CTRL_CMD_I2C_READ_REG returns zeros without
 * touching the bus, and this board's DCB lists no INA power sensor) and are not reported.
 */
#include <linux/delay.h>
#include <linux/hwmon.h>
#include <linux/ktime.h>
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

static int fan_line = -1;	/* FAN_SENSE GPIO line from the VBIOS, -1 = none */
static int fan_rpm(struct nvdrv_dev *nv, long *rpm);

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
	{
		long rpm;

		if (!fan_rpm(nv, &rpm))
			seq_printf(m, "fan: %ld RPM (tach on GPIO line %d)\n", rpm, fan_line);
	}
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

/* ---- fan tachometer ---------------------------------------------------------- */

#define DCB_GPIO_FAN_SENSE	0x3d
#define NV_GPIO_LINE(n)		(0x00021200 + (n) * 4)
#define NV_GPIO_LINE_SENSE	BIT(14)

static u32 vb_rd(const struct nvdrv_dev *nv, u32 off, u32 bytes)
{
	u32 v = 0, i;

	if (!nv->vbios.data || off + bytes > nv->vbios.size)
		return 0;
	for (i = 0; i < bytes; i++)
		v |= (u32)nv->vbios.data[off + i] << (8 * i);
	return v;
}

/* DCB 4.x GPIO Assignment Table: the line with function FAN_SENSE. */
static int fan_sense_line(struct nvdrv_dev *nv)
{
	u32 dcb = vb_rd(nv, 0x36, 2), gpio, hdr, cnt, len, i;

	if (!dcb || vb_rd(nv, dcb, 1) < 0x30 || vb_rd(nv, dcb + 1, 1) < 0x0c)
		return -ENODEV;
	gpio = vb_rd(nv, dcb + 0x0a, 2);
	if (!gpio || vb_rd(nv, gpio, 1) < 0x40 || vb_rd(nv, gpio, 1) > 0x41)
		return -ENODEV;
	hdr = vb_rd(nv, gpio + 1, 1);
	cnt = vb_rd(nv, gpio + 2, 1);
	len = vb_rd(nv, gpio + 3, 1);
	for (i = 0; i < cnt; i++) {
		u32 info = vb_rd(nv, gpio + hdr + i * len, 4);

		if (((info >> 8) & 0xff) == DCB_GPIO_FAN_SENSE)
			return info & 0x3f;
	}
	return -ENOENT;
}

/* Fan speed in RPM, 0 if stopped (no rotation within 250 ms). Sleeps ~4 tach periods. */
static int fan_rpm(struct nvdrv_dev *nv, long *rpm)
{
	u32 prev, cur, changes = 0;
	u64 start = 0, end, t0 = ktime_get_ns();

	if (fan_line < 0)
		return -ENODEV;
	prev = nv_rd32(nv, NV_GPIO_LINE(fan_line)) & NV_GPIO_LINE_SENSE;
	do {
		usleep_range(500, 1000);	/* up to 7500 RPM */
		cur = nv_rd32(nv, NV_GPIO_LINE(fan_line)) & NV_GPIO_LINE_SENSE;
		if (cur != prev) {
			if (!start)
				start = ktime_get_ns();
			changes++;
			prev = cur;
		}
	} while (changes < 5 && ktime_get_ns() - t0 < 250 * NSEC_PER_MSEC);
	end = ktime_get_ns();
	/* five changes = four intervals = one rotation (two tach pulses) */
	*rpm = changes == 5 && end > start ? div64_u64(60ULL * NSEC_PER_SEC, end - start) : 0;
	return 0;
}

/* ---- hwmon: temp1, fan1 ------------------------------------------------------- */

static umode_t hwmon_visible(const void *data, enum hwmon_sensor_types type, u32 attr, int channel)
{
	if (type == hwmon_temp && (attr == hwmon_temp_input || attr == hwmon_temp_label))
		return 0444;
	if (type == hwmon_fan && (attr == hwmon_fan_input || attr == hwmon_fan_label) && fan_line >= 0)
		return 0444;
	return 0;
}

static int hwmon_read(struct device *dev, enum hwmon_sensor_types type, u32 attr, int channel, long *val)
{
	struct nvdrv_dev *nv = dev_get_drvdata(dev);

	if (type == hwmon_temp && attr == hwmon_temp_input)
		return nv_sensor_temp(nv, val);
	if (type == hwmon_fan && attr == hwmon_fan_input)
		return fan_rpm(nv, val);
	return -EOPNOTSUPP;
}

static int hwmon_read_string(struct device *dev, enum hwmon_sensor_types type, u32 attr, int channel,
			     const char **str)
{
	*str = type == hwmon_fan ? "GPU fan" : "GPU";
	return 0;
}

static const struct hwmon_ops nv_hwmon_ops = {
	.is_visible = hwmon_visible,
	.read = hwmon_read,
	.read_string = hwmon_read_string,
};

static const struct hwmon_channel_info *const nv_hwmon_info[] = {
	HWMON_CHANNEL_INFO(temp, HWMON_T_INPUT | HWMON_T_LABEL),
	HWMON_CHANNEL_INFO(fan, HWMON_F_INPUT | HWMON_F_LABEL),
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

	fan_line = fan_sense_line(nv);
	if (fan_line >= 0)
		dev_info(&nv->pdev->dev, "hwmon: fan tachometer on GPIO line %d (VBIOS DCB)\n", fan_line);
	hw = devm_hwmon_device_register_with_info(&nv->pdev->dev, "nvdrv", nv, &nv_hwmon_chip, NULL);
	if (IS_ERR(hw))
		dev_warn(&nv->pdev->dev, "hwmon: registration failed: %ld\n", PTR_ERR(hw));
}

/* ---- experiment: board I2C devices through RM (read-only) ----------------- */

#define NV40_I2C				0x402c
#define NVDRV_H_I2C				0xd0004020
#define NV402C_CTRL_CMD_I2C_GET_PORT_INFO	0x402c0101
#define NV402C_CTRL_CMD_I2C_TABLE_GET_DEV_INFO	0x402c0104
#define NV2080_CTRL_CMD_I2C_READ_REG		0x20800603

struct nv_i2c_dev_table {		/* NV402C_CTRL_I2C_TABLE_GET_DEV_INFO_PARAMS */
	u8 count, pad;
	struct { u8 type, pad; __le16 addr; u8 port, idx; } dev[32];
};
static_assert(sizeof(struct nv_i2c_dev_table) == 194);

struct nv_i2c_rw_reg {			/* NV2080_CTRL_I2C_RW_REG_PARAMS */
	__le32 version, port, flags, addr;
	u8 reg, bufsize, buffer[255];
	u8 pad[3];
};
static_assert(sizeof(struct nv_i2c_rw_reg) == 276);

/* DCB 4.x I2C Devices Table types (open-gpu-doc DCB-4.x-Specification) */
static const char *dcb_i2c_type(u8 t)
{
	switch (t) {
	case 0x01: return "ADM1032 thermal";	case 0x02: return "MAX6649 thermal";
	case 0x03: return "LM99 thermal";	case 0x06: return "MAX1617 thermal";
	case 0x07: return "LM64 thermal";	case 0x0a: return "ADT7473 thermal";
	case 0x0b: return "LM89 thermal";	case 0x0c: return "TMP411 thermal";
	case 0x0d: return "ADT7461 thermal";	case 0x30: return "ADS1112 ADC";
	case 0x40: return "VT1103 power ctrl";	case 0x41: return "PX3540 power ctrl";
	case 0x42: return "VT1165 power ctrl";	case 0x43: return "CHL82xx power ctrl";
	case 0x44: return "NCP4208 power ctrl";	case 0x48: return "CHL8112/8225 power ctrl";
	case 0x49: return "CHL8266/8316 power ctrl"; case 0x4a: return "DS4424N";
	case 0x4b: return "NCT3933U";		case 0x4c: return "INA219 power sensor";
	case 0x4d: return "INA209 power sensor"; case 0x4e: return "INA3221 power sensor";
	case 0x50: return "CY2XP304 clock gen";	case 0x60: return "PCA9555 GPIO";
	case 0x70: return "ADT7473 fan control"; case 0x80: return "SI1930uC HDMI";
	case 0x82: return "PCA9536 GPIO";	case 0xb0: case 0xb1: return "GPU I2CS";
	case 0xd0: return "ANX9805 encoder";	case 0xff: return "skip";
	default: return "?";
	}
}

static char i2c_log[2048];
static size_t i2c_log_len;

static void i2c_logf(const char *fmt, ...)
{
	va_list ap;

	va_start(ap, fmt);
	i2c_log_len += vscnprintf(i2c_log + i2c_log_len, sizeof(i2c_log) - i2c_log_len, fmt, ap);
	va_end(ap);
}

/*
 * debugfs i2c: "table" lists the board's I2C devices (from the VBIOS DCB through RM);
 * "read <port> <addr7> <reg> <len>" reads <len> bytes of register <reg>. No writes, on
 * purpose: board I2C devices include voltage regulators. RM serialises the buses with
 * its own users. Reading the file shows the last result.
 */
int nv_i2c_experiment(struct nvdrv_dev *nv, const char *cmd)
{
	u32 port, addr, reg, len, i;
	int ret;

	i2c_log_len = 0;
	if (!nv->rm.h_subdevice)
		return -ENODEV;
	mutex_lock(&nv->gsp_lock);
	if (sysfs_streq(cmd, "table")) {
		struct nv_i2c_dev_table *t = kzalloc(sizeof(*t), GFP_KERNEL);
		u8 pi[16] = {};

		ret = t ? 0 : -ENOMEM;
		if (!ret) {
			ret = nv_rm_alloc(nv, nv->rm.h_client, nv->rm.h_subdevice, NVDRV_H_I2C, NV40_I2C, NULL, 0);
			if (ret == -EEXIST)
				ret = 0;
			i2c_logf("NV40_I2C object: %d\n", ret);
		}
		if (!ret) {
			int r = nv_rm_control(nv, nv->rm.h_client, NVDRV_H_I2C, NV402C_CTRL_CMD_I2C_GET_PORT_INFO,
					      pi, sizeof(pi));

			i2c_logf("port info (%d):", r);
			for (i = 0; i < 16; i++)
				i2c_logf(" %02x", pi[i]);
			i2c_logf("\n");
			ret = nv_rm_control(nv, nv->rm.h_client, NVDRV_H_I2C, NV402C_CTRL_CMD_I2C_TABLE_GET_DEV_INFO,
					    t, sizeof(*t));
			i2c_logf("device table (%d): %u devices\n", ret, t->count);
			for (i = 0; !ret && i < t->count && i < 32; i++)
				i2c_logf("  %2u: type 0x%02x %-24s addr 0x%02x (8-bit 0x%02x) port %u idx %u\n", i,
					 t->dev[i].type, dcb_i2c_type(t->dev[i].type), le16_to_cpu(t->dev[i].addr) >> 1,
					 le16_to_cpu(t->dev[i].addr), t->dev[i].port, t->dev[i].idx);
		}
		kfree(t);
	} else if (sscanf(cmd, "read %u %x %x %u", &port, &addr, &reg, &len) == 4 && len && len <= 16 &&
		   addr < 0x80 && reg < 0x100) {
		struct nv_i2c_rw_reg *p = kzalloc(sizeof(*p), GFP_KERNEL);

		ret = p ? 0 : -ENOMEM;
		if (!ret) {
			p->port = cpu_to_le32(port);
			p->addr = cpu_to_le32(addr << 1);	/* RM takes the 8-bit form */
			p->reg = reg;
			p->bufsize = len;
			ret = nv_rm_control(nv, nv->rm.h_client, nv->rm.h_subdevice, NV2080_CTRL_CMD_I2C_READ_REG,
					    p, sizeof(*p));
			i2c_logf("read port %u addr 0x%02x reg 0x%02x len %u: %d:", port, addr, reg, len, ret);
			for (i = 0; !ret && i < len; i++)
				i2c_logf(" %02x", p->buffer[i]);
			i2c_logf("\n");
		}
		kfree(p);
	} else {
		ret = -EINVAL;
	}
	mutex_unlock(&nv->gsp_lock);
	return ret;
}

void nv_i2c_experiment_show(struct seq_file *m)
{
	seq_write(m, i2c_log, i2c_log_len);
}
