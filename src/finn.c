// SPDX-License-Identifier: GPL-2.0 AND MIT
/*
 * FINN serialisation for the one RM control nvdrv needs it for:
 * NV402C_CTRL_CMD_I2C_TRANSACTION (board I2C through GSP-RM).
 *
 * RM controls whose parameters hold pointers travel to GSP-RM FINN-encoded, marked
 * RMAPI_RPC_FLAGS_SERIALIZED: a FINN_RM_API header { version 0, payloadSize, interface =
 * cmd >> 8, message = cmd & 0xff } followed by fields packed LSB-first into 64-bit words,
 * each field preceded by a 1-bit "present" marker, a union variant by one more marker.
 * The bit pump and the field order follow NVIDIA's generated code in open-gpu-kernel-modules
 * (src/nvidia/interface/rmapi/src/g_finn_rm_api.c: finn_write_buffer, finn_read_buffer,
 * finnSerializeMessage_NV402C_CTRL_I2C_TRANSACTION_PARAMS and its records), which is
 * MIT-licensed:
 *
 *   Copyright (c) 2008-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *   Permission is hereby granted, free of charge, to any person obtaining a copy of this
 *   software and associated documentation files (the "Software"), to deal in the Software
 *   without restriction, including without limitation the rights to use, copy, modify,
 *   merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
 *   permit persons to whom the Software is furnished to do so, subject to the following
 *   conditions: The above copyright notice and this permission notice shall be included
 *   in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS",
 *   WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED.
 */
#include <linux/module.h>
#include <linux/slab.h>

#include "nvdrv.h"

/*
 * NV402C_CTRL_I2C_FLAGS_SPEED_MODE for the experiments. 0 = 100 kHz: with SPEED_MODE_DEFAULT
 * (6) GSP-RM rejects every transaction with NV_ERR_INVALID_ARGUMENT, at 100 kHz port 1
 * carries real bus traffic (a NACK at the reserved address 0x7f comes back as I2C_ERROR).
 * Port 2, where the DCB places the VRM, is rejected at every speed and mode.
 */
static uint nv_i2c_speed;
module_param(nv_i2c_speed, uint, 0644);
MODULE_PARM_DESC(nv_i2c_speed, "I2C experiments: NV402C speed mode (default 0 = 100 kHz)");

#define FINN_HDR_WORDS		4	/* version, payloadSize, interface, message */
#define FINN_BUF_BYTES		256

struct finn_w {
	u64 *pos, *end, acc;
	u32 empty;		/* free bits in acc */
	bool overrun;
};

static void fw_bits(struct finn_w *w, u64 v, u32 bits)
{
	if (bits >= w->empty) {
		bits -= w->empty;
		w->acc |= v << (64 - w->empty);
		v = w->empty == 64 ? 0 : v >> w->empty;
		if (w->pos >= w->end) {
			w->overrun = true;
			return;
		}
		*w->pos++ = w->acc;
		w->acc = 0;
		w->empty = 64;
	}
	if (bits)
		w->acc |= v << (64 - w->empty);
	w->empty -= bits;
}

static void fw_field(struct finn_w *w, u64 v, u32 bits)
{
	fw_bits(w, 1, 1);	/* present */
	fw_bits(w, v, bits);
}

static void fw_close(struct finn_w *w)
{
	if (w->empty < 64 && w->pos < w->end)
		*w->pos++ = w->acc;
}

struct finn_r {
	const u64 *pos, *end;
	u64 acc;
	u32 left;		/* unread bits in acc */
};

static u64 fr_bits(struct finn_r *r, u32 bits)
{
	u64 v;

	if (bits > r->left) {
		bits -= r->left;
		v = r->acc;
		r->acc = r->pos < r->end ? *r->pos++ : 0;
		if (bits >= 64) {
			v = r->acc;
			r->acc = 0;
			return v;
		}
		v |= (r->acc & ((1ULL << bits) - 1)) << r->left;
		r->left = 64;
	} else {
		v = r->acc & ((1ULL << bits) - 1);
	}
	r->acc >>= bits;
	r->left -= bits;
	return v;
}

static int fr_field(struct finn_r *r, u32 bits, u64 *v)
{
	if (!fr_bits(r, 1))
		return -EPROTO;	/* field missing: version mismatch */
	*v = fr_bits(r, bits);
	return 0;
}

/*
 * One SMBus transaction on board I2C port `port` with GSP-RM as bus master.
 * type: NV_I2C_QUICK (record { bWrite, warFlags }; with ping = true no data is transferred),
 * NV_I2C_SMBUS_BYTE or NV_I2C_SMBUS_WORD (command `reg`, data in *val).
 * addr8: 8-bit form of the 7-bit address. Caller holds gsp_lock; h_i2c is an NV40_I2C.
 */
int nv_i2c_xfer(struct nvdrv_dev *nv, u32 h_i2c, u8 port, u16 addr8, enum nv_i2c_type type, bool write,
		u8 reg, bool ping, u16 *val)
{
	const u32 cmd = 0x402c0105;	/* NV402C_CTRL_CMD_I2C_TRANSACTION */
	u32 flags = nv_i2c_speed << 1 | (ping ? 1 << 10 : 0);	/* 7-bit, SPEED_MODE, NORMAL/PING */
	struct finn_w w = {};
	struct finn_r r = {};
	u64 *buf, v;
	int ret;

	buf = kzalloc(FINN_BUF_BYTES, GFP_KERNEL);
	if (!buf)
		return -ENOMEM;
	w.pos = buf + FINN_HDR_WORDS;
	w.end = buf + FINN_BUF_BYTES / 8;
	w.empty = 64;
	fw_field(&w, port, 8);			/* portId */
	fw_field(&w, flags, 32);		/* flags */
	fw_field(&w, addr8, 16);		/* deviceAddress */
	fw_field(&w, type, 32);			/* transType */
	fw_bits(&w, 1, 1);			/* transData present */
	fw_bits(&w, 1, 1);			/* union variant present */
	fw_field(&w, write, 1);			/* bWrite */
	if (type == NV_I2C_QUICK)
		fw_field(&w, 0, 32);		/* warFlags */
	if (type == NV_I2C_SMBUS_BYTE || type == NV_I2C_SMBUS_WORD) {
		fw_field(&w, reg, 8);		/* registerAddress */
		fw_field(&w, val ? *val : 0, type == NV_I2C_SMBUS_WORD ? 16 : 8);	/* message */
	}
	fw_close(&w);
	if (w.overrun) {
		kfree(buf);
		return -E2BIG;
	}
	buf[0] = 0;				/* FINN_SERIALIZATION_VERSION */
	buf[1] = (u8 *)w.pos - (u8 *)buf;	/* payloadSize */
	buf[2] = cmd >> 8;			/* interface: FINN_NV40_I2C_I2C */
	buf[3] = cmd & 0xff;			/* message: TRANSACTION */

	ret = nv_rm_control_flags(nv, nv->rm.h_client, h_i2c, cmd, buf, FINN_BUF_BYTES,
				  NV_RMAPI_RPC_FLAGS_SERIALIZED);
	if (ret)
		goto out;
	/* the reply is the same message, serialised by GSP-RM */
	if (buf[0] != 0 || buf[1] < FINN_HDR_WORDS * 8 || buf[1] > FINN_BUF_BYTES) {
		ret = -EPROTO;
		goto out;
	}
	r.pos = buf + FINN_HDR_WORDS;
	r.end = buf + buf[1] / 8;
	ret = fr_field(&r, 8, &v) ?: fr_field(&r, 32, &v) ?: fr_field(&r, 16, &v) ?: fr_field(&r, 32, &v);
	if (ret)
		goto out;
	if (!fr_bits(&r, 1) || !fr_bits(&r, 1) || fr_field(&r, 1, &v)) {
		ret = -EPROTO;
		goto out;
	}
	if (type == NV_I2C_QUICK)
		ret = fr_field(&r, 32, &v);	/* warFlags */
	if (!ret && (type == NV_I2C_SMBUS_BYTE || type == NV_I2C_SMBUS_WORD)) {
		ret = fr_field(&r, 8, &v) ?: fr_field(&r, type == NV_I2C_SMBUS_WORD ? 16 : 8, &v);
		if (!ret && val)
			*val = v;
	}
out:
	kfree(buf);
	return ret;
}
