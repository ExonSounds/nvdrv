// SPDX-License-Identifier: MIT
/*
 * Texture and sampler headers for CUDA texture objects, built with Mesa's nil (the TIC, MIT) and
 * NVIDIA's open class headers for the TSC (cl9097tex.h and later, MIT, as NVK's nvk_sampler.c).
 * Only this file sees Mesa's headers; texobj.c owns the pools and the API.
 *
 * CUDA texture objects live on linear device memory: CU_RESOURCE_TYPE_LINEAR (a 1-D "buffer"
 * texture, fetched by integer index) or CU_RESOURCE_TYPE_PITCH2D (a pitch-linear 2-D image).
 * CUDA arrays (block-linear, mipmapped) are not supported yet.
 */
#include <stdio.h>
#include <string.h>

#include "nil.h"
#include "util/format/u_format.h"
#include "util/bitpack_helpers.h"
#include "drf.h"
#include "nvidia/classes/cl9097tex.h"
#include "nvidia/classes/cla097tex.h"
#include "nvidia/classes/clb197tex.h"

#include "cuda.h"
#include "compile.h"

/* CUDA (format, channels, read-as-integer) -> pipe_format */
static enum pipe_format pipe_fmt(CUarray_format f, unsigned int ch, bool as_int)
{
	switch (f) {
	case CU_AD_FORMAT_UNSIGNED_INT8:
		if (as_int)
			return ch == 1 ? PIPE_FORMAT_R8_UINT : ch == 2 ? PIPE_FORMAT_R8G8_UINT : PIPE_FORMAT_R8G8B8A8_UINT;
		return ch == 1 ? PIPE_FORMAT_R8_UNORM : ch == 2 ? PIPE_FORMAT_R8G8_UNORM : PIPE_FORMAT_R8G8B8A8_UNORM;
	case CU_AD_FORMAT_UNSIGNED_INT16:
		if (as_int)
			return ch == 1 ? PIPE_FORMAT_R16_UINT : ch == 2 ? PIPE_FORMAT_R16G16_UINT : PIPE_FORMAT_R16G16B16A16_UINT;
		return ch == 1 ? PIPE_FORMAT_R16_UNORM : ch == 2 ? PIPE_FORMAT_R16G16_UNORM : PIPE_FORMAT_R16G16B16A16_UNORM;
	case CU_AD_FORMAT_UNSIGNED_INT32:
		return ch == 1 ? PIPE_FORMAT_R32_UINT : ch == 2 ? PIPE_FORMAT_R32G32_UINT : PIPE_FORMAT_R32G32B32A32_UINT;
	case CU_AD_FORMAT_SIGNED_INT8:
		if (as_int)
			return ch == 1 ? PIPE_FORMAT_R8_SINT : ch == 2 ? PIPE_FORMAT_R8G8_SINT : PIPE_FORMAT_R8G8B8A8_SINT;
		return ch == 1 ? PIPE_FORMAT_R8_SNORM : ch == 2 ? PIPE_FORMAT_R8G8_SNORM : PIPE_FORMAT_R8G8B8A8_SNORM;
	case CU_AD_FORMAT_SIGNED_INT16:
		if (as_int)
			return ch == 1 ? PIPE_FORMAT_R16_SINT : ch == 2 ? PIPE_FORMAT_R16G16_SINT : PIPE_FORMAT_R16G16B16A16_SINT;
		return ch == 1 ? PIPE_FORMAT_R16_SNORM : ch == 2 ? PIPE_FORMAT_R16G16_SNORM : PIPE_FORMAT_R16G16B16A16_SNORM;
	case CU_AD_FORMAT_SIGNED_INT32:
		return ch == 1 ? PIPE_FORMAT_R32_SINT : ch == 2 ? PIPE_FORMAT_R32G32_SINT : PIPE_FORMAT_R32G32B32A32_SINT;
	case CU_AD_FORMAT_HALF:
		return ch == 1 ? PIPE_FORMAT_R16_FLOAT : ch == 2 ? PIPE_FORMAT_R16G16_FLOAT : PIPE_FORMAT_R16G16B16A16_FLOAT;
	case CU_AD_FORMAT_FLOAT:
		return ch == 1 ? PIPE_FORMAT_R32_FLOAT : ch == 2 ? PIPE_FORMAT_R32G32_FLOAT : PIPE_FORMAT_R32G32B32A32_FLOAT;
	default:
		return PIPE_FORMAT_NONE;
	}
}

static uint32_t address_mode(CUaddress_mode m)
{
	switch (m) {
	case CU_TR_ADDRESS_MODE_WRAP:	return NV9097_TEXSAMP0_ADDRESS_U_WRAP;
	case CU_TR_ADDRESS_MODE_MIRROR:	return NV9097_TEXSAMP0_ADDRESS_U_MIRROR;
	case CU_TR_ADDRESS_MODE_BORDER:	return NV9097_TEXSAMP0_ADDRESS_U_BORDER;
	default:			return NV9097_TEXSAMP0_ADDRESS_U_CLAMP_TO_EDGE;
	}
}

#define SET(w, drf, v)	((w) |= util_bitpack_uint((v), DRF_LO(drf), DRF_HI(drf)))

int nvc_tex_headers(const CUDA_RESOURCE_DESC *res, const CUDA_TEXTURE_DESC *desc, uint32_t tic[8], uint32_t tsc[8],
		    char *err, size_t errlen)
{
	const struct nv_device_info *dev = nvc_compiler_devinfo();
	bool as_int = desc->flags & CU_TRSF_READ_AS_INTEGER;
	bool normalized = desc->flags & CU_TRSF_NORMALIZED_COORDINATES;
	struct nil_descriptor d;
	enum pipe_format pf;
	unsigned int ch;
	uint32_t u, v, p, filt;
	int i;

	memset(tic, 0, 32);
	memset(tsc, 0, 32);
	switch (res->resType) {
	case CU_RESOURCE_TYPE_LINEAR: {
		uint32_t bpe;

		ch = res->res.linear.numChannels;
		pf = pipe_fmt(res->res.linear.format, ch, as_int);
		if (pf == PIPE_FORMAT_NONE || (ch != 1 && ch != 2 && ch != 4)) {
			snprintf(err, errlen, "unsupported texture format %u x%u", res->res.linear.format, ch);
			return -1;
		}
		bpe = util_format_get_blocksize(pf);
		if (res->res.linear.devPtr & 255 || !res->res.linear.sizeInBytes ||
		    res->res.linear.sizeInBytes / bpe > (1u << 27)) {
			snprintf(err, errlen, "linear texture: bad address or size");
			return -1;
		}
		d = nil_buffer_descriptor(dev, res->res.linear.devPtr, nil_format(pf),
					  (uint32_t)(res->res.linear.sizeInBytes / bpe));
		break;
	}
	case CU_RESOURCE_TYPE_PITCH2D: {
		struct nil_image_init_info info;
		struct nil_image img;
		struct nil_view view;

		ch = res->res.pitch2D.numChannels;
		pf = pipe_fmt(res->res.pitch2D.format, ch, as_int);
		if (pf == PIPE_FORMAT_NONE || (ch != 1 && ch != 2 && ch != 4)) {
			snprintf(err, errlen, "unsupported texture format %u x%u", res->res.pitch2D.format, ch);
			return -1;
		}
		if (res->res.pitch2D.devPtr & 255 || res->res.pitch2D.pitchInBytes & 31 || !res->res.pitch2D.width ||
		    !res->res.pitch2D.height || res->res.pitch2D.width > 65536 || res->res.pitch2D.height > 65536 ||
		    res->res.pitch2D.pitchInBytes < res->res.pitch2D.width * util_format_get_blocksize(pf)) {
			snprintf(err, errlen, "pitch2D texture: address must be 256-byte aligned, pitch a multiple of 32");
			return -1;
		}
		memset(&info, 0, sizeof(info));
		info.dim = NIL_IMAGE_DIM_2D;
		info.format = nil_format(pf);
		info.extent_px.width = res->res.pitch2D.width;
		info.extent_px.height = res->res.pitch2D.height;
		info.extent_px.depth = 1;
		info.extent_px.array_len = 1;
		info.levels = 1;
		info.samples = 1;
		info.usage = NIL_IMAGE_USAGE_LINEAR_BIT;
		info.modifier = 0;	/* DRM_FORMAT_MOD_LINEAR */
		info.explicit_row_stride_B = res->res.pitch2D.pitchInBytes;
		if (!nil_image_init(dev, &img, &info)) {
			snprintf(err, errlen, "pitch2D texture: nil_image_init failed");
			return -1;
		}
		memset(&view, 0, sizeof(view));
		view.view_type = NIL_VIEW_TYPE_2D;
		view.access = NIL_VIEW_ACCESS_TEXTURE;
		view.format = nil_format(pf);
		view.num_levels = 1;
		view.array_len = 1;
		view.swizzle[0] = PIPE_SWIZZLE_X;
		view.swizzle[1] = PIPE_SWIZZLE_Y;
		view.swizzle[2] = PIPE_SWIZZLE_Z;
		view.swizzle[3] = PIPE_SWIZZLE_W;
		d = nil_image_view_descriptor(dev, &img, &view, res->res.pitch2D.devPtr);
		break;
	}
	default:
		snprintf(err, errlen, "CUDA arrays are not supported yet (only linear and pitch2D memory)");
		return -1;
	}
	memcpy(tic, d.bits, 32);

	/* the sampler, as nvk_sampler_get_header */
	u = address_mode(desc->addressMode[0]);
	v = address_mode(desc->addressMode[1]);
	p = address_mode(desc->addressMode[2]);
	SET(tsc[0], NV9097_TEXSAMP0_ADDRESS_U, u);
	SET(tsc[0], NV9097_TEXSAMP0_ADDRESS_V, v);
	SET(tsc[0], NV9097_TEXSAMP0_ADDRESS_P, p);
	SET(tsc[0], NV9097_TEXSAMP0_FONT_FILTER_WIDTH, NV9097_TEXSAMP0_FONT_FILTER_WIDTH_SIZE_2);
	SET(tsc[0], NV9097_TEXSAMP0_FONT_FILTER_HEIGHT, NV9097_TEXSAMP0_FONT_FILTER_HEIGHT_SIZE_2);
	filt = desc->filterMode == CU_TR_FILTER_MODE_LINEAR;
	SET(tsc[1], NV9097_TEXSAMP1_MAG_FILTER, filt ? NV9097_TEXSAMP1_MAG_FILTER_MAG_LINEAR : NV9097_TEXSAMP1_MAG_FILTER_MAG_POINT);
	SET(tsc[1], NV9097_TEXSAMP1_MIN_FILTER, filt ? NV9097_TEXSAMP1_MIN_FILTER_MIN_LINEAR : NV9097_TEXSAMP1_MIN_FILTER_MIN_POINT);
	SET(tsc[1], NV9097_TEXSAMP1_MIP_FILTER, NV9097_TEXSAMP1_MIP_FILTER_MIP_NONE);
	SET(tsc[1], NVB197_TEXSAMP1_REDUCTION_FILTER, NVB197_TEXSAMP1_REDUCTION_FILTER_RED_NONE);
	SET(tsc[1], NVA097_TEXSAMP1_FLOAT_COORD_NORMALIZATION,
	    normalized ? NVA097_TEXSAMP1_FLOAT_COORD_NORMALIZATION_USE_HEADER_SETTING :
			 NVA097_TEXSAMP1_FLOAT_COORD_NORMALIZATION_FORCE_UNNORMALIZED_COORDS);
	SET(tsc[2], NV9097_TEXSAMP2_MAX_LOD_CLAMP, 0);
	for (i = 0; i < 4; i++) {
		uint32_t bits;

		memcpy(&bits, &desc->borderColor[i], 4);
		tsc[4 + i] = bits;
	}
	return 0;
}
