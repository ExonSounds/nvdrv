#!/bin/sh
# Print sizeof/offsetof for GspStaticConfigInfo (the GET_GSP_STATIC_INFO reply) and the
# nested structs nvdrv decodes, by compiling NVIDIA's own definitions (ref/ogkm) in a
# small host program. Used to derive the layout asserted in src/rpc.c; rerun when
# moving to a new firmware version.
set -eu
R=${1:-$(cd "$(dirname "$0")/.." && pwd)/ref/ogkm/src}
N=$R/nvidia/inc/kernel
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
{
echo '#include <stdio.h>'; echo '#include <stddef.h>'; echo '#include "nvtypes.h"'
echo '#include "ctrl/ctrl0080/ctrl0080gpu.h"'; echo '#include "ctrl/ctrl0080/ctrl0080gr.h"'
echo '#include "ctrl/ctrl2080/ctrl2080bios.h"'; echo '#include "ctrl/ctrl2080/ctrl2080fb.h"'
echo '#include "ctrl/ctrl2080/ctrl2080gpu.h"'
sed -n 's/^\s*RM_ENGINE_TYPE_LAST\s*=\s*(\(0x[0-9a-fA-F]*\)).*/#define RM_ENGINE_TYPE_LAST \1/p' "$N/gpu/gpu_engine_type.h"
grep -E '#define NVGPU_ENGINE_CAPS_MASK_(BITS|ARRAY_MAX)\s' "$N/gpu/nvbitmask.h"
grep -E '#define MAX_GROUP_COUNT\s' "$R/nvidia/generated/g_gpu_nvoc.h"
sed -n '/^typedef struct VIRTUAL_DISPLAY_GET_MAX_RESOLUTION_PARAMS/,/^} VIRTUAL_DISPLAY_GET_NUM_HEADS_PARAMS;/p' "$N/vgpu/rpc_headers.h"
awk '/^typedef struct$/{buf=""; p=1} p{buf=buf"\n"$0} p && /^} (EcidManufacturingInfo|FW_WPR_LAYOUT_OFFSET);/{print buf; p=0}' "$N/gpu/gsp/gsp_static_config.h"
sed -n '/^typedef struct GspStaticConfigInfo_t/,/^} GspStaticConfigInfo;/p' "$N/gpu/gsp/gsp_static_config.h"
echo 'int main(void) {'
echo 'printf("sizeof GspStaticConfigInfo %zu\n", sizeof(GspStaticConfigInfo));'
sed -n '/^typedef struct GspStaticConfigInfo_t/,/^} GspStaticConfigInfo;/p' "$N/gpu/gsp/gsp_static_config.h" |
	grep -oE '^\s+[A-Za-z_0-9]+\s+[A-Za-z_0-9]+(\[[A-Za-z_0-9]+\])?;' |
	sed -E 's/\[.*\]//' |
	awk '{gsub(";","",$2); printf "printf(\"%%-36s %%5zu %%5zu\\n\", \"%s\", offsetof(GspStaticConfigInfo, %s), sizeof(((GspStaticConfigInfo*)0)->%s));\n", $2, $2, $2}'
echo 'printf("FB_REGION_INFO: sizeof %zu, fbRegion @%zu, entry %zu\n", sizeof(NV2080_CTRL_CMD_FB_GET_FB_REGION_INFO_PARAMS), offsetof(NV2080_CTRL_CMD_FB_GET_FB_REGION_INFO_PARAMS, fbRegion), sizeof(NV2080_CTRL_CMD_FB_GET_FB_REGION_FB_REGION_INFO));'
for f in base limit reserved performance supportCompressed supportISO bProtected blackList; do
	echo "printf(\"  region.%-18s %3zu\\n\", \"$f\", offsetof(NV2080_CTRL_CMD_FB_GET_FB_REGION_FB_REGION_INFO, $f));"
done
echo 'printf("GID_INFO: sizeof %zu, length @%zu, data @%zu\n", sizeof(NV2080_CTRL_GPU_GET_GID_INFO_PARAMS), offsetof(NV2080_CTRL_GPU_GET_GID_INFO_PARAMS, length), offsetof(NV2080_CTRL_GPU_GET_GID_INFO_PARAMS, data));'
echo 'printf("SKU_INFO: sizeof %zu, chipSKU @%zu, project @%zu, projectSKU @%zu\n", sizeof(NV2080_CTRL_BIOS_GET_SKU_INFO_PARAMS), offsetof(NV2080_CTRL_BIOS_GET_SKU_INFO_PARAMS, chipSKU), offsetof(NV2080_CTRL_BIOS_GET_SKU_INFO_PARAMS, project), offsetof(NV2080_CTRL_BIOS_GET_SKU_INFO_PARAMS, projectSKU));'
echo 'return 0; }'
} > "$T/probe.c"
gcc -std=gnu11 -I"$R/common/sdk/nvidia/inc" -o "$T/probe" "$T/probe.c"
"$T/probe"
