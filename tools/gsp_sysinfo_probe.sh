#!/bin/sh
# Print sizeof/offsetof for GspSystemInfo by compiling NVIDIA's own struct definitions
# (open-gpu-kernel-modules, ref/ogkm) in a small host program. Used to derive the
# layout asserted in src/rpc.c; rerun when moving to a new firmware version.
set -eu
R=${1:-$(cd "$(dirname "$0")/.." && pwd)/ref/ogkm/src}
H=$R/nvidia/inc/kernel/gpu
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
{
echo '#include <stdio.h>'; echo '#include <stddef.h>'; echo '#include "nvtypes.h"'
echo 'typedef NvU32 NV_STATUS;'; echo '#define NV0073_CTRL_SYSTEM_ACPI_ID_MAP_MAX_DISPLAYS (16U)'
awk '/^typedef struct$/{buf=""; p=1} p{buf=buf"\n"$0} /} BUSINFO;/{print buf; exit}' "$R/nvidia/generated/g_chipset_nvoc.h"
sed -n '/^typedef struct DOD_METHOD_DATA/,/^} ACPI_METHOD_DATA;/p' "$H/gpu_acpi_data.h"
sed -n '/^typedef struct GSP_VF_INFO/,/^} GSP_PCIE_CONFIG_REG;/p' "$H/gsp/gsp_static_config.h"
sed -n '/^typedef struct GspSystemInfo/,/^} GspSystemInfo;/p' "$H/gsp/gsp_static_config.h"
echo 'int main(void) { printf("sizeof %zu\n", sizeof(GspSystemInfo));'
sed -n '/^typedef struct GspSystemInfo/,/^} GspSystemInfo;/p' "$H/gsp/gsp_static_config.h" |
	grep -oE '^\s+[A-Za-z_0-9]+\s+[A-Za-z_0-9]+;' |
	awk '{gsub(";","",$2); printf "printf(\"%%-36s %%4zu %%3zu\\n\", \"%s\", offsetof(GspSystemInfo, %s), sizeof(((GspSystemInfo*)0)->%s));\n", $2, $2, $2}'
echo 'return 0; }'
} > "$T/probe.c"
gcc -std=gnu11 -I"$R/common/sdk/nvidia/inc" -o "$T/probe" "$T/probe.c"
"$T/probe"
