#!/bin/sh
# Print sizeof/offsetof for the RM API structs nvdrv sends through GSP_RM_ALLOC /
# GSP_RM_CONTROL, by compiling NVIDIA's SDK headers (ref/ogkm) in a host program.
# Used to derive the layouts asserted in src/rm.c; rerun when moving firmware versions.
set -eu
R=${1:-$(cd "$(dirname "$0")/.." && pwd)/ref/ogkm/src}
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
cat > "$T/probe.c" <<'EOF'
#include <stdio.h>
#include <stddef.h>
#include "nvtypes.h"
#include "class/cl0000.h"
#include "class/cl0080.h"
#include "class/cl2080.h"
#include "ctrl/ctrl2080/ctrl2080gpu.h"
#include "ctrl/ctrl2080/ctrl2080internal.h"
#include "nvos.h"
#include "ctrl/ctrl0073/ctrl0073dfp.h"
#include "ctrl/ctrl0073/ctrl0073dp.h"

#define S(t)    printf("%-56s size %4zu\n", #t, sizeof(t))
#define F(t, f) printf("  %-54s %4zu %3zu\n", #f, offsetof(t, f), sizeof(((t *)0)->f))

int main(void)
{
	S(NV0000_ALLOC_PARAMETERS);
	F(NV0000_ALLOC_PARAMETERS, hClient); F(NV0000_ALLOC_PARAMETERS, processID);
	F(NV0000_ALLOC_PARAMETERS, processName); F(NV0000_ALLOC_PARAMETERS, pOsPidInfo);
	S(NV0080_ALLOC_PARAMETERS);
	F(NV0080_ALLOC_PARAMETERS, deviceId); F(NV0080_ALLOC_PARAMETERS, hClientShare);
	F(NV0080_ALLOC_PARAMETERS, hTargetClient); F(NV0080_ALLOC_PARAMETERS, hTargetDevice);
	F(NV0080_ALLOC_PARAMETERS, flags); F(NV0080_ALLOC_PARAMETERS, vaSpaceSize);
	F(NV0080_ALLOC_PARAMETERS, vaStartInternal); F(NV0080_ALLOC_PARAMETERS, vaLimitInternal);
	F(NV0080_ALLOC_PARAMETERS, vaMode);
	S(NV2080_ALLOC_PARAMETERS);
	F(NV2080_ALLOC_PARAMETERS, subDeviceId);
	S(NV2080_CTRL_GPU_GET_CHIP_DETAILS_PARAMS);
	F(NV2080_CTRL_GPU_GET_CHIP_DETAILS_PARAMS, pciDevId);
	F(NV2080_CTRL_GPU_GET_CHIP_DETAILS_PARAMS, chipSku);
	F(NV2080_CTRL_GPU_GET_CHIP_DETAILS_PARAMS, chipMajor);
	F(NV2080_CTRL_GPU_GET_CHIP_DETAILS_PARAMS, chipMinor);
#define M(f) F(NV2080_CTRL_INTERNAL_MEMSYS_GET_STATIC_CONFIG_PARAMS, f)
	S(NV2080_CTRL_INTERNAL_MEMSYS_GET_STATIC_CONFIG_PARAMS);
	M(bOneToOneComptagLineAllocation); M(bEnabledEccFBPA); M(bL2PreFill); M(l2CacheSize);
	M(bFbpaPresent); M(comprPageSize); M(comprPageShift); M(ramType); M(ltcCount);
	M(ltsPerLtcCount);
#define P(f) F(NV2080_CTRL_INTERNAL_DISPLAY_CHANNEL_PUSHBUFFER_PARAMS, f)
	S(NV2080_CTRL_INTERNAL_DISPLAY_CHANNEL_PUSHBUFFER_PARAMS);
	P(addressSpace); P(physicalAddr); P(limit); P(cacheSnoop); P(hclass); P(channelInstance);
	P(valid); P(pbTargetAperture); P(channelPBSize); P(subDeviceId);
#define W(f) F(NV2080_CTRL_INTERNAL_DISPLAY_WRITE_INST_MEM_PARAMS, f)
	S(NV2080_CTRL_INTERNAL_DISPLAY_WRITE_INST_MEM_PARAMS);
	W(instMemPhysAddr); W(instMemSize); W(instMemAddrSpace); W(instMemCpuCacheAttr);
#define V(f) F(NV50VAIO_CHANNELDMA_ALLOCATION_PARAMETERS, f)
	S(NV50VAIO_CHANNELDMA_ALLOCATION_PARAMETERS);
	V(channelInstance); V(hObjectBuffer); V(hObjectNotify); V(offset); V(pControl); V(flags);
	V(channelPBSize); V(subDeviceId);
#define A(f) F(NV0073_CTRL_DFP_ASSIGN_SOR_PARAMS, f)
	S(NV0073_CTRL_DFP_ASSIGN_SOR_PARAMS);
	A(displayId); A(sorExcludeMask); A(slaveDisplayId); A(forceSublinkConfig); A(bIs2Head1Or);
	A(sorAssignList); A(sorAssignListWithTag); A(reservedSorMask); A(flags);
	S(NV0073_CTRL_DFP_ASSIGN_SOR_INFO);
	F(NV0073_CTRL_DFP_ASSIGN_SOR_INFO, displayMask); F(NV0073_CTRL_DFP_ASSIGN_SOR_INFO, sorType);
#define X(f) F(NV0073_CTRL_DP_AUXCH_CTRL_PARAMS, f)
	S(NV0073_CTRL_DP_AUXCH_CTRL_PARAMS);
	X(displayId); X(bAddrOnly); X(cmd); X(addr); X(data); X(size); X(replyType); X(retryTimeMs);
#define C(f) F(NV0073_CTRL_CMD_DP_CONFIG_STREAM_PARAMS, f)
	S(NV0073_CTRL_CMD_DP_CONFIG_STREAM_PARAMS);
	C(head); C(sorIndex); C(dpLink); C(bEnableOverride); C(bMST); C(singleHeadMultistreamMode);
	C(hBlankSym); C(vBlankSym); C(colorFormat); C(bEnableTwoHeadOneOr); C(MST); C(SST);
	C(SST.bEnhancedFraming); C(SST.tuSize); C(SST.waterMark); C(SST.bEnableAudioOverRightPanel);
	S(NV0073_CTRL_CMD_DP_GET_CAPS_PARAMS);
	F(NV0073_CTRL_CMD_DP_GET_CAPS_PARAMS, sorIndex);
	F(NV0073_CTRL_CMD_DP_GET_CAPS_PARAMS, maxLinkRate);
	F(NV0073_CTRL_CMD_DP_GET_CAPS_PARAMS, bHasIncreasedWatermarkLimits);
	return 0;
}
EOF
gcc -std=gnu11 -I"$R/common/sdk/nvidia/inc" -o "$T/probe" "$T/probe.c"
"$T/probe"
