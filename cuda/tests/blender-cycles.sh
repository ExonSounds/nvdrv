#!/bin/sh
# Render Blender's default scene with Cycles on nvcuda (GPU, CUDA device), headless.
#   cuda/tests/blender-cycles.sh /path/to/blender-5.2.3-linux-x64 [output-dir]
# Needs nvdrv loaded with nvk=1 and nvcuda built (make -C cuda). Blender's lib/ ships a prebuilt
# kernel_sm_86.cubin.zst, which Cycles picks before the PTX and never falls back from, so it is
# set aside (renamed) for the run; the PTX (kernel_compute_75.ptx.zst) is what nvcuda compiles.
set -e
B=${1:?blender directory}
OUT=${2:-/tmp/nvcuda-blender}
HERE=$(cd "$(dirname "$0")" && pwd)
LIB=$(ls -d "$B"/*/scripts/addons_core/cycles/lib | head -1)
CUBIN=$LIB/kernel_sm_86.cubin.zst
[ -f "$CUBIN" ] && mv "$CUBIN" "$CUBIN.hidden"
mkdir -p "$OUT"
export LD_LIBRARY_PATH="$HERE/../build${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export NVCUDA_DEBUG=1
"$B/blender" -b --factory-startup --python-expr '
import bpy
p = bpy.context.preferences.addons["cycles"].preferences
p.compute_device_type = "CUDA"
p.get_devices()
print("Cycles devices:", [(d.name, d.type, d.use) for d in p.devices])
for d in p.devices:
    d.use = d.type == "CUDA"
s = bpy.context.scene
s.cycles.device = "GPU"
s.cycles.samples = 16
s.cycles.use_denoising = False
s.render.resolution_percentage = 50
' -E CYCLES -o "$OUT/frame_#" -F PNG -f 1 2>&1 | tee "$OUT/blender.log" | grep -vE '^$' | tail -40
echo "output: $OUT"
