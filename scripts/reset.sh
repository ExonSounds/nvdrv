#!/bin/sh
# Unbind nvdrv and FLR the GPU. Clears WPR2; the VBIOS re-runs GFW boot (~300 ms),
# which nvdrv's probe waits for. Blanks the local console until reboot. Run as root.
set -eu
DEV=${1:-0000:01:00.0}
HERE=$(cd "$(dirname "$0")/.." && pwd)
"$HERE/scripts/unbind.sh" "$DEV" >/dev/null 2>&1 || true
echo 1 > "/sys/bus/pci/devices/$DEV/reset"
echo "reset $DEV (methods: $(cat /sys/bus/pci/devices/$DEV/reset_method))"
