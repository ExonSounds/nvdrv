#!/bin/sh
# Load nvdrv and bind it to the GPU via driver_override. Run as root.
#   HW=1 scripts/bind.sh   load with allow_hw_writes=1 (exposes run_* triggers)
set -eu
DEV=${1:-0000:01:00.0}
SYS=/sys/bus/pci/devices/$DEV
HERE=$(cd "$(dirname "$0")/.." && pwd)

if [ -e "$SYS/driver" ]; then
	echo "$DEV is already bound to $(basename "$(readlink -f "$SYS/driver")")" >&2
	exit 1
fi
if lsmod | grep -q '^nvdrv '; then
	want=$([ "${HW:-0}" = 1 ] && echo Y || echo N)
	have=$(cat /sys/module/nvdrv/parameters/allow_hw_writes)
	[ "$want" = "$have" ] || rmmod nvdrv
fi
lsmod | grep -q '^nvdrv ' || insmod "$HERE/nvdrv.ko" allow_hw_writes=$([ "${HW:-0}" = 1 ] && echo 1 || echo 0)
echo nvdrv > "$SYS/driver_override"
echo "$DEV" > /sys/bus/pci/drivers_probe
dmesg | grep nvdrv | tail -n 3
