#!/bin/sh
# Unbind nvdrv from the GPU, clear the override and unload the module. Run as root.
set -eu
DEV=${1:-0000:01:00.0}
SYS=/sys/bus/pci/devices/$DEV

[ -e /sys/bus/pci/drivers/nvdrv/$DEV ] && echo "$DEV" > /sys/bus/pci/drivers/nvdrv/unbind
echo > "$SYS/driver_override"
lsmod | grep -q '^nvdrv ' && rmmod nvdrv
echo "unbound $DEV"
