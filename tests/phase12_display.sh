#!/bin/sh
# Display test: boot GSP-RM, discover the DisplayPort sink, train the link, modeset its
# EDID preferred mode and scan out a test pattern (colour bars + grey ramp + white
# border) from VRAM. Checks what the hardware reports; the picture itself has to be
# looked at. Writes GPU registers and takes over the monitor. Run as root over SSH.
set -eu
DEV=${1:-0000:01:00.0}
HERE=$(cd "$(dirname "$0")/.." && pwd)
DBG=/sys/kernel/debug/nvdrv/$DEV
fail() { echo "FAIL: $*" >&2; dmesg | grep -E "nvdrv|AMD-Vi" | tail -n 25 >&2; exit 1; }
rd() { dd if="$DBG/mmio" bs=4 skip=$(($1 / 4)) count=1 status=none | od -A n -t x4 | tr -d ' '; }

faults0=$(dmesg | grep -c IO_PAGE_FAULT || true)
"$HERE/tests/phase6_boot.sh" "$DEV" 1 >/dev/null || fail "GSP boot"
echo 1 > "$DBG/disp_probe" || fail "display probe"
grep -qE "connected SOR[0-9-]+ DP-" "$DBG/info" || fail "no connected DisplayPort sink"
echo 1 > "$DBG/disp_link" || fail "DP link training"
dmesg | tail -n 5 | grep -q "disp: link up" || fail "link not reported up"
echo 0 > "$DBG/disp_modeset" || fail "modeset"

W=$((0x690000)); C=$((0x680000))
# The window update must not have been skipped (FE_EXCEPT RESTART_MODE = SKIP).
wexc=$(rd $((0x611020 + 12)))
[ $((0x$wexc & 0x30000000)) -eq 0 ] || fail "window update raised exception 0x$wexc (err 0x$(rd $((0x611028 + 12))))"
# Assembly state == armed state for the surface methods (size, ctxdma, format, pitch).
for off in 0x224 0x240 0x22c 0x230; do
	a=$(rd $((W + off))); b=$(rd $((W + 0x800 + off)))
	[ "$a" = "$b" ] || fail "window method $off not armed (assembly 0x$a, armed 0x$b)"
done
[ "$(rd $((W + 0x800 + 0x240)))" != "00000000" ] || fail "no surface armed"
# Raster generator of head 0 is running.
s1=$(rd $((C + 0x220))); sleep 0.01; s2=$(rd $((C + 0x220)))
[ "$s1" != "$s2" ] || fail "head 0 scanline not moving (0x$s1)"
size=$(rd $((W + 0x800 + 0x224)))
echo "scanning out $((0x$size & 0xffff))x$((0x$size >> 16)) from ctxdma 0x$(rd $((W + 0x800 + 0x240))), scanline 0x$s1 -> 0x$s2"

if [ "${KEEP:-0}" != 1 ]; then
	echo "$DEV" > /sys/bus/pci/drivers/nvdrv/unbind
	echo "$DEV" > /sys/bus/pci/drivers_probe
fi
faults=$(( $(dmesg | grep -c IO_PAGE_FAULT || true) - faults0 ))
[ "$faults" -eq 0 ] || fail "$faults IOMMU faults"
echo "PASS (hardware checks; the test pattern itself must be verified by looking at the monitor)"
