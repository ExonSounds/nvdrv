#!/bin/sh
# Display test: boot GSP-RM, bring up the DisplayPort sink (settle, train, modeset,
# verify the link holds, retry), and scan out a test pattern from VRAM.
#   DRM=1   also register the DRM device and put a text console (tty3) on it
#   KEEP=1  leave the display up afterwards (otherwise unbind and rebind at the end)
# Checks what the hardware and the sink report; the picture itself has to be looked at.
# Writes GPU registers and takes over the monitor. Run as root over SSH.
set -eu
DEV=${1:-0000:01:00.0}
HERE=$(cd "$(dirname "$0")/.." && pwd)
DBG=/sys/kernel/debug/nvdrv/$DEV
fail() { echo "FAIL: $*" >&2; dmesg | grep -E "nvdrv|AMD-Vi" | tail -n 25 >&2; exit 1; }
rd() { dd if="$DBG/mmio" bs=4 skip=$(($1 / 4)) count=1 status=none | od -A n -t x4 | tr -d ' '; }

faults0=$(dmesg | grep -c IO_PAGE_FAULT || true)
"$HERE/tests/phase6_boot.sh" "$DEV" 1 >/dev/null || fail "GSP boot"
echo 1 > "$DBG/disp_up" || fail "display bring-up (see dmesg for the attempts)"
grep -q "(in sync)" "$DBG/dp_status" || fail "sink not in sync: $(cat "$DBG/dp_status")"

W=$((0x690000)); C=$((0x680000))
# Assembly state == armed state for the surface methods (size, ctxdma, format, pitch).
for off in 0x224 0x240 0x22c 0x230; do
	a=$(rd $((W + off))); b=$(rd $((W + 0x800 + off)))
	[ "$a" = "$b" ] || fail "window method $off not armed (assembly 0x$a, armed 0x$b)"
done
[ "$(rd $((W + 0x800 + 0x240)))" != "00000000" ] || fail "no surface armed"
s1=$(rd $((C + 0x220))); sleep 0.01; s2=$(rd $((C + 0x220)))
[ "$s1" != "$s2" ] || fail "head 0 scanline not moving (0x$s1)"
size=$(rd $((W + 0x800 + 0x224)))
echo "scanning out $((0x$size & 0xffff))x$((0x$size >> 16)), link: $(grep 0x200 "$DBG/dp_status" | sed 's/^ *//')"

if [ "${DRM:-0}" = 1 ]; then
	echo 1 > "$DBG/drm_register" || fail "DRM registration"
	[ "$(cat /sys/class/graphics/fb0/name)" = nvdrvdrmfb ] || fail "fb0 is not nvdrv's"
	chvt 3
	sleep 0.5
	printf '\n\033[1;32m*** nvdrv: this console is drawn by nvdrv (GSP-RM display, BAR1 scanout) ***\033[0m\n' > /dev/tty3
	card=$(basename "$(ls -d /sys/bus/pci/devices/"$DEV"/drm/card* | head -n 1)")
	grep -q "active=1" "/sys/kernel/debug/dri/${card#card}/state" || fail "CRTC not active after console switch"
	echo "DRM: $card, fb0 = nvdrvdrmfb, console on tty3"
fi

if [ "${KEEP:-0}" != 1 ]; then
	echo "$DEV" > /sys/bus/pci/drivers/nvdrv/unbind
	echo "$DEV" > /sys/bus/pci/drivers_probe
fi
faults=$(( $(dmesg | grep -c IO_PAGE_FAULT || true) - faults0 ))
[ "$faults" -eq 0 ] || fail "$faults IOMMU faults"
echo "PASS (hardware checks; the picture itself must be verified by looking at the monitor)"
