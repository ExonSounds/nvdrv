#!/bin/sh
# Phase 5 (host side) test: FB layout + GspFwWprMeta vs independent Python computation. Read-only.
set -eu
DEV=${1:-0000:01:00.0}
HERE=$(cd "$(dirname "$0")/.." && pwd)
DBG=/sys/kernel/debug/nvdrv/$DEV
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT

[ -e "$DBG/info" ] || "$HERE/scripts/bind.sh" "$DEV" >/dev/null
[ -e "$DBG/wpr_meta.bin" ] || { echo "FAIL: no wpr_meta.bin (see dmesg)"; exit 1; }
cat "$DBG/wpr_meta.bin" > "$T/meta.bin"
# Hardware inputs read directly, not from the driver's computed layout
fb_mb=$(dd if="$DBG/mmio" bs=4 skip=$((0x1183a4 / 4)) count=1 status=none | od -A n -t u4 | tr -d ' ')
vga_reg=$(dd if="$DBG/mmio" bs=4 skip=$((0x625f04 / 4)) count=1 status=none | od -A n -t u4 | tr -d ' ')
fb=$((fb_mb << 20))
if [ $((vga_reg & 8)) -ne 0 ]; then echo "VGA workspace valid in HW: extend this test" >&2; exit 1; fi
vga=$((fb - 0x100000))
addr() { sed -n "s/^$1.* \(0x[0-9a-f]\{16\}\).*/\1/p" "$DBG/info" | head -1; }
python3 -I "$HERE/tools/layout.py" "$T/meta.bin" /lib/firmware/nvidia/ga104/gsp 570.144 \
	"$fb" "$vga" "$(addr 'gsp image:')" "$(addr 'bootloader:')" "$(addr 'gsp signature:')"
