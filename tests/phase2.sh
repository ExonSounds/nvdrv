#!/bin/sh
# Phase 2 test: kernel VBIOS/FWSEC/FRTS preparation vs independent Python parse. Run as root.
set -eu
DEV=${1:-0000:01:00.0}
HERE=$(cd "$(dirname "$0")/.." && pwd)
DBG=/sys/kernel/debug/nvdrv/$DEV
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT

[ -e "$DBG/info" ] || "$HERE/scripts/bind.sh" "$DEV" >/dev/null
cat "$DBG/vbios" > "$T/vbios.bin"
cat "$DBG/fwsec_frts.bin" > "$T/fwsec_frts.bin"
frts=$(sed -n 's/^frts: *\(0x[0-9a-f]*\) .*/\1/p' "$DBG/info")
fuse=$(sed -n 's/^ucode id:.*fuse version \([0-9]*\).*/\1/p' "$DBG/info")
python3 -I "$HERE/tools/vbios.py" "$T/vbios.bin" > /dev/null
python3 -I "$HERE/tests/check_fwsec_frts.py" "$T/vbios.bin" "$T/fwsec_frts.bin" "$frts" "$fuse"
