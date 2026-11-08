#!/bin/sh
# Phase 5 (host side) test: LibOS init args + RPC queue memory vs independent decode. Read-only.
set -eu
DEV=${1:-0000:01:00.0}
HERE=$(cd "$(dirname "$0")/.." && pwd)
DBG=/sys/kernel/debug/nvdrv/$DEV
[ -e "$DBG/info" ] || "$HERE/scripts/bind.sh" "$DEV" >/dev/null
[ -e "$DBG/libos_args.bin" ] || { echo "FAIL: no libos_args.bin (see dmesg)"; exit 1; }
python3 -I "$HERE/tools/libos.py" "$DBG"
