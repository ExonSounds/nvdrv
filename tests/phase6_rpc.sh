#!/bin/sh
# Phase 6 (host side) test: pre-boot RPCs in the command queue vs independent decode. Read-only.
set -eu
DEV=${1:-0000:01:00.0}
HERE=$(cd "$(dirname "$0")/.." && pwd)
DBG=/sys/kernel/debug/nvdrv/$DEV
[ -e "$DBG/info" ] || "$HERE/scripts/bind.sh" "$DEV" >/dev/null
python3 -I "$HERE/tools/rpc.py" "$DBG" "$DEV"
