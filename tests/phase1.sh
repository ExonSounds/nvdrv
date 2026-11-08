#!/bin/sh
# Phase 1 smoke test: bind, identify, read MMIO + VBIOS, unbind, rebind. Run as root.
set -eu
DEV=${1:-0000:01:00.0}
HERE=$(cd "$(dirname "$0")/.." && pwd)
DBG=/sys/kernel/debug/nvdrv/$DEV
fail() { echo "FAIL: $*" >&2; exit 1; }

"$HERE/scripts/unbind.sh" "$DEV" >/dev/null 2>&1 || true
for round in 1 2; do
	"$HERE/scripts/bind.sh" "$DEV" >/dev/null
	grep -q '^chip:' "$DBG/info" || fail "no chip in info"
	boot0=$(od -A n -t x4 -N 4 "$DBG/mmio" | tr -d ' ')
	[ "$boot0" != "ffffffff" ] || fail "BOOT_0 dead"
	sig=$(od -A n -t x1 -N 2 "$DBG/vbios" | tr -d ' ')
	[ "$sig" = "55aa" ] || fail "VBIOS signature $sig"
	"$HERE/scripts/unbind.sh" "$DEV" >/dev/null
	[ ! -e "$DBG" ] || fail "debugfs not cleaned up"
	echo "round $round ok (boot0=$boot0)"
done
grep -q simpledrm /proc/fb || fail "console framebuffer disappeared"
echo "PASS"
