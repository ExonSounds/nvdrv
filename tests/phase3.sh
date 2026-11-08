#!/bin/sh
# Phase 3 test: FLR -> bind (waits for GFW boot) -> FWSEC-FRTS -> verify WPR2. Repeats N times.
# Writes GPU registers and blanks the local console. Run as root over SSH.
set -eu
DEV=${1:-0000:01:00.0}; N=${2:-1}
HERE=$(cd "$(dirname "$0")/.." && pwd)
DBG=/sys/kernel/debug/nvdrv/$DEV
fail() { echo "FAIL: $*" >&2; exit 1; }

i=1
while [ "$i" -le "$N" ]; do
	"$HERE/scripts/reset.sh" "$DEV" >/dev/null
	HW=1 "$HERE/scripts/bind.sh" "$DEV" >/dev/null
	grep -q "^wpr2:.*not set up" "$DBG/info" || fail "WPR2 survived FLR"
	frts=$(sed -n 's/^frts: *\(0x[0-9a-f]*\) .*/\1/p' "$DBG/info")
	echo 1 > "$DBG/run_fwsec_frts" || fail "FWSEC-FRTS failed (see dmesg)"
	lo=$(sed -n 's/^wpr2: *\(0x[0-9a-f]*\) .*/\1/p' "$DBG/info")
	[ "$lo" = "$frts" ] || fail "WPR2 at $lo, expected $frts"
	echo "cycle $i ok: WPR2 at $lo"
	i=$((i + 1))
done
echo "PASS"
