#!/bin/sh
# Phase 6 test: first synchronous RPC. Boots GSP-RM N times (FLR in between); each boot ends
# with GET_GSP_STATIC_INFO, whose reply is decoded independently and cross-checked.
# Writes GPU registers and blanks the console. Run as root over SSH.
set -eu
DEV=${1:-0000:01:00.0}; N=${2:-1}
HERE=$(cd "$(dirname "$0")/.." && pwd)
DBG=/sys/kernel/debug/nvdrv/$DEV
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
"$HERE/tools/gsp_static_info_probe.sh" > "$T/probe.txt"
first=

i=1
while [ "$i" -le "$N" ]; do
	"$HERE/tests/phase6_boot.sh" "$DEV" 1 >/dev/null || { echo "FAIL: boot $i" >&2; exit 1; }
	grep -q "^name:" "$DBG/info" || { echo "FAIL: no static info after boot $i" >&2; exit 1; }
	python3 -I "$HERE/tools/static_info.py" "$T/probe.txt" "$DBG" > "$T/out" || { cat "$T/out"; exit 1; }
	# Identity must not change between boots (handles and BAR PDEs may).
	id=$(grep -E "^(name|uuid|fb region)" "$T/out" "$DBG/info" | sort | md5sum)
	[ -z "$first" ] && { first=$id; cat "$T/out"; }
	[ "$id" = "$first" ] || { echo "FAIL: static info changed on boot $i" >&2; exit 1; }
	echo "boot $i ok: $(dmesg | grep -o 'rm: ".*' | tail -n 1)"
	i=$((i + 1))
done
echo "PASS"
