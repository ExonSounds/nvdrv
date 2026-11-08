#!/bin/sh
# Phase 6 test: full GSP-RM boot (FWSEC-FRTS, Booter on SEC2, sequencer/CORE_RESUME,
# GSP_INIT_DONE), N times. Writes GPU registers and blanks the console. Run as root over SSH.
set -eu
DEV=${1:-0000:01:00.0}; N=${2:-1}
HERE=$(cd "$(dirname "$0")/.." && pwd)
DBG=/sys/kernel/debug/nvdrv/$DEV
fail() { echo "FAIL: $*" >&2; dmesg | grep -E "nvdrv|AMD-Vi|DMAR" | tail -n 20 >&2; exit 1; }

i=1
while [ "$i" -le "$N" ]; do
	faults0=$(dmesg | grep -c IO_PAGE_FAULT || true)
	"$HERE/scripts/reset.sh" "$DEV" >/dev/null	# unbind (FLRs a running GSP first), then FLR
	HW=1 "$HERE/scripts/bind.sh" "$DEV" >/dev/null
	start=$(date +%s%N)
	echo 1 > "$DBG/run_gsp_boot" || fail "GSP boot failed"
	ms=$(( ($(date +%s%N) - start) / 1000000 ))
	grep -q "^gsp:.*RUNNING" "$DBG/info" || fail "driver does not report GSP running"
	sleep 1
	cpuctl=$(dd if="$DBG/mmio" bs=4 skip=$(( (0x111000 + 0x388) / 4 )) count=1 status=none | od -A n -t x4 | tr -d ' ')
	[ $((0x$cpuctl & 0x80)) -ne 0 ] || fail "GSP RISC-V not active 1s after INIT_DONE (cpuctl $cpuctl)"
	faults=$(( $(dmesg | grep -c IO_PAGE_FAULT || true) - faults0 ))
	[ "$faults" -eq 0 ] || fail "$faults IOMMU faults during boot"
	echo "cycle $i ok: GSP_INIT_DONE in ${ms} ms, RISC-V active, 0 IOMMU faults"
	i=$((i + 1))
done
echo "PASS"
