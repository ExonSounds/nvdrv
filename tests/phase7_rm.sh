#!/bin/sh
# Phase 7 test: nvdrv's own RM objects inside GSP-RM. Boots GSP-RM N times (FLR in between);
# each boot allocates client -> device -> subdevice and runs two controls. Checks the answers
# against independent sources, runs the RM error-path self test, and checks that unbind
# frees the client. Writes GPU registers and blanks the console. Run as root over SSH.
set -eu
DEV=${1:-0000:01:00.0}; N=${2:-1}
HERE=$(cd "$(dirname "$0")/.." && pwd)
DBG=/sys/kernel/debug/nvdrv/$DEV
SYS=/sys/bus/pci/devices/$DEV
fail() { echo "FAIL: $*" >&2; dmesg | grep -E "nvdrv|AMD-Vi" | tail -n 15 >&2; exit 1; }
field() { sed -n "s/^$1:[[:space:]]*//p" "$DBG/info"; }

i=1
while [ "$i" -le "$N" ]; do
	faults0=$(dmesg | grep -c IO_PAGE_FAULT || true)
	"$HERE/tests/phase6_boot.sh" "$DEV" 1 >/dev/null || fail "boot $i"

	handles=$(field handles)
	[ "$handles" = "client 0xc1d00001 device 0xd0000080 subdevice 0xd0002080" ] || fail "handles: '$handles'"

	# GPU_GET_CHIP_DETAILS vs PCI config space, BOOT_0 and the static info's SKU
	set -- $(field "chip details" | tr ',' ' ')	# pci dev 0x00002489 sku 202 rev a.1
	pcidev=$(cat "$SYS/device")
	[ $(($3)) -eq $((pcidev)) ] || fail "chip details pci dev $3 != config space $pcidev"
	boot_rev=$(field chip | sed -n 's/.*rev \([0-9a-f]\)\([0-9a-f]\)).*/\1.\2/p')
	[ "$7" = "$boot_rev" ] || fail "chip details rev $7 != BOOT_0 rev $boot_rev"
	sku=$(field sku | sed -n 's/^chip \([^,]*\),.*/\1/p')
	[ "$5" = "$sku" ] || fail "chip details sku $5 != static info sku $sku"

	# INTERNAL_MEMSYS_GET_STATIC_CONFIG through GSP-RM's internal client: plausible values
	ms=$(field memsys)
	echo "$ms" | grep -qE "^L2 [1-9][0-9]* KiB, ram type 0x[0-9a-f]+ \((GDDR[0-9X]+|HBM[0-9])\), [1-9][0-9]* LTC x [1-9][0-9]* LTS" ||
		fail "memsys: '$ms'"

	# error paths + a second client alloc/free, then a normal call
	echo 1 > "$DBG/rm_selftest" || fail "rm_selftest"
	dmesg | tail -n 20 | grep -q "rm selftest: PASS" || fail "rm_selftest did not report PASS"

	echo "$DEV" > /sys/bus/pci/drivers/nvdrv/unbind
	dmesg | tail -n 5 | grep -q "rm: client 0xc1d00001 freed" || fail "client not freed on unbind"
	echo "$DEV" > /sys/bus/pci/drivers_probe
	faults=$(( $(dmesg | grep -c IO_PAGE_FAULT || true) - faults0 ))
	[ "$faults" -eq 0 ] || fail "$faults IOMMU faults"
	echo "boot $i ok: $(echo "$ms" | cut -d, -f1-3); selftest PASS; client freed on unbind"
	i=$((i + 1))
done
echo "PASS"
