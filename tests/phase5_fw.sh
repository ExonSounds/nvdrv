#!/bin/sh
# Phase 5 (host side) test: kernel firmware loading vs independent Python decode. Read-only. Run as root.
set -eu
DEV=${1:-0000:01:00.0}
HERE=$(cd "$(dirname "$0")/.." && pwd)
DBG=/sys/kernel/debug/nvdrv/$DEV
FWDIR=/lib/firmware/nvidia/ga104/gsp
VER=570.144
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
fail() { echo "FAIL: $*" >&2; exit 1; }

[ -e "$DBG/info" ] || "$HERE/scripts/bind.sh" "$DEV" >/dev/null
grep -q "^gsp image:.*radix3 root" "$DBG/info" || fail "firmware not loaded (see dmesg)"
dmesg | grep -q "fw: gsp $VER:.*(verified)" || fail "radix3 self-verification not reported"

# SEC2 fuse for Booter's ucode id (3) -> version = highest set bit + 1
fuse_reg=$(dd if="$DBG/mmio" bs=4 skip=$(( (0x824140 + 4 * 2) / 4 )) count=1 status=none | od -A n -t u4 | tr -d ' ')
fuse=0; v=$fuse_reg; while [ "$v" -gt 0 ]; do fuse=$((fuse + 1)); v=$((v >> 1)); done

python3 -I "$HERE/tools/fw.py" "$FWDIR" "$VER" "$fuse" > "$T/py.txt"
cat "$DBG/booter_load.bin" > "$T/booter_load.bin"
python3 -I - "$FWDIR/booter_load-$VER.bin" "$fuse" "$T/booter_load.bin" "$HERE/tools" <<'PY'
import sys
sys.path.insert(0, sys.argv[4])
import fw
ref = fw.booter(fw.load(sys.argv[1]), int(sys.argv[2]))["image"]
got = open(sys.argv[3], "rb").read()
assert got == ref, f"booter_load differs ({len(got):#x} vs {len(ref):#x} bytes)"
print(f"booter_load: kernel image == Python (sig {fw.booter(fw.load(sys.argv[1]), int(sys.argv[2]))['sig_index']} patched)")
PY
py_img=$(sed -n 's/.*\.fwimage 0x[0-9a-f]* crc32 \([0-9a-f]*\).*/\1/p' "$T/py.txt")
py_sig=$(sed -n 's/.*fwsignature_ga10x 0x[0-9a-f]* crc32 \([0-9a-f]*\).*/\1/p' "$T/py.txt")
py_bl=$(sed -n 's/^bootloader: .*crc32 \([0-9a-f]*\).*/\1/p' "$T/py.txt")
grep -q "^gsp image:.*crc $py_img" "$DBG/info" || fail "gsp image crc mismatch (python $py_img)"
grep -q "^gsp signature:.*crc $py_sig" "$DBG/info" || fail "gsp signature crc mismatch (python $py_sig)"
dmesg | grep -q "fw: bootloader:.*crc $py_bl" || fail "bootloader crc mismatch (python $py_bl)"
echo "gsp image/signature/bootloader CRCs match ($py_img $py_sig $py_bl), SEC2 fuse $fuse"
echo "PASS"
