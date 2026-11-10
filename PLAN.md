# nvdrv — an NVIDIA GPU driver on top of GSP-RM

Target hardware: **GA104 (RTX 3060 Ti LHR, 10de:2489), Ampere**, PCI 0000:01:00.0
Kernel: Arch 7.2.x, out-of-tree C module (Kbuild), GPL-2.0.

## Ground rules

- **References:** NVIDIA `open-gpu-doc` (register/class headers, MIT), `open-gpu-kernel-modules`
  (MIT/GPL dual) for register offsets and GSP message layouts, public specs (PCI Firmware spec,
  RISC-V), and, **since 2026-11-08, nouveau** (the user lifted the original no-nouveau rule to speed
  things up). nouveau code may be used and built on; keep its copyright/license headers (MIT) on
  anything taken from it, and note the origin in the file.
- **No proprietary code** may be created for or added to the project unless its actual source code is
  provided with it (user rule, 2026-11-08; also in README). NVIDIA's signed firmware loaded at runtime is
  not part of the repository.
- **One firmware version, pinned: `570.144`** (`/lib/firmware/nvidia/ga104/gsp/*-570.144.bin.zst`,
  from linux-firmware). The GSP RPC ABI changes with every release, so every
  struct we define must match that tag of `open-gpu-kernel-modules`.
- **Read-only until it has to write.** Each phase adds the minimum number of register
  writes it needs, and documents each one.
- **Develop over SSH.** This GPU also drives the console (simpledrm on BAR1).
  From Phase 3 on, expect the local display to go dark. Recovery is
  `scripts/unbind.sh` then `echo 1 > /sys/bus/pci/devices/0000:01:00.0/reset` (FLR), or a reboot.
  Since 2026-11-09 unbind shuts GSP-RM down cleanly (no FLR needed); the FLR stays as the fallback.
- Never auto-bind: no `MODULE_DEVICE_TABLE`. The driver binds only via `driver_override`.

## Architecture (where we are going)

```
 userspace ── DRM render node / ioctls ──┐
                                         ▼
 nvdrv.ko: PCI ─ MMIO ─ VBIOS ─ Falcon/RISC-V loader ─ GSP boot ─ RPC queues ─ RM objects
                                                                  │
                                         channels (GPFIFO) ─ copy engine ─ compute ─ (display)
```

On Turing and newer GPUs the kernel driver does not program most engines directly.
It boots **GSP-RM** (NVIDIA's signed resource-manager firmware running on the GPU's
RISC-V core), then sends it RPCs to allocate memory, channels and engines.
Most of the work is the boot chain and the RPC layer.

## Phases

| # | Phase | Deliverable / exit criterion |
|---|-------|------------------------------|
| 0 | **Environment** | headers, build, bind/unbind scripts, SSH workflow. ✅ |
| 1 | **PCI skeleton + chip ID** | Binds to 01:00.0, maps BAR0, decodes `NV_PMC_BOOT_0/42` → "GA104", detects VRAM size, read-only debugfs (`info`, `mmio`, `vbios`). Clean unbind with no display disturbance. ✅ (`tests/phase1.sh`) |
| 2 | **VBIOS parsing** | Walk PCI ROM image chain (PCIR + NVIDIA NPDE ext.), find BIT table, follow the Falcon-data token to the **FWSEC** ucode descriptor. Prototype `tools/vbios.py`, C port `src/vbios.c`, FWSEC image patched for FRTS in memory. ✅ (`tests/phase2.sh`: byte-identical to independent rebuild) |
| 3 | **Falcon / RISC-V engine core** | Reset, IMEM/DMEM upload via DMA, BROM setup, start, halt wait, mailboxes (`src/falcon.c`). ✅ GSP falcon runs signed HS ucode (FWSEC) via DMA through the IOMMU. SEC2 pending. |
| 4 | **Framebuffer layout** | Compute the VRAM carve-out layout (FRTS, WPR2, GSP heap, non-WPR) the way GSP firmware expects it (`GspFwWprMeta`). ✅ `src/fb.c` + `src/wpr.c` (`tests/phase5_meta.sh`: every field matches an independent recomputation) |
| 5 | **Secure boot chain** ✅ | Run FWSEC-FRTS (carves WPR2) ✅, load firmware files (`booter_load`, `bootloader`, `gsp` ELF) ✅, build the radix-3 page table for the GSP image ✅ (`tests/phase5_fw.sh`), WPR meta ✅, run Booter-Load on SEC2 → GSP-RM starts. **✅ GSP_INIT_DONE on hardware (2026-11-08), `tests/phase6_boot.sh`** |
| 6 | **GSP RPC layer** | Shared-memory message/status queues ✅, send/receive + checksums ✅, CPU sequencer incl. CORE_RESUME ✅, pre-boot RPCs ✅ (`tests/phase6_rpc.sh`), doorbell + synchronous calls + event draining ✅, **GET_GSP_STATIC_INFO ✅ (`tests/phase6_static.sh`)**; clean `booter_unload` shutdown ✅ (2026-11-09, FLR as fallback), interrupts ✅. |
| 7 | **RM object model** | GSP_RM_ALLOC / GSP_RM_CONTROL / FREE ✅; root client → device → subdevice ✅; routed control on our subdevice and internal control on GSP's internal subdevice ✅; error paths verified ✅ (`tests/phase7_rm.sh`). |
| 8 | **Memory management** | VRAM carve-out allocator ✅, GPU VA space with host-written GP100 page tables + RM's server-reserved PDEs ✅, VRAM/sysmem mappings ✅. Next: free/unmap, 64K pages. |
| 9 | **Channels + Copy Engine** | GPFIFO channel (host-provided inst/USERD/method buffer, BIND, SCHEDULE), doorbell, push buffer ✅; **CE memcpy sysmem → VRAM with semaphore release: first real GPU work (2026-11-08)** ✅ (debugfs `ce_test`). |
| 10 | **userspace API** | DRM render node and a small uAPI (BO alloc/map, channel create, submit, wait). |
| 11 | **Compute** | Allocate compute class, launch a hand-assembled or NAK-compiled kernel. |
| 12 | **Display** (pulled forward) | DP sink discovery + EDID ✅, SOR assignment + DP link training via RM ✅, display instmem/ctxdma/push buffers done by nvdrv in CPU-RM's place ✅, core + window channels ✅, **modeset with a test pattern scanned out from VRAM, confirmed on the monitor (2026-11-08)** ✅, BAR1 CPU mapping ✅, robust bring-up (sink settle, link verify, detach + retrain) ✅, **DRM/KMS device with fbdev: the Linux text console runs on nvdrv, confirmed on the monitor** ✅ (`DRM=1 tests/phase12_display.sh`), **Hyprland 0.57.1 renders on nvdrv** (Mesa llvmpipe through GBM/EGL, confirmed on the monitor, 2026-11-08) ✅, **all EDID modes incl. DisplayID, mode changes through DRM, 2560x1440@180 Hz measured on the head (180 scanline wraps/s)** ✅. Next: faster frame copies, link-loss recovery, gamma LUT, cursor plane. |

Each phase has a test in `tests/` that runs over SSH and is safe to repeat (`phase3.sh` FLRs the GPU).

## Findings log

- **Phase 1:** BOOT_0 = `0xb74000a1`, BOOT_42 = `0x174a1000` → GA104 rev a1. VRAM: both
  `0x100ce0` (LOCAL_MEMORY_RANGE) and `0x1183a4` (USABLE_FB_SIZE_IN_MB) report 8192 MiB.
- **Phase 2, ROM chain:** x86 (0x0, PCIR) → EFI (0xfe00, PCIR) → FWSEC (0x24600,
  `VN`/`NPDS`, type 0xE0) → FWSEC (0x29c00, NPDE last flag). The walk must accept `VN`/`NPDS`
  (and `0xBB77`/`RGIS`). NPDE at `align16(pcir + pcir_len)`, revision 0x100/0x101: `subimage_len`
  ×512 is the image length, `+0x0a` bit 7 is the last-image flag. *(Corrected: an earlier version
  of this log claimed the EFI image is flagged last. It is not. The first parser simply
  stopped at the unrecognised `VN` signature.)*
- **Phase 2, pointers:** falcon-table pointers are offset by `expansionRomOffset` = (offset of
  the first 0xE0 image) − (size of the x86 base image) = 0x14800 here, which equals the EFI image length.
  BIT header has a byte checksum. Falcon ucode table: `{ver, hdr, entry, count, descVer, descSize}`
  then `{u8 app, u8 target, u32 desc}`. FWSEC_PROD = app 0x85, target 0x07 (DBG = 0x45, chosen
  when fuse `0x82074c` bit 0 = 0).
- **FWSEC_PROD desc v3** @0x41208: hdr `0x04ac0301` (0x2c struct + 3×0x180 RSA-3K sigs),
  IMEM 0xe200 + DMEM 0x800 = stored 0xea00, PKC (signature) at DMEM 0x5a4, ucode_id 9,
  engine mask 0x400. Image at desc + 0x4ac, `align256(stored)` bytes.
- **FWSEC interface:** table at DMEM 0x1c → id 4 = DMEM mapper at 0x560, signature `DMAP`,
  v3, `cmd_in_buffer` 0x7c0/0x40. FRTS = `init_cmd 0x15` + 48-byte `FWSECLIC_FRTS_CMD`.
- **Signature choice:** fuse `0x8241c0 + 4*(ucode_id-1)` → version = highest set bit + 1 = **2** here.
  Index = popcount(sig_versions & (BIT(ver)-1)) = **2**.
- **FB layout (GA104):** usable FB 0x200000000. VGA workspace register not valid, so the default is FB−1 MiB = 0x1fff00000.
  `wpr_end` 0x1fff00000, **FRTS 0x1ffe00000** (1 MiB).
  **Lesson:** MMU_LOCK (0x10000000–0x1effff000) is read only on GA100. GA10x uses a stub.
  Always check `generated/g_*_nvoc.c` for the HAL a chip actually uses before porting an RM function.
- **GSP falcon at rest** (after GFW boot): cpuctl 0x10 (halted), hwcfg2 0x47f7 (RISC-V present,
  scrub done), BCR 0x1 (falcon core selected), WPR2 not up, GFW progress 0xff.

- **FWSEC-FRTS on hardware (2026-11-08): SUCCESS on the first attempt.** Reset → DMA (IOVA
  0xfff00000 through AMD-Vi, no faults) → BROM RSA-3K → halt in ~210 ms. mailbox0 = 0, FRTS err = 0,
  **WPR2 = 0x1ffe00000 – 0x1ffee0000** (HI reads as the base of the last 128K block, so the region is the full
  1 MiB). Afterwards: falcon halted, BCR still on falcon core, console (simpledrm) unaffected,
  unbind/rebind fine, WPR2 persists across rebind, and the re-run guard refuses (`-EBUSY`).
- **FLR (2026-11-08): clears WPR2 and re-runs GFW boot.** Progress `01 → 02 → 09 → ff` in
  ~300 ms, after which the state matches a cold boot (GSP falcon halted, same FB size and VGA defaults,
  FWSEC fuse/sig unchanged). BOOT_0 stays readable throughout. **Development loop: about 0.8 s per
  boot attempt, no reboot needed** (`tests/phase3.sh N` runs 3 cycles in 2.4 s).
- **Races found by binding right after FLR:** (1) probe must wait for GFW boot (now polls up to
  2.05 s like `gpuWaitForGfwBootComplete_TU102`); (2) LOCAL_MEMORY_RANGE reads 0 until devinit
  finishes, so the VRAM size is read only after that wait.

- **Firmware files (570.144, linux-firmware):** the kernel loader decompresses `.zst` itself
  (`CONFIG_FW_LOADER_COMPRESS_ZSTD=y`). Container = `{0x10de, 1, size, hdr_off, data_off, data_size}`.
  The `size` field is *not* the file length (0xf000 vs 0xef78). Bound reads by `data_off + data_size`.
  - **Booter** (`booter_load`/`_unload`): the HS v2 header holds offsets to sig (2×0x180), `patch_loc`,
    `patch_sig` (must be 0), meta `{fuseVer, engineId=1, ucodeId=3}`, `num_sigs`, and the 9-u32
    header `{osCode, osData, numApps=1, appCode, appData}`. IMEM = appCode (0x8900@0x100, va =
    offset, pa 0), DMEM = osData (0x6200@0x8a00), signature DMEM addr = patch_loc − osDataOffset = 0x10.
    **Signature index = numSigs − 1 − SEC2 fuse version** (the reverse order from FWSEC; uses SEC2 fuses at
    0x824140). Fuse slot 3 = 1, so v1 and sig 0 here, matching Booter's own meta fuseVer 1.
  - **bootloader:** RM_RISCV_UCODE_DESC v5 (21 u32), image 0x6000: manifest 0x0+0x800,
    monitor data 0x800+0x1000, monitor code 0x1800+0x2900, BL 0x5000+0x880, fbReserved 0x6000.
  - **gsp ELF:** `.fwversion` = `"570.144\0"` (exact match required), `.fwimage` 0x3c99000,
    signature for GA10x in `.fwsignature_ga10x` (0x1000). Sections for other families exist too.
  - **Radix-3:** 1/1/31/15513 pages `[L0][L1][L2][data]`, u64 bus addresses, 512 per page. Built from
    individually allocated pages + one sg_table mapping (no large contiguous allocation), verified by
    walking the table. All of it (~61 MB) is freed on unbind.

- **GSP FB layout (GA104, 8 GiB):** GSP owns the top **192 MiB**:
  non-WPR heap 0x1f4000000 (1M) · WPR start/meta 0x1f4100000 (1M) · heap 0x1f4200000 (**127 MiB**) ·
  ELF 0x1fc160000 (0x3c99000, 64K-aligned) · boot bin 0x1ffdfa000 (0x6000, 4K-aligned) ·
  FRTS 0x1ffe00000 (1M) · WPR end 0x1fff00000 · VGA ws (1M) · FB end 0x200000000.
  Heap = 22 (LIBOS3 OS) + 8 (base) + 1 (96K × 8 GB) + 96 (2048 channels × 48K) = 127 MiB,
  clamped to [88, 256 − pre(2) − post(63)] MiB. **The cap comes from the pre-scrubbed top 256 MiB**,
  because GA10x has no scrubber ucode (`bScrubberUcodeSupported` is AD10x-only).
- **GspFwWprMeta:** 256 bytes in one coherent 4K sysmem page (RM: `kgspAllocBootArgs_TU102`).
  magic 0xdc3aae21371a60b3, rev 1. For GA104, flags = CLOCK_BOOST (`bBootGspRmWithBoostClocks` is
  true for Turing..Ada), no CrashCat queue (`kgspGetCrashcatSysmemBufferSize` is 0 except GB20x),
  `pmuReservedSize` is Hopper-only, signature size = align256(0x1000). bootloader{Code,Data,Manifest}Offset
  = monitorCode / monitorData / manifest offsets from RM_RISCV_UCODE_DESC.

- **LibOS boot args (GA10x):** a 4K page of `LibosMemoryRegionInitArgument` {id8, pa, size, kind, loc}
  (32 bytes each), in this order: LOGINIT, LOGINTR, LOGRM, LOGMNOC, LOGKRNL (LOGIDX_SIZE = 5; LIBOS2 chips
  ≤ GA100 drop one), then RMARGS → GSP_ARGUMENTS_CACHED page. id8 = name packed big-endian into a u64.
  Log buffers: 64K contiguous sysmem, word 0 = put pointer, words 1..16 = bus address of each page.
  The init arg `pa` is the first page.
- **GSP_ARGUMENTS_CACHED** (72 bytes, natural alignment): queue shm PA = PTE[0], 129 PTEs, cmdq @0x1000,
  statq @0x41000, `bDmemStack` = 1 (registry default), SR args 0, gpuInstance 0, no profiler.
- **RPC queue shm:** 0x81000 bytes = [PTE page: 129 × u64 covering the whole block, itself included]
  [cmdq 256K][statq 256K]. Cmdq `msgqTxHeader` = {ver 0, size 0x40000, msgSize 0x1000, msgCount 63,
  writePtr 0, flags SWAP_RX, rxHdrOff 0x20, entryOff 0x1000}. **GSP-RM writes the statq header itself**
  when it starts; we link to it after boot.
- **Leak check method:** free memory drifts ±50 MB between runs (system noise), so it can't detect a ~1 MB
  leak. Counting the driver's vmap areas in `/proc/vmallocinfo` across 20 bind/unbind cycles can: it is
  exact (223 areas / 2408 KiB before and after).

- **RPC framing (570.144):** element = {authTag[16], aad[16], checkSum, seqNum, elemCount, pad} (48 bytes)
  + `rpc_message_header_v03_00` {header_version 0x03000000, signature "VRPC", length (header + payload),
  function, rpc_result/private = 0xffffffff, sequence, spare} (32 bytes) + payload at +80. Checksum =
  XOR of u64 words over 48 + length (zero-padded to 8), folded hi^lo, so a valid message folds to 0.
  Messages take whole 4K elements and may wrap the ring.
- **GspSystemInfo is 928 bytes**, with padding a hand-written struct would miss (clPdbProperties @112,
  acpiMethodData 676 B @160, sysTimerOffsetNs @848). Derived by compiling NVIDIA's header:
  `tools/gsp_sysinfo_probe.sh`. **SET_REGISTRY must not be empty** (RM treats it as failure); we send the
  Unix per-GPU defaults RMSecBusResetEnable = RMForcePcieConfigSave = 1.
- **GSP boot on hardware, attempt 1:** FWSEC ✓, Booter ✓ (WPR2 grew to 0x1f4100000 = our gspFwWprStart),
  GSP consumed both pre-boot RPCs, then (a) 10 IOMMU faults at bus address 0 and (b) a sequencer request with
  **opcode 8 CORE_RESUME, which GA10x sends on every normal boot**, not only on resume.
  Fixes: **program the sysmem flush buffer** (`NV_PFB_NISO_FLUSH_SYSMEM_ADDR{,_HI}` = page >> 8; reads 0
  after FLR, which explains the faults at address 0), allocate the **notify-op surface**, and implement
  CORE_RESUME (`kgspExecuteSequencerCommand_GA102`: reset GSP into RISC-V, LibOS args again, start SEC2
  with Booter still resident, poll `BSI_SECURE_SCRATCH_14` bit 26, SEC2 mbox0 = 0, FALCON_OS, RISC-V active).
- **Attempt 2: GSP-RM 570.144 initialized.** Sequence observed: RUN_CPU_SEQUENCER (420 ops, ends in
  CORE_RESUME) → GSP_POST_NOCAT_RECORD (component-name table) → 2× UCODE_LIBOS_PRINT → GSP_INIT_DONE
  (result 0). About 1.14 s from trigger to INIT_DONE, 3/3 then 2/2 cycles, 0 IOMMU faults. Afterwards RISC-V
  cpuctl = 0x80 (active), and falcon-side registers read 0xbadf57xx (locked by GSP-RM).
- **Teardown:** a running GSP keeps DMAing into our queues and logs. `.remove` FLRs the GPU
  (`pci_reset_function_locked`) before freeing them. Verified: 0 faults after unbind, vmap areas back to baseline.

- **Synchronous RPCs (2026-11-08):** send, then write 0 to the doorbell `NV_PGSP_QUEUE_HEAD(0)` (0x110c00), then
  drain the status queue until a message with the same `function` arrives (RM matches on function only and allows one
  outstanding call; `gsp_lock` enforces that). Other messages go through the same event handler as during boot. Timeout
  6 s (1.5 × RM default). Poll backoff 20 µs → 1 ms.
- **GET_GSP_STATIC_INFO (65) works on the first attempt:** request = 1656 zero bytes, reply in ~4.5 ms. GspStaticConfigInfo
  is **1656 bytes** (`tools/gsp_static_info_probe.sh`). GA104 reports "NVIDIA GeForce RTX 3060 Ti" / "GA104-A", fb_length
  8 GiB (= LMR), a stable UUID, internal handles client 0xc2000006 / device 0xabcd0080 / subdevice 0xabcd2080, and
  6 contiguous FB regions. **The last region is exactly our 192 MiB GSP carve-out** (0x1f4000000, fully reserved),
  which independently confirms the Phase 4 layout.
  - `engineCaps` uses **NV2080_ENGINE_TYPE** numbering (not RM_ENGINE_TYPE; `kern_bus.c` calls it nv2080EngineCaps):
    GR, CE0–4, NVDEC0, NVENC0, SW, SEC2, OFA.
  - `fwWprLayoutOffset` = 0 is expected: it is set only with `PDB_PROP_GPU_FW_WPR_OFFSET_SET_BY_ACR` (not GA10x).
  - `vbiosSubVendor/Device` = 0x4d50/0x4449 (ASCII "PM"/"ID"). This is not the PCI subsystem ID (19da:1630).
  - Bytes 1240..1259 (`fb_bus_width, fb_ram_type, fbp_mask, l2_cache_size` in the header) read `0, 0x100, 0x11, 0xf`
    and **do not follow the header**: MEMSYS_GET_STATIC_CONFIG (Phase 7) gives GDDR6 = 0x11 and L2 = 3 MiB. 0x11 sits in
    the `fbp_mask` slot, and 3 MiB appears nowhere. RM never reads these fields, so nvdrv no longer decodes them.
- **Leak check, corrected method:** keep the module loaded and cycle through sysfs unbind/probe (`unbind.sh` unloads
  the module, which removes the `[nvdrv]` attribution). `[nvdrv]` areas: 0 unbound, 1 (61 MB radix3) bound,
  0 after 10 boot/unbind cycles, 0 IOMMU faults.

- **RM objects (2026-11-08), first attempt:** exactly as `NV_RM_RPC_ALLOC_SHARE_DEVICE_FWCLIENT` does it:
  - Root: `GSP_RM_ALLOC(hClient=H, hParent=0, hObject=0, NV01_ROOT)` with NV0000_ALLOC_PARAMETERS {hClient H, processID
    KERNEL_PID 0xffffffff} (120 bytes).
  - Device: `(H, H, hDev, 0x80)` with NV0080 {deviceId 0, hClientShare H} (56 bytes).
  - Subdevice: `(H, hDev, hSub, 0x2080)` with {subDeviceId 0}.
  - Handles: client 0xc1d00001 (CPU-RM's client range), device 0xd0000080, subdevice 0xd0002080. Layouts come from
    `tools/rm_struct_probe.sh`.
  - The three allocs + two controls take ~2 ms in total.
- **Which controls GSP-RM serves:** CPU-RM forwards a control verbatim only when it has `RMCTRL_FLAGS_ROUTE_TO_PHYSICAL`
  (0x40, see `g_subdevice_nvoc.c`). Others, such as FB_GET_INFO_V2 (flags 0x118), CPU-RM answers itself from cached
  data, so GSP-RM may not implement them. Controls flagged INTERNAL (0x80) go to **GSP's internal client/subdevice**
  (hInternalClient/hInternalSubdevice from static info). That is how CPU-RM reaches "physical" RM, and it works for us.
  None of these are FINN-serialized (FINN covers only messages with embedded pointers; the list is in `g_finn_rm_api.c`).
  - `GPU_GET_CHIP_DETAILS` (0x208001a4) on our subdevice: pciDevId 0x2489 (= config space), sku "202" (= static info),
    rev a.1 (= BOOT_0).
  - `INTERNAL_MEMSYS_GET_STATIC_CONFIG` (0x20800a1c): GDDR6, L2 3072 KiB, 8 LTC × 4 LTS, 64 KiB compression page,
    ECC off.
- **Error reporting:** a failed **alloc** sets the header `rpc_result` *and* the payload `status` (e.g. 0x19
  INSERT_DUPLICATE_NAME). A failed **control** leaves `rpc_result` at 0 and reports only in payload `status` (0x56
  NOT_SUPPORTED for an unknown cmd, 0x57 OBJECT_NOT_FOUND for a bad handle, 0x1f INVALID_ARGUMENT for a wrong params
  size). Check both. `nv_rpc_call` copies the reply out even on failure.
- **FREE (10)** {hRoot, hObjectParent 0, hObjectOld, status} on the client frees device and subdevice too. Unbind
  frees the client before the FLR. A second client alloc+free in the self test leaves the queues in sync.

- **Display, first image (2026-11-08).** Monitor on DP (display 0x2000, DCB 0, DP-B), EDID preferred 2560x1440@60
  (241.7 MHz). Sequence, all on top of GSP-RM 570.144:
  1. `NV04_DISPLAY_COMMON` (0x73): GET_NUM_HEADS (4) / GET_SUPPORTED (0x7f00) / GET_CONNECT_STATE / GET_TYPE /
     OR_GET_INFO / GET_EDID_V2. All are ROUTE_TO_PHYSICAL.
  2. **Things CPU-RM does that nvdrv must do itself in client mode:**
     - Display **instance memory**: 64 KiB in VRAM (8 KiB hash table of {handle, ctx} + 32-byte ctxdma objects), given
       to GSP with `INTERNAL_DISPLAY_WRITE_INST_MEM` (ADDR_FBMEM, WC). Hash and object format are from
       `disp_inst_mem_0300.c`; ctxdmas are never sent to GSP.
     - **Push buffers**: `INTERNAL_DISPLAY_CHANNEL_PUSHBUFFER` (sysmem, PHYS_PCI_COHERENT, 4 KiB) on the internal
       subdevice, *then* GSP_RM_ALLOC of the channel class (`hObjectBuffer` is a dummy handle; GSP never resolves it).
     - **VRAM**: nvdrv carves from the top of the last FB region GSP left unreserved (region 2, below 0x1f07e0000),
       the same place CPU-RM puts its reserved memory. CPU access goes through the BAR0 PRAMIN window (0x1700/0x700000).
  3. DP: AUX read of DPCD caps (DPCD 1.2, HBR2 x4, enhanced framing), sink D0, `DFP_ASSIGN_SOR` (SOR0),
     `DP_CTRL` SET_LANE_COUNT|SET_LINK_BW|ENHANCED_FRAMING. **RM does the CR/EQ link training itself**: 4 x HBR2 on the first try.
  4. `NVC670_DISPLAY`, core `NVC67D` (user area BAR0 0x680000: PUT/GET, GET_RG_SCAN_LINE @0x220) and window
     `NVC67E` 0 (0x690000). Method header = count<<18 | offset; JUMP = 0x20000000.
  5. Core update A: **window→head ownership and usage bounds in their own update, not interlocked with the window**.
     Doing it in the modeset update makes the window's update fail (nvkms comment: ownership cannot change in the
     update that uses it).
  6. Core update B (interlocked with window 0): SOR_SET_CONTROL (owner head0, protocol DP-B), raster
     (nvkms conversion: syncEnd = sw-1, blankEnd = total-syncStart-1, blankStart = active+total-syncStart-1),
     pixel clock in Hz, viewport, usage bounds, HEAD_SET_DISPLAY_ID. GSP-RM handles the supervisor interrupts.
  7. Window: X8R8G8B8 pitch-linear (pitch/64; pitch vs block linear comes from the ctxdma KIND bit), ctxdma
     ISO 0, identity FMT matrix, composition bypass (since 2026-11-09: normal composition with identity ILUT/OLUT,
     see the hardware cursor entry).
     - Normal blending without an input LUT → window error 0x2d (nvkms always sets an identity ILUT for 8-bit formats).
     - Bypass + explicit CSC/LUT "disable" methods → window error 0x2.
- **Debugging display exceptions without docs:** `NV_PDISP_FE_EXCEPT(chn)` (0x611020 + 12·chn) latches METHOD_OFFSET
  (0x80 = UPDATE) and RESTART_MODE. SKIP (0x1000_0000) means GSP-RM rejected the update. `EXCEPTERR` (+8) holds the
  class error code. Comparing the assembly copy (user area) with the armed copy (+0x800 window, +0x8000 core) shows
  what actually took effect.
- **Scanout from sysmem is rejected** on this dGPU: a window ctxdma with TARGET_NODE PHYSICAL_PCI_COHERENT fails
  with window error 0x1d, so ISO surfaces stay in VRAM.
- **BAR1 (2026-11-08), done the way nouveau does it** (r535 bar.c): under GSP-RM, BAR1 is in virtual mode. GSP builds a
  root → PD2 → PD1 → PD0 chain at `bar1PdeBase` (0x1f28ba000) with one empty 64K page table. nvdrv replaces root
  entry 0 with its own PD2/PD1/PD0 + 64 KiB page tables (GP100 formats: PDE/PTE = addr>>4 | aperture/valid) in its
  carve-out, invalidates via 0xb830a0/a4/b0 (type 7), and ioremap_wc()s BAR1 (8 GiB with resizable BAR). CPU writes
  through BAR1 read back identically via PRAMIN.
- **The monitor has two identities** (direct DP cable): asleep it answers with a fallback EDID/DPCD (1920x1080,
  DPCD `12 14 c4 81 ... 00 00`). A moment after it wakes, and *woken by our first modeset*, it switches to its real
  one (2560x1440, DPCD 1.4 with extended receiver caps, byte 0x00E = 0x83). That kills the trained link (all lanes
  0x00, AUX briefly dead), and the OSD says "No signal". Once the head is active, RM refuses `DP_CTRL` retraining
  (NV_ERR_GENERIC). **`disp_up` therefore:** probes until EDID+DPCD are unchanged over 3 reads (250 ms apart,
  sink set to D0), trains, modesets, checks DPCD 0x202..0x205 one second later, and on link loss **detaches the
  stream** (window ISO ctxdma 0, SOR owner 0, HEAD_SET_DISPLAY_ID 0, one interlocked update) and repeats. In practice
  attempt 1 wakes the monitor and attempt 2 holds.
- **DRM/KMS + console (2026-11-08):** `src/drm.c`, modeled on tiny/cirrus-qemu: GEM-shmem buffers, a shadow-buffered
  primary plane whose damage rectangles are memcpy'd into the VRAM scanout surface through BAR1, software vblank
  timer (`DRM_CRTC_VBLANK_TIMER_FUNCS`), fbdev via `drm_client_setup`. Only the mode the head runs is offered.
  `aperture_remove_conflicting_pci_devices` retires simpledrm. The scanout surface is one fixed 64 MiB VRAM block
  (one ctxdma, mapped through BAR1 once), so re-modesets only re-send methods.
  - On this machine the **desktop session (Hyprland on tty1, via logind) grabs the new card and holds DRM master**,
    so fbdev cannot commit on tty1. The console shows on the text VTs (`chvt 3`). On a hotplug-only status change
    (`detect`), fbdev does not recommit.
  - Unbind with fbcon bound: `drm_dev_unplug` + `drm_atomic_helper_shutdown` before the FLR. The console falls back to
    the dummy device with no warnings, and later BAR1 access is impossible.
- **Hyprland on nvdrv (2026-11-08).** Userspace stack on a KMS-only device: GBM uses Mesa's `dri_gbm` (no
  `nvdrv_gbm.so`), and EGL/GLES 3.2 run on **llvmpipe**. That works with the default glvnd setup even though NVIDIA's
  EGL vendor (`10_nvidia.json`, from the nvidia-utils fallback) is installed. Checked with `tools/egltest.c`, which
  renders into a SCANOUT|RENDERING BO and exports it as a dma-buf. The test's own pitfall: eglChooseConfig ignores
  EGL_NATIVE_VISUAL_ID, so the XR24 config has to be picked by hand. Hyprland must be started in a **fresh session**
  (a VT login): a running instance cannot move its primary renderer to a hotplugged GPU (aquamarine's mgpu
  renderer fails on `eglQueryDevicesEXT`). Hyprland then modesets DP-1 at 2560x1440@60 with three XR24 LINEAR
  buffers and page flips on the vblank timer. Missing gamma/HDR properties are logged and ignored. The user's
  monitor rule asks for @180 (AOC Q27G42XE), which nvdrv does not offer yet.
  - **Pitfall:** after a DRM master (modetest) exits, the CRTC stays disabled and fbdev does not restore it on
    its own; a VT switch does (`chvt`).
  - GSP-RM's OS_ERROR_LOG explains the core exception: `CMDre ... 00002004 00000041 00000004` = method 0x2004
    HEAD_SET_CONTROL_OUTPUT_RESOURCE, data 0x41 (0x49 with -vsync), error 4. The modeset still applies; the field
    combination needs checking against what RM expects for DP (nvkms programs the same bits, so maybe it is
    ordering or a missing companion method).
- **180 Hz (2026-11-08).** The AOC Q27G42XE's EDID: base DTD 2560x1440@60, CTA DTDs @144/120/100, DisplayID type I
  @165/170/**180** (713.96 MHz, htotal 2680, vtotal 1480). The extended caps at DPCD 0x2200 (read when 0x00E bit 7
  is set) say DP 1.4 but **HBR2 max** (0x2201 = 0x14), so 180 Hz runs on HBR2 x4: 17.135 of 17.17 Gbps; the
  isModePossibleSST watermark/hblank checks pass (wm 22, hblank symbols 65). DRM now offers every EDID mode
  (`drm_edid_connector_add_modes`) that `nv_disp_mode_fits` the trained link and the 64 MiB framebuffer. A mode
  change in an atomic commit reprograms the head from `mode_set_nofb` (`nv_disp_set_mode`: in-place re-modeset,
  link check, detach + retrain + modeset on loss). The 60 → 180 switch worked in place, and the link stayed locked.
  - The core exception is fixed: HEAD_SET_CONTROL_OUTPUT_RESOURCE.EXT_PACKET_WIN must be `_NONE` = 0x3F. 0 selects
    window 0 as the infoframe source, which GSP-RM rejected as an invalid argument (OS_ERROR_LOG `CMDre ... 00002004`).
  - **Full-screen frame rate: the PCIe link, not the copy.** After a few seconds of idle, GSP-RM's governor parks
    the GPU in **P8 with the PCIe link at 2.5 GT/s** (Gen1), because CPU traffic through BAR1 does not count as
    load: BAR1 writes then reach 2.9 GB/s (~155 full frames/s through DRM). In **P0 the link runs Gen3 x16**
    (8 GT/s, the root port's maximum) and BAR1 takes **11.2 GB/s** (~760 full 2560x1440 frames/s).
    `NV2080_CTRL_CMD_INTERNAL_PERF_BOOST_SET_2X` on nvdrv's own subdevice (what CPU-RM sends for
    NV2080_CTRL_CMD_PERF_BOOST; params {NvBool flags, u32 duration}, BOOST_TO_MAX = 2, duration in s or
    0xffffffff = until cleared) moves it to P0/Gen3 in ~250 ms. The DRM plane update re-arms a 3 s boost
    (work item, at most 1/s) once more than 64 MB were copied within a second. Cursor blinks never trigger it, a
    static screen drops back to P8 after ~6 s, and full-screen flips hold 180.00 Hz. Debugfs `perf` shows the
    pstate and link and accepts max/clear/<seconds>; `bar1_bench` times full-frame copies. Confirmed in Hyprland at 180 Hz
    ("smooth"): 5969 plane updates / 20.2 GB copied in a ~20 s motion test (debugfs `drm_stats`), back to P8 when idle.
  - A module reload is impossible while a compositor still holds fds of the unplugged DRM device (each open
    file pins the module); stop it first.
- **First GPU work: copy engine (2026-11-08).** Following nouveau's r535 GSP code (`src/fifo.c`):
  - **VA space:** FERMI_VASPACE_A (index GPU_NEW). The host writes the page tables (GP100 format, 4K pages through PD0's
    small-table slot) and passes RM the PD3/PD2/PD1 pages covering the server-reserved 512 MiB at 4 GiB with
    `VASPACE_COPY_SERVER_RESERVED_PDES`.
  - **Instance block:** joined by the host (gp100/gv100_vmm_join: PDB | VER2 | 64K at 0x200, limit 2^49-1 at 0x208,
    subcontext 0 PDB at 0x2a0, valid mask at 0x298).
  - **Channel:** AMPERE_CHANNEL_GPFIFO_A with instanceMem/ramfcMem (VRAM), userdMem (VRAM, 0x200 per chid, USERD index
    fields), mthdbufMem (sysmem, size from CE_GET_FAULT_METHOD_BUFFER_SIZE) and flags 0x00200020 (+ runqueue, USERD
    index/page). Then NVA06F BIND (engine type) and GPFIFO_SCHEDULE(enable), and AMPERE_DMA_COPY_B {version 1,
    engineType} on the channel.
  - **Submission:** GPFIFO entry `{va & ~3, va>>32 | dwords<<10}`, GP_PUT at USERD+0x8c (through BAR1), doorbell
    `(runlist doorbell id << 16) | chid` to BAR0 0xbb0090 (function base 0xb80000 + NV_FUNC_DOORBELL 0x30090). On GA104
    the doorbell id (runlist + 0x008 >> 16) equals the runlist number. `GPFIFO_GET_WORK_SUBMIT_TOKEN` returns
    INVALID_STATE for this privileged channel, so the token is computed as nouveau does.
  - **Pitfalls found:** (1) internalFlags notifier types: UNKNOWN = 0 is a CPU-RM placeholder; NONE = 1, MEMORY = 3.
    nvdrv uses a memory **error notifier**, and RM writes the robust-channel code there (info32) when it recovers a
    channel; it sends no RC event. (2) **CE4 has no PCEs** on this board (`CE_GET_CE_PCE_MASK` = 0; CE2 = 0x10,
    CE3 = 0x20). Using it gives ROBUST_CHANNEL_CE4_ERROR (71) with CE LAUNCHERR INVALID_CONFIG (0x0d): RM then
    submits an empty runlist and disables the channel (chram 0x108/0x000). nvdrv now picks a non-GR CE with PCEs.
- **GR engine context (2026-11-08)** (`gr_ctx_promote`, nouveau r535 gr.c): `INTERNAL_STATIC_KGR_GET_CONTEXT_BUFFERS_INFO`
  (0x20800a32, internal subdevice, [8][26] {size, align}) gives the sizes; GA104: MAIN 1132 KiB (incl. 64 subcontext
  headers), PATCH 20 KiB, BUNDLE_CB 12 KiB, PAGEPOOL 128 KiB, ATTRIBUTE_CB 8.3 MiB (VA aligned to its pow2 size),
  RTV_CB_GLOBAL 512 KiB, FECS_EVENT 64 KiB, PRIV_ACCESS_MAP 512 KiB (unmapped) + an UNRESTRICTED copy (mapped ro).
  Mapped with PTE PRIV (bit 5) / RO (bit 6); initialisable buffers carry their physical address (physAttr 4).
  `GPU_PROMOTE_CTX` (0x2080012b) on nvdrv's subdevice, then AMPERE_B (0xc797) on the channel makes RM build the golden
  context; AMPERE_COMPUTE_B (0xc7c0) next to it. nvdrv uses its single GR channel (chid 2, runlist 0) as the golden
  channel (nouveau uses a throwaway one). debugfs `gr_test`: the compute class releases a semaphore
  (SET_REPORT_SEMAPHORE_A..D at 0x1b00) — passes.
- **nouveau uAPI for Mesa NVK (2026-11-08, `src/uapi.c`, module parameter `nvk=1`).** The DRM driver is named
  "nouveau" 1.4.0 (NVK checks the name, needs >= 1.3.1; 1.4.1 would enable its transfer queue) with a render node,
  syncobjs and: GETPARAM (GRAPH_UNITS from GR_GET_GPC_MASK/TPC_MASK: 5 GPCs, 19 TPCs), NVIF device/info/sclass/
  engine objects (custom ioctl path: variable size), GEM_NEW/INFO/CPU_PREP, VM_INIT/VM_BIND, CHANNEL_ALLOC/FREE,
  EXEC. Per DRM file: own RM client + *externally owned* FERMI_VASPACE_A (flag bit 3) with the host's page
  directory given by `NV0080_CTRL_CMD_DMA_SET_PAGE_DIRECTORY` (nouveau r535 vmm.c external path; no RM-reserved
  range; NVK puts the kernel range at [2^39, 2^40)). VRAM objects: drm_mm pool, zeroed by the CE in physical mode
  (remap const), mmap through BAR1 (mapped on first mmap). GART: shmem pages, SYS_COH|VOL PTEs. EXEC: GPFIFO
  entries `{va, va>>32 | BIT(9) subroutine | dwords<<10 | no_prefetch<<31}` + host SEM_RELEASE (0x5c..0x6c,
  RELEASE_WFI); fences signalled by a 100 µs hrtimer reading the semaphore; error notifier -> fence error.
  - **Pitfall: client channels must be unprivileged** (NVOS04 bit 5 clear, internal PRIVILEGE_USER). With ADMIN,
    RM gives the context the *unrestricted* priv access map, which (like nouveau) is not promoted for client
    channels: NVK's first priv-register write via FECS (macro writing 0x419ba4) faulted at VA 0 + 0x419ba4/32
    (RC 44 GR_FAULT_DURING_CTXSW, MMU fault 0x20000). Found by dumping NVK's push (`dump_push=1`) and decoding it.
  - GR context promoted on a channel's first object (as r535_gr_chan_new); RC_TRIGGERED decoded in gsp.c.
  - Results: vulkaninfo (NVK GA104, Vulkan 1.4), vkmark headless ~17k fps, Zink GLES 3.2 via GBM, kmsgl.
- **Hyprland on NVK/Zink (2026-11-09): runs smoothly.** Problems found on the way, all fixed:
  - Window push buffer: `nv_disp_scanout` reserved 16 bytes for 48 bytes of methods; the overflowed batch was
    half-dropped and the software PUT ran ahead of the hardware, so every wrap waited 2 s (one frame per 2 s).
    Overflowed batches now roll back to where they began.
  - Xwayland deadlock (not nvdrv): Zink made the Vulkan loader load the proprietary nvidia-utils ICD too, whose
    init calls XOpenDisplay(":0") on Xwayland itself while it starts. Session-side fix: `VK_DRIVER_FILES` set to
    the NVK ICD (guarded on `/sys/module/nvdrv/parameters/nvk`).
  - Stutter every 1-2 s: VM_BIND ~30 ms (four uncached PRAMIN reads per level per 4K PTE) under gsp_lock, flips
    queued behind it (78 ms seen). Last page table cached per VM: one walk per 2 MiB.
  - Mouse lag when apps start: GR context setup 59 ms (PRAMIN zero-fill ~20 MB/s) -> CE clears, now 6-8 ms; and
    page flips no longer take gsp_lock at all (evo.win_lock, order gsp_lock -> win_lock): kmsgl holds 60 fps
    while 12 Vulkan clients start in parallel.
  - GPU stays in P8 under a compositor (RM's governor does not raise it); EXEC re-arms a 3 s boost. Not the
    stutter cause, but correct.
  - Teardown with files open after unbind oopsed (BAR0 gone); nvdrv_dev is refcounted and file/object teardown
    after DRM unplug keeps to software state (verified: orphans closing after unbind, clean unload).
- **Zero-copy scanout:** DRM framebuffers on nvdrv VRAM objects flip window 0 to them (ctxdma over the pool,
  SET_OFFSET >> 8, PLANAR_STORAGE pitch >> 6). Pitfall: the modeset leaves the window INTERLOCK_WITH_CORE, so a
  window-only UPDATE stalls until a core update; flips clear the interlock flags.
  - Debug aids: debugfs `chan_status` (USERD/GPFIFO/push buffer), `chan_kick` (doorbell variants), and reading PBDMA
    (0x40000 + id·0x800, id from runlist +0x010), runlist (+0x080..0x098, +0x200 engine status) and channel RAM
    (runlist +0x004 base) registers against the GA100 manuals in open-gpu-doc.
- **Open:** `DP_CONFIG_STREAM` returns NV_ERR_ILLEGAL_ACTION (0x16) before the modeset; the link works with RM's
  defaults. The core channel also latches EXCEPT 0x10000080 arg 0x41 during the modeset update, but the update takes
  effect. The EDID read during one early probe gave 1920x1080; later reads give 2560x1440 (sink waking up?).

## Next steps

Done so far: GSP-RM boot, RM objects, DisplayPort display + KMS/fbdev, GPU VA spaces, channels, copy engine, GR
contexts, nouveau uAPI subset for Mesa NVK (`nvk=1`), Zink GL, zero-copy scanout; Hyprland runs smoothly on the GPU.

**Status (2026-11-09):** every to-do item that needs no display checking is done (per-object fences and async
VM_BIND, 2 MiB pages, GPU recovery, sparse binding, preemption buffers, clean unload), and on the display side
hardware vblank, hardware cursor, gamma/CTM, block-linear scanout, tearing flips, VRR, DPMS and hotplug. Left:
multi-monitor, NVDEC/NVENC (items marked ▶ first). Switching between nvdrv
and NVIDIA's driver: "Switching drivers" under Hardware workflow.

**Long-term goal (user, 2026-11-09): make the whole driver independent of nouveau.** Today nvdrv leans on it in
two ways, both meant as stepping stones:
- *Interface:* with `nvk=1` the DRM device calls itself "nouveau" and implements a subset of nouveau's uAPI, only
  so that Mesa's NVK binds to it unchanged. That collides with a real nouveau (see Platform) and ties nvdrv to
  nouveau's ioctl design and versioning.
- *Code and knowledge:* parts are ported from or modelled on nouveau (r535 GSP paths, BAR1, display/cursor/LUT
  sequences, the uAPI semantics), with nouveau's MIT headers kept.
The end state is nvdrv under its own name with its own uAPI and its own Mesa winsys/backend, and every
nouveau-derived piece either rewritten from primary sources (NVIDIA's open-gpu-kernel-modules / open-gpu-doc and
the firmware's RPC interface) or justified on its own. Steps, when the time comes: design the nvdrv uAPI (GEM, VM
bind, exec, syncobjs, channel/engine queries) from what NVK actually needs; add an nvdrv backend to NVK
(nvkmd) next to nouveau's; keep the nouveau-compatible uAPI as an optional mode during the transition; then retire
it and audit the remaining nouveau-derived code.

To-do list (GA104 / RTX 3060 Ti), roughly by user impact within each group:

### Rendering and memory
- [x] **Interrupts** (`src/intr.c`, 2026-11-09): one MSI vector. Vectors from
      `INTERNAL_INTR_GET_KERNEL_TABLE` (0x20800a5c, internal subdevice; 25 entries on GA104; r570 MC indices GSP 50,
      GR0 84, CE0 15..): GR nonstall = 0, GSP stall = 155. CPU interrupt tree as nouveau (vfn tu102.c): leaf status
      0xb81000+4l (W1C), enable set/clear +0x1200/+0x1400, top +0x1600, arm/unarm +0x1608/+0x1610. Foreign enabled
      sources are masked, not serviced.
      - Fences: EXEC's fence = host semaphore release + membar (MEM_OP_A..D, MEMBAR) + NON_STALL_INTERRUPT
        (nouveau gv100_fence_emit32); the handler signals fences in hardirq; the hrtimer poll stays as a 4 ms
        safety net (100 us without interrupts). vkmark: 17.5k -> 35.9k fps (fence latency was the limit).
      - GSP events: GSP falcon SWGEN0 (IRQSTAT 0x110008 bit 6, IRQSCLR 0x110004, retrigger 0x1103e8) -> work item
        drains the message queue under gsp_lock (`nv_rpc_drain_events`). GSP raises it for *asynchronous* events
        only (replies stay polled, as in RM's _kgspRpcRecvPoll): verified with a deliberate channel fault
        (`nvktest fault`): MMU_FAULT_QUEUED, NOCAT and RC_TRIGGERED arrive with no RPC in flight.
      - GSP health check every 5 s (RISC-V still active), reported once; debugfs `intr` shows counters.
- [x] **Transfer queue** (2026-11-09): CHANNEL_ALLOC with `NOUVEAU_FIFO_ENGINE_CE` creates an unprivileged channel
      on nvdrv's async copy engine (the CE with physical engines, CE2 / runlist 1 on GA104), class list just
      AMPERE_DMA_COPY_B (engine type COPYn, no GR context). The DRM driver reports 1.4.1, so NVK exposes a
      transfer queue family (2 queues, TRANSFER | SPARSE_BINDING) and uses copy channels for uploads. nvktest:
      CE copy 4 KiB VRAM -> GART bit-exact under a syncobj; vkmark and Zink unchanged.
- [x] **VRAM eviction / overcommit** (2026-11-09): a VRAM allocation that does not fit evicts the oldest evictable
      objects (TTM-style overcommit, they stay usable from system memory): `evict_sem` (write) blocks EXEC/VM_BIND/
      CPU faults, waits for every client channel to go idle, the CE copies VRAM -> system pages in physical mode on
      both sides (`nv_ce_copy_to_sg`), every GPU mapping in every client VM is rewritten to the pages (objects now
      track their mappings), CPU mappings are zapped (next fault maps the pages), the VRAM is freed. Scanout objects
      (display reads VRAM only) and dma-buf exports/imports are never evicted. Test `nvktest overcommit 10`: 10 GiB
      on the 7.5 GiB pool, 3 GiB evicted, all contents intact (CPU), GPU writes into an evicted object, 3.1 s.
      Not yet: moving objects back when VRAM frees up; LRU by use (eviction order is allocation order).
- [x] **BAR1 beyond 512 MiB** (2026-11-09): one PD0 page per 512 MiB up to the whole BAR (8 GiB with Resizable BAR,
      16 PD0 pages + 4096 64 KiB-page tables, 1.1 MiB). NVK's host-visible VRAM is no longer capped at 446 MiB.
- [x] **Resizable BAR** (2026-11-09):
      - *Resize at probe* (as amdgpu_device_resize_fb_bar; module parameter `rebar`: -1 largest (default), 0 leave,
        N MiB): before anything uses BAR1, the firmware framebuffer in BAR1 is removed, memory decoding is turned
        off, `pci_resize_resource(pdev, 1, size, BIT(0))` (BAR0 excluded; BAR3 shares the window and is
        reassigned), `pci_assign_unassigned_bus_resources`. Only with allow_hw_writes. Tested both ways on this
        machine (firmware already sets 8 GiB): `rebar=256` -> 256 MiB (BAR moved to 0x830000000; driver uses
        per-object BAR1 mappings), default -> 8 GiB again (moved to 0xa00000000); GSP-RM, display, uAPI, overcommit
        all fine after each.
      - *Use of a large BAR*: the whole VRAM pool is mapped 1:1 into BAR1 (BAR1 offset == VRAM address; 51 ms once),
        so GEM mmap needs no BAR1 space or PTE writes and cannot fail for lack of BAR; GETPARAM VRAM_BAR_SIZE
        reports the pool. NVK then offers DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT over all 7.4 GiB.
- [x] **64 KiB GPU pages + compression** (2026-11-09, uAPI 1.4.2):
      - Client VMs use GP100 dual page tables: per 64 KiB chunk either *big* (one LPTE in the PD0 [0] table, 32 per
        2 MiB; any VRAM mapping with 64 KiB-aligned VA and physical address) or *small* (LPTE 0 = "use the small
        table", 16 SPTEs). Transitions write the SPTEs before an LPTE is invalidated, so the MMU never sees an
        invalid LPTE over stale SPTEs (what 1.4.2 tells NVK: LPTE/SPTE races fixed). `nvktest pages` punches a
        4 KiB hole into a big chunk and refills it from GART: all GPU writes land right. 1 GiB maps: 16384 PTEs
        instead of 262144 (overcommit test 3.1 s -> 1.65 s).
      - Kinds: Turing/Ampere table (nouveau tu102_mmu_kind); invalid kinds (0x07, 0x0f) rejected; compressible
        kinds (0x08..0x0e) kept on big pages, downgraded to their uncompressed kind on small pages / system memory.
        On GA100+ compression needs only the kind (GSP-RM keeps the backing store for all VRAM). Compressed objects
        are never evicted (a physical copy would move raw compressed bytes).
      - NVK compresses dedicated unshared optimal-tiling images (64 KiB / 2 MiB aligned); everything else it binds
        at 4 KiB alignment (its nouveau backend's choice), so most mappings stay small. `tools/gltest.c` (Zink
        off-screen FBO, gradient + depth test, full readback): exact with a compressed, big-page color target.
- [x] **2 MiB GPU pages** (2026-11-09): a 2 MiB region of a client VM covered by one VRAM mapping at a 2 MiB-aligned
      VA and physical address is a single PTE in the PD0 entry ([0] = PTE with VALID, [1] = 0).
      - Split (partial unmap, eviction, different mapping): the small table is hooked into [1] and filled first
        while the huge PTE still applies, then [0] becomes a zeroed big table; the region's 64 KiB chunks are
        then re-upgraded. Tables of a region are kept (`vm.huge` xarray) while it is huge and reused on split.
      - Merge: VM_BIND coalesces adjacent mappings of the same object range and kind, so remapping a hole makes
        the region huge again.
      - VRAM objects of >= 2 MiB are placed 2 MiB aligned when the pool allows (fallback: requested alignment).
      - `nvktest huge`: 3 x 2 MiB, split by a 4 KiB hole, merged again; GPU writes correct in each state.
        Overcommit (10 GiB) maps 5127 huge pages, eviction splits them correctly. NVK's small allocations and
        4 KiB-aligned binds still get small pages (VA placement is NVK's).
- [x] **Z-cull** (2026-11-09): `DRM_NOUVEAU_GET_ZCULL_INFO` answered from `NV2080_CTRL_CMD_GR_GET_ZCULL_INFO`
      (0x20801206, GSP's internal subdevice) plus the GRAPHICS_ZCULL context-buffer size/alignment (as nouveau
      r570). GA104: align 304x32, aliquots 4864/5120, region x160, headers 224/1344, 16 subregions 304x64, ctxsw
      821760/4096. NVK manages Z-cull storage itself (nothing bound per channel, as nouveau). Verified by dumping
      NVK's pushes: SET_ZCULL = 1, region/storage/aliquots/bounds programmed (region width 1216 = 1024 rounded to
      304); `tools/gltest.c` depth scene (64 rectangles, random depth order) exact with Z-cull on.
- [x] **Per-object fences / async VM_BIND** (2026-11-09):
      - VM_BIND unmap no longer waits for the client's GPU work: once PTEs are cleared and the TLB is flushed the
        GPU cannot reach the memory (Vulkan already requires the app to be done; a buggy app faults its own
        channel). Unmapped objects are released only after the flush, so their memory is never reused before it.
        0.06 ms while the GPU is busy (was: up to the 10 s idle wait).
      - GEM_CPU_PREP waits per object: its reservation fences plus the work of every client that maps it.
      - EXEC adds its fence to shared objects' (dma-buf exported/imported) reservations, BOOKKEEP usage, as nouveau
        does through drm_gpuvm.
      - DRM_NOUVEAU_VM_BIND_RUN_ASYNC: ops run in an ordered per-client worker after their wait syncobjs; the signal
        syncobjs get a fence the worker signals. Synchronous binds drain the queue first; file close drains it.
      - `nvktest sync`: stalled channel (host semaphore acquire released by the CPU), instant unmap, CPU_PREP
        busy/idle across two clients with a dma-buf, async bind returning at once and taking effect after the GPU
        signals its wait.
- [x] **Sparse-binding semantics** (2026-11-09), as nouveau's uvmm:
      - Sparse regions (VM_BIND MAP with SPARSE, no object): holes read 0 and drop writes (PTE VALID_FALSE + VOL);
        object mappings inside work; a plain unmap inside reverts the range to sparse; a sparse unmap removes the
        region and every mapping in it.
      - Rules: regions never overlap (-ENOSPC); an object mapping is entirely inside one region or outside all of
        them (-ENOSPC); a sparse unmap names exactly one region (-ENOENT).
      - A region created over existing mappings keeps them (it was overwriting their PTEs before).
      - Fully sparse 2 MiB blocks are one PD0 entry (VALID_FALSE + VOL_BIG, nouveau gp100_vmm_pd0_sparse), no page
        tables: a 16 GiB region takes 79 ms and no VRAM (was ~4 M PTE writes and 2 tables per 2 MiB). Such an entry
        splits into tables when an object is mapped into it, and becomes invalid when nothing covers it any more.
      - Residency is the MMU's job (sparse PTEs); NVK exposes sparseBinding and sparseResidency*.
      - `nvktest sparse` (17 checks): CE reads/writes of sparse pages, map/unmap inside, the rules, region over an
        existing mapping, 16 GiB create/remove, fault after removal.
- [x] **Preemption buffers** (2026-11-09):
      - Without them GSP-RM leaves every GR channel at WFI: a long dispatch or draw holds the engine and every other
        app's GPU work waits for all of it (`tools/preempttest.c`, 100 waves of ~15 ms workgroups: a trivial
        compute job from a second process waited 1496 ms).
      - nvdrv now does what NVIDIA's user-mode driver does: per GR channel it allocates the PREEMPT (preemption
        context image, 1.7 MB), SPILL (1.8 MB), PAGEPOOL (128 KB), BETACB (5.1 MB) and RTV (520 KB) buffers (sizes from
        the KGR context buffers info), maps them privileged into the channel's VM and binds them with
        `NV2080_CTRL_CMD_GR_CTXSW_PREEMPTION_BIND` (graphics GFXP, compute CTA). The global GFXP_CTRL_BLK is promoted
        with the golden context (CPU-RM promote3d). debugfs `preempt` shows the sizes and each channel's modes.
      - Compute result: the same test gives the probe a 7.5 ms median / 15 ms max wait. Workgroups that themselves run
        for a long time are not interrupted: CILP is accepted by RM but behaves like CTA (no SM trap handler from
        NVK), so CTA is the default.
      - Module parameters `preempt` (bind on/off), `preempt_gfx` (0 WFI, 1 GFXP), `preempt_compute` (0 WFI, 1 CTA, 2 CILP).
      - Graphics result (`preempttest <ms> <dir> gfx`: one 2048x2048 draw with a long fragment shader through Zink):
        under WFI the probe gets in once, after the draw (119 ms); with GFXP it gets in 5 times during one 330 ms draw
        (median 51 ms, max 78 ms). The remaining latency is the scheduler's timeslice, not the draw.
      - Cost: ~9.3 MB of VRAM per GR channel (freed with it); vkmark unchanged (33.6k).
- [x] **GPU recovery** (2026-11-09):
      - Faults: RM recovers (RCs) the faulting channel itself; its fences fail with -EIO and EXEC returns -ENODEV
        (NVK: VK_ERROR_DEVICE_LOST). Other channels and apps keep running.
      - Hangs: watchdog (`job_timeout_ms`, default 10000 like nouveau's drm_sched timeout, 0 = off). A channel whose
        fence semaphore has not moved for that long while work is pending is stopped through RM
        (`NVA06F_CTRL_CMD_STOP_CHANNEL`, bImmediate; the params are exactly 1 byte, 4 gave INVALID_ARGUMENT) before
        its fences fail with -ETIMEDOUT, so the GPU never writes memory the app frees on device lost.
      - nvdrv's own copy channel: an RC or a timeout in a clear/eviction rebuilds it in place (old VAs unmapped,
        RM free, re-created on the same chid); that one operation fails, later ones work.
      - GSP death (RPC wait or the 5 s health check sees the RISC-V stopped): device lost for everyone at once;
        every pending fence fails with -ENODEV, and EXEC, CHANNEL_ALLOC, GEM_NEW and VM_BIND return -ENODEV.
        After a GSP death the in-place reset below brings the GPU back (before 2026-11-10: a module reload).
      - Tests: `nvktest hang` (job_timeout_ms=2000: another channel runs meanwhile, the hung one is stopped after
        2.2 s, its write never lands, EXEC -ENODEV, a new channel works), debugfs `ce_fault` (faulting kernel copy,
        rebuild, clear works), `nvktest lost` with debugfs `gpu_lost` (fence fails at once, 4 ioctls -ENODEV).
- [x] **In-place GPU reset** (2026-11-10, `src/reset.c`): when GSP-RM dies the GPU comes back on the same bind,
      without a reload.
      - Detection: the 5 s health check (RISC-V ACTIVE_STAT), an RPC that finds GSP dead, or the *vblank
        watchdog* (`src/drm.c`). Every vblank interrupt re-arms a 50 ms timer. If it fires while vblanks are on
        and GSP's RISC-V is stopped, software vblanks release DRM's vblank waiters and the reset starts. While an
        FLR is still running, BAR0 reads are all ones, so the watchdog looks again every 100 ms. Before the
        watchdog, the fbdev damage worker hit DRM's 1 s vblank-wait WARNs while the 5 s check had not yet fired.
      - Sequence: programs get the device lost → `nv_drm_reset` holds **every DRM modeset lock** for the whole
        reset. Pending vblank events are sent first, then the state is duplicated and everything disabled. No
        commit can interleave: the first version used the suspend/resume helpers, and vkmark's file release
        committed in the middle of them, which oopsed in `drm_atomic_helper_commit_cleanup_done`. nvdrv then
        forgets the dead RM's objects: display channels, kernel channels, VM and GR ctxbufs, with pool memory
        freed. A generation counter (`rm_gen`) makes stale frees from programs' files skip RM. Then FLR → cold
        GSP boot (FWSEC-FRTS, Booter-Load, INIT_DONE, new WPR meta) → BAR1 tables rewritten from the tracked
        mappings (VRAM after an FLR reads 0xBAD0ACxx until GSP boots: treated as lost) → kernel channels →
        display object → the duplicated state is committed (full DP bring-up).
      - The timestamp hook returns no timestamp while GSP-RM is down. `drm_crtc_vblank_off` clears the hwmode
        clock but not the frame duration, so a commit already in flight would otherwise hit the helper's
        "uninitialized mode" WARN through fence deadlines.
      - Kernel VMM page tables, channel instance blocks and GR context buffers now come from the VRAM pool
        (freeable) instead of the carve-out bump allocator. That also ends the 4 KiB carve-out leak per channel.
      - At most 3 resets within 10 minutes (`auto_reset=0` turns it off); debugfs `gpu_reset` resets on demand.
      - Verified: a sysfs FLR during vkmark (vkmark gets ErrorDeviceLost) is noticed by the watchdog about 150 ms
        later, and the GPU is back after 4.1 s. 2560x1440@180 is up again with no WARN, oops or IOMMU fault. Two
        deaths in a row both recover. Afterwards vkmark gives 32-33k, nvktest PASS, pm_test 2 (suspend + FLR +
        resume) OK, and DPMS off/on OK. Unbind after resets is clean (no pool leak warning).

### Display
- [x] **Hotplug (HPD) and DP IRQ** (2026-11-09): subscribed like nouveau r535_gsp_device_event_ctor:
      NV01_EVENT_KERNEL_CALLBACK_EX (0x7e) objects under the subdevice with notifyIndex CLIENT_RM | HOTPLUG (1) /
      DP_IRQ (7), NV2080_CTRL_CMD_EVENT_SET_NOTIFICATION REPEAT; GSP-RM sends POST_EVENT (0x1003, data at 29:
      plug/unplug display masks, or the displayId). A work item refreshes connect state and EDID without the full
      probe, retrains on a DP IRQ whose link status is bad, and raises a DRM hotplug event; the connector has a
      real `detect`. After a replug the link is untrained: modes are judged by the sink's maximum link, and
      power-on does the full bring-up. Verified with the monitor's power button: off → display off, on → back at
      2560x1440@180 in ~2.2 s (the AOC reports DP-B + TMDS-B unplugged, DP-B plugged).
- [ ] **Multiple monitors**: the other heads/SORs, HDMI (TMDS/FRL).
- [ ] **Robust mode changes** with the AOC monitor (identity switching), DP_CONFIG_STREAM, core exceptions.
- [x] **Hardware cursor plane** (2026-11-09): DRM cursor plane (ARGB8888 premultiplied, 32..256 square), image
      through the core channel (HEAD_SET_CONTROL_CURSOR/COMPOSITION/CONTEXT_DMA/OFFSET_CURSOR; VRAM pool objects
      zero-copy through a pool ctxdma on the core channel, others copied into two 256x256 slots), position through
      the cursor immediate PIO channel `NVC67A` (allocated as nouveau r535_curs_init: CHANNEL_PUSHBUFFER valid=0,
      then NV50VAIO_CHANNELPIO params; user area 0x6d8000 + 0x1000·head; SET_CURSOR_HOT_SPOT_POINT_OUT + UPDATE).
      Verified on screen (`tools/curstest`). What it took (each missing piece is its own INVALID_STATE code):
      - The window must use **normal composition**, not bypass: with bypass, enabling the cursor fails (core 0x43).
      - Composition needs an **identity input LUT** on the window (window 0x2d without; nouveau wndwc57e_ilut:
        VSS header + 1025 FP16 entries, DIRECT10) **and** an **identity output LUT + OCSC0** on the head (window
        0x2e without; OLUT alone fails with core 0x41: nvkms EvoSetOutputLutC5 enables OCSC0 after the OLUT, "as
        required by the specification"; bypass means OLUT and OCSC0 off, nvSetupOutputLUT5).
      - Bypass ↔ composition must switch in **one** window-interlocked core update (each half alone is invalid).
        The modeset does it; module param `compose` (default 1) keeps the old bypass path.
      - A disabled cursor still needs FORMAT = A8R8G8B8 (core 0x2 on HEAD_SET_CONTROL_CURSOR otherwise).
      - Interlock flags are sticky method state: the modeset leaves the core interlocked with window 0, so every
        later core update sets both flags explicitly (otherwise the core waits forever in WAIT_FOR_UPD).
      - DRM's damage helper reports full damage on plain moves: only a new fb, reappearing cursor or explicit
        damage clips re-upload the image; moves are one PIO write.
      - The old "bypass + CSC/LUT disables → window error 0x2" was TMO_CONTROL = 0 (invalid argument); not sent.
- [ ] **Overlay planes** (direct scanout of fullscreen apps).
- [x] **Adaptive sync (VRR / FreeSync)** (2026-11-09), nvkms nvEnableVrr + SetVrrActive for DP Adaptive-Sync:
      connector `vrr_capable` = DPCD 0x007 bit 6 (MSA_TIMING_PAR_IGNORED) + an EDID range (AOC Q27G42XE:
      48–180 Hz); CRTC VRR_ENABLED → DPCD 0x107 bit 7 (MSA_TIMING_PAR_IGNORE_EN, once), VRR_DISPLAY_INFO
      (0x73012c), then a core update with HEAD_SET_STALL_LOCK (enable, ONE_SHOT, line-lock unstall) and
      HEAD_SET_DISPLAY_RATE (ONE_SHOT, MIN_REFRESH_INTERVAL = 1/min refresh) with ARM_LIGHTWEIGHT_SUPERVISOR
      (0x73012f) armed around it; off: CONTINUOUS + CLEAR_ELV_BLOCK (0x73012e). Flips release the stall with
      UPDATE RELEASE_ELV; cursor moves too while VRR is on. VRR is switched off around modesets.
      `KMSGL_VRR=1 KMSGL_FPS=n tools/kmsgl`: 100 fps → flips 9.99 ms apart (stddev 0.14; 2.2 without VRR,
      5.56/11.1 ms alternating), 72 fps → 13.9 ms; the monitor OSD shows 72 Hz.
- [x] **Real vblank interrupts and timestamps** (2026-11-09): GSP-RM's kernel interrupt table gives the CPU the
      DISP stall vector (engine 2, vector 154 on GA104); head timing as kdispServiceLowLatencyIntrs_KERNEL /
      nouveau tu102_disp_intr: DISPATCH 0x611ec0 bit h → RM_INTR_STAT 0x611c00+4h bit 1 (LAST_DATA) → ack
      EVT_STAT 0x611800+4h, enable RM_INTR_EN 0x611d80+4h bit 1 in enable/disable_vblank. Timestamps from the
      raster line counter (0x616330+0x800h) and the ARMed head timing (0x68a064/6c/70+0x400h), nouveau's
      scanoutpos conversion. `tools/vblanktest`: 180.001 Hz, interval jitter ±0.5 µs; kmsgl flips at 179.8 of
      180 Hz. The software timer stays as the fallback without interrupts.
- [x] **Async (tearing) page flips** (2026-11-09): `async_page_flip`; the primary plane's atomic_async_check accepts
      zero-copy VRAM framebuffers; a flip with `async_flip` sets window SET_PRESENT_CONTROL BEGIN_MODE IMMEDIATE
      (sticky: every flip writes it), sends its event at once, and the commit tail waits for flip_done instead of
      a vblank (else two flips per frame). kmsgl: vsync 180.0 fps, async 7,300–9,800 fps; `KMSGL_ALT=1`
      (alternating red/blue frames) shows even purple with vsync and bands with tearing, verified on screen.
      Note: the window's armed copy (+0x800) changes at UPDATE for both modes, so it cannot show when a flip
      reaches the screen.
- [x] **Gamma LUT and CTM** (2026-11-09, needs `compose=1`): DRM GAMMA_LUT (1024, any size resampled; the legacy
      gamma ioctl goes through it) → the head OLUT, written into the slot not on screen (two slots) and switched
      with OCSC0 in one core update at vblank; CTM → OCSC0 (before the OLUT, DRM's order), S31.32 sign-magnitude
      → S5.16 as nvCscCoefConvertS514. Both survive modesets (the modeset programs the current state).
      `tools/colortest`: red tint / red-blue swap / warm legacy gamma, verified on screen and in the ARMed state.
- [ ] **Colour**: 10-bit formats, HDR metadata; DSC.
- [x] **Block-linear scanout modifiers** (2026-11-09): the primary plane offers
      DRM_FORMAT_MOD_NVIDIA_BLOCK_LINEAR_2D(0, 1, 2, 0x06, 0..5) (nouveau wndwc57e_modifiers, what NVK negotiates)
      besides LINEAR, for VRAM framebuffers (copied ones stay linear). Scanout through a second VRAM-pool ctxdma
      with the KIND bit (bit 20, nouveau gv100_dmaobj_new) and SET_STORAGE = log2 block height; the pitch stays
      in 64-byte units. `KMSGL_BL=1 tools/kmsgl`: GBM picks 32-GOB blocks, picture correct on screen.
- [x] **Audio over DisplayPort** (2026-11-09): the connector's ELD (DRM builds it from the EDID's CTA audio block)
      goes to NV0073_CTRL_CMD_DFP_SET_ELD_AUDIO_CAPS (0x731144; nouveau r535_sor_hda_eld) with PD + ELDV and the
      head when the display comes on; cleared when it goes off. The buffer is padded to its full 96 bytes (the
      codec reads all of it: "invalid ELD data byte 33" otherwise), and the first set after load is preceded by a
      clear (else the HDA side keeps eld_valid 0). snd_hda_intel on 01:00.1 then shows monitor Q27G42XE, LPCM
      32/44.1/48 kHz, and PipeWire offers the HDMI/DP sink. DFP_SET_AUDIO_ENABLE (0x731150) before the ELD,
      DP_SET_AUDIO_MUTESTREAM (0x731359) + disable when off, max sample rate from the SADs (nvkms
      nvHdmiDpEnableDisableAudio).
      - **Text jittered sideways while audio played** at 2560x1440@180. Not bandwidth (no underflow, NVIDIA's
        driver is steady in the same mode, ruling out the monitor). Cause: without
        **NV0073_CTRL_CMD_DP_SET_MANUAL_DISPLAYPORT** (0x731365, DPLib's first call, also nouveau r535) RM
        programs the DP stream itself and refuses DP_CONFIG_STREAM (NV_ERR_ILLEGAL_ACTION, the long-standing
        "refused, continuing with RM defaults"): its audio hblank symbols stayed at 313 (the 60 Hz bring-up mode)
        where 180 Hz has room for 65, so audio packets overran the blanking. In library mode nvdrv's values are
        set. Also: the sink's MSA_TIMING_PAR_IGNORE_EN (DPCD 0x107 bit 7) is now cleared at link-up and when VRR
        goes off (it stayed set after a VRR session); debugfs dp_status shows it.
- [x] **Display power (DPMS)** (2026-11-09): CRTC ACTIVE=0 (not a mode change) turns VRR off, detaches the head
      and puts the sink in D3 (DPCD 0x600 = 2): the monitor shows "no signal" and goes to standby. ACTIVE=1: D0,
      link retrain, mode set with the usual retries. `tools/dpmstest`: off in ~10 ms; on in 0.26 s right away,
      ~3.5 s after the AOC entered real standby (retrain fails until the 4th attempt's full bring-up re-probes the
      sink). Not the driver: AUX answers at once and the first retrain succeeds, then the AOC drops off AUX
      during the first modeset while it re-identifies (as after mode changes); waiting for AUX first did not help.
- [ ] fbdev restore after the last DRM master closes; mode changes through a real per-commit modeset path.

### Power, thermals, stability
- [x] **Sensors (public interfaces only)** (`src/sensors.c`, 2026-11-09): GPU temperature through hwmon
      (`sensors` -> `nvdrv-pci-0100`, `temp1_input`) and debugfs `sensors`: temperature, P-state + PCIe link,
      utilisation (graphics/memory/NVENC/NVDEC, `PERF_GET_GPUMON_PERFMON_UTIL_SAMPLES_V2`), per-perf-level clocks
      (`PERF_GET_LEVEL_INFO_V2`: 5 levels, domain 0x10 = memory clock up to 7001 MHz, domain 0x1 ~420 MHz).
      Temperature register 0x020460 as nouveau reads it; on GA104 bits 31:30 mark it (nouveau's Pascal valid bit 29
      is clear), checked against load: 35 C idle -> 54 C under vkmark -> falling after.
- [x] **Fan speed** (2026-11-09): hwmon `fan1_input` ("GPU fan"). The VBIOS DCB GPIO table names the FAN_SENSE
      (0x3d) line (GA104: line 13; FAN PWM is line 16); its input is bit 14 of 0x021200 + 4·line (nouveau ga102
      gpio), timed over four level changes = one rotation (nouveau therm/fan.c). Passive; GSP-RM keeps driving the
      fan. Checked under load: ~1290 RPM up to ~50 C, then rising to 1800 RPM at 58 C.
- [x] **Second fan** (2026-11-09): hwmon `fan2_input`. DCB GPIO function 0xd2 (not in the public DCB 4.x spec)
      is the second fan's tach on this board (line 7): found by sampling every GPIO input passively; it toggles
      at fan rates next to line 13 and follows the same curve independently under load (1270 -> 1810 RPM vs
      1280 -> 1800). Both fans share the PWM (line 16). Lines 0 (function 0x81) and 22 (0xd4) toggle at PWM rates.
- [ ] **Board power** - experiments (debugfs `i2c`, read-only):
      - **FINN-encoded `NV402C_CTRL_CMD_I2C_TRANSACTION` works** (`src/finn.c`, port of NVIDIA's MIT bit pump;
        RMAPI_RPC_FLAGS_SERIALIZED). Pitfall: the SMBUS_QUICK record is {bWrite, warFlags}; missing warFlags gave
        NV_ERR_LIB_RM_VERSION_MISMATCH. With SPEED_MODE_DEFAULT every call is INVALID_ARGUMENT; at 100 kHz port 1
        carries real traffic (PING/quick/byte at the reserved 0x7f -> NV_ERR_I2C_ERROR = NACK).
      - **The VRM's bus (DCB logical port 2) is closed to clients**: INVALID_ARGUMENT at every speed/mode before
        any bus traffic; the VRM does not answer on port 1. Treated as a deliberate RM protection of the voltage
        regulator bus; not worked around. So no board power through public interfaces on this card.
      - Older open SDK releases (515/525/535) stub the thermal/fan/clk/pmgr controls too: no published route.
      - RM-mediated I2C: `NV40_I2C` (0x402c) allocates; `I2C_TABLE_GET_DEV_INFO` lists one board device (type 0x47,
        not in the DCB 4.x type list but in its power-controller range: the VRM, addr 0x25, port 2) and no INA
        power sensor. `NV2080_CTRL_CMD_I2C_READ_REG` "succeeds" with zeros even at the reserved address 0x7f, so
        in GSP-RM it does not reach the bus. The working path would be `NV402C_CTRL_CMD_I2C_TRANSACTION`, which
        needs FINN serialisation (public, in the open RM's generated code); then PMBus reads (READ_VOUT/IOUT/POUT)
        of the VRM, if it is PMBus at all.
      - The BIT 'P' power-sensor pointer leads to a version 0x68 table (nouveau only knows 0x10/0x20, Maxwell/
        Pascal); Ampere's layout is not public, so it is not decoded.
      - No blind I2C address scans: a register read writes a pointer byte first, not harmless on a VRM.
- [ ] **Live graphics clock**: no public description of GA104 PLL or clock-counter registers (nouveau's desktop
      clock code ends at Kepler; open RM and open-gpu-doc have none). Per-level clocks are reported instead.
- [x] **P-state governor**: GSP-RM's governor works by itself (P8 -> P0 within 2.5 s at ~90 % graphics
      utilisation, back to P8 at idle; a desktop's light load correctly stays at P8). The EXEC boost is now off
      (module parameter `exec_boost=1` re-enables it).
- [ ] ▶ **System suspend/resume** (S3; `src/pm.c`, 2026-11-10). Works through suspend → FLR → resume (debugfs
      `pm_test`: 1 = plain cycle, 2 = FLR in between, 3 = FLR without DRM, then bring-up); a real S3 cycle is
      the remaining check. Hibernation (freeze/poweroff) is still refused with -EBUSY.
      - Sequence (r570 kernel RM + nouveau r535/r570, no closed code): DRM suspend (heads off) → no new
        submissions, GPU idle → display channels freed, each after `NV5070_CTRL_CMD_SET_RMFREE_FLAGS`
        PRESERVE_HW (nvkms' suspend path) → `FIFO_TOGGLE_ACTIVE_CHANNEL_SCHEDULING` off (preempts every
        channel: GR contexts and RAMFCs are written back) → new kernel copy channel (channels made after the
        toggle still run) copies the nvdrv carve-out minus unused pool to sysmem → that copy channel freed →
        FBSR sysmem buffer to RM (`ALLOC_MEMORY` RPC with a page list, continuation records, then
        `FBSR_INIT`) → GSP's BAR1 root entry 0 back → `UNLOADING_GUEST_DRIVER` {PM 1, level 3} → Booter-Unload
        with the `GspFwSRMeta` (SR data radix3, 190 MiB) mailboxes. ~0.55 s.
      - Resume: GFW boot wait → Booter-Load with the SR meta, rmargs `srInitArguments` {old level 3, flags
        PRESERVING | PM_TRANSITION, bInPMTransition 1}, no FRTS → carve-out restored (PRAMIN outside the pool,
        then BAR1 root entry 0 again, pool through BAR1) → scheduling on → new kernel copy channel →
        display object and core channel → DRM resume (full DP bring-up). ~1.7 s, display back on attempt 1.
      - What it took: (1) every RM display-channel free used to run into RM's 4 s timeout (an ASSERT NOCAT
        each); with PRESERVE_HW it is 15 ms, and the GSP PM unload no longer hangs behind it. (2) the VRAM
        snapshot must be taken after the preemption: a resident GR context is only written back when
        preempted (stale: RC 13 GR exception after resume), and the copy channel's own RAMFC is in the
        snapshot from before its copies (restored, it replays them into the freed buffers: IOMMU
        IO_PAGE_FAULTs). (3) the carve-out restore must come after GSP-RM's boot (before it: RC 45 on a GR
        channel). (4) `DP_SET_MANUAL_DISPLAYPORT` again after resume; display-common object and hotplug
        events are kept (re-allocated after a PM resume, the events cannot be armed: no event list).
      - Verified: vkmark (NVK) keeps running across suspend → FLR → resume (32.5k, no device loss, no RC, no
        IOMMU faults); DPMS and DP IRQ events after resume; 2560x1440 @ 180 Hz back.
      - A failed suspend before the unload recovers (display back), but RM refuses to re-enable scheduling
        outside a PM resume (INVALID_STATE): client channels stay stopped until a reload.
- [ ] Runtime power-down when idle (GC6 / D3cold): not available on a desktop board with an active display.
- [x] **Clean unload** instead of FLR (2026-11-09, `gsp_unload_locked` in `src/gsp.c`, as kgspUnloadRm_IMPL +
      kgspTeardown_TU102): after our RM client is freed, `UNLOADING_GUEST_DRIVER` (RPC 47, {bInPMTransition 0,
      bGc6Entering 0, newLevel 0}, GSP replies) → GSP MAILBOX0 = 0x80000000 (LibOS PROCESSOR_SUSPENDED) →
      FWSEC-SB on the GSP falcon (DMEM-mapper command 0x19 with only a READ_VBIOS_DESC; checks: GFW PLM lowered,
      GFW progress 0xff, scratch 0x15 bits 15:0 = 0) → Booter-Unload on SEC2 (mailboxes 0xff/0xff in, 0 out)
      → WPR2 must be down. ~215 ms. Any failure (or `clean_unload=0`, or GSP dead) falls back to the FLR;
      bus mastering is cleared either way before the shared memory is freed.
      - Verified: 8 unload/boot cycles without FLR; nvdrv → NVIDIA 615.78 (its own, newer GSP firmware) →
        nvdrv, both hand-offs without FLR or reboot; FLR fallback with `clean_unload=0`; regression all PASS
        (nvktest modes, gltest, kmsgl 57 fps, vkmark 32.2k).
      - GSP-RM posts one NOCAT record per boot (engine, "Display Subsystem"/"Display Engine", error 0x27) and
        one at unload (error 0x28), with or without FLR. NOCAT records are now decoded into one log line and
        counted in debugfs `info`.
      - Seen once in 8 no-FLR boots, not reproduced: ~60 NOCAT ASSERT records ("NV_PGC6_AON_SECURE_SCRATCH_
        GROUP_05_0_GFW_BOOT_PROGRESS...", every 40 ms for 2.4 s) inside GSP-RM's closed display code; the boot
        still finished and passed. GFW_BOOT reads 0x3ff and the PLM 0x8b8f after both FLR and clean unload.
- [ ] Fan control / power limits (read-only first; writing them is an explicit user decision).

### Hardware blocks not used yet
- [ ] **NVDEC** (H.264/HEVC/VP9/AV1 decode) and **NVENC** channels for Vulkan Video / VA-API paths.
- [ ] OFA (optical flow).

### Platform
- [ ] Firmware versions other than 570.144 (the RPC ABI changes per release).
- [ ] Coexistence with nouveau (the uAPI device is named "nouveau" with `nvk=1`); solved for good by the
      long-term independence goal under Next steps.
- [ ] The Hyprland session needs `VK_DRIVER_FILES` pointing at NVK while the proprietary `nvidia-utils` ICD is
      installed (it deadlocks Xwayland otherwise).
- [ ] LibOS log decoding (`.fwlogging_*` format strings are not shipped in linux-firmware); raw logs in debugfs.
- [x] Large-RPC continuation path (`_issueRpcAndWaitLarge`, `nv_rpc_call_large`) for messages > 64 KiB: send side
      done for the FBSR page lists (suspend); replies come back the same way and their continuation records are
      drained as events.

## Hardware workflow

```
sudo ./scripts/reset.sh                 # FLR: clears WPR2, GFW boot re-runs (blanks console)
sudo HW=1 ./scripts/bind.sh             # allow_hw_writes=1; probe waits for GFW boot
sudo sh -c 'echo 1 > /sys/kernel/debug/nvdrv/0000:01:00.0/run_fwsec_frts'   # once per reset
# or the whole boot: FWSEC-FRTS + Booter + GSP-RM, ~1.1 s
sudo sh -c 'echo 1 > /sys/kernel/debug/nvdrv/0000:01:00.0/run_gsp_boot'
sudo ./tests/phase6_boot.sh 0000:01:00.0 3                                   # 3 full cycles
sudo ./tests/phase6_static.sh 0000:01:00.0 3                                 # + GET_GSP_STATIC_INFO cross-check
sudo ./tests/phase7_rm.sh 0000:01:00.0 3                                     # + RM objects, controls, error paths
sudo KEEP=1 ./tests/phase12_display.sh                                       # boot + DP link + modeset, image stays up
sudo KEEP=1 DRM=1 ./tests/phase12_display.sh                                 # ... + DRM device, text console on tty3
sudo sh -c 'echo 1 > /sys/kernel/debug/nvdrv/0000:01:00.0/rm_selftest'      # RM error paths on a booted GSP
```

### Switching drivers

NVIDIA's driver (nvidia-open/nvidia-utils 615.x) stays installed. While nvdrv is in use it is disabled by
`/etc/modprobe.d/blacklist-nvidia.conf` (blacklist + `install ... /bin/false` for nvidia, nvidia_drm,
nvidia_modeset, nvidia_uvm), and the initramfs is rebuilt with `mkinitcpio -P`.

- **Quick test without touching the blacklist** (no reboot, no FLR, both directions verified): stop Hyprland,
  `sudo ./scripts/unbind.sh`, `sudo modprobe -C /usr/lib/modprobe.d nvidia_drm` (reads only the packaged
  config, so the blacklist's `install ... /bin/false` is skipped); back with
  `sudo rmmod nvidia_drm nvidia_modeset nvidia` (plus `nvidia_uvm` if loaded) and the nvdrv bind.
- **Back to NVIDIA:** stop Hyprland, `sudo ./scripts/unbind.sh`, move the blacklist to
  `/etc/modprobe.d/blacklist-nvidia.conf.disabled` (modprobe reads only `*.conf`), `sudo mkinitcpio -P`,
  `sudo modprobe nvidia_drm` (pulls in nvidia, nvidia_modeset). If it fails to initialise the GPU, reboot.
- **Back to nvdrv:** move the file back to `blacklist-nvidia.conf`, `sudo mkinitcpio -P`, stop Hyprland (apps
  holding the GPU close with it), `sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia` (or reboot), then `sudo NVDRV_PARAMS="nvk=1" KEEP=1 DRM=1
  ./tests/phase12_display.sh` and start Hyprland.
- The Hyprland config only sets `VK_DRIVER_FILES` to NVK when `/sys/module/nvdrv/parameters/nvk` is `Y`, so it
  needs no change for either driver (pre-nvdrv backup: `~/.config/hypr/hyprland.lua.bak-nvdrv`).

## Key risks

- **GSP ABI churn.** Mitigated by pinning 570.144.
- **Signed firmware.** We cannot modify FWSEC, Booter or GSP; we can only load them
  exactly as NVIDIA expects. Layout bugs show up as silent hangs, so we need good falcon
  mailbox and error-register dumping (Phase 3).
- **Single GPU = console.** Use SSH only. Keep `nvidia-open` installed as a known-good fallback.
- **Hard hangs.** A bad DMA or PRI access can lock the PCIe link. Use netconsole
  or `journalctl -k -b -1` after reboot to diagnose.

## Layout

```
PLAN.md           this file
Makefile, Kbuild  out-of-tree build
src/              kernel module sources
tools/            userspace helpers (VBIOS parser, register decoders)
scripts/          bind / unbind
tests/            per-phase smoke tests
ref/              (gitignored) NVIDIA reference trees
```
