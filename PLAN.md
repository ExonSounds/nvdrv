# nvdrv — an NVIDIA GPU driver on top of GSP-RM

Target hardware: **GA104 (RTX 3060 Ti LHR, 10de:2489), Ampere**, PCI 0000:01:00.0
Kernel: Arch 7.2.x, out-of-tree C module (Kbuild), GPL-2.0.

## Ground rules

- **References:** NVIDIA `open-gpu-doc` (register/class headers, MIT), `open-gpu-kernel-modules`
  (MIT/GPL dual) for register offsets and GSP message layouts, public specs (PCI Firmware spec,
  RISC-V), and, **since 2026-11-08, nouveau** (the user lifted the original no-nouveau rule to speed
  things up). nouveau code may be used and built on; keep its copyright/license headers (MIT) on
  anything taken from it, and note the origin in the file.
- **One firmware version, pinned: `570.144`** (`/lib/firmware/nvidia/ga104/gsp/*-570.144.bin.zst`,
  from linux-firmware). The GSP RPC ABI changes with every release, so every
  struct we define must match that tag of `open-gpu-kernel-modules`.
- **Read-only until it has to write.** Each phase adds the minimum number of register
  writes it needs, and documents each one.
- **Develop over SSH.** This GPU also drives the console (simpledrm on BAR1).
  From Phase 3 on, expect the local display to go dark. Recovery is
  `scripts/unbind.sh` then `echo 1 > /sys/bus/pci/devices/0000:01:00.0/reset` (FLR), or a reboot.
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
| 6 | **GSP RPC layer** | Shared-memory message/status queues ✅, send/receive + checksums ✅, CPU sequencer incl. CORE_RESUME ✅, pre-boot RPCs ✅ (`tests/phase6_rpc.sh`), doorbell + synchronous calls + event draining ✅, **GET_GSP_STATIC_INFO ✅ (`tests/phase6_static.sh`)**; clean `booter_unload` shutdown, interrupts. |
| 7 | **RM object model** | GSP_RM_ALLOC / GSP_RM_CONTROL / FREE ✅; root client → device → subdevice ✅; routed control on our subdevice and internal control on GSP's internal subdevice ✅; error paths verified ✅ (`tests/phase7_rm.sh`). |
| 8 | **Memory management** | VRAM allocator, GPU virtual address spaces and page tables, sysmem mappings. |
| 9 | **Channels + Copy Engine** | GPFIFO channel, USERD/doorbell, push buffer, run a CE memcpy between VRAM and sysmem and fence on a semaphore. **First "real work" milestone.** |
| 10 | **userspace API** | DRM render node and a small uAPI (BO alloc/map, channel create, submit, wait). |
| 11 | **Compute** | Allocate compute class, launch a hand-assembled or NAK-compiled kernel. |
| 12 | **Display** (pulled forward) | DP sink discovery + EDID ✅, SOR assignment + DP link training via RM ✅, display instmem/ctxdma/push buffers done by nvdrv in CPU-RM's place ✅, core + window channels ✅, **modeset with a test pattern scanned out from VRAM, confirmed on the monitor (2026-11-08)** ✅ (`tests/phase12_display.sh`). Next: KMS/DRM interface, input LUT, cursor. |

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
     ISO 0, identity FMT matrix, **composition bypass, no CSC/ILUT/TMO methods**.
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
- **Open:** `DP_CONFIG_STREAM` returns NV_ERR_ILLEGAL_ACTION (0x16) before the modeset; the link works with RM's
  defaults. The core channel also latches EXCEPT 0x10000080 arg 0x41 during the modeset update, but the update takes
  effect. The EDID read during one early probe gave 1920x1080; later reads give 2560x1440 (sink waking up?).

## Next steps

0. **DRM/KMS device** on top of BAR1: dumb buffers in VRAM mapped through BAR1, one CRTC/plane/encoder/connector,
   fbdev emulation so the console runs on nvdrv. Then identity input LUT, DP_CONFIG_STREAM, core exception.
1. **Phase 8 groundwork:** a VA space (`FERMI_VASPACE_A`, 0x90f1) under our device and memory allocations
   (`NV01_MEMORY_LOCAL_USER` / sysmem). Check the GA10x class list via `NV2080_CTRL_CMD_GPU_GET_ENGINES_V2` /
   classlist first (note: GET_ENGINES / GET_ENGINE_CLASSLIST *are* FINN-serialized; use the V2 variants or check).
   Messages > 64 KiB need the large-RPC continuation path (`_issueRpcAndWaitLarge`), which is not implemented yet.
2. **Clean unload** instead of FLR: UNLOADING_GUEST_DRIVER RPC, wait for halt, Booter-Unload on SEC2 (WPR2 torn down).
3. **Interrupts** (GSP → CPU message notification) instead of polling; health check of GSP while running.
4. LibOS log decoding needs `.fwlogging_*` format strings, which the linux-firmware ELF does not ship. Until then, keep raw logs in debugfs.

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
sudo sh -c 'echo 1 > /sys/kernel/debug/nvdrv/0000:01:00.0/rm_selftest'      # RM error paths on a booted GSP
```

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
