# nvdrv

A Linux kernel driver for NVIDIA Ampere GPUs that boots NVIDIA's GSP-RM firmware and
talks to it directly, written mostly from scratch. It does not use the NVIDIA kernel
modules; parts taken from nouveau keep their original license headers.

> [!WARNING]
> **Work in progress and unstable.** This repository is pushed after every change,
> including half-finished and broken states. Nothing here is a release, the code
> changes without notice, and any commit may fail to build or misbehave.
>
> The driver writes GPU registers, runs signed firmware on the GPU, DMAs into
> host memory and takes over the display. A wrong step can hang the GPU or the
> whole machine. Only run it on hardware you can afford to reset, over SSH, with a
> way to recover (`scripts/reset.sh`, or a reboot).

## Status

Developed and tested on a single GPU: **GA104 (RTX 3060 Ti), Arch Linux, kernel 7.2,
GSP firmware 570.144** from linux-firmware. Other GPUs and firmware versions are not
supported. The GSP RPC ABI changes with every firmware release.

What works today (see [PLAN.md](PLAN.md) for the phase table, findings and next steps):

- chip identification, VBIOS parsing, FWSEC extraction
- secure boot chain: FWSEC-FRTS, Booter on SEC2, GSP-RM boot to `GSP_INIT_DONE`
- GSP RPC queues, synchronous calls, `GET_GSP_STATIC_INFO`
- RM objects (client / device / subdevice), RM controls, error paths
- display: DisplayPort sink discovery, EDID, link training, modeset scanning out
  from VRAM, BAR1 CPU mapping
- a DRM/KMS device with fbdev emulation: the Linux text console runs on nvdrv

Not there yet: mode changes and hotplug through DRM, memory management, GPU
channels/copy engine, compute, interrupts, and a clean firmware unload (unbind
currently resets the GPU).

## Building and running

```sh
make                                    # out-of-tree module against the running kernel
sudo HW=1 ./scripts/bind.sh             # load with allow_hw_writes=1, bind via driver_override
sudo KEEP=1 DRM=1 ./tests/phase12_display.sh  # boot GSP-RM, light the DP display, console on tty3
sudo ./scripts/unbind.sh                # unbind (resets the GPU) and unload
```

The module never binds automatically: it has no device table and binds only through
`driver_override`. Without `allow_hw_writes=1` it is read-only. Debug state and
triggers are under `/sys/kernel/debug/nvdrv/<pci-address>/`. `tests/` has one test per
phase; most of them write GPU registers.

Firmware is not included. The driver loads the 570.144 GSP/booter files from
`/lib/firmware/nvidia/ga104/gsp/` (linux-firmware).

## References

Register offsets, class definitions and RPC layouts come from NVIDIA's
[open-gpu-doc](https://github.com/NVIDIA/open-gpu-doc) and
[open-gpu-kernel-modules](https://github.com/NVIDIA/open-gpu-kernel-modules) (tag 570.144),
which are not part of this repository. The `tools/*_probe.sh` scripts derive struct
layouts by compiling NVIDIA's own headers from a local checkout in `ref/ogkm`.

## License

GPL-2.0-only, see [COPYING](COPYING).
