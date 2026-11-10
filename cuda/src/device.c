// SPDX-License-Identifier: MIT
/*
 * cuInit and device queries. Devices are nvdrv's render nodes (driver "nouveau", description
 * "nvdrv ..."), queried the way Mesa NVK does: VM_INIT, the NVIF device object and its info,
 * GETPARAMs. Fixed per-architecture limits are the documented compute capability 8.6 values
 * (CUDA C++ Programming Guide, "Compute Capabilities").
 */
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <xf86drm.h>
#include <drm/nouveau_drm.h>

#include "internal.h"

#define DRM_NOUVEAU_NVIF 0x07

struct nvif_hdr { uint8_t version, type, pad02[4], owner, route; uint64_t token, object; };
struct nvif_new { uint8_t version, pad01[6], route; uint64_t token, object; uint32_t handle; int32_t oclass; };
struct nvif_mthd { uint8_t version, method, pad02[6]; };
struct nvif_info { uint8_t version, platform; uint16_t chipset; uint8_t revision, family, pad06[2];
		   uint64_t ram_size, ram_user; char chip[16], name[64]; };

struct nvc_device nvc_devices[NVC_MAX_DEVICES];
int nvc_ndevices;
bool nvc_initialized;
static pthread_once_t init_once = PTHREAD_ONCE_INIT;

/* The device object every nouveau client creates first (NVK order: after VM_INIT). */
int nvc_nvif_device(int fd)
{
	struct { struct nvif_hdr h; struct nvif_new n; uint64_t dev; } dn = {
		.h = { .type = 2, .owner = 0xff }, .n = { .oclass = 0x80 }, .dev = ~0ull };

	return drmCommandWrite(fd, DRM_NOUVEAU_NVIF, &dn, sizeof(dn));
}

static int probe(drmDevicePtr d, struct nvc_device *out)
{
	struct { struct nvif_hdr h; struct nvif_mthd m; struct nvif_info i; } di = { .h = { .type = 4, .owner = 0xff } };
	struct drm_nouveau_vm_init vi = { NVC_KERNEL_VA, NVC_KERNEL_VA_SIZE };
	struct drm_nouveau_getparam gp = { .param = NOUVEAU_GETPARAM_GRAPH_UNITS };
	drmVersionPtr ver;
	int fd, ok = 0;

	if (!(d->available_nodes & (1 << DRM_NODE_RENDER)) || d->bustype != DRM_BUS_PCI)
		return 0;
	fd = open(d->nodes[DRM_NODE_RENDER], O_RDWR | O_CLOEXEC);
	if (fd < 0)
		return 0;
	ver = drmGetVersion(fd);
	if (!ver || strcmp(ver->name, "nouveau") || strncmp(ver->desc, "nvdrv", 5))
		goto out;
	if (drmCommandWrite(fd, DRM_NOUVEAU_VM_INIT, &vi, sizeof(vi)) || nvc_nvif_device(fd) ||
	    drmCommandWriteRead(fd, DRM_NOUVEAU_NVIF, &di, sizeof(di)))
		goto out;
	if (di.i.chipset < 0x170 || di.i.chipset >= 0x180)	/* GA10x: the classes nvcuda programs */
		goto out;
	memset(out, 0, sizeof(*out));
	snprintf(out->path, sizeof(out->path), "%s", d->nodes[DRM_NODE_RENDER]);
	snprintf(out->name, sizeof(out->name), "%s", di.i.name);
	out->chipset = di.i.chipset;
	out->vram = di.i.ram_size;
	out->vram_user = di.i.ram_user;
	/* GRAPH_UNITS: GPCs 7:0, TPCs 15:8 (nouveau gf100_gr_units); two SMs per TPC since Turing */
	if (!drmCommandWriteRead(fd, DRM_NOUVEAU_GETPARAM, &gp, sizeof(gp)))
		out->sm_count = ((gp.value >> 8) & 0xff) * 2;
	out->gpc_count = gp.value & 0xff;
	out->tpc_count = (gp.value >> 8) & 0xff;
	out->pci_domain = d->businfo.pci->domain;
	out->pci_bus = d->businfo.pci->bus;
	out->pci_dev = d->businfo.pci->dev;
	out->pci_func = d->businfo.pci->func;
	out->pci_device_id = d->deviceinfo.pci->device_id;
	ok = 1;
out:
	drmFreeVersion(ver);
	close(fd);
	return ok;
}

static void do_init(void)
{
	drmDevicePtr devs[16];
	int n = drmGetDevices2(0, devs, 16), i;

	for (i = 0; i < n; i++)
		if (nvc_ndevices < NVC_MAX_DEVICES && probe(devs[i], &nvc_devices[nvc_ndevices]))
			nvc_ndevices++;
	if (n > 0)
		drmFreeDevices(devs, n);
	nvc_initialized = true;
}

CUAPI CUresult cuInit(unsigned int flags)
{
	if (flags)
		return CUDA_ERROR_INVALID_VALUE;
	pthread_once(&init_once, do_init);
	return nvc_ndevices ? CUDA_SUCCESS : CUDA_ERROR_NO_DEVICE;
}

CUAPI CUresult cuDriverGetVersion(int *version)
{
	if (!version)
		return CUDA_ERROR_INVALID_VALUE;
	*version = CUDA_VERSION;
	return CUDA_SUCCESS;
}

#define CHECK_INIT()	do { if (!nvc_initialized) return CUDA_ERROR_NOT_INITIALIZED; } while (0)
#define CHECK_DEV(d)	do { CHECK_INIT(); if ((d) < 0 || (d) >= nvc_ndevices) return CUDA_ERROR_INVALID_DEVICE; } while (0)

CUAPI CUresult cuDeviceGetCount(int *count)
{
	CHECK_INIT();
	if (!count)
		return CUDA_ERROR_INVALID_VALUE;
	*count = nvc_ndevices;
	return CUDA_SUCCESS;
}

CUAPI CUresult cuDeviceGet(CUdevice *device, int ordinal)
{
	CHECK_INIT();
	if (!device)
		return CUDA_ERROR_INVALID_VALUE;
	if (ordinal < 0 || ordinal >= nvc_ndevices)
		return CUDA_ERROR_INVALID_DEVICE;
	*device = ordinal;
	return CUDA_SUCCESS;
}

CUAPI CUresult cuDeviceGetName(char *name, int len, CUdevice dev)
{
	CHECK_DEV(dev);
	if (!name || len <= 0)
		return CUDA_ERROR_INVALID_VALUE;
	snprintf(name, len, "%s", nvc_devices[dev].name);
	return CUDA_SUCCESS;
}

/* No GPU UUID through the uAPI yet: a stable one derived from the PCI address (version 8 layout). */
CUAPI CUresult cuDeviceGetUuid(CUuuid *uuid, CUdevice dev)
{
	struct nvc_device *d;

	CHECK_DEV(dev);
	if (!uuid)
		return CUDA_ERROR_INVALID_VALUE;
	d = &nvc_devices[dev];
	memset(uuid, 0, sizeof(*uuid));
	memcpy(uuid->bytes, "nvdrv", 5);
	uuid->bytes[6] = 0x80;
	uuid->bytes[8] = 0x80;
	uuid->bytes[10] = d->pci_domain >> 8;
	uuid->bytes[11] = d->pci_domain;
	uuid->bytes[12] = d->pci_bus;
	uuid->bytes[13] = d->pci_dev;
	uuid->bytes[14] = d->pci_func;
	uuid->bytes[15] = d->chipset;
	return CUDA_SUCCESS;
}

CUAPI CUresult cuDeviceTotalMem_v2(size_t *bytes, CUdevice dev)
{
	CHECK_DEV(dev);
	if (!bytes)
		return CUDA_ERROR_INVALID_VALUE;
	*bytes = nvc_devices[dev].vram_user;
	return CUDA_SUCCESS;
}

CUAPI CUresult cuDeviceComputeCapability(int *major, int *minor, CUdevice dev)
{
	CHECK_DEV(dev);
	if (!major || !minor)
		return CUDA_ERROR_INVALID_VALUE;
	*major = 8;
	*minor = 6;
	return CUDA_SUCCESS;
}

CUAPI CUresult cuDeviceGetPCIBusId(char *busid, int len, CUdevice dev)
{
	struct nvc_device *d;

	CHECK_DEV(dev);
	if (!busid || len <= 0)
		return CUDA_ERROR_INVALID_VALUE;
	d = &nvc_devices[dev];
	snprintf(busid, len, "%08x:%02x:%02x.%x", d->pci_domain, d->pci_bus, d->pci_dev, d->pci_func);
	return CUDA_SUCCESS;
}

CUAPI CUresult cuDeviceGetAttribute(int *value, CUdevice_attribute attrib, CUdevice dev)
{
	struct nvc_device *d;
	int v = 0;

	CHECK_DEV(dev);
	if (!value)
		return CUDA_ERROR_INVALID_VALUE;
	d = &nvc_devices[dev];
	switch ((int)attrib) {
	case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK:		v = 1024; break;
	case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X:		v = 1024; break;
	case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y:		v = 1024; break;
	case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z:		v = 64; break;
	case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X:		v = 0x7fffffff; break;
	case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y:		v = 65535; break;
	case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z:		v = 65535; break;
	case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK:	v = 48 << 10; break;
	case 9 /* TOTAL_CONSTANT_MEMORY */:			v = 64 << 10; break;
	case 10 /* WARP_SIZE */:				v = 32; break;
	case CU_DEVICE_ATTRIBUTE_MAX_PITCH:			v = 0x7fffffff; break;
	case CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK:	v = 64 << 10; break;
	case 13 /* CLOCK_RATE, kHz: RTX 3060 Ti boost clock (no live clock query yet) */: v = 1665000; break;
	case 14 /* TEXTURE_ALIGNMENT */:			v = 512; break;
	case 15 /* GPU_OVERLAP */:				v = 1; break;
	case 16 /* MULTIPROCESSOR_COUNT */:			v = d->sm_count; break;
	case 17 /* KERNEL_EXEC_TIMEOUT */:			v = 1; break;
	case 18 /* INTEGRATED */:				v = 0; break;
	case 19 /* CAN_MAP_HOST_MEMORY */:			v = 1; break;
	case 20 /* COMPUTE_MODE: default */:			v = 0; break;
	case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH:	v = 131072; break;
	case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH:	v = 131072; break;
	case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT:	v = 65536; break;
	case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH:
	case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT:
	case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH:	v = 16384; break;
	case 30 /* SURFACE_ALIGNMENT */:			v = 512; break;
	case 31 /* CONCURRENT_KERNELS */:			v = 1; break;
	case 32 /* ECC_ENABLED */:				v = 0; break;
	case 33 /* PCI_BUS_ID */:				v = d->pci_bus; break;
	case 34 /* PCI_DEVICE_ID */:				v = d->pci_dev; break;
	case 35 /* TCC_DRIVER */:				v = 0; break;
	case 36 /* MEMORY_CLOCK_RATE, kHz: 14 Gbps GDDR6 */:	v = 7001000; break;
	case 37 /* GLOBAL_MEMORY_BUS_WIDTH */:			v = 256; break;
	case 38 /* L2_CACHE_SIZE (GSP-RM: 3072 KiB on this board) */: v = 3 << 20; break;
	case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR: v = 1536; break;
	case 40 /* ASYNC_ENGINE_COUNT */:			v = 1; break;
	case 41 /* UNIFIED_ADDRESSING */:			v = 1; break;
	case 50 /* PCI_DOMAIN_ID */:				v = d->pci_domain; break;
	case 51 /* TEXTURE_PITCH_ALIGNMENT */:			v = 32; break;
	case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR:	v = 8; break;
	case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR:	v = 6; break;
	case 78 /* STREAM_PRIORITIES_SUPPORTED */:		v = 0; break;
	case 79 /* GLOBAL_L1_CACHE_SUPPORTED */:		v = 1; break;
	case 80 /* LOCAL_L1_CACHE_SUPPORTED */:			v = 1; break;
	case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR: v = 100 << 10; break;
	case CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR: v = 64 << 10; break;
	case 83 /* MANAGED_MEMORY: no unified memory yet */:	v = 0; break;
	case 84 /* MULTI_GPU_BOARD */:				v = 0; break;
	case 85 /* MULTI_GPU_BOARD_GROUP_ID */:			v = 0; break;
	case 86 /* HOST_NATIVE_ATOMIC_SUPPORTED */:		v = 0; break;
	case 87 /* SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO */:	v = 64; break;
	case 90 /* COMPUTE_PREEMPTION_SUPPORTED */:		v = 1; break;
	case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN: v = 99 << 10; break;
	case 106 /* MAX_BLOCKS_PER_MULTIPROCESSOR */:		v = 16; break;
	default:
		if ((int)attrib <= 0 || (int)attrib > 140)
			return CUDA_ERROR_INVALID_VALUE;
		v = 0;		/* a feature nvcuda does not offer */
	}
	*value = v;
	return CUDA_SUCCESS;
}
