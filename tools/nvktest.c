// SPDX-License-Identifier: GPL-2.0
/*
 * Exercise nvdrv's nouveau uAPI subset the way Mesa NVK does (nvk=1): device info, VM_INIT,
 * a GR channel with its engine objects, GEM objects in VRAM and system memory, VM_BIND,
 * EXEC with a syncobj, and the compute engine writing semaphores into both objects.
 *   cc -O2 -o nvktest tools/nvktest.c $(pkg-config --cflags --libs libdrm)
 */
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <xf86drm.h>
#include <drm/nouveau_drm.h>

#define DRM_NOUVEAU_NVIF 0x07

struct nvif_hdr { uint8_t version, type, pad02[4], owner, route; uint64_t token, object; };
struct nvif_new { uint8_t version, pad01[6], route; uint64_t token, object; uint32_t handle; int32_t oclass; };
struct nvif_sclass { uint8_t version, count, pad02[6]; struct { int32_t oclass; int16_t minver, maxver; } list[16]; };
struct nvif_mthd { uint8_t version, method, pad02[6]; };
struct nvif_info { uint8_t version, platform; uint16_t chipset; uint8_t revision, family, pad06[2];
		   uint64_t ram_size, ram_user; char chip[16], name[64]; };

#define CHECK(x) do { int _r = (x); if (_r) { fprintf(stderr, "FAIL %s: %d (%s)\n", #x, _r, strerror(-_r)); exit(1); } } while (0)

static int fd;

static int param(uint64_t p, uint64_t *v)
{
	struct drm_nouveau_getparam g = { .param = p };
	int r = drmCommandWriteRead(fd, DRM_NOUVEAU_GETPARAM, &g, sizeof(g));

	*v = g.value;
	return r;
}

static int bo_new(uint64_t size, uint32_t domain, uint32_t *handle, void **map)
{
	struct drm_nouveau_gem_new n = { .info = { .size = size, .domain = domain | NOUVEAU_GEM_DOMAIN_MAPPABLE },
					 .align = 0x10000 };
	int r = drmCommandWriteRead(fd, DRM_NOUVEAU_GEM_NEW, &n, sizeof(n));

	if (r)
		return r;
	*handle = n.info.handle;
	*map = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, n.info.map_handle);
	return *map == MAP_FAILED ? -errno : 0;
}

static int bind(uint32_t handle, uint64_t va, uint64_t size, uint32_t op)
{
	struct drm_nouveau_vm_bind_op o = { .op = op, .handle = handle, .addr = va, .range = size };
	struct drm_nouveau_vm_bind b = { .op_count = 1, .op_ptr = (uintptr_t)&o };

	return drmCommandWriteRead(fd, DRM_NOUVEAU_VM_BIND, &b, sizeof(b));
}

static uint32_t *p;
static void mthd(int subc, uint32_t m, int n, ...)
{
	__builtin_va_list ap;

	*p++ = 0x20000000 | n << 16 | subc << 13 | m >> 2;
	__builtin_va_start(ap, n);
	while (n--)
		*p++ = __builtin_va_arg(ap, uint32_t);
	__builtin_va_end(ap);
}

int main(int argc, char **argv)
{
	drmDevicePtr devs[16];
	int n = drmGetDevices2(0, devs, 16), i;
	uint64_t v;

	fd = -1;
	for (i = 0; i < n && fd < 0; i++) {
		if (!(devs[i]->available_nodes & (1 << DRM_NODE_RENDER)))
			continue;
		int f = open(devs[i]->nodes[DRM_NODE_RENDER], O_RDWR | O_CLOEXEC);
		drmVersionPtr ver = f >= 0 ? drmGetVersion(f) : NULL;

		if (ver && !strcmp(ver->name, "nouveau")) {
			printf("%s: %s %d.%d.%d (%s)\n", devs[i]->nodes[DRM_NODE_RENDER], ver->name, ver->version_major,
			       ver->version_minor, ver->version_patchlevel, ver->desc);
			fd = f;
		} else if (f >= 0) {
			close(f);
		}
		drmFreeVersion(ver);
	}
	if (fd < 0) {
		fprintf(stderr, "no nouveau render node (load nvdrv with nvk=1)\n");
		return 1;
	}

	/* the order NVK uses: VM_INIT, device object, PCI id, device info, params */
	struct drm_nouveau_vm_init vi = { 1ull << 39, (1ull << 40) - (1ull << 39) };
	CHECK(drmCommandWrite(fd, DRM_NOUVEAU_VM_INIT, &vi, sizeof(vi)));
	struct { struct nvif_hdr h; struct nvif_new n; uint64_t dev; } dn = {
		.h = { .type = 2, .owner = 0xff }, .n = { .oclass = 0x80 }, .dev = ~0ull };
	CHECK(drmCommandWrite(fd, DRM_NOUVEAU_NVIF, &dn, sizeof(dn)));
	struct { struct nvif_hdr h; struct nvif_mthd m; struct nvif_info i; } di = {
		.h = { .type = 4, .owner = 0xff } };
	CHECK(drmCommandWriteRead(fd, DRM_NOUVEAU_NVIF, &di, sizeof(di)));
	printf("device: chipset 0x%x %s \"%s\", family %u, VRAM %" PRIu64 " MiB (user %" PRIu64 ")\n", di.i.chipset,
	       di.i.chip, di.i.name, di.i.family, di.i.ram_size >> 20, di.i.ram_user >> 20);
	static const struct { int p; const char *n; } params[] = {
		{ NOUVEAU_GETPARAM_PCI_DEVICE, "pci device" }, { NOUVEAU_GETPARAM_CHIPSET_ID, "chipset" },
		{ NOUVEAU_GETPARAM_FB_SIZE, "fb size" }, { NOUVEAU_GETPARAM_AGP_SIZE, "gart size" },
		{ NOUVEAU_GETPARAM_GRAPH_UNITS, "graph units" }, { NOUVEAU_GETPARAM_EXEC_PUSH_MAX, "push max" },
		{ NOUVEAU_GETPARAM_VRAM_BAR_SIZE, "bar size" }, { NOUVEAU_GETPARAM_VRAM_USED, "vram used" },
		{ NOUVEAU_GETPARAM_PTIMER_TIME, "ptimer" }, { NOUVEAU_GETPARAM_HAS_VMA_TILEMODE, "vma tilemode" },
	};
	for (i = 0; i < (int)(sizeof(params) / sizeof(params[0])); i++) {
		int r = param(params[i].p, &v);

		printf("  %-13s %s0x%" PRIx64 "\n", params[i].n, r ? "ERROR " : "", v);
	}

	struct drm_nouveau_channel_alloc ca = { .fb_ctxdma_handle = ~0u, .tt_ctxdma_handle = NOUVEAU_FIFO_ENGINE_GR };
	CHECK(drmCommandWriteRead(fd, DRM_NOUVEAU_CHANNEL_ALLOC, &ca, sizeof(ca)));
	struct { struct nvif_hdr h; struct nvif_sclass s; } sc = {
		.h = { .type = 1, .route = 0xff, .token = ca.channel }, .s = { .count = 16 } };
	CHECK(drmCommandWriteRead(fd, DRM_NOUVEAU_NVIF, &sc, sizeof(sc)));
	printf("channel %d, %u classes:", ca.channel, sc.s.count);
	for (i = 0; i < sc.s.count; i++)
		printf(" %04x", sc.s.list[i].oclass);
	printf("\n");
	for (i = 0; i < sc.s.count; i++) {
		struct { struct nvif_hdr h; struct nvif_new n; } on = {
			.h = { .type = 2, .route = 0xff, .token = ca.channel },
			.n = { .handle = 0xbeef0000 + i, .oclass = sc.s.list[i].oclass } };
		CHECK(drmCommandWrite(fd, DRM_NOUVEAU_NVIF, &on, sizeof(on)));
	}

	uint32_t hpb, hvram, hgart;
	void *pb, *vram, *gart;
	CHECK(bo_new(0x10000, NOUVEAU_GEM_DOMAIN_GART, &hpb, &pb));
	CHECK(bo_new(0x10000, NOUVEAU_GEM_DOMAIN_VRAM, &hvram, &vram));
	CHECK(bo_new(0x10000, NOUVEAU_GEM_DOMAIN_GART, &hgart, &gart));
	printf("VRAM object reads back 0x%08x after allocation (cleared: %s)\n", *(volatile uint32_t *)vram,
	       *(volatile uint32_t *)vram ? "no" : "yes");
	*(volatile uint32_t *)vram = 0x11111111;
	*(volatile uint32_t *)gart = 0x22222222;
	CHECK(bind(hpb, 0x10000000, 0x10000, DRM_NOUVEAU_VM_BIND_OP_MAP));
	CHECK(bind(hvram, 0x20000000, 0x10000, DRM_NOUVEAU_VM_BIND_OP_MAP));
	CHECK(bind(hgart, 0x30000000, 0x10000, DRM_NOUVEAU_VM_BIND_OP_MAP));

	/* compute class on subchannel 1: report semaphores into VRAM and GART */
	p = pb;
	mthd(1, 0x0000, 1, 0xc7c0);
	mthd(1, 0x1b00, 4, 0x0, 0x20000000, 0xcafe0001, 1u << 28);
	mthd(1, 0x1b00, 4, 0x0, 0x30000000, 0xcafe0002, 1u << 28);
	struct drm_nouveau_exec_push ep = { .va = 0x10000000, .va_len = (uint32_t)((char *)p - (char *)pb) };
	uint32_t so;
	CHECK(drmSyncobjCreate(fd, 0, &so));
	struct drm_nouveau_sync sig = { .flags = DRM_NOUVEAU_SYNC_SYNCOBJ, .handle = so };
	struct drm_nouveau_exec ex = { .channel = ca.channel, .push_count = 1, .push_ptr = (uintptr_t)&ep,
				       .sig_count = 1, .sig_ptr = (uintptr_t)&sig };
	CHECK(drmCommandWriteRead(fd, DRM_NOUVEAU_EXEC, &ex, sizeof(ex)));
	int r = drmSyncobjWait(fd, &so, 1, INT64_MAX, 0, NULL);
	uint32_t a = *(volatile uint32_t *)vram, b = *(volatile uint32_t *)gart;
	printf("exec: syncobj wait %d, VRAM 0x%08x, GART 0x%08x -> %s\n", r, a, b,
	       !r && a == 0xcafe0001 && b == 0xcafe0002 ? "PASS" : "FAIL");

	/* a second submission waiting on the first (in-fence path) */
	p = pb;
	mthd(1, 0x1b00, 4, 0x0, 0x20000000, 0xcafe0003, 1u << 28);
	ep.va_len = (uint32_t)((char *)p - (char *)pb);
	uint32_t so2;
	CHECK(drmSyncobjCreate(fd, 0, &so2));
	struct drm_nouveau_sync sig2 = { .flags = DRM_NOUVEAU_SYNC_SYNCOBJ, .handle = so2 };
	ex.wait_count = 1;
	ex.wait_ptr = (uintptr_t)&sig;
	ex.sig_ptr = (uintptr_t)&sig2;
	CHECK(drmCommandWriteRead(fd, DRM_NOUVEAU_EXEC, &ex, sizeof(ex)));
	r = drmSyncobjWait(fd, &so2, 1, INT64_MAX, 0, NULL);
	a = *(volatile uint32_t *)vram;
	printf("exec 2: syncobj wait %d, VRAM 0x%08x -> %s\n", r, a, !r && a == 0xcafe0003 ? "PASS" : "FAIL");

	CHECK(bind(0, 0x20000000, 0x10000, DRM_NOUVEAU_VM_BIND_OP_UNMAP));
	struct drm_nouveau_channel_free cf = { .channel = ca.channel };
	CHECK(drmCommandWrite(fd, DRM_NOUVEAU_CHANNEL_FREE, &cf, sizeof(cf)));
	close(fd);
	printf("done\n");
	return 0;
}
