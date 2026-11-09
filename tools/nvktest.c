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
#include <time.h>
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

static int bo_new_align(uint64_t size, uint32_t domain, uint32_t align, uint32_t *handle, void **map)
{
	struct drm_nouveau_gem_new n = { .info = { .size = size, .domain = domain | NOUVEAU_GEM_DOMAIN_MAPPABLE },
					 .align = align };
	int r = drmCommandWriteRead(fd, DRM_NOUVEAU_GEM_NEW, &n, sizeof(n));

	if (r)
		return r;
	*handle = n.info.handle;
	*map = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, n.info.map_handle);
	return *map == MAP_FAILED ? -errno : 0;
}

static int bo_new(uint64_t size, uint32_t domain, uint32_t *handle, void **map)
{
	return bo_new_align(size, domain, 0x10000, handle, map);
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

	{
		struct drm_nouveau_get_zcull_info z = {};
		int zr = drmCommandRead(fd, DRM_NOUVEAU_GET_ZCULL_INFO, &z, sizeof(z));

		printf("zcull (%d): align %ux%u, aliquots %u/%u, region x%u, headers %u/%u, subregions %u (%ux%u), ctxsw %u/%u\n",
		       zr, z.width_align_pixels, z.height_align_pixels, z.pixel_squares_by_aliquots, z.aliquot_total,
		       z.zcull_region_byte_multiplier, z.zcull_region_header_size, z.zcull_subregion_header_size,
		       z.subregion_count, z.subregion_width_align_pixels, z.subregion_height_align_pixels, z.ctxsw_size,
		       z.ctxsw_align);
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

	/* a second GR channel in the same VM; alternate work between the two (context switches) */
	struct drm_nouveau_channel_alloc ca2 = { .fb_ctxdma_handle = ~0u, .tt_ctxdma_handle = NOUVEAU_FIFO_ENGINE_GR };
	CHECK(drmCommandWriteRead(fd, DRM_NOUVEAU_CHANNEL_ALLOC, &ca2, sizeof(ca2)));
	{
		struct { struct nvif_hdr h; struct nvif_new n; } on = {
			.h = { .type = 2, .route = 0xff, .token = ca2.channel }, .n = { .handle = 1, .oclass = 0xc7c0 } };
		CHECK(drmCommandWrite(fd, DRM_NOUVEAU_NVIF, &on, sizeof(on)));
		on.n.oclass = 0xc797;
		CHECK(drmCommandWrite(fd, DRM_NOUVEAU_NVIF, &on, sizeof(on)));
	}
	for (i = 0; i < 6; i++) {
		int c = i & 1 ? ca2.channel : ca.channel;

		p = (uint32_t *)((char *)pb + 0x1000 * (i + 1));
		uint32_t *start = p;
		if (getenv("THREED")) {	/* 3D class on subchannel 0: SET_REPORT_SEMAPHORE (0x1b00) */
			mthd(0, 0x0000, 1, 0xc797);
			mthd(0, 0x1b00, 4, 0x0, 0x20000000 + 4 * i, 0xd0000000 + i, 1u << 28);
		} else {
			mthd(1, 0x0000, 1, 0xc7c0);
			mthd(1, 0x1b00, 4, 0x0, 0x20000000 + 4 * i, 0xd0000000 + i, 1u << 28);
		}
		ep.va = 0x10000000 + 0x1000 * (i + 1);
		ep.va_len = (uint32_t)((char *)p - (char *)start);
		ex.channel = c;
		ex.wait_count = 0;
		ex.sig_ptr = (uintptr_t)&sig2;
		CHECK(drmCommandWriteRead(fd, DRM_NOUVEAU_EXEC, &ex, sizeof(ex)));
	}
	r = drmSyncobjWait(fd, &so2, 1, INT64_MAX, 0, NULL);
	usleep(100000);
	for (i = 0; i < 6; i++)
		printf("  alternate %d (channel %d): 0x%08x\n", i, i & 1 ? ca2.channel : ca.channel,
		       ((volatile uint32_t *)vram)[i]);
	struct drm_nouveau_channel_free cf2 = { .channel = ca2.channel };
	CHECK(drmCommandWrite(fd, DRM_NOUVEAU_CHANNEL_FREE, &cf2, sizeof(cf2)));

	/*
	 * nvktest huge: 2 MiB pages. A 6 MiB VRAM object (2 MiB aligned) at a 2 MiB-aligned VA is
	 * three 2 MiB PTEs; a 4 KiB hole in the middle region splits it (the others stay huge);
	 * remapping the hole with the same object range merges it back. GPU writes must land
	 * correctly in every state. Read debugfs mem for the PTE counters.
	 */
	if (argc > 1 && !strcmp(argv[1], "huge")) {
		uint32_t hh;
		void *mh;
		const uint64_t base = 0x80000000, hole = 0x200000 + 0x81000;
		static const uint32_t offs[] = { 0x0, 0x1ff000, 0x200000, 0x280000, 0x2ff004, 0x400100, 0x5ffff0 };
		int bad = 0, rounds = 3;

		CHECK(bo_new_align(0x600000, NOUVEAU_GEM_DOMAIN_VRAM, 0x200000, &hh, &mh));
		CHECK(bind(hh, base, 0x600000, DRM_NOUVEAU_VM_BIND_OP_MAP));
		for (int round = 0; round < rounds; round++) {
			if (round == 1)	/* split the middle region */
				CHECK(bind(0, base + hole, 0x1000, DRM_NOUVEAU_VM_BIND_OP_UNMAP));
			if (round == 2) {	/* the same object range back: the region can merge */
				struct drm_nouveau_vm_bind_op o = { .op = DRM_NOUVEAU_VM_BIND_OP_MAP, .handle = hh,
								    .addr = base + hole, .bo_offset = hole, .range = 0x1000 };
				struct drm_nouveau_vm_bind b = { .op_count = 1, .op_ptr = (uintptr_t)&o };
				CHECK(drmCommandWriteRead(fd, DRM_NOUVEAU_VM_BIND, &b, sizeof(b)));
			}
			p = pb;
			mthd(1, 0x0000, 1, 0xc7c0);
			for (i = 0; i < (int)(sizeof(offs) / sizeof(offs[0])); i++)
				mthd(1, 0x1b00, 4, (uint32_t)((base + offs[i]) >> 32), (uint32_t)(base + offs[i]),
				     0x48000000u + round * 0x100 + i, 1u << 28);
			ep.va = 0x10000000;
			ep.va_len = (uint32_t)((char *)p - (char *)pb);
			ex.channel = ca.channel;
			ex.wait_count = 0;
			ex.sig_ptr = (uintptr_t)&sig2;
			CHECK(drmCommandWriteRead(fd, DRM_NOUVEAU_EXEC, &ex, sizeof(ex)));
			r = drmSyncobjWait(fd, &so2, 1, INT64_MAX, 0, NULL);
			for (i = 0; i < (int)(sizeof(offs) / sizeof(offs[0])); i++) {
				uint32_t got = ((volatile uint32_t *)mh)[offs[i] / 4], want = 0x48000000u + round * 0x100 + i;

				bad += got != want || r;
				if (got != want)
					printf("  round %d +0x%06x: 0x%08x, expected 0x%08x\n", round, offs[i], got, want);
			}
			printf("  round %d (%s): GPU writes %s\n", round, round == 0 ? "3 x 2 MiB" : round == 1 ?
			       "middle split" : "merged again", bad ? "WRONG" : "ok");
		}
		printf("huge: %d wrong -> %s\n", bad, bad ? "FAIL" : "PASS");
		return 0;
	}

	/*
	 * nvktest sync: (1) a channel stalled on a semaphore the CPU releases later; meanwhile
	 * (2) an unmap returns at once, (3) CPU_PREP from a second client: busy for an object the
	 * stalled client maps (dma-buf shared), idle for one only the second client holds,
	 * (4) an asynchronous VM_BIND waiting on a syncobj returns at once and runs once the GPU
	 * signals it; the new mapping then works.
	 */
	if (argc > 1 && !strcmp(argv[1], "sync")) {
		uint32_t hshare, hgate, hnew, so_w, so_s, so_job;
		void *mshare, *mgate, *mnew;
		struct timespec t0, t1;
		int fd2, prime, okc = 0;

		CHECK(bo_new(0x10000, NOUVEAU_GEM_DOMAIN_VRAM, &hshare, &mshare));
		CHECK(bo_new(0x10000, NOUVEAU_GEM_DOMAIN_GART, &hgate, &mgate));
		CHECK(bo_new(0x10000, NOUVEAU_GEM_DOMAIN_VRAM, &hnew, &mnew));
		*(volatile uint32_t *)mgate = 0;
		CHECK(bind(hshare, 0x60000000, 0x10000, DRM_NOUVEAU_VM_BIND_OP_MAP));
		CHECK(bind(hgate, 0x61000000, 0x10000, DRM_NOUVEAU_VM_BIND_OP_MAP));
		CHECK(drmSyncobjCreate(fd, 0, &so_job));
		/* (1) stall: host acquire (>= 1) on the gate word, then a compute write */
		p = pb;
		mthd(0, 0x005c, 5, 0x61000000, 0x0, 1, 0, 0x2 /* ACQ_STRICT_GEQ */);
		mthd(1, 0x0000, 1, 0xc7c0);
		mthd(1, 0x1b00, 4, 0x0, 0x60000000, 0x51a11ed, 1u << 28);
		ep.va = 0x10000000;
		ep.va_len = (uint32_t)((char *)p - (char *)pb);
		struct drm_nouveau_sync sj = { .flags = DRM_NOUVEAU_SYNC_SYNCOBJ, .handle = so_job };
		ex.channel = ca.channel;
		ex.wait_count = 0;
		ex.sig_count = 1;
		ex.sig_ptr = (uintptr_t)&sj;
		CHECK(drmCommandWriteRead(fd, DRM_NOUVEAU_EXEC, &ex, sizeof(ex)));
		usleep(20000);
		printf("  stalled job: syncobj %s\n", drmSyncobjWait(fd, &so_job, 1, 0, 0, NULL) ? "busy (expected)" : "signalled?!");

		/* (2) unmap something unrelated while the GPU is busy */
		clock_gettime(CLOCK_MONOTONIC, &t0);
		CHECK(bind(0, 0x20000000, 0x10000, DRM_NOUVEAU_VM_BIND_OP_UNMAP));
		clock_gettime(CLOCK_MONOTONIC, &t1);
		double ums = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
		printf("  unmap while busy: %.2f ms\n", ums);
		okc += ums < 100;

		/* (3) second client: CPU_PREP on the shared object vs. its own object */
		{	/* a second client on the same render node */
			char lnk[64], path[256] = "";

			snprintf(lnk, sizeof(lnk), "/proc/self/fd/%d", fd);
			if (readlink(lnk, path, sizeof(path) - 1) < 0)
				return 1;
			fd2 = open(path, O_RDWR | O_CLOEXEC);
			if (fd2 < 0)
				return 1;
		}
		uint32_t himp, hown;
		CHECK(drmPrimeHandleToFD(fd, hshare, DRM_CLOEXEC, &prime));
		CHECK(drmPrimeFDToHandle(fd2, prime, &himp));
		{
			int save = fd;
			void *dummy;

			fd = fd2;
			CHECK(bo_new(0x10000, NOUVEAU_GEM_DOMAIN_VRAM, &hown, &dummy));
			fd = save;
		}
		struct drm_nouveau_gem_cpu_prep cp = { .handle = himp, .flags = NOUVEAU_GEM_CPU_PREP_NOWAIT };
		int r1 = drmCommandWrite(fd2, DRM_NOUVEAU_GEM_CPU_PREP, &cp, sizeof(cp));
		cp.handle = hown;
		int r2 = drmCommandWrite(fd2, DRM_NOUVEAU_GEM_CPU_PREP, &cp, sizeof(cp));
		printf("  CPU_PREP nowait from client 2: shared object %d (%s), own object %d (%s)\n", r1,
		       r1 == -EBUSY ? "busy, correct" : "WRONG", r2, r2 == 0 ? "idle, correct" : "WRONG");
		okc += r1 == -EBUSY && r2 == 0;

		/* (4) asynchronous map of hnew, waiting on so_w (nothing submitted to it yet) */
		CHECK(drmSyncobjCreate(fd, 0, &so_w));
		CHECK(drmSyncobjCreate(fd, 0, &so_s));
		struct drm_nouveau_sync bw = { .flags = DRM_NOUVEAU_SYNC_SYNCOBJ, .handle = so_w };
		struct drm_nouveau_sync bs = { .flags = DRM_NOUVEAU_SYNC_SYNCOBJ, .handle = so_s };
		struct drm_nouveau_vm_bind_op bo = { .op = DRM_NOUVEAU_VM_BIND_OP_MAP, .handle = hnew,
						     .addr = 0x62000000, .range = 0x10000 };
		struct drm_nouveau_vm_bind vb = { .op_count = 1, .op_ptr = (uintptr_t)&bo,
						  .flags = DRM_NOUVEAU_VM_BIND_RUN_ASYNC, .wait_count = 1,
						  .wait_ptr = (uintptr_t)&bw, .sig_count = 1, .sig_ptr = (uintptr_t)&bs };
		clock_gettime(CLOCK_MONOTONIC, &t0);
		CHECK(drmCommandWriteRead(fd, DRM_NOUVEAU_VM_BIND, &vb, sizeof(vb)));
		clock_gettime(CLOCK_MONOTONIC, &t1);
		ums = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
		int pending = drmSyncobjWait(fd, &so_s, 1, 0, 0, NULL) != 0;
		printf("  async VM_BIND returned in %.2f ms, its fence %s\n", ums, pending ? "pending (expected)" : "signalled?!");
		okc += ums < 100 && pending;

		/* release the stalled job: it signals so_job; chain so_w to it via a tiny job */
		*(volatile uint32_t *)mgate = 1;
		r = drmSyncobjWait(fd, &so_job, 1, INT64_MAX, 0, NULL);
		printf("  stalled job done: %d, shared object reads 0x%08x\n", r, *(volatile uint32_t *)mshare);
		okc += !r && *(volatile uint32_t *)mshare == 0x51a11ed;
		p = (uint32_t *)((char *)pb + 0x2000);
		uint32_t *st = p;
		mthd(1, 0x1b00, 4, 0x0, 0x60000000, 0x0f, 1u << 28);
		ep.va = 0x10002000;
		ep.va_len = (uint32_t)((char *)p - (char *)st);
		ex.sig_ptr = (uintptr_t)&bw;	/* this job signals the bind's wait */
		CHECK(drmCommandWriteRead(fd, DRM_NOUVEAU_EXEC, &ex, sizeof(ex)));
		r = drmSyncobjWait(fd, &so_s, 1, INT64_MAX, 0, NULL);
		/* use the new mapping */
		p = (uint32_t *)((char *)pb + 0x3000);
		st = p;
		mthd(1, 0x1b00, 4, 0x0, 0x62000000, 0xa5ac0de, 1u << 28);
		ep.va = 0x10003000;
		ep.va_len = (uint32_t)((char *)p - (char *)st);
		ex.sig_ptr = (uintptr_t)&sj;
		CHECK(drmCommandWriteRead(fd, DRM_NOUVEAU_EXEC, &ex, sizeof(ex)));
		int r3 = drmSyncobjWait(fd, &so_job, 1, INT64_MAX, 0, NULL);
		printf("  async bind fence: %d, GPU write through the new mapping: %d, reads 0x%08x\n", r, r3,
		       *(volatile uint32_t *)mnew);
		okc += !r && !r3 && *(volatile uint32_t *)mnew == 0xa5ac0de;
		printf("sync: %d/5 -> %s\n", okc, okc == 5 ? "PASS" : "FAIL");
		return 0;
	}

	/*
	 * nvktest pages: 64 KiB pages and their transitions. A 256 KiB VRAM object (big pages)
	 * gets GPU writes in several chunks; then one 4 KiB page in the middle of a big chunk is
	 * unmapped (that chunk becomes small pages) and refilled with a GART page; GPU writes
	 * must land in VRAM next to the hole and in the GART page inside it.
	 */
	if (argc > 1 && !strcmp(argv[1], "pages")) {
		uint32_t hbig, hsys;
		void *mbig, *msys;
		const uint64_t base = 0x50000000;
		static const uint32_t offs[] = { 0x0, 0x10100, 0x30ff0, 0x11000, 0x12000 };
		int bad = 0;

		CHECK(bo_new(0x40000, NOUVEAU_GEM_DOMAIN_VRAM, &hbig, &mbig));
		CHECK(bo_new(0x10000, NOUVEAU_GEM_DOMAIN_GART, &hsys, &msys));
		CHECK(bind(hbig, base, 0x40000, DRM_NOUVEAU_VM_BIND_OP_MAP));
		for (int round = 0; round < 2; round++) {
			if (round == 1) {	/* punch a 4 KiB hole into big chunk 1 and fill it from GART */
				CHECK(bind(0, base + 0x11000, 0x1000, DRM_NOUVEAU_VM_BIND_OP_UNMAP));
				struct drm_nouveau_vm_bind_op o = { .op = DRM_NOUVEAU_VM_BIND_OP_MAP, .handle = hsys,
								    .addr = base + 0x11000, .bo_offset = 0, .range = 0x1000 };
				struct drm_nouveau_vm_bind b = { .op_count = 1, .op_ptr = (uintptr_t)&o };
				CHECK(drmCommandWriteRead(fd, DRM_NOUVEAU_VM_BIND, &b, sizeof(b)));
			}
			p = pb;
			mthd(1, 0x0000, 1, 0xc7c0);
			for (i = 0; i < 5; i++)
				mthd(1, 0x1b00, 4, (uint32_t)((base + offs[i]) >> 32), (uint32_t)(base + offs[i]),
				     0x70000000u + round * 0x100 + i, 1u << 28);
			ep.va = 0x10000000;
			ep.va_len = (uint32_t)((char *)p - (char *)pb);
			ex.channel = ca.channel;
			ex.wait_count = 0;
			ex.sig_ptr = (uintptr_t)&sig2;
			CHECK(drmCommandWriteRead(fd, DRM_NOUVEAU_EXEC, &ex, sizeof(ex)));
			r = drmSyncobjWait(fd, &so2, 1, INT64_MAX, 0, NULL);
			for (i = 0; i < 5; i++) {
				uint32_t want = 0x70000000u + round * 0x100 + i, got;

				if (round == 1 && offs[i] == 0x11000)	/* the hole: now the GART page */
					got = ((volatile uint32_t *)msys)[0];
				else
					got = ((volatile uint32_t *)mbig)[offs[i] / 4];
				bad += got != want;
				printf("  round %d write +0x%05x: 0x%08x%s\n", round, offs[i], got, got != want ? " (WRONG)" : "");
			}
		}
		printf("pages: wait %d, %d wrong -> %s\n", r, bad, !r && !bad ? "PASS" : "FAIL");
		return 0;
	}

	/*
	 * nvktest overcommit [GiB]: allocate more VRAM than the pool has; nvdrv evicts older
	 * objects to system memory. Every object must keep its contents (CPU check) and stay
	 * usable by the GPU (compute writes into the oldest, evicted one).
	 */
	if (argc > 1 && !strcmp(argv[1], "overcommit")) {
		int nb = argc > 2 ? atoi(argv[2]) : 10, b, badobj = 0;
		uint32_t hb[64];
		volatile uint32_t *mb[64];

		for (b = 0; b < nb && b < 64; b++) {
			void *m;

			CHECK(bo_new(1ull << 30, NOUVEAU_GEM_DOMAIN_VRAM, &hb[b], &m));
			mb[b] = m;
			for (uint64_t o = 0; o < (1ull << 30); o += 64 << 20)	/* a word every 64 MiB */
				mb[b][o / 4] = 0xa0000000u + b * 0x100 + (uint32_t)(o >> 26);
			CHECK(bind(hb[b], 0x100000000ull + (uint64_t)b * (1ull << 30), 1ull << 30, DRM_NOUVEAU_VM_BIND_OP_MAP));
			printf("  object %2d: 1 GiB allocated, mapped at 0x%llx\n", b,
			       0x100000000ull + (unsigned long long)b * (1ull << 30));
		}
		for (b = 0; b < nb && b < 64; b++)
			for (uint64_t o = 0; o < (1ull << 30); o += 64 << 20)
				if (mb[b][o / 4] != 0xa0000000u + b * 0x100 + (uint32_t)(o >> 26)) {
					badobj++;
					break;
				}
		printf("overcommit: %d GiB in %d objects, %d with wrong contents (CPU)\n", nb, nb, badobj);
		/* GPU write into object 0 (evicted first) through its existing mapping */
		p = pb;
		mthd(1, 0x0000, 1, 0xc7c0);
		mthd(1, 0x1b00, 4, 0x1, 0x00000100, 0x600d0bad, 1u << 28);	/* VA 0x1_0000_0100 */
		ep.va = 0x10000000;
		ep.va_len = (uint32_t)((char *)p - (char *)pb);
		ex.channel = ca.channel;
		ex.wait_count = 0;
		ex.sig_ptr = (uintptr_t)&sig2;
		CHECK(drmCommandWriteRead(fd, DRM_NOUVEAU_EXEC, &ex, sizeof(ex)));
		r = drmSyncobjWait(fd, &so2, 1, INT64_MAX, 0, NULL);
		printf("GPU write into object 0: wait %d, reads 0x%08x -> %s\n", r, mb[0][0x100 / 4],
		       !r && mb[0][0x100 / 4] == 0x600d0bad && !badobj ? "PASS" : "FAIL");
		return 0;
	}

	/* transfer queue: a copy-engine channel copies 4 KiB VRAM -> GART (uAPI 1.4.1) */
	{
		struct drm_nouveau_channel_alloc cc = { .fb_ctxdma_handle = ~0u, .tt_ctxdma_handle = NOUVEAU_FIFO_ENGINE_CE };
		uint32_t *vsrc = (uint32_t *)vram + 0x1000 / 4, *gdst = (uint32_t *)gart + 0x8000 / 4, bad = 0, k;

		CHECK(drmCommandWriteRead(fd, DRM_NOUVEAU_CHANNEL_ALLOC, &cc, sizeof(cc)));
		struct { struct nvif_hdr h; struct nvif_sclass s; } cs = {
			.h = { .type = 1, .route = 0xff, .token = cc.channel }, .s = { .count = 16 } };
		CHECK(drmCommandWriteRead(fd, DRM_NOUVEAU_NVIF, &cs, sizeof(cs)));
		printf("copy channel %d, %u classes:", cc.channel, cs.s.count);
		for (i = 0; i < cs.s.count; i++)
			printf(" %04x", cs.s.list[i].oclass);
		printf("\n");
		struct { struct nvif_hdr h; struct nvif_new n; } on = {
			.h = { .type = 2, .route = 0xff, .token = cc.channel }, .n = { .oclass = 0xc7b5 } };
		CHECK(drmCommandWrite(fd, DRM_NOUVEAU_NVIF, &on, sizeof(on)));
		for (k = 0; k < 1024; k++) {
			vsrc[k] = 0x5eed0000u ^ (k * 2654435761u);
			gdst[k] = 0;
		}
		p = (uint32_t *)((char *)pb + 0x8000);
		uint32_t *start = p;
		mthd(4, 0x0000, 1, 0xc7b5);					/* SET_OBJECT */
		mthd(4, 0x0400, 4, 0x0, 0x20001000, 0x0, 0x30008000);		/* OFFSET_IN/OUT_UPPER/LOWER */
		mthd(4, 0x0418, 1, 4096);					/* LINE_LENGTH_IN */
		mthd(4, 0x0300, 1, 0x186);	/* LAUNCH_DMA: non-pipelined, flush, pitch -> pitch */
		ep.va = 0x10008000;
		ep.va_len = (uint32_t)((char *)p - (char *)start);
		uint32_t so3;
		CHECK(drmSyncobjCreate(fd, 0, &so3));
		struct drm_nouveau_sync sig3 = { .flags = DRM_NOUVEAU_SYNC_SYNCOBJ, .handle = so3 };
		ex.channel = cc.channel;
		ex.wait_count = 0;
		ex.sig_ptr = (uintptr_t)&sig3;
		CHECK(drmCommandWriteRead(fd, DRM_NOUVEAU_EXEC, &ex, sizeof(ex)));
		r = drmSyncobjWait(fd, &so3, 1, INT64_MAX, 0, NULL);
		for (k = 0; k < 1024; k++)
			bad += gdst[k] != vsrc[k];
		printf("copy engine: syncobj wait %d, %u/1024 mismatches -> %s\n", r, bad, !r && !bad ? "PASS" : "FAIL");
		struct drm_nouveau_channel_free cf3 = { .channel = cc.channel };
		CHECK(drmCommandWrite(fd, DRM_NOUVEAU_CHANNEL_FREE, &cf3, sizeof(cf3)));
	}

	/*
	 * nvktest fault: a compute semaphore release to an unmapped address. RM recovers the
	 * channel (MMU fault, RC) and sends RC_TRIGGERED asynchronously: exercises the GSP
	 * message interrupt. The channel is dead afterwards; the test ends here.
	 */
	if (argc > 1 && !strcmp(argv[1], "fault")) {
		p = pb;
		mthd(1, 0x1b00, 4, 0x0, 0x70000000 /* never mapped */, 0xbad, 1u << 28);
		ep.va = 0x10000000;
		ep.va_len = (uint32_t)((char *)p - (char *)pb);
		ex.channel = ca.channel;
		ex.wait_count = 0;
		ex.sig_ptr = (uintptr_t)&sig2;
		CHECK(drmCommandWriteRead(fd, DRM_NOUVEAU_EXEC, &ex, sizeof(ex)));
		r = drmSyncobjWait(fd, &so2, 1, INT64_MAX, 0, NULL);
		printf("fault: syncobj wait %d (the fence carries the error)\n", r);
		sleep(1);
		return 0;
	}

	CHECK(bind(0, 0x20000000, 0x10000, DRM_NOUVEAU_VM_BIND_OP_UNMAP));
	struct drm_nouveau_channel_free cf = { .channel = ca.channel };
	CHECK(drmCommandWrite(fd, DRM_NOUVEAU_CHANNEL_FREE, &cf, sizeof(cf)));
	close(fd);
	printf("done\n");
	return 0;
}
