# nvcuda: libcuda.so.1 (CUDA Driver API) on nvdrv's nouveau uAPI
# Needs Mesa's NAK compiler built first: mesa/build-nak.sh (writes mesa/mesa.mk).
CC ?= cc
CFLAGS ?= -O2 -g
CFLAGS += -std=gnu11 -D_GNU_SOURCE -fPIC -fvisibility=hidden -Wall -Wextra -Wno-unused-parameter \
	  -Iinclude -I../uapi $(shell pkg-config --cflags libdrm)
LIBS = $(shell pkg-config --libs libdrm) -lpthread

-include mesa/mesa.mk
ifeq ($(MESA_LIBS),)
$(error Mesa's NAK is not built: run mesa/build-nak.sh first)
endif

MESA_SRCS = src/compile.c src/ptx.c
SRCS = $(filter-out $(MESA_SRCS),$(wildcard src/*.c))
OBJS = $(SRCS:src/%.c=build/%.o) $(MESA_SRCS:src/%.c=build/%.o)

all: build/libcuda.so.1 build/cudatest build/nakc build/kerneltest build/kernels.spv build/kernels.ptx build/ptxops.ptx build/ptxtest

build/%.o: src/%.c src/internal.h src/compile.h include/cuda.h include/cuda_enums.h
	@mkdir -p build
	$(CC) $(CFLAGS) -c -o $@ $<

# the compiler and the PTX front end see Mesa's headers (its own defines first, then nvcuda's)
build/compile.o build/ptx.o: build/%.o: src/%.c src/compile.h src/ptx.h
	@mkdir -p build
	$(CC) -O2 -g -std=gnu11 -fPIC -fvisibility=hidden -Wall $(MESA_CFLAGS) -Iinclude -c -o $@ $<

build/libcuda.so.1: $(OBJS) src/exports.map
	$(CC) -shared -Wl,-soname,libcuda.so.1 -Wl,--no-undefined -Wl,--version-script=src/exports.map \
		-Wl,--exclude-libs,ALL -Wl,--gc-sections -o $@ $(OBJS) \
		-Wl,--start-group $(MESA_LIBS) -Wl,--end-group $(MESA_SYSLIBS) $(LIBS)
	ln -sf libcuda.so.1 build/libcuda.so

# the test links against libcuda.so like any CUDA program; run with LD_LIBRARY_PATH=build
build/cudatest: tests/cudatest.c build/libcuda.so.1
	$(CC) -O2 -g -Wall -Iinclude -o $@ $< -Lbuild -lcuda

# offline compiler check: build/nakc file.spv entry...
build/nakc: tests/nakc.c build/compile.o build/ptx.o
	$(CC) -O2 -g -o $@ $< build/compile.o build/ptx.o -Wl,--start-group $(MESA_LIBS) -Wl,--end-group $(MESA_SYSLIBS)

# CUDA C++ -> SPIR-V with clang (no NVIDIA SDK): include/nvcuda/device.h supplies threadIdx & co.
build/kernels.spv: tests/kernels.cu include/nvcuda/device.h
	@mkdir -p build
	clang -x cuda --cuda-device-only -nocudainc -nocudalib --offload=spirv64-unknown-unknown \
		-Iinclude -include nvcuda/device.h -O2 -S -emit-llvm -o build/kernels.ll $<
	llc -mtriple=spirv64-unknown-unknown -filetype=obj -o $@ build/kernels.ll
	spirv-val $@

# the same kernels as PTX, through clang's NVPTX back end (input of the PTX front end)
build/kernels.ptx: tests/kernels.cu include/nvcuda/device.h
	@mkdir -p build
	clang -x cuda --cuda-device-only -nocudainc -nocudalib --cuda-gpu-arch=sm_86 -Xclang -target-feature -Xclang +ptx75 \
		-Iinclude -include nvcuda/device.h -O2 -S -o $@ $<

build/ptxops.ptx: tests/ptxops.cu
	@mkdir -p build
	clang -x cuda --cuda-device-only -nocudainc -nocudalib --cuda-gpu-arch=sm_86 -Xclang -target-feature -Xclang +ptx75 \
		-Iinclude -include nvcuda/device.h -O2 -S -o $@ $<

build/ptxtest: tests/ptxtest.c build/libcuda.so.1
	$(CC) -O2 -g -Wall -Iinclude -o $@ $< -Lbuild -lcuda -lm

build/kerneltest: tests/kerneltest.c build/libcuda.so.1
	$(CC) -O2 -g -Wall -Iinclude -o $@ $< -Lbuild -lcuda

clean:
	rm -rf build

.PHONY: all clean
