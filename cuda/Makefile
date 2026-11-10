# nvcuda: libcuda.so.1 (CUDA Driver API) on nvdrv's nouveau uAPI
CC ?= cc
CFLAGS ?= -O2 -g
CFLAGS += -std=gnu11 -D_GNU_SOURCE -fPIC -fvisibility=hidden -Wall -Wextra -Wno-unused-parameter \
	  -Iinclude $(shell pkg-config --cflags libdrm)
LIBS = $(shell pkg-config --libs libdrm) -lpthread
SRCS = $(wildcard src/*.c)
OBJS = $(SRCS:src/%.c=build/%.o)

all: build/libcuda.so.1 build/cudatest

build/%.o: src/%.c src/internal.h include/cuda.h include/cuda_enums.h
	@mkdir -p build
	$(CC) $(CFLAGS) -c -o $@ $<

build/libcuda.so.1: $(OBJS)
	$(CC) -shared -Wl,-soname,libcuda.so.1 -Wl,--no-undefined -o $@ $(OBJS) $(LIBS)
	ln -sf libcuda.so.1 build/libcuda.so

# the test links against libcuda.so like any CUDA program; run with LD_LIBRARY_PATH=build
build/cudatest: tests/cudatest.c build/libcuda.so.1
	$(CC) -O2 -g -Wall -Iinclude -o $@ $< -Lbuild -lcuda

clean:
	rm -rf build

.PHONY: all clean
