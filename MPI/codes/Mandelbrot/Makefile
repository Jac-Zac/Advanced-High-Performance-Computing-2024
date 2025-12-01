# Makefile for MPI Mandelbrot demonstrator
#
# Targets:
#   all       - Build the unified executable (all backends compiled in)
#   clean     - Remove build artifacts
#   test      - Run basic tests with each backend
#   benchmark - Run timing comparison across backends
#
# The executable contains all backends; select at runtime with:
#   mpirun -np 4 ./mandelbrot_mpi lock
#   mpirun -np 4 ./mandelbrot_mpi shmem

CC = mpicc
CFLAGS = -Wall -Wextra -O3 -std=c11 -march=native -mtune=native
LDFLAGS = -lm

# Source files
CORE_SRC = mandelbrot_core.c
BACKEND_SRC = backend_lock.c backend_shmem.c
MAIN_SRC = mandelbrot_main.c

# Object files
CORE_OBJ = $(CORE_SRC:.c=.o)
BACKEND_OBJ = $(BACKEND_SRC:.c=.o)
MAIN_OBJ = $(MAIN_SRC:.c=.o)

ALL_OBJ = $(CORE_OBJ) $(BACKEND_OBJ) $(MAIN_OBJ)

# Target executable
TARGET = mandelbrot_mpi

.PHONY: all clean test benchmark help

all: $(TARGET)

$(TARGET): $(ALL_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c mandelbrot_common.h
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(ALL_OBJ) $(TARGET) *.pgm

# Quick functionality test with each backend
test: $(TARGET)
	@echo "=== Testing lock backend ==="
	mpirun --oversubscribe -np 4 ./$(TARGET) lock -w 256 -h 256 -p 32 -o test_lock.pgm
	@echo ""
	@echo "=== Testing shmem backend ==="
	mpirun --oversubscribe -np 4 ./$(TARGET) shmem -w 256 -h 256 -p 32 -o test_shmem.pgm
	@echo ""
	@echo "=== Comparing outputs ==="
	@if cmp -s test_lock.pgm && cmp -s test_shmem.pgm; then \
		echo "All backends produce identical output. SUCCESS!"; \
	else \
		echo "WARNING: Output files differ!"; \
	fi

# Benchmark comparison
benchmark: $(TARGET)
	@echo "=== Benchmark: 1024x1024, 1000 iterations ==="
	@echo ""
	@for backend in lock shmem; do \
		echo "--- Backend: $$backend ---"; \
		mpirun --oversubscribe -np 4 ./$(TARGET) $$backend -w 1024 -h 1024 -i 1000 -p 64 -o bench_$$backend.pgm 2>&1 | grep -E "(time|patches)"; \
		echo ""; \
	done

help:
	@echo "MPI Mandelbrot Set Calculator - Build Targets"
	@echo ""
	@echo "  make all       - Build the unified executable"
	@echo "  make clean     - Remove build artifacts"
	@echo "  make test      - Run functionality tests"
	@echo "  make benchmark - Run timing comparison"
	@echo ""
	@echo "Usage:"
	@echo "  mpirun -np N ./mandelbrot_mpi [backend] [options]"
	@echo ""
	@echo "Backends: lock (default), shmem"
	@echo ""
	@echo "Options:"
	@echo "  -w WIDTH      Image width (default: 1024)"
	@echo "  -h HEIGHT     Image height (default: 1024)"  
	@echo "  -i MAXITER    Max iterations (default: 1000)"
	@echo "  -p PATCHSIZE  Initial patch size (default: 64)"
	@echo "  -o FILENAME   Output file (default: mandelbrot.pgm)"
