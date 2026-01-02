# Makefile for compiling C files with common HPC flags
# To use: make TARGET=filename (without .c extension)
# Example: make TARGET=sum_loop

# Compiler (change to clang if preferred)
CC = gcc
# CC = clang

# Common optimization flags (uncomment/comment to add/remove)
OFLAG = -O3
MARCH = -march=native
# MTUNE = -mtune=native
FVEC = -ftree-vectorize
FUNROLL = -funroll-loops

# Diagnostic flags (for gcc)
FOPT_VEC = -fopt-info-vec-all
# FOPT_VEC = -fopt-info-vec-optimized -fopt-info-vec-missed -fopt-info-loop-optimized -fopt-info-loop-missed

# For clang, use these instead:
# FOPT_VEC = -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize

# Defines
# DDTYPE = -DDTYPE=FPDP
# DTYPE = -DDTYPE=INTEGER

# Other flags
OPENMP = -fopenmp
# OPENMP = -fopenmp-simd
LM = -lm
DEBUG = -g
# DEBUG = -DDEBUG

# Assembly output (uncomment to generate .s instead of executable)
# ASM = -S -fverbose-asm
# ASM_MASM = -masm=intel

# Combine flags
CFLAGS = $(OFLAG) $(MARCH) $(MTUNE) $(FVEC) $(FUNROLL) $(FOPT_VEC) $(DDTYPE) $(OPENMP) $(DEBUG) $(ASM) $(ASM_MASM)
LDFLAGS = $(LM)

# Default target
TARGET ?= default

$(TARGET): $(TARGET).c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $<

# Clean target
clean:
	rm -f $(TARGET)

.PHONY: clean
