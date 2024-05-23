# Makefile

# Compiler and compiler flags
NVCC = nvcc
GCC = gcc
CFLAGS = -lm

# Targets
CUDA_TARGET = cuda_main
SEQ_TARGET = seq_main

# Source files
CUDA_SOURCE = cuda_main.cu
SEQ_SOURCE = seq_main.c

# Rules
all: $(CUDA_TARGET) $(SEQ_TARGET)

$(CUDA_TARGET): $(CUDA_SOURCE)
	$(NVCC) -Xcompiler -fopenmp -o $@ $^ $(CFLAGS)

$(SEQ_TARGET): $(SEQ_SOURCE)
	$(GCC) -o $@ $^ $(CFLAGS)

clean:
	rm -f $(CUDA_TARGET) $(SEQ_TARGET)

.PHONY: all clean

