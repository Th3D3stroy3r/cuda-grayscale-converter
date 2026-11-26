NVCC = nvcc
NVCC_FLAGS = -std=c++17 -O2 -I./src -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart
TARGET = bin/grayscale.exe
SRC = src/grayscale.cu

.PHONY: all build run clean

all: build

build: $(SRC)
	mkdir -p bin
	$(NVCC) $(NVCC_FLAGS) $(SRC) -o $(TARGET)

run: build
	./$(TARGET) --input data/input/sample_00.ppm --output data/output/sample_00_gray.pgm

clean:
	rm -rf bin/* data/output/* logs/*
