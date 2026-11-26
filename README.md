# CUDA GPU-Accelerated Image Grayscale Converter

## Overview
This project demonstrates GPU-accelerated image processing using CUDA and C++. The application converts color images into grayscale images using a CUDA kernel and parallel GPU execution.

The project highlights:
- CUDA kernel execution
- GPU memory allocation and transfer
- Parallel image processing
- Command-line argument support and automated build/run scripts

## Project Structure
- `bin/`: Contains compiled executable files (`./bin/grayscale.exe`).
- `data/`: Contains input and output image files used for processing.
- `lib/`: Contains external libraries if needed.
- `src/`: Contains CUDA C++ source code files (`src/grayscale.cu`).
- `Makefile`: Builds the executable into `bin/grayscale.exe`.
- `run.sh`: Automates building and running the grayscale converter.
- `logs/`: Stores execution proof logs.

## Technologies Used
- CUDA
- C++
- NVIDIA GPU Computing

## Build Instructions
Run the following command:
```bash
make
```

## Run Instructions
Execute the program using:
```bash
./bin/grayscale.exe --input data/input/sample_00.ppm --output data/output/sample_00_gray.pgm
```
Or run the automated script:
```bash
./run.sh
```
