#!/usr/bin/env bash
set -e

mkdir -p bin data/input data/output logs
make build
./bin/grayscale.exe --input data/input/sample_00.ppm --output data/output/sample_00_gray.pgm | tee logs/execution_proof.log
