#!/bin/bash

export CXXFLAGS="-std=c++11"
export CFLAGS="-std=c99"

TORCH=$(python -c "import os; import torch; print(os.path.dirname(torch.__file__))")

rm -rf ../_ext
rm -rf ../__pycache__
rm -rf src/cudaCG_gpu.o
cp src/cudaCG_gpu.c src/cudaCG_gpu.cu

nvcc -c -o src/cudaCG_gpu.o src/cudaCG_gpu.cu --gpu-architecture=compute_52 --gpu-code=compute_52 --compiler-options -fPIC -I ${TORCH}/lib/include/TH -I ${TORCH}/lib/include/THC -I /usr/include/python2.7

python builder.py

cp -rf ./_ext ../_ext