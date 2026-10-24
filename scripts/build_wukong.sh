#!/bin/bash
# llama_wukong build script for NOUGHT (Kepler sm_37)
# Run from: ${LLAMA_WUKONG_HOME}

set -e

# Use environment variables for paths - set LLAMA_WUKONG_HOME if needed
LLAMA_WUKONG_HOME="${LLAMA_WUKONG_HOME:-${LLAMA_WUKONG_HOME}}"

cd ${LLAMA_WUKONG_HOME}
rm -rf build && mkdir build && cd build

echo "Configuring llama_wukong for Kepler sm_37..."

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_F16=ON \
  -DCMAKE_CUDA_HOST_COMPILER=g++-11 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.8/bin/nvcc \
  -DGGML_CUDA_NCCL=ON \
  -DCMAKE_CUDA_ARCHITECTURES="37" \
  -DLLAMA_CURL=OFF \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DGGML_CUDA_FORCE_MMQ=ON \
  -DGGML_CUDA_GRAPHS=OFF \
  -DCMAKE_C_COMPILER=gcc-11 \
  -DCMAKE_CXX_COMPILER=g++-11 \
  -DGGML_CUDA_CUBLAS=ON \
  -DCMAKE_C_FLAGS="-DGGML_NUMA_REPLICATE" \
  -DCMAKE_CXX_FLAGS="-DGGML_NUMA_REPLICATE" \
  -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-rpath,/usr/local/cuda-11.8/targets/x86_64-linux/lib"

echo "Building (36 cores)..."
make -j36 llama-server llama-bench

echo "Build complete. Binaries in build/bin/"
ls -la bin/llama-server bin/llama-bench
