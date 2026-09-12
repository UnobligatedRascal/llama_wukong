# llama.cpp for NVIDIA Kepler (sm_35/sm_37)

> This fork enables llama.cpp on NVIDIA Kepler GPUs (Tesla K80, K40, K20) that were dropped when CUDA 12 removed compute capability 3.x support.

## What This Fixes

Official llama.cpp fails on Kepler with:

- `CUBLAS_STATUS_ARCH_MISMATCH` - tensor ops / TF32 math used on hardware without support
- `CUBLAS_STATUS_INVALID_VALUE` - FP16 pointers passed to FP32 cuBLAS functions
- `GGML_ASSERT(to_fp32_src0 != nullptr)` - F32 type passed to conversion function
- Build failures - sm_37 never compiled into binary

All resolved. Multi-GPU tensor parallelism works.

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

## Tested Hardware

| GPU | Chip | CC | Status |
|-----|------|----|--------|
| Tesla K80 | GK210 | sm_37 | Tested, working with -np 4 |
| Tesla K40 | GK110B | sm_35 | Compiled, not tested |
| Tesla K20 | GK110 | sm_35 | Compiled, not tested |

## Requirements

- CUDA Toolkit 11.x (11.0-11.8) - Kepler support removed in CUDA 12
- Driver 470.x or later
- GCC 9 or 10 (GCC 11+ not supported by CUDA 11.x)
- CMake 3.18+
- Linux x86_64

## Build

```bash
git clone https://github.com/UnobligatedRascal/llama_lazarus
cd llama_lazarus
mkdir build && cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_F16=ON \
  -DGGML_CUDA_CUBLAS=ON \
  -DGGML_CUDA_FORCE_MMQ=ON \
  -DGGML_CUDA_GRAPHS=OFF \
  -DGGML_CUDA_NCCL=ON \
  -DCMAKE_CUDA_ARCHITECTURES="37" \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.8/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=g++-11 \
  -DCMAKE_C_COMPILER=gcc-11 \
  -DCMAKE_CXX_COMPILER=g++-11 \
  -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-rpath,/usr/local/cuda-11.8/targets/x86_64-linux/lib"

make -j$(nproc) llama-server
```

For Tesla K40/K20 (sm_35), replace `37` with `35` above.

## Usage

Single GPU:

```bash
./bin/llama-server --model model.gguf -ngl 999 -c 4096
```

Dual K80 (2 GPUs on one card):

```bash
./bin/llama-server --model model.gguf -ngl 999 -c 4096 --tensor-split 1,1
```

Multi-GPU with parallelism (4+ K80s):

```bash
./bin/llama-server --model model.gguf -ngl 999 -c 4096 --tensor-split 1,1,1,1 -np 4
```

## What Changed (From Upstream)

Three fixes in `ggml/src/ggml-cuda/`:

1. **ggml-cuda.cu** - cuBLAS GEMM algorithm selection: use `CUBLAS_GEMM_DEFAULT` instead of `CUBLAS_GEMM_DEFAULT_TENSOR_OP` on cc < Volta. Batched GEMM paths use `cublasSgemmBatched`/`cublasSgemmStridedBatched` (FP32) instead of Ex variants on cc < Kepler, with proper FP16->FP32 conversion. Strides corrected for contiguous quantized-to-compute_type conversion.

2. **common.cuh** - cuBLAS handle initialized with `CUBLAS_DEFAULT_MATH` instead of `CUBLAS_TF32_TENSOR_OP_MATH`.

3. **solve_tri.cu** - Same cuBLAS math mode fix on handle restore.

See `docs/kepler/` for full technical details.

## Performance Notes

- Kepler is memory-bandwidth bound (240 GB/s per GK210), not compute-bound
- Expect 4-6 tok/s on 14B models, 10-20 tok/s on <5B models (Q4_K_M quantization)
- No Tensor Cores, no BF16, no Flash Attention on this architecture
- Multi-GPU tensor split helps fit larger models but does not increase generation throughput significantly
- Independent llama-server processes per GPU may be preferable to aggressive model splitting for some workloads

## Patches Needed for Upstream Builds

When rebasing to a new llama.cpp version, scan for:

- Hardcoded `CUBLAS_GEMM_DEFAULT_TENSOR_OP`
- `CUBLAS_TF32_TENSOR_OP_MATH`
- `cublasGemmBatchedEx` / `cublasGemmStridedBatchedEx` without cc guards
- `ggml_get_to_fp32_cuda()` calls without F32 type handling

See `docs/kepler/SCAN_KEYWORDS.md` for the complete scanner.

## License

MIT License (inherited from upstream llama.cpp)

This is a fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp). Upstream authors retain all rights to their original code.
