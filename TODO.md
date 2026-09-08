# Project Llama Lazarus - Kepler sm_37 cuBLAS Fix

## Current Status: READY FOR TESTING

**Latest Fix** (2026-09-01 ~15:36 UTC):
Assertion failure `GGML_ASSERT(to_fp32_src0 != nullptr)` RESOLVED.

**Root Cause**:
In the Kepler cc<500 batched path, code always called `ggml_get_to_fp32_cuda(traits::ggml_type_val)` to convert src0/src1 to FP32 before `cublasSgemmBatched`. When template instantiated with `GGML_TYPE_F32`, this returns nullptr — no F32→F32 conversion function exists.

**Fix Applied** (ggml-cuda.cu ~line 1609):
Split Kepler cc<500 batched path into two branches:

1. `if (cc < 500 && compute_type != GGML_TYPE_F32)`: F16/BF16 compute → convert to FP32, rebuild ptr arrays, call cublasSgemmBatched (same as before)
2. `else if (cc < 500)`: F32 compute → use existing ptrs_src/ptrs_dst directly with cublasSgemmBatched (they already point to FP32 data with correct strides)
3. `else`: Non-Kepler → cublasGemmBatchedEx (unchanged)

**Build**: Clean build completed at ./build/bin/llama-server (15:36 UTC)
**Backup**: <backup-path>/llama_lazarus_20260901/ggml-cuda.cu.fixed

**Next**: Test with multi-GPU concurrent requests (-np 4) on quantized models.

---

## Previous Issues (RESOLVED)

### Issue 1: CUBLAS_STATUS_INVALID_VALUE on cublasSgemmBatched
**Trigger**: Multi-GPU server (-np 4), concurrent requests
**Error**:
```
On entry to SgemmBatched parameter number 13 had an illegal value
CUBLAS_STATUS_INVALID_VALUE
cublasSgemmBatched(...) at ggml-cuda.cu:1613
```
**Root Cause**: Original Kepler batched path called `cublasSgemmBatched` (expects FP32 pointers) but passed pointers from `k_compute_batched_ptrs` which point to FP16 data. cuBLAS reads FP16 bytes as FP32 → garbage → invalid stride param.
**Fix**: Convert FP16/BF16 → FP32 before cublasSgemmBatched (now with F32 path handled separately).

### Issue 2: Assertion fail `GGML_ASSERT(to_fp32_src0 != nullptr)` at ggml-cuda.cu:1627
**Trigger**: F32 compute path hitting the Kepler batched code
**Template**: `ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_F32>()` → traits::ggml_type_val = GGML_TYPE_F32
**Root Cause**: Called `ggml_get_to_fp32_cuda(GGML_TYPE_F32)` → returns nullptr. When compute_type=F32, src0_ptr/src1_ptr are ALREADY FP32 (converted earlier or natively F32).
**Fix**: Added `compute_type != GGML_TYPE_F32` guard; F32 uses ptrs directly with proper FP32 strides.

---

## System Info

- **Server**: NOUGHT (<internal-ip>, Debian/Q4OS)
- **Project Path**: `<project-root>`
- **Build Path**: `<project-root>/build`
- **GPU**: 8× Tesla K80 (Kepler sm_37, 11GB each)
- **Driver**: 470.256.02, CUDA Runtime: 11.4
- **CUDA Toolkit**: 11.8 at `/usr/local/cuda-11.8`

## Build Config
```bash
cd <project-root> && rm -rf build && mkdir build && cd build
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
  -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-rpath,/usr/local/cuda-11.8/targets/x86_64-linux/lib"
make -j$(nproc) llama-server
```

## Existing Patches (All Applied)

1. ggml-cuda.cu (~line 1513): gemm_algo selection for Kepler
   - `const cublasGemmAlgo_t gemm_algo = (cc >= GGML_CUDA_CC_VOLTA) ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT;`
   - Replaced 3× hardcoded `CUBLAS_GEMM_DEFAULT_TENSOR_OP` with `gemm_algo`

2. common.cuh (~line 1506): CUBLAS_DEFAULT_MATH instead of TF32 in cublas_handle()

3. solve_tri.cu (~line 75): Same cuBLAS math mode fix

4. ggml-cuda.cu (~line 1609): Kepler batched path — F32 vs F16/BF16 split (FIXED)

## Key Files
- Main: `ggml/src/ggml-cuda/ggml-cuda.cu`
- Template: `ggml_cuda_mul_mat_cublas_impl<ggml_type T>` at ~line 1395
- Kepler batched path: ~line 1610
- Pointer kernel: `k_compute_batched_ptrs` at ~line 1340
- Traits: `batched_mul_mat_traits<T>` at ~line 1367
- Math mode: `common.cuh` ~line 1506, `solve_tri.cu` ~line 75

---
Last updated: 2026-09-01 15:36 UTC
Status: Build complete, ready for testing.
