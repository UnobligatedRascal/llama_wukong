# llama_lazarus - Task Tracker

> llama.cpp fork with Kepler sm_37 cuBLAS fixes and KV cache tensor-split support.
> Upstream: ggml-org/llama.cpp (currently 159 commits behind master).

**Last updated:** 2026-09-10

---

## Current Status: KEPLER FIXES APPLIED, NEEDS UPSTREAM SYNC

**Verified working:** Kepler sm_37 cuBLAS path, KV cache quantization with tensor-split (Option A fix).

---

## Completed

### Kepler sm_37 cuBLAS Fixes ✓
- gemm_algo selection: CUBLAS_GEMM_DEFAULT for Kepler (no tensor ops)
- CUBLAS_DEFAULT_MATH instead of TF32 in cublas_handle()
- solve_tri.cu math mode fix
- Kepler batched path: F32 vs F16/BF16 split (fixes nullptr assertion)
- Commit: 93c888df1

### Dimension-based Strides Fix ✓
- Replaced stride multiplication by block_size with dimension-based strides
- Fixes contiguous quantized-to-compute_type conversion
- Commit: 992315e0e

### KV Cache Block Size Fix ✓
- Use KV cache tensor's own block size for split granularity
- Fixes misaligned quantization block boundaries with tensor-split
- Commit: 77657e77f
- See: QUANTIZED_KV_TENSOR_SPLIT_ROTA.md (in llama_wukong)

---

## Active Work

### P0: Sync with ggml-org/llama.cpp Upstream
**Problem:** llama_lazarus is 159 commits behind ggml-org/llama.cpp master.

**Plan:**
- [ ] Add ggml-org/llama.cpp as upstream remote
- [ ] Fetch current master (72797e891)
- [ ] Rebase llama_lazarus master onto upstream master
- [ ] Resolve conflicts in 4 lazarus-specific commits
- [ ] Force-push updated llama_lazarus
- [ ] Verify build and Kepler fixes still work

**Risk:** Conflicts expected in ggml-cuda.cu (Kepler fixes may need updating).

### P0: Verify KV Cache Quantization with Tensor-Split
**Problem:** Reports of f16=q8_0=q4_0 speed — no measurable improvement from quantization.

**Action items:**
- [ ] Add instrumentation logging at KV cache allocation
- [ ] Run nvprof to measure actual memory bandwidth vs KV cache type
- [ ] Profile CUDA kernel times: dequantize vs compute
- [ ] Document findings
- [ ] See: RESEARCH/KV_CACHE_AUDIT.md (in llama_wukong)

---

## System Info

- **GPU:** 8× Tesla K80 (Kepler sm_37, 11GB each)
- **Driver:** 470.256.02, CUDA Runtime: 11.4
- **CUDA Toolkit:** 11.8

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

1. ggml-cuda.cu: gemm_algo selection for Kepler
2. common.cuh: CUBLAS_DEFAULT_MATH instead of TF32
3. solve_tri.cu: cuBLAS math mode fix
4. ggml-cuda.cu: Kepler batched path — F32 vs F16/BF16 split
5. llama-model.cpp: KV cache block size fix for tensor-split

## Key Files
- Main: `ggml/src/ggml-cuda/ggml-cuda.cu`
- Kepler batched path: ~line 1610
- Math mode: `common.cuh` ~line 1506, `solve_tri.cu` ~line 75
- KV cache split: `src/llama-model.cpp` ~line 791

---

*UnobligatedRascal — Making old hardware sing.*
