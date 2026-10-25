# Project Llama Lazarus — Bug Patterns & Scan Keywords

## Known Bug Categories (Kepler sm_37)

### 1. Tensor Ops / Compute Capability Assumptions
**Pattern**: Code uses Volta+ (sm_70+) tensor ops or TF32 math without checking `cc`.
**Files affected**: ggml-cuda.cu, common.cuh, solve_tri.cu

| File | Issue | Fix |
|------|-------|-----|
| ggml-cuda.cu ~1513 | Hardcoded `CUBLAS_GEMM_DEFAULT_TENSOR_OP` | Conditional: `cc >= VOLTA ? TENSOR_OP : CUBLAS_GEMM_DEFAULT` |
| common.cuh ~1506 | `cublasSetMathMode(CUBLAS_TF32_TENSOR_OP_MATH)` | → `CUBLAS_DEFAULT_MATH` |
| solve_tri.cu ~75 | Same TF32 math mode | → `CUBLAS_DEFAULT_MATH` |

**Scan Keywords**:
- `CUBLAS_GEMM_DEFAULT_TENSOR_OP`
- `CUBLAS_TF32_TENSOR_OP_MATH`
- `cublasSetMathMode`
- `cublasGemmEx` (check gemm_algo param)
- `cublasGemmStridedBatchedEx` (check gemm_algo param)
- `gemm_algo` (ensure not hardcoded to TENSOR_OP)

---

### 2. cuBLAS Batched GEMM FP16/BF16 Support
**Pattern**: Kepler doesn't support FP16/BF16 in `cublasGemmBatchedEx` / `cublasGemmStridedBatchedEx`. Must fall back to FP32 (`cublasSgemmBatched`) for cc<500.

| File | Issue | Fix |
|------|-------|-----|
| ggml-cuda.cu ~1609 | Used `cublasSgemmBatched` with FP16 pointers → INVALID_VALUE | Convert FP16→FP32 first |
| ggml-cuda.cu ~1627 | Called `ggml_get_to_fp32_cuda(GGML_TYPE_F32)` → nullptr | Check `compute_type != GGML_TYPE_F32` before conversion |

**Scan Keywords**:
- `cublasSgemmBatched`
- `cublasGemmBatchedEx`
- `cublasGemmStridedBatchedEx`
- `cublasHgemmBatched` (FP16 variant — Kepler may not support)
- `batched_mul_mat_traits`
- `k_compute_batched_ptrs`
- `ggml_get_to_fp32_cuda` (check for missing nullptr guards on F32 type)
- `GGML_ASSERT.*to_fp32.*nullptr`

---

### 3. FP16 Hardware Capability Assumptions
**Pattern**: Code assumes FP16 math is "fast" or available. Kepler has hardware FP16 but it's slow (emulated via FP32); pre-Kepler has none.

| File | Issue | Fix |
|------|-------|-----|
| ggml-cuda.cu | `fast_fp16_hardware_available()` gates compute_type selection | Already correct; verify not bypassed |

**Scan Keywords**:
- `fast_fp16_hardware_available`
- `GGML_TYPE_F16` in compute_type selection (check for cc guards)
- `CUBLAS_COMPUTE_16F`
- `CUDA_R_16F`, `CUDA_R_16BF`, `CUDA_R_8F_*`

---

### 4. CUDA API Version / Arch Requirements
**Pattern**: Uses CUDA APIs or features requiring newer compute capability than Kepler (sm_37).

**Scan Keywords**:
- `cooperative_groups` (Pascal+ only)
- `__shfl_sync`, `__shfl_up_sync`, `__shfl_down_sync`, `__ballot_sync` (Fermi+ with mask)
- `__popc`, `__clz` (Kepler OK, but check usage patterns)
- `atomicCAS` on non-32-bit types
- `__ldg()` (Kepler OK)
- `cudaGraph*` (disabled for Kepler via `-DGGML_CUDA_GRAPHS=OFF`)

---

### 5. Shared Memory / Block Config Assumptions
**Pattern**: Kernels assume large shared memory or specific warp sizes not available on Kepler.

**Scan Keywords**:
- `sharedMemPerBlock` usage (Kepler K80: 48KB/block)
- `sharedMemPerBlockOptin`
- `block_dims` or grid/block launch configs > 1024 threads
- `__shared__` array sizes > 48KB

---

### 6. cuBLAS Handle / Context Setup
**Pattern**: cuBLAS handle initialized with modes unsupported on Kepler.

**Scan Keywords**:
- `cublasCreate` followed by `cublasSetMathMode`
- `cublasSetPointerMode`
- `cublasSetStream`

---

## Scan Script (for new llama.cpp builds)

```bash
#!/bin/bash
# Run from llama.cpp root to find Kepler-risk patterns

echo "=== cuBLAS Tensor Op / TF32 patterns ==="
grep -rn "CUBLAS_GEMM_DEFAULT_TENSOR_OP" ggml/src/ggml-cuda/
grep -rn "CUBLAS_TF32_TENSOR_OP_MATH" ggml/src/ggml-cuda/

echo "=== cuBLAS batched GEMM patterns ==="
grep -rn "cublasSgemmBatched\|cublasHgemmBatched\|cublasGemmBatchedEx\|cublasGemmStridedBatchedEx" ggml/src/ggml-cuda/

echo "=== FP32 conversion assertions (potential nullptr issues) ==="
grep -rn "GGML_ASSERT.*to_fp32.*nullptr" ggml/src/ggml-cuda/
grep -rn "ggml_get_to_fp32_cuda" ggml/src/ggml-cuda/

echo "=== FP16 compute type selections ==="
grep -rn "GGML_TYPE_F16.*compute_type\|fast_fp16_hardware_available" ggml/src/ggml-cuda/

echo "=== Cooperative groups (Pascal+ only) ==="
grep -rn "cooperative_groups" ggml/src/ggml-cuda/

echo "=== cuBLAS math mode settings ==="
grep -rn "cublasSetMathMode" ggml/src/ggml-cuda/
```

---

## Patch Application Checklist

When building a new llama.cpp version for Kepler:

- [ ] Scan for `CUBLAS_GEMM_DEFAULT_TENSOR_OP` → add `gemm_algo` conditional
- [ ] Scan for `CUBLAS_TF32_TENSOR_OP_MATH` → replace with `CUBLAS_DEFAULT_MATH`
- [ ] Scan for `cublasGemmBatchedEx`/`cublasGemmStridedBatchedEx` in batched paths → add Kepler cc<500 fallback
- [ ] Any `ggml_get_to_fp32_cuda()` + `GGML_ASSERT` → verify F32 type is handled (no conversion needed)
- [ ] Verify `-DGGML_CUDA_GRAPHS=OFF` in CMake
- [ ] Verify `-DCMAKE_CUDA_ARCHITECTURES="37"` (no higher archs)
- [ ] Verify `-DGGML_CUDA_CUBLAS=ON` (cuBLAS path handles legacy better than custom kernels)
- [ ] Verify `-DGGML_CUDA_FORCE_MMQ=ON` (uses quantization kernels that support sm_37)

---
Generated: 2026-09-01
Project: Llama Lazarus (Kepler sm_37 compatibility)
