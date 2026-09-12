# TurboQuant sm_37 (K80) Compatibility Audit

**Date:** 2026-09-08  
**Auditor:** Pi (llama_wukong project)  
**Target:** 8x Tesla K80, GK210, sm_37, 12GB VRAM each

## Executive Summary

**VERDICT: TurboQuant is sm_37 COMPATIBLE with minimal changes.**

All TurboQuant CUDA kernels use FP32 arithmetic. No tensor cores, no half-precision math intrinsics, no sm_75+ features required. The only half-precision usage is `__half2float()` for reading stored fp16 norm scalars from block headers — this intrinsic has been available since CUDA 5.0 and runs fine on Kepler.

## Detailed Audit

### 1. turbo-quant.cuh (17KB)

**Status: ✅ CLEAN**

| Feature | Usage | sm_37 Compatible? |
|---------|-------|-------------------|
| FP16 math (`__hadd2`, etc.) | NONE | N/A |
| Tensor cores (WMMA) | NONE | N/A |
| `__half` / `__half2` computation | NONE (only `__half2float()` for reading stored norms) | ✅ |
| `__constant__` memory | Yes (centroid tables, WHT signs) | ✅ (4MB available on K80) |
| Async copy (`cp.async`) | NONE | N/A |
| Warp shuffle (`__shfl_sync`) | NONE | N/A |
| `if constexpr` (C++17) | Yes | ✅ (nvcc 11.8 supports) |
| Math functions | sqrtf, powf, cosf, sinf | ✅ All available since CUDA 3.0 |

**Centroid tables in `__constant__` memory:**
- TURBO_CENTROIDS_2BIT[4] = 16 bytes
- TURBO_CENTROIDS_3BIT[8] = 32 bytes
- TURBO_CENTROIDS_4BIT[16] = 64 bytes
- TURBO_WHT_SIGNS1[128] = 512 bytes
- TURBO_WHT_SIGNS2[128] = 512 bytes
- Total: ~1.1KB — trivial vs 4MB K80 limit

**FWHT implementation:** Pure add/subtract butterfly network. O(n log n). Zero precision concerns.

### 2. triattention-score.cu (542 lines)

**Status: ✅ CLEAN**

| Feature | Usage | sm_37 Compatible? |
|---------|-------|-------------------|
| Shared memory | `(hd + fc) * sizeof(float)` | ✅ (~640B for hd=128, fc=32; K80 has 48KB/block) |
| Block size | `dim3(fc, 1, 1)` | ✅ (fc typically 32-64; K80 supports 1024) |
| `extern __shared__` | Yes | ✅ Fully supported on Kepler |
| `__syncthreads()` | Yes | ✅ Standard since CUDA 1.0 |
| Math functions | cosf, sinf, sqrtf, atan2f, fmaxf | ✅ All available |
| FP16 math | NONE | N/A |
| Tensor cores | NONE | N/A |

**Shared memory budget:**
- K vector buffer: head_dim × 4 bytes = 512B (for head_dim=128)
- Score reduction: freq_count × 4 bytes = 128B (for fc=32)
- Total: ~640B per block — well within 48KB limit

### 3. Block Type Structures (ggml-common.h)

**Status: ✅ COMPATIBLE**

```c
// turbo3_0: 14 bytes per 128 elements (3.5 bpw, 4.6× compression)
typedef struct {
    ggml_half  norm;           // 2 bytes — stored, not computed with
    uint8_t    qs[32];         // 32 bytes — lower 2-bit indices
    uint8_t    signs[16];      // 16 bytes — upper 1-bit
} block_turbo3_0;

// turbo4_0: 68 bytes per 128 elements (4.25 bpw, 3.8× compression)
typedef struct {
    ggml_half  norm;           // 2 bytes
    ggml_half  rnorm;          // 2 bytes (reserved)
    uint8_t    qs[64];         // 64 bytes — 4-bit indices
} block_turbo4_0;

// turbo2_0: 10 bytes per 128 elements (2.5 bpw, 6.4× compression)
typedef struct {
    ggml_half  norm;           // 2 bytes
    uint8_t    qs[32];         // 32 bytes — 2-bit indices
} block_turbo2_0;
```

Note: `ggml_half` is used ONLY for storage (2 bytes). Dequantization converts to float immediately via `__half2float()`. No FP16 arithmetic.

### 4. ggml-turbo-quant.c (CPU reference)

**Status: ✅ N/A** — Pure CPU code, compiled with gcc. Irrelevant for GPU compatibility.

## Potential Issues & Mitigations

### Issue 1: `__half2float()` on sm_37
**Risk:** LOW  
`__half2float()` is a standard CUDA intrinsic since CUDA 5.0. Kepler fully supports it. The K80 uses it for all existing quantized formats (Q4_0, Q5_0, etc.) that store scales in fp16.

### Issue 2: `if constexpr` (C++17)
**Risk:** LOW  
nvcc 11.8 supports C++17 for all compute capabilities including sm_37. The fork already compiles with C++17.

### Issue 3: Register Pressure
**Risk:** MEDIUM  
The FWHT_128 function uses local arrays and multiple loop iterations. Could pressure registers on Kepler (64 per thread max for good occupancy).  
**Mitigation:** Use `__launch_bounds__` if needed. Profile with `nvcc --ptxas-options=-v` to check register usage.

### Issue 4: Shared Memory Bank Conflicts
**Risk:** LOW  
The triattention kernel uses shared memory for K vectors. With head_dim=128 and 32-64 threads, there could be some bank conflicts.  
**Mitigation:** Acceptable for first pass. Optimize with padding if profiling shows it's a bottleneck.

## Integration Plan

### Files to Port from llama-cpp-turboquant:

1. **Headers:**
   - `ggml/include/ggml.h` — Add GGML_TYPE_TURBO3_0(41), TURBO4_0(42), TURBO2_0(43)
   - `ggml/src/ggml-common.h` — Add QK_TURBO* macros, block_turbo* structs

2. **CUDA kernels:**
   - `ggml/src/ggml-cuda/turbo-quant.cuh` — Core quantization kernels
   - `ggml/src/ggml-cuda/turbo-innerq.cuh` — InnerQ header
   - `ggml/src/ggml-cuda/turbo-innerq.cu` — InnerQ implementation
   - `ggml/src/ggml-cuda/turbo-wht.cuh` — WHT transform header
   - `ggml/src/ggml-cuda/turbo-wht.cu` — WHT transform implementation
   - `ggml/src/ggml-cuda/triattention-score.cuh` — TriAttention header
   - `ggml/src/ggml-cuda/triattention-score.cu` — TriAttention implementation

3. **CPU code:**
   - `ggml/src/ggml-turbo-quant.c` — CPU reference quant/dequant

4. **Build system:**
   - `ggml/src/ggml-cuda/CMakeLists.txt` — Add turbo source files
   - `ggml/src/ggml.c` — Add type info for TURBO types

5. **KV cache integration:**
   - `ggml/src/ggml-cuda/set-rows.cu` — Add turbo quantization path for KV writes
   - `ggml/src/ggml-cuda/dequantize.cuh` — Add turbo dequant functions

6. **FlashAttention templates:**
   - `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-*-turbo*.cu`

### sm_37-Specific Build Flags:

```cmake
-DCMAKE_CUDA_ARCHITECTURES="37"
-DGGML_CUDA_TURBOQUANT=ON      # New flag to enable TurboQuant
```

### Testing Strategy:

1. Build with TurboQuant enabled, sm_37 target
2. Run CPU-only quantize/dequant roundtrip test
3. Run GPU set_rows with turbo3_0 cache type
4. Verify attention output matches fp16 baseline
5. Benchmark KV cache memory savings

## Conclusion

TurboQuant is an excellent fit for the K80 cluster. The FP32-first design means it runs natively on Kepler without any hacks. Expected benefits:

- **turbo4_0 KV cache:** 3.8× compression → ~1/4 the VRAM for KV cache
- **turbo3_0 KV cache:** 4.6× compression → ~1/5 the VRAM
- **turbo2_0 KV cache:** 6.4× compression → ~1/7 the VRAM (quality tradeoff)

This directly addresses our primary bottleneck: 12GB VRAM per GPU limiting context length and batch size.

---
*UnobligatedRascal — Making old hardware sing.*
