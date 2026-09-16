# Llama Lazarus: IMAD + LUT-GEMM Research Report
## Kepler sm_37 Quantized Inference Supercharging

Date: 2026-09-02

---

## Executive Summary

**IMAD**: Theoretically viable but practically irrelevant — K80 is memory-bound, not compute-bound.

**LUT-GEMM**: Not viable for Q4_K_M — benefits non-uniform quantization (NF4, learned tables), not ggml's uniform scheme.

**Recommendation**: Focus on memory bandwidth optimization, not integer arithmetic micro-optimization.

---

## Current Execution Path (Verified)

With `-DGGML_CUDA_FORCE_MMQ=ON` and `-DCMAKE_CUDA_ARCHITECTURES="37"`:

1. Q4_K_M matmul → `ggml_cuda_mul_mat_q()` (MMQ path)
2. Config: `mmq-config-pascal-older.cuh` (no tensor cores, no DP4A)
3. Inner dot product: `ggml_cuda_dp4a()` in `common.cuh:731-738`

The dp4a fallback for sm_37 (< GGML_CUDA_CC_DP4A = 610):
```cpp
// common.cuh:736-738
const int8_t * a8 = (const int8_t *) &a;
const int8_t * b8 = (const int8_t *) &b;
return c + a8[0]*b8[0] + a8[1]*b8[1] + a8[2]*b8[2] + a8[3]*b8[3];
```

This is plain C integer multiply-add — no hardware DP4A acceleration.

---

## IMAD on Kepler sm_37

### Architecture Facts

- Kepler integer ALU: 1:1 performance ratio with FP32 ALU (confirmed via CGBN project)
- IMAD instruction exists: `imad.u32.u32 d, a, b, c` → d = a*b + c (single instruction)
- 1 IMAD cycle vs ~2-4 cycles for separate mul+add sequence

### Can IMAD Help?

**Theoretical benefit**: Yes. Replacing `c + a*b` with IMAD saves:
- 1 register (no intermediate product)
- 1 instruction per multiply-add pair
- Potential for better warp scheduling while memory stalls

**Practical benefit for K80**: Negligible.

### Why Negligible: Memory-Bound Bottleneck

Measured K80 performance (KeyBridge Wireless, Q4_K_M, full GPU offload):

| Model | Size | Measured | Theoretical max (BW-only) | Efficiency |
|-------|------|----------|---------------------------|------------|
| Phi-4-mini | 3.8B | 13 tok/s | ~126 tok/s | ~10% |
| Qwen3-class | 14B | 4.2-4.4 tok/s | ~34 tok/s | ~13% |

KeyBridge explicitly states: *"Generation speed is essentially flat... the limiting factor is memory bandwidth, not compute."*

For a 14B model (~7 GB weights):
- Required: load all weights per token
- K80 bandwidth: 240 GB/s (per GK210)
- Theoretical max: 240 / 7 ≈ 34 tok/s
- Actual: 4.4 tok/s → **13% of theoretical max**

The 87% gap is overhead: kernel launches, cuBLAS/MMQ dispatch, memory access patterns, non-ideal coalescing — not arithmetic instruction choice.

### If You Must Try IMAD

Use inline PTX for the inner loop of `ggml_cuda_dp4a`:

```cpp
// In common.cuh, replace the sm_37 dp4a fallback:
static __device__ __forceinline__ int ggml_cuda_dp4a(const int a, const int b, int c) {
    int result;
    const int8_t * a8 = (const int8_t *) &a;
    const int8_t * b8 = (const int8_t *) &b;
    // Chain IMAD: each byte pair, accumulate
    asm volatile(
        "imad.u32.u32 %0, %1, %2, %3;\n"   // c + a[0]*b[0]
        "imad.u32.u32 %0, %4, %5, %0;\n"   // + a[1]*b[1]
        "imad.u32.u32 %0, %6, %7, %0;\n"   // + a[2]*b[2]
        "imad.u32.u32 %0, %8, %9, %0;\n"   // + a[3]*b[3]
        : "=r"(result)
        : "r"(a8[0]), "r"(b8[0]), "r"(c),
          "r"(a8[1]), "r"(b8[1]),
          "r"(a8[2]), "r"(b8[2]),
          "r"(a8[3]), "r"(b8[3])
    );
    return result;
}
```

**Expected gain**: <2% — you're shaving cycles off something that spends 90% of its time waiting for memory.

---

## LUT-GEMM Analysis

### What It Is

LUT-GEMM replaces arithmetic dequantization with shared memory lookup:
- 4-bit quant → 16-entry table per group
- Each index fetches pre-dequantized float value
- Eliminates multiply/add per element

### Why It Doesn't Help Q4_K_M

1. **Q4_K_M uses uniform quantization**: `value = scale * (quant - 8)`
   - This is already optimal arithmetic: 1 mul, 1 add/sub
   - No benefit from replacing with table lookup

2. **LUT shines for non-uniform quantization**: NF4, BCQ, learned tables
   - These require expensive nonlinear transforms per element
   - Table lookup replaces complex math with memory read
   - Q4_K_M's linear formula is cheaper than the lookup latency

3. **Shared memory bandwidth constraint**:
   - 16-entry LUT = 64 bytes (for 8-bit entries) or 128 bytes (for fp16)
   - Every thread in a warp needs the table → broadcast/shared access pattern
   - On Kepler's 48KB shared memory: ~750 groups per SMX simultaneously
   - But lookup still incurs shared memory latency (~20-30 cycles)

4. **Modern hardware targeting**:
   - NAVER LUT-GEMM: requires sm_80+ (Blackwell tensor cores)
   - FLUTE (Google EMNLP 2024): targets Ampere/Volta+ with async copy
   - Neither design accounts for Kepler's limitations

### LUT Feasibility on K80

If you absolutely wanted to try:

```cpp
// For Q4_K_M group of 64 elements with scale d:
// Pre-compute: lut[i] = d * (i - 8.0f) for i in [0,15]
__device__ float dequant_with_lut(uint8_t quant, const float4* lut) {
    uint4 indices;
    // Extract 4 nibbles from 2 bytes
    asm volatile("prmt.b32 %0, %1, %2, 0x0123;" : "=r"(indices.x) 
                 : "r"(quant), "r"(0x88888888));
    return lut[0].x;  // Simplified; actual impl needs byte shuffles
}
```

**Verdict**: More code, same or worse performance. The uniform dequant is already ~4-5 cycles; LUT lookup adds shared memory latency overhead.

---

## What Would Actually Help

### 1. Memory Access Pattern Optimization

- Ensure weight loads are fully coalesced across warps
- Check if `mmq-config-pascal-older.cuh` tile sizes are optimal for K80's 32-thread warps

### 2. cuBLAS F32 vs MMQ Comparison

Run benchmarks comparing:
- `GGML_CUDA_FORCE_MMQ=ON` (current: integer MMQ)
- `GGML_CUDA_FORCE_CUBLAS=ON` (cuBLAS F32 with dequantize)
- cuBLAS FP16 compute with FP32 output (if supported on Kepler)

The cuBLAS F32 path may actually be competitive since:
- cuBLAS is highly optimized for Kepler
- Dequantize kernel runs once, GEMM runs on optimized FP32 hardware
- MMQ's integer arithmetic may not beat cuBLAS's memory-aware blocking

### 3. Multi-GPU Optimization (2x K80)

Current: `--tensor-split 1,1` distributes layers across GPUs.

Alternative: Run independent llama-server processes (one per GPU), no tensor splitting. KeyBridge found this preferable for their workload.

### 4. Model Size Selection

For K80's bandwidth limits:
- <5B parameter models: competitive (10-20 tok/s expected)
- 7-14B: marginal (4-6 tok/s)
- >14B: impractical (single GPU can't hold weights)

---

## Conclusion

- **IMAD**: Nice to have, won't move the needle. Try only if everything else is exhausted.
- **LUT-GEMM**: Wrong tool for Q4_K_M's uniform quantization. Not worth implementing.
- **Real bottleneck**: Memory bandwidth + overhead. Optimize weight loading, not arithmetic.
- **Priority**: Benchmark cuBLAS vs MMQ, optimize memory access, consider smaller models.

**The K80 is a memory bandwidth appliance, not a compute powerhouse. Respect that constraint.**
