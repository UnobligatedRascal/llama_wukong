# TurboQuant / TriAttention / FlashAttention Research Notes

**Date:** 2026-09-08
**Hardware target:** NOUGHT (8x Tesla K80, sm_37, 12GB VRAM each, GK210)

## Repo Corrections

The original references were incorrect. Correct repos:

| Task | Original (wrong) | Correct |
|------|------------------|---------|
| TurboQuant | TheTom/TurboQuant | TheTom/turboquant_plus (research) + atomicmilkshake/llama-cpp-turboquant (llama.cpp fork) |
| TriAttention | atomicmilkshake/TriAttention | domvox/triattention-ggml (standalone) + integrated in atomicmilkshake/llama-cpp-turboquant |

## 1. TurboQuant Integration

### What it is
- KV cache compression using PolarQuant: WHT rotation + Lloyd-Max scalar quantization
- Formats: turbo4 (4-bit, 3.8x compression), turbo3 (3-bit, 4.6x), turbo2 (2-bit, 6.4x)
- Key finding: V compression is free, K compression is the quality bottleneck

### Integration status in llama.cpp-turboquant fork

**Already implemented:**
- CUDA kernels in `ggml/src/ggml-cuda/turbo-quant.cuh`:
  - FWHT rotation (128-element, O(n log n))
  - 2/3/4-bit centroids (Lloyd-Max for N(0, 1/128))
  - Block quantize/dequantize for turbo2_0, turbo3_0, turbo4_0
  - InnerQ: per-channel variance equalization before rotation (TURBO_INNERQ env)
- FlashAttention integration: `fattn-vec-instance-turbo4_0-*.cu` template files
- Weight formats: TQ3_1S, TQ4_1S

**Integration path for llama_wukong:**
1. Port turbo-quant.cuh and turbo-innerq.cuh from fork
2. Add GGML_TYPE_TURBO2_0, TURBO3_0, TURBO4_0 to ggml type system
3. Wire into set_rows for KV cache writes
4. Add FA vec instances for turbo formats

### ⚠️ K80 COMPATIBILITY ISSUE

The llama.cpp-turboquant fork targets Turing+ (sm_75+):
- CUDA build: `-DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;120;121"`
- Uses `half2`, FP16 math extensively
- FWHT kernels use fp16 path by default

**For K80 sm_37:**
- No FP16 tensor cores
- Must use FP32 path or software-emulated FP16
- Need to add sm_37 to architectures and ensure fp32 fallback in turbo kernels
- Performance will be worse than on Turing+, but should work

**Action needed:** Audit turbo-quant.cuh for FP16-only paths, add sm_37-compatible fp32 variants.

### Key files to cherry-pick from atomicmilkshake/llama-cpp-turboquant:
- `ggml/src/ggml-cuda/turbo-quant.cuh`
- `ggml/src/ggml-cuda/turbo-innerq.cuh`
- `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-turbo*.cu`
- `ggml/include/ggml.h` (type definitions for TURBO2/3/4)
- `common/common.cpp` (cache type parsing for --cache-type-k/v flags)

## 2. TriAttention Integration

### What it is
- KV cache pruning via trigonometric frequency scoring
- Uses RoPE-inverted key vectors to score token importance
- Evicts low-scoring tokens, keeps fixed KV budget
- GPU scoring is ~1000x faster than CPU (4-9ms vs 5900ms per event)

### Implementation in llama.cpp-turboquant fork

**Already implemented:**
- `ggml/src/ggml-cuda/triattention-score.cu`:
  - Full GPU kernel: dequant K → inverse WHT (if turbo) → inverse RoPE → score
  - Template for each K type: TURBO2, TURBO3, TURBO4, Q8_0, F16, F32
  - Supports calibration files for per-head statistics
  - CUDA stream-aware, async scoring

### ⚠️ K80 COMPATIBILITY

The triattention kernel:
- Uses standard CUDA operations (cosf, sinf, atan2f, cosf — all work on sm_37)
- Shared memory usage: (head_dim + freq_count) * 4 bytes = ~1KB for head_dim=128
  - K80 has 48KB shared mem per block — fine
- No tensor cores needed

**Should work on K80 with minimal changes.**

### ⚠️ domvox/triattention-ggml notes

The standalone repo is HIP/ROCm-focused:
- Uses HIP kernels, not CUDA
- Better reference for algorithm than direct integration source
- The atomicmilkshake fork's CUDA implementation is the one to use

### Integration path:
1. Port triattention-score.cu/cuh from fork
2. Port calibration file format and loading code
3. Add --triattention-* flags to llama-server
4. Hook scoring into decode loop (trigger every N tokens when KV > budget)

### Key files from atomicmilkshake/llama-cpp-turboquant:
- `ggml/src/ggml-cuda/triattention-score.cu`
- `ggml/src/ggml-cuda/triattention-score.cuh`
- Search fork for `triattention_calibrate.py` equivalent
- Server flag parsing for --triattention-budget, --triattention-stats, etc.

## 3. FlashAttention / SlidingWindowAttention on NOUGHT

### Current state in llama_wukong

FA implementation exists in `ggml/src/ggml-cuda/`:
- `fattn.cu` — kernel dispatcher
- `fattn-common.cuh` — common utilities
- `fattn-mma-f16.cuh` — tensor-core kernels (Volta+)
- `fattn-tile.cuh` — generic tile-based kernels
- `fattn-vec.cuh` — vector kernels

**Kernel selection logic (from fattn.cu):**
```
if (turing_mma_available(cc)) → MMA_F16
elif (volta_mma_available(cc)) → TILE or MMA_F16
else → TILE (fallback)
```

**For K80 sm_37:** Falls through to BEST_FATTN_KERNEL_TILE or BEST_FATTN_KERNEL_VEC.

### ⚠️ K80 COMPATIBILITY

From common.cuh:
- `GGML_CUDA_CC_PASCAL = 600`
- `GGML_CUDA_CC_VOLTA = 700`
- K80 is sm_37 = 370

The tile kernel path is guarded by `#ifdef FLASH_ATTN_AVAILABLE` only, not compute capability. Need to verify:
1. Does FLASH_ATTN_AVAILABLE get defined for sm_37?
2. Does the tile kernel actually compile for sm_37?
3. Any `__half` usage that breaks on Kepler?

**Build config for llama_wukong currently:**
```
-DCMAKE_CUDA_ARCHITECTURES="37"
-DGGML_CUDA_FA_ALL_QUANTS=ON
```

So FA is being compiled for sm_37. Need to **test** if it actually runs.

### SlidingWindowAttention

**No dedicated SWA implementation found** in llama_wukong CUDA codebase.
- Only one "SWA" reference: comment in fattn.cu line 143 about MiMo-V2.5 gqa_ratio
- SWA is typically implemented via attention masks, not separate kernels
- Could be wired through the existing FA mask parameter

**Options:**
1. Implement SWA via attention mask: set mask to -inf for positions outside window
   - Works with existing FA, no new kernel needed
   - Need to generate appropriate mask tensors in C++ code
2. Implement dedicated SWA kernel:
   - Only loads K/V within window into shared memory
   - Saves bandwidth but more complex

**Recommendation:** Start with mask-based SWA (option 1), measure if it's fast enough.

### Action items for FA on NOUGHT:

1. **Test current FA:** Build with -DGGML_CUDA_FA_ALL_QUANTS=ON and run attention-heavy model
2. **If FA fails:** Check for fp16-only paths in fattn-tile.cuh
3. **If FA works but slow:** Tune tile/block sizes for GK210 constraints
4. **SWA:** Implement via mask first, benchmark vs dense attention on long context

## 4. Cross-Repo Compatibility Notes

### llama_wukong vs llama.cpp-turboquant

Both are forks of llama.cpp but may have diverged:
- llama_wukong has custom NUMA replication code
- llama_wukong has async pipeline code (partially rolled back)
- llama.cpp-turboquant has TurboQuant + TriAttention

**Integration strategy:**
- Cherry-pick CUDA files from turboquant fork (turbo-quant.cuh, triattention-score.cu)
- These are mostly self-contained, shouldn't conflict with NUMA code
- Watch out for ggml type enum changes (TURBO2/3/4 types need to be added)

### Priority recommendation

Given K80 sm_37 constraints:

1. **FlashAttention verification** — already in codebase, test if it works on sm_37
2. **SlidingWindowAttention** — low-cost via attention masks
3. **TurboQuant** — needs FP32 fallback audit, but high ROI for KV cache savings
4. **TriAttention** — should work on K80, complements TurboQuant for extreme compression

All three are synergistic: TurboQuant compresses KV, TriAttention prunes it, FA/SWA speeds up attention on the remaining tokens.

## 5. Cloned Repos (for reference)

- `/home/whistler/turboquant_plus` — TheTom's research reference (Python)
- `/home/whistler/llama-cpp-turboquant` — atomicmilkshake's llama.cpp fork (CUDA kernels)
- `/home/whistler/triattention-ggml` — domvox's HIP/ROCm implementation (algorithm reference)
