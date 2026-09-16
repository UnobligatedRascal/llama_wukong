# TurboQuant Integration Research

**Author:** UnobligatedRascal  
**Date:** 2026-09-04  
**Sources:** TheTom/turboquant_plus, TheTom/llama-cpp-turboquant, atomicmilkshake/llama-cpp-turboquant  
**Target:** llama_wukong (Kepler sm_37, 8x K80)  

---

## Executive Summary

TurboQuant achieves 3.8-6.4x KV cache compression via PolarQuant codec (norm extraction + Walsh-Hadamard rotation + optimal scalar quantization) at near-q8_0 speed and ~0.9x decode throughput at long context. Two production forks exist:

| Implementation | Focus | Backends | Status |
|---------------|-------|----------|--------|
| TheTom/llama-cpp-turboquant | Complete TurboQuant+ stack | Metal, CUDA, HIP, CPU | Production fork, ~300 commits ahead of upstream |
| atomicmilkshake/llama-cpp-turboquant | Windows MSVC+CUDA + TriAttention | CUDA only | Pre-built binaries, SM75+ only |

For NOUGHT (Kepler sm_37): TheTom's implementation is the integration target. Requires custom sm_37 WHT and dequant kernels. atomicmilkshake's SM75+ requirement excludes Kepler entirely.

---

## Architecture: The PolarQuant Codec

The algorithm operates per-block (block_size=128 default) of KV cache vectors. Three stages:

### Stage 1: Norm Extraction
```
Input: KV cache vector x ∈ R^d (one head, one block)
γ = ||x||₂                          # L2 norm, stored as FP32 per block
x̂ = x / γ                           # Unit-normalized vector
```
Norm is stored uncompressed because reconstruction quality depends on it. ~2.5% overhead.

### Stage 2: Walsh-Hadamard Rotation
```
x_pad = pad(x̂) to next power of 2   # e.g., d=4096 → 4096 (exact), d=1280 → 2048
y = D₁ ⊙ x_pad                      # Random sign flips (pre-generated, stored once)
y = WHT(y) / √n                     # Fast Walsh-Hadamard Transform, O(d log d)
z = D₂ ⊙ y                          # Second random sign flips
z = truncate(z, d)                  # Back to original dimension
```

**Critical insight:** After rotation, coordinates follow N(0, 1/d) by CLT. Kurtosis drops from ~900 (raw KV) to ~2.9 (Gaussian baseline). This enables optimal scalar quantization where raw vectors would fail due to heavy-tailed outliers.

**Rotation matrix:** D₁ and D₂ are random ±1 vectors generated once per model/head dimension at load time. Stored as packed bits (~d/8 bytes each). Same rotation applied on both store and retrieve.

### Stage 3: Lloyd-Max Scalar Quantization
```
For each rotated coordinate z[i]:
    idx[i] = argmin_c |z[i] - c|    # Nearest centroid lookup via binary search on boundaries

Output: indices (bit-packed) + stored norms
```

Centroids are precomputed per bit-width using Lloyd's algorithm on N(0, 1/d):
- **turbo4:** 16 centroids (4-bit), indices stored as nibbles
- **turbo3:** 8 centroids (3-bit), 4 indices per byte
- **turbo2:** 4 centroids (2-bit), 4 indices per byte

### Reconstruction (Dequantize)
```
y_hat = centroids[idx]              # Lookup each coordinate
y_hat = y_hat / ||y_hat||           # Re-normalize to unit norm
x_hat_unit = Hᵀ @ (D₂ ⊙ y_hat) ⊙ D₁ # Inverse rotation
x_hat = x_hat_unit * γ              # Rescale by stored norm
```

---

## Quantization Formats

| Format | Bits/val | Compression | Centroids | Notes |
|--------|----------|-------------|-----------|-------|
| turbo4 | 4.25 bpv | 3.8x vs fp16 | 16 | Best quality, ~0.9% PPL delta vs q8_0 |
| turbo3 | 3.125 bpv | 5.12x vs fp16 | 8 | Best compression/quality balance |
| turbo2 | 2.125 bpv | 7.5x vs fp16 | 4 | Extreme compression, use asymmetric only |

Compression formula: `16 / avg_bpv`. For asymmetric K/V: `avg_bpv = (k_bpv + v_bpv) / 2`.

---

## Critical Findings

### 1. K Precision Dominates Quality
K determines attention routing via softmax. Softmax exponentially amplifies small errors: a tiny shift in Q*K scores can flip which tokens dominate. V errors scale linearly through the weighted sum.

**Evidence (Qwen2.5-7B Q4_K_M):**
- q8_0-K + turbo3-V: PPL 6.71 (+2.0%) — healthy
- turbo3-K + q8_0-V: PPL 3556 — catastrophic

Same total bits, opposite directions, 500x quality difference.

### 2. V Compression Is "Free"
Compressing V down to 2 bits has zero measurable effect on attention quality when K precision is maintained. Confirmed across Metal, CUDA, and HIP backends.

### 3. Boundary Layers Are Sensitive
First 2 + last 2 transformer layers disproportionately affect quality. Protecting them at q8_0 recovers 37-91% of quality gap when using aggressive compression elsewhere.

### 4. Sparse V Dequant Saves Decode Time
At long context, 90%+ of attention weights are negligible (<1e-6). Skipping V dequantization for those positions:
- +22.8% decode throughput on MoE at 32K
- Zero perplexity impact
- Not TurboQuant-specific (works on q8_0 too)

---

## Implementation Patterns from TheTom's Fork

### KV Cache Type Integration

Added to llama.cpp as cache-type flags:
```bash
-ctk turbo3 -ctv turbo3    # Symmetric
-ctk q8_0 -ctv turbo4      # Asymmetric (recommended default)
```

Integration points:
1. **kv_cache.h:** New cache_type enum values (e.g., GGML_KV_CACHE_TURBO4, TURBO3, TURBO2)
2. **ggml-backend-impl.h:** Cache size calculation per turbo format
3. **flash-attention kernels:** Store and load paths handle turbo-encoded blocks
4. **ggml-cuda.cu:** WHT rotation kernels + dequant kernels
5. **ggml-cpu.c:** Reference CPU implementation

### Memory Layout

Per turbo block (128 elements default):
- 4 bytes: stored norm (FP32)
- N bytes: bit-packed indices (depends on turbo level)

For turbo3 with d=4096:
- Norm: 4 bytes
- Indices: 4096 * 3 bits = 1536 bytes
- Total per block: 1540 bytes vs 8192 bytes fp16 = 5.3x compression

### WHT Kernel Pattern

The Hadamard transform is implemented as butterfly operations:
```cuda
// Forward WHT (in-place)
__device__ void wht_forward(float* x, int n) {
    for (int h = 1; h < n; h *= 2) {
        for (int i = 0; i < n; i += h * 2) {
            for (int j = 0; j < h; j++) {
                float a = x[i + j];
                float b = x[i + j + h];
                x[i + j] = a + b;
                x[i + j + h] = a - b;
            }
        }
    }
    // Normalize by 1/√n (applied once at end)
    float scale = 1.0f / sqrtf(n);
    for (int i = 0; i < n; i++) x[i] *= scale;
}
```

### Dequant Kernel Pattern

Dequantization must fuse: centroid lookup → inverse rotation → attention compute. Separate passes kill performance.

---

## Upstream Status

**Merged into llama.cpp:**
- Hadamard KV cache rotation (PR #21038, cites TurboQuant)
- CPU WHT kernels (PR #22631)
- CUDA WHT kernels (PR #23615)
- Vulkan WHT kernels (PR #23687)
- Sparse V dequant (PR #21119)

**Not merged (PR rejected June 2026):**
- Full PolarQuant codec (PR #21089) — upstream wanted smaller incremental patches

**vLLM:**
- Full TurboQuant upstream (PR #38479) as `--kv-cache-dtype turboquant_k8v4`

**Result:** llama.cpp rotation + q4_0 cache approximates turbo4's rotation stage. Full PolarQuant with optimal centroids requires the fork.

---

## atomicmilkshake Implementation Notes

atomicmilkshake's fork targets Windows MSVC + CUDA with two additions:

### TurboQuant Weight Formats
Formats turbo2_0, turbo3_0, turbo4_0 as model weight quantization (not just KV cache):
- turbo4_0: WHT-space centroids, drop-in q4_0 replacement
- turbo3_0: 3-bit with Hadamard pre-rotation  
- turbo2_0: Aggressive 2-bit compression

CUDA kernels optimized for Turing+ (SM75) and Ampere (SM80/86). Not applicable to Kepler.

### TriAttention (KV Cache Pruning)
GPU-accelerated token eviction using RoPE geometry:
- Scores token importance via RoPE-inverted key vectors
- Evicts low-value tokens when cache exceeds budget
- GPU scoring: 4-9ms/event vs 5900ms CPU (1000x faster)
- 4.3x generation speedup on RTX 3080 by keeping cache in VRAM

This is orthogonal to TurboQuant compression. Could be useful for long context on limited VRAM.

---

## Kepler sm_37 Adaptation Strategy

### What Transfers Directly
- Algorithm: PolarQuant codec is mathematically identical on any architecture
- CPU WHT: Upstream merged CPU kernels work on any x86
- Block layout: Same memory layout, just different backend kernels
- Asymmetric K/V config: Same logic applies

### What Needs Custom Kernels
1. **CUDA WHT:** Upstream CUDA WHT (PR #23615) uses newer instructions. Needs sm_37 verification and likely adaptation (no warp shuffle, limited shared mem).
2. **Dequant kernel:** Must implement centroid lookup + inverse WHT + attention fusion for Kepler.
3. **Store kernel:** Quantize + pack indices path needs sm_37 implementation.

### Recommended Approach
1. Start with CPU WHT (already upstream) + q8_0 K path for correctness validation
2. Implement simple CUDA WHT for sm_37 (butterfly ops are architecture-neutral)
3. Implement basic turbo3 dequant kernel (8 centroids = simple LUT)
4. Test asymmetric q8_0-K + turbo4-V first (most conservative, proven on all hardware)
5. Optimize: fuse dequant into attention kernel, add sparse V

### Memory Budget Impact
For Qwen3.8-Flash-Next at 128K context:
- fp16 KV cache: ~X GB (depends on full_attention layers only, ~16 of 64)
- q8_0 KV: ~X/2 GB
- turbo3 KV: ~X/5 GB (5.12x vs fp16)
- Asymmetric q8_0-K + turbo3-V: ~X/3 GB

Given GDN layers avoid KV entirely (75% of layers), actual savings scale with hybrid architecture. TurboQuant targets only the QSA layers (every 4th layer = ~16 layers with KV cache).

### ROI Assessment for llama_wukong
**Medium-High priority.** Key factors:
- Proven quality across backends (Metal, CUDA, HIP)
- Rotation already upstream = easier integration
- Complements GDN's KV-free design (both reduce memory pressure)
- Sparse V dequant helps decode speed regardless of compression level
- Requires sm_37 kernel work but algorithm is architecture-agnostic

Recommended: Phase 1.5 task (between 1.4 and 1.5 in current plan).

---

## Configuration Matrix (For NOUGHT Testing)

| Model weights | Safe config | Aggressive | Why |
|---------------|-------------|------------|-----|
| Q8_0+ | `-ctk turbo4 -ctv turbo4` | `-ctk turbo3 -ctv turbo3` | Strong weights absorb quantization |
| Q4_K_M, unknown | `-ctk q8_0 -ctv turbo4` | `-ctk q8_0 -ctv turbo3` | Asymmetric rescue |
| Q4_K_M, large (24B+) | `-ctk turbo4 -ctv turbo4` | `-ctk turbo3 -ctv turbo3` | Large models tolerant |
| Maximum compression | `-ctk q8_0 -ctv turbo2` | N/A | Boundary V auto-enables |

---

## References

- TheTom/turboquant_plus: https://github.com/TheTom/turboquant_plus (research home)
- TheTom/llama-cpp-turboquant: https://github.com/TheTom/llama-cpp-turboquant (production fork)
- atomicmilkshake/llama-cpp-turboquant: https://github.com/atomicmilkshake/llama-cpp-turboquant (Windows/CUDA)
- TurboQuant paper: arXiv 2504.19874 (ICLR 2026)
- PolarQuant paper: arXiv 2502.02617 (AISTATS 2026)
- Upstream rotation: PR #21038, WHT kernels #22631/#23615/#23687
- TheTom recommendations: docs/turboquant-recommendations.md
- llama.cpp discussion #20969: TurboQuant + MTP correctness battery

---
*End of research document — UnobligatedRascal*
