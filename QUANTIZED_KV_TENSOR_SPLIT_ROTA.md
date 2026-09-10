# ROTA: Enable Quantized KV Cache with Tensor-Split Mode

**Date:** 2026-09-09  
**Author:** UnobligatedRascal  
**Status:** Option A fix applied (2026-09-09), pending rebuild/test  
**Priority:** P0 — blocks all KV cache quantization benefits on NOUGHT (8-GPU tensor-split)

---

## Problem Statement

KV cache quantization (`--cache-type-k q4_0 --cache-type-v q4_0`, etc.) is completely non-functional when used with `--split-mode tensor`. The system silently falls back to unquantized behavior, making quantization settings ineffective.

**Symptoms:**
- Setting `--cache-type-k q4_0 --cache-type-v q4_0` with tensor-split produces identical token/sec as F16 KV cache
- No error messages — the fallback is silent
- Memory usage does not decrease as expected from quantization
- Custom turbo3_0/turbo4_0 types previously crashed with "tensor read out of bounds" (now fixed by reverting broken MIRRORED shortcut)

**Impact:** On NOUGHT with 8 GPUs in tensor-split mode, we cannot reduce KV cache memory from F16 to save VRAM for longer context windows. This is a critical blocker.

---

## Root Cause Analysis

### What's Already Fixed (Upstream PR #23792)

PR #23792 by JohannesGaessler (merged June 1, 2026) is **already present** in llama_wukong:
- `uint32_t nr[16]` field in `ggml_backend_meta_split_state` ✓
- Guard removal in `llama-context.cpp` ✓
- Split-state propagation fixes in `ggml-backend-meta.cpp` ✓
- Segment+repetition format in `llama-model.cpp` ✓

**This was NOT the missing piece.**

### Actual Root Cause: Meta Backend + Quantized Type Mismatch

The problem is in how the meta backend handles quantized KV cache tensors when they're split across GPUs.

**Key files:**
1. `ggml/src/ggml-backend-meta.cpp` — meta backend tensor operations
2. `src/llama-model.cpp:llama_meta_device_get_split_state()` — split state computation
3. `src/llama-kv-cache.cpp` — KV cache allocation

**The bug chain:**

1. KV cache tensors are created with quantized types (q4_0, q8_0, turbo3_0, etc.) in `llama-kv-cache.cpp:242-243`:
   ```cpp
   ggml_tensor * k = ggml_new_tensor_3d(ctx, type_k, n_embd_k_gqa, kv_size, n_stream);
   ggml_tensor * v = ggml_new_tensor_3d(ctx, type_v, n_embd_v_gqa, kv_size, n_stream);
   ```

2. Split state is computed in `llama-model.cpp:526`:
   ```cpp
   if (std::regex_match(tensor_name, pattern_kv_cache)) {
       return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "attn_output.weight");
   }
   ```
   KV cache is split on axis 0 (embedding dimension), using `attn_output.weight` as reference.

3. Block size is computed from the reference tensor, NOT the KV cache tensor (`llama-model.cpp:788`):
   ```cpp
   const int64_t blck_size = ggml_blck_size(tc.tensor_axis_0->type);
   ```
   For KV cache, `tc.tensor_axis_0` is `attn_output.weight` (e.g., Q4_K_M), but the KV cache itself may be q4_0, q8_0, or turbo3_0 — **different block sizes**.

4. The meta backend's `init_tensor` and `get_tensor/set_tensor` operations use element counts from the split state, but quantized types have non-trivial byte layouts. When the split boundaries don't align with quantization block boundaries, data corruption occurs.

5. The CUDA flash attention kernel has limited type support (`ggml-cuda/fattn.cu:338-355`):
   ```cpp
   static bool ggml_cuda_fattn_kv_type_supported(ggml_type type) {
       switch (type) {
           case GGML_TYPE_F32: case GGML_TYPE_F16: return true;
           case GGML_TYPE_Q4_0: case GGML_TYPE_Q8_0: case GGML_TYPE_BF16: return true;
           default: return false;  // turbo3_0, turbo4_0, etc. NOT supported
       }
   }
   ```
   Custom types (turbo3_0, turbo4_0) return `false`, causing fallback to non-FA path.

### Why F16 Speed = Q4/Q8 Speed

When quantization is corrupted by misaligned splits:
- Quantization overhead (dequantization in kernels) is still present
- Quantization benefit (reduced memory bandwidth) is lost due to incorrect data
- Result: same or worse performance than F16 with no memory savings

---

## Fix Strategy

### Option A: Fix Block Size Reference (Quick, Partial)

**Change:** Use the KV cache tensor's own type for block size calculation.

**File:** `src/llama-model.cpp:788`

**Current:**
```cpp
const int64_t blck_size = ggml_blck_size(tc.tensor_axis_0->type);
```

**Fix:**
```cpp
// For KV cache tensors, use their own type's block size, not the reference weight tensor
const int64_t blck_size = (std::regex_match(tensor_name, pattern_kv_cache))
    ? ggml_blck_size(tensor->type)
    : ggml_blck_size(tc.tensor_axis_0->type);
```

**Pros:** Simple, one-line change. May fix q4_0/q8_0.  
**Cons:** Doesn't fix custom types (turbo3_0, etc.) — CUDA FA doesn't support them.

### Option B: Add Quantized Type Support to CUDA Flash Attention (Medium)

**Task:** Extend `ggml_cuda_fattn_kv_type_supported()` and add kernel templates for turbo3_0/turbo4_0.

**Files:**
- `ggml/src/ggml-cuda/fattn.cu:338-355`
- `ggml/src/ggml-cuda/fattn-vec.cuh`
- `ggml/src/ggml-cuda/template-instances/`

**Pros:** Enables full quantization stack.  
**Cons:** Requires writing new CUDA kernels; significant work.

### Option C: Rework Meta Backend for Quantized Split Tensors (Hard, Complete)

**Task:** Make the meta backend properly handle quantized split tensors by:
1. Computing split boundaries in bytes, aligned to quantization blocks
2. Ensuring each GPU's portion is a valid quantized tensor
3. Adding proper dequantization paths for unsupported types

**Files:**
- `ggml/src/ggml-backend-meta.cpp`
- `ggml/src/ggml-backend.cpp`
- `src/llama-model.cpp`

**Pros:** Complete fix; enables all quantized types.  
**Cons:** Major refactoring; high risk of introducing bugs.

### Recommended Approach: A + B

1. Apply Option A immediately (1-line fix, low risk)
2. Test with q4_0/q8_0 — if working, we have partial victory
3. For turbo3_0/turbo4_0, either:
   - Add CUDA FA support (Option B), or
   - Fall back to non-FA path with corrected meta backend handling

---

## Critical Code References

### 1. KV Cache Allocation (`src/llama-kv-cache.cpp:242-243`)
```cpp
ggml_tensor * k = has_k ? ggml_new_tensor_3d(ctx, type_k, n_embd_k_gqa, kv_size, n_stream) : nullptr;
ggml_tensor * v = has_v ? ggml_new_tensor_3d(ctx, type_v, n_embd_v_gqa, kv_size, n_stream) : nullptr;
```
KV cache tensors are created with user-specified quantized types.

### 2. Split State Reference (`src/llama-model.cpp:526`)
```cpp
if (std::regex_match(tensor_name, pattern_kv_cache) || std::regex_match(tensor_name, pattern_attn_sinks)) {
    return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "attn_output.weight");
}
```
KV cache split state uses `attn_output.weight` as reference tensor.

### 3. Block Size Computation (`src/llama-model.cpp:788`)
```cpp
const int64_t blck_size = ggml_blck_size(tc.tensor_axis_0->type);
```
**BUG:** Uses reference tensor's block size, not KV cache tensor's block size.

### 4. CUDA FA Type Support (`ggml/src/ggml-cuda/fattn.cu:338-355`)
```cpp
static bool ggml_cuda_fattn_kv_type_supported(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32: case GGML_TYPE_F16: return true;
        case GGML_TYPE_Q4_0: case GGML_TYPE_Q8_0: case GGML_TYPE_BF16: return true;
        default: return false;  // turbo3_0, turbo4_0 NOT supported
    }
}
```
Custom types are rejected by CUDA flash attention.

### 5. Meta Backend Split Tensor Handling (`ggml/src/ggml-backend-meta.cpp:1577-1620`)
```cpp
if (split_state.n_segments != 1 || split_state.nr[0] != 1) {
    // Split tensor get_tensor logic
    // Uses element counts from split_state.ne[] but doesn't account for
    // quantized type byte layout differences between reference and actual tensor
}
```
Split tensor operations assume element-aligned splits work for all types.

---

## Test Plan

### Pre-fix Baseline
```bash
# Current broken behavior (document for comparison)
llama-server -m model.gguf --split-mode tensor --tensor-split 1,1,1,1,1,1,1,1 \
    --cache-type-k q4_0 --cache-type-v q4_0 --flash-attn 1 \
    -lv 3 2>&1 | grep -E "KV buffer|cache_type|split_state"
```

### Post-fix Tests

1. **Basic functionality (q4_0):**
   ```bash
   llama-server -m model.gguf --split-mode tensor --tensor-split 1,1,1,1,1,1,1,1 \
       --cache-type-k q4_0 --cache-type-v q4_0 --flash-attn 1 -lv 3
   ```
   - Verify KV buffer size is ~2x smaller than F16
   - Verify token/sec is higher than F16 (less memory bandwidth)
   - Verify no "out of bounds" errors

2. **Mixed types (q8_0 K, q4_0 V):**
   ```bash
   llama-server -m model.gguf --split-mode tensor --tensor-split 1,1,1,1,1,1,1,1 \
       --cache-type-k q8_0 --cache-type-v q4_0 --flash-attn 1
   ```

3. **Custom types (turbo3_0) — expected to fall back to non-FA:**
   ```bash
   llama-server -m model.gguf --split-mode tensor --tensor-split 1,1,1,1,1,1,1,1 \
       --cache-type-k turbo3_0 --cache-type-v turbo3_0 --flash-attn 1
   ```

4. **Correctness check:**
   - Run same prompt with F16 vs q4_0 KV cache
   - Compare outputs — should be nearly identical (quantization noise only)

---

## Upstream References

- **PR #23792** (merged): "TP: quantized KV cache support" — already in llama_wukong
  - https://github.com/ggml-org/llama.cpp/pull/23792
- **Issue #21788** (closed): "Allow SPLIT_MODE_TENSOR with KV cache quantization"
  - https://github.com/ggml-org/llama.cpp/issues/21788
- **Issue #23567** (closed): "Support quantized KV cache with --split-mode tensor"
  - https://github.com/ggml-org/llama.cpp/issues/23567
- **Megatron-LM tensor parallel mappings** (reference implementation):
  - https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/mappings.py

---

## Action Items

- [x] Apply Option A fix (block size reference) — committed 2026-09-09
  - llama_wukong: 7d7944e3b
  - llama_lazarus: 77657e77f
- [ ] Rebuild and test with q4_0/q8_0
- [ ] Document results
- [ ] If q4_0/q8_0 works: decide on turbo3_0/turbo4_0 strategy
- [ ] If q4_0/q8_0 still broken: investigate meta backend byte layout handling
- [ ] Consider upstreaming any fixes to ggml-org/llama.cpp

## Notes

- Same fix applied to both llama_wukong and llama_lazarus
- llama_lazarus is the upstream base; llama_wukong was a cowboy branch
- After validating this fix, should properly rebase/fork wukong from updated lazarus
- CUDA FA type support (turbo3_0/turbo4_0) is a SEPARATE issue — not fixed here

---

*UnobligatedRascal — "If it's worth doing, it's worth doing RIGHT."*
