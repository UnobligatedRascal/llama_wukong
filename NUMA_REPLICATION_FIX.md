# NUMA Replication Fix: Implementation Status

## Critical Findings

### Benchmark (Qwen2.5-0.5B, 36 threads, 256 tokens)

| Test | Gen TPS | vs Best |
|------|---------|---------|
| numactl bind node0 (18 threads) | 56.7 | 100% |
| numactl bind node1 (18 threads) | 55.5 | 97.8% |
| numactl interleave (36 threads) | 29.8 | 52.6% |
| baseline no NUMA control (36 threads) | 25.3 | 44.7% |
| **--numa mirror (ours, BROKEN)** | 24.6 | 43.3% |

Single-node binding is **2.3x faster** than our broken NUMA mirror.

### Root Cause

Original `ggml-cpu-numa-replicate.{c,h}` had dead code — functions defined but never called. MIRROR strategy in ggml-cpu.c was identical to DISTRIBUTE (spread threads, no weight replication).

## Implementation Attempt

Created offset-based replication:
1. Allocate weights normally via ggml allocator
2. After allocation, replicate buffer data to per-node copies using `numa_alloc_onnode()`
3. Track `original_base` and per-node `node_bases[]`
4. `ggml_numa_replicate_get_local_ptr(global_ptr)` returns node-local pointer via TLS-cached node detection

**Problem:** Compute kernels read `tensor->data` directly, never calling `ggml_numa_replicate_get_local_ptr()`. Replicated data is unused.

## Why Buffer Wrapper Didn't Work

Attempted wrapping buffers with custom `get_base()` returning per-thread pointers. Failed because:
- `ggml_backend_tensor_alloc()` computes tensor addresses from `get_base()` in main thread
- If `get_base()` returns different values per thread, tensor->data pointers become invalid for other threads
- Tensor addresses are static; can't change per-thread

## Required Solution

**Kernel-level integration:** Modify weight-reading kernels to call `ggml_numa_replicate_get_local_ptr(tensor->data)` before dereferencing. This is invasive — needs changes to:
- Quantized GEMM kernels in ggml-cpu-quants.c
- Dequantization functions that read weights
- Any direct weight tensor access

Alternatively: **Don't implement NUMA replication.** Just use `numactl --cpunodebind=0 --membind=0` with 18 threads. Single-node is 2.3x faster than our broken mirror and requires zero code changes.

## Option A (Recommended): Single-Node Binding

Run llama_wukong with:
```bash
numactl --cpunodebind=0 --membind=0 ./build/bin/llama-server <model> -t 18 --numa isolate ...
```

This gives 56.7 gen TPS vs 24.6 for broken mirror. No code changes needed.

## Files Modified

- `ggml/src/ggml-cpu/ggml-cpu-numa-replicate.c` — rewritten with offset-based replication (functional but unused)
- `ggml/include/ggml-cpu.h` — added NUMA replicate declarations
- `src/llama-model.cpp` — calls `ggml_numa_replicate_weights()` after buffer allocation

## Next Steps

1. **SHORT TERM:** Use Option A (numactl bind) — proven 2.3x speedup, zero risk
2. **LONG TERM (P1):** If NUMA replication is worth it, modify kernels to use `ggml_numa_replicate_get_local_ptr()`. Estimate: 50-100 LOC changes across quantization/GEMM kernels. ROI questionable vs single-node binding.
