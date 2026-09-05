# NUMA Replication Benchmark Results

**Date:** 2026-09-04  
**System:** NOUGHT — Dual Xeon E5-2697 v4 (2 nodes x 36 CPUs, 64GB DDR4 each, UPI distance 21)  
**Model:** Qwen2.5-0.5B-Instruct-Q4_K_M (CPU inference, 256 tokens generated)  
**numa_balancing:** 0 (disabled)

## Benchmark Results

| Test | Prompt TPS | Gen TPS | Prompt ms | Gen ms | vs Best |
|------|-----------|---------|-----------|--------|---------|
| numactl bind node0 (18 threads) | 315.0 | 56.7 | 41.3 | 4499.5 | 100.0% |
| numactl bind node1 (18 threads) | 282.6 | 55.5 | 46.0 | 4598.4 | 97.8% |
| numactl interleave (36 threads) | 178.4 | 29.8 | 72.9 | 8553.8 | 52.6% |
| baseline no NUMA control (36 threads) | 150.3 | 25.3 | 86.5 | 10061.3 | 44.7% |
| **--numa mirror (ours, 36 threads)** | 175.0 | 24.6 | 74.3 | 10379.8 | **43.3%** |

## Critical Finding: NUMA MIRROR Is Broken

`--numa mirror` is **the worst strategy** tested — 2.3x slower than single-node binding.

### Root Cause

The `ggml-cpu-numa-replicate.{c,h}` implementation provides:
- `ggml_numa_replicate_alloc_per_node()` — per-node memory allocation
- `ggml_numa_replicate_get_local_ptr()` — thread-local pointer lookup

But **these functions are never called**. Zero callers outside the defining file.

The MIRROR strategy in `ggml-cpu.c` (line 2192) does:
```c
case GGML_NUMA_STRATEGY_MIRROR:
    // MIRROR: distribute threads and enable weight replication
    node_num = thread_n % g_state.numa.n_nodes;
    break;
```

This is identical to DISTRIBUTE — it spreads threads across NUMA nodes but **never replicates the weights**. Result: threads on node 1 read all model weights through the UPI link from node 0's memory, causing exactly the cross-node latency we're trying to eliminate.

### Single-Node Binding: 2.3x Speedup

Binding to a single NUMA node with 18 threads is the clear winner:
- node0: 56.7 gen TPS
- node1: 55.5 gen TPS (97.8% of node0, minor asymmetry likely from free memory difference)

This proves the hardware NUMA penalty is real and that locality matters.

## Option A: Immediate Fix (Backup)

Run llama_wukong with single-node binding:
```bash
numactl --cpunodebind=0 --membind=0 ./build/bin/llama-server <model> -t 18 ...
```

This gives a **2.3x speedup over the broken MIRROR** and **2.2x over baseline** with zero code changes.

## Option B: Proper NUMA Replication (In Progress)

Wire up `ggml_numa_replicate_alloc_per_node()` into the tensor allocation path so model weights are duplicated per-NUMA-node at load time, and threads read from their local copy.

See NUMA_REPLICATION_FIX.md for implementation plan.
