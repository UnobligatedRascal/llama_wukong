# NUMA Replication Research

## Problem

NOUGHT has dual-socket Xeon E5-2697 v4 with separate DDR4 controllers:
- NUMA node 0: cores 0-17,36-53 + 64GB RAM
- NUMA node 1: cores 18-35,54-71 + 64GB RAM
- Cross-node latency: 21 (vs 10 local) via Intel UPI
- Cross-node bandwidth: ~80 GB/s (vs ~460 GB/s local)

Standard llama.cpp allocates weights via malloc (not NUMA-aware). When threads
on node 1 read weights allocated on node 0, every access crosses UPI. For a
70GB Q8 model, this is catastrophic.

## Solution: Weight Replication

roblee04/numa_llamacpp proves 1.4-1.75x speedup via:
1. Duplicate weights: load model twice, once per NUMA node
2. Thread pinning: bind threads to their NUMA-local cores
3. Pointer swap: each thread reads from its node's copy

Memory cost: 2x weights in RAM. For Qwen3.8-Flash-Next (~82GB at IQ3_XXS),
that's ~164GB -- exceeds our 128GB. Trade-off: replicate only hot paths
(attention weights, norms) or use partial replication.

## Implementation Points

### Buffer Type Extension
llama.cpp backend alloc interface (ggml_backend_buffer_type_i):
- Add NUMA-aware alloc_buffer that calls numa_alloc_onnode()
- Track per-node base pointers

### Pointer Swap in GEMM
ggml_compute_forward_mul_mat receives ggml_compute_params { nth, ith }:
- ith = thread index, determines which NUMA node
- Swap src0->data to node-local copy at start of kernel
- Single pointer dereference change, zero latency cost

### Thread Affinity
Use sched_setaffinity or pthread_setaffinity_np:
```c
cpu_set_t set;
CPU_ZERO(&set);
for (int i = 0; i < 18; i++) CPU_SET(i, &set);      // node 0 local
for (int i = 0; i < 18; i++) CPU_SET(36+i, &set);   // node 0 HT
pthread_setaffinity_np(thread, sizeof(set), &set);
```

### Sync Barrier Optimization
Standard barrier: all threads do cross-node atomic operations.
Optimized: per-node barriers first, then only N_NODE threads sync globally.

## References
- roblee04/numa_llamacpp: https://github.com/roblee04/numa_llamacpp
- GGML_NUMA_MIRROR PR #14969: per-node wdata buffers
- SemiEngineering blog (Feb 2026): 55% speedup on Neoverse dual-NUMA

## NOUGHT-Specific Notes

- 36 physical cores total, use -t 36 (not 72 with HT)
- Each node: 18 physical cores + 18 hyperthreads
- 128GB RAM limits full replication for large models
- For Qwen3.8-Flash-Next: selective replication or accept memory cost
- For 27B baseline model (~13GB Q4): replication is cheap (26GB total)
