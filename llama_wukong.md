# Project llama_wukong

Base: llama_lazarus (UnobligatedRascal fork), commit 93c888df1
Hardware: NOUGHT (dual Xeon E5-2697 v4, 8x Tesla K80 GK210, 128GB RAM)
Target: Qwen3.8-Flash-Next (Qwen4 preview) optimized inference

## Vision

Two-phase optimization of llama.cpp for NOUGHT's specific hardware to run
Qwen3.8-Flash-Next (180B total params, 6B active/token) at practical throughput.

Phase 1: NOUGHT-specific hardware optimizations (NUMA, async pipeline,
custom kernels, memory tricks)

Phase 2: Qwen4-exp architecture integration (GDN/QSA, hyper-connection
tensors, ultra-sparse MoE routing, MTP head)

See ARCHITECTURE_REFERENCE.md for the original architectural pitch with
full memory topology diagrams, execution pipeline mermaid visualization,
and Kepler exploitation vectors.

## Phase 1: NOUGHT Hardware Optimization

### 1.1 NUMA Replication (highest ROI)

Goal: 1.4-1.75x speedup on dual-socket, zero GPU changes required.

Implementation:
- Duplicate model weights across both NUMA nodes at load time
- Per-thread pointer swap in GEMM kernels (each core reads local copy)
- Strict core pinning: NUMA node 0 gets cores 0-17,36-53; node 1 gets 18-35,54-71
- NUMA-local wdata buffers for quantized dequantization
- NUMA-aware sync barriers (reduce cross-node atomics)

Reference: roblee04/numa_llamacpp (proven 1.4-1.75x on dual EPYC)

Key files:
- ggml/src/ggml-cpu/ggml-cpu.c (thread pool, buffer alloc)
- ggml/src/ggml-cpu/ggml-cpu-quants.c (wdata handling)
- New: ggml/src/ggml-cpu/ggml-cpu-numa-replicate.c

### 1.2 Async Multi-GPU Pipeline

Goal: Hide PCIe Gen3 latency between K80 GPUs via compute/transfer overlap.

Implementation:
- Each K80 card has 2 independent GK210 chips via PLX switch
- Per-card: 2 CUDA streams (compute stream + prefetch stream)
- Event-based synchronization: Chip A computes layer N while Chip B pre-fetches
  layer N+1 weights via cudaStreamWaitEvent
- 8-GPU layer-split with event barriers between tensor boundaries

Key APIs:
- cudaStreamCreate, cudaEventCreate, cudaEventRecord, cudaStreamWaitEvent
- cudaMemcpyAsync with pinned host memory

Key files:
- ggml/src/ggml-cuda/ggml-cuda.cu (multi-GPU dispatch)
- ggml/src/ggml-backend/ggml-backend-impl.h (async interfaces)

### 1.3 GDN Kernel for Kepler

Goal: Exploit GK210's 512KB/SM register file for linear attention state.

Context: Qwen3.8-Flash-Next uses Gated DeltaNet (GDN) for 75% of layers.
GDN maintains a fixed-size recurrent state matrix S instead of growing KV cache.

Current state: llama.cpp has gated_delta_net.cu (upstream), needs sm_37 verification.

Optimization target:
- Pin state matrix S in shared memory per thread block
- Use registers for normalized q/k/v vectors (avoid global memory per step)
- Fuse decay/read/delta/update/readout into single kernel launch

GDN per-step math:
- S_decay = gate * S_prev (element-wise)
- kv_mem = S_decay^T * k_hat (matrix-vector)
- delta = beta_s * (v - kv_mem) (correction)
- S_new = S_decay + outer(k_hat, delta) (rank-1 update)
- out = S_new^T * q_hat (readout)

Key files:
- ggml/src/ggml-cuda/gated_delta_net.cu (existing, verify sm_37 compatibility)

### 1.4 Zero-Copy N-Gram Table

Goal: Access 51B parameter N-gram lookup from DDR4 without VRAM cost.

Implementation:
- cudaHostAlloc (page-locked) for N-gram table in system RAM
- cudaMemPrefetchAsync to pull needed pages before lookup kernels
- Indexed lookup kernel streams vectors over PCIe on-demand (~14 GB/s practical)
- Treat DDR4 as extended L2 cache for engram tensors

Trade-offs:
- PCIe Gen3 x16 bandwidth: ~14 GB/s practical vs 240 GB/s VRAM
- Only hit on bigram/trigram matches (not every token)
- Frees ~25GB equivalent VRAM for active model weights

Key APIs:
- cudaHostAlloc, cudaMemPrefetchAsync, cudaDeviceSynchronize

### 1.5 MoE Coarse-Grained Expert Batching

Goal: Reduce scattered weight loads for 512-expert ultra-sparse routing.

Problem: Standard MoE dispatch loads different experts per token -> scattered
memory access pattern, poor bandwidth utilization.

Solution:
- Hold tokens briefly, sort by expert index (top-10 of 512 per token)
- Group tokens targeting same expert -> stream expert weights once
- Process all tokens for expert E, then release, move to next expert
- Converts latency-bound MoE routing to throughput-bound weight streaming

Implementation:
- Modify topk-moe.cu to output sorted expert indices
- New kernel: expert_batch_dispatch (group tokens, stream weights, compute)

Key files:
- ggml/src/ggml-cuda/topk-moe.cu (existing routing kernel)

### 1.6 cuBLAS F32 vs MMQ Benchmarking

Goal: Determine optimal compute path for Kepler sm_37.

Current: GGML_CUDA_FORCE_MMQ=ON with integer MMQ fallback.

Hypothesis: cuBLAS F32 may be competitive or better because:
- cuBLAS highly optimized for Kepler (decades of tuning)
- Dequantize kernel runs once, GEMM on optimized FP32 hardware
- MMQ integer path has poor memory coalescing on Kepler

Action: Run controlled benchmarks comparing:
- MMQ path (current)
- cuBLAS F32 path (GGML_CUDA_FORCE_CUBLAS=ON)
- Mixed: cuBLAS for dense layers, MMQ for small ops

## Phase 2: Qwen4-exp Architecture Integration

### 2.1 Group-of-4 Layer Scheduling

Qwen3.8-Flash-Next alternates layers in groups of 4:
- Layer 0-2: GDN (linear attention recurrence)
- Layer 3: QSA (sparse global attention every 4th layer)

Implementation:
- New model definition: qwen4exp.cpp
- Override layer dispatch to handle GDN/QSA alternation
- QSA layers use standard KV cache; GDN layers use recurrent state

### 2.2 Hyper-Connection Tensors

Replaces traditional layernorms with 4 specialized tensors per block:
- hc_norm: structural normalization scale
- block_inject_weight: direct residual projection
- input_mix_weight_down/up: dynamic gating matrices

Implementation:
- Remove input_layernorm and post_attention_layernorm from block
- Add hyper-connection tensor loading and application
- Fuse with GDN/QSA computation where possible

### 2.3 Ultra-Sparse MoE (512 experts, top-10)

Qwen3.8-Flash-Next specs:
- 512 total experts, each 640 channels wide
- Top-10 routing + 1 shared expert per token
- 6B active parameters per token (vs 125B total MoE weight)

Implementation:
- Extend topk-moe.cu for 512-expert routing
- Integrate coarse-grained batching (Phase 1.5)
- Handle shared expert path separately (always active)

### 2.4 Muon Optimizer Split Handling

Trained with Muon orthogonalizer -> fused weight matrices must be split.

Implementation:
- Ensure quantization preserves Muon partition lines
- Do not merge QKV projections, SwiGLU gates, or GDN layers
- Validate tensor names/structure in GGUF loading

### 2.5 MTP Head Integration (optional)

4B parameter multi-token prediction head for speculative decoding.

Status: llama.cpp's speculative decoding does not yet support qwen4exp MTP.

Action: Monitor upstream, implement when pipeline lands.

## Hardware-Aware Design Decisions

### Why This Hardware Fits

Qwen3.8-Flash-Next is memory-bound, compute-light:
- 6B active params -> fits in 96GB VRAM at IQ3_XXS (~82GB)
- GDN layers avoid KV cache growth -> constant memory per step
- MoE sparsity -> most weights never loaded simultaneously

Kepler GK210 strengths exploited:
- 512KB/SM register file -> GDN state pinning
- 240 GB/s memory bandwidth -> weight streaming for MoE
- 8 independent chips -> layer-split parallelism
- Hyper-Q -> concurrent CUDA streams for async pipeline

Kepler limitations accepted:
- No FP16 compute -> FP32 path throughout
- No Tensor Cores -> no WMMA, no fast INT4
- PCIe Gen3 -> async pipeline hides latency but can't eliminate
- No P2P between K80s -> tensor boundaries via host memory

## File Organization

llama_wukong/
- ARCHITECTURE_REFERENCE.md (original pitch: memory topology, execution pipeline, strategic key points)
- VERIFIED_CONFIG.md (current working config)
- llama_wukong.md (this file)
- PHASE1_TODO.md (detailed implementation tasks)
- RESEARCH/ (supporting research notes)
  - NUMA_REPLICATION.md
  - ASYNC_PIPELINE.md
  - GDN_KERNEL.md
  - ZEROCOPY_ENGRAM.md
  - MOE_BATCHING.md
- scripts/ (build, benchmark, profiling)
  - build_wukong.sh
  - bench_numa.sh
- src/ (custom source files)
  - ggml-cpu-numa-replicate.c
  - ggml-cpu-numa-replicate.h
  - NUMA_INTEGRATION_NOTES.md

## Version Control

- Repo: UnobligatedRascal/llama_wukong (private fork of llama_lazarus)
- Base: UnobligatedRascal/llama_lazarus @ commit 93c888df1
- Branch strategy: master (stable), phase1/* (feature branches), research/* (experiments)
- Commit often, document all changes with reasoning

## NUMA Replication Status (Updated 2026-09-04)

### Findings

Benchmark on NOUGHT dual-socket Xeon (Qwen2.5-0.5B, 36 threads):
- Single-node binding (numactl --cpunodebind=0, 18 threads): **56.7 gen TPS**
- Our --numa mirror implementation: **24.6 gen TPS** (BROKEN - 2.3x slower)
- Root cause: dead code - replication functions defined but never called in compute path

### Recommendation

Use `numactl --cpunodebind=0 --membind=0 -t 18` instead of `--numa mirror`. Proven 2.3x speedup, zero code changes.

Full NUMA replication requires kernel-level changes to call per-node pointer mapping. ROI questionable vs single-node binding.

See NUMA_BENCHMARK_RESULTS.md and NUMA_REPLICATION_FIX.md for details.
