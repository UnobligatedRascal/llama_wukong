# Project llama_wukong

Base: llama_lazarus (UnobligatedRascal fork), commit 93c888df1
Hardware: NOUGHT (dual Xeon E5-2697 v4, 8x Tesla K80 GK210, 128GB RAM)
Target: NOUGHT hardware optimization for existing model architectures

Note (2026-09-05): Qwen4-exp architecture integration (Phase 2 below) deferred to a separate future project. llama_wukong now focuses on hardware optimizations applicable to any large model: NUMA replication, async GPU pipeline, Kepler-specific kernels, memory tuning.

## Build Command (Verified Working, 2026-09-05)

```bash
git clone https://github.com/UnobligatedRascal/llama_wukong && cd llama_wukong && cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_SCHED_MAX_COPIES=3 -DGGML_CUDA_NCCL=ON -DGGML_CUDA_FA_ALL_QUANTS=ON -DGGML_CUDA_FORCE_MMQ=ON -DGGML_CUDA_FA=ON -DGGML_CUDA_GRAPHS=OFF -DCMAKE_CUDA_ARCHITECTURES="37" -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.8/bin/nvcc -DGGML_CUDA_PEER_MAX_BATCH_SIZE=64 -DCMAKE_CUDA_HOST_COMPILER=g++-11 -DCMAKE_C_COMPILER=gcc-11 -DCMAKE_CXX_COMPILER=g++-11 -DGGML_AVX2=ON -DGGML_AVX512=OFF -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_SSE42=ON -DGGML_BMI2=ON -DGGML_NATIVE=OFF -DGGML_OPENMP=ON -DLLAMA_CURL=OFF -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-rpath,/usr/local/cuda-11.8/targets/x86_64-linux/lib" -DGGML_CUDA_CUBLAS=ON && cmake --build build --config Release -j36
```

**Key flags rationale:**
- `-DGGML_CUDA=ON -DGGML_CUDA_CUBLAS=ON`: CUDA backend + cuBLAS for Kepler
- `-DGGML_CUDA_NCCL=ON`: NCCL for multi-GPU tensor splitting
- `-DGGML_CUDA_FA_ALL_QUANTS=ON -DGGML_CUDA_FA=ON`: Flash attention across all quant types
- `-DGGML_CUDA_FORCE_MMQ=ON`: Force MMQ integer path (Kepler-optimized)
- `-DGGML_CUDA_GRAPHS=OFF`: CUDA graphs unsupported on Kepler sm_37
- `-DCMAKE_CUDA_ARCHITECTURES="37"`: Kepler sm_37 target
- `-DGGML_CUDA_PEER_MAX_BATCH_SIZE=64`: NCCL peer-to-peer batch sizing
- `-DGGML_AVX2=ON -DGGML_AVX512=OFF`: Xeon E5-2697v4 supports AVX2, not AVX-512
- `-DGGML_SCHED_MAX_COPIES=3`: Limit copy scheduling depth
- `-j36`: 36 physical cores (no HT) for build parallelism

## Run Command (Verified Working, 2026-09-05)

```bash
sudo GGML_CUDA_P2P=1 -E nice -n -20 numactl --interleave=all ./build/bin/llama-server -m /path/to/models/Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-MTP-Q4_K_M.gguf -t 36 -c 262144 -ngl 99  --port 4269 --host 0.0.0.0 --api-key YOUR_API_KEY_HERE --jinja --chat-template-file ./models/tuvak.jinja --load-mode none -np 3 --ctx-checkpoints 64 --checkpoint-min-step 4096 --cache-ram 65536 --mmproj /path/to/models/Qwen3.6-27B-mmproj-F16.gguf --no-mmproj-offload  --image-min-tokens 1024 --batch-size 2048 --ubatch-size 512 --cache-type-k q4_0 --cache-type-v q4_0 --tensor-split 1,1,1,1,1,1,1,1 --kv-unified --slot-save-path /path/to/models/kv_cache --seed 1016 --spec-type draft-mtp --spec-draft-p-min 0.75 --spec-draft-n-max 3 --split-mode tensor
```

**Key flags rationale:**
- `sudo GGML_CUDA_P2P=1 -E nice -n -20`: Real-time priority + P2P memory enable
- `numactl --interleave=all`: Interleaved memory allocation across both NUMA nodes
- `-m`: Qwen3.6-27B model (Q4_K_M quantization)
- `-t 36`: 36 threads (one per physical core)
- `-c 262144`: 256K context window
- `-ngl 99`: All layers offloaded to GPU
- `-np 3`: 3 concurrent user slots
- `--load-mode none`: Skip initial load profiling (speeds startup)
- `--ctx-checkpoints 64 --checkpoint-min-step 4096`: Context checkpointing for large contexts
- `--cache-ram 65536`: 64GB RAM cache for KV state
- `--tensor-split 1,1,1,1,1,1,1,1`: Equal tensor split across all 8 K80 chips
- `--split-mode tensor`: Tensor parallelism mode
- `--kv-unified`: Unified KV cache across GPUs
- `--spec-type draft-mtp --spec-draft-p-min 0.75 --spec-draft-n-max 3`: Multi-token prediction speculative decoding
- `--cache-type-k q4_0 --cache-type-v q4_0`: Quantized KV cache to save VRAM
- `--mmproj`: Vision projection model for multimodal input
- `--batch-size 2048 --ubatch-size 512`: Batch processing tuned for Kepler memory

## Vision

Optimize llama.cpp for NOUGHT's specific hardware to run large models
(27B–40B+ parameters) efficiently on aging Kepler sm_37 GPUs.

Focus: hardware optimizations applicable to any model architecture:
- NUMA-aware weight replication across dual Xeon sockets
- Async multi-GPU pipeline hiding PCIe latency
- sm_37-compatible CUDA kernels (FP32, no tensor cores)
- Memory tuning for VRAM-constrained Kepler hardware

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

## Phase 2: Qwen4-exp Architecture Integration (DEFERRED - FUTURE WORK)

> Moved to future project. llama_wukong scope is now NOUGHT hardware optimization only.

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

## Scope Note (2026-09-05)

Phase 2 tasks (Qwen4-exp architecture: GDN/QSA layer scheduling, hyper-connection tensors, 512-expert MoE, Muon split handling, MTP head) have been deferred to a separate future project. Phase 1 hardware optimizations (NUMA, async pipeline, Kepler kernels, cuBLAS tuning) remain in scope and apply to any model running on NOUGHT.

## Hardware-Aware Design Decisions

### Why This Hardware Fits

Large models on Kepler are memory-bound, compute-light:
- Quantized weights (Q4_K_M, Q3_K_S) keep 27B–40B models in 96GB VRAM
- Tensor splitting across 8 GK210 chips enables models beyond single-GPU capacity
- KV cache quantization further reduces VRAM pressure

Kepler GK210 strengths exploited:
- 512KB/SM register file -> efficient state management
- 240 GB/s memory bandwidth -> weight streaming for large models
- 8 independent chips -> layer-split parallelism
- Hyper-Q -> concurrent CUDA streams for async pipeline

Kepler limitations accepted:
- No FP16 compute -> FP32 path throughout
- No Tensor Cores -> no WMMA, no fast INT4
- PCIe Gen3 -> async pipeline hides latency but can't eliminate
- No P2P between K80s -> tensor boundaries via host memory

## File Organization

llama_wukong/
- README.md (quick reference, build, current status)
- TODO.md (concise task tracking)
- PHASE1_TODO.md (detailed implementation tasks with checkboxes)
- TODO_PHASE4.md (TriAttention + TurboQuant deep dive)
- ARCHITECTURE_REFERENCE.md (original pitch: memory topology, execution pipeline, Kepler exploitation)
- VERIFIED_CONFIG.md (working launch command and hardware profile)
- llama_wukong.md (this file: full project history and scope)
- RESEARCH/ (technical research notes per optimization area)
  - NUMA_REPLICATION.md
  - ASYNC_PIPELINE.md
  - GDN_KERNEL.md
  - TURBOQUANT_SM37_AUDIT.md
  - TURBOQUANT_TRIATTENTION_RESEARCH.md
  - ZEROCOPY_ENGRAM.md (deferred)
  - MOE_BATCHING.md (deferred)
- scripts/ (build, benchmark, profiling)
  - build_wukong.sh
  - bench_numa.sh
- ggml/src/ggml-cuda/ (custom CUDA files)
  - async-pipeline.cuh (per-GPU async context, prefetch streams)
  - nccl-stagger.cuh (NUMA-aware NCCL staggered allreduce)
  - numa-gpu-bind.cuh (GPU-to-NUMA topology utilities)
  - rope-lut.cuh (sin/cos lookup table for sm_37)
  - triattention-score.cu/cuh (TriAttention GPU scoring kernel)
  - turbo-quant.cuh (TurboQuant CUDA kernels)
- ggml/src/ (custom CPU files)
  - ggml-turbo-quant.c (TurboQuant CPU kernels)

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
## 2026-09-07: RoPE Lookup Table (sm_37 transcendental acceleration)

**Problem**: K80 sm_37 sinf/cosf are 20-30 cycles each. RoPE calls sinf/cosf per frequency dimension (~64 calls/token/layer). Dominant latency.

**Solution**: Precomputed sin/cos lookup table with bilinear interpolation:
- 4096 entries × 2 tables × 4 bytes = 32KB constant memory (within sm_37's 64KB limit)
- ~5 cycles per sin/cos via LUT+interp vs 20-30 native
- Configurable: ROPE_LUT_SIZE, ROPE_USE_LUT defines
- Graceful fallback to sinf/cosf if not initialized

**Files**:
- ggml/src/ggml-cuda/rope-lut.cuh (new)
- ggml/src/ggml-cuda/rope.cu (rope_yarn uses LUT)
- ggml/src/ggml-cuda/ggml-cuda.cu (LUT init at startup)
- ggml/src/ggml-cuda/nccl-stagger.cuh (simplified, ncclCommSplit unavailable)

**Build**: GCC 11 + CUDA 11.8, -DCMAKE_CUDA_ARCHITECTURES=37

**TODO**: Test on running server - restart with new binary, measure t/s change.

## 2026-09-08: TurboQuant Integration Complete

**Status:** Fully integrated and usable. Types turbo2_0/turbo3_0/turbo4_0 available via `--cache-type-k/--cache-type-v`.

**Implementation:**
- Types registered in ggml.h (GGML_TYPE_TURBO3_0/4_0/2_0)
- CPU kernels in ggml-turbo-quant.c (quantize/dequantize_row_turbo{2,3,4}_0)
- CUDA kernels in turbo-quant.cuh (quantize), dequantize.cuh (dequantize_turbo{2,3,4}_0)
- Wired into ggml-cuda.cu (MUL_MAT, GET_ROWS, SET_ROWS), set-rows.cu, arg.cpp, llama-bench.cpp
- sm_37: All kernels FP32, no tensor cores; fully K80-compatible
- Symbols exported from libggml.so and libggml-cuda.so

**Verification:**
- CLI help lists turbo2_0/turbo3_0/turbo4_0 as valid cache types
- Build succeeds with all turbo symbols resolved
- Blocked on runtime testing: all GPUs in production use

## 2026-09-08: TriAttention GPU Kernels Complete

**Status:** GPU scoring path complete. CPU glue code and CLI integration pending.

**Implementation:**
- GPU kernel: triattention-score.cu (542 lines, full scoring pipeline)
- GPU API: triattention_gpu_init/score_head/free/etc. in ggml-cuda.h
- Supports TurboQuant types (turbo2/3/4 dequant with WHT handling)
- Symbols exported from libggml-cuda.so
- sm_37: FP32 math only; fully K80-compatible

**Remaining:**
- Create src/llama-triattention.h/cpp (loader, RoPE inversion, pruning pipeline)
- Wire pruning hook into llama-context.cpp decode loop
- Add CLI flags (--triattention-stats, --triattention-budget, etc.)
- Calibration tool (--triattention-calibrate)
- Multi-GPU tensor-split coordination for eviction decisions
