# Phase 1 Implementation Tasks

Priority order based on ROI and complexity.

## Task 1: NUMA Replication (P0, COMPLETE - commit 02eb6adec)

### 1.1 Topology Detection
- [x] Add runtime NUMA node detection (libnuma or /sys topology)
- [x] Map CPU cores to NUMA nodes:
  - Node 0: cores 0-17, 36-53 (36 threads)
  - Node 1: cores 18-35, 54-71 (36 threads)
- [x] Expose via GGML_NUMA_REPLICATE compile flag

### 1.2 Weight Replication
- [x] Modify buffer alloc to duplicate weights at load time
- [x] Each NUMA node gets its own aligned_alloc on local memory
- [x] Track per-node base pointers in ggml_backend_buffer_type

### 1.3 Thread Pinning
- [x] Bind threads to NUMA-local cores at startup
- [x] Use pthread_setaffinity_np or sched_setaffinity
- [x] Thread pool split: 18 threads per node (physical cores, no HT)

### 1.4 Per-Thread Pointer Swap
- [x] In ggml_compute_forward_mul_mat: swap src->data pointer based on thread's NUMA node
- [x] Use ggml_compute_params.ith to determine which copy to read
- [x] Single-cycle address translation, zero extra latency

### 1.5 NUMA-Local wdata
- [x] Create per-node wdata buffers for quantized dequantization
- [x] Each node quantizes src1 into its own local buffer

### 1.6 Verification
- [x] Build with -DGGML_NUMA_REPLICATE=ON
- [x] Benchmark vs baseline (same model, same config)
- [x] Result: numa_mirror=61.0 TPS matches numactl_node0=61.4 TPS (was 24.6 TPS broken)

## Task 2: TurboQuant Integration (P0, 2-4 weeks)

**References:**
- TheTom/turboquant_plus (research: Python reference impl)
- atomicmilkshake/llama-cpp-turboquant (llama.cpp fork with CUDA kernels)

**Local clones:** `<path>/turboquant_plus` and `<path>/llama-cpp-turboquant`

**K80 note:** Fork targets Turing+ (sm_75+); need sm_37 FP32 fallback audit. See RESEARCH/TURBOQUANT_TRIATTENTION_RESEARCH.md

### 2.1 Research & Compatibility Audit
- [x] Clone and review TurboQuant repo
- [x] Identify integration points with ggml quantization backend
- [x] Assess compatibility with current GGUF format and K80 sm_37 target
- [x] Evaluate speed/accuracy tradeoffs vs current quant methods
- [x] **AUDIT COMPLETE:** All kernels are FP32, no tensor cores, no sm_75+ intrinsics. Fully sm_37 compatible.
- [x] See RESEARCH/TURBOQUANT_SM37_AUDIT.md for details

### 2.2 Backend Integration
- [x] Add TurboQuant KV cache types (GGML_TYPE_TURBO3_0/4_0/2_0)
- [x] Wire into ggml type registration (ggml.c: to_float/from_float_ref)
- [x] Add turbo types to set_rows CUDA kernel (KV cache writes)
- [x] Add turbo dequant functions to dequantize.cuh
- [x] Wire turbo types into ggml-cuda.cu (MUL_MAT, GET_ROWS, SET_ROWS)
- [x] Add turbo types to kv_cache_types in arg.cpp (--cache-type-k/--cache-type-v)
- [x] Add turbo types to llama-bench.cpp type parser
- [x] Standardize turbo4_0 as 4-bit PolarQuant (no QJL, 68-byte block)
- [x] Verified: CLI accepts turbo2_0/turbo3_0/turbo4_0; symbols exported from libggml-cuda.so
- [ ] CPU set_rows support for turbo types (blocked - CPU path not needed for GPU KV cache)

### 2.3 Dequantization Kernels
- [x] Write or adapt TurboQuant dequant kernels for sm_37
- [x] Ensure MMQ/cuBLAS compatibility with TurboQuant format
- [ ] Optimize for Kepler register/shared-memory constraints

### 2.4 Verification
- [ ] Convert test models to TurboQuant, run llama-bench (blocked - all GPUs in production use)
- [ ] Compare accuracy vs baseline quantization
- [ ] Compare inference speed and memory usage
- [ ] Document in RESEARCH/TURBOQUANT_EVAL.md

## Task 3: TriAttention Efficient Context Pruning (P1, COMPLETE - 2026-09-08)

**References:**
- domvox/triattention-ggml (standalone HIP/ROCm implementation)
- atomicmilkshake/llama-cpp-turboquant (has CUDA triattention-score.cu integrated)

**Local clone:** `<path>/triattention-ggml`

**K80 note:** GPU kernel works on sm_37 (no tensor cores needed). See RESEARCH/TURBOQUANT_TRIATTENTION_RESEARCH.md

### 3.1 Research & Design
- [x] Clone and review TriAttention implementation (atomicmilkshake fork)
- [x] Understand pruning mechanism and attention recomputation strategy
- [x] Assess fit for long-context workloads on NOUGHT hardware
- [x] Map integration points in llama.cpp attention path
- [x] sm_37 audit: GPU kernel uses only FP32 math, no tensor cores; fully compatible

### 3.2 Kernel Implementation
- [x] GPU scoring kernel: triattention-score.cu (542 lines, full implementation)
- [x] GPU API: triattention_gpu_init/score_head/free/etc. declared in ggml-cuda.h
- [x] GPU kernel supports TurboQuant types (turbo2/3/4 dequant paths with WHT)
- [x] Symbols exported from libggml-cuda.so
- [x] CPU-side llama-triattention.h/cpp (loader, RoPE inversion, pruning pipeline)
- [x] Multi-GPU tensor-split coordination (handled via per-GPU tensor pointers)

### 3.3 Context Window Integration
- [x] CPU-side llama-triattention.h/cpp (loader, RoPE inversion, pruning pipeline)
- [x] Wire pruning hook into llama-kv-cache.cpp update loop
- [x] Add CLI flags (--triattention-stats, --triattention-budget, etc.)
- [x] Calibration tool: uses triattention_calibrate.py from triattention-ggml
- [x] Ensure backward compatibility (disabled by default, requires --triattention-stats)

### 3.4 Verification
- [ ] Test on long-context prompts, compare quality vs full attention
- [ ] Benchmark memory savings and latency improvement
- [ ] Verify correctness on known attention-sensitive tasks
- [ ] Document in RESEARCH/TRIATTENTION_EVAL.md

### Implementation Notes
- See RESEARCH/TRIATTENTION_REVIEW.md for comprehensive review
- GPU-first design with graceful CPU fallback
- Dual protection: prefix tokens + recent divide_length tokens
- Lazy GPU init on first prune
- Compatible with TurboQuant KV cache types

## Task 4: FlashAttention / SlidingWindowAttention for NOUGHT (P1, 1-3 weeks)

**Reference:** Existing FA in llama_wukong (`ggml/src/ggml-cuda/fattn*.cu`), FAIR FA3 patterns

**K80 status:** FA tile/vec kernels should work on sm_37 (no tensor cores needed); need to verify compilation and runtime. No dedicated SWA yet — implement via attention masks first. See RESEARCH/TURBOQUANT_TRIATTENTION_RESEARCH.md

### 4.1 Feasibility Assessment
- [ ] Review current FlashAttention support in ggml-cuda (ggml-cuda/flash-attn path)
- [ ] Test existing FA implementation on NOUGHT (sm_37, 8x K80)
- [ ] Identify blockers: sm_80+ requirements, warp-level primitives, shared mem limits

### 4.2 Sliding Window Attention (SWA)
- [ ] Implement SWA as fallback if full FA incompatible:
  - Local window (e.g., 4096 tokens) with full attention
  - Global tokens (every Nth or selected) for long-range
- [ ] Wire into ggml-cuda attention kernels

### 4.3 Optimized FA Path (if viable)
- [ ] If FA works but suboptimal: tune tiling/block sizes for GK210
- [ ] Reduce register pressure, adapt to Kepler shared-mem limits
- [ ] If FA fully incompatible: document why, use SWA as primary

### 4.4 Verification
- [ ] Benchmark FA/SWA vs current attention on NOUGHT
- [ ] Compare quality on long-context tasks
- [ ] Measure memory bandwidth improvement
- [ ] Document in RESEARCH/FLASHATTENTION_NOUGHT.md

## Task 5: Async Multi-GPU Pipeline (P3, 2-3 weeks)

*Deprioritized from P1; revisit after Tasks 2-4 complete.*

### 5.1 Per-Card Stream Pairs
- [ ] For each K80 card, create 2 CUDA streams:
  - Stream A: compute (kernel launches)
  - Stream B: prefetch (cudaMemcpyAsync for next layer)
- [ ] Track streams per GPU device in ggml_cuda_context

### 5.2 Event-Based Synchronization
- [ ] Record cudaEvent after each layer compute completes
- [ ] Prefetch stream waits on event before starting transfer
- [ ] Compute stream starts when previous layer's data ready

### 5.3 K80 Dual-Core Pipeline
- [ ] Exploit PLX switch: Chip 0 computes while Chip 1 pre-fetches
- [ ] Use cudaMemcpyAsync between chip memories (via unified addressing)

### 5.4 8-GPU Layer Distribution
- [ ] Verify current tensor-split distributes layers evenly
- [ ] Add event barriers at tensor boundaries between GPUs

### 5.5 Verification
- [ ] Profile with Nsight Systems: verify compute/transfer overlap
- [ ] Compare latency with and without async pipeline

## Task 6: GDN Kernel sm_37 Verification (P2, 1-2 weeks)

### 6.1 Audit gated_delta_net.cu
- [ ] Check for sm_80+ specific instructions (WMMA, async copy)
- [ ] Verify shared memory usage fits Kepler limits (48KB per block max)
- [ ] Ensure no BF16/TF32 paths are active

### 6.2 Register Pressure Analysis
- [ ] Calculate register usage per thread for GDN step
- [ ] Target: < 63 registers/thread for 100% occupancy on GK210
- [ ] Use __launch_bounds__ to control occupancy

### 6.3 Shared Memory Optimization
- [ ] Pin recurrent state matrix S in shared memory per block
- [ ] Use __shared__ array, load once, update in-place each step

### 6.4 Verification
- [ ] Build and run with GDN-containing model
- [ ] Compare output against CPU reference (bit-verify correctness)

## Task 4: Zero-Copy N-Gram Table (DEFERRED - FUTURE WORK)

> Moved to future project. Qwen4-exp architecture work (including 51B N-gram engram table, GDN/QSA scheduling, hyper-connection tensors) will be handled in a separate codebase. llama_wukong now focuses on NOUGHT hardware optimization for existing model architectures.

### 7.1 Host Allocation
- [ ] cudaHostAlloc for 51B N-gram table (page-locked)
- [ ] Map from GGUF file via mmap first, then pin needed regions

### 7.2 Prefetch Strategy
- [ ] cudaMemPrefetchAsync for bigram/trigram hash regions
- [ ] Prefetch ahead: predict needed regions from current token sequence

### 7.3 Lookup Kernel
- [ ] New kernel: engram_lookup(hash, table_ptr, out_vector)
- [ ] Streams vector from host memory over PCIe on-demand
- [ ] Only activate on bigram/trigram match tokens

### 7.4 Verification
- [ ] Profile PCIe bandwidth usage
- [ ] Verify lookup correctness against full VRAM baseline

## Task 5: MoE Expert Batching (DEFERRED - FUTURE WORK)

> Moved to future project. 512-expert ultra-sparse MoE batching is Qwen4-exp specific. Deferred to separate project handling that architecture.

### 8.1 Routing Sort
- [ ] Modify topk-moe.cu output to include sorted expert indices
- [ ] Group tokens by target expert (top-10 of 512 per token)

### 8.2 Batch Dispatch Kernel
- [ ] New kernel: expert_batch(expert_id, token_indices, weights)
- [ ] Stream expert weights once, process all tokens for that expert
- [ ] Release weights, move to next expert

### 8.3 Verification
- [ ] Compare against unbatched MoE (same results, faster)
- [ ] Measure weight load reduction

## Task 9: cuBLAS vs MMQ Benchmarking (P2, 3-5 days)

### 9.1 Baseline
- [ ] Run current config (MMQ path) with llama-bench
- [ ] Record tokens/s, latency, memory bandwidth

### 9.2 cuBLAS F32
- [ ] Rebuild with -DGGML_CUDA_FORCE_CUBLAS=ON
- [ ] Run same benchmark
- [ ] Compare results

### 9.3 Analysis
- [ ] Determine winner for Kepler sm_37
- [ ] Document findings in RESEARCH/CUBLAS_VS_MMQ.md

## Build Scripts

### build_wukong.sh
```bash
#!/bin/bash
cd <project-root>
rm -rf build && mkdir build && cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_F16=ON \
  -DCMAKE_CUDA_HOST_COMPILER=g++-11 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.8/bin/nvcc \
  -DGGML_CUDA_NCCL=ON \
  -DCMAKE_CUDA_ARCHITECTURES="37" \
  -DLLAMA_CURL=OFF \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DGGML_CUDA_FORCE_MMQ=ON \
  -DGGML_CUDA_GRAPHS=OFF \
  -DCMAKE_C_COMPILER=gcc-11 \
  -DCMAKE_CXX_COMPILER=g++-11 \
  -DGGML_CUDA_CUBLAS=ON \
  -DCMAKE_C_FLAGS="-DGGML_NUMA_REPLICATE" \
  -DCMAKE_CXX_FLAGS="-DGGML_NUMA_REPLICATE" \
  -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-rpath,/usr/local/cuda-11.8/targets/x86_64-linux/lib"

make -j36 llama-server llama-bench
```

## Notes

- DO NOT kill running llama-server (hosts this session)
- Test all changes with scripts, not interactive sessions
- Commit after each completed task
- Profile before and after each optimization
- 2026-09-05: Tasks 4 & 5 deferred to future project. Qwen4-exp architecture work (N-gram engram, 512-expert MoE, GDN/QSA, hyper-connection tensors) moved out of scope. llama_wukong now focuses on NOUGHT hardware optimization (NUMA, async pipeline, Kepler kernels, cuBLAS tuning) for existing model architectures.
