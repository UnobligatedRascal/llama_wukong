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

## Task 2: Async Multi-GPU Pipeline (P1, 2-3 weeks)

### 2.1 Per-Card Stream Pairs
- [ ] For each K80 card, create 2 CUDA streams:
  - Stream A: compute (kernel launches)
  - Stream B: prefetch (cudaMemcpyAsync for next layer)
- [ ] Track streams per GPU device in ggml_cuda_context

### 2.2 Event-Based Synchronization
- [ ] Record cudaEvent after each layer compute completes
- [ ] Prefetch stream waits on event before starting transfer
- [ ] Compute stream starts when previous layer's data ready

### 2.3 K80 Dual-Core Pipeline
- [ ] Exploit PLX switch: Chip 0 computes while Chip 1 pre-fetches
- [ ] Use cudaMemcpyAsync between chip memories (via unified addressing)

### 2.4 8-GPU Layer Distribution
- [ ] Verify current tensor-split distributes layers evenly
- [ ] Add event barriers at tensor boundaries between GPUs

### 2.5 Verification
- [ ] Profile with Nsight Systems: verify compute/transfer overlap
- [ ] Compare latency with and without async pipeline

## Task 3: GDN Kernel sm_37 Verification (P1, 1-2 weeks)

### 3.1 Audit gated_delta_net.cu
- [ ] Check for sm_80+ specific instructions (WMMA, async copy)
- [ ] Verify shared memory usage fits Kepler limits (48KB per block max)
- [ ] Ensure no BF16/TF32 paths are active

### 3.2 Register Pressure Analysis
- [ ] Calculate register usage per thread for GDN step
- [ ] Target: < 63 registers/thread for 100% occupancy on GK210
- [ ] Use __launch_bounds__ to control occupancy

### 3.3 Shared Memory Optimization
- [ ] Pin recurrent state matrix S in shared memory per block
- [ ] Use __shared__ array, load once, update in-place each step

### 3.4 Verification
- [ ] Build and run with GDN-containing model
- [ ] Compare output against CPU reference (bit-verify correctness)

## Task 4: Zero-Copy N-Gram Table (P1, 1-2 weeks)

### 4.1 Host Allocation
- [ ] cudaHostAlloc for 51B N-gram table (page-locked)
- [ ] Map from GGUF file via mmap first, then pin needed regions

### 4.2 Prefetch Strategy
- [ ] cudaMemPrefetchAsync for bigram/trigram hash regions
- [ ] Prefetch ahead: predict needed regions from current token sequence

### 4.3 Lookup Kernel
- [ ] New kernel: engram_lookup(hash, table_ptr, out_vector)
- [ ] Streams vector from host memory over PCIe on-demand
- [ ] Only activate on bigram/trigram match tokens

### 4.4 Verification
- [ ] Profile PCIe bandwidth usage
- [ ] Verify lookup correctness against full VRAM baseline

## Task 5: MoE Expert Batching (P2, 1-2 weeks)

### 5.1 Routing Sort
- [ ] Modify topk-moe.cu output to include sorted expert indices
- [ ] Group tokens by target expert (top-10 of 512 per token)

### 5.2 Batch Dispatch Kernel
- [ ] New kernel: expert_batch(expert_id, token_indices, weights)
- [ ] Stream expert weights once, process all tokens for that expert
- [ ] Release weights, move to next expert

### 5.3 Verification
- [ ] Compare against unbatched MoE (same results, faster)
- [ ] Measure weight load reduction

## Task 6: cuBLAS vs MMQ Benchmarking (P2, 3-5 days)

### 6.1 Baseline
- [ ] Run current config (MMQ path) with llama-bench
- [ ] Record tokens/s, latency, memory bandwidth

### 6.2 cuBLAS F32
- [ ] Rebuild with -DGGML_CUDA_FORCE_CUBLAS=ON
- [ ] Run same benchmark
- [ ] Compare results

### 6.3 Analysis
- [ ] Determine winner for Kepler sm_37
- [ ] Document findings in RESEARCH/CUBLAS_VS_MMQ.md

## Build Scripts

### build_wukong.sh
```bash
#!/bin/bash
cd /home/whistler/llama_wukong
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

## Task 7: TurboQuant KV Cache Compression (P1.5, 2-3 weeks)

### 7.1 Foundation Audit
- [ ] Verify upstream Hadamard rotation is present (PR #21038 merged)
- [ ] Verify CPU WHT kernels present (PR #22631 merged)
- [ ] Verify CUDA WHT kernels present (PR #23615 merged) — audit sm_37 compatibility
- [ ] Read RESEARCH/TURBOQUANT.md for architecture understanding

### 7.2 CUDA WHT sm_37 Adaptation
- [ ] Audit ggml-cuda.cu WHT kernels for sm_80+ instructions
- [ ] Implement sm_37-compatible WHT:
  - Butterfly ops are architecture-neutral, should work
  - Shared memory: max 48KB/block on Kepler
  - No warp shuffle; use shared memory for inter-thread comms
  - Block size: 256 threads per WHT operation (fits GK210)
- [ ] Test: wht_forward/wht_inverse on known vectors, verify against CPU reference

### 7.3 PolarQuant Core Implementation
- [ ] Add turbo cache type enums to kv_cache.h:
  - GGML_KV_CACHE_TURBO4, TURBO3, TURBO2
- [ ] Implement centroid tables (precomputed Lloyd-Max for N(0,1/d)):
  - turbo4: 16 centroids
  - turbo3: 8 centroids
  - turbo2: 4 centroids
- [ ] Implement rotation matrix storage (D₁, D₂ sign vectors per head dim)
- [ ] Implement block_size=128 memory layout:
  - Per block: FP32 norm (4B) + bit-packed indices
  - turbo3 indices: 3 bits per coordinate

### 7.4 Quantize (Store) Path
- [ ] CPU quantize kernel:
  - Extract norm, normalize, WHT rotate, nearest-centroid lookup, pack
  - Block-by-block over KV cache vectors
- [ ] CUDA quantize kernel (sm_37):
  - Each thread block handles one head's block of 128 tokens
  - Use shared memory for WHT butterfly stages
  - Output: packed indices + norms per block

### 7.5 Dequantize (Load) Path
- [ ] CPU dequantize kernel:
  - Unpack indices, centroid lookup, inverse WHT, rescale by norm
- [ ] CUDA dequantize kernel (sm_37):
  - Fuse dequantize + attention compute where possible
  - Centroid LUT in shared memory
  - Inverse WHT via Hᵀ (same butterfly, reverse sign application)

### 7.6 Asymmetric K/V Support
- [ ] Implement -ctk / -ctv flags for independent K/V cache types
- [ ] Ensure cache size calculation handles mixed types
- [ ] Priority: q8_0-K + turbo4-V (proven safe on all tested hardware)
- [ ] Then: turbo4-K + turbo4-V for large tolerant models

### 7.7 Sparse V Dequant
- [ ] Implement attention-weight thresholding (<1e-6 → skip dequant)
- [ ] Not TurboQuant-specific: works with any V cache type
- [ ] +22.8% decode on MoE at 32K (TheTom's measured)

### 7.8 Boundary V (Auto for turbo2)
- [ ] First 2 + last 2 layers: q8_0-V
- [ ] Middle layers: turbo2-V
- [ ] Auto-enable when -ctv turbo2 is set
- [ ] 37-91% quality recovery vs pure turbo2

### 7.9 Integration and Testing
- [ ] Build with TurboQuant support enabled
- [ ] Test sequence (conservative to aggressive):
  1. `-ctk q8_0 -ctv q8_0` — baseline, ensure no regression
  2. `-ctk q8_0 -ctv turbo4` — asymmetric, should work on any model
  3. `-ctk q8_0 -ctv turbo3` — more V compression
  4. `-ctk turbo4 -ctv turbo4` — symmetric on tolerant models
- [ ] PPL validation: llama-perplexity vs q8_0 baseline
- [ ] Speed validation: llama-bench at pp512, pp8192, pp32768
- [ ] NIAH retrieval test: verify long-context quality

### 7.10 Kepler-Specific Optimizations
- [ ] Register pressure analysis for dequant kernel (<63 reg/thread)
- [ ] Shared memory usage for centroid LUT (fits 48KB)
- [ ] Evaluate: fused FA with turbo dequant vs separate passes
- [ ] Consider: cuBLAS F32 path for dequant-heavy ops

## Notes

- DO NOT kill running llama-server (hosts this session)
- Test all changes with scripts, not interactive sessions
- Commit after each completed task
- Profile before and after each optimization
- See RESEARCH/TURBOQUANT.md for deep architectural reference

## Task Dependencies

- Task 7 can proceed in parallel with Tasks 2-4
- Task 7.10 depends on Task 6 (cuBLAS vs MMQ results)
- Sparse V (7.7) has no dependencies, implement after core turbo3 works
- atomicmilkshake's TriAttention is NOT required; orthogonal feature

## Recommended Source Order

1. Start with TheTom's fork as reference: https://github.com/TheTom/llama-cpp-turboquant
2. Python reference for algorithm validation: https://github.com/TheTom/turboquant_plus
3. Upstream llama.cpp WHT kernels first (already merged, verify sm_37 works)
4. Implement PolarQuant on top of existing WHT infrastructure
5. Test asymmetric first (q8_0-K + turbo-V) — proven safe across hardware

## Expected ROI for NOUGHT

- Qwen3.8-Flash-Next: 75% GDN layers (no KV), 25% QSA layers (KV cache)
- TurboQuant compresses only the QSA KV cache (~16 of 64 layers)
- At 128K context: ~5x KV cache reduction on those layers
- Combined with GDN's constant memory: enables longer contexts than fp16 KV
- Sparse V dequant: decode speedup regardless of compression level
- Memory savings feed into Phase 2: more VRAM for MoE weight streaming

## Build Scripts

### build_wukong.sh
```bash
#!/bin/bash
cd /home/whistler/llama_wukong
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
- See RESEARCH/TURBOQUANT.md for deep architectural reference

---
UnobligatedRascal
