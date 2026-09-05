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

## Notes

- DO NOT kill running llama-server (hosts this session)
- Test all changes with scripts, not interactive sessions
- Commit after each completed task
- Profile before and after each optimization
