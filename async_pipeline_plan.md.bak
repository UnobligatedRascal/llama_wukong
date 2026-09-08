# Async Pipeline Plan — llama_wukong

UnobligatedRascal | Last updated: 2026-09-06

Standalone reference for the async GPU pipeline on NOUGHT's 8x Tesla K80 (Kepler sm_37).
Covers architecture, integration status, performance, remaining work, and rollback paths.

---

## Status

**Phase: INTEGRATED, RUNNING, OPTIMIZING** (2026-09-06)

- All 6 patches applied to ggml-cuda.cu
- Build: clean (Release, CUDA 11.8, NCCL, cuBLAS)
- Runtime: confirmed working on long-context prompts
- Latest benchmark: ~116 tokens/sec prompt processing
- Trend: improved from ~90 tok/s (before pointer optimization) and improved over prior async run
- Still below pre-async-baseline (~130-140 tok/s) — tuning in progress

## Hardware Topology

NOUGHT: dual Xeon E5-2697v4 (72 cores total), 128GB DDR4 ECC, 8x Tesla K80

| NUMA node | GPUs | Cores | Connection |
|-----------|------|-------|------------|
| NUMA0 | GPU0,1,2,3 | 0-17, 36-53 (36 cores) | PIX within K80, PHB between cards, QPI cross-node |
| NUMA1 | GPU4,5,6,7 | 18-35, 54-71 (36 cores) | PIX within K80, PHB between cards, QPI cross-node |

Key constraints:
- PCIe Gen3 x16 (~14 GB/s practical, ~16 GB/s theoretical)
- No NVLink, no P2P between K80 cards (only within-card via PLX/PIX)
- Cross-NUMA traffic via QPI/UPI — expensive, avoid simultaneous contention
- Each K80 card: 2 independent GK210 chips sharing one PCIe slot

## Architecture

Goal: hide PCIe latency between GPUs via compute/transfer overlap using CUDA streams and events.

### Stream Model (per GPU)

- **Stream 0 (compute):** kernel launches, layer processing
- **Stream 1 (prefetch):** cudaMemcpyAsync for next-layer weights/activations
- **Event:** cudaEventRecord after each layer complete; prefetch stream waits via cudaStreamWaitEvent

Timeline (without pipeline):
```
GPU: [compute layer N] -> [wait PCIe transfer] -> [compute layer N+1]
```

Timeline (with pipeline):
```
GPU compute:  [compute layer N] ----------------> [compute layer N+1]
GPU prefetch:        [prefetch layer N+1] --------> [done]
                    ^ overlap hides PCIe latency
```

### K80 Dual-Core Intra-Card Pipeline

Each K80 card has 2 GK210 chips via PLX switch:

```
Card 0:
  Chip 0 compute:  [token N, layers 0-12] -> event
  Chip 1 prefetch:          [token N+1 weights] waits on event
```

Cross-card: event barriers at tensor boundaries, NCCL stagger for allreduce.

### NUMA-Staggered NCCL AllReduce

Problem: 8-GPU simultaneous allreduce causes QPI contention (NUMA0↔NUMA1).

Solution: sequential NUMA-node allreduce groups:
1. NUMA0 GPUs (0-3) complete allreduce
2. Synchronize NUMA0 completion
3. NUMA1 GPUs (4-7) start their allreduce

Avoids QPI contention by serializing cross-node communication.

NCCL env vars (set by nccl-stagger.cuh if unset):
```
NCCL_P2P_LEVEL=1      # PCIe P2P only (PIX/PHB), no SYS
NCCL_IB_DISABLE=1     # No InfiniBand on NOUGHT
NCCL_SOCKET_NTHREADS=2
NCCL_ALGO=Ring        # Ring preferred for PCIe-only topology
NCCL_STAGGER=1        # Enable NUMA-staggered allreduce
```

## Implementation Files

### Core Headers

| File | Purpose |
|------|---------|
| ggml/src/ggml-cuda/async-pipeline.cuh | Per-GPU stream/event context, lazy init, mark_layer_complete, async_prefetch |
| ggml/src/ggml-cuda/nccl-stagger.cuh | NUMA-grouped allreduce, NCCL env tuning |
| ggml/src/ggml-cuda/numa-gpu-bind.cuh | Thread-to-core pinning by NUMA node |

### async-pipeline.cuh Design

Key structures:
- `ggml_cuda_async_gpu_ctx`: per-GPU prefetch stream, double-buffered events, lazy init
- `ggml_cuda_async_prefetch_desc`: transfer descriptor (dst, src, size, kind)
- `ggml_cuda_async_pipeline`: global context, enables/disables pipeline, NUMA GPU grouping

Lazy init: NO cudaSetDevice() during backend init (deadlock risk). GPU context initializes on first access per device.

Public integration points:
- `ggml_cuda_async_mark_layer_complete(device, compute_stream)` — call after each layer
- `ggml_cuda_async_schedule_prefetch(device, dst, src, size, kind)` — queue async transfer
- `ggml_cuda_async_pipeline_global.enable()` — lightweight flag-only enable

### nccl-stagger.cuh Design

- `ggml_cuda_nccl_init_env()` — set NCCL env vars for K80 topology
- `ggml_cuda_nccl_staggered_allreduce<T>(...)` — NUMA-phase allreduce (NUMA0 then NUMA1)

### numa-gpu-bind.cuh Design

- `ggml_cuda_numa_get_node_for_gpu(gpu_id)` — hardcoded mapping (0-3→NUMA0, 4-7→NUMA1)
- `ggml_cuda_numa_pin_thread_for_gpu(gpu_id, thread_local_id)` — sched_setaffinity to NUMA-local core
- `ggml_cuda_numa_create_mask(numa_node)` — cpu_set_t for NUMA node

## Applied Patches to ggml-cuda.cu

### Patch 1: Include Headers (~line 50)
```cpp
#include "ggml-cuda/async-pipeline.cuh"
#include "ggml-cuda/nccl-stagger.cuh"
#include "ggml-cuda/numa-gpu-bind.cuh"
```

### Patch 2: Global Async Pipeline Instance (~line 680)
```cpp
ggml_cuda_async_pipeline ggml_cuda_async_pipeline_global;
```

### Patch 3: Backend Init (~init function)
After NCCL init:
```cpp
std::vector<int> devices;
for (int i = 0; i < ggml_cuda_info().device_count; i++)
    devices.push_back(i);
ggml_cuda_async_pipeline_global.init(devices);
```

### Patch 4: NCCL Staggered AllReduce (~line 1000-1075)
Replace NCCL group allreduce with staggered NUMA-aware version when:
- n_devices >= 4
- NCCL_STAGGER not explicitly disabled
- Contiguous F32 tensor

### Patch 5: Async Mark After Layer (~line 4100+)
After each node compute in ggml_cuda_compute_forward():
```cpp
ggml_cuda_async_mark_layer_complete(cuda_ctx->device);
```

### Patch 6: NUMA Thread Pinning (~line 1140+)
After NCCL comm creation in ggml_backend_cuda_comm_context_init():
```cpp
for (int i = 0; i < n_devices; i++)
    ggml_cuda_numa_pin_thread_for_gpu(i, i);
```

## Build Configuration

```bash
cd /home/whistler/llama_wukong
rm -rf build && mkdir build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_NCCL=ON \
  -DGGML_CUDA_CUBLAS=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DGGML_CUDA_FORCE_MMQ=ON \
  -DGGML_CUDA_GRAPHS=OFF \
  -DCMAKE_CUDA_ARCHITECTURES="37" \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.8/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=g++-11 \
  -DCMAKE_C_COMPILER=gcc-11 \
  -DCMAKE_CXX_COMPILER=g++-11 \
  -DGGML_CUDA_PEER_MAX_BATCH_SIZE=64 \
  -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-rpath,/usr/local/cuda-11.8/targets/x86_64-linux/lib"
make -j36 llama-server llama-bench
```

Key flags explained:
- GGML_CUDA_NCCL=ON: multi-GPU NCCL communication
- GGML_CUDA_CUBLAS=ON: cuBLAS path (required for Kepler F32)
- GGML_CUDA_GRAPHS=OFF: CUDA graphs not supported on sm_37
- GGML_CUDA_FORCE_MMQ=ON: integer MMQ fallback (benchmark vs cuBLAS path)

## Testing Commands

### Baseline Comparison
```bash
# Standard run
./build/bin/llama-bench -m <model> -t 36 -ngl 99 --tensor-split 1,1,1,1,1,1,1,1 --split-mode tensor ...

# With NUMA-staggered NCCL
NCCL_STAGGER=1 ./build/bin/llama-bench -m <model> -t 36 -ngl 99 --tensor-split 1,1,1,1,1,1,1,1 ...

# NUMA0-only sanity check (18 threads, 1 NUMA node)
numactl --cpunodebind=0 --membind=0 ./build/bin/llama-bench -m <model> -t 18 -ngl 99 --tensor-split 1,1,1,1,1,1,1,1 ...
```

### Nsight Profiling
```bash
nsys profile --trace=cuda,nvtx,osrt,cudnn,cublas,nccl --stats=true \
    --duration=60 -o profile.nsight \
    ./build/bin/llama-bench -m <model> ...
nsys-ui profile.nsight   # Analyze compute/transfer overlap
```

## Performance Observations

### Current State (2026-09-06)
- ~116 tok/s prompt processing (long-context test)
- Improved from ~90 tok/s (after moving pointers to heavy mul/mat ops only)
- Improved over prior async run (async pipeline re-integration working)
- Gap vs pre-async baseline (~130-140 tok/s) indicates tuning opportunity

### Expected Improvements (targeted)
1. NCCL stagger: ~5-15% all-reduce latency reduction (avoid QPI contention)
2. Async pipeline: ~3-8% via PCIe transfer hidden behind compute
3. NUMA binding: ~5-10% host-side overhead reduction

Combined target: ~10-25% improvement on tensor-split 8-GPU workloads.

### Current Gap Analysis
Why ~116 tok/s vs expected ~130+?
- Async prefetch scheduling not yet integrated into actual tensor transfer paths
- `ggml_cuda_async_mark_layer_complete()` called but `async_schedule_prefetch()` not wired into layer loop
- Pipeline is structurally present but not fully exploiting weight pre-fetch opportunities
- Overhead of double-buffered event recording/sync on Kepler (no CUDA graphs to hide)

## Remaining Work

### High Priority
1. **Wire async prefetch into layer loop**: currently mark_layer_complete is called but prefetch is not scheduled for next-layer weights. Need to identify tensor transfer points in ggml_cuda_compute_forward() and call async_schedule_prefetch() there.

2. **Verify NCCL stagger is active**: add runtime logging to confirm staggered path is taken vs fallback standard NCCL.

3. **Nsight profiling**: run nsys to verify compute/transfer overlap is actually happening (not just added latency).

4. **cuBLAS vs MMQ benchmark**: determine optimal compute path for Kepler (GGML_CUDA_FORCE_CUBLAS=ON vs MMQ).

### Medium Priority
5. **Prefetch buffer sizing**: determine optimal pinned host memory for staging buffers (limited by 128GB system RAM).

6. **NUMA replication revisit**: original implementation was broken (2.3x slower). Consider pointer-swap approach per roblee04/numa_llamacpp (proven 1.4-1.75x on dual EPYC).

7. **K80 intra-card pipeline optimization**: exploit PIX connection for true chip-to-chip pipelining within each K80 card.

### Phase 2 (Qwen4 Architecture)
8. GDN kernel integration for Kepler (ggml-cuda/gated_delta_net.cu)
9. Hyper-connection tensor support
10. Ultra-sparse MoE (512 experts, top-10 routing) with coarse-grained batching
11. Zero-copy N-gram table via cudaHostAlloc + cudaMemPrefetchAsync

## Known Limitations

- K80 lacks P2P between cards (only within-card via PLX)
- PCIe Gen3 bandwidth (~14 GB/s) limits async benefit ceiling
- CUDA graphs disabled on Kepler (no graph-based optimization)
- Async pipeline requires contiguous tensor transfers to be effective
- No FP16 compute on sm_37 → FP32 path throughout
- No Tensor Cores → no WMMA, no fast INT4

## Rollback Paths

Runtime disable (no rebuild needed):
```bash
# Disable async pipeline
export GGML_CUDA_ASYNC=0

# Disable NCCL stagger
export NCCL_STAGGER=0

# Fallback to original ggml-cuda.cu
git checkout ggml/src/ggml-cuda/ggml-cuda.cu
```

## Related Documents

- ASYNC_PIPELINE_INTEGRATION.md — patch details and step-by-step integration
- RESEARCH/ASYNC_PIPELINE.md — problem/solution analysis
- llama_wukong.md — overall project plan
- PHASE1_TODO.md — detailed implementation tasks
- VERIFIED_CONFIG.md — current working configuration
- NUMA_BENCHMARK_RESULTS.md — NUMA replication findings

---
END async_pipeline_plan.md — UnobligatedRascal
