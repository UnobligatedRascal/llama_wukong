# llama_wukong - Task Tracking

## Current Status: BUILD COMPLETE — TurboQuant READY, TriAttention GPU DONE

**Last updated:** 2026-09-08

---

## Completed

### Phase 1: NUMA Replication ✓
- NUMA-aware weight replication for ggml-cpu
- Per-node weight pointers in kernels
- Benchmark: 2.5× speedup on dual-socket (vs cross-NUMA baseline)
- Files: ggml/src/ggml-cpu/ggml-cpu-numa-replicate.c

### Phase 2: Async GPU Pipeline ✓
- async-pipeline.cuh: Per-GPU lazy-init context, prefetch streams, double-buffered events
- nccl-stagger.cuh: NUMA-aware NCCL env init for staggered allreduce
- numa-gpu-bind.cuh: GPU-to-NUMA binding utilities
- Integration in ggml-cuda.cu: Global context, enable() in backend init, mark_layer_complete() on MUL_MAT ops only
- Defensive null checks prevent crashes during lazy-init races
- Restricted to heavy ops (MUL_MAT) to avoid ~35% regression on lightweight ops

### Phase 3: fit.cpp Tensor-Split Support ✓
- Tensor-split mode works with --fit path
- Memory targets validated per-device after context reduction
- Clear error if tensor-split model can't fit (no layer redistribution possible)

### Phase 4: TurboQuant KV Cache ✓ (Task 2.1 + 2.2)
- Types: GGML_TYPE_TURBO3_0/4_0/2_0 registered in ggml.h
- CPU: quantize/dequantize_row_turbo{2,3,4}_0 in ggml-turbo-quant.c
- CUDA: dequantize_turbo{2,3,4}_0 in dequantize.cuh; quantize in turbo-quant.cuh
- Wired: ggml-cuda.cu (MUL_MAT, GET_ROWS, SET_ROWS), set-rows.cu, arg.cpp, llama-bench.cpp
- CLI: `--cache-type-k turbo4_0 --cache-type-v turbo4_0` works
- sm_37: All kernels FP32, no tensor cores; fully K80-compatible
- Symbols: Exported from libggml.so and libggml-cuda.so

### Phase 4: TriAttention GPU Kernels ✓ (Task 3 partial)
- GPU scoring kernel: triattention-score.cu (542 lines)
- GPU API: triattention_gpu_init/score_head/free/etc. in ggml-cuda.h
- Supports TurboQuant types (turbo2/3/4 dequant with WHT handling)
- Symbols exported from libggml-cuda.so
- sm_37 compatible (FP32 math only)

---

## Remaining

### Async Pipeline: NCCL Staggered Allreduce (Patch 4)
- Wire ggml_cuda_nccl_staggered_allreduce() into ggml_backend_cuda_comm_allreduce_nccl()
- Stagger NUMA0 (GPU0-3) and NUMA1 (GPU4-7) allreduce to avoid QPI contention
- nccl-stagger.cuh exists; needs wiring into comm path

### Async Pipeline: NUMA Thread Pinning for NCCL (Patch 6)
- Pin NCCL worker threads to NUMA-local cores in ggml_backend_cuda_comm_context_init()
- numa-gpu-bind.cuh exists; needs integration

### TriAttention: CPU/Server Integration (Task 3 remaining)
- Create src/llama-triattention.h/cpp (loader, RoPE inversion, pruning pipeline)
- Wire pruning hook into llama-context.cpp decode loop
- Add CLI flags (--triattention-stats, --triattention-budget, etc.)
- Calibration tool (--triattention-calibrate)
- Multi-GPU tensor-split coordination for eviction decisions

### FlashAttention / Sliding Window on sm_37 (Task 4)
- Test existing FA implementation on K80 (sm_37, 8x K80)
- Identify blockers: sm_80+ requirements, warp primitives, shared mem limits
- Implement SWA as fallback if FA incompatible
- Document in RESEARCH/FLASHATTENTION_NOUGHT.md

### cuBLAS vs MMQ Benchmarking (Task 9)
- Run llama-bench with current config (MMQ path)
- Run with -DGGML_CUDA_FORCE_CUBLAS=ON
- Determine winner for Kepler sm_37
- Document in RESEARCH/CUBLAS_VS_MMQ.md

### TurboQuant Verification (blocked)
- Convert test models to TurboQuant, run llama-bench
- Compare accuracy vs baseline quantization
- Compare inference speed and memory usage
- Blocked: all GPUs in production use

---

## Build Config (NOUGHT — Verified)

```bash
cd /home/whistler/llama_wukong && rm -rf build && mkdir build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_F16=ON \
  -DCMAKE_CUDA_HOST_COMPILER=g++-11 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.8/bin/nvcc \
  -DGGML_CUDA_NCCL=ON \
  -DCMAKE_CUDA_ARCHITECTURES="37" \
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

## Test Command (Working — 8x K80 tensor-split)

```bash
sudo GGML_CUDA_P2P=1 -E nice -n -20 numactl --interleave=all \
  ./build/bin/llama-server \
  -m /path/to/models/Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-MTP-Q4_K_M.gguf \
  -t 36 -c 262144 -ngl 99 \
  --port 4269 --host 0.0.0.0 --api-key YOUR_API_KEY_HERE \
  --jinja --chat-template-file ./models/tuvak.jinja \
  --load-mode none -np 3 \
  --ctx-checkpoints 64 --checkpoint-min-step 4096 --cache-ram 65536 \
  --mmproj /path/to/models/Qwen3.6-27B-mmproj-F16.gguf --no-mmproj-offload \
  --image-min-tokens 1024 \
  --batch-size 2048 --ubatch-size 512 \
  --cache-type-k turbo4_0 --cache-type-v turbo4_0 \
  --tensor-split 1,1,1,1,1,1,1,1 \
  --kv-unified --slot-save-path /path/to/models/kv_cache \
  --seed 1016 \
  --spec-type draft-mtp --spec-draft-p-min 0.75 --spec-draft-n-max 3 \
  --split-mode tensor
```

## Key Files

| File | Purpose |
|------|---------|
| ggml-cuda/async-pipeline.cuh | Per-GPU async context, prefetch streams |
| ggml-cuda/nccl-stagger.cuh | NUMA-aware NCCL staggered allreduce |
| ggml-cuda/numa-gpu-bind.cuh | GPU-to-NUMA topology utilities |
| ggml-cuda/triattention-score.cu | TriAttention GPU scoring kernel |
| ggml-cuda/turbo-quant.cuh | TurboQuant CUDA kernels |
| ggml-cuda/dequantize.cuh | TurboQuant dequantization |
| ggml/src/ggml-turbo-quant.c | TurboQuant CPU kernels |
| common/fit.cpp | Tensor-split-aware memory fitting |

## Hardware

- **Server**: NOUGHT (Debian/Q4OS)
- **CPU**: Dual Xeon E5-2697 v4 (36 cores each, 72 total)
- **RAM**: 128GB DDR4 ECC
- **GPU**: 8x Tesla K80 GK210 sm_37 (11GB each)
  - NUMA0: GPU0-3 (PIX-linked pairs: 0-1, 2-3)
  - NUMA1: GPU4-7 (PIX-linked pairs: 4-5, 6-7)
- **Driver**: 470.256.02, CUDA 11.8

---
*UnobligatedRascal — Making old hardware sing.*
