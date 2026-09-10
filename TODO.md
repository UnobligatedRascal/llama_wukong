# llama_wukong - Master Task Tracker

> Optimized llama.cpp fork for NVIDIA Kepler sm_37 hardware (8x Tesla K80).
> Base: llama_lazarus (UnobligatedRascal fork), ggml-org/llama.cpp upstream.

**Last updated:** 2026-09-10

---

## Current Status: BUILD COMPLETE — TurboQuant READY, TriAttention GPU DONE

**Verified working:** Tensor-split across all 8 GK210 GPUs, -np 2–6, Qwen3.6-27B Q4_K_M at 256K context with speculative decoding.

---

## Completed

### NUMA Replication ✓
- NUMA-aware weight replication for ggml-cpu
- Per-node weight pointers in kernels
- Benchmark: 2.5× speedup on dual-socket (vs cross-NUMA baseline)
- Files: ggml/src/ggml-cpu/ggml-cpu-numa-replicate.c

### Async GPU Pipeline (Partial) ✓
- async-pipeline.cuh: Per-GPU lazy-init context, prefetch streams, double-buffered events
- nccl-stagger.cuh: NUMA-aware NCCL env init for staggered allreduce
- numa-gpu-bind.cuh: GPU-to-NUMA binding utilities
- Integration in ggml-cuda.cu: Global context, enable() in backend init, mark_layer_complete() on MUL_MAT ops only
- Defensive null checks prevent crashes during lazy-init races
- Restricted to heavy ops (MUL_MAT) to avoid ~35% regression on lightweight ops
- **Note:** Full async pipeline rolled back (malloc corruption); core files preserved for re-implementation

### fit.cpp Tensor-Split Support ✓
- Tensor-split mode works with --fit path
- Memory targets validated per-device after context reduction
- Clear error if tensor-split model can't fit (no layer redistribution possible)

### TurboQuant KV Cache ✓
- Types: GGML_TYPE_TURBO3_0/4_0/2_0 registered in ggml.h
- CPU: quantize/dequantize_row_turbo{2,3,4}_0 in ggml-turbo-quant.c
- CUDA: dequantize_turbo{2,3,4}_0 in dequantize.cuh; quantize in turbo-quant.cuh
- Wired: ggml-cuda.cu (MUL_MAT, GET_ROWS, SET_ROWS), set-rows.cu, arg.cpp, llama-bench.cpp
- CLI: `--cache-type-k turbo4_0 --cache-type-v turbo4_0` works
- sm_37: All kernels FP32, no tensor cores; fully K80-compatible
- Symbols: Exported from libggml.so and libggml-cuda.so

### TriAttention GPU Kernels ✓
- GPU scoring kernel: triattention-score.cu (542 lines)
- GPU API: triattention_gpu_init/score_head/free/etc. in ggml-cuda.h
- Supports TurboQuant types (turbo2/3/4 dequant with WHT handling)
- Symbols exported from libggml-cuda.so
- sm_37 compatible (FP32 math only)

### RoPE Lookup Table ✓
- Precomputed sin/cos LUT for sm_37 transcendental acceleration
- 4096 entries × 2 tables × 4 bytes = 32KB constant memory
- ~5 cycles per sin/cos via LUT+interp vs 20-30 native
- Files: ggml/src/ggml-cuda/rope-lut.cuh

### KV Cache Quantization + Tensor-Split Fix ✓
- Option A fix: Use KV cache tensor's own block size for split granularity
- Committed: 7d7944e3b (llama_wukong), 77657e77f (llama_lazarus)
- See: QUANTIZED_KV_TENSOR_SPLIT_ROTA.md

---

## Active Work

### P0: Verify KV Cache Quantization Actually Works
**Problem:** User reports f16=q8_0=q4_0 speed — no measurable improvement from quantization.

**Hypotheses:**
1. Quantization not being applied (silent fallback to f16)
2. Dequantization overhead masks bandwidth savings
3. Tensor-split meta backend overhead negates quantization benefits
4. Pipeline bottleneck elsewhere (PCIe, NCCL comms, compute-bound)

**Action items:**
- [ ] Add instrumentation logging at KV cache allocation (llama-kv-cache.cpp)
- [ ] Add logging at split state computation (llama-model.cpp)
- [ ] Add logging at attention path selection (llama-graph.cpp)
- [ ] Run nvprof to measure actual memory bandwidth vs KV cache type
- [ ] Profile CUDA kernel times: dequantize vs compute
- [ ] Compare: single-GPU vs tensor-split with quantized KV cache
- [ ] Document findings in RESEARCH/KV_CACHE_AUDIT.md

**See:** RESEARCH/KV_CACHE_AUDIT.md for full methodology

### P0: Properly Fork llama_wukong from llama_lazarus
**Problem:** llama_wukong was forked improperly — no ancestry relationship with llama_lazarus.

**Current state:**
- llama_lazarus: 159 commits behind ggml-org/llama.cpp master
- llama_wukong: Full ggml-org history + ~22 custom commits, NOT forked from lazarus

**Plan:**
- [ ] Update llama_lazarus: rebase onto ggml-org/llama.cpp master
- [ ] Resolve conflicts in 4 lazarus-specific commits (Kepler sm_37 fixes)
- [ ] Force-push updated llama_lazarus
- [ ] Rebase llama_wukong custom commits onto updated llama_lazarus
- [ ] Cherry-pick ~22 wukong-specific commits, resolve conflicts
- [ ] Force-push llama_wukong main to new clean history
- [ ] Verify build and all features work post-rebase

**Risk:** Conflicts expected in ggml-cuda.cu, ggml.h, llama-model.cpp, llama-graph.cpp, llama-kv-cache.cpp, llama-context.cpp, arg.cpp (all modified upstream in 159 commits).

### P1: TriAttention CPU/Server Integration
- [ ] Create src/llama-triattention.h/cpp (loader, RoPE inversion, pruning pipeline)
- [ ] Wire pruning hook into llama-context.cpp decode loop
- [ ] Add CLI flags (--triattention-stats, --triattention-budget, etc.)
- [ ] Calibration tool (--triattention-calibrate)
- [ ] Multi-GPU tensor-split coordination for eviction decisions

### P1: Async Pipeline Re-Implementation
- [ ] Fix lazy init race condition (no cudaSetDevice in constructors)
- [ ] Move async pipeline enable() BEFORE return in ggml_cuda_init()
- [ ] Test NUMA replication FIRST, then add async pipeline incrementally
- [ ] Wire nccl-stagger.cuh staggered allreduce into comm path
- [ ] Pin NCCL worker threads to NUMA-local cores

---

## Remaining

### P2: FlashAttention / Sliding Window on sm_37
- [ ] Test existing FA implementation on K80 (sm_37, 8x K80)
- [ ] Identify blockers: sm_80+ requirements, warp primitives, shared mem limits
- [ ] Implement SWA as fallback if FA incompatible
- [ ] Document in RESEARCH/FLASHATTENTION_NOUGHT.md

### P2: cuBLAS vs MMQ Benchmarking
- [ ] Run llama-bench with current config (MMQ path)
- [ ] Run with -DGGML_CUDA_FORCE_CUBLAS=ON
- [ ] Determine winner for Kepler sm_37
- [ ] Document in RESEARCH/CUBLAS_VS_MMQ.md

### P2: TurboQuant Verification
- [ ] Convert test models to TurboQuant, run llama-bench
- [ ] Compare accuracy vs baseline quantization
- [ ] Compare inference speed and memory usage
- [ ] Blocked: all GPUs in production use

### P2: TriAttention + TurboQuant Stacked Testing
- [ ] Verify TriAttention uses turbo2/turbo3 dequant functions correctly
- [ ] Run stacked config: `--cache-type-k turbo3 --cache-type-v turbo3 --triattention-stats model.triattention --triattention-budget 2048`
- [ ] Expected: ~40× effective KV compression vs FP16 baseline

---

## Build Config (Verified)

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

---

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

---

## Hardware

- **Server:** Dual Xeon E5-2697 v4 (36 cores each, 72 total), 128GB DDR4 ECC
- **GPU:** 8x Tesla K80 GK210 sm_37 (11GB each)
  - NUMA0: GPU0-3 (PIX-linked pairs: 0-1, 2-3)
  - NUMA1: GPU4-7 (PIX-linked pairs: 4-5, 6-7)
- **Driver:** 470.256.02, CUDA 11.8
- **OS:** Debian Bookworm / Q4OS

---

## Key Files

| File | Purpose |
|------|---------|
| ggml-cuda/async-pipeline.cuh | Per-GPU async context, prefetch streams |
| ggml-cuda/nccl-stagger.cuh | NUMA-aware NCCL staggered allreduce |
| ggml-cuda/numa-gpu-bind.cuh | GPU-to-NUMA topology utilities |
| ggml-cuda/triattention-score.cu | TriAttention GPU scoring kernel |
| ggml-cuda/turbo-quant.cuh | TurboQuant CUDA kernels |
| ggml-cuda/dequantize.cuh | TurboQuant dequantization |
| ggml-cuda/rope-lut.cuh | RoPE sin/cos lookup table |
| ggml/src/ggml-turbo-quant.c | TurboQuant CPU kernels |
| ggml/src/ggml-cpu/ggml-cpu-numa-replicate.c | NUMA weight replication |
| common/fit.cpp | Tensor-split-aware memory fitting |
| src/llama-triattention.cpp/h | TriAttention CPU integration |

---

## Troubleshooting Keywords

**For future debugging sessions, search these terms:**

- **KV cache quantization not working:** `type_k`, `type_v`, `ggml_new_tensor_3d`, `kv_cache_types`, `blck_size`, `split_state`, `pattern_kv_cache`
- **Tensor-split issues:** `meta backend`, `split_state`, `GGML_BACKEND_SPLIT_AXIS`, `get_split_state`, `non-contiguous split tensor`
- **Flash attention fallback:** `ggml_cuda_fattn_kv_type_supported`, `use_flash_attn`, `BEST_FATTN_KERNEL_NONE`, `GGML_CUDA_FA_ALL_QUANTS`
- **TurboQuant:** `GGML_TYPE_TURBO`, `turbo-quant.cuh`, `dequantize_turbo`, `WHT`, `PolarQuant`
- **TriAttention:** `triattention-score.cu`, `triattention_gpu_init`, `triattention_prune`, `RoPE inversion`
- **NUMA replication:** `ggml_numa_replicate`, `GGML_NUMA_REPLICATE`, `numa_replicate_get_local_ptr`
- **Kepler sm_37 specific:** `CUBLAS_GEMM_DEFAULT_TENSOR_OP`, `sm_37`, `__constant__`, `__device__`, `rope_lut`
- **Meta backend quantized splits:** `ggml_backend_meta_get_split_state`, `nr[]`, `segments`, `granularity`
- **Dequantization overhead:** `dequantize_row`, `mul_mat`, `quantize`, `to_float`, `from_float_ref`
- **NCCL/P2P:** `GGML_CUDA_P2P`, `ncclCommInit`, `ncclAllReduce`, `GGML_CUDA_PEER_MAX_BATCH_SIZE`

---

## Documentation

- **QUANTIZED_KV_TENSOR_SPLIT_ROTA.md** — KV cache + tensor-split fix documentation
- **RESEARCH/KV_CACHE_AUDIT.md** — KV cache visualization & audit methodology
- **RESEARCH/TURBOQUANT_SM37_AUDIT.md** — TurboQuant sm_37 compatibility audit
- **RESEARCH/TURBOQUANT_TRIATTENTION_RESEARCH.md** — Combined research notes
- **RESEARCH/TRIATTENTION_REVIEW.md** — TriAttention implementation review
- **ASYNC_PIPELINE_ATTEMPTED_CHANGES.md** — Async pipeline rollback record
- **ARCHITECTURE_REFERENCE.md** — Original architectural pitch (historical)
- **VERIFIED_CONFIG.md** — Working launch command and hardware profile

---

*UnobligatedRascal — Making old hardware sing.*
