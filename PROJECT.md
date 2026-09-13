# llama_wukong — Project Reference

> llama.cpp fork optimized for NVIDIA Kepler sm_37 (8x Tesla K80). Base: llama_lazarus → ggml-org/llama.cpp.
> UnobligatedRascal — Making old hardware sing.

**Last updated:** 2026-09-12 (turbo2_0/turbo3_0 meta backend fix applied)

---

## Hardware Target (NOUGHT)

- Dual Xeon E5-2697 v4 (36 cores each, 72 total), 128GB DDR4 ECC
- 8x Tesla K80 GK210 sm_37 (11GB each): NUMA0=GPU0-3, NUMA1=GPU4-7 (PIX-linked pairs)
- Driver 470.256.02, CUDA 11.8, PCIe Gen3 x16
- OS: Debian Bookworm / Q4OS

---

## Current Status

**Verified working:** Tensor-split across all 8 GK210 GPUs, -np 2–6, Qwen3.6-27B Q4_K_M at 256K context with speculative decoding.

**Build:** Clean, branch main, working tree clean.

### Completed Features

| Feature | Status | Key Files |
|---------|--------|-----------|
| NUMA replication | ✓ Complete (2.5× dual-socket speedup) | ggml-cpu-numa-replicate.c |
| TurboQuant KV cache | ✓ Integrated; CLI flags work; symbols exported | turbo-quant.cuh, dequantize.cuh, ggml-turbo-quant.c |
| TriAttention GPU kernels | ✓ GPU scoring complete (542 lines) | triattention-score.cu, ggml-cuda.h |
| RoPE LUT | ✓ Per-GPU sin/cos lookup (32KB const mem) | rope-lut.cuh |
| KV cache + tensor-split fix (Option A) | ✓ Applied (commit 7d7944e3b) | llama-model.cpp |
| FA + turbo3_0 V cache + tensor-split | ✓ Contiguity fix (commit 5970fe28a) | llama-graph.cpp |

### Active Issues

#### P0: Meta backend assertion with turbo2_0/turbo3_0 V cache

```
ggml/src/ggml-backend-meta.cpp:1645: GGML_ASSERT(size % chunk_size_full == 0) failed
```

- **Trigger:** `--cache-type-v turbo2_0` or `turbo3_0` with `--split-mode tensor`
- **Root cause:** Meta backend's linear stride scaling (`nb[i] = full_nb * split_ne/full_ne`) produces incorrect row strides for quantized types split along axis 0, because it doesn't respect quantization block alignment. Turbo types (blck_size=128) with uneven splits get misaligned strides.
- **Fix applied (commit 70481fa68):** For quantized types (blck_size > 1) split along axis 0, use `ggml_row_size(type, ne[0])` which correctly computes block-aligned row strides.
- **Verified:** llama-server starts and runs with turbo2_0/turbo3_0/turbo4_0 KV cache + tensor-split without assertion failures.

#### ✓ P0: KV cache quantization verification (COMPLETE)

- Fixed and verified.

#### P1: Speculative decoding backend offload failure

```
W set_sampler: backend sampling not supported with SPLIT_MODE_TENSOR; using CPU
W spec common_specu: backend offload failed for seq_id=0; using CPU sampler
```

- **Root cause 1:** Overly conservative blanket block in `llama_context::set_sampler()` blocking backend sampling for SPLIT_MODE_TENSOR.
- **Fix 1 (commit 50fd7cfc3):** Removed blanket block — output layer is on single GPU, samplers can offload there.
- **Root cause 2:** MTP draft context inherited TENSOR split mode, causing meta backend crash in `handle_per_row` (RMS_NORM on axis-0-split tensors).
- **Fix 2 (commit 11c63fee5):** Force `LLAMA_SPLIT_MODE_NONE` for MTP draft context — MTP head runs on single GPU with full activations.
- **Root cause 3 (DISCOVERED):** Fix 2 only applied when loading separate draft model (`has_draft` branch). MTP-with-same-model (`else if (spec_mtp)`) reused `model_tgt` directly, inheriting TENSOR split mode.
- **Fix 3 (IN PROGRESS):** For MTP-with-same-model, load model a second time with NONE split mode. MTP head needs full weights on single GPU.
- **Status:** Fix 3 applied in working tree. Requires rebuild + testing.
- **Test:** `--spec-type draft-mtp --split-mode tensor --tensor-split 1,1,1,1,1,1,1,1` — expect no crashes, GPU sampling active.

#### ✓ P0: Git fork hygiene (COMPLETE)

- **Phase 1:** Synced llama_lazarus with ggml-org via PR #2 (merged)
- **Phase 2:** Created clean llama_wukong fork — 1 commit on top of synced lazarus
- origin/main now has clean ancestry: ggml-org/master → lazarus fixes → wukong features
- Old messy history preserved as main-backup branch

**Phase 2 (PENDING):** Create clean llama_wukong fork
- After PR #1 merges, fetch updated upstream (llama_lazarus)
- Create wukong from synced lazarus + wukong-specific commits (~25)
- Script: scripts/create_clean_fork.sh (will need update after PR #1 merges)
- Conflicts expected in: ggml-cuda.cu, ggml.h, llama-model.cpp, llama-graph.cpp, llama-kv-cache.cpp, llama-context.cpp, arg.cpp

**Current backup:** main-backup branch preserves all current work.

### Deferred / Pending

- **P1:** Async pipeline re-implementation (rolled back 2026-09-08, malloc corruption). See ASYNC_PIPELINE_ATTEMPTED_CHANGES.md for root cause analysis. Re-impl notes: no lazy init during load, fix unreachable enable(), test NUMA first, no cudaSetDevice in constructors.
- **P2:** FlashAttention/SWA feasibility on sm_37
- **P2:** cuBLAS vs MMQ benchmarking
- **P2:** TurboQuant runtime verification (blocked: GPUs in production)

---

## Build

```bash
cd /home/whistler/llama_wukong && rm -rf build && mkdir build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON -DGGML_CUDA_F16=ON \
  -DCMAKE_CUDA_HOST_COMPILER=g++-11 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.8/bin/nvcc \
  -DGGML_CUDA_NCCL=ON -DCMAKE_CUDA_ARCHITECTURES="37" \
  -DGGML_CUDA_FA_ALL_QUANTS=ON -DGGML_CUDA_FORCE_MMQ=ON -DGGML_CUDA_GRAPHS=OFF \
  -DCMAKE_C_COMPILER=gcc-11 -DCMAKE_CXX_COMPILER=g++-11 \
  -DGGML_CUDA_CUBLAS=ON \
  -DCMAKE_C_FLAGS="-DGGML_NUMA_REPLICATE" \
  -DCMAKE_CXX_FLAGS="-DGGML_NUMA_REPLICATE" \
  -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-rpath,/usr/local/cuda-11.8/targets/x86_64-linux/lib"
make -j36 llama-server llama-bench
```

Disable NUMA at runtime: `GGML_NUMA_REPLICATE=0 ./build/bin/llama-server ...`

---

## Run (Verified)

```bash
sudo GGML_CUDA_P2P=1 -E nice -n -20 numactl --interleave=all \
  ./build/bin/llama-server \
  -m /path/to/model.gguf -t 36 -c 262144 -ngl 99 \
  --port 4269 --host 0.0.0.0 --api-key KEY \
  --jinja --chat-template-file ./models/tuvak.jinja \
  --load-mode none -np 3 \
  --ctx-checkpoints 64 --checkpoint-min-step 4096 --cache-ram 65536 \
  --mmproj /path/to/mmproj.gguf --no-mmproj-offload --image-min-tokens 1024 \
  --batch-size 2048 --ubatch-size 512 \
  --cache-type-k turbo4_0 --cache-type-v turbo4_0 \
  --tensor-split 1,1,1,1,1,1,1,1 --kv-unified --slot-save-path /path/to/kv_cache \
  --seed 1016 --spec-type draft-mtp --spec-draft-p-min 0.75 --spec-draft-n-max 3 \
  --split-mode tensor
```

---

## TurboQuant Reference

| Type | Bits/value | Compression vs FP16 | Notes |
|------|-----------|---------------------|-------|
| turbo4_0 | 4.25 bpw | 3.8× | Recommended |
| turbo3_0 | 3.5 bpw | 4.6× | FA fallback (not supported in fattn.cu) |
| turbo2_0 | 2.5 bpw | 6.4× | FA fallback; breaks meta backend with tensor-split |

Use: `--cache-type-k turbo4_0 --cache-type-v turbo4_0`

---

## Key Custom Files

| File | Purpose |
|------|---------|
| ggml-cuda/async-pipeline.cuh | Per-GPU async context, prefetch streams (rolled back) |
| ggml-cuda/nccl-stagger.cuh | NUMA-aware NCCL staggered allreduce |
| ggml-cuda/numa-gpu-bind.cuh | GPU-to-NUMA topology utilities |
| ggml-cuda/triattention-score.cu | TriAttention GPU scoring kernel |
| ggml-cuda/turbo-quant.cuh | TurboQuant CUDA kernels |
| ggml-cuda/dequantize.cuh | TurboQuant dequantization |
| ggml-cuda/rope-lut.cuh | RoPE sin/cos lookup table |
| ggml-turbo-quant.c | TurboQuant CPU kernels |
| ggml-cpu/ggml-cpu-numa-replicate.c | NUMA weight replication |
| common/fit.cpp | Tensor-split-aware memory fitting |

---

## Troubleshooting Keywords

| Issue | Search Terms |
|-------|-------------|
| KV cache quant not working | `type_k`, `type_v`, `ggml_new_tensor_3d`, `kv_cache_types`, `blck_size`, `split_state`, `pattern_kv_cache` |
| Tensor-split issues | `meta backend`, `split_state`, `GGML_BACKEND_SPLIT_AXIS`, `get_split_state`, `non-contiguous split tensor` |
| TurboQuant meta crash | `chunk_size_full`, `GGML_ASSERT(size % chunk_size_full`, `ggml-backend-meta.cpp:1645` |
| FA fallback | `ggml_cuda_fattn_kv_type_supported`, `use_flash_attn`, `BEST_FATTN_KERNEL_NONE` |
| Spec decoding CPU fallback | `set_sampler`, `backend sampling not supported`, `SPLIT_MODE_TENSOR` |
| TriAttention | `triattention-score.cu`, `triattention_gpu_init`, `RoPE inversion` |
| NUMA | `ggml_numa_replicate`, `GGML_NUMA_REPLICATE`, `numa_replicate_get_local_ptr` |
| sm_37 specific | `CUBLAS_GEMM_DEFAULT_TENSOR_OP`, `__constant__`, `rope_lut` |

---

## Git History (Last 10)

```
70481fa68 fix(meta): correct stride scaling for quantized split tensors
1219e9e4f docs: consolidate all documentation into PROJECT.md; remove redundant files
220da23a7 docs: consolidate TODO, add KV cache audit methodology, clean README
7046c46a1 docs: mark Option A fix as applied in ROTA
7d7944e3b fix: use KV cache tensor's own block size for split granularity
5177c1716 docs: add ROTA for quantized KV cache + tensor-split fix
1103f55de revert: remove broken MIRRORED shortcut in meta backend get_tensor
3709ed8ba Fix turbo3_0/turbo4_0 KV cache with tensor-split: meta backend split state bugs
5970fe28a fix: make FA output contiguous before reshape for tensor-split + turbo3_0 V cache
844a11324 fix: handle_bin_bcast for MIRRORED+split sources
```

---

*UnobligatedRascal — Making old hardware sing.*
