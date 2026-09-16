# llama_wukong — Project Reference

> llama.cpp fork optimized for NVIDIA Kepler sm_37 (8x Tesla K80). Base: llama_lazarus → ggml-org/llama.cpp.
> UnobligatedRascal — Making old hardware sing.

**Last updated:** 2026-09-16

**OBJECTIVE: Maximum single-user inference speed on 8x Tesla K80.**

---

## Hardware Target (NOUGHT)

- Dual Xeon E5-2697 v4 (36 cores each, 72 total), 128GB DDR4 ECC
- 8x Tesla K80 GK210 sm_37 (11GB each): NUMA0=GPU0-3, NUMA1=GPU4-7 (PIX-linked pairs)
- Driver 470.256.02, CUDA 11.8, PCIe Gen3 x16
- OS: Debian Bookworm / Q4OS

**Constraint:** Memory-bound bottleneck (~13% of theoretical max). Arithmetic optimization is waste; focus on bandwidth reduction.

---

## Current Status

**Verified working:** Tensor-split across all 8 GK210 GPUs, Qwen3.6-27B Q4_K_M at 256K context.

### Completed

| Feature | Status |
|---------|--------|
| NUMA replication | ✓ Complete |
| TurboQuant KV cache | ✓ Integrated |
| TriAttention GPU kernels | ✓ GPU scoring complete |
| RoPE LUT | ✓ Per-GPU sin/cos lookup |
| KV cache + tensor-split fix | ✓ Applied |
| FA + turbo3_0 V cache + tensor-split | ✓ Contiguity fix |

---

## Priority Order — Maximum Inference Speed

### P0 — BLOCKING
1. **TurboQuant/meta backend crash** — turbo2_0/turbo3_0/turbo4_0 crashes with tensor-split. Blocks all turbo KV cache usage. **Must fix: turbo = the bandwidth reduction we need.**

### P1 — HIGH
2. **cuBLAS vs MMQ benchmark** — cuBLAS F32 may beat integer MMQ on Kepler. Measure: `-DGGML_CUDA_FORCE_CUBLAS=ON` vs `-DGGML_CUDA_FORCE_MMQ=ON`.
3. **TurboQuant with tensor-split** — After crash fix, use turbo4_0 KV cache. Less data off chip = more tokens/sec.

### P2 — MEDIUM
4. **NUMA replication validation** — Was 2.3x slower when broken; verify current state actually helps.
5. **KV cache persistence to disk** — Skip for single-user unless reloading same context repeatedly.

### P3 — LOW / DEFERRED
6. IMAD micro-optimization — Research says <2% gain on memory-bound hardware. Skip.
7. LUT-GEMM — Wrong tool for Q4_K_M uniform quant. Skip.
8. Speculative decoding — Tensor-split incompatibility. Abandoned.

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
  --seed 1016
  --split-mode tensor
```

---

## TurboQuant Reference

| Type | Bits/value | Compression vs FP16 | Notes |
|------|-----------|---------------------|-------|
| turbo4_0 | 4.25 bpw | 3.8× | Recommended |
| turbo3_0 | 3.5 bpw | 4.6× | FA fallback |
| turbo2_0 | 2.5 bpw | 6.4× | FA fallback; breaks meta backend with tensor-split |

---

*UnobligatedRascal — Making old hardware sing.*
