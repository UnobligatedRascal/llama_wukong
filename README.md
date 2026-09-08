# llama_wukong

> llama.cpp fork optimized for NVIDIA Kepler sm_37 hardware (8x Tesla K80).

## What This Is

llama_wukong extends llama.cpp with NOUGHT-specific optimizations for running large models efficiently on aging Kepler hardware. Focuses on existing model architectures (Qwen3.6/3.8, L3.2-MoE, etc.). Qwen4-exp architecture work deferred to a separate project.

**Core optimizations:**
- NUMA-aware weight replication across dual Xeon sockets
- TurboQuant KV cache compression (3.8×–6.4× VRAM savings) — **READY**
- TriAttention GPU scoring kernels for efficient context pruning — **GPU DONE, server integration pending**
- sm_37-compatible CUDA kernels (FP32, no tensor cores)
- 8-GPU tensor-split support for models up to 40B+ parameters

## Hardware Target

- Server: NOUGHT (dual Xeon E5-2697 v4, 128GB DDR4 ECC, 8x Tesla K80 GK210 sm_37)
- Use case: Running 27B–40B parameter models at 256K+ context with speculative decoding

## Quick Reference

| File | Purpose |
|------|---------|
| TODO.md | Current task tracking (concise) |
| PHASE1_TODO.md | Detailed implementation tasks with checkboxes |
| TODO_PHASE4.md | TriAttention + TurboQuant deep dive |
| llama_wukong.md | Full project history and scope |
| ARCHITECTURE_REFERENCE.md | Original pitch: memory topology, execution pipeline, Kepler exploitation |
| VERIFIED_CONFIG.md | Working launch command and hardware profile |
| RESEARCH/ | Technical research notes per optimization area |

## Build (NOUGHT)

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

## TurboQuant KV Cache — Ready to Use

KV cache compression via PolarQuant + WHT rotation. Use with `--cache-type-k`/`--cache-type-v`:

| Type | Bits/value | Compression vs FP16 | Use case |
|------|-----------|---------------------|----------|
| turbo4_0 | 4.25 bpw | 3.8× | Recommended starting point |
| turbo3_0 | 3.5 bpw | 4.6× | Good quality/compression tradeoff |
| turbo2_0 | 2.5 bpw | 6.4× | Maximum compression, quality loss |

Example:
```bash
./build/bin/llama-server -m model.gguf --cache-type-k turbo4_0 --cache-type-v turbo4_0
```

## Current Status

- **NUMA replication:** Complete (2.5× speedup on dual-socket)
- **TurboQuant backend:** Fully integrated; CLI flags working; symbols exported
- **TriAttention:** GPU scoring kernels complete; CPU/server integration pending
- **FlashAttention/SWA on sm_37:** Pending feasibility testing
- **Qwen4-exp architecture:** Deferred to separate project

Verified working: tensor-split across all 8 GK210 GPUs, -np 2–6, Qwen3.6-27B Q4_K_M at 256K context with speculative decoding.

## License

MIT License (inherited from upstream llama.cpp)

This is a fork of [UnobligatedRascal/llama_lazarus](https://github.com/UnobligatedRascal/llama_lazarus), which is itself a fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp). Upstream authors retain all rights to their original code.

---
*UnobligatedRascal — Making old hardware sing.*
