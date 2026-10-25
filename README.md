# llama_wukong

[![Fork of ggml-org/llama.cpp](https://img.shields.io/badge/fork-llama.cpp-blue)](https://github.com/ggml-org/llama.cpp)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**High-performance LLM inference on legacy NVIDIA hardware — making old GPUs sing.**

`llama_wukong` is a specialized fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) optimized for NVIDIA Kepler architecture (sm_37) and multi-GPU tensor-split deployments. Targets production workloads on Tesla K80 and similar legacy hardware with modern quantization, speculative decoding, and NUMA-aware optimizations.

---

## Key Features

- **Kepler sm_37 support:** Optimized CUDA kernels and cuBLAS fallbacks for Tesla K80/K40
- **TurboQuant KV cache:** Proprietary KV cache quantization (turbo2_0/turbo3_0/turbo4_0) — up to 6.4× compression vs FP16
- **TriAttention:** GPU-accelerated attention scoring for MoE models
- **NUMA replication:** Dual-socket aware weight replication (~2.5× speedup on multi-socket systems)
- **RoPE LUT:** Precomputed sin/cos lookup tables reducing transcendental overhead
- **Tensor-split across 8+ GPUs:** Verified with Qwen models at 256K context on 8x K80

---

## Hardware Target

Optimized for NVIDIA Kepler sm_37 with dual-socket Intel Xeon:

| Component | Spec |
|-----------|------|
| CPU | Dual Intel Xeon E5-2697 v4 (72 cores total) |
| GPU | 8× Tesla K80 GK210 (sm_37, 11GB each) |
| RAM | 128GB DDR4 ECC |
| CUDA | 11.8 + NCCL |

Works on any sm_37+ NVIDIA hardware with appropriate adjustments.

---

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON -DGGML_CUDA_FA_QUANTS=all -DGGML_CUDA_FORCE_MMQ=ON \
  -DGGML_CUDA_FA=ON -DGGML_CUDA_NCCL=ON -DGGML_CUDA_CUBLAS=ON \
  -DCMAKE_CUDA_ARCHITECTURES="37" \
  -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_OPENMP=ON \
  -DLLAMA_CURL=OFF && cmake --build build -j$(nproc)
```

See [docs/build.md](docs/build.md) for full options.

---

## Run

```bash
./build/bin/llama-server \
  -m model.gguf -t 36 -c 262144 -ngl 99 \
  --tensor-split 1,1,1,1,1,1,1,1 --kv-unified \
  --cache-type-k turbo4_0 --cache-type-v turbo4_0 \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  --jinja --port 8080
```

For production deployment on multi-socket/NUMA systems:

```bash
numactl --interleave=all ./build/bin/llama-server [options]
```

---

## TurboQuant KV Cache

| Type | Bits/value | Compression vs FP16 | Notes |
|------|-----------|---------------------|-------|
| turbo4_0 | 4.25 bpw | 3.8× | Recommended default |
| turbo3_0 | 3.5 bpw | 4.6× | FlashAttention fallback |
| turbo2_0 | 2.5 bpw | 6.4× | FlashAttention fallback |

Use: `--cache-type-k turbo4_0 --cache-type-v turbo4_0`

---

## Architecture

Built on top of [llama.cpp](https://github.com/ggml-org/llama.cpp) and [ggml](https://github.com/ggml-org/ggml).

Key customizations:

- `ggml-cuda/triattention-score.cu` — TriAttention GPU scoring
- `ggml-cuda/turbo-quant.cuh` — TurboQuant CUDA kernels
- `ggml-cpu/ggml-cpu-numa-replicate.c` — NUMA-aware weight replication
- `common/fit.cpp` — Tensor-split memory fitting

---

## Contributing

Contributions welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

For upstream issues or general llama.cpp questions, please use the [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) repository.

---

## License

MIT License — see [LICENSE](LICENSE).

---

*llama_wukong — UnobligatedRascal*
