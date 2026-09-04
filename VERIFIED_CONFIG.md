# llama_wukong - Verified Working Configuration

Source: llama_lazarus, commit 93c888df1 (Kepler sm_37 cuBLAS fixes)

## NOUGHT Hardware Profile

- GPUs: 8 x Tesla K80 GK210 (sm_37, 12GB VRAM each, 96GB aggregate)
- CPUs: 2 x Intel Xeon E5-2697 v4 (18c/36t each, 72t total, 2 NUMA nodes)
- RAM: 128GB DDR4 ECC (64GB per NUMA node)
- Storage: 256GB NVMe + 512GB NVMe (models on 512GB SSD)
- CUDA: Toolkit 11.8, Driver 470.256.02, Runtime 11.4
- OS: Debian Bookworm / Q4OS

## Verified Launch Command

```bash
sudo GGML_CUDA_P2P=1 -E nice -n -20 numactl --interleave=all \
  /home/whistler/muthafukka/llama_lazarus/build/bin/llama-server \
  -m /mnt/512gb_ssd/models/Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-MTP-Q4_K_M.gguf \
  -t 28 \
  -c 262144 \
  -ngl 99 \
  --port 4269 \
  --host 0.0.0.0 \
  --api-key Squigg5McPeter! \
  --jinja \
  --chat-template-file /home/whistler/models/tuvak.jinja \
  --load-mode none \
  -np 3 \
  --ctx-checkpoints 32 \
  --checkpoint-min-step 4096 \
  --cache-ram 32768 \
  --mmproj /mnt/512gb_ssd/models/Qwen3.6-27B-mmproj-F16.gguf \
  --no-mmproj-offload \
  --image-min-tokens 1024 \
  --batch-size 2048 \
  --ubatch-size 512 \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --tensor-split 1,1,1,1,1,1,1,1 \
  --kv-unified \
  --slot-save-path /mnt/512gb_ssd/models/kv_cache \
  --seed 1016 \
  --spec-type draft-mtp \
  --spec-draft-p-min 0.74 \
  --spec-draft-n-max 3 \
  --split-mode tensor
```

## Key Parameters Explained

| Flag | Value | Purpose |
|------|-------|---------|
| -t | 28 | Thread count (28 physical cores, leaves headroom over 36) |
| -c | 262144 | Context window (256K) |
| -ngl | 99 | Full GPU offload (all layers to GPU) |
| -np | 3 | Parallel processing slots |
| --tensor-split | 1,1,1,1,1,1,1,1 | Equal split across all 8 GPUs |
| --split-mode | tensor | Tensor parallelism (row-split across GPUs) |
| --cache-type-k/v | q4_0 | KV cache quantization (critical for memory) |
| --cache-ram | 32768 | 32GB cache RAM allocation |
| --ctx-checkpoints | 32 | Context checkpointing for long context |
| --batch-size | 2048 | Prefill batch size |
| --ubatch-size | 512 | Ubatch size |
| --kv-unified | | Unified KV cache across GPUs |
| GGML_CUDA_P2P | 1 | Enable peer-to-peer GPU access |
| numactl --interleave | | Interleave memory across NUMA nodes |

## Tested Configurations

Working: tensor-split 1,1,1,1,1,1,1,1 with -np 2,3,4,5,6

## Build Configuration

```bash
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
  -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-rpath,/usr/local/cuda-11.8/targets/x86_64-linux/lib"
```

## Kepler sm_37 Patches Applied (llama_lazarus)

### Commit 992315e0e: Dimension-based strides fix
- File: ggml/src/ggml-cuda/ggml-cuda.cu
- Lines: ~1458-1464, ~1483-1489
- Fix: Replaced stride multiplication by block_size with dimension-based strides for contiguous quantized-to-compute_type conversion

### Commit 93c888df1: cuBLAS Kepler fixes
- File: ggml/src/ggml-cuda/ggml-cuda.cu
- Issues fixed:
  1. F16-to-FP32 pointer mismatch in batched path
  2. nullptr assertion on F32 compute path
  3. Wrong cuBLAS math mode for Kepler (CUBLAS_GEMM_DEFAULT_TENSOR_OP -> legacy path)

## Critical Notes

- DO NOT kill the running llama-server (PID ~200202) -- it hosts this session
- Model is Qwen3.6-27B with custom fine-tune (Fable-Fus-UnHeretic)
- Speculative decoding enabled (MTP draft, p-min 0.74, max 3 draft tokens)
- Multi-modal enabled with mmproj but offloaded to RAM
- KV cache saved/restored from NVMe
