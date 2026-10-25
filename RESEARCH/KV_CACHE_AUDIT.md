# KV Cache Visualization & Audit Methodology

**Purpose:** Diagnose why KV cache quantization (q4_0/q8_0/turbo*) shows no speedup vs f16 on NOUGHT with tensor-split.

**Hypothesis:** Either (a) quantization isn't being applied, (b) dequantization overhead masks bandwidth savings, or (c) tensor-split meta backend adds overhead that negates quantization benefits.

---

## KV Cache Pipeline Trace

### 1. Allocation (llama-kv-cache.cpp:235)
```cpp
ggml_tensor * k = ggml_new_tensor_3d(ctx, type_k, n_embd_k_gqa, kv_size, n_stream);
ggml_tensor * v = ggml_new_tensor_3d(ctx, type_v, n_embd_v_gqa, kv_size, n_stream);
```
- **type_k/type_v** come from CLI args (--cache-type-k/--cache-type-v)
- Tensor is created with quantized type directly
- **VERIFY:** Add log at allocation to confirm actual type

### 2. Split State Computation (llama-model.cpp:791)
```cpp
const int64_t blck_size = (std::regex_match(tensor_name, pattern_kv_cache))
    ? ggml_blck_size(tensor->type)      // KV cache's OWN block size
    : ggml_blck_size(tc.tensor_axis_0->type);  // Reference weight tensor
```
- Option A fix: uses KV cache tensor's own block size
- Ensures split boundaries align with quantization blocks
- **VERIFY:** Log split_state for KV cache tensors

### 3. Attention Computation (llama-graph.cpp:2560+)
Two paths:

**Flash Attention (fattn.cu):**
```cpp
static bool ggml_cuda_fattn_kv_type_supported(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32: case GGML_TYPE_F16: return true;
        case GGML_TYPE_Q4_0: case GGML_TYPE_Q8_0: case GGML_TYPE_BF16: return true;
        default: return false;  // turbo2/3/4 NOT supported
    }
}
```
- q4_0/q8_0: supported, dequantize inline in FA kernel
- turbo*: NOT supported, falls back to non-FA path

**Non-FA Path (mul_mat):**
- k = ggml_permute(ctx0, k, 0, 2, 1, 3);  // quantized tensor passed directly
- kq = ggml_mul_mat(ctx0, k, q);  // dequantize in CUDA kernel
- Dequantization happens in ggml-cuda.cu mul_mat dispatch

### 4. CUDA Kernel Dequantization (ggml-cuda.cu)
- mul_mat dispatches to appropriate kernel based on src0->type (K's type)
- For quantized K: dequantize_row_* called inline in kernel
- Dequantization is FP32 math on sm_37 (no tensor cores)

---

## Audit Points (Instrumentation)

### Point A: Confirm KV Cache Allocation Type
**File:** src/llama-kv-cache.cpp, after line 240
```cpp
LLAMA_LOG_INFO("%s: layer %d: KV cache allocated as K=%s V=%s\n",
    __func__, il, ggml_type_name(k->type), ggml_type_name(v->type));
```

### Point B: Confirm Split State for KV Cache
**File:** src/llama-model.cpp, after line 820
```cpp
if (std::regex_match(tensor_name, pattern_kv_cache)) {
    LLAMA_LOG_INFO("%s: KV cache split: axis=%d blck_size=%ld segments=%d\n",
        __func__, split_state.axis, blck_size, split_state.n_segments);
}
```

### Point C: Confirm Attention Path Taken
**File:** src/llama-graph.cpp, in build_attn after line 2560
```cpp
LLAMA_LOG_DEBUG("%s: layer %d: use_flash_attn=%d K_type=%s V_type=%s\n",
    __func__, il, use_flash_attn, ggml_type_name(k->type), ggml_type_name(v->type));
```

### Point D: CUDA Kernel Timing
Use nvprof/nvvp to profile:
```bash
nvprof --metrics sm__instructionsissued.sum,sm__warps_active.sum,mem__throughput.avg.dram.read \
    ./build/bin/llama-server [args]
```

Key metrics:
- `mem__throughput.avg.dram.read` - should be LOWER with q4_0 vs f16
- `sm__instructionsissued.sum` - dequantization adds instructions
- Kernel time breakdown: dequantize vs compute

---

## Expected Behavior (If Working Correctly)

| KV Type | VRAM vs F16 | Expected Speedup | Notes |
|---------|-------------|------------------|-------|
| f16 | 1.0× | baseline | Full precision |
| q8_0 | 0.5× | 1.1-1.3× | 8-bit, minimal dequant overhead |
| q4_0 | 0.25× | 1.3-1.8× | 4-bit, more dequant overhead |
| turbo4_0 | ~0.27× | 1.2-1.6× | Non-FA path, WHT rotation |
| turbo3_0 | ~0.22× | 1.1-1.5× | Non-FA path, more compression |

**If speed is identical across all types, the bottleneck is:**
1. Compute-bound (not memory-bound) - dequant + FP32 math dominates
2. PCIe bottleneck (tensor-split across 8 GPUs via host memory)
3. Meta backend overhead masking quantization benefits

---

## Diagnostic Commands

### 1. Check actual KV cache type at runtime
```bash
./build/bin/llama-server -m model.gguf --cache-type-k q4_0 --cache-type-v q4_0 \
    --split-mode tensor --tensor-split 1,1,1,1,1,1,1,1 -lv 3 2>&1 | grep -i "kv.*cache.*type"
```

### 2. Profile memory bandwidth
```bash
nvprof --metrics mem__throughput.avg.dram.read,mem__throughput.avg.dram.write \
    ./build/bin/llama-bench -m model.gguf -t 36 -ngl 99 \
    --cache-type-k q4_0 --cache-type-v q4_0 --tensor-split 1,1,1,1,1,1,1,1 \
    --split-mode tensor -b 128 -ngl 99 -n 256
```

### 3. Compare kernel times
```bash
nvprof --print-gpu-trace ./build/bin/llama-bench [same args as above] 2>&1 | \
    grep -E "(mul_mat|flash_attn|dequant)" | head -30
```

### 4. Check FA path usage
```bash
./build/bin/llama-server -m model.gguf --cache-type-k q4_0 --cache-type-v q4_0 \
    --flash-attn 1 --split-mode tensor --tensor-split 1,1,1,1,1,1,1,1 -lv 4 2>&1 | \
    grep -i "flash_attn"
```

---

## Visualization Tools

### Option 1: Python Profiler Script
Create `scripts/kv_cache_profiler.py`:
- Parse nvprof output
- Graph memory bandwidth vs KV cache type
- Show kernel time breakdown
- Compare quantized vs f16

### Option 2: Nsight Systems GUI
```bash
nsys profile --trace=cuda,nvtx,osrt \
    ./build/bin/llama-server [args]
nsys-ui report.qdrep
```
Shows full timeline: CPU, CUDA kernels, memory transfers, NCCL comms.

### Option 3: Simple Timing Script
```bash
for type in f16 q8_0 q4_0; do
    echo "=== KV Cache Type: $type ==="
    ./build/bin/llama-bench -m model.gguf \
        --cache-type-k $type --cache-type-v $type \
        --tensor-split 1,1,1,1,1,1,1,1 --split-mode tensor \
        -b 128 -n 256 2>&1 | grep -E "(tps|mem)"
done
```

---

## Critical Questions to Answer

1. **Is KV cache actually allocated as quantized type?** (Point A)
2. **Is flash attention being used or falling back to mul_mat?** (Point C)
3. **Is memory bandwidth actually lower with quantized KV?** (nvprof metrics)
4. **Is dequantization overhead dominating compute time?** (kernel timing)
5. **Is tensor-split NCCL communication the bottleneck?** (Nsight timeline)

---

*UnobligatedRascal — "If it's worth doing, it's worth doing RIGHT."*
