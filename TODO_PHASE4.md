# Phase 4: TriAttention + TurboQuant KV Cache Optimization

**Goal:** Integrate TriAttention (trigonometric KV eviction) and TurboQuant (WHT-rotated polar quantization) for massive KV memory savings on NOUGHT's 8x K80 cluster. Enables 256K+ context at practical VRAM usage, and opens path to 1M+ context.

**Timing:** Post-async-pipeline-polish. Both modify KV cache layer; do after Patches 4 + 6 are stable.

**Reference implementations:**
- **atomicmilkshake/llama-cpp-turboquant** (feature/triattention): Complete TriAttention + TurboQuant integration
  - llama-triattention.h: Full API, calibration format, scoring math
  - llama-triattention.cpp: CPU scoring, RoPE inversion, pruning pipeline
  - ggml-turbo-quant.c: TurboQuant CPU kernels (turbo2_0, turbo3_0, turbo4_0)
  - GPU scoring kernels via ggml-cuda.h declarations (triattention_gpu_init, _score_head)
  - CLI flags: --triattention-stats, --triattention-budget, --triattention-window, --triattention-trigger, --triattention-log, --triattention-calibrate
- **TheTom/llama-cpp-turboquant** (feature/turboquant-kv-cache): TurboQuant+ codec stack
  - ggml-turbo-quant.c: PolarQuant with WHT rotation, QJL residual correction
  - ggml-quants.h: block_turbo2_0, block_turbo3_0, block_turbo4_0 type definitions
  - Cross-backend: CUDA, Metal, Vulkan kernels
  - Uses --cache-type-k turboN interface

**Why these help llama_wukong:**
- **TurboQuant:** Replaces q4_0 KV cache with turbo3 (3-bit). Your current config uses `--cache-type-k q4_0 --cache-type-v q4_0`; turbo3 gives ~3x savings over q4_0, ~6x over FP16. Direct drop-in via new cache types.
- **TriAttention:** Evicts low-importance KV entries using pre-RoPE trigonometric scoring. At 10% budget: ~10.7x compression vs FP16 baseline, zero reasoning accuracy loss on AIME25/MATH500. GPU scoring ~1000x faster than CPU (4-9ms vs 5900ms/prune).
- **Stacked:** TriAttention @10% + turbo3 = ~40x effective KV compression. Enables 1M+ context on 8x K80.

## 4.1 TurboQuant Integration (P1, 1-2 weeks)

### 4.1.1: Port TurboQuant quant types
- Source: TheTom/llama-cpp-turboquant ggml/src/ggml-turbo-quant.c
- Define block_turbo2_0 (2-bit), block_turbo3_0 (3-bit), block_turbo4_0 (4-bit) in ggml-quants.h
- Implement quantize_row_turboN_0_ref, dequantize_row_turboN_0, quantize_turboN_0
- Key math: WHT rotation (TURBO_D=128), Lloyd-Max centroids for N(0,1/128), norm correction
- Files to modify:
  - ggml/src/ggml-quants.h: Add block_turboN_0 structs, function declarations
  - ggml/src/ggml-quants.c: Add quantize/dequantize implementations (port from TheTom)
  - ggml/src/ggml-common.h: Register GGML_TYPE_TURBO2_0, TURBO3_0, TURBO4_0
  - ggml/src/CMakeLists.txt: Add ggml-turbo-quant.c to build

### 4.1.2: Port CUDA dequant kernels
- TurboQuant CPU path works; CUDA path needed for GPU offloaded KV cache
- Source: atomicmilkshake GPU kernels + TheTom CPU reference
- Implement dequantize_row_turboN_0 CUDA versions in ggml-cuda/quantize.cu
- Implement vec_dot_fattn_vec_KQ_turboN in ggml-cuda/fattn-common.cuh
- Wire into get_vec_dot_KQ() template switch (follows q4_0/q8_0 pattern)
- Kepler note: sm_37 lacks async copy; dequant-to-FP32 path is only option (same as q4_0)

### 4.1.3: Wire into KV cache type selection
- Extend --cache-type-k/v parsing to accept turbo2/turbo3/turbo4
- Source: TheTom common/arg.cpp patterns
- Files: common/arg.cpp, src/llama-context.cpp (KV cache creation path)

### 4.1.4: Test on NOUGHT
- Build with -DCMAKE_CUDA_ARCHS="37" for K80
- Test: `--cache-type-k turbo3 --cache-type-v turbo3` on Qwen3.6-27B
- Verify: correctness (compare outputs vs q4_0), VRAM savings (should be ~3x vs q4_0)
- Profile: no regression in decode TPS

## 4.2 TriAttention Integration (P1, 2-3 weeks)

### 4.2.1: Port calibration file format + loader
- Source: atomicmilkshake src/llama-triattention.h, llama-triattention.cpp
- Binary format: magic 0x54524941, version, per-head pre-RoPE Q stats (q_mean_real, q_mean_imag, q_abs_mean)
- Create src/llama-triattention.h (API), src/llama-triattention.cpp (loader, CPU scoring)
- Add triattention_init(), triattention_free(), triattention_prune() public API
- Calibration generation: port --triattention-calibrate flag to llama-cli

### 4.2.2: Port RoPE inversion + trigonometric scoring (CPU)
- triattention_invert_rope(): Post-RoPE K to pre-RoPE K using known positions and RoPE frequencies
- triattention_score_head(): Score cached keys using calibration stats (Eqs. 6-10 from paper)
- triattention_prune(): Combine per-head scores, select top-budget tokens, evict rest via seq_rm
- Files: src/llama-triattention.cpp
- Key structures: triattention_calibration, triattention_config, triattention_state

### 4.2.3: Integrate pruning hook into KV cache
- Hook into llama-context.cpp decode loop: after N tokens, check if cache exceeds budget
- Call triattention_prune() when budget + divide_length exceeded (SLACK mode) or interval hit
- Protect prefill tokens (--triattention-no-protect-prefill flag)
- Track cell_positions[] for correct RoPE inversion after eviction
- Files: src/llama-context.cpp, src/llama-kv-cache.h/.cpp (position tracking)

### 4.2.4: Port GPU scoring kernel
- Source: atomicmilkshake GPU state structure (triattention_gpu_state)
- Create ggml-cuda/triattention-gpu.cu with:
  - triattention_gpu_init(): Upload calibration data to device
  - triattention_gpu_score_head(): GPU kernel for per-head scoring (avoids 5900ms CPU cost)
- Wire into llama-triattention.cpp via ggml-cuda.h declarations
- Lazy init: only on first prune call; fallback to CPU if GPU init fails
- Kepler note: scoring kernel is math-only (sin/cos/mul); sm_37 handles it

### 4.2.5: Add CLI flags to llama-server/llama-cli
- Source: atomicmilkshake README flags
- --triattention-stats <file>: Calibration file (required)
- --triattention-budget <n>: Max KV entries post-prune (default: 2048)
- --triattention-window <n>: Protected recent tokens (default: 64)
- --triattention-trigger <mode>: interval or slack
- --triattention-log: Enable prune event logging
- --triattention-calibrate <corpus>: Generate calibration file (llama-cli only)
- --triattention-calibrate-out <file>: Output path for calibration
- Files: common/arg.cpp, tools/server/server.cpp, tools/cli/main.cpp

### 4.2.6: Generate calibration for target model
- Run on Qwen3.6-27B with representative corpus (code + reasoning text)
- Command: `llama-cli -m <model> -ngl 99 --triattention-calibrate corpus.txt --triattention-calibrate-out qwen3.6-27b.triattention`
- Validate: check file size (expect few MB), model name matches

### 4.2.7: Test on NOUGHT
- Test: `llama-server --triattention-stats qwen3.6-27b.triattention --triattention-budget 4096 --triattention-log`
- Verify: pruning triggers, tokens evicted, generation continues correctly
- Benchmark: compare TPS with/without TriAttention at same context length
- Benchmark: compare max context achievable vs without

## 4.3 Stacked: TriAttention + TurboQuant (P2, 1 week)

### 4.3.1: Verify compatibility
- TriAttention uses turbo2/turbo3 dequant functions already (from atomicmilkshake)
- Ensure llama_wukong's turbo3 dequant produces compatible output
- Key: TriAttention's RoPE inversion expects post-RoPE K in original embedding space

### 4.3.2: Run stacked config
- Test: `--cache-type-k turbo3 --cache-type-v turbo3 --triattention-stats model.triattention --triattention-budget 2048`
- Expected: ~40x effective KV compression vs FP16 baseline
- Expected on NOUGHT: Qwen3.6-27B at 1M+ context within 8x K80 VRAM

### 4.3.3: Correctness battery
- Run AIME25-style prompts (math reasoning, long context)
- Run NIAH (Needle-In-A-Haystack) at varied depths
- Verify outputs match full-attention baseline at same temperature

## 4.4 Kepler-Specific Optimizations

### 4.4.1: TurboQuant kernel tuning for sm_37
- Review CUDA kernels for sm_75+ only paths (dp4a, async copy)
- Ensure fallback to standard FP32 paths for Kepler
- Test build with -DCMAKE_CUDA_ARCHS="37" only

### 4.4.2: Tensor-split interaction
- TriAttention pruning is per-cache-instance; verify it works correctly with --tensor-split
- Each GPU holds a shard; eviction decisions must be coordinated or per-shard consistent
- Source: atomicmilkshake's handling; adapt for NOUGHT's 8-GPU tensor-split config
