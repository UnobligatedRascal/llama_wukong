# Phase 4: TriAttention + TurboQuant KV Cache Optimization

**Goal:** Massive KV memory savings on NOUGHT's 8x K80 cluster via TurboQuant compression and TriAttention pruning. Enables 256K–1M+ context at practical VRAM usage.

**Last updated:** 2026-09-08

---

## TurboQuant KV Cache — COMPLETE ✓

**Status:** Fully integrated and usable via `--cache-type-k/--cache-type-v`.

| Type | Bits/value | Compression vs FP16 | Use case |
|------|-----------|---------------------|----------|
| turbo4_0 | 4.25 bpw | 3.8× | Recommended starting point |
| turbo3_0 | 3.5 bpw | 4.6× | Good quality/compression tradeoff |
| turbo2_0 | 2.5 bpw | 6.4× | Maximum compression, quality loss |

**Usage:**
```bash
./build/bin/llama-server -m model.gguf --cache-type-k turbo4_0 --cache-type-v turbo4_0
```

**Implementation:**
- Types: GGML_TYPE_TURBO3_0/4_0/2_0 in ggml.h
- CPU: ggml-turbo-quant.c (quantize/dequantize_row_turbo{2,3,4}_0)
- CUDA: turbo-quant.cuh (quantize), dequantize.cuh (dequantize_turbo{2,3,4}_0)
- Wired: ggml-cuda.cu (MUL_MAT, GET_ROWS, SET_ROWS), set-rows.cu, arg.cpp, llama-bench.cpp
- sm_37: All kernels FP32, no tensor cores; fully K80-compatible

**Remaining:**
- Verification benchmarking (blocked: all GPUs in production use)

---

## TriAttention — GPU KERNELS DONE, SERVER INTEGRATION PENDING

**Status:** GPU scoring path complete. CPU glue code and CLI integration needed.

**What it does:** Evicts low-importance KV entries using pre-RoPE trigonometric scoring. At 10% budget: ~10.7× compression vs FP16 baseline, zero reasoning accuracy loss on AIME25/MATH500. GPU scoring ~1000× faster than CPU (4–9ms vs 5900ms/prune).

**Stacked benefit:** TriAttention @10% + turbo3 = ~40× effective KV compression. Enables 1M+ context on 8x K80.

### Completed ✓

- **GPU scoring kernel:** triattention-score.cu (542 lines)
  - Full scoring: dequantize → inverse WHT (turbo2/3) → inverse RoPE → trig scoring
  - Supports turbo2_0/turbo3_0/turbo4_0, Q8_0, F16, F32
- **GPU API:** triattention_gpu_init/score_head/scores_to_host/upload_cells/alloc_scores/free_dev/free
  - Declared in ggml-cuda.h, exported from libggml-cuda.so
- **sm_37 audit:** FP32 math only, no tensor cores; fully K80-compatible

### Remaining

#### 4.2.1: CPU-side llama-triattention.h/cpp (P1, 1-2 weeks)
- Port from atomicmilkshake/llama-cpp-turboquant (feature/triattention)
- Binary calibration format: magic 0x54524941, version, per-head pre-RoPE Q stats
- triattention_init(), triattention_free(), triattention_prune() public API
- triattention_invert_rope(): Post-RoPE K → pre-RoPE K
- triattention_score_head(): CPU fallback scoring (Eqs. 6-10 from paper)
- triattention_prune(): Combine per-head scores, select top-budget tokens, evict via seq_rm

#### 4.2.2: Pruning hook in llama-context.cpp (P1, 3-5 days)
- After N tokens, check if cache exceeds budget
- Call triattention_prune() when budget + divide_length exceeded (SLACK mode) or interval hit
- Protect prefill tokens (--triattention-no-protect-prefill flag)
- Track cell_positions[] for correct RoPE inversion after eviction

#### 4.2.3: CLI flags (P1, 2-3 days)
- --triattention-stats <file>: Calibration file (required)
- --triattention-budget <n>: Max KV entries post-prune (default: 2048)
- --triattention-window <n>: Protected recent tokens (default: 64)
- --triattention-trigger <mode>: interval or slack
- --triattention-log: Enable prune event logging
- --triattention-calibrate <corpus>: Generate calibration (llama-cli only)
- --triattention-calibrate-out <file>: Output path for calibration

#### 4.2.4: Calibration tool (P1, 3-5 days)
- Port --triattention-calibrate to llama-cli
- Generate calibration on Qwen3.6-27B with representative corpus
- Command: `llama-cli -m <model> -ngl 99 --triattention-calibrate corpus.txt --triattention-calibrate-out qwen3.6-27b.triattention`

#### 4.2.5: Multi-GPU tensor-split coordination (P2, 1 week)
- TriAttention pruning is per-cache-instance
- Verify eviction decisions are coordinated or per-shard consistent across 8 GPUs
- Source: atomicmilkshake's handling; adapt for NOUGHT's tensor-split config

#### 4.2.6: Testing (P2, 1 week)
- Test: `llama-server --triattention-stats model.triattention --triattention-budget 4096 --triattention-log`
- Verify: pruning triggers, tokens evicted, generation continues correctly
- Benchmark: TPS with/without TriAttention at same context length
- Benchmark: max context achievable vs without
- Correctness: AIME25-style prompts, NIAH at varied depths

### Stacked: TriAttention + TurboQuant (P2, 1 week)

#### 4.3.1: Verify compatibility
- TriAttention uses turbo2/turbo3 dequant functions
- Ensure llama_wukong's turbo3 dequant produces compatible output
- Key: TriAttention's RoPE inversion expects post-RoPE K in original embedding space

#### 4.3.2: Run stacked config
- Test: `--cache-type-k turbo3 --cache-type-v turbo3 --triattention-stats model.triattention --triattention-budget 2048`
- Expected: ~40× effective KV compression vs FP16 baseline
- Expected on NOUGHT: Qwen3.6-27B at 1M+ context within 8x K80 VRAM

---

## Reference Implementations

- **atomicmilkshake/llama-cpp-turboquant** (feature/triattention): Complete TriAttention + TurboQuant integration
  - llama-triattention.h/cpp: Full API, calibration format, scoring math
  - ggml-turbo-quant.c: TurboQuant CPU kernels
  - GPU scoring via ggml-cuda.h declarations
  - CLI flags: --triattention-stats, --triattention-budget, etc.
- **TheTom/llama-cpp-turboquant** (feature/turboquant-kv-cache): TurboQuant+ codec stack
  - ggml-turbo-quant.c: PolarQuant with WHT rotation, QJL residual correction
  - ggml-quants.h: block_turbo2_0, block_turbo3_0, block_turbo4_0 definitions
  - Cross-backend: CUDA, Metal, Vulkan kernels

---
*UnobligatedRascal — Making old hardware sing.*
