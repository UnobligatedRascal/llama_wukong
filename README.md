# llama_wukong

> llama.cpp fork optimized for NVIDIA Kepler sm_37 hardware running Qwen3.8-Flash-Next (Qwen4 preview).

## What This Is

llama_wukong extends llama_lazarus with NOUGHT-specific optimizations targeting the Qwen3.8-Flash-Next architecture (180B total params, 6B active/token). Implements two-phase optimization strategy:

**Phase 1:** NOUGHT hardware optimizations (NUMA replication, async GPU pipeline, custom Kepler kernels, zero-copy memory)

**Phase 2:** Qwen4-exp architecture integration (GDN/QSA layer scheduling, hyper-connection tensors, ultra-sparse MoE routing, MTP head)

## Hardware Target

- Server: NOUGHT (dual Xeon E5-2697 v4, 128GB DDR4 ECC, 8x Tesla K80 GK210 sm_37)
- Model: Qwen3.8-Flash-Next (Qwen4 preview) — linear attention + MoE ultra-sparse

## Quick Reference

| File | Purpose |
|------|---------|
| ARCHITECTURE_REFERENCE.md | Original architectural pitch: memory topology, execution pipeline, Kepler exploitation vectors |
| llama_wukong.md | Full project plan (Phase 1 + Phase 2) |
| VERIFIED_CONFIG.md | Working launch command and hardware profile |
| PHASE1_TODO.md | Detailed implementation tasks |
| RESEARCH/ | Technical research notes per optimization area |

## Build (NOUGHT)

```bash
cd /home/whistler/llama_wukong
./scripts/build_wukong.sh
```

## Current Status

Phase 1: NUMA replication implemented and tested. Async pipeline, GDN kernel, zero-copy engram, and MoE batching pending.

Verified working: tensor-split across all 8 GK210 GPUs, -np 2-6, Qwen3.6-27B Q4_K_M at 256K context with speculative decoding.

## License

MIT License (inherited from upstream llama.cpp)

This is a fork of [UnobligatedRascal/llama_lazarus](https://github.com/UnobligatedRascal/llama_lazarus), which is itself a fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp). Upstream authors retain all rights to their original code.
