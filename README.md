# llama_wukong

NOUGHT-optimized llama.cpp fork targeting Qwen3.8-Flash-Next (Qwen4 preview).

Base: llama_lazarus (Kepler sm_37 cuBLAS fixes), commit 93c888df1

## Quick Reference

- VERIFIED_CONFIG.md - working launch command and hardware profile
- llama_wukong.md - full project plan (Phase 1 + Phase 2)
- PHASE1_TODO.md - detailed implementation tasks
- RESEARCH/ - technical research notes per optimization area
- scripts/ - build and benchmark scripts

## Build (NOUGHT)

```bash
cd /home/whistler/llama_wukong
./scripts/build_wukong.sh
```

## Current Status

Phase 1: Hardware optimization planning complete. Implementation pending.

Verified working: tensor-split across all 8 GK210 GPUs, -np 2-6,
Qwen3.6-27B Q4_K_M at 256K context with speculative decoding.

## Version Control

Private repo: UnobligatedRascal/llama_wukong
