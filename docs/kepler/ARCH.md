# GK210 / Tesla K80 Architectural Reference

**GPU Codename**: GK210 (GK210-885-A1 in Tesla K80)
**Architecture**: Kepler 2.0 (compute capability 3.7)
**Process**: TSMC 28 nm HKMG
**Primary Product**: NVIDIA Tesla K80 (dual-GPU board, 2x GK210)

---

## High-Level Chip Organization

| Component | Full GK210 | Tesla K80 (per GPU) |
|-----------|------------|---------------------|
| Streaming Multiprocessors (SMX) | 15 | 13 (2 disabled for TDP) |
| CUDA Cores (FP32) | 2880 | 2496 |
| FP64 Units | 960 | 832 |
| SFUs | 480 | 416 |
| Memory Controllers | 6 x 64-bit | 6 x 64-bit |
| Memory | 24 GB GDDR5 | 12 GB per GPU |
| Memory Bandwidth | 480 GB/s aggregate | 240 GB/s per GPU |
| TDP | - | 300 W (passive) |

## SMX Compute Unit (per SMX)

- CUDA Cores (FP32): 192 fully pipelined
- FP64 Cores: 64 (1:3 FP32 ratio)
- SFU: 32 (sin, cos, exp, rsqrt, etc.)
- LD/ST Units: 32
- Texture Units: 16
- Warp Schedulers: 4 (dual-issue)
- Warp Size: 32 threads
- Max Warps / SMX: 64
- Max Threads / SMX: 2048
- Registers: 131072 x 32-bit (512 KB)
- Shared Memory + L1: 128 KB configurable (112/16, 96/32, or 80/48 KB split)
- Read-Only / Texture Cache: 48 KB per SMX

## Key Features (cc 3.7)

- Hyper-Q: 32 concurrent hardware work queues
- Dynamic Parallelism: kernels can launch child kernels
- Shuffle Instructions: warp-level data exchange without shared memory
- Improved Atomics: full 64-bit support
- GPU Boost 2.0: dynamic clock scaling
- PCIe 3.0

**No Tensor Cores, no WMMA, no independent thread scheduling (pre-Volta)**

## Limits

- Max resident warps / SMX: 64
- Max shared memory / block: 48 KB
- Max registers / thread: 255
- Grid dimensions: 2^31-1 in X

## Optimization Notes for llama.cpp

- Force FP32 compute paths (no fast FP16 arithmetic on Kepler)
- Avoid tensor-core / BF16 / WMMA paths entirely
- Prefer shared memory configurations maximizing shared (112/16) for matrix tiles
- Exploit high register file for larger tile sizes
- Hyper-Q enables better multi-stream / multi-slot concurrency
- Memory bandwidth is the hard limit - not compute
- MMQ-style integer kernels viable (DP4A not present; use pure integer or FP32)
