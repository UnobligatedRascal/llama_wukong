# GK210 / Tesla K80 Architectural Reference

**GPU Codename**: GK210 (GK210-885-A1 in Tesla K80)  
**Architecture**: Kepler 2.0 (compute capability 3.7)  
**Process**: TSMC 28 nm HKMG  
**Transistors**: ~7.1 billion  
**Die Size**: ~561–606 mm²  
**Primary Product**: NVIDIA Tesla K80 (dual-GPU board, 2× GK210)  
**References**:
- NVIDIA Kepler GK110/GK210 Architecture Whitepaper (primary source)
- NVIDIA Kepler Tuning Guide (CUDA Toolkit archive)
- CUDA Programming Guide / Compute Capability tables (cc 3.7)
- TechPowerUp GPU Database (Tesla K80 / GK210)
- Chips and Cheese microarchitectural analysis
- NVIDIA CUDA Handbook (SMX diagrams)

---

## 1. High-Level Chip Organization

| Component                  | Full GK210 | Tesla K80 (per GPU) | Notes |
|----------------------------|------------|---------------------|-------|
| Graphics Processing Clusters (GPCs) | 5         | 5                   | Each GPC contains multiple SMX + raster engines |
| Streaming Multiprocessors (SMX)    | 15        | 13                  | 2 SMX disabled for power/TDP |
| Texture Processing Clusters (TPC)  | 15        | 13                  | 1 TPC = 1 SMX in Kepler |
| Memory Controllers         | 6 × 64-bit | 6 × 64-bit         | 384-bit GDDR5 interface |
| ROPs                       | 48        | 48                  | |
| TMUs                       | 240       | 208                 | |
| CUDA Cores (FP32)          | 2880      | 2496                | 192 per SMX |
| FP64 Units                 | 960       | 832                 | 64 per SMX (1:3 FP32 ratio) |
| SFUs                       | 480       | 416                 | 32 per SMX |
| LD/ST Units                | 480       | 416                 | 32 per SMX |

**Tesla K80 Board**:
- 2× independent GK210 GPUs
- Total CUDA cores: 4992
- Memory: 24 GB GDDR5 (12 GB per GPU)
- Memory bandwidth: 480 GB/s aggregate (240 GB/s per GPU @ 5 Gbps effective)
- TDP: 300 W (passive cooling common)
- PCIe 3.0 ×16
- Base clock ~560–562 MHz, Boost up to 824–875 MHz (GPU Boost 2.0, dynamic)

---

## 2. Streaming Multiprocessor (SMX) – Core Compute Unit

### 2.1 Execution Resources (per SMX)
- **CUDA Cores (SP)**: 192 fully pipelined FP32 ALUs + INT32 ALUs
- **FP64 Cores (DP)**: 64 dedicated double-precision units
- **Special Function Units (SFU)**: 32 (sin, cos, exp, rsqrt, etc.)
- **Load/Store Units (LD/ST)**: 32
- **Texture Units**: 16 (shared with texture pipeline)
- **Quad Warp Schedulers**: 4
- **Instruction Dispatch Units**: 8 (2 per scheduler → dual-issue)
- **Warp Size**: 32 threads
- **Max Warps / SMX**: 64
- **Max Threads / SMX**: 2048
- **Max Thread Blocks / SMX**: 16
- **Max Threads / Block**: 1024

### 2.2 Register File
- **Total 32-bit Registers / SMX**: 131072 (512 KB)
  - Doubled vs. GK110 (65536 / 256 KB)
- **Registers per Scheduler Partition**: ~32k (4 partitions)
- **Max Registers / Thread**: 255
- **Max Registers / Thread Block**: 65536
- **Banking**: Multi-banked for concurrent access by schedulers
- **Purpose**: High occupancy for register-heavy kernels (especially FP64)

### 2.3 Shared Memory + L1 Cache (Configurable)
- **Total Configurable On-Chip Memory**: 128 KB (doubled vs. GK110’s 64 KB)
- **Possible Configurations** (Shared / L1):
  - 112 KB / 16 KB
  - 96 KB / 32 KB
  - 80 KB / 48 KB
- **Shared Memory Bandwidth**: 256 B/clock (64-bit and larger loads)
- **Banks**: 32 banks (4-byte width default; 8-byte for 64-bit)
- **L1 Cache Behavior**:
  - Default: L1 used only for local memory (register spills, stack)
  - Global loads → L2 only (can opt-in to Fermi-style L1 caching of global via compiler flag `-dlcm=ca` or `cudaDeviceSetCacheConfig`)
- **Read-Only Data Cache (Texture Cache)**: 48 KB per SMX
  - Compiler-directed or `__ldg()` intrinsic
  - Higher tag bandwidth, supports unaligned accesses at full speed
  - Independent of L1/shared

### 2.4 Instruction Pipeline & Scheduling
- **Quad Warp Scheduler**: Selects up to 4 warps per cycle
- **Dual Issue**: Each scheduler can issue 2 independent instructions per cycle
- **Instruction Pairing**: FP32 + FP64, integer + memory, etc. allowed
- **Control Words**: 64-bit control word every 7 instructions (scheduling hints)
- **Scoreboarding**: Hardware dependency tracking
- **No out-of-order execution** beyond dual-issue

---

## 3. Memory Hierarchy

| Level                  | Size                  | Scope          | Bandwidth / Notes |
|------------------------|-----------------------|----------------|-------------------|
| Registers              | 512 KB / SMX         | Per-thread    | Highest bandwidth |
| Shared Memory          | up to 112 KB / SMX   | Per-block     | 256 B/clock      |
| L1 Cache               | up to 48 KB / SMX    | Per-SMX       | Local + optional global |
| Read-Only / Texture Cache | 48 KB / SMX       | Per-SMX       | High tag bandwidth |
| L2 Cache               | 1536 KB (1.5 MB)     | Chip-wide     | Unified for loads/stores/textures; ~12 × 128 KB slices |
| Device Memory (GDDR5)  | 12 GB / GPU          | Chip-wide     | 240 GB/s (384-bit @ 5 Gbps) |
| Constant Memory        | 64 KB                | Chip-wide     | Cached in constant cache |
| Texture Memory         | Bound to device mem  | Chip-wide     | Uses texture units + cache |

- **ECC**: Supported (reduces effective capacity/bandwidth when enabled)
- **Atomic Operations**: Native 32-bit and 64-bit atomics (Add, CAS, Min, Max, And, Or, Xor, Exch)
- **Unified Address Space**: Yes (Kepler)
- **Memory Controllers**: 6 × 64-bit GDDR5

---

## 4. Key Architectural Features (cc 3.7)

- **Hyper-Q**: 32 concurrent hardware work queues (MPI-friendly, reduces false serialization)
- **Dynamic Parallelism**: Kernels can launch child kernels
- **Shuffle Instructions**: Warp-level data exchange (`__shfl`, `__shfl_up`, etc.) without shared memory
- **Increased Registers per Thread**: 255 (vs. 63 on Fermi)
- **GPU Boost 2.0**: Dynamic clock scaling based on power/thermal headroom (many intermediate boost points on K80)
- **Bindless Textures**: Yes
- **Improved Atomics**: Full 64-bit support
- **PCIe 3.0**: Full support
- **No Tensor Cores**, no WMMA, no independent thread scheduling (pre-Volta)

---

## 5. Functional Units Summary (per SMX)

| Unit Type          | Count | Throughput Notes |
|--------------------|-------|------------------|
| FP32 CUDA Core     | 192   | FMA per clock   |
| INT32 ALU          | 192   | Shared with FP32 path |
| FP64 Unit          | 64    | 1/3 of FP32 rate |
| SFU                | 32    | Special functions |
| LD/ST              | 32    | Memory ops      |
| Texture Unit       | 16    | Filtering + cache |
| Warp Scheduler     | 4     | Dual-issue each |

---

## 6. Limits & Occupancy (cc 3.7)

- Max resident warps / SMX: 64
- Max resident threads / SMX: 2048
- Max resident blocks / SMX: 16
- Max shared memory / block: 48 KB (hardware limit still applies even if SMX has more)
- Max registers / thread: 255
- Grid dimensions: 2³¹−1 in X

**Occupancy Notes**:
- GK210’s doubled registers + shared memory dramatically improves occupancy for register- or shared-memory-bound kernels compared to GK110.
- Example: 63 regs/thread, 256 threads/block → 100% occupancy possible on GK210 vs. 50% on GK110.

---

## 7. Optimization Opportunities for Legacy Code (llama.cpp context)

- Prefer **shared memory** configurations that maximize shared (112/16) for matrix tiles.
- Use **shuffle** instructions aggressively to reduce shared memory pressure.
- Force **FP32** compute paths (no fast FP16 arithmetic on Kepler).
- Exploit high register file for larger tile sizes / reduced spilling.
- Hyper-Q enables better multi-stream / multi-slot concurrency.
- Leverage read-only cache (`__ldg`) for weights.
- MMQ-style integer kernels still viable (DP4A not present; use pure integer or FP32).
- Avoid tensor-core / BF16 / WMMA paths entirely.
- GPU Boost + power management via `nvidia-smi` / NVML for sustained clocks.

---

## 8. Machine-Readable Key-Value Summary

```yaml
architecture: Kepler_2.0
compute_capability: 3.7
chip: GK210
product: Tesla_K80
smx_count_full: 15
smx_count_k80: 13
cuda_cores_per_smx: 192
fp64_units_per_smx: 64
sfu_per_smx: 32
ldst_per_smx: 32
register_file_kb_per_smx: 512
registers_32bit_per_smx: 131072
max_regs_per_thread: 255
shared_l1_total_kb: 128
shared_configs:
  - {shared: 112, l1: 16}
  - {shared: 96, l1: 32}
  - {shared: 80, l1: 48}
readonly_cache_kb_per_smx: 48
l2_cache_kb: 1536
memory_bus_bits: 384
memory_type: GDDR5
memory_per_gpu_gb: 12
bandwidth_per_gpu_gbs: 240.6
warp_schedulers: 4
dual_issue: true
max_warps_per_smx: 64
max_threads_per_smx: 2048
hyper_q: true
dynamic_parallelism: true
shuffle_instructions: true
gpu_boost: 2.0