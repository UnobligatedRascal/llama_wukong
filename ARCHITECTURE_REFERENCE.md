# Qwen3.8-Flash-Next Architecture Reference

Source: initial_pitch_llama_wukong_RECOVERED.md — original architectural pitch recovered 2026-09-05.

## Unified Tensor Map & Memory Topography

The architecture decouples heavy parametric weights from dense VRAM constraints by splitting
execution across three memory boundaries. Total parameter footprint: ~180B parameters.

```
[System RAM / NVMe via MMAP] ────────► 51B N-Gram Engram Table (Bigram/Trigram Token-Space Hash)

│ (Zero-Copy/Host-Alloc Pointer Lookups)
▼

[VRAM Base Pool (96GB Boundary)] ────► 125B Sparse MoE Backbone (BF16/FP8 native; UD-IQ3_XXS target)

│ (Hyper-Connection & Gated Residual Streams)
▼

[SRAM / SM Registers (512KB/SM)] ────► Fixed-Size Recurrent State Matrix (Linear DeltaNet layers)
```

### Tensor Topology Overrides (qwen4exp)

**Eliminated Tensors:**
- Traditional input_layernorm and post_attention_layernorm deleted from block definitions

**Injected Tensors** (four explicitly bound hyper-connection tensors per block layer):
- hc_norm (Structural normalization scale)
- block_inject_weight (Direct residual projection)
- input_mix_weight_down / input_mix_weight_up (Dynamic gating matrices)

**Speculative Head:**
- output_hc_* tensors encapsulate a 4B parameter Multi-Token Prediction (MTP) head
- Designed for single-pass parallel decoding

## Grouped Architectural Transforms

Layers discard uniform block progression. They cluster into rigid groups of four, trading
quadratic attention cost for recurrent linear state retention.

```
┌────────────────────────────────────────────────────────┐
▼ │
[Input Stream] ──► [GDN Layer] ──► [GDN Layer] ──► [GDN Layer] ──┴──► [QSA Layer] ──► [Output Stream]

(Linear Recurrence via Matrix State)              (Global Sparse Indexer)
```

### Gated DeltaNet (GDN) Transform

- **Operation:** Computes linear attention by converting the sequence history into a fixed-size matrix state
- **Mathematical Bound:** Replaces KV-cache matrix calculations with an update step per token
- **Hardware Target:** Optimized for high-register hardware. The recurrent matrix state can be pinned inside SM local memory/registers, bypassing global memory reads during execution loops

### Qwen Sparse Attention (QSA) Transform

- **Operation:** Acts as the global context anchor every fourth layer
- **Mechanism:** Utilizes a low-rank micro-block indexer to dynamically select high-salience historic tokens, preventing the linear context drift common in pure recurrent models

### Gated Residual (GR) Pipeline

- **Operation:** Breaks traditional single-line residual streams
- **Mechanism:** Widens the activation pipe into 4 parallel branches. Cross-talk is regulated at runtime via the input_mix_weight gating tensor to maximize cross-layer information transfer without deepening the block stack

## Execution Algorithms & Matrix Routing

```
┌──► [Shared Expert] (1 Always Active)
│

[MLP Block Input] ──┼──► [Top-K Router] ──► [Selects 10 of 512 Sparse Experts]
│ (Coarse-Grained Batch Grouping via Pointers)
▼
[Fused Multi-Token Prediction Head] ──► (Generates Tokens N+1...N+4 via Speculative Buffers)
```

### 512-Expert Ultra-Sparse Routing

- **Structure:** The Feed-Forward Network (FFN) contains 512 total experts. Each expert is structurally clamped to a thin 640-channel width
- **Routing Algorithm:** High-sparsity Top-K selection. Only 10 routed experts + 1 permanently bound shared expert fire per token
- **Active Compute:** Drops runtime load down to ~6B active parameters per token execution pass, making a 125B parameter model compute-light but highly memory-bandwidth bound

### Fused Multi-Token Prediction (MTP)

- **Operation:** The execution head uses a 4B parameter speculative structure
- **Mechanism:** Calculates logits for the immediate token while concurrently utilizing intermediate hidden layers to output speculative probabilities for tokens N+2, N+3, and N+4 in a single forward execution

### Muon Optimizer Splitting

- **Algorithmic Constraint:** Weight files are trained via the Muon orthogonalizer
- **Inference Expectation:** Combined parameter blocks (like QKV projections, SwiGLU gates, and internal GDN linear layers) are mathematically segregated into precise independent linear transformations. Upstream quantization or tensor-splitting algorithms must preserve these structural partition lines to prevent severe mathematical skew

## Kepler Hardware Exploitation & Porting Vectors

```
[Host System RAM] ──(Zero-Copy Pointer)──► [PCIe Gen3 Bus] ──► [Tesla K80 Physical Card]
- 51B N-Gram Matrix Lookups ├─ Chip 0: Compute Token N
- Global KV Scratch Space └─ Chip 1: Pre-Fetch Token N+1 (via PLX)
```

### Key Porting Requirements

1. **FP32 Native Fallback Routing:** Force tensor evaluation through legacy cublasSgemmBatched paths. All modern TF32 and half-precision hardware execution primitives must be stripped or explicitly emulated via single-precision paths to prevent runtime illegal instruction faults

2. **Zero-Copy Host Pointer Lookups:** Allocate the 51B N-gram table in host memory via cudaHostAlloc. Kepler kernels look up bigram/trigram pointers dynamically across the PCIe bus on-demand, reserving physical VRAM for active MoE weights

3. **Asynchronous Ring Pipelining:** Exploit dual-core K80 layouts by setting up explicit cudaStreamWaitEvent boundaries. Map Chip 0 to process layer calculations while asynchronously streaming the sparse MoE expert memory locations for the next token to Chip 1 across the onboard PLX switch, hiding PCIe Gen3 transit overhead behind execution loops

4. **Coarse-Grained Expert Batching:** Rearrange routing arrays at the framework level to group multiple token sequences targeting overlapping expert indices. This converts unpredictable routing jumps into linear data streaming across the 240 GB/s internal memory bus

## Execution Pipeline Diagram

```mermaid
graph TD
%% Custom Styles for Clarity
classDef memory fill:#1a1a2e,stroke:#162447,stroke-width:2px,color:#fff;
classDef tensor fill:#0f3460,stroke:#e94560,stroke-width:1px,color:#fff;
classDef cpu fill:#2b2b2b,stroke:#4e4e4e,stroke-width:1px,color:#e0e0e0;
classDef gpu fill:#1b4d3e,stroke:#2d7256,stroke-width:2px,color:#fff;
classDef sync fill:#4a3b1a,stroke:#d4af37,stroke-width:1px,color:#fff;

%% PHASE 1: INPUT PROCESSING & SYSTEM RAM PRE-FETCH
subgraph PHASE_1 [Phase 1: Token Input & System RAM Lookups]
A[Token Input Sequence] --> B[Host Token Array]
B --> C[Compute Bigram/Trigram Hashes]
C --> D[System RAM: 51B N-Gram Engram Table]:::memory
D -->|cudaHostAlloc Zero-Copy Pointer| E[Extract Bias Vectors over PCIe Gen3]:::cpu
end

%% PHASE 2: PIPELINED BLOCK TRANSFORMS
subgraph PHASE_2 [Phase 2: Hybrid Group-of-4 Block Transforms]
E --> F[Hyper-Connection Stream Initialized]
F --> G[GDN Layers 1-3: Linear Attention Recurrence]:::gpu
G1[hc_norm / block_inject_weight / input_mix_weight_up+down]:::tensor -.->|Inject Block Topology| G
G -->|Update Matrix State| H[Pin Fixed-Size Recurrent State Matrix in SM Registers]:::memory
H -->|O(1) Sequence Compress| I[Pass Compressed Hidden State]
I --> J[QSA Layer 4: Global Context Anchor]:::gpu
J1[Low-Rank Micro-Block Indexer]:::tensor -.->|Filter Salient Tokens| J
J -->|Cache Intermittent State| K[KV Cache Pool: Forced q8_0 Precision in VRAM]:::memory
end

%% PHASE 3: MOE MATRIX MULTIPLICATION & ROUTING
subgraph PHASE_3 [Phase 3: Ultra-Sparse MoE Routing & Expert Batching]
K --> L[Gated Residual GR Pipeline: Widen to 4 Parallel Branches]
L --> M[Top-K Router Assessment]
M -->|Selects 10 of 512 Sparse Experts + 1 Shared| N[Framework Pointer Sorting Queue]
O[Coarse-Grained Expert Batching]:::sync -.->|Rearrange Token Sequence Arrays| N
N --> P[Legacy GPU Path: cublasSgemmBatched Execution]:::gpu
P -->|Asynchronous Ring Pipelining| Q[K80 Chip 0 computes Layer N / Chip 1 pre-fetches Layer N+1 via PLX]:::sync
end

%% PHASE 4: HEAD DECODING & PARALLEL SPECULATION
subgraph PHASE_4 [Phase 4: Output Decoding & Speculative MTP Head]
Q --> R[Gather Block Outputs via final hyper-connection mix]
R --> S[Fused Multi-Token Prediction Head]:::gpu
S1[output_hc_* Speculative Tensors]:::tensor -.-> S
S --> T[Predict Immediate Token N+1]
S --> U[Parallel Output Speculative Token N+2]
S --> V[Parallel Output Speculative Token N+3]
S --> W[Parallel Output Speculative Token N+4]
end

%% Style assignments
class D,H,K memory;
class G1,J1,S1 tensor;
class E cpu;
class G,J,P,S gpu;
class O,Q sync;
```

## Strategic Key Points

1. **The Zero-Copy Bridge:** The pipeline begins by parsing the 51B N-Gram table out of System RAM dynamically over PCIe, ensuring that the heavy 125B MoE base backbone has max footprint availability in VRAM

2. **The Memory-to-Register Flip:** The GDN Layers cycle data back and forth locally into SM Registers via a recurrent loop, keeping execution away from global memory boundaries until hitting the QSA Layer where a standard (but compressed) KV Cache update occurs

3. **The Token Grouping Pipeline:** In Phase 3, standard token sequencing is broken up by a pointer sorting queue. By sorting token indexing around shared expert requirements, the execution feeds data cleanly into the legacy cublasSgemmBatched execution path while letting the onboard PLX switch hide transfer latencies between Chip 0 and Chip 1
