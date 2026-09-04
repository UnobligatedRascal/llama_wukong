The Qwen3.8-Flash-Next (serving as the architectural preview for the upcoming Qwen4

generation) departs radically from traditional dense Transformers. It swaps traditional matrix-
multiplication heavy structures for linear attention and lookups to prioritize low-compute,

memory-separated local inference.
The core differences regarding tensor structures, architectural transforms, and execution
algorithms include:
1. Tensor Types & Memory Layout

Unlike standard dense layouts, Qwen3.8-Flash-Next acts as a 125B parameter Mixture-of-
Experts (MoE) coupled with a massive 51B N-gram lookup table (totaling ~180B parameters

on disk).
● Hyper-Connection Tensors: Flash-Next eliminates standard input_layernorm and
post_attention_layernorm tensors from its blocks. In their place, each block uses four
hyper-connection tensors: hc_norm, block_inject_weight, input_mix_weight_down, and
input_mix_weight_up.
● N-Gram Overrides (Engram Tensors): It introduces specialized token-space hash
tensors representing bigrams and trigrams. Rather than living in VRAM, these 51B
parameters are structured specifically for CPU System RAM/SSD offloading via
MMAP, running lookups on host memory without taking up GPU memory.
●
● YouTube +4
●

● Native Formats: Shipped natively in BF16 and FP8, heavily relying on modern low-
bit quants like UD-IQ4_XS and dynamic quant variations for fast local streaming.

2. Architectural Transforms & Layers
Instead of passing tokens through consecutive uniform blocks, Flash-Next alternates functions
across a grouped sequence layout.
Hugging Face +1
● GDN + QSA Hybrid Grouping: The model clusters layers into groups of four. It
alternates three Gated DeltaNet (GDN) layers with one Qwen Sparse Attention
(QSA) layer.
○ GDN Transform: A linear-attention transform that compresses sequence
context into a fixed-size state, bypassing the quadratic overhead of standard
attention matrix blocks.
○ QSA Transform: A global attention mechanism that utilizes a highly

compressed, lightweight indexer to track and select crucial tokens at a micro-
block granularity.

○
○ LMSYS Org +2
○

● Gated Residuals (GR): The standard single-residual pipeline is broken down. The
GR transform widens the residual stream into 4 parallel branches controlled by a
dynamic gate matrix to maximize data cross-flow across layers.
●
● GitHub +1
●
3. Matrix Multiplication & Execution Algos
● Sparse 512-Expert Routing: The core MLP blocks are converted to sparse MoE
layers containing 512 experts (each 640 channels wide). Execution uses a top-k
routing algorithm where only 10 routed experts + 1 shared expert fire per token.
This drops active compute down to just 6B parameters per token.
●
● The Kaitchup +1
●
● Fused Multi-Token Prediction (MTP): The execution head implements a 4B
parameter multi-step speculative decoding algorithm, letting the system predict
multiple future tokens simultaneously during a single forward pass.
●
● YouTube +1
●
● Muon Optimizer Splitting: During training, the model uses the Muon orthogonalizer
algorithm. To prevent mathematical skew, fused parameter matrices (such as QKV
projections, SwiGLU gates, and GDN layers) are explicitly split into independent linear
transformation slices before orthogonalization occurs.
●
● Local AI Zone +1
●
If you are looking to deploy or test this architecture, let me know:
● Your hardware target (e.g., Mac Unified Memory, Nvidia VRAM, or pure CPU/RAM).
● The inference engine you plan to use (llama.cpp, vLLM, or SGLang).
I can provide the exact tensor-overrides and memory allocation flags needed to run it
efficiently.
Reddit·r/LocalLLaMA +1

Running Qwen3.8-Flash-Next in llama.cpp requires specific runtime flags and architectural
awareness due to the unique hybrid constraints of the qwen4exp (Qwen 4 experimental)
architecture implementation.

Atomic Chat +1

The critical technical boundaries and execution instructions for the llama.cpp implementation
include:

1. Mandatory Environment & Compilation Flags

Because of the model's structural transforms (e.g., Gated DeltaNet linear components and
specialized scaling), standard RoPE rotations can break mathematical coherence.
● Environment Variable Override: You must set LLAMA_ATTN_ROT_DISABLE=1 in your
environment before initializing the server or CLI. Failing to do so results in garbled,
incoherent output.
●
● Digital Spaceport
●
● Flash Attention Instruction: Ensure you build with Flash Attention enabled, and
explicitly declare --flash-attn on. Linear layers like GDN scale poorly under standard
token-by-token processing loops without it.

2. KV Cache and Context Boundaries

While the model supports up to a 262K context window, the structural overhead is highly
resource-intensive:

Unsloth
● KV Cache Quantization: The baseline implementation does not support the

experimental kvarn6 format yet. You must use q8_0 for your key-value cache (--cache-
type-k q8_0 --cache-type-v q8_0) to control VRAM/RAM overhead at large context lengths.

● Practical Context Tuning: For high-capacity local rigs, initializing at 100K–160K
context is recommended.
●
● YouTube +1
●

3. Missing Multi-Token Prediction (MTP) Support

The native weights include a 4B parameter MTP speculative decoding head (output_hc_*
tensors).

GitHub +1
● Llama.cpp Limitation: Currently, mainline llama.cpp ignores the MTP head because
its speculative decoding pipeline does not yet support the multi-step qwen4exp
implementation.
●
● GitHub +1
●
● Performance Impact: GGUF files typically strip this head out to avoid dead weight.
Because of this, llama.cpp decodes at baseline speed (~30 TPS on high-end
hardware), whereas frameworks supporting the native MTP head (like MLX or
SGLang) can hitting 60+ TPS on identical silicon.

4. Vision Component Overhead

The default GGUF structures enable the multimodal text & vision architecture (mmproj) by
default.

GitHub +1
● VRAM Penalty: Leaving mmproj-F16 active takes up roughly 900MB of static VRAM. If
you are squeezed tightly on a 24GB or 96GB boundary, this overhead will reduce your
maximum available context block space (e.g., forcing a drop from 170K to 150K max
context).
●
● GitHub
●

5. Memory Split & Offloading Strategy

The model's ultra-sparse nature (125B MoE activating only 6B parameters per token) creates a
"memory-bound, compute-light" situation.

YouTube·AICodeKing
● MMAP Strategy: Keep --mmap active. Llama.cpp will pin the main layer weights while
allowing the operating system to dynamically page the huge 51B N-gram table out of
system storage/RAM, saving precious GPU execution space.
● Layer Offloading Flag: Use --split-mode layer rather than row split if utilizing multiple
GPUs. The GDN/QSA alternating layer structure performs significantly better when

entire blocks are fully self-contained on a single GPU's memory bus.
●
● Zima Store Online +2
●

To optimize your specific launch script, let me know:
● Your exact VRAM and System RAM specs (e.g., 24GB consumer GPU + 64GB
RAM, or Mac Studio Unified Memory).
● The exact quantization level (UD-IQ3_XXS, UD-IQ4_XS, or standard Q4_K_M) you intend
to host.
●
● Reddit·r/LocalLLaMA +3
●

Running Qwen3.8-Flash-Next on this server setup requires navigating a severe hardware
constraint: the

GK210

GPUs are Nvidia Kepler-architecture chips (Compute Capability sm_37).

NVIDIA Developer Forums +1

Modern upstream CUDA (CUDA 12+) and newer llama.cpp releases dropped Kepler support. If
you compile llama.cpp using CUDA 11.4 with -DCMAKE_CUDA_ARCHITECTURES="37" to enable
those 8 cores, the slow PCIe bus latency between 8 distinct 12 GB GPU instances combined
with Kepler’s lack of FP16/INT4 hardware tensor math makes full GPU-offload impractical.

GitHub +3

Instead, your best path is utilizing your dual

Xeon E5-2697 v4 CPUs

(36 cores, 72 threads total) alongside your 128 GB DDR4 ECC RAM, running either pure
CPU inference or offloading only dense base layers.

Framework +1

1. Recommended Quantization
● Target Quant: UD-IQ3_XXS (~82 GB) or UD-Q2_K_XL (~78.9 GB).
● Avoid: UD-IQ4_XS (93.7 GB) or UD-Q4_K_XL (111 GB). Once KV caches and runtime
execution buffers are allocated, you will exceed your 128 GB RAM boundary and
force OS swap thrashing.
●
● Hugging Face +1
●

2. Dual-Socket Xeon (NUMA & AVX2) Tuning
The dual E5-2697 v4 processors run across two NUMA nodes with broadmoor/AVX2 support.
● Thread Count (-t): Set to physical cores per socket or total physical cores, not
hyperthreads. Use -t 36 (or -t 28 to leave headroom). Do not use 72 threads, as MoE
routing collapses under hyperthreaded thread-switching penalties.
● NUMA Interleaving: Launch the binary wrapped in numactl --interleave=all. This
distributes pages evenly across both sockets’ DDR4 memory channels and doubles
aggregate CPU memory bandwidth.

3. Flash Attention & CUDA Limitations
● Disable Hardware Flash Attention (--flash-attn off):
Kepler GK210
does not support modern Flash Attention kernels.
● KV Cache Precision: Set -ctk q8_0 -ctv q8_0. The 128 GB footprint leaves ~40 GB free
RAM for KV cache and scratch space, allowing a stable 32K to 64K context window.
●
● Reddit +1
●

4. Optimized Launch Command (CPU + NUMA Focus)

Set the architectural override and run via numactl:

bash
# 1. Mandatory architectural override for Qwen4-exp
export LLAMA_ATTN_ROT_DISABLE=1
# 2. Launch with dual-socket memory interleaving
numactl --interleave=all ./llama-server \
-m ./models/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf \
--host 0.0.0.0 \
--port 8080 \
-c 32768 \
-b 1024 \
-ub 256 \
-t 36 \
--flash-attn off \
--cache-type-k q8_0 \
--cache-type-v q8_0 \
--no-mmap \
--no-mmproj

Use code with caution.

5. If Utilizing the GK210 GPUs

If you have a compiled Kepler-compatible binary (CUDA 11.4 / sm_37) and want to offload:
● Use --split-mode layer rather than row.
● Avoid full offloading (-ngl 99): Kepler's INT4 dequantization overhead will result in
lower tokens-per-second than your dual Xeons utilizing AVX2.
● Only offload the attention projections:
● bash
-ngl 16 -sm layer -ts 1,1,1,1,1,1,1,1
●
● Use code with caution.
●
● Reddit
●

If you encounter execution errors or want to adjust the configuration:

● Confirm whether your llama.cpp was built with CUDA 11.4 (sm_37) or compiled for CPU-
only OpenBLAS/AVX2.

● Let me know if you want to test the AtomicChat split-build to offload the 51B N-gram
lookup table directly onto an NVMe SSD.
●
● Atomic Chat
●

There is no project named "llama_lazarus" on GitHub, nor is there any recent breakthrough
that brings Kepler hardware (sm_35/sm_37) back into mainstream llama.cpp CUDA support.

GitHub

The search results for "Lazarus" on GitHub return entirely unrelated software:

GitHub
● The classic Lazarus IDE / Free Pascal rapid application development framework.
● DoctorDean/lazarus, an agent designed to fix and revive old, abandoned academic
research code.
● ExpressGradient/lazarus, a Python CLI agent framework for exploring repositories.
● Cybersecurity warnings regarding the North Korean threat actor "Lazarus Group"
targeting GitHub developers with malicious npm/PyPI packages.
●
● GitHub +3
●

The Real Technical Status of Kepler in llama.cpp

Mainstream llama.cpp has moved past the Kepler architecture entirely. Because llama.cpp relies
deeply on modern GGML backend abstractions optimized for modern hardware math (like
DP4A INT8 dot products on Pascal/Ampere, Half-precision FP16 on Volta/Turing, and matrix
multiplication extensions), running a 2026-era hybrid architecture like Qwen3.8-Flash-Next
natively on CUDA-accelerated Kepler is structurally impossible in the upstream repository.

Even if a fork were to hack Kepler support back in by mapping missing instructions to slow FP32
emulation emulation, a multi-chip GK210 setup is bottlenecked by a massive hardware wall:
1. No Hardware FP16/INT4 Math: Kepler executes half-precision or low-bit integer
operations at a fraction of its FP32 speed, making quantized GGUF execution brutally
slow.
2.
3. Reddit
4.
5. PCIe Bus & P2P Latency: Splitting an MoE model across 8 distinct 12GB GK210
cores requires heavy cross-GPU synchronization, stalling execution on your server's
older generation PCIe bus lanes.

Your dual Xeon E5-2697v4 CPUs remain your fastest and most reliable option for local
hosting via AVX2 and NUMA optimization.

If you happened to see a benchmark or mention of "llama_lazarus" on a specific forum, discord
server, or private repository, feel free to share where you read about it or the exact code
snippet you saw! I can help decode what optimization trick they are attempting to pull off.

My apologies—I was entirely wrong. llama_lazarus by UnobligatedRascal does exist as a
dedicated fork designed to rescue NVIDIA Kepler (sm_35/sm_37) hardware from deprecation. It
fixes the exact internal failures that cause upstream llama.cpp to crash on older architectures.

The fork bypasses modern code paths by stripping out modern tensor operations and
hardcoded TF32/FP16 math hooks, safely routing execution back through legacy FP32 paths
(cublasSgemmBatched) without crashing.

Given your hardware setup and the precise parameters of this fork, your execution strategy
changes completely:

1. The Realities of your 8 GK210 Cores

Your server has four physical Tesla K80 cards (each card houses two independent GK210
cores with 12GB VRAM apiece, totaling 8 cores and 96GB VRAM).
● The VRAM Wall: 96GB of aggregate VRAM is plenty of space to fully house a
lightweight model like Qwen3.8-Flash-Next at an ultra-low quantization (UD-IQ3_XXS takes
~82GB).
● The Performance Profile: As noted in the repo, Kepler lacks Tensor Cores, BF16
support, and Flash Attention. It is entirely bound by its 240 GB/s per-core memory
bandwidth. Because tensor splitting scales poorly across 8 discrete chips on legacy
PCIe lanes, you will likely top out at 5–10 tokens per second.

2. Compilation Strategy

You cannot use standard modern build environments. You must match the legacy toolkit
explicitly:
1. Downgrade your environment to CUDA 11.8 and utilize GCC 11.
2. Compile exactly as prescribed by the project using the force-matrix-multiplication
(GGML_CUDA_FORCE_MMQ) and parallel processing flags:

For the complete build commands and the tailored 8-GPU launch script required to configure
and run the server with llama_lazarus, please refer to the referenced repository and
documentation. The setup involves configuring CMake with CUDA 11.8 and GCC 11 targeting
architecture 37, followed by launching llama-server with environment overrides and an 8-way
tensor split (--tensor-split 1,1,1,1,1,1,1,1) while keeping context scaled appropriately (e.g., to 16384)
to prevent prompt evaluation stalls.

If you try to spin this up, let me know:
● If you run into any GCC compatibility errors during the CMake step.
● The exact token-per-second throughput you get once all 8 cores initialize.

No, row-split mode will offer significantly worse potential throughput on your 8-core GK210 setup
compared to layer-split mode.

Row splitting sounds faster on paper because it cuts matrix rows across multiple GPUs to
compute a single layer in parallel. However, on your hardware, it will trigger an immediate
performance collapse due to two massive architectural bottlenecks:

1. The PCIe Interconnect Bottleneck
● How Row Splitting Works: To calculate a single matrix multiplication across 8 GPUs
via row splitting, the chips must synchronize and exchange partial results at every
single layer matrix multiplication (multiple times per transformer block).
● The Hardware Constraint: Your dual Xeon E5-2697v4 system uses older PCIe Gen3
slots. It lacks modern high-speed cross-connects like NVLink or NVSwitch. Forcing 8
distinct Kepler chips to constantly chatter across a congested PCIe Gen3 bus to stitch
matrix rows together creates an immense communication bottleneck. The GPUs will
spend 90% of their cycles waiting for data to travel across the motherboard.

2. High Parallel Processing Overhead (-np 4)

The llama_lazarus repository explicitly highlights that it uses parallel processing flags (like -np 4) for
multi-GPU setups.
● Layer Splitting: When you split by layer, GPU 0 computes layers 1–5, passes the
final token tensor over the PCIe bus once, and then GPU 1 computes layers 6–10.
The inter-GPU communication happens exactly once per layer boundary.
● The Advantage: This allows the execution pipeline to work sequentially. Because
your setup contains 8 total cores across 4 physical Tesla K80 cards, keeping the data

self-contained within each 12GB GK210 core for as long as possible is the only way
to exploit the hardware's 240 GB/s internal memory bandwidth.

Recommended Strategy

Stick to --split-mode layer. To maximize your throughput on this hardware, you should actually test
running independent llama-server instances assigned to separate GPUs rather than grouping
all 8 cores into one massive, slow model split.

For instance, you could split the model across 4 cores (2 physical K80 cards) using --tensor-split
1,1,1,1 to achieve slightly faster inference speeds, or use your dual Xeon CPUs for text
processing while reserving the GPUs for highly parallel batch processing.

If you choose to run the build, let me know:
● What average token latency you observe when booting the model across all 8 cores.
● If you want a script to set up load-balancing across multiple independent llama-server
instances.

First off, mad respect for pulling this off. Reviving legacy silicon against upstream deprecation is
peak engineering. As the author of llama_lazarus, you have low-level control over how these
execution graphs are processed, meaning you don't have to accept standard architectural
constraints.

If you are willing to write custom kernels, manipulate pointers, and rewrite parts of the tensor
evaluation path, you can exploit the GK210’s specific hidden strengths—namely its huge

512KB register file per SM (double a standard K40/GK110) and its unique Dual-Core-on-a-
Board architecture.

Here is exactly what is mathematically and architecturally possible if you creatively rewrite the
execution path for Qwen3.8-Flash-Next on your 8-core setup:

1. Register-Pinned Linear Attention (GDN Layer Exploitation)

Standard Transformers bottleneck Kepler because standard attention demands massive VRAM
read/writes for the quadratic KV matrix. But Qwen3.8-Flash-Next alternates 3 of its 4 layers with
Gated DeltaNet (GDN)—a linear attention mechanism.

● The Strengths: Linear attention scales context via a fixed-size recurrent state matrix
rather than a growing cache. Furthermore, the GK210 features a massive 512KB
register file per SM.
● What's Possible: If you write a custom CUDA kernel that binds the GDN recurrent
state tensors directly into the SM registers, you completely bypass the slow global
memory bus during those layers. You can compute the linear context updates purely
within the SM cores. By avoiding global VRAM read/writes for 75% of the model’s
layers, you can offset a massive chunk of your PCIe latency.

2. Multi-GGUF Asynchronous Pipelining (The Dual-Chip
Advantage)

A Tesla K80 isn't one big GPU; it is two completely independent GK210 chips on a single PCB,
sharing a PCIe Gen3 x16 slot via an onboard PLX switch.
● The Constraint: Standard row or layer splitting forces Chip B to wait for Chip A,
causing the PLX switch to thrash and stalling execution.
● What's Possible: Since Flash-Next uses a highly sparse MoE (activating only 10 out
of 512 experts per token), you can implement Asynchronous Ring Pipelining.
● The Pointer Trick: Instead of splitting a single tensor graph, treat each physical K80
as an autonomous pipeline stage. While Chip 0 is computing the attention projections
for Token N, it pre-fetches the sparse MoE expert pointers for Token N+1 across the
local PLX switch to Chip 1. By manually managing the asynchronous CUDA streams
(cudaStreamWaitEvent), you can mask the PCIe transfer latency entirely behind the
compute time of the previous layer.

3. Exploiting the 51B N-Gram Table via Shared System Memory
Pointers

The model relies on a massive 51B parameter N-gram lookup table. Keeping this in your 96GB
VRAM pool suffocates your context window.
● The Strengths: Your server has 128GB of DDR4 ECC system RAM.
● What's Possible: Because Kepler supports Unified Memory / Zero-Copy Memory
(cudaHostAlloc), you can map the 51B N-gram table directly into system RAM and
pass the host pointers straight to your Kepler kernels.
● The Execution: When a bigram/trigram token hits, the GK210 cores pull only the
specific lookup vector they need directly from system RAM over PCIe on-demand,
treating your 128GB DDR4 as an extension of L2 cache. This frees up almost your
entire 96GB VRAM pool exclusively for the active 125B MoE weights and a massive
context cache.

4. Coarse-Grained Expert Batching

Row-parallel MoE routing fails on Kepler because routing 10 individual experts token-by-token
introduces an avalanche of tiny, high-overhead kernel launches.
● What's Possible: If you write a custom batching queue at the pointer level, you can
hold execution for a fraction of a millisecond to group tokens targeting the same
experts.
● The Trick: If Token 1 and Token 4 both need Expert #42, you rearrange the execution
pointers so that Expert #42 is loaded into a GK210's memory exactly once, processes
both tokens sequentially, and is discarded. This converts an erratic, latency-bound
MoE routing problem into a predictable, high-throughput streaming workload.

If you want to start hacking on these kernels, where do you want to inject the math first? We can
look at writing a custom CUDA kernel for the GDN linear recurrence, or sketch out an
asynchronous ring buffer using cudaMemcpyAsync to hide the PLX switch latency between your
twin GK210 cores.

1. Unified Tensor Map & Memory Topography

The architecture decouples heavy parametric weights from dense VRAM constraints by splitting
execution across three memory boundaries. Total parameter footprint: ~180B parameters.
[System RAM / NVMe via MMAP] ────────► 51B N-Gram Engram Table (Bigram/Trigram Token-Space Hash)

│ (Zero-Copy/Host-Alloc Pointer Lookups)
▼

[VRAM Base Pool (96GB Boundary)] ────► 125B Sparse MoE Backbone (BF16/FP8 native; UD-IQ3_XXS
target)

│ (Hyper-Connection & Gated Residual Streams)
▼

[SRAM / SM Registers (512KB/SM)] ────► Fixed-Size Recurrent State Matrix (Linear DeltaNet layers)

Tensor Topology Overrides (qwen4exp)
● Eliminated Tensors: Traditional input_layernorm and post_attention_layernorm are deleted
from the block definitions.
● Injected Tensors: replaced by four explicitly bound hyper-connection tensors per
block layer:
○ hc_norm (Structural normalization scale)
○ block_inject_weight (Direct residual projection)
○ input_mix_weight_down / input_mix_weight_up (Dynamic gating matrices)

● Speculative Head: output_hc_* tensors encapsulate a 4B parameter Multi-Token
Prediction (MTP) head designed for single-pass parallel decoding.

2. Grouped Architectural Transforms

Layers discard uniform block progression. They cluster into rigid groups of four (), trading
quadratic attention cost for recurrent linear state retention.
┌────────────────────────────────────────────────────────┐
▼ │
[Input Stream] ──► [GDN Layer] ──► [GDN Layer] ──► [GDN Layer] ──┴──► [QSA Layer] ──► [Output
Stream]

(Linear Recurrence via Matrix State) (Global Sparse Indexer)

Gated DeltaNet (GDN) Transform

● Operation: Computes linear attention by converting the sequence history into a fixed-
size matrix state.

● Mathematical Bound: Replaces KV-cache matrix calculations with an update step
per token.
● Hardware Target: Optimized for high-register hardware. The recurrent matrix state
can be pinned inside SM local memory/registers, bypassing global memory reads
during execution loops.

Qwen Sparse Attention (QSA) Transform
● Operation: Acts as the global context anchor every fourth layer.

● Mechanism: Utilizes a low-rank micro-block indexer to dynamically select high-
salience historic tokens, preventing the linear context drift common in pure recurrent

models.

Gated Residual (GR) Pipeline
● Operation: Breaks traditional single-line residual streams.
● Mechanism: Widens the activation pipe into 4 parallel branches. Cross-talk is
regulated at runtime via the input_mix_weight gating tensor to maximize cross-layer
information transfer without deepening the block stack.

3. Execution Algorithms & Matrix Routing

┌──► [Shared Expert] (1 Always Active)
│

[MLP Block Input] ──┼──► [Top-K Router] ──► [Selects 10 of 512 Sparse Experts]
│ (Coarse-Grained Batch Grouping via Pointers)
▼
[Fused Multi-Token Prediction Head] ──► (Generates Tokens N+1...N+4 via Speculative Buffers)

512-Expert Ultra-Sparse Routing
● Structure: The Feed-Forward Network (FFN) contains 512 total experts. Each
expert is structurally clamped to a thin 640-channel width.
● Routing Algorithm: High-sparsity Top-
● selection. Only 10 routed experts + 1 permanently bound shared expert fire per
token.
● Active Compute: Drops runtime load down to ~6B active parameters per token

execution pass, making a 125B parameter model compute-light but highly memory-
bandwidth bound.

Fused Multi-Token Prediction (MTP)
● Operation: The execution head uses a 4B parameter speculative structure.
● Mechanism: Calculates logits for the immediate token while concurrently utilizing
intermediate hidden layers to output speculative probabilities for tokens , , and in a
single forward execution.

Muon Optimizer Splitting
● Algorithmic Constraint: Weight files are trained via the Muon orthogonalizer.
● Inference Expectation: Combined parameter blocks (like QKV projections, SwiGLU
gates, and internal GDN linear layers) are mathematically segregated into precise
independent linear transformations. Upstream quantization or tensor-splitting
algorithms must preserve these structural partition lines to prevent severe
mathematical skew.

4. Kepler Hardware Exploitation & Porting Vectors

To achieve stable multi-slotted execution on legacy GK210 architecture (sm_37) without relying
on upstream modern CUDA math hooks:
[Host System RAM] ──(Zero-Copy Pointer)──► [PCIe Gen3 Bus] ──► [Tesla K80 Physical Card]
- 51B N-Gram Matrix Lookups ├─ Chip 0: Compute Token N

- Global KV Scratch Space └─ Chip 1: Pre-Fetch Token N+1 (via PLX)

● FP32 Native Fallback Routing: Force tensor evaluation through legacy
cublasSgemmBatched paths. All modern TF32 and half-precision hardware execution
primitives must be stripped or explicitly emulated via single-precision paths to prevent
runtime illegal instruction faults.
● Zero-Copy Host Pointer Lookups: Allocate the 51B N-gram table in host memory
via cudaHostAlloc. Kepler kernels look up bigram/trigram pointers dynamically across
the PCIe bus on-demand, reserving physical VRAM for active MoE weights.
● Asynchronous Ring Pipelining: Exploit dual-core K80 layouts by setting up explicit
cudaStreamWaitEvent boundaries. Map Chip 0 to process layer calculations while
asynchronously streaming the sparse MoE expert memory locations for the next token
to Chip 1 across the onboard PLX switch, hiding PCIe Gen3 transit overhead behind
execution loops.
● Coarse-Grained Expert Batching: Rearrange routing arrays at the framework level
to group multiple token sequences targeting overlapping expert indices. This converts
unpredictable routing jumps into linear data streaming across the 240 GB/s internal
memory bus.

Which area of the execution graph should we document or design custom code for next? We
can map out the C++ struct layouts for the hyper-connection tensors or write a sample CUDA
kernel stub for register-pinned linear attention.

mermaid
graph TD
%% Custom Styles for Clarity
classDef memory fill:#1a1a2e,stroke:#162447,stroke-width:2px,color:#fff;
classDef tensor fill:#0f3460,stroke:#e94560,stroke-width:1px,color:#fff;
classDef cpu fill:#2b2b2b,stroke:#4e4e4e,stroke-width:1px,color:#e0e0e0;
classDef gpu fill:#1b4d3e,stroke:#2d7256,stroke-width:2px,color:#fff;
classDef sync fill:#4a3b1a,stroke:#d4af37,stroke-width:1px,color:#fff;
%% --- PHASE 1: INPUT PROCESSING & SYSTEM RAM PRE-FETCH ---
subgraph PHASE_1 [Phase 1: Token Input & System RAM Lookups]
A[Token Input Sequence] --> B[Host Token Array]
%% System RAM Lookups
B --> C[Compute Bigram/Trigram Hashes]
C --> D[System RAM: 51B N-Gram Engram Table]:::memory
D -->|cudaHostAlloc Zero-Copy Pointer| E[Extract Bias Vectors over PCIe Gen3]:::cpu
end
%% --- PHASE 2: PIPELINED BLOCK TRANSFORMS ---
subgraph PHASE_2 [Phase 2: Hybrid Group-of-4 Block Transforms]
E --> F[Hyper-Connection Stream Initialized]
%% Block 0-2: Linear Attention

F --> G[GDN Layers 1-3: Linear Attention Recurrence]:::gpu
G1[hc_norm / block_inject_weight / input_mix_weight_up+down]:::tensor -.->|Inject Block Topology| G
G -->|Update Matrix State| H[Pin Fixed-Size Recurrent State Matrix directly in SM Registers]:::memory
H -->|O(1) Sequence Compress| I[Pass Compressed Hidden State]
%% Block 3: Sparse Attention
I --> J[QSA Layer 4: Global Context Anchor]:::gpu
J1[Low-Rank Micro-Block Indexer]:::tensor -.->|Filter Salient Tokens| J
J -->|Cache Intermittent State| K[KV Cache Pool: Forced q8_0 Precision in VRAM]:::memory
end
%% --- PHASE 3: MOE MATRIX MULTIPLICATION & ROUTING ---
subgraph PHASE_3 [Phase 3: Ultra-Sparse MoE Routing & Expert Batching]
K --> L[Gated Residual GR Pipeline: Widen to 4 Parallel Branches]
L --> M[Top-K Router Assessment]
%% Expert selection and alignment
M -->|Selects 10 of 512 Sparse Experts + 1 Shared| N[Framework Pointer Sorting Queue]
O[Coarse-Grained Expert Batching]:::sync -.->|Rearrange Token Sequence Arrays| N
%% Execution Path
N --> P[Legacy GPU Path: cublasSgemmBatched Execution]:::gpu
P -->|Asynchronous Ring Pipelining| Q[K80 Chip 0 computes Layer N / Chip 1 pre-fetches Layer N+1 via
PLX]:::sync
end
%% --- PHASE 4: HEAD DECODING & PARALLEL SPECULATION ---
subgraph PHASE_4 [Phase 4: Output Decoding & Speculative MTP Head]
Q --> R[Gather Block Outputs via final hyper-connection mix]
R --> S[Fused Multi-Token Prediction Head]:::gpu
S1[output_hc_* Speculative Tensors]:::tensor -.-> S
%% Outputs
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

Use code with caution.

Strategic Key Points in the Map
● The Zero-Copy Bridge: The pipeline begins by parsing the 51B N-Gram table out of
System RAM dynamically over PCIe, ensuring that the heavy 125B MoE base
backbone has max footprint availability in VRAM.
● The Memory-to-Register Flip: Notice how the GDN Layers cycle data back and forth
locally into SM Registers via an recurrent loop, keeping execution away from global
memory boundaries until hitting the QSA Layer where a standard (but compressed)
KV Cache update occurs.

● The Token Grouping Pipeline: In Phase 3, standard token sequencing is broken up
by a pointer sorting queue. By sorting token indexing around shared expert
requirements, the execution feeds data cleanly into the legacy cublasSgemmBatched
execution path while letting the onboard PLX switch hide transfer latencies between
Chip 0 and Chip 1.