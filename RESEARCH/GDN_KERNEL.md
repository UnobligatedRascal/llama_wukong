# GDN Kernel Research

## Gated DeltaNet (GDN) Overview

GDN is a linear-attention recurrence used in Qwen3.8-Flash-Next for 75%
of layers (3 of every 4 layers). Replaces quadratic KV-cache attention
with fixed-size state matrix.

## Per-Step Math (per head)

Inputs:
- q, k, v: [state_dim] vectors
- beta: scalar (write gate)
- g: scalar (decay gate)
- S_prev: [state_dim x state_dim] recurrent state matrix

Steps:
1. Normalize: q_hat = L2norm(q) / sqrt(state_dim), k_hat = L2norm(k)
2. Gate: beta_s = sigmoid(beta), gate = exp(g)
3. Decay: S_decay = gate * S_prev (element-wise multiply)
4. Read: kv_mem = S_decay^T * k_hat (matrix-vector multiply)
5. Delta: delta = beta_s * (v - kv_mem) (correction vector)
6. Write: S_new = S_decay + outer(k_hat, delta) (rank-1 update)
7. Output: out = S_new^T * q_hat (readout)

S_new becomes S_prev for next token. State size is constant regardless of
context length (vs KV cache which grows).

## GPU Optimization Targets

### GK210 Register File Advantage
- 512KB registers per SM (doubled vs GK110)
- State matrix S can fit in registers for small state_dim
- Avoid global memory reads for S during recurrence

Example: state_dim = 128 -> S = 128*128*4 bytes = 64KB
- Fits in shared memory per block
- Small enough to keep hot in L1/L2

### Shared Memory Strategy
- Pin S in __shared__ memory per thread block
- Each decode step: decay in-place, update in-place
- Only global memory access: read q/k/v, write output

### Register Usage for q/k/v
- Normalize q_hat/k_hat once, keep in registers
- Used in steps 4, 5, 6, 7 -> multiple reuse without reload
- Target: < 63 registers/thread for 100% occupancy

### Kernel Fusion
Fuse all 7 steps into single kernel launch:
- Avoid launching separate kernels for decay/read/write
- Reduces kernel launch overhead on Kepler (no fast launch)
- Keeps intermediate results in registers/shared memory

## Current State in llama.cpp

File: ggml/src/ggml-cuda/gated_delta_net.cu
- CPU has AVX2/AVX-512 implementations (c-kernel-engine reference)
- CUDA kernel exists upstream; needs sm_37 verification
- Check for modern ISA features (async copy, WMMA) -- none should be used

## Verification Checklist

- [ ] No BF16/TF32 instructions (Kepler has no support)
- [ ] Shared memory usage < 48KB per block (Kepler hardware limit)
- [ ] Register usage allows adequate occupancy
- [ ] Output matches CPU reference (bit-exact FP32)

## References
- Gated DeltaNet Deep Dive: https://c-kernel-engine.github.io/C-Kernel-Engine/deltanet-deep-dive.html
- Linear Attention & GDN: https://carlyou.github.io/all-attentions-you-need/posts/08-linear-attention/
- From KV Cache to State Matrix: https://winterrykim.github.io/blog/2026/linear-attention-and-deltanet/
