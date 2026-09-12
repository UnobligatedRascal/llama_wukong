# MoE Coarse-Grained Expert Batching Research

## Problem

Qwen3.8-Flash-Next: 512 experts, each 640 channels. Top-10 routing + 1 shared
expert per token. Only ~6B params active per token.

Standard MoE dispatch:
- For each token, route to top-10 experts
- Each expert processes its assigned tokens independently
- Problem: scattered memory access pattern

Without batching:
- Expert 42 weights loaded -> process token 1 -> unload
- Expert 17 weights loaded -> process token 3 -> unload
- Expert 42 weights loaded -> process token 7 -> unload (reload!)

Wasted bandwidth: same expert weights loaded multiple times.

## Solution: Token Grouping by Expert

Hold tokens briefly, sort by expert index, process in expert order:

```
Token 0: experts [3, 15, 42, 88, 102, 156, 201, 289, 334, 445]
Token 1: experts [3, 7, 42, 91, 120, 178, 233, 289, 355, 401]
Token 2: experts [12, 42, 55, 88, 134, 199, 267, 312, 390, 445]
...

Group by expert:
Expert 3:  [token 0, token 1]
Expert 15: [token 0]
Expert 42: [token 0, token 1, token 2]
Expert 88: [token 0, token 2]
...
```

Process:
1. Load Expert 3 weights (640-channel MLP)
2. Process token 0, token 1 sequentially
3. Unload Expert 3, load Expert 15
4. Process token 0
5. Continue...

## Implementation

### Modified Routing Output
Current topk-moe.cu outputs: per-token expert indices and weights.
Add: sorted expert -> token_indices mapping.

```cpp
struct expert_batch {
    int expert_id;
    int32_t* token_indices;  // which tokens use this expert
    int num_tokens;
    float* routing_weights;  // per-token gate weights
};
```

### Batch Dispatch Kernel
New kernel processes all tokens for one expert in sequence:

```cpp
__global__ void expert_batch_kernel(
    const float* expert_weights,
    const float* token_inputs,
    const int32_t* token_indices,
    int num_tokens,
    float* outputs
) {
    // Expert MLP: gate, up, down projections
    // Process token_indices[0], token_indices[1], ... sequentially
    // All use same expert_weights -> stays in cache/L2
}
```

## Expected Gains

- Fewer weight loads: expert weights loaded once per batch, not per token
- Better cache utilization: expert weights stay hot in L2
- Sequential memory access: weight streaming vs random jumps
- For 512 experts: potential 2-5x bandwidth improvement in MoE layers

## NOUGHT-Specific Notes

- Kepler L2: 1.5MB chip-wide, 48KB shared memory per SM
- Expert size: 640 channels * 640 (hidden) * 4 bytes = ~1.6MB per expert (FP32)
- At quantization: smaller, fits better in cache
- Pipeline: stream expert weights from VRAM -> process batch -> next expert
