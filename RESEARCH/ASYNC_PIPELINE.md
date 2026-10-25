# Async Multi-GPU Pipeline Research

## Problem

8 GK210 cores across 4 K80 cards connected via PCIe Gen3 x16.
No NVLink, no P2P between cards. Tensor-split mode sends activations
across PCIe between GPUs -- latency visible in token generation.

Each K80 card: 2 independent GK210 chips sharing one PCIe slot via PLX switch.
Chip-to-chip on same card goes through PLX (PCIe-like latency).

## Solution: Compute/Transfer Overlap

CUDA streams + events enable hiding PCIe latency:
- Stream A: compute kernel (layer N processing)
- Stream B: prefetch transfer (layer N+1 weights via cudaMemcpyAsync)
- Event E: marks end of compute; Stream B waits on E

Timeline without pipeline:
```
GPU0: [compute layer 0] -> [wait PCIe transfer] -> [compute layer 1]
```

Timeline with pipeline:
```
GPU0 stream A: [compute layer 0] ---------> [compute layer 1]
GPU0 stream B:         [prefetch layer 1] ---> [done]
                       ^ PCIe transfer overlaps compute
```

## K80 Dual-Core Pipeline

Each K80 card has 2 GK210 chips. Pipeline within a card:

```
Card 0:
  Chip 0 stream compute: [compute token N, layers 0-12] -> event
  Chip 1 stream prefetch:          [prefetch token N+1] waits on event

Card 1:
  Chip 2 stream compute: [compute token N, layers 13-24] waits on card0 event
  Chip 3 stream prefetch:          [prefetch token N+1] waits on compute event
```

## Key CUDA APIs

```cpp
// Per-GPU context
cudaStream_t compute_stream, prefetch_stream;
cudaEvent_t layer_complete;

cudaStreamCreate(&compute_stream);
cudaStreamCreate(&prefetch_stream);
cudaEventCreate(&layer_complete);

// In layer loop:
launch_kernel(compute_stream, layer_weights, input);
cudaEventRecord(layer_complete, compute_stream);

// Prefetch next layer's weights:
cudaStreamWaitEvent(prefetch_stream, layer_complete);
cudaMemcpyAsync(next_layer_weights, host_ptr, size,
                cudaMemcpyHostToDevice, prefetch_stream);
```

## PCIe Bandwidth Reality

- PCIe Gen3 x16: 16 GB/s theoretical, ~14 GB/s practical
- K80 per-chip VRAM bandwidth: 240 GB/s
- Ratio: PCIe is 17x slower than VRAM
- Goal: keep PCIe transfers hidden behind compute, never visible

## Memory Considerations

cudaMemcpyAsync requires pinned (page-locked) host memory:
- cudaHostAlloc for staging buffers
- Limited system RAM (128GB), manage pinned allocations carefully

## References
- NVIDIA CUDA Programming Guide: Section 2.5 Asynchronous Execution
- NVIDIA CUDA Programming Guide: Section 3.4 Multi-GPU Systems

## NOUGHT-Specific Notes

- 8 independent GPUs, no NVLink, PCIe Gen3 topology
- GGML_CUDA_P2P=1 enables peer access where available (within K80 cards)
- Tensor-split distributes layers; async pipeline hides inter-GPU sync
- Layer-split mode preferred over row-split for Kepler (less sync)
