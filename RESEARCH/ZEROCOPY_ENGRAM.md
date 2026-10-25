# Zero-Copy N-Gram Engram Table Research

## Problem

Qwen3.8-Flash-Next has a 51B parameter N-gram lookup table (engram tensors)
representing bigram/trigram token-space hashes. This is NOT stored in VRAM --
designed for CPU system RAM/SSD via mmap.

Holding 51B params in VRAM would consume ~25GB+ at Q4, eating context capacity.

## Solution: cudaHostAlloc Zero-Copy

Kepler GK210 supports mapped host memory (canMapHostMemory = true):
- cudaHostAlloc allocates page-locked (pinned) host memory
- GPU kernels can read this memory directly over PCIe
- No explicit cudaMemcpy needed -- on-demand streaming

## Implementation

```cpp
// Allocate pinned host memory for engram table
float* engram_table;
size_t table_size = /* 51B params at target quantization */;
cudaHostAlloc(&engram_table, table_size, cudaHostAllocDefault);

// Map from GGUF file via mmap, then copy to pinned memory
// OR: mmap directly into pinned region if OS allows

// In kernel, access via device-accessible pointer:
__global__ void engram_lookup_kernel(
    const float* engram_table,  // points to host memory
    int32_t hash_value,
    float* out_vector
) {
    // GPU reads from engram_table[hash_value] directly
    // PCIe transfer happens automatically (on-demand page fetch)
    int offset = hash_value * vector_dim;
    for (int i = 0; i < vector_dim; i++) {
        out_vector[i] = engram_table[offset + i];
    }
}
```

## Prefetching Strategy

On-demand page faults on Kepler are slow (software coherence only).
Explicit prefetch recommended:

```cpp
cudaMemPrefetchAsync(engram_table + region_start, region_size,
                     gpu_device, stream);
```

Predictive prefetch: look ahead at token sequence, prefetch likely bigram/trigram
hash regions before lookup kernels execute.

## Bandwidth Reality

- PCIe Gen3 x16: ~14 GB/s practical
- K80 VRAM bandwidth: 240 GB/s (17x faster)
- Only hit on bigram/trigram matches, not every token
- Accept latency for massive VRAM savings

## Trade-offs

Pro:
- Frees ~25GB VRAM for active model weights and KV cache
- No explicit data movement code in hot path
- Simple kernel interface (same pointer, different memory domain)

Con:
- PCIe latency visible on first access per page
- Pinned memory reduces available system RAM for OS swapping
- Must manage prefetching to avoid page fault storms

## NOUGHT-Specific Notes

- 128GB system RAM, model weights ~82GB at IQ3_XXS
- Engram table adds ~25GB pinned -> total ~107GB, leaves 21GB headroom
- NVMe backup: mmap from disk, cudaHostAlloc working set only
- Monitor system memory pressure; fallback to explicit copy if needed
