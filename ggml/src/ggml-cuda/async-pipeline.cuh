// Async GPU Pipeline for 8x K80 on NOUGHT
// Implements compute/transfer overlap via per-GPU stream pairs
// Exploits K80 PLX switch (PIX connection) for intra-card parallelism
//
// Architecture:
//   - Stream 0: compute (kernel launches, layer processing)
//   - Stream 1: prefetch (cudaMemcpyAsync for next-layer weights/data)
//   - cudaEvent after each layer compute completion
//   - Prefetch stream waits on event before starting transfer
//
// NOUGHT topology (nvidia-smi topo):
//   NUMA0: GPU0-PIX-GPU1, GPU2-PIX-GPU3, PHB to each other
//   NUMA1: GPU4-PIX-GPU5, GPU6-PIX-GPU7, PHB to each other
//   Cross-NUMA: SYS (QPI/UPI) - expensive
//
// Strategy:
//   - Intra-card pipeline: Chip A computes while Chip B pre-fetches via PIX
//   - Inter-card: event barriers at tensor boundaries
//   - NCCL stagger: NUMA0 all-reduce first, then NUMA1 to avoid QPI contention
//
// Integration points:
//   - ggml_backend_cuda_context: add async_pipeline field
//   - ggml_compute_forward_mul_mat: use async streams where applicable
//   - NCCL allreduce: call via staggered groups
//
// NOTE: Uses LAZY INIT to avoid cudaSetDevice() deadlock during backend init.
//       Per-GPU context initializes only on first access for that GPU.

#pragma once

#include "common.cuh"
#include "ggml-cuda.h"
#include <cuda_runtime.h>
#include <vector>
#include <atomic>
#include <mutex>

// Per-GPU async pipeline context
struct ggml_cuda_async_gpu_ctx {
    cudaStream_t prefetch_stream = nullptr;    // Stream 1: async prefetch
    cudaEvent_t layer_complete_events[2];      // Double-buffered completion events
    int current_event = 0;                     // Current event index (ping-pong)
    std::atomic<bool> initialized {false};
    std::mutex init_mutex;
    int device = -1;

    ggml_cuda_async_gpu_ctx() {
        layer_complete_events[0] = nullptr;
        layer_complete_events[1] = nullptr;
    }

    ~ggml_cuda_async_gpu_ctx() {
        if (initialized.load()) {
            if (prefetch_stream) {
                cudaStreamDestroy(prefetch_stream);
            }
            for (int i = 0; i < 2; i++) {
                if (layer_complete_events[i]) {
                    cudaEventDestroy(layer_complete_events[i]);
                }
            }
        }
    }

    // LAZY init: only called once per GPU, on first access
    void ensure_init(int dev_id) {
        bool expected = false;
        if (!initialized.compare_exchange_strong(expected, true)) {
            return; // Already initialized (another thread beat us)
        }
        std::lock_guard<std::mutex> lock(init_mutex);
        // Double-check after lock
        if (initialized.load()) return;
        device = dev_id;
        ggml_cuda_set_device(dev_id);
        CUDA_CHECK(cudaStreamCreateWithFlags(&prefetch_stream, cudaStreamNonBlocking));
        for (int i = 0; i < 2; i++) {
            CUDA_CHECK(cudaEventCreateWithFlags(&layer_complete_events[i],
                                                cudaEventDisableTiming));
        }
        initialized.store(true);
    }

    // Record completion of layer compute on main stream
    // Caller must ensure current CUDA device matches this ctx's device
    // Defensive: validates stream+event before CUDA call to catch race conditions
    void record_layer_complete(cudaStream_t compute_stream) {
        if (!initialized.load()) return;
        cudaEvent_t evt = layer_complete_events[current_event];
        if (!evt || !compute_stream) {
            // Resources not yet created - likely a race with ensure_init
            // Silently skip; caller can retry on next layer
            return;
        }
        CUDA_CHECK(cudaEventRecord(evt, compute_stream));
    }

    // Wait for layer completion on prefetch stream
    // Caller must ensure current CUDA device matches this ctx's device
    void wait_layer_complete_prefetch() {
        if (!initialized.load()) return;
        if (!prefetch_stream || !layer_complete_events[current_event]) {
            // Resources not yet created - skip
            return;
        }
        CUDA_CHECK(cudaStreamWaitEvent(prefetch_stream,
                                        layer_complete_events[current_event], 0));
    }

    // Get current prefetch stream (may be null if not initialized)
    cudaStream_t get_prefetch_stream() const {
        return prefetch_stream;
    }

    // Advance event index for next layer
    void advance_event() {
        current_event = 1 - current_event;
    }

    bool is_initialized() const {
        return initialized.load();
    }

    int get_device() const {
        return device;
    }
};

// Per-layer async prefetch descriptor
struct ggml_cuda_async_prefetch_desc {
    void* dst_ptr = nullptr;          // Destination device pointer
    const void* src_ptr = nullptr;    // Source pointer (device or pinned host)
    size_t size = 0;                  // Bytes to transfer
    cudaMemcpyKind kind = cudaMemcpyDeviceToDevice;
};

// Global async pipeline context (one per compute graph)
struct ggml_cuda_async_pipeline {
    std::vector<ggml_cuda_async_gpu_ctx> gpu_ctx;
    bool enabled = false;
    bool use_staggered_nccl = true;    // Stagger NCCL by NUMA node

    ggml_cuda_async_pipeline() : gpu_ctx(GGML_CUDA_MAX_DEVICES) {}

    // Lightweight enable: NO CUDA calls, just sets a flag.
    // Actual GPU context creation is lazy on first use.
    void enable() {
        enabled = true;
    }

    // Lazy init for a specific GPU (called from compute path)
    void ensure_gpu_init(int dev_id) {
        if (!enabled) return;
        gpu_ctx[dev_id].ensure_init(dev_id);
    }

    // NUMA-aware GPU grouping for staggered NCCL
    // Returns GPU indices for NUMA node 0 and 1
    static void get_numa_gpu_groups(
        std::vector<int>& numa0_gpus,
        std::vector<int>& numa1_gpus) {
        numa0_gpus.clear();
        numa1_gpus.clear();
        // Hardcoded for NOUGHT topology
        // GPUs 0-3 on NUMA0, GPUs 4-7 on NUMA1
        for (int i = 0; i < 4; i++) numa0_gpus.push_back(i);
        for (int i = 4; i < 8; i++) numa1_gpus.push_back(i);
    }

    // Async prefetch: queue transfer on prefetch stream
    static void async_prefetch(ggml_cuda_async_gpu_ctx& ctx,
                               const ggml_cuda_async_prefetch_desc& desc) {
        if (!ctx.is_initialized() || ctx.get_prefetch_stream() == nullptr) {
            // Fallback: use main stream (synchronous behavior)
            return;
        }
        ctx.wait_layer_complete_prefetch();
        CUDA_CHECK(cudaMemcpyAsync(desc.dst_ptr, desc.src_ptr, desc.size,
                                    desc.kind, ctx.get_prefetch_stream()));
        ctx.advance_event();
    }
};

// Global accessor for async pipeline (initialized at backend init)
extern ggml_cuda_async_pipeline ggml_cuda_async_pipeline_global;

// Integration helpers

// Call this after layer compute completes on main stream
// Lazily initializes GPU context on first call.
// CRITICAL: ensures device context is correct before CUDA calls,
// since ensure_gpu_init may have switched devices under lock.
inline void ggml_cuda_async_mark_layer_complete(int device, cudaStream_t compute_stream) {
    if (!ggml_cuda_async_pipeline_global.enabled) return;
    ggml_cuda_async_pipeline_global.ensure_gpu_init(device);
    // ensure_gpu_init may have changed the current device under lock;
    // restore to target device before cudaEventRecord (stream+event must match device)
    ggml_cuda_set_device(device);
    ggml_cuda_async_pipeline_global.gpu_ctx[device]
        .record_layer_complete(compute_stream);
}

// Schedule async prefetch of next tensor
// Lazily initializes GPU context on first call
inline void ggml_cuda_async_schedule_prefetch(
    int device, void* dst, const void* src, size_t size,
    cudaMemcpyKind kind = cudaMemcpyDeviceToDevice) {
    if (!ggml_cuda_async_pipeline_global.enabled) return;
    ggml_cuda_async_pipeline_global.ensure_gpu_init(device);
    // Ensure we're on the correct device for the CUDA calls below
    ggml_cuda_set_device(device);
    ggml_cuda_async_prefetch_desc desc;
    desc.dst_ptr = dst;
    desc.src_ptr = src;
    desc.size = size;
    desc.kind = kind;
    ggml_cuda_async_pipeline::async_prefetch(
        ggml_cuda_async_pipeline_global.gpu_ctx[device], desc);
}
