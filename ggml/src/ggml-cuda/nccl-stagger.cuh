// NCCL AllReduce for NOUGHT 8x K80
//
// NOTE: ncclCommSplit is not available in this NCCL version, so we use
// standard NCCL allreduce. NUMA-aware optimization can be added when
// ncclCommSplit becomes available.

#pragma once

#include "common.cuh"
#include <cuda_runtime.h>
#include <vector>

// ============================================================================
// NCCL type trait (must be defined before use)
// ============================================================================

template<typename T> struct ncclType;
template<> struct ncclType<float> { static constexpr ncclDataType_t value = ncclFloat; };
template<> struct ncclType<half> { static constexpr ncclDataType_t value = ncclFloat16; };
template<> struct ncclType<nv_bfloat16> { static constexpr ncclDataType_t value = ncclBfloat16; };


// ============================================================================
// NUMA topology helpers
// ============================================================================

inline int ggml_cuda_nccl_numa_get_node_for_gpu(int gpu_id) {
    if (gpu_id >= 0 && gpu_id <= 3) return 0;
    if (gpu_id >= 4 && gpu_id <= 7) return 1;
    return 0;
}

inline void ggml_cuda_nccl_init_env() {
    // Tune NCCL for K80 PCIe topology before any NCCL calls
    if (!getenv("NCCL_P2P_LEVEL")) {
        setenv("NCCL_P2P_LEVEL", "1", 0);
        GGML_LOG_INFO("NCCL: set NCCL_P2P_LEVEL=1 (PCIe P2P only)\n");
    }
    if (!getenv("NCCL_IB_DISABLE")) {
        setenv("NCCL_IB_DISABLE", "1", 0);
    }
    if (!getenv("NCCL_SOCKET_NTHREADS")) {
        setenv("NCCL_SOCKET_NTHREADS", "2", 0);
    }
    if (!getenv("NCCL_ALGO")) {
        setenv("NCCL_ALGO", "Ring", 0);
    }
    if (!getenv("NCCL_DEBUG")) {
        setenv("NCCL_DEBUG", "VERSION", 0);
    }
}

// ============================================================================
// NUMA-aware communicator management (stub - ncclCommSplit unavailable)
// ============================================================================

struct ggml_cuda_nccl_numa_comms {
    bool valid = false;
};

// ncclCommSplit not available in this NCCL version; NUMA comms not usable.
inline bool ggml_cuda_nccl_init_numa_comms(
        const std::vector<ncclComm_t>& global_comms,
        const std::vector<int>& dev_ids,
        ggml_cuda_nccl_numa_comms& out) {

    GGML_LOG_DEBUG("NCCL: ncclCommSplit unavailable, using standard allreduce\n");
    out.valid = false;
    return false;
}

// ============================================================================
// Standard NCCL allreduce
// ============================================================================

inline bool ggml_cuda_nccl_numa_allreduce(
    const std::vector<ncclComm_t>& global_comms,
    const ggml_cuda_nccl_numa_comms& numa_comms,
    const std::vector<cudaStream_t>& streams,
    const std::vector<void*>& tensors,
    int64_t count,
    ncclRedOp_t op) {

    const size_t n = global_comms.size();
    NCCL_CHECK(ncclGroupStart());
    for (size_t i = 0; i < n; i++) {
        NCCL_CHECK(ncclAllReduce(tensors[i], tensors[i], count,
                                  ncclFloat, op,
                                  global_comms[i], streams[i]));
    }
    NCCL_CHECK(ncclGroupEnd());
    return true;
}
