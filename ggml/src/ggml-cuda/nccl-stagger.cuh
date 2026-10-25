// NCCL Staggered AllReduce for NOUGHT 8x K80
// Reduces QPI/UPI contention by grouping all-reduce ops by NUMA node
//
// Problem: 8-GPU all-reduce simultaneously causes QPI contention when
// GPUs 0-3 (NUMA0) communicate with GPUs 4-7 (NUMA1) via SYS links.
//
// Solution: Stagger NCCL operations by NUMA node:
//   1. NUMA0 GPUs (0-3) complete their all-reduce first
//   2. NUMA1 GPUs (4-7) start their all-reduce after NUMA0 completes
//   3. Cross-node ring/all-gather happens without contention
//
// Also tunes NCCL env vars for K80 topology:
//   - NCCL_P2P_LEVEL=1: PCIe P2P only (PIX/PHB), no SYS
//   - NCCL_IB_DISABLE=1: No InfiniBand (K80s use PCIe only)
//   - NCCL_SOCKET_NTHREADS=2: Per-NUMA socket threads

#pragma once

#include "common.cuh"
#include <cuda_runtime.h>
#include <vector>

// Helper trait for NCCL types (must be defined before use)
template<typename T> struct ncclType;
template<> struct ncclType<float> { static constexpr ncclDataType_t value = ncclFloat; };
template<> struct ncclType<half> { static constexpr ncclDataType_t value = ncclFloat16; };
template<> struct ncclType<nv_bfloat16> { static constexpr ncclDataType_t value = ncclBfloat16; };

// NUMA-aware NCCL environment initialization
inline void ggml_cuda_nccl_init_env() {
    // Set NCCL env vars for K80 topology if not already set
    // NCCL_P2P_LEVEL: 0=none, 1=PCIe only, 2=all (including SYS)
    // For K80: prefer PCIe P2P (PIX/PHB) over SYS crossing
    if (!getenv("NCCL_P2P_LEVEL")) {
        setenv("NCCL_P2P_LEVEL", "1", 0);
        GGML_LOG_INFO("NCCL: set NCCL_P2P_LEVEL=1 (PCIe P2P only, no SYS crossing)\n");
    }

    // Disable InfiniBand (not available on NOUGHT)
    if (!getenv("NCCL_IB_DISABLE")) {
        setenv("NCCL_IB_DISABLE", "1", 0);
    }

    // Per-NUMA socket threads for better CPU binding
    if (!getenv("NCCL_SOCKET_NTHREADS")) {
        setenv("NCCL_SOCKET_NTHREADS", "2", 0);
    }

    // Ring algorithm preferred for PCIe-only topology
    if (!getenv("NCCL_ALGO")) {
        setenv("NCCL_ALGO", "Ring", 0);
    }
}

// Staggered AllReduce: execute NCCL all-reduce in NUMA node groups
// This avoids QPI contention when all 8 GPUs communicate simultaneously
template<typename T>
bool ggml_cuda_nccl_staggered_allreduce(
    const std::vector<ncclComm_t>& comms,
    const std::vector<cudaStream_t>& streams,
    T** tensors,
    size_t count,
    ncclRedOp_t op) {

    // Group GPUs by NUMA node
    std::vector<int> numa0_gpus, numa1_gpus;
    for (size_t i = 0; i < comms.size(); i++) {
        if (i < 4) numa0_gpus.push_back(i);
        else numa1_gpus.push_back(i);
    }

    // Phase 1: NUMA0 all-reduce (GPUs 0-3)
    if (!numa0_gpus.empty()) {
        NCCL_CHECK(ncclGroupStart());
        for (size_t j = 0; j < numa0_gpus.size(); j++) {
            int rank = numa0_gpus[j];
            NCCL_CHECK(ncclAllReduce(tensors[rank], tensors[rank], count,
                                      ncclType<T>::value, op,
                                      comms[rank], streams[rank]));
        }
        NCCL_CHECK(ncclGroupEnd());

        // Wait for NUMA0 completion before NUMA1 starts
        for (size_t j = 0; j < numa0_gpus.size(); j++) {
            int rank = numa0_gpus[j];
            CUDA_CHECK(cudaStreamSynchronize(streams[rank]));
        }
    }

    // Phase 2: NUMA1 all-reduce (GPUs 4-7)
    if (!numa1_gpus.empty()) {
        NCCL_CHECK(ncclGroupStart());
        for (size_t j = 0; j < numa1_gpus.size(); j++) {
            int rank = numa1_gpus[j];
            NCCL_CHECK(ncclAllReduce(tensors[rank], tensors[rank], count,
                                      ncclType<T>::value, op,
                                      comms[rank], streams[rank]));
        }
        NCCL_CHECK(ncclGroupEnd());
    }

    return true;
}
