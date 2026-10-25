// NUMA-Aware GPU Thread Binding for NOUGHT
// Pins compute threads to NUMA-local cores matching their GPU assignment
//
// NOUGHT topology:
//   NUMA0: CPU cores 0-17,36-53 | GPUs 0,1,2,3
//   NUMA1: CPU cores 18-35,54-71 | GPUs 4,5,6,7
//
// Strategy:
//   - GPU compute threads for GPUs 0-3 bound to NUMA0 cores
//   - GPU compute threads for GPUs 4-7 bound to NUMA1 cores
//   - Host-side memcpy threads pinned to source/dest NUMA node
//
// Usage:
//   - Call ggml_cuda_numa_pin_thread(gpu_id, thread_id) at thread init
//   - Use numactl --cpunodebind for initial process binding, then pin individual threads

#pragma once

#include "common.cuh"
#include <sched.h>
#include <cstring>
#include <vector>

// Get NUMA node for a given GPU (hardcoded for NOUGHT topology)
inline int ggml_cuda_numa_get_node_for_gpu(int gpu_id) {
    if (gpu_id >= 0 && gpu_id <= 3) return 0;
    if (gpu_id >= 4 && gpu_id <= 7) return 1;
    return 0; // fallback
}

// Get NUMA node's core list
inline void ggml_cuda_numa_get_cores(int numa_node, std::vector<int>& cores) {
    cores.clear();
    if (numa_node == 0) {
        // NUMA0: cores 0-17, 36-53 (36 physical cores)
        for (int i = 0; i < 18; i++) cores.push_back(i);
        for (int i = 36; i < 54; i++) cores.push_back(i);
    } else {
        // NUMA1: cores 18-35, 54-71 (36 physical cores)
        for (int i = 18; i < 36; i++) cores.push_back(i);
        for (int i = 54; i < 72; i++) cores.push_back(i);
    }
}

// Pin thread to specific core
inline void ggml_cuda_numa_pin_to_core(int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    if (sched_setaffinity(0, sizeof(cpuset), &cpuset) != 0) {
        GGML_LOG_WARN("NUMA: failed to pin thread to core %d: %s\n",
                      core_id, strerror(errno));
    }
}

// Pin thread to NUMA-local core for a given GPU
// thread_local_id: thread index within the GPU's thread pool
inline void ggml_cuda_numa_pin_thread_for_gpu(int gpu_id, int thread_local_id) {
    int numa_node = ggml_cuda_numa_get_node_for_gpu(gpu_id);
    std::vector<int> cores;
    ggml_cuda_numa_get_cores(numa_node, cores);
    if (cores.empty()) return;
    int core = cores[thread_local_id % cores.size()];
    ggml_cuda_numa_pin_to_core(core);
    GGML_LOG_DEBUG("NUMA: pinned GPU%d thread%d to core%d (NUMA node %d)\n",
                   gpu_id, thread_local_id, core, numa_node);
}

// Create a CPU affinity mask for a NUMA node
inline cpu_set_t* ggml_cuda_numa_create_mask(int numa_node) {
    cpu_set_t* mask = new cpu_set_t();
    CPU_ZERO(mask);
    std::vector<int> cores;
    ggml_cuda_numa_get_cores(numa_node, cores);
    for (int core : cores) {
        CPU_SET(core, mask);
    }
    return mask;
}
