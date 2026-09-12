#ifndef GGML_CPU_NUMA_REPLICATE_H
#define GGML_CPU_NUMA_REPLICATE_H

#ifdef __cplusplus
extern "C" {
#endif

void ggml_numa_replicate_init(void);
int ggml_numa_replicate_get_n_nodes(void);
int ggml_numa_replicate_is_enabled(void);

/* Replicate a weight buffer's data to per-node copies.
 * Called after tensor allocation is complete. */
void ggml_numa_replicate_weights(void *base, size_t size);

/* Get the local node's pointer for a global weight address.
 * Kernels MUST use this for weight tensor accesses to get NUMA-local data. */
void *ggml_numa_replicate_get_local_ptr(void *global_ptr);

/* Trigger replication for a buffer. Returns NULL (replication via offset mapping). */
ggml_backend_buffer_t ggml_backend_cpu_numa_buffer_wrap(ggml_backend_buffer_t inner);

void ggml_numa_replicate_stats(void);

/*
 * Inline helper: get NUMA-local pointer for weight tensor data.
 * Use this to wrap weight tensor->data accesses in compute kernels.
 * Zero overhead when NUMA replication is disabled.
 */
static inline void *ggml_numa_local_ptr(void *p) {
#if defined(GGML_NUMA_REPLICATE) && defined(__gnu_linux__)
    if (ggml_numa_replicate_is_enabled()) {
        return ggml_numa_replicate_get_local_ptr(p);
    }
#endif
    return p;
}

#ifdef __cplusplus
}
#endif

#endif
