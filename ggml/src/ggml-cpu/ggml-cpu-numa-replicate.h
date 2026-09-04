/*
 * ggml-cpu-numa-replicate.h
 * Public interface for NUMA weight replication
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/* Initialize NUMA topology for replication (call once at startup) */
void ggml_numa_replicate_init(void);

/* Get number of NUMA nodes (for replication) */
int ggml_numa_replicate_get_n_nodes(void);

/* Check if NUMA replication is active */
int ggml_numa_replicate_is_enabled(void);

/* Get thread's NUMA node based on thread_id */
int ggml_numa_replicate_get_thread_node(int thread_id);

/* Allocate per-node buffers: returns array of pointers (one per node).
 * Caller must use ggml_numa_replicate_free_per_node() to release. */
void **ggml_numa_replicate_alloc_per_node(size_t size);

/* Free per-node buffers */
void ggml_numa_replicate_free_per_node(void **ptrs);

/* Get node-local pointer for a thread from per-node array */
void *ggml_numa_replicate_get_local_ptr(void **ptrs, int thread_id);

/* Print allocation statistics */
void ggml_numa_replicate_stats(void);

#ifdef __cplusplus
}
#endif
