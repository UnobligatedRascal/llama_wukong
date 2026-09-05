/*
 * ggml-cpu-numa-replicate.c
 * NUMA weight replication for llama_wukong (NOUGHT dual-socket Xeon)
 *
 * REVISED DESIGN: Post-allocation data replication with TLS-cached pointer swap.
 *
 * Problem: ggml_backend_tensor_alloc() computes absolute tensor->data pointers
 * from ggml_backend_buffer_get_base(). If get_base() returns different values
 * per thread, tensor addresses break.
 *
 * Solution:
 * 1. Allocate weights normally (single buffer on node 0)
 * 2. After allocation completes, replicate data to per-node buffers
 * 3. Use per-thread base pointer offset: each thread's local copy sits at
 *    the SAME virtual offset as the original, so tensor->data + offset gives
 *    the local pointer.
 *
 * Implementation: We cannot change tensor->data per-thread (it's a static
 * pointer). Instead we use the fact that compute kernels access tensors through
 * their data pointers. We replicate data and provide a mapping function:
 *    ggml_numa_replicate_get_local_ptr(void *global_ptr)
 *
 * But kernels don't call this — they use tensor->data directly.
 *
 * REAL SOLUTION: Override tensor data pointers AFTER replication.
 * We track the original buffer base and each node's base. After load_tensors()
 * completes, we adjust every tensor's data pointer to point to the local node
 * copy based on the thread that will use it. But threads are assigned AFTER load.
 *
 * SIMPLEST WORKING SOLUTION: Use per-node base offset trick.
 * Allocate all per-node copies contiguously: [node0_data][node1_data].
 * The base pointer returned by get_base() depends on the thread's NUMA node,
 * but the tensor->data offsets are the same. So:
 *   node0: base=N0, tensor->data = N0 + offset
 *   node1: base=N1, tensor->data = N1 + offset
 * But tensor->data is FIXED after allocation...
 *
 * ULTIMATE SIMPLE SOLUTION: Just allocate weights per-node using numa_alloc_onnode,
 * then after tensor allocation, copy node0's data to other nodes. Then override
 * the allocator's view by storing per-node base pointers and using them in the
 * NUMA-aware compute path. For kernels that read tensor->data directly, we need
 * to patch tensor->data per thread — which we can't do atomically.
 *
 * FINAL WORKING APPROACH: Don't try to trick the allocator.
 * After model load, we have:
 *   - One buffer with all weights
 *   - All tensor->data pointers pointing into it
 *
 * We replicate the ENTIRE buffer to per-node copies.
 * Then we create a "virtual" mapping: tensor local ptr = node_base + (tensor->data - original_base)
 * This is a simple pointer offset operation that can be done in TLS.
 *
 * For kernels: they MUST be modified to use ggml_numa_replicate_get_local_ptr(tensor->data)
 * OR we use a shared memory trick: map the same virtual address to different physical pages
 * per thread using madvise/MAP_PRIVATE. Too complex.
 *
 * PRACTICAL APPROACH for now:
 * - Replicate data to per-node buffers
 * - Provide ggml_numa_replicate_get_local_tensor_data(void *global_ptr) for kernels
 * - Document that compute kernels must use this for weight accesses
 * - For now, verify the replication works by checking memory locality
 */

#include "ggml.h"
#include "ggml-cpu-impl.h"
#include "ggml-impl.h"
#include "ggml-backend.h"

#if defined(__gnu_linux__)
#include <numa.h>
#include <numaif.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <string.h>

#ifndef SYS_getcpu
#ifdef SYS_get_cpu
#define SYS_getcpu SYS_get_cpu
#endif
#endif

#define GGML_NUMA_REPLICATE_MAX_NODES 8
#define GGML_NUMA_REPLICATE_MAX_CPUS 512

struct ggml_numa_replicate_node {
    int node_id;
    int n_cpus;
    int cpus[GGML_NUMA_REPLICATE_MAX_CPUS];
};

struct ggml_numa_replicate_state {
    int enabled;
    int n_nodes;
    int current_node;
    struct ggml_numa_replicate_node nodes[GGML_NUMA_REPLICATE_MAX_NODES];

    /* Per-node data copies for weight replication */
    void * node_bases[GGML_NUMA_REPLICATE_MAX_NODES];
    void * original_base;
    size_t original_size;
    int has_replicated;

    int alloc_count;
    size_t alloc_bytes[GGML_NUMA_REPLICATE_MAX_NODES];
};

static struct ggml_numa_replicate_state g_numa_rep = {0};

static __thread int tls_numa_node = -1;

static int ggml_numa_rep_get_thread_node_cached(void) {
    if (tls_numa_node < 0) {
        unsigned node;
        if (syscall(SYS_getcpu, NULL, &node, NULL) == 0 &&
            node < GGML_NUMA_REPLICATE_MAX_NODES) {
            tls_numa_node = (int)node;
        } else {
            tls_numa_node = 0;
        }
    }
    return tls_numa_node;
}

static void *ggml_numa_rep_alloc_onnode(size_t size, int node) {
    void *ptr = numa_alloc_onnode(size, node);
    if (ptr != NULL) {
        g_numa_rep.alloc_bytes[node] += size;
        g_numa_rep.alloc_count++;
    }
    return ptr;
}

void ggml_numa_replicate_init(void) {
    if (g_numa_rep.enabled) return;

    if (numa_available() < 0) {
        fprintf(stderr, "ggml_numa_replicate: libnuma not available\n");
        return;
    }

    int max_node = numa_max_node();
    if (max_node < 1) {
        fprintf(stderr, "ggml_numa_replicate: only 1 NUMA node, replication disabled\n");
        return;
    }

    if (max_node >= GGML_NUMA_REPLICATE_MAX_NODES) {
        fprintf(stderr, "ggml_numa_replicate: too many nodes (%d), max %d\n",
                max_node + 1, GGML_NUMA_REPLICATE_MAX_NODES);
        return;
    }

    g_numa_rep.n_nodes = max_node + 1;
    unsigned node;
    syscall(SYS_getcpu, NULL, &node, NULL);
    g_numa_rep.current_node = (int)node;

    int n_possible_cpus = numa_num_possible_cpus();
    for (int n = 0; n < g_numa_rep.n_nodes; n++) {
        struct ggml_numa_replicate_node *node = &g_numa_rep.nodes[n];
        node->node_id = n;
        node->n_cpus = 0;

        struct bitmask *cpumask = numa_allocate_cpumask();
        if (cpumask == NULL) continue;

        int rc = numa_node_to_cpus(n, cpumask);
        if (rc == 0) {
            for (int cpu = 0; cpu < n_possible_cpus && node->n_cpus < GGML_NUMA_REPLICATE_MAX_CPUS; cpu++) {
                if (numa_bitmask_isbitset(cpumask, cpu)) {
                    node->cpus[node->n_cpus++] = cpu;
                }
            }
        }
        numa_free_cpumask(cpumask);

        fprintf(stderr, "ggml_numa_replicate: node %d has %d CPUs\n", n, node->n_cpus);
    }

    g_numa_rep.enabled = 1;
    fprintf(stderr, "ggml_numa_replicate: ENABLED across %d nodes, main process on node %d\n",
            g_numa_rep.n_nodes, g_numa_rep.current_node);
}

int ggml_numa_replicate_get_n_nodes(void) {
    return g_numa_rep.enabled ? g_numa_rep.n_nodes : 1;
}

int ggml_numa_replicate_is_enabled(void) {
    return g_numa_rep.enabled;
}

/*
 * Replicate a buffer's data to per-node copies.
 * Called after tensor allocation completes.
 */
void ggml_numa_replicate_weights(void *base, size_t size) {
    if (!g_numa_rep.enabled || g_numa_rep.n_nodes < 2) return;
    if (g_numa_rep.has_replicated) return; /* only once */

    g_numa_rep.original_base = base;
    g_numa_rep.original_size = size;
    g_numa_rep.node_bases[0] = base; /* node 0 uses original */

    for (int n = 1; n < g_numa_rep.n_nodes; n++) {
        g_numa_rep.node_bases[n] = ggml_numa_rep_alloc_onnode(size, n);
        if (g_numa_rep.node_bases[n]) {
            memcpy(g_numa_rep.node_bases[n], base, size);
            fprintf(stderr, "ggml_numa_replicate: replicated %.2f MB to node %d\n",
                    size / (1024.0 * 1024.0), n);
        } else {
            fprintf(stderr, "ggml_numa_replicate: FAILED to allocate on node %d, falling back to shared\n", n);
            g_numa_rep.node_bases[n] = base;
        }
    }

    g_numa_rep.has_replicated = 1;
}

/* Get local node's pointer for a global weight address */
void *ggml_numa_replicate_get_local_ptr(void *global_ptr) {
    if (!g_numa_rep.has_replicated) return global_ptr;

    char *global = (char *)global_ptr;
    char *orig_base = (char *)g_numa_rep.original_base;

    /* Check if pointer is within replicated range */
    if (global < orig_base ||
        global >= orig_base + g_numa_rep.original_size) {
        return global_ptr; /* not in replicated region */
    }

    size_t offset = global - orig_base;
    int node = ggml_numa_rep_get_thread_node_cached();

    if (node < g_numa_rep.n_nodes && g_numa_rep.node_bases[node]) {
        return (char *)g_numa_rep.node_bases[node] + offset;
    }
    return global_ptr;
}

/* Wrap buffer: keeps original base for allocator, replicates data */
ggml_backend_buffer_t ggml_backend_cpu_numa_buffer_wrap(ggml_backend_buffer_t inner) {
    /* For now, just trigger replication and return original buffer unchanged.
     * The replication is done by data offset mapping in get_local_ptr(). */
    if (!ggml_numa_replicate_is_enabled() ||
        ggml_numa_replicate_get_n_nodes() < 2) {
        return NULL;
    }

    void *base = ggml_backend_buffer_get_base(inner);
    size_t size = ggml_backend_buffer_get_size(inner);

    ggml_numa_replicate_weights(base, size);

    fprintf(stderr, "ggml_numa_replicate: wrapped buffer of size %.2f MB\n",
            size / (1024.0 * 1024.0));

    return NULL; /* no wrapper needed yet - replication is via offset mapping */
}

void ggml_numa_replicate_stats(void) {
    if (!g_numa_rep.enabled) return;
    fprintf(stderr, "ggml_numa_replicate: %d allocations, replicated=%d\n",
            g_numa_rep.alloc_count, g_numa_rep.has_replicated);
    for (int n = 0; n < g_numa_rep.n_nodes; n++) {
        fprintf(stderr, "  node %d: %.2f MB\n", n,
                g_numa_rep.alloc_bytes[n] / (1024.0 * 1024.0));
    }
}

#else /* !__gnu_linux__ */

typedef struct ggml_backend_buffer ggml_backend_buffer_t;
void ggml_numa_replicate_init(void) {}
int ggml_numa_replicate_get_n_nodes(void) { return 1; }
int ggml_numa_replicate_is_enabled(void) { return 0; }
void ggml_numa_replicate_weights(void *base, size_t size) { GGML_UNUSED(base); GGML_UNUSED(size); }
void *ggml_numa_replicate_get_local_ptr(void *global_ptr) { return global_ptr; }
ggml_backend_buffer_t ggml_backend_cpu_numa_buffer_wrap(ggml_backend_buffer_t inner) { GGML_UNUSED(inner); return NULL; }
void ggml_numa_replicate_stats(void) {}

#endif
