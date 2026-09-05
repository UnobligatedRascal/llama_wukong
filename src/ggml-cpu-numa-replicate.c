/*
 * ggml-cpu-numa-replicate.c
 * NUMA weight replication for llama_wukong (NOUGHT dual-socket Xeon)
 *
 * Implements GGML_NUMA_STRATEGY_MIRROR: duplicate weights on each NUMA node
 * so threads read from local memory instead of crossing UPI links.
 *
 * Key design:
 * - numa_alloc_onnode() for per-node buffer allocation
 * - Thread-local NUMA node → local weight pointer swap in compute
 * - NUMA-local wdata buffers for quantized dequantization
 */

#include "ggml.h"
#include "ggml-cpu-impl.h"
#include "ggml-impl.h"

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

/* Maximum NUMA nodes and CPUs we support */
#define GGML_NUMA_REPLICATE_MAX_NODES 8
#define GGML_NUMA_REPLICATE_MAX_CPUS 512

/* Per-node state */
struct ggml_numa_replicate_node {
    int node_id;
    int n_cpus;
    int cpus[GGML_NUMA_REPLICATE_MAX_CPUS];
};

/* Global replicate state */
struct ggml_numa_replicate_state {
    int enabled;
    int n_nodes;
    int current_node; /* main process node */
    struct ggml_numa_replicate_node nodes[GGML_NUMA_REPLICATE_MAX_NODES];
    /* Per-node allocation tracking */
    int alloc_count;
    size_t alloc_bytes[GGML_NUMA_REPLICATE_MAX_NODES];
};

static struct ggml_numa_replicate_state g_numa_rep = {0};

/* Get current NUMA node for calling thread */
static int ggml_numa_rep_get_current_node(void) {
    unsigned node;
    int ret = syscall(SYS_getcpu, NULL, &node, NULL);
    return (ret == 0 && node < GGML_NUMA_REPLICATE_MAX_NODES) ? (int)node : 0;
}

/* Map CPU ID to NUMA node */
static int ggml_numa_rep_cpu_to_node(int cpu_id) {
    if (cpu_id < 0 || cpu_id >= GGML_NUMA_REPLICATE_MAX_CPUS) {
        return 0;
    }
    for (int n = 0; n < g_numa_rep.n_nodes; n++) {
        struct ggml_numa_replicate_node *node = &g_numa_rep.nodes[n];
        for (int i = 0; i < node->n_cpus; i++) {
            if (node->cpus[i] == cpu_id) {
                return n;
            }
        }
    }
    return 0;
}

/* Allocate memory on a specific NUMA node */
static void *ggml_numa_rep_alloc_onnode(size_t size, int node) {
    void *ptr = numa_alloc_onnode(size, node);
    if (ptr != NULL) {
        g_numa_rep.alloc_bytes[node] += size;
        g_numa_rep.alloc_count++;
    }
    return ptr;
}

/* Allocate and copy data to specific NUMA node */
static void *ggml_numa_rep_alloc_copy_onnode(const void *src, size_t size, int node) {
    void *ptr = ggml_numa_rep_alloc_onnode(size, node);
    if (ptr != NULL && src != NULL) {
        memcpy(ptr, src, size);
    }
    return ptr;
}

/* Initialize NUMA topology for replication */
void ggml_numa_replicate_init(void) {
    if (g_numa_rep.enabled) {
        return; /* already initialized */
    }

    /* Check if libnuma is usable */
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
    g_numa_rep.current_node = ggml_numa_rep_get_current_node();

    /* Enumerate CPUs per node using numa_node_to_cpus with bitmask */
    int n_possible_cpus = numa_num_possible_cpus();
    for (int n = 0; n < g_numa_rep.n_nodes; n++) {
        struct ggml_numa_replicate_node *node = &g_numa_rep.nodes[n];
        node->node_id = n;
        node->n_cpus = 0;

        struct bitmask *cpumask = numa_allocate_cpumask();
        if (cpumask == NULL) {
            fprintf(stderr, "ggml_numa_replicate: failed to allocate cpumask for node %d\n", n);
            continue;
        }

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

/* Get number of NUMA nodes (for replication) */
int ggml_numa_replicate_get_n_nodes(void) {
    return g_numa_rep.enabled ? g_numa_rep.n_nodes : 1;
}

/* Check if NUMA replication is active */
int ggml_numa_replicate_is_enabled(void) {
    return g_numa_rep.enabled;
}

/* Get thread's NUMA node */
int ggml_numa_replicate_get_thread_node(int thread_id) {
    if (!g_numa_rep.enabled) return 0;
    return ggml_numa_rep_cpu_to_node(thread_id);
}

/* Allocate per-node buffers: returns array of pointers (one per node) */
void **ggml_numa_replicate_alloc_per_node(size_t size) {
    if (!g_numa_rep.enabled || g_numa_rep.n_nodes < 2) {
        /* Single node: allocate once, all nodes share */
        void **ptrs = (void **)malloc(sizeof(void *) * GGML_NUMA_REPLICATE_MAX_NODES);
        if (ptrs == NULL) return NULL;
        ptrs[0] = malloc(size);
        for (int n = 1; n < g_numa_rep.n_nodes; n++) {
            ptrs[n] = ptrs[0]; /* shared */
        }
        return ptrs;
    }

    void **ptrs = (void **)malloc(sizeof(void *) * g_numa_rep.n_nodes);
    if (ptrs == NULL) return NULL;

    /* Allocate on node 0 first (source) */
    ptrs[0] = ggml_numa_rep_alloc_onnode(size, 0);
    if (ptrs[0] == NULL) {
        free(ptrs);
        return NULL;
    }

    /* Copy to remaining nodes */
    for (int n = 1; n < g_numa_rep.n_nodes; n++) {
        ptrs[n] = ggml_numa_rep_alloc_copy_onnode(ptrs[0], size, n);
        if (ptrs[n] == NULL) {
            /* Fallback: share with node 0 if allocation fails */
            ptrs[n] = ptrs[0];
            fprintf(stderr, "ggml_numa_replicate: fallback to shared alloc for node %d\n", n);
        }
    }

    return ptrs;
}

/* Free per-node buffers */
void ggml_numa_replicate_free_per_node(void **ptrs) {
    if (ptrs == NULL) return;

    if (!g_numa_rep.enabled || g_numa_rep.n_nodes < 2) {
        if (ptrs[0]) free(ptrs[0]);
        free(ptrs);
        return;
    }

    /* Free unique allocations, skip duplicates */
    for (int n = 0; n < g_numa_rep.n_nodes; n++) {
        int is_unique = 1;
        for (int m = 0; m < n; m++) {
            if (ptrs[m] == ptrs[n]) {
                is_unique = 0;
                break;
            }
        }
        if (is_unique && ptrs[n]) {
            numa_free(ptrs[n], 0); /* size 0 = auto-detect */
        }
    }
    free(ptrs);
}

/* Get node-local pointer for a thread */
void *ggml_numa_replicate_get_local_ptr(void **ptrs, int thread_id) {
    if (!g_numa_rep.enabled) return ptrs[0];
    int node = ggml_numa_replicate_get_thread_node(thread_id);
    if (node < g_numa_rep.n_nodes) {
        return ptrs[node];
    }
    return ptrs[0]; /* fallback */
}

/* Print allocation stats */
void ggml_numa_replicate_stats(void) {
    if (!g_numa_rep.enabled) return;
    fprintf(stderr, "ggml_numa_replicate: %d allocations\n", g_numa_rep.alloc_count);
    for (int n = 0; n < g_numa_rep.n_nodes; n++) {
        fprintf(stderr, "  node %d: %.2f MB\n", n,
                g_numa_rep.alloc_bytes[n] / (1024.0 * 1024.0));
    }
}

#else /* !__gnu_linux__ */

/* Stub implementations for non-Linux */
void ggml_numa_replicate_init(void) {}
int ggml_numa_replicate_get_n_nodes(void) { return 1; }
int ggml_numa_replicate_is_enabled(void) { return 0; }
int ggml_numa_replicate_get_thread_node(int thread_id) { (void)thread_id; return 0; }
void **ggml_numa_replicate_alloc_per_node(size_t size) {
    void **ptrs = (void **)malloc(sizeof(void *));
    if (ptrs) ptrs[0] = malloc(size);
    return ptrs;
}
void ggml_numa_replicate_free_per_node(void **ptrs) {
    if (ptrs) { if (ptrs[0]) free(ptrs[0]); free(ptrs); }
}
void *ggml_numa_replicate_get_local_ptr(void **ptrs, int thread_id) {
    (void)thread_id;
    return ptrs[0];
}
void ggml_numa_replicate_stats(void) {}

#endif
