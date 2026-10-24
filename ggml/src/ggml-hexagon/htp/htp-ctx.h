#ifndef HTP_CTX_H
#define HTP_CTX_H

#include "dma-queue.h"
#include "hmx-queue.h"
#include "htp-dcache.h"
#include "htp-ops.h"
#include "hex-profile.h"
#include "work-queue.h"
#include "hex-fastdiv.h"

#include <assert.h>
#include <dspqueue.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>

#ifndef HTP_MAX_NTHREADS
#define HTP_MAX_NTHREADS 10
#endif

<<<<<<< HEAD
#define HTP_MAX_DIRTY_RANGES 16
=======
#define HTP_MAX_MMAPS 64
>>>>>>> ggml-org/master

// Memory mapping
struct htp_mmap {
    uint64_t size;
    uint64_t base;
    uint32_t fd;
<<<<<<< HEAD
    uint32_t reserved;
=======
    uint32_t flags;
>>>>>>> ggml-org/master
};

// Scratchpad state
struct htp_spad {
    const struct htp_tensor * src;             // original src of the data (for reuse)
    uint8_t *                 data;            // pointer to an area in vtcm
    uint32_t                  stride;          // stride used inside this spad
    uint32_t                  size;            // total size
    uint32_t                  size_per_thread; // size per thread
};

struct htp_context;

// Context while processing an Op
// TODO: fold this into the main context
struct htp_ops_context {
    struct htp_context * ctx;

    enum htp_op_code    op; // FIXME: rename to opcode
    int32_t             op_params[HTP_OP_MAX_PARAMS];
    int32_t             kernel_params[HTP_OP_MAX_KERN_PARAMS];

    const struct htp_tensor * src[HTP_OP_MAX_INPUTS];
    union {
        const struct htp_tensor * dst;
        const struct htp_tensor * dsts[HTP_OP_MAX_OUTPUTS];
    };

    // TODO convert these to an array
    struct htp_spad src0_spad;
    struct htp_spad src1_spad;
    struct htp_spad dst_spad;

<<<<<<< HEAD
    uint32_t n_threads;
    uint32_t flags;
=======
    uint32_t              flags;
    uint32_t              n_threads;
    struct fastdiv_values n_threads_div;
    int                   status;

    // Must stay last: not cleared by htp_ops_context_reset(), ops init the caches they use
    union {
        dma_cache_dm dma_cache_dm[HTP_MAX_NTHREADS];
        dma_cache_fa dma_cache_fa[HTP_MAX_NTHREADS];
    } __attribute__((aligned(128)));
>>>>>>> ggml-org/master
};

// Main context for htp DSP backend
struct htp_context {
    dspqueue_t             dsp_queue;

    struct htp_mmap        mmap[HTP_MAX_MMAPS];
    dma_queue_t            dma[HTP_MAX_NTHREADS];
<<<<<<< HEAD
    dma_queue_t            dma_cached[HTP_MAX_NTHREADS];
=======
    struct htp_thread_trace trace[HTP_MAX_NTHREADS + 1];
>>>>>>> ggml-org/master
    work_queue_t           work_queue;
    hmx_queue_t            hmx_queue;

    uint32_t               n_threads;
    struct fastdiv_values  n_threads_div;

    int                    thread_id;
    int                    thread_prio;

    bool                   hmx_enabled;
    bool                   etm;
    uint32_t               profiler;
    struct htp_thread_trace trace[HTP_MAX_NTHREADS + 1];

    uint8_t *              vtcm_base;
    size_t                 vtcm_size;
    uint32_t               vtcm_rctx;
    atomic_bool            vtcm_valid;
    atomic_bool            vtcm_needs_release;

    uint64_t               max_vmem;
    struct htp_dirty_range {
        uint32_t start;
        uint32_t end;
        uint32_t bi;
    } dirty_ranges[HTP_MAX_DIRTY_RANGES];

    // Persistent DDR scratchpad for MUL_MAT_ID mappings
    void *                 ddr_spad_base;
    size_t                 ddr_spad_size;

    struct htp_ops_context octx;

    qurt_thread_t          main_thread;
    void *                 main_stack;
    atomic_bool            killed;
    size_t                 footprint;
};

<<<<<<< HEAD
=======
static inline void htp_ops_context_reset(struct htp_ops_context * octx, struct htp_context * ctx) {
    memset(octx, 0, offsetof(struct htp_ops_context, dma_cache_dm));
    octx->ctx           = ctx;
    octx->n_threads     = ctx->n_threads;
    octx->n_threads_div = ctx->n_threads_div;
}

static inline bool htp_ops_context_set_n_threads(struct htp_ops_context * octx, uint32_t n_threads) {
    if (n_threads == 0 || n_threads > octx->ctx->n_threads) {
        return false;
    }

    if (n_threads != octx->n_threads) {
        octx->n_threads = n_threads;
        octx->n_threads_div = n_threads == octx->ctx->n_threads
            ? octx->ctx->n_threads_div
            : init_fastdiv_values(n_threads);
    }

    return true;
}

static inline void htp_ops_context_set_status(struct htp_ops_context * octx, int status) {
    if (status > HTP_STATUS_OK && octx->status == HTP_STATUS_OK) {
        octx->status = status;
    }
}

>>>>>>> ggml-org/master
int op_matmul(struct htp_ops_context * octx);
int op_matmul_id(struct htp_ops_context * octx);
int op_matmul_nx(struct htp_ops_context * octx);
int op_binary(struct htp_ops_context * octx);
int op_unary(struct htp_ops_context * octx);
int op_sum(struct htp_ops_context * octx);
int op_sum_rows(struct htp_ops_context * octx);
int op_argmax(struct htp_ops_context * octx);
int op_glu(struct htp_ops_context * octx);
int op_softmax(struct htp_ops_context * octx);
int op_add_id(struct htp_ops_context * octx);
int op_rope(struct htp_ops_context * octx);
int op_flash_attn_ext(struct htp_ops_context * octx);
int op_set_rows(struct htp_ops_context * octx);
int op_get_rows(struct htp_ops_context * octx);
int op_cpy(struct htp_ops_context * octx);
int op_repeat(struct htp_ops_context * octx);
int op_argsort(struct htp_ops_context * octx);
int op_top_k(struct htp_ops_context * octx);
int op_ssm_conv(struct htp_ops_context * octx);
int op_cumsum(struct htp_ops_context * octx);
int op_fill(struct htp_ops_context * octx);
int op_concat(struct htp_ops_context * octx);
int op_diag(struct htp_ops_context * octx);
int op_solve_tri(struct htp_ops_context * octx);
int op_gated_delta_net(struct htp_ops_context * octx);
int op_pad(struct htp_ops_context * octx);
int op_im2col(struct htp_ops_context * octx);
int op_im2col_3d(struct htp_ops_context * octx);
int op_pool_2d(struct htp_ops_context * octx);
int op_pool_1d(struct htp_ops_context * octx);
int op_allreduce(struct htp_ops_context * octx);
int op_roll(struct htp_ops_context * octx);

#endif /* HTP_CTX_H */
