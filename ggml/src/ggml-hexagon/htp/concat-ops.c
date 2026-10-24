<<<<<<< HEAD
=======
#include "concat-ops.h"
#include "dma-queue.h"
#include "hex-common.h"
#include "dma-copy.h"
#include "hex-fastdiv.h"
#include "hex-profile.h"
#include "hexagon_protos.h"
#include "hexagon_types.h"
>>>>>>> ggml-org/master
#include "htp-ctx.h"
#include "htp-fence.h"
#include "htp-ops.h"
<<<<<<< HEAD
#include "hexagon_types.h"
#include "hexagon_protos.h"
#include "hvx_hexagon_protos.h"
#include "hex-dma.h"
=======
#include "htp-tensor.h"
>>>>>>> ggml-org/master
#include "htp-vtcm.h"
#include "hvx-utils.h"
#include "hvx_hexagon_protos.h"
#include "gather-rows.h"

#include <string.h>

struct htp_concat_context {
    struct htp_ops_context * octx;
<<<<<<< HEAD
    uint32_t dim;
    uint32_t nrows_per_thread;
    struct fastdiv_values div_ne0;
    struct fastdiv_values div_ne1;
=======
    uint8_t * spad0_base;
    uint8_t * spad1_base;
    uint32_t  spad0_size_per_thread;
    uint32_t  spad1_size_per_thread;
    uint32_t  row_start;
    uint32_t  nrows;
    uint32_t  nrows_per_thread;
    uint32_t  nplanes;
>>>>>>> ggml-org/master
    struct fastdiv_values div_ne2;
};

static inline dma_addr_t concat_plane_addr(const struct htp_tensor * t, uint32_t p, const struct fastdiv_values * div_ne2, uint32_t ne2) {
    const uint32_t i3 = fastdiv(p, div_ne2);
    const uint32_t i2 = p - i3 * ne2;
    return t->data + i2 * t->nb[2] + i3 * t->nb[3];
}

static void concat_2d_f32_transposed(unsigned int nth, unsigned int ith, void * data) {
    struct htp_concat_context * cctx = (struct htp_concat_context *) data;
    struct htp_ops_context * octx = cctx->octx;

    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * src1 = octx->src[1];
    const struct htp_tensor * dst  = octx->dst;

    const uint32_t src0_ne0 = src0->ne[0];
    const uint32_t src1_ne0 = src1->ne[0];
    const uint32_t ne1      = dst->ne[1];

    const uint32_t start_i = ith * cctx->nrows_per_thread;
    const uint32_t end_i   = (start_i + cctx->nrows_per_thread < ne1) ? (start_i + cctx->nrows_per_thread) : ne1;
    if (start_i >= end_i) return;

    dma_queue * dma_q = octx->ctx->dma[ith];

    uint8_t * spad0_base = cctx->spad0_base + ith * cctx->spad0_size_per_thread;
    uint8_t * spad1_base = cctx->spad1_base + ith * cctx->spad1_size_per_thread;

    const uint32_t block_i = 32;
    const uint32_t spad1_stride = block_i * sizeof(float);

    const HVX_Vector offsets = hvx_vec_gather_offsets_w(spad1_stride);
    const uint32_t src1_ne0_padded = hex_round_up(src1_ne0, 32);
    const uint32_t src0_row_bytes  = src0_ne0 * sizeof(float);
    const uint32_t src0_row_padded = hex_round_up(src0_row_bytes, VLEN);
    const uint32_t src0_pre        = src0_row_padded - src0_row_bytes;
    const uint32_t spad0_row_bytes = src0_row_padded + src1_ne0_padded * sizeof(float);

<<<<<<< HEAD
    for (uint32_t i = start_i; i < end_i; i += block_i) {
        uint32_t current_block_i = (end_i - i < block_i) ? (end_i - i) : block_i;
=======
    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    const struct fastdiv_values * div_ne2 = &cctx->div_ne2;
    const uint32_t ne2 = dst->ne[2];
>>>>>>> ggml-org/master

    uint32_t p = 0;
    uint32_t i = start_i;

    const dma_addr_t src1_addr = concat_plane_addr(src1, p, div_ne2, ne2) + i * src1->nb[1];
    dma_queue_push(dma_q, dma_make_data(spad1_base, src1_addr), spad1_stride, src1->nb[0], MIN(end_i - i, block_i) * sizeof(float), src1_ne0);

    const dma_addr_t src0_addr = concat_plane_addr(src0, p, div_ne2, ne2) + i * src0->nb[1];
    dma_queue_push(dma_q, dma_make_data(spad0_base + src0_pre, src0_addr), spad0_row_bytes, src0->nb[1], src0_row_bytes, MIN(end_i - i, block_i));

    dma_queue_pop(dma_q); // src1

    while (p < cctx->nplanes) {
        const uint32_t current_block_i = MIN(end_i - i, block_i);

<<<<<<< HEAD
        for (uint32_t j = 0; j < src1_ne0_padded; j += 32) {
            #pragma unroll(4)
            for (uint32_t ii = 0; ii < current_block_i; ii++) {
                size_t rt = (size_t)(spad1_base + j * spad1_stride + ii * sizeof(float));
                Q6_vgather_ARMVw(&vtcm_tmp[ii], rt, mu, vv);
                uint8_t * dst_ptr = spad0_base + ii * spad0_row_bytes + (src0_ne0 + j) * sizeof(float);
                hvx_vmemu(dst_ptr) = vtcm_tmp[ii];
            }
        }
=======
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        for (uint32_t j = 0; j < src1_ne0; j += 32) {
            const uint8_t * src_ptr = spad1_base + j * spad1_stride;
            uint8_t * dst_ptr = spad0_base + src0_row_padded + j * sizeof(float);
            hvx_transpose_32x32_w_gather(dst_ptr, spad0_row_bytes, src_ptr, spad1_stride, offsets, current_block_i, MIN(src1_ne0 - j, 32));
        }
        hvx_gather_sync(spad0_base + src0_row_padded);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
>>>>>>> ggml-org/master

        uint32_t np = p;
        uint32_t ni = i + block_i;
        if (ni >= end_i) {
            ni = start_i;
            np++;
        }
        const bool has_next = np < cctx->nplanes;
        const uint32_t next_block_i = MIN(end_i - ni, block_i);

        // spad1 is free after the gather sync, prefetch next src1 ahead of the dst write
        if (has_next) {
            const dma_addr_t nsrc1_addr = concat_plane_addr(src1, np, div_ne2, ne2) + ni * src1->nb[1];
            dma_queue_push(dma_q, dma_make_data(spad1_base, nsrc1_addr), spad1_stride, src1->nb[0], next_block_i * sizeof(float), src1_ne0);
        }

        dma_queue_pop(dma_q); // src0

        const dma_addr_t dst_addr = concat_plane_addr(dst, p, div_ne2, ne2) + i * dst->nb[1];
        dma_queue_push(dma_q, dma_make_data(dst_addr, spad0_base + src0_pre), dst->nb[1], spad0_row_bytes, (src0_ne0 + src1_ne0) * sizeof(float), current_block_i);

        if (has_next) {
            dma_queue_pop(dma_q); // next src1
        }
        dma_queue_pop(dma_q); // dst

        // spad0 is free after the dst write
        if (has_next) {
            const dma_addr_t nsrc0_addr = concat_plane_addr(src0, np, div_ne2, ne2) + ni * src0->nb[1];
            dma_queue_push(dma_q, dma_make_data(spad0_base + src0_pre, nsrc0_addr), spad0_row_bytes, src0->nb[1], src0_row_bytes, next_block_i);
        }

        p = np;
        i = ni;
    }
    dma_queue_flush(dma_q);
}

static void concat_2d_f16_transposed(unsigned int nth, unsigned int ith, void * data) {
    struct htp_concat_context * cctx = (struct htp_concat_context *) data;
    struct htp_ops_context * octx = cctx->octx;

    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * src1 = octx->src[1];
    const struct htp_tensor * dst  = octx->dst;

    const uint32_t src0_ne0 = src0->ne[0];
    const uint32_t src1_ne0 = src1->ne[0];
    const uint32_t ne1      = dst->ne[1];

    const uint32_t start_i = ith * cctx->nrows_per_thread;
    const uint32_t end_i   = (start_i + cctx->nrows_per_thread < ne1) ? (start_i + cctx->nrows_per_thread) : ne1;
    if (start_i >= end_i) return;

    dma_queue * dma_q = octx->ctx->dma[ith];

    uint8_t * spad0_base = cctx->spad0_base + ith * cctx->spad0_size_per_thread;
    uint8_t * spad1_base = cctx->spad1_base + ith * cctx->spad1_size_per_thread;

    const uint32_t block_i = 64;
    const uint32_t spad1_stride = block_i * sizeof(__fp16);

    const HVX_Vector offsets = hvx_vec_gather_offsets_h(spad1_stride);
    const uint32_t src1_ne0_padded = hex_round_up(src1_ne0, 64);
    const uint32_t src0_row_bytes  = src0_ne0 * sizeof(__fp16);
    const uint32_t src0_row_padded = hex_round_up(src0_row_bytes, VLEN);
    const uint32_t src0_pre        = src0_row_padded - src0_row_bytes;
    const uint32_t spad0_row_bytes = src0_row_padded + src1_ne0_padded * sizeof(__fp16);

<<<<<<< HEAD
    for (uint32_t i = start_i; i < end_i; i += block_i) {
        uint32_t current_block_i = (end_i - i < block_i) ? (end_i - i) : block_i;
=======
    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    const struct fastdiv_values * div_ne2 = &cctx->div_ne2;
    const uint32_t ne2 = dst->ne[2];
>>>>>>> ggml-org/master

    uint32_t p = 0;
    uint32_t i = start_i;

    const dma_addr_t src1_addr = concat_plane_addr(src1, p, div_ne2, ne2) + i * src1->nb[1];
    dma_queue_push(dma_q, dma_make_data(spad1_base, src1_addr), spad1_stride, src1->nb[0], MIN(end_i - i, block_i) * sizeof(__fp16), src1_ne0);

    const dma_addr_t src0_addr = concat_plane_addr(src0, p, div_ne2, ne2) + i * src0->nb[1];
    dma_queue_push(dma_q, dma_make_data(spad0_base + src0_pre, src0_addr), spad0_row_bytes, src0->nb[1], src0_row_bytes, MIN(end_i - i, block_i));

    dma_queue_pop(dma_q); // src1

    while (p < cctx->nplanes) {
        const uint32_t current_block_i = MIN(end_i - i, block_i);

<<<<<<< HEAD
        for (uint32_t j = 0; j < src1_ne0_padded; j += 64) {
            #pragma unroll(4)
            for (uint32_t ii = 0; ii < current_block_i; ii++) {
                size_t rt = (size_t)(spad1_base + j * spad1_stride + ii * sizeof(__fp16));
                Q6_vgather_ARMVh(&vtcm_tmp[ii], rt, mu, vv);
                uint8_t * dst_ptr = spad0_base + ii * spad0_row_bytes + (src0_ne0 + j) * sizeof(__fp16);
                hvx_vmemu(dst_ptr) = vtcm_tmp[ii];
            }
        }

        dma_queue_pop(q); // src0

        uint8_t * dst_ptr = (uint8_t *)dst->data + i * dst->nb[1];
        dma_queue_push(q, dma_make_ptr(dst_ptr, spad0_base), dst->nb[1], spad0_row_bytes, (src0_ne0 + src1_ne0) * sizeof(__fp16), current_block_i);

        dma_queue_pop(q);
    }
}

static void concat_generic(unsigned int nth, unsigned int ith, void * data) {
    struct htp_concat_context * cctx = (struct htp_concat_context *) data;
    struct htp_ops_context * octx = cctx->octx;

    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * src1 = octx->src[1];
    const struct htp_tensor * dst  = octx->dst;

    const int dim = cctx->dim;
    const uint32_t type_size = (dst->type == HTP_TYPE_F32 || dst->type == HTP_TYPE_I32) ? 4 : 2;

    const uint32_t ne[4] = {dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]};
    const uint32_t total_elements = ne[0] * ne[1] * ne[2] * ne[3];
    const uint32_t chunk_size = (total_elements + nth - 1) / nth;

    const uint32_t start_idx = MIN(ith * chunk_size, total_elements);
    const uint32_t end_idx   = MIN(start_idx + chunk_size, total_elements);

    // Naive scalar element-wise copy
    for (uint32_t idx = start_idx; idx < end_idx; idx++) {
        uint32_t idx_div_ne0 = fastdiv(idx, &cctx->div_ne0);
        uint32_t i0 = idx - idx_div_ne0 * ne[0];

        uint32_t idx_div_ne01 = fastdiv(idx_div_ne0, &cctx->div_ne1);
        uint32_t i1 = idx_div_ne0 - idx_div_ne01 * ne[1];

        uint32_t idx_div_ne012 = fastdiv(idx_div_ne01, &cctx->div_ne2);
        uint32_t i2 = idx_div_ne01 - idx_div_ne012 * ne[2];
        uint32_t i3 = idx_div_ne012;

        uint8_t * dst_ptr = (uint8_t *)dst->data + i3 * dst->nb[3] + i2 * dst->nb[2] + i1 * dst->nb[1] + i0 * dst->nb[0];

        uint32_t idx_dim = 0;
        if (dim == 0) idx_dim = i0;
        else if (dim == 1) idx_dim = i1;
        else if (dim == 2) idx_dim = i2;
        else if (dim == 3) idx_dim = i3;

        const struct htp_tensor * src = (idx_dim < src0->ne[dim]) ? src0 : src1;

        uint32_t s0 = i0;
        uint32_t s1 = i1;
        uint32_t s2 = i2;
        uint32_t s3 = i3;

        if (dim == 0 && src == src1) s0 -= src0->ne[0];
        if (dim == 1 && src == src1) s1 -= src0->ne[1];
        if (dim == 2 && src == src1) s2 -= src0->ne[2];
        if (dim == 3 && src == src1) s3 -= src0->ne[3];

        uint8_t * src_ptr = (uint8_t *)src->data + s3 * src->nb[3] + s2 * src->nb[2] + s1 * src->nb[1] + s0 * src->nb[0];

        if (type_size == 4) {
            *(float*)dst_ptr = *(float*)src_ptr;
        } else {
            *(__fp16*)dst_ptr = *(__fp16*)src_ptr;
=======
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        for (uint32_t j = 0; j < src1_ne0; j += 64) {
            const uint8_t * src_ptr = spad1_base + j * spad1_stride;
            uint8_t * dst_ptr = spad0_base + src0_row_padded + j * sizeof(__fp16);
            hvx_transpose_64x64_h_gather(dst_ptr, spad0_row_bytes, src_ptr, spad1_stride, offsets, current_block_i, MIN(src1_ne0 - j, 64));
        }
        hvx_gather_sync(spad0_base + src0_row_padded);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);

        uint32_t np = p;
        uint32_t ni = i + block_i;
        if (ni >= end_i) {
            ni = start_i;
            np++;
>>>>>>> ggml-org/master
        }
        const bool has_next = np < cctx->nplanes;
        const uint32_t next_block_i = MIN(end_i - ni, block_i);

        // spad1 is free after the gather sync, prefetch next src1 ahead of the dst write
        if (has_next) {
            const dma_addr_t nsrc1_addr = concat_plane_addr(src1, np, div_ne2, ne2) + ni * src1->nb[1];
            dma_queue_push(dma_q, dma_make_data(spad1_base, nsrc1_addr), spad1_stride, src1->nb[0], next_block_i * sizeof(__fp16), src1_ne0);
        }

        dma_queue_pop(dma_q); // src0

        const dma_addr_t dst_addr = concat_plane_addr(dst, p, div_ne2, ne2) + i * dst->nb[1];
        dma_queue_push(dma_q, dma_make_data(dst_addr, spad0_base + src0_pre), dst->nb[1], spad0_row_bytes, (src0_ne0 + src1_ne0) * sizeof(__fp16), current_block_i);

        if (has_next) {
            dma_queue_pop(dma_q); // next src1
        }
        dma_queue_pop(dma_q); // dst

        // spad0 is free after the dst write
        if (has_next) {
            const dma_addr_t nsrc0_addr = concat_plane_addr(src0, np, div_ne2, ne2) + ni * src0->nb[1];
            dma_queue_push(dma_q, dma_make_data(spad0_base + src0_pre, nsrc0_addr), spad0_row_bytes, src0->nb[1], src0_row_bytes, next_block_i);
        }

        p = np;
        i = ni;
    }
    dma_queue_flush(dma_q);
}

static int concat_regular(struct htp_ops_context * octx, int dim, uint32_t type_size) {
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * src1 = octx->src[1];
    const struct htp_tensor * dst  = octx->dst;

    struct htp_tensor view0 = *dst;
    struct htp_tensor view1 = *dst;
    for (int d = 0; d < HTP_OP_MAX_DIMS; d++) {
        view0.ne[d] = src0->ne[d];
        view1.ne[d] = src1->ne[d];
    }
    view1.data += src0->ne[dim] * dst->nb[dim];

    const uint32_t total_rows_0 = src0->ne[1] * src0->ne[2] * src0->ne[3];
    const uint32_t total_rows_1 = src1->ne[1] * src1->ne[2] * src1->ne[3];

    uint32_t rstart0 = 0, nrows0 = total_rows_0;
    uint32_t rstart1 = 0, nrows1 = total_rows_1;

    if (octx->ctx->mdev.count > 1) {
        const struct htp_tensor_mdev_range range0 = htp_tensor_mdev_partition(
            total_rows_0, 1, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        rstart0 = range0.start;
        nrows0  = range0.count;

        const struct htp_tensor_mdev_range range1 = htp_tensor_mdev_partition(
            total_rows_1, 1, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        rstart1 = range1.start;
        nrows1  = range1.count;
    }

    dma_queue * q = octx->ctx->dma[0];

    dma_cpy_sametype_sameshape_range(q, &view0, src0, type_size, rstart0, nrows0);
    dma_cpy_sametype_sameshape_range(q, &view1, src1, type_size, rstart1, nrows1);
    dma_queue_flush(q);

    return HTP_STATUS_OK;
}

static int concat_transposed(struct htp_ops_context * octx, const struct htp_concat_kernel_params * kparams, uint32_t type_size) {
    if (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    const struct htp_tensor * dst = octx->dst;

    const uint32_t total_rows = dst->ne[1];
    uint32_t row_start = 0;
    uint32_t nrows     = total_rows;
    if (octx->ctx->mdev.count > 1) {
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_rows, 1, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        row_start = range.start;
        nrows     = range.count;
    }

    if (nrows == 0 || dst->ne[2] == 0 || dst->ne[3] == 0) {
        return HTP_STATUS_OK;
    }

    if (kparams->vtcm_size > octx->ctx->vtcm_size) {
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    const uint32_t n_threads = octx->n_threads;

    struct htp_concat_transposed_vtcm_layout layout;
    htp_concat_transposed_vtcm_layout_build(&layout, octx->src[0]->ne[0], octx->src[1]->ne[0], type_size, n_threads);

    uint8_t * vtcm_base = (uint8_t *) octx->ctx->vtcm_base;

    struct htp_concat_context cctx;
<<<<<<< HEAD
    cctx.octx = octx;
    cctx.dim = dim;
    cctx.div_ne0 = init_fastdiv_values(dst->ne[0]);
    cctx.div_ne1 = init_fastdiv_values(dst->ne[1]);
    cctx.div_ne2 = init_fastdiv_values(dst->ne[2]);

    void (*worker_func)(unsigned int, unsigned int, void *) = concat_generic;

    if (dim == 0 && is_2d && is_src1_transposed && !is_src0_transposed) {
        n_threads = MIN(dst->ne[1], n_threads);
        if (n_threads < 1) {
            n_threads = 1;
        }
        uint32_t block_i = (type_size == 4) ? 32 : 64;

        cctx.nrows_per_thread = hmx_ceil_div(dst->ne[1], n_threads);

        // Allocate VTCM
        uint32_t spad1_stride = block_i * type_size;

        uint32_t src1_ne0_padded = hex_round_up(src1->ne[0], block_i);
        uint32_t spad0_row_bytes = hex_round_up((src0->ne[0] + src1_ne0_padded) * type_size, VLEN);

        octx->src0_spad.size_per_thread = block_i * spad0_row_bytes;
        octx->src1_spad.size_per_thread = src1_ne0_padded * spad1_stride + block_i * VLEN;

        octx->src0_spad.size = n_threads * octx->src0_spad.size_per_thread;
        octx->src1_spad.size = n_threads * octx->src1_spad.size_per_thread;

        if (octx->src0_spad.size + octx->src1_spad.size > octx->ctx->vtcm_size) {
            return HTP_STATUS_VTCM_TOO_SMALL;
        }

        octx->src0_spad.data = octx->ctx->vtcm_base;
        octx->src1_spad.data = octx->src0_spad.data + octx->src0_spad.size;
        octx->src0_spad.src  = NULL;
        octx->src1_spad.src  = NULL;

        if (type_size == 4) {
            worker_func = concat_2d_f32_transposed;
        } else {
            worker_func = concat_2d_f16_transposed;
        }
    }

    worker_pool_run_func(octx->ctx->worker_pool, worker_func, &cctx, n_threads);
=======
    cctx.octx                  = octx;
    cctx.spad0_base            = VTCM_LAYOUT_PTR(uint8_t, vtcm_base, layout.off_spad0);
    cctx.spad1_base            = VTCM_LAYOUT_PTR(uint8_t, vtcm_base, layout.off_spad1);
    cctx.spad0_size_per_thread = layout.src0_spad_size_per_thread;
    cctx.spad1_size_per_thread = layout.src1_spad_size_per_thread;
    cctx.row_start             = row_start;
    cctx.nrows                 = nrows;
    cctx.nplanes               = dst->ne[2] * dst->ne[3];
    cctx.div_ne2               = kparams->u.transposed.div_ne2;
    cctx.nrows_per_thread      = fastdiv(nrows + n_threads - 1, &octx->n_threads_div);

    work_queue_func_t worker_func = (type_size == 4) ? concat_2d_f32_transposed : concat_2d_f16_transposed;
    work_queue_run(octx->ctx->work_queue, worker_func, &cctx, n_threads);
>>>>>>> ggml-org/master
    return HTP_STATUS_OK;
}

int op_concat(struct htp_ops_context * octx) {
    const struct htp_concat_kernel_params * kparams = (const struct htp_concat_kernel_params *) octx->kernel_params;
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * src1 = octx->src[1];
    const struct htp_tensor * dst  = octx->dst;

    int status = HTP_STATUS_OK;
    const uint32_t type_size = (dst->type == HTP_TYPE_F32 || dst->type == HTP_TYPE_I32) ? 4 : 2;

    switch (kparams->kernel_type) {
        case HTP_CONCAT_KERNEL_REGULAR:
            status = concat_regular(octx, kparams->dim, type_size);
            break;

        case HTP_CONCAT_KERNEL_TRANSPOSED:
            status = concat_transposed(octx, kparams, type_size);
            break;

        case HTP_CONCAT_KERNEL_GATHER_ROWS:
            status = htp_gather_rows(octx, src0, src1, &kparams->u.gather, kparams->n_threads);
            break;

        default:
            status = HTP_STATUS_NO_SUPPORT;
            break;
    }

    htp_ops_context_set_status(octx, status);

    return octx->status;
}
