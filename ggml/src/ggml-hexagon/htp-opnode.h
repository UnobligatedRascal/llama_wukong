#ifndef HTP_OPNODE_H
#define HTP_OPNODE_H

#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "htp/htp-ops.h"

#include <cstring>
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

#define HTP_OP_MAX_FUSED 16

// local std::inplace_vector replacement
template <typename T, size_t MAX_N>
struct htp_opvec {
    T      data[MAX_N] {};
    size_t n = 0;

    htp_opvec() = default;
    htp_opvec(std::initializer_list<T> init) {
        GGML_ASSERT(init.size() <= MAX_N);
        for (const auto & v : init) {
            data[n++] = v;
        }
    }

    size_t    size()  const { return n; }
    bool      empty() const { return n == 0; }
    void      clear()       { n = 0; }
    T *       begin()       { return data; }
    T *       end()         { return data + n; }
    const T * begin() const { return data; }
    const T * end()   const { return data + n; }
    T &       back()        { return data[n - 1]; }
    const T & back()  const { return data[n - 1]; }
    T &       operator[](size_t i)       { return data[i]; }
    const T & operator[](size_t i) const { return data[i]; }

    void push_back(T val) {
        GGML_ASSERT(n < MAX_N);
        data[n++] = val;
    }

    void insert(T * pos, T val) {
        GGML_ASSERT(n < MAX_N);
        for (T * p = data + n; p > pos; --p) {
            *p = *(p - 1);
        }
        *pos = val;
        n++;
    }
};

static inline bool htp_is_hadamard(const struct ggml_tensor * t) {
    return t && t->op == GGML_OP_MUL_MAT && ggml_get_op_params_i32(t, 1) == GGML_HINT_SRC0_IS_HADAMARD;
}

struct htp_opnode {
    const ggml_tensor * node   { nullptr };
    htp_op_code         opcode { HTP_OP_INVALID };
    int32_t             params[HTP_OP_MAX_PARAMS] {0};
    int32_t             kernel_params[HTP_OP_MAX_KERN_PARAMS] {0};

    std::vector<std::shared_ptr<ggml_tensor>>          dummy;

    htp_opvec<const ggml_tensor *, HTP_OP_MAX_FUSED>   fused;
    htp_opvec<const ggml_tensor *, HTP_OP_MAX_INPUTS>  inputs;
    htp_opvec<const ggml_tensor *, HTP_OP_MAX_OUTPUTS> outputs;

    int n_active_src(const ggml_tensor * t) const {
        if (!t) return 0;
        for (int i = GGML_MAX_SRC - 1; i >= 0; i--) {
            if (t->src[i]) {
                return i + 1;
            }
        }
        return 0;
    }

    void init(const ggml_tensor * node) {
        static_assert(sizeof(this->params) == sizeof(this->node->op_params), "params size mismatch");

        this->node = node;
        if (!node) return;

        memcpy(this->params, this->node->op_params, sizeof(this->params));

        // Build inputs (preserving optional nullptrs)
        if (htp_is_hadamard(this->node)) {
            this->inputs.push_back(this->node->src[1]);
        } else {
            int n_inputs = n_active_src(this->node);
            for (int i = 0; i < n_inputs; i++) {
                this->inputs.push_back(this->node->src[i]);
            }
        }

        // Build outputs
        this->outputs.push_back(this->dst());
    }

    htp_opnode(htp_op_code opcode = HTP_OP_INVALID, const ggml_tensor * node = nullptr) : opcode(opcode) {
        init(node);
    }

    htp_opnode(const ggml_tensor * node) : opcode(HTP_OP_INVALID) {
        init(node);
    }

    const ggml_tensor * src0() const { return node->src[0]; }
    const ggml_tensor * src1() const { return node->src[1]; }
    const ggml_tensor * dst()  const { return outputs.empty() ? node : outputs.back(); }

    ggml_tensor * add_dummy(const ggml_tensor & t) {
        dummy.push_back(std::make_shared<ggml_tensor>(t));
        return dummy.back().get();
    }

    const htp_opvec<const ggml_tensor *, HTP_OP_MAX_INPUTS> & get_inputs() const {
        return inputs;
    }

    const htp_opvec<const ggml_tensor *, HTP_OP_MAX_OUTPUTS> & get_outputs() const {
        return outputs;
    }

    template <typename T>
    T * kparams() {
        static_assert(sizeof(T) <= sizeof(kernel_params), "kparams size exceeds HTP_OP_MAX_KERN_PARAMS");
        static_assert(alignof(T) <= alignof(int32_t), "kparams alignment exceeds int32_t");
        return reinterpret_cast<T *>(kernel_params);
    }

    template <typename T>
    const T * kparams() const {
        static_assert(sizeof(T) <= sizeof(kernel_params), "kparams size exceeds HTP_OP_MAX_KERN_PARAMS");
        static_assert(alignof(T) <= alignof(int32_t), "kparams alignment exceeds int32_t");
        return reinterpret_cast<const T *>(kernel_params);
    }

    template <typename T>
    void set_kparams(const T & kp) {
        static_assert(sizeof(T) <= sizeof(kernel_params), "kparams size exceeds HTP_OP_MAX_KERN_PARAMS");
        static_assert(alignof(T) <= alignof(int32_t), "kparams alignment exceeds int32_t");
        memcpy(kernel_params, &kp, sizeof(T));
    }

    std::string name() const;
};

enum htp_opformat_flags {
    HTP_OPFMT_OP      = (1 << 0),
    HTP_OPFMT_NAMES   = (1 << 1),
    HTP_OPFMT_DIMS    = (1 << 2),
    HTP_OPFMT_TYPES   = (1 << 3),
    HTP_OPFMT_STRIDES = (1 << 4),
    HTP_OPFMT_BUFFS   = (1 << 5),
    HTP_OPFMT_KPARAMS = (1 << 6),

<<<<<<< HEAD
    int format_tensor_dims(char * str, size_t max_size, const struct ggml_tensor * t) {
        if (!t) {
            return snprintf(str, max_size, "NONE");
        }
        if (t->ne[2] == 1 && t->ne[3] == 1) {
            return snprintf(str, max_size, "%d:%d", (int) t->ne[0], (int) t->ne[1]);
        } else {
            return snprintf(str, max_size, "%d:%d:%d:%d", (int) t->ne[0], (int) t->ne[1], (int) t->ne[2], (int) t->ne[3]);
        }
    }

    void format_op_dims(char * str, size_t max_size, const htp_opnode & node) {
        char * p = str;
        char * p_end = str + max_size;
        auto inputs = node.get_inputs();

        if (!inputs.empty()) {
            p += std::min((size_t)format_tensor_dims(p, p_end - p, inputs[0]), (size_t)(p_end - p));

            for (size_t i = 1; i < inputs.size(); i++) {
                if (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
                }
                if (p < p_end) {
                    p += std::min((size_t)format_tensor_dims(p, p_end - p, inputs[i]), (size_t)(p_end - p));
                }
            }

            if (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
            }
        }

        char self[64];
        format_tensor_dims(self, sizeof(self), node.dst());
        if (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, "%s", self), (size_t)(p_end - p));
        }
    }

    int format_tensor_strides(char * str, size_t max_size, const struct ggml_tensor * t) {
        if (!t) {
            return snprintf(str, max_size, "NONE");
        }
        const char * c = ggml_is_contiguous(t) ? "" : "!";

        if (t->ne[2] == 1 && t->ne[3] == 1) {
            return snprintf(str, max_size, "%zu:%zu%s", (size_t) t->nb[0], (size_t) t->nb[1], c);
        } else {
            return snprintf(str, max_size, "%zu:%zu:%zu:%zu%s", (size_t) t->nb[0], (size_t) t->nb[1], (size_t) t->nb[2], (size_t) t->nb[3], c);
        }
    }

    void format_op_strides(char * str, size_t max_size, const htp_opnode & node) {
        char * p = str;
        char * p_end = str + max_size;
        auto inputs = node.get_inputs();

        if (!inputs.empty()) {
            p += std::min((size_t)format_tensor_strides(p, p_end - p, inputs[0]), (size_t)(p_end - p));

            for (size_t i = 1; i < inputs.size(); i++) {
                if (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
                }
                if (p < p_end) {
                    p += std::min((size_t)format_tensor_strides(p, p_end - p, inputs[i]), (size_t)(p_end - p));
                }
            }

            if (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
            }
        }

        char self[64];
        format_tensor_strides(self, sizeof(self), node.dst());
        if (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, "%s", self), (size_t)(p_end - p));
        }
    }

    void format_op_types(char * str, size_t max_size, const htp_opnode & node) {
        char * p = str;
        char * p_end = str + max_size;
        auto inputs = node.get_inputs();

        if (!inputs.empty()) {
            if (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, "%s", inputs[0] ? ggml_type_name(inputs[0]->type) : "NONE"), (size_t)(p_end - p));
            }

            for (size_t i = 1; i < inputs.size(); i++) {
                if (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
                }
                if (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, "%s", inputs[i] ? ggml_type_name(inputs[i]->type) : "NONE"), (size_t)(p_end - p));
                }
            }

            if (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
            }
        }

        if (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, "%s", ggml_type_name(node.dst()->type)), (size_t)(p_end - p));
        }
    }

    const char * tensor_buff_name(const struct ggml_tensor * t) {
        if (t && t->buffer) {
            return ggml_backend_buffer_name(t->buffer);
        }
        return "NONE";
    }

    void format_op_buffs(char * str, size_t max_size, const htp_opnode & node) {
        char * p = str;
        char * p_end = str + max_size;
        auto inputs = node.get_inputs();

        if (!inputs.empty()) {
            if (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, "%s", tensor_buff_name(inputs[0])), (size_t)(p_end - p));
            }

            for (size_t i = 1; i < inputs.size(); i++) {
                if (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
                }
                if (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, "%s", tensor_buff_name(inputs[i])), (size_t)(p_end - p));
                }
            }

            if (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
            }
        }

        if (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, "%s", tensor_buff_name(node.dst())), (size_t)(p_end - p));
        }
    }

    void format_op_names(char * str, size_t max_size, const htp_opnode & node) {
        char * p = str;
        char * p_end = str + max_size;
        auto inputs = node.get_inputs();

        if (!inputs.empty()) {
            if (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, "%s", inputs[0] ? inputs[0]->name : "NONE"), (size_t)(p_end - p));
            }

            for (size_t i = 1; i < inputs.size(); i++) {
                if (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
                }
                if (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, "%s", inputs[i] ? inputs[i]->name : "NONE"), (size_t)(p_end - p));
                }
            }

            if (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
            }
        }

        if (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, "%s", node.dst()->name), (size_t)(p_end - p));
        }
    }
    void format_kernel_params(char * str, size_t max_size, const htp_opnode & node) {
        if (node.opcode == HTP_OP_MUL_MAT || node.opcode == HTP_OP_MUL_MAT_ID ||
            node.opcode == HTP_OP_MUL_MAT_NX || node.opcode == HTP_OP_MUL_MAT_ADD) {
            const auto * kparams = (const struct htp_mm_kernel_params *) node.kernel_params;
            const char * path = "unknown";
            int32_t type = kparams->kernel_type;
            if (type == HTP_MM_KERNEL_HMX_2D || type == HTP_MM_KERNEL_HMX_F16_BATCHED) {
                path = "hmx-tiled";
            } else if (type == HTP_MM_KERNEL_HVX_F16_F16_VTCM || type == HTP_MM_KERNEL_HVX_F32_F32_VTCM ||
                       type == HTP_MM_KERNEL_HVX_QUANT_ROW    || type == HTP_MM_KERNEL_HVX_QUANT_BLOCK) {
                path = "hvx-tiled";
            } else if (type == HTP_MM_KERNEL_HVX_F16_F16_DDR  || type == HTP_MM_KERNEL_HVX_F16_F32_DDR ||
                       type == HTP_MM_KERNEL_HVX_F32_F32_DDR  || type == HTP_MM_KERNEL_HVX_F32_F16_DDR ||
                       type == HTP_MM_KERNEL_HVX_QUANT_ROW_FLAT) {
                path = "hvx-flat";
            }
            snprintf(str, max_size, "%s vtcm %d", path, (int) kparams->vtcm_size);
        } else if (node.opcode == HTP_OP_FLASH_ATTN_EXT) {
            const auto * kparams = (const struct htp_fa_kernel_params *) node.kernel_params;
            const char * path = "unknown";
            int32_t type = kparams->kernel_type;
            if (type == HTP_FA_KERNEL_HMX) {
                path = kparams->u.hmx.pipeline ? "hmx-pipe" : "hmx-seq";
            } else if (type == HTP_FA_KERNEL_HVX) {
                path = "hvx";
            }
            snprintf(str, max_size, "%s vtcm %d", path, (int) kparams->vtcm_size);
        } else if (htp_op_is_unary(node.opcode)) {
            const auto * kparams = (const struct htp_unary_kernel_params *) node.kernel_params;
            snprintf(str, max_size, "%s vtcm %d", kparams->col_tile ? "wide-row" : "row-block", (int) kparams->vtcm_size);
        } else {
            snprintf(str, max_size, "----");
        }
    }

    void format(const htp_opnode & node) {
        format_op_dims(dims, sizeof(dims), node);
        format_op_strides(strides, sizeof(strides), node);
        format_op_types(types, sizeof(types), node);
        format_op_buffs(buffs, sizeof(buffs), node);
        format_op_names(names, sizeof(names), node);
        format_kernel_params(kparams, sizeof(kparams), node);
    }

    htp_opformat() {
        strides[0] = '\0';
        dims[0]    = '\0';
        types[0]   = '\0';
        buffs[0]   = '\0';
        names[0]   = '\0';
        kparams[0] = '\0';
    }
    htp_opformat(const htp_opnode & node) { format(node); }
=======
    HTP_OPFMT_EXEC    = HTP_OPFMT_OP | HTP_OPFMT_NAMES | HTP_OPFMT_DIMS | HTP_OPFMT_TYPES | HTP_OPFMT_STRIDES | HTP_OPFMT_BUFFS | HTP_OPFMT_KPARAMS,
    HTP_OPFMT_SUPP    = HTP_OPFMT_OP | HTP_OPFMT_NAMES | HTP_OPFMT_DIMS | HTP_OPFMT_TYPES | HTP_OPFMT_STRIDES | HTP_OPFMT_BUFFS,
    HTP_OPFMT_PROF    = HTP_OPFMT_OP | HTP_OPFMT_NAMES | HTP_OPFMT_DIMS | HTP_OPFMT_TYPES | HTP_OPFMT_STRIDES | HTP_OPFMT_KPARAMS,
    HTP_OPFMT_TENSOR  = HTP_OPFMT_NAMES | HTP_OPFMT_DIMS | HTP_OPFMT_TYPES | HTP_OPFMT_STRIDES,
>>>>>>> ggml-org/master
};

std::string htp_format_op(const htp_opnode & node, uint32_t flags = HTP_OPFMT_EXEC);
std::string htp_format_tensor(const ggml_tensor * t, uint32_t flags = HTP_OPFMT_TENSOR);

#endif // HTP_OPNODE_H
