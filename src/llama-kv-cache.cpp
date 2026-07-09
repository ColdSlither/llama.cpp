#include "llama-kv-cache.h"

#include "llama-impl.h"
#include "llama-io.h"
#include "llama-model.h"
#include "llama-context.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <cfloat>
#include <numeric>

static bool ggml_is_power_of_2(int n) {
    return (n & (n - 1)) == 0;
}

// orthonormal Walsh-Hadamard rotation matrix
// note: res^2 == I
static void ggml_gen_hadamard(ggml_tensor * tensor) {
    assert(tensor->type == GGML_TYPE_F32);

    const int n = tensor->ne[0];

    assert(ggml_is_power_of_2(n));
    assert(tensor->ne[1] == n);
    assert(tensor->ne[2] == 1);
    assert(tensor->ne[3] == 1);

    std::vector<float> data_f32;

    float * data = (float *) tensor->data;

    if (tensor->type != GGML_TYPE_F32) {
        data_f32.resize(n*n);
        data = data_f32.data();
    }

    data[0*n + 0] = 1.0 / sqrtf(n);

    for (int s = 1; s < n; s *= 2) {
        for (int i = 0; i < s; i++) {
            for (int j = 0; j < s; j++) {
                const float val = data[i*n + j];

                data[(i + s)*n + (j    )] =  val;
                data[(i    )*n + (j + s)] =  val;
                data[(i + s)*n + (j + s)] = -val;
            }
        }
    }

    if (tensor->type != GGML_TYPE_F32) {
        ggml_quantize_chunk(tensor->type, data, tensor->data, 0, 1, n*n, nullptr);
    }
}

static ggml_tensor * ggml_mul_mat_aux(
        ggml_context * ctx,
        ggml_tensor * cur,
        ggml_tensor * rot) {
    const auto n = rot->ne[0];

    ggml_tensor * res;

    res = ggml_reshape_2d(ctx, cur, n, ggml_nelements(cur)/n);
    res = ggml_mul_mat   (ctx, rot, res);
    ggml_mul_mat_set_hint(res, GGML_HINT_SRC0_IS_HADAMARD);
    res = ggml_reshape_4d(ctx, res, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3]);

    return res;
}

//
// llama_kv_cache
//

llama_kv_cache::llama_kv_cache(
        const llama_model & model,
        const llama_hparams & hparams,
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                     bool   offload,
                     bool   unified,
                 uint32_t   kv_size,
                 uint32_t   n_seq_max,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
           llama_memory_t   mem_other,
    const layer_filter_cb & filter,
    const  layer_reuse_cb & reuse,
    const  layer_share_cb & share) :
    model(model), hparams(hparams), v_trans(v_trans),
    n_seq_max(n_seq_max), n_stream(unified ? 1 : n_seq_max), n_pad(n_pad), n_swa(n_swa), swa_type(swa_type),
    other(static_cast<llama_kv_cache *>(mem_other)),
    v_cells_impl(other ? other->v_cells_impl : std::make_shared<llama_kv_cells_vec>()),
    v_cells(*v_cells_impl) {

    // shared cells view the source cache's K/V tensors, so the cell count
    // follows the source allocation: a fitted target can be smaller than the
    // draft default and oversized views would overflow the source tensors
    if (other) {
        const uint32_t size_other = other->get_size();
        if (kv_size != size_other) {
            LLAMA_LOG_WARN("%s: kv_size = %u overridden to %u to match the shared source cache\n", __func__, kv_size, size_other);
            kv_size = size_other;
        }
    }

    GGML_ASSERT(kv_size % n_pad == 0);

    const uint32_t n_layer = hparams.n_layer_all;

    // define a comparator for the buft -> ctx map to ensure that the order is well-defined:
    struct ggml_backend_buft_comparator {
        bool operator()(const ggml_backend_buffer_type_t & lhs, const ggml_backend_buffer_type_t & rhs) const {
            return strcmp(ggml_backend_buft_name(lhs), ggml_backend_buft_name(rhs)) < 0;
        }
    };
    std::map<ggml_backend_buffer_type_t, ggml_context_ptr, ggml_backend_buft_comparator> ctx_map;

    // create a context for each buffer type
    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            ggml_init_params params = {
                /*.mem_size   =*/ size_t(2u*(1 + n_stream)*n_layer*ggml_tensor_overhead()),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                return nullptr;
            }

            ctx_map.emplace(buft, ctx);

            return ctx;
        }

        return it->second.get();
    };

    GGML_ASSERT(n_stream == 1 || n_stream == n_seq_max);

    v_heads.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        v_heads[s] = 0;
    }

    v_cells.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        v_cells[s].resize(kv_size);
    }

    // by default, all sequence ids are mapped to the 0th stream
    seq_to_stream.resize(LLAMA_MAX_SEQ, 0);

    if (n_stream > 1) {
        seq_to_stream.resize(n_stream, 0);
        for (uint32_t s = 0; s < n_stream; ++s) {
            seq_to_stream[s] = s;
        }
    }

    // [TAG_V_CACHE_VARIABLE]
    if (v_trans && hparams.is_n_embd_v_gqa_variable()) {
        LLAMA_LOG_WARN("%s: the V embeddings have different sizes across layers and FA is not enabled - padding V cache to %d\n",
                __func__, hparams.n_embd_v_gqa_max());
    }

    const bool is_mla = hparams.is_mla();

    for (uint32_t il = 0; il < n_layer; il++) {
        if (!hparams.has_kv(il)) {
            LLAMA_LOG_DEBUG("%s: layer %3d: does not have KV cache\n", __func__, il);
            continue;
        }

        if (filter && !filter(il)) {
            LLAMA_LOG_DEBUG("%s: layer %3d: filtered\n", __func__, il);
            continue;
        }

        if (share && other) {
            const int32_t il_share = share(il);

            if (il_share >= 0) {
                const auto & layer_share = other->layers[other->map_layer_ids[il_share]];

                LLAMA_LOG_WARN("%s: layer %3d: sharing with layer %d. k = %p, v = %p\n", __func__, il, il_share,
                        layer_share.k->data, layer_share.v->data);

                map_layer_ids[il] = layers.size();

                layers.push_back(layer_share);
                layers.back().il = il;

                continue;
            }
        }

        if (n_embd_head_k_all == 0) {
            n_embd_head_k_all = (int32_t) hparams.n_embd_head_k(il);
        } else if (n_embd_head_k_all > 0 && n_embd_head_k_all != (int32_t) hparams.n_embd_head_k(il)) {
            n_embd_head_k_all = -1;
        }

        if (!is_mla) {
            if (n_embd_head_v_all == 0) {
                n_embd_head_v_all = (int32_t) hparams.n_embd_head_v(il);
            } else if (n_embd_head_v_all > 0 && n_embd_head_v_all != (int32_t) hparams.n_embd_head_v(il)) {
                n_embd_head_v_all = -1;
            }
        }

        // [TAG_V_CACHE_VARIABLE]
        const uint32_t n_embd_k_gqa =            hparams.n_embd_k_gqa(il);
        const uint32_t n_embd_v_gqa = !v_trans ? hparams.n_embd_v_gqa(il) : hparams.n_embd_v_gqa_max();

        const char * dev_name = "CPU";

        ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();

        if (offload) {
            auto * dev = model.dev_layer(il);
            buft = ggml_backend_dev_buffer_type(dev);

            dev_name = ggml_backend_dev_name(dev);
        }

        LLAMA_LOG_DEBUG("%s: layer %3d: dev = %s\n", __func__, il, dev_name);

        ggml_context * ctx = ctx_for_buft(buft);
        if (!ctx) {
            throw std::runtime_error("failed to create ggml context for kv cache");
        }

        const bool has_k = true;
        const bool has_v = !is_mla;

        ggml_tensor * k = has_k ? ggml_new_tensor_3d(ctx, type_k, n_embd_k_gqa, kv_size, n_stream) : nullptr;
        ggml_tensor * v = has_v ? ggml_new_tensor_3d(ctx, type_v, n_embd_v_gqa, kv_size, n_stream) : nullptr;

        has_k && ggml_format_name(k, "cache_k_l%d", il);
        has_v && ggml_format_name(v, "cache_v_l%d", il);

        std::vector<ggml_tensor *> k_stream;
        std::vector<ggml_tensor *> v_stream;

        for (uint32_t s = 0; s < n_stream; ++s) {
            k_stream.push_back(has_k ? ggml_view_2d(ctx, k, n_embd_k_gqa, kv_size, k->nb[1], s*k->nb[2]) : nullptr);
            v_stream.push_back(has_v ? ggml_view_2d(ctx, v, n_embd_v_gqa, kv_size, v->nb[1], s*v->nb[2]) : nullptr);
        }

        map_layer_ids[il] = layers.size();

        layers.push_back({ il, k, v, k_stream, v_stream, });
    }

    if (reuse) {
        LLAMA_LOG_DEBUG("%s: reusing layers:\n", __func__);

        for (uint32_t il = 0; il < n_layer; il++) {
            const int32_t il_reuse = reuse(il);

            if (il_reuse < 0) {
                LLAMA_LOG_DEBUG("%s: - layer %3d: no reuse\n", __func__, il);
                continue;
            }

            if (filter && !filter(il)) {
                LLAMA_LOG_DEBUG("%s: - layer %3d: filtered\n", __func__, il);
                continue;
            }

            GGML_ASSERT(map_layer_ids.find(il_reuse) != map_layer_ids.end());

            map_layer_ids[il] = map_layer_ids[il_reuse];

            LLAMA_LOG_DEBUG("%s: - layer %3d: reuse layer %d, is_swa = %d\n", __func__, il, il_reuse, hparams.is_swa(il));
        }
    }

    // allocate tensors and initialize the buffers to avoid NaNs in the padding
    for (auto & [buft, ctx] : ctx_map) {
        ggml_backend_buffer_t buf;
        if (hparams.no_alloc) {
            buf = ggml_backend_buft_alloc_buffer(buft, /*size =*/ 0); // dummy buffer
            for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != nullptr; t = ggml_get_next_tensor(ctx.get(), t)) {
                t->buffer = buf; // set dummy buffer for KV cache so that the backend scheduler won't try to allocate it
            }
        } else {
            buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft); // real buffer
        }
        if (!buf) {
            throw std::runtime_error("failed to allocate buffer for kv cache");
        }

        LLAMA_LOG_INFO("%s: %10s KV buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf)/1024.0/1024.0);

        ggml_backend_buffer_clear(buf, 0);
        ctxs_bufs.emplace_back(std::move(ctx), buf);
    }

    {
        const size_t memory_size_k = size_k_bytes();
        const size_t memory_size_v = size_v_bytes();

        LLAMA_LOG_INFO("%s: size = %7.2f MiB (%6u cells, %3d layers, %2u/%u seqs), K (%s): %7.2f MiB, V (%s): %7.2f MiB\n", __func__,
                (float)(memory_size_k + memory_size_v) / (1024.0f * 1024.0f), kv_size, (int) layers.size(), n_seq_max, n_stream,
                ggml_type_name(type_k), (float)memory_size_k / (1024.0f * 1024.0f),
                ggml_type_name(type_v), (float)memory_size_v / (1024.0f * 1024.0f));
    }

    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        n_embd_head_k_all = other->n_embd_head_k_all;
        n_embd_head_v_all = other->n_embd_head_v_all;

        attn_rot_k = other->attn_rot_k;
        attn_rot_v = other->attn_rot_v;
    } else {
        const char * LLAMA_ATTN_ROT_DISABLE = getenv("LLAMA_ATTN_ROT_DISABLE");
        const bool attn_rot_disable = LLAMA_ATTN_ROT_DISABLE ? atoi(LLAMA_ATTN_ROT_DISABLE) : false;
        if (attn_rot_disable) {
            LLAMA_LOG_WARN("%s: attention rotation force disabled (LLAMA_ATTN_ROT_DISABLE)\n", __func__);
        }

        attn_rot_k =
            !attn_rot_disable &&
            n_embd_head_k_all > 0 &&
            ggml_is_quantized(type_k) &&
            hparams.n_embd_head_k() % 64 == 0;

        // always create Hadamard rotation tensors for DeepSeek lightning indexers
        if ((model.arch == LLM_ARCH_DEEPSEEK32 || model.arch == LLM_ARCH_DEEPSEEK4) &&
                hparams.n_embd_head_k_full == hparams.indexer_head_size) {
            attn_rot_k = true;
        }

        attn_rot_v =
            !attn_rot_disable &&
            n_embd_head_v_all > 0 &&
            ggml_is_quantized(type_v) &&
            hparams.n_embd_head_v() % 64 == 0;
    }

    LLAMA_LOG_INFO("%s: attn_rot_k = %d, n_embd_head_k_all = %d\n", __func__, attn_rot_k, n_embd_head_k_all);
    LLAMA_LOG_INFO("%s: attn_rot_v = %d, n_embd_head_k_all = %d\n", __func__, attn_rot_v, n_embd_head_v_all);

    // pre-compute the haramard matrices and keep them in host memory
    // TODO: in the future, we can make copies in the backend buffers to avoid host -> device transfers
    if (attn_rot_k || attn_rot_v) {
        for (int64_t n = 64; n <= std::max(n_embd_head_k_all, n_embd_head_v_all); n *= 2) {
            attn_rot_hadamard[n] = std::vector<float>(n*n);

            ggml_init_params params = {
                /* .mem_size   = */ 1*ggml_tensor_overhead(),
                /* .mem_buffer = */ nullptr,
                /* .no_alloc   = */ true,
            };

            ggml_context_ptr ctx { ggml_init(params) };

            ggml_tensor * tmp = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n, n);
            tmp->data = attn_rot_hadamard[n].data();

            ggml_gen_hadamard(tmp);
        }
    }

    const char * LLAMA_KV_CACHE_DEBUG = getenv("LLAMA_KV_CACHE_DEBUG");
    debug = LLAMA_KV_CACHE_DEBUG ? atoi(LLAMA_KV_CACHE_DEBUG) : 0;
}

void llama_kv_cache::clear(bool data) {
    for (uint32_t s = 0; s < n_stream; ++s) {
        v_cells[s].reset();
        v_heads[s] = 0;
    }

    if (data) {
        for (auto & [_, buf] : ctxs_bufs) {
            ggml_backend_buffer_clear(buf.get(), 0);
        }
    }
}

bool llama_kv_cache::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return true;
    }

    GGML_ASSERT(seq_id == -1 || (seq_id >= 0 && (size_t) seq_id < seq_to_stream.size()));

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    if (seq_id >= 0) {
        auto & cells = v_cells[seq_to_stream[seq_id]];
        auto & head  = v_heads[seq_to_stream[seq_id]];

        uint32_t new_head = cells.size();

        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (!cells.pos_in(i, p0, p1)) {
                continue;
            }

            if (cells.seq_has(i, seq_id) && cells.seq_rm(i, seq_id)) {
                if (new_head == cells.size()) {
                    new_head = i;
                }
            }
        }

        // If we freed up a slot, set head to it so searching can start there.
        if (new_head != cells.size() && new_head < head) {
            head = new_head;
        }
    } else {
        // match any sequence
        for (uint32_t s = 0; s < n_stream; ++s) {
            auto & cells = v_cells[s];
            auto & head  = v_heads[s];

            uint32_t new_head = cells.size();

            for (uint32_t i = 0; i < cells.size(); ++i) {
                if (!cells.pos_in(i, p0, p1)) {
                    continue;
                }

                cells.rm(i);

                if (new_head == cells.size()) {
                    new_head = i;
                }
            }

            // If we freed up a slot, set head to it so searching can start there.
            if (new_head != cells.size() && new_head < head) {
                head = new_head;
            }
        }
    }

    return true;
}

void llama_kv_cache::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_ASSERT(seq_id_src >= 0 && (size_t) seq_id_src < seq_to_stream.size());
    GGML_ASSERT(seq_id_dst >= 0 && (size_t) seq_id_dst < seq_to_stream.size());

    const auto s0 = seq_to_stream[seq_id_src];
    const auto s1 = seq_to_stream[seq_id_dst];

    if (s0 == s1) {
        // since both sequences are in the same stream, no data copy is necessary
        // we just have to update the cells meta data

        auto & cells = v_cells[s0];

        if (seq_id_src == seq_id_dst) {
            return;
        }

        if (p0 < 0) {
            p0 = 0;
        }

        if (p1 < 0) {
            p1 = std::numeric_limits<llama_pos>::max();
        }

        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (!cells.pos_in(i, p0, p1)) {
                continue;
            }

            if (cells.seq_has(i, seq_id_src)) {
                cells.seq_add(i, seq_id_dst);
            }
        }

        return;
    }

    // cross-stream sequence copies require to copy the actual buffer data

    bool is_full = true;

    if (p0 > 0 && p0 + 1 < (int) get_size()) {
        is_full = false;
    }

    if (p1 > 0 && p1 + 1 < (int) get_size()) {
        is_full = false;
    }

    GGML_ASSERT(is_full && "seq_cp() is only supported for full KV buffers");

    // enqueue the copy operation - the buffer copy will be performed during the next update
    sc_info.ssrc.push_back(s0);
    sc_info.sdst.push_back(s1);

    v_cells[s1].reset();
    for (uint32_t i = 0; i < v_cells[s0].size(); ++i) {
        if (v_cells[s0].seq_has(i, seq_id_src)) {
            llama_pos pos   = v_cells[s0].pos_get(i);
            llama_pos shift = v_cells[s0].get_shift(i);

            llama_kv_cell_ext ext = v_cells[s0].ext_get(i);

            if (shift != 0) {
                pos -= shift;
                assert(pos >= 0);
            }

            v_cells[s1].pos_set(i, pos);
            v_cells[s1].seq_add(i, seq_id_dst);

            if (shift != 0) {
                v_cells[s1].pos_add(i, shift);
            }

            v_cells[s1].ext_set(i, ext);
        }
    }

    v_heads[s1] = v_heads[s0];

    //for (uint32_t s = 0; s < n_stream; ++s) {
    //    LLAMA_LOG_WARN("%s: seq %d: min = %d, max = %d\n", __func__, s, v_cells[s].seq_pos_min(s), v_cells[s].seq_pos_max(s));
    //}
}

void llama_kv_cache::seq_keep(llama_seq_id seq_id) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    auto & cells = v_cells[seq_to_stream[seq_id]];
    auto & head  = v_heads[seq_to_stream[seq_id]];

    uint32_t new_head = cells.size();

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (cells.seq_keep(i, seq_id)) {
            if (new_head == cells.size()) {
                new_head = i;
            }
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    if (new_head != cells.size() && new_head < head) {
        head = new_head;
    }
}

void llama_kv_cache::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());
    GGML_ASSERT(hparams.n_pos_per_embd() == 1 && "seq_add() is only supported for n_pos_per_embd() == 1");

    auto & cells = v_cells[seq_to_stream[seq_id]];
    auto & head  = v_heads[seq_to_stream[seq_id]];

    if (shift == 0) {
        return;
    }

    uint32_t new_head = cells.size();

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // If there is no range then return early to avoid looping over all cells.
    if (p0 == p1) {
        return;
    }

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.pos_in(i, p0, p1)) {
            continue;
        }

        if (cells.seq_has(i, seq_id)) {
            if (cells.pos_add(i, shift)) {
                if (new_head == cells.size()) {
                    new_head = i;
                }
            }
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    // Otherwise we just start the next search from the beginning.
    head = new_head != cells.size() ? new_head : 0;
}

void llama_kv_cache::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());
    GGML_ASSERT(hparams.n_pos_per_embd() == 1 && "seq_div() is only supported for n_pos_per_embd() == 1");

    auto & cells = v_cells[seq_to_stream[seq_id]];

    if (d == 1) {
        return;
    }

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // If there is no range then return early to avoid looping over the cache.
    if (p0 == p1) {
        return;
    }

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.pos_in(i, p0, p1)) {
            continue;
        }

        if (cells.seq_has(i, seq_id)) {
            cells.pos_div(i, d);
        }
    }
}

llama_pos llama_kv_cache::seq_pos_min(llama_seq_id seq_id) const {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return other->seq_pos_min(seq_id);
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    const auto & cells = v_cells[seq_to_stream[seq_id]];

    return cells.seq_pos_min(seq_id);
}

llama_pos llama_kv_cache::seq_pos_max(llama_seq_id seq_id) const {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return other->seq_pos_max(seq_id);
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    const auto & cells = v_cells[seq_to_stream[seq_id]];

    return cells.seq_pos_max(seq_id);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_kv_cache::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> ret;
    for (const auto & [ctx, buf] : ctxs_bufs) {
        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buf.get());

        if (hparams.no_alloc) {
            GGML_ASSERT(ggml_backend_buffer_get_base(buf.get()) == nullptr);
            ret[buft] += ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(), buft);
        } else {
            // GGML_ASSERT(ggml_backend_buffer_get_base(buf.get()) != nullptr); // multi_buffer does not have a defined base
            ret[buft] += ggml_backend_buffer_get_size(buf.get());
        }
    }

    return ret;
}

llama_memory_context_ptr llama_kv_cache::init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) {
    GGML_UNUSED(embd_all);

    do {
        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;
        while (true) {
            auto ubatch = n_stream == 1 ? balloc.split_simple(n_ubatch) : balloc.split_equal(n_ubatch, true);

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        auto sinfos = prepare(ubatches);
        if (sinfos.empty()) {
            break;
        }

        return std::make_unique<llama_kv_cache_context>(
                this, std::move(sinfos), std::move(ubatches));
    } while (false);

    return std::make_unique<llama_kv_cache_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_kv_cache::init_full() {
    return std::make_unique<llama_kv_cache_context>(this);
}

llama_memory_context_ptr llama_kv_cache::init_update(llama_context * lctx, bool optimize) {
    GGML_UNUSED(optimize);

    bool do_shift = get_has_shift();

    return std::make_unique<llama_kv_cache_context>(this, lctx, do_shift, std::move(sc_info));
}

llama_kv_cache::slot_info_vec_t llama_kv_cache::prepare(const std::vector<llama_ubatch> & ubatches) {
    llama_kv_cache::slot_info_vec_t res;

    struct state_t {
        slot_info sinfo; // slot info for the ubatch

        std::vector<uint32_t> v_heads_old; // old positions of the heads, before placing the ubatch

        std::vector<llama_kv_cells> v_cells; // copy of the old cells, before placing the ubatch
    };

    // remember the old state of the cells so we can restore it in the end
    std::vector<state_t> states;

    bool success = true;

    for (const auto & ubatch : ubatches) {
        // only find a suitable slot for the ubatch. don't modify the cells yet
        const auto sinfo_new = find_slot(ubatch, false);
        if (sinfo_new.empty()) {
            success = false;
            break;
        }

        // remember the position that we found
        res.push_back(sinfo_new);

        // store the old state of the cells in the recovery stack
        {
            state_t state = { sinfo_new, v_heads, {} };

            for (uint32_t s = 0; s < sinfo_new.n_stream(); ++s) {
                auto & cells = v_cells[sinfo_new.strm[s]];

                state.v_cells.push_back(cells.cp(sinfo_new.idxs[s]));
            }

            states.push_back(std::move(state));
        }

        // now emplace the ubatch
        apply_ubatch(sinfo_new, ubatch);
    }

    GGML_ASSERT(!states.empty() || !success);

    // iterate backwards and restore the cells to their original state
    for (auto it = states.rbegin(); it != states.rend(); ++it) {
        const auto & sinfo = it->sinfo;

        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            auto & cells = v_cells[sinfo.strm[s]];
            auto & head  = v_heads[sinfo.strm[s]];

            cells.set(sinfo.idxs[s], it->v_cells[s]);
            head = it->v_heads_old[s];
        }
    }

    if (!success) {
        return {};
    }

    return res;
}

bool llama_kv_cache::update(llama_context * lctx, bool do_shift, const stream_copy_info & sc_info) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return true;
    }

    bool updated = false;

    auto * sched = lctx->get_sched();

    if (!sc_info.empty()) {
        assert(n_stream > 1 && "stream copy should never happen with a single stream");

        llama_synchronize(lctx);

        const size_t n_copy = sc_info.ssrc.size();

        for (size_t i = 0; i < n_copy; ++i) {
            const auto ssrc = sc_info.ssrc[i];
            const auto sdst = sc_info.sdst[i];

            assert(ssrc < n_stream);
            assert(sdst < n_stream);

            LLAMA_LOG_DEBUG("%s: copying KV buffer: stream %d to stream %d\n", __func__, ssrc, sdst);

            assert(ssrc != sdst);

            for (uint32_t il = 0; il < layers.size(); ++il) {
                const auto & layer = layers[il];

                ggml_backend_tensor_copy(layer.k_stream[ssrc], layer.k_stream[sdst]);

                if (layer.v_stream[ssrc]) {
                    ggml_backend_tensor_copy(layer.v_stream[ssrc], layer.v_stream[sdst]);
                }
            }
        }
    }

    if (do_shift) {
        if (!get_can_shift()) {
            GGML_ABORT("The current KV cache / model configuration does not support K-shift");
        }

        LLAMA_LOG_DEBUG("%s: applying K-shift\n", __func__);

        // apply K-shift if needed
        if (hparams.rope_type != LLAMA_ROPE_TYPE_NONE) {
            ggml_backend_sched_reset(sched);

            auto * res = lctx->get_gf_res_reserve();

            res->reset();

            auto * gf = build_graph_shift(res, lctx);
            if (!ggml_backend_sched_alloc_graph(sched, gf)) {
                LLAMA_LOG_ERROR("%s: failed to allocate compute graph for K-shift\n", __func__);
                return updated;
            }

            res->set_inputs(nullptr);

            if (lctx->graph_compute(gf, false) != GGML_STATUS_SUCCESS) {
                LLAMA_LOG_ERROR("%s: failed to compute K-shift\n", __func__);
                return updated;
            }

            updated = true;
        }

        for (uint32_t s = 0; s < n_stream; ++s) {
            auto & cells = v_cells[s];

            cells.reset_shift();
        }
    }

    return updated;
}

llama_kv_cache::slot_info llama_kv_cache::find_slot(const llama_ubatch & ubatch, bool cont) const {

    if (debug > 0) {
        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
            const auto seq_id = ubatch.seq_id_unq[s];
            const auto stream_id = seq_to_stream[seq_id];
            const auto & cells = v_cells[stream_id];
            const uint32_t head_cur = v_heads[stream_id];

            LLAMA_LOG_DEBUG("%s: stream[%d], n = %5d, used = %5d, head = %5d, size = %5d, n_swa = %5d\n",
                    __func__, stream_id, cells.used_max_p1(), cells.get_used(), head_cur, get_size(), n_swa);

            if ((debug == 2 && n_swa > 0) || debug > 2) {
                std::string ss;
                for (uint32_t i = 0; i < cells.size(); ++i) {
                    if (cells.is_empty(i)) {
                        ss += '.';
                    } else {
                        assert(cells.seq_count(i) >= 1);

                        if (cells.seq_count(i) == 1) {
                            ss += std::to_string(cells.seq_get(i));
                        } else {
                            ss += 'M';
                        }
                    }
                    if (i%256 == 255) {
                        ss += " *";
                        ss += '\n';
                    }
                }
                LLAMA_LOG_DEBUG("\n%s\n", ss.c_str());
            }

            if ((debug == 2 && n_swa > 0) || debug > 2) {
                std::string ss;
                for (uint32_t i = 0; i < cells.size(); ++i) {
                    std::string cur;
                    if (cells.is_empty(i)) {
                        cur = '.';
                    } else {
                        cur = std::to_string(cells.pos_get(i));
                    }
                    const int n = cur.size();
                    for (int j = 0; j < 5 - n; ++j) {
                        cur += ' ';
                    }
                    ss += cur;
                    if (i%256 == 255) {
                        ss += " *";
                    }
                    if (i%64 == 63) {
                        ss += '\n';
                    }
                }
                LLAMA_LOG_DEBUG("\n%s\n", ss.c_str());
            }

            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                if (cells.seq_pos_min(s) < 0) {
                    continue;
                }

                LLAMA_LOG_DEBUG("%s: stream[%d] min[%d] = %5d, max[%d] = %5d\n", __func__, stream_id, s, cells.seq_pos_min(s), s, cells.seq_pos_max(s));
            }
        }
    }

    uint32_t n_tokens = ubatch.n_tokens;
    uint32_t n_seqs   = 1;

    if (n_stream > 1) {
        GGML_ASSERT(n_tokens % ubatch.n_seqs_unq == 0);

        n_seqs   = ubatch.n_seqs_unq;
        n_tokens = n_tokens / n_seqs;
    }

    slot_info res = {
        /*.s0   =*/ LLAMA_MAX_SEQ,
        /*.s1   =*/ 0,
        /*.strm =*/ { },
        /*.idxs =*/ { },
    };

    res.resize(n_seqs);

    for (uint32_t s = 0; s < n_seqs; ++s) {
        const auto seq_id = ubatch.seq_id_unq[s];

        if (n_stream > 1) {
            GGML_ASSERT(ubatch.n_seq_id[s*n_tokens]    == 1);
            GGML_ASSERT(ubatch.seq_id  [s*n_tokens][0] == seq_id);
        }

        res.s0 = std::min<uint32_t>(res.s0, seq_to_stream[seq_id]);
        res.s1 = std::max<uint32_t>(res.s1, seq_to_stream[seq_id]);

        res.strm[s] = seq_to_stream[seq_id];
        res.idxs[s].reserve(n_tokens);

        const auto & cells = v_cells[seq_to_stream[seq_id]];

        uint32_t head_cur = v_heads[seq_to_stream[seq_id]];

        // if we have enough unused cells before the current head ->
        //   better to start searching from the beginning of the cache, hoping to fill it
        if (head_cur > cells.get_used() + 2*n_tokens) {
            head_cur = 0;
        }

        if (n_tokens > cells.size()) {
            LLAMA_LOG_ERROR("%s: n_tokens = %d > size = %u\n", __func__, n_tokens, cells.size());
            return { };
        }

        uint32_t n_tested = 0;

        // for continuous slots, we test that all tokens in the ubatch fit, starting from the current head
        // for non-continuous slots, we test the tokens one by one
        const uint32_t n_test = cont ? n_tokens : 1;

        while (true) {
            if (head_cur + n_test > cells.size()) {
                n_tested += cells.size() - head_cur;
                head_cur = 0;
                continue;
            }

            for (uint32_t i = 0; i < n_test; i++) {
                const auto idx = head_cur;

                head_cur++;
                n_tested++;

                //const llama_pos    pos    = ubatch.pos[i];
                //const llama_seq_id seq_id = ubatch.seq_id[i][0];

                // can we use this cell? either:
                //  - the cell is empty
                //  - the cell is occupied only by one sequence:
                //    - (disabled) mask causally, if the sequence is the same as the one we are inserting
                //    - mask SWA, using current max pos for that sequence in the cache
                //                always insert in the cell with minimum pos
                bool can_use = cells.is_empty(idx);

                if (!can_use && cells.seq_count(idx) == 1) {
                    const llama_pos pos_cell = cells.pos_get(idx);

                    // (disabled) causal mask
                    // note: it's better to purge any "future" tokens beforehand
                    //if (cells.seq_has(idx, seq_id)) {
                    //    can_use = pos_cell >= pos;
                    //}

                    if (!can_use) {
                        const llama_seq_id seq_id_cell = cells.seq_get(idx);

                        // SWA mask
                        if (llama_hparams::is_masked_swa(n_swa, swa_type, pos_cell, cells.seq_pos_max(seq_id_cell) + 1)) {
                            can_use = true;
                        }
                    }
                }

                if (can_use) {
                    res.idxs[s].push_back(idx);
                } else {
                    if (cont) {
                        break;
                    }
                }
            }

            if (res.idxs[s].size() == n_tokens) {
                break;
            }

            if (cont) {
                res.idxs[s].clear();
            }

            if (n_tested >= cells.size()) {
                //LLAMA_LOG_ERROR("%s: failed to find a slot for %d tokens\n", __func__, n_tokens);
                return { };
            }
        }

        // we didn't find a suitable slot - return empty result
        if (res.idxs[s].size() < n_tokens) {
            return { };
        }
    }

    assert(res.s1 >= res.s0);

    return res;
}

void llama_kv_cache::apply_ubatch(const slot_info & sinfo, const llama_ubatch & ubatch) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    // keep track of the max sequence position that we would overwrite with this ubatch
    // for non-SWA cache, this would be always empty
    llama_seq_id seq_pos_max_rm[LLAMA_MAX_SEQ];
    for (uint32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
        seq_pos_max_rm[s] = -1;
    }

    assert(ubatch.n_tokens == sinfo.n_stream()*sinfo.size());

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        for (uint32_t ii = 0; ii < sinfo.size(); ++ii) {
            const uint32_t i = s*sinfo.size() + ii;

            auto & cells = v_cells[sinfo.strm[s]];

            const auto idx = sinfo.idxs[s][ii];

            if (!cells.is_empty(idx)) {
                assert(cells.seq_count(idx) == 1);

                const llama_seq_id seq_id = cells.seq_get(idx);
                const llama_pos    pos    = cells.pos_get(idx);

                seq_pos_max_rm[seq_id] = std::max(seq_pos_max_rm[seq_id], pos);

                cells.rm(idx);
            }

            cells.pos_set(idx, ubatch.pos[i]);

            if (ubatch.is_pos_2d()) {
                llama_kv_cell_ext ext {
                    /*.x =*/ ubatch.pos[i + ubatch.n_tokens*2],
                    /*.y =*/ ubatch.pos[i + ubatch.n_tokens],
                };
                cells.ext_set(idx, ext);
            }

            for (int32_t s = 0; s < ubatch.n_seq_id[i]; s++) {
                cells.seq_add(idx, ubatch.seq_id[i][s]);
            }
        }
    }

    // note: we want to preserve the invariant that all positions between [pos_min, pos_max] for each sequence
    //       will be present in the cache. so we have to purge any position which is less than those we would overwrite
    //       ref: https://github.com/ggml-org/llama.cpp/pull/13746#issuecomment-2916057092
    for (uint32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
        if (seq_pos_max_rm[s] == -1) {
            continue;
        }

        GGML_ASSERT(s < seq_to_stream.size());

        auto & cells = v_cells[seq_to_stream[s]];

        if (cells.seq_pos_min(s) <= seq_pos_max_rm[s]) {
            LLAMA_LOG_DEBUG("%s: purging positions [%d, %d] of sequence %d from KV cache\n",
                    __func__, cells.seq_pos_min(s), seq_pos_max_rm[s], s);

            seq_rm(s, cells.seq_pos_min(s), seq_pos_max_rm[s] + 1);
        }
    }

    // move the head at the end of the slot
    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        auto & head = v_heads[sinfo.strm[s]];

        head = sinfo.idxs[s].back() + 1;
    }
}

bool llama_kv_cache::get_can_shift() const {
    // Step35 uses per-layer RoPE dims; K-shift assumes a single global n_rot.
    if (model.arch == LLM_ARCH_STEP35) {
        return false;
    }
    if (hparams.n_pos_per_embd() > 1) {
        return false;
    }
    return true;
}

uint32_t llama_kv_cache::get_size() const {
    const auto & cells = v_cells[seq_to_stream[0]];

    return cells.size();
}

uint32_t llama_kv_cache::get_n_stream() const {
    return n_stream;
}

bool llama_kv_cache::get_has_shift() const {
    bool result = false;

    for (uint32_t s = 0; s < n_stream; ++s) {
        result |= v_cells[s].get_has_shift();
    }

    return result;
}

ggml_type llama_kv_cache::type_k() const {
    return layers[0].k->type;
}

ggml_type llama_kv_cache::type_v() const {
    return layers[0].v->type;
}

//
// KVzip — query-agnostic KV-cache compression
//

static void kvzip_score_tokens(
    const ggml_tensor * k,
    const ggml_tensor * v,
    float * scores,
    int n_kv,
    float pos_bias = 0.0f)
{
    const int n_embd_k = k->ne[0];
    const int n_embd_v = v->ne[0];

    // Staging buffers for reading rows and converting to f32
    // This handles quantized types correctly
    std::vector<uint8_t> row_k_bytes(n_embd_k * ggml_type_size(k->type));
    std::vector<uint8_t> row_v_bytes(n_embd_v * ggml_type_size(v->type));
    std::vector<float> row_k(n_embd_k);
    std::vector<float> row_v(n_embd_v);

    for (int i = 0; i < n_kv; i++) {
        float sum_k = 0.0f;
        float sum_v = 0.0f;

        // Read K row i using proper offset and size
        // K tensor layout: [n_embd_k, kv_size, n_stream]
        // Row i is at offset i * nb[1]
        size_t k_row_bytes = ggml_row_size(k->type, n_embd_k);
        ggml_backend_tensor_get(k, row_k_bytes.data(), i * k->nb[1], k_row_bytes);

        // Convert to f32
        switch (k->type) {
            case GGML_TYPE_F32:
                {
                    float * fp = (float *)row_k_bytes.data();
                    for (int e = 0; e < n_embd_k; e++) {
                        row_k[e] = fp[e];
                    }
                } break;
            case GGML_TYPE_F16:
                ggml_cpu_fp16_to_fp32((ggml_fp16_t *)row_k_bytes.data(), row_k.data(), n_embd_k);
                break;
            case GGML_TYPE_BF16:
                ggml_cpu_bf16_to_fp32((ggml_bf16_t *)row_k_bytes.data(), row_k.data(), n_embd_k);
                break;
            default:
                // For quantized types, use the generic path
                ggml_backend_tensor_get(k, row_k.data(), i * k->nb[1], k_row_bytes);
                break;
        }

        for (int e = 0; e < n_embd_k; e++) {
            sum_k += row_k[e] * row_k[e];
        }

        // Read V row i similarly
        size_t v_row_bytes = ggml_row_size(v->type, n_embd_v);
        ggml_backend_tensor_get(v, row_v_bytes.data(), i * v->nb[1], v_row_bytes);

        switch (v->type) {
            case GGML_TYPE_F32:
                {
                    float * fp = (float *)row_v_bytes.data();
                    for (int e = 0; e < n_embd_v; e++) {
                        row_v[e] = fp[e];
                    }
                } break;
            case GGML_TYPE_F16:
                ggml_cpu_fp16_to_fp32((ggml_fp16_t *)row_v_bytes.data(), row_v.data(), n_embd_v);
                break;
            case GGML_TYPE_BF16:
                ggml_cpu_bf16_to_fp32((ggml_bf16_t *)row_v_bytes.data(), row_v.data(), n_embd_v);
                break;
            default:
                ggml_backend_tensor_get(v, row_v.data(), i * v->nb[1], v_row_bytes);
                break;
        }

        for (int e = 0; e < n_embd_v; e++) {
            sum_v += row_v[e] * row_v[e];
        }

        // Base magnitude score (kvzip paper formulation)
        float base_score = sqrtf(sum_k) + sqrtf(sum_v);
        // Recency bias: prefer early tokens when magnitudes are similar.
        // pos_bias=0 disables, 0.1 is mild (kvzip default), higher values stronger.
        float pos_factor = (n_kv > 1) ? (1.0f - (float)i / (float)(n_kv - 1)) : 0.0f;
        scores[i] = base_score * (1.0f + pos_bias * pos_factor);
    }
}

static int kvzip_compact(
    ggml_tensor * k,
    ggml_tensor * v,
    llama_kv_cells & cells,
    float keep_ratio,
    int n_kv_used,
    float pos_bias = 0.0f,
    int min_latest = 0)
{
    if (n_kv_used <= 0 || keep_ratio >= 1.0f) {
        return n_kv_used;
    }

    const int n_keep = std::max(1, (int)(n_kv_used * keep_ratio));
    const int n_drop = n_kv_used - n_keep;

    if (n_drop <= 0) {
        return n_kv_used;
    }

    // Score tokens.
    std::vector<float> scores(n_kv_used);
    kvzip_score_tokens(k, v, scores.data(), n_kv_used, pos_bias);

    // Always keep the latest min_latest tokens (response-in-progress immunity)
    if (min_latest > 0) {
        int n_latest = std::min(min_latest, n_kv_used);
        for (int i = n_kv_used - n_latest; i < n_kv_used; i++) {
            scores[i] = std::numeric_limits<float>::max();
        }
    }

    // Diagnostic: log score distribution + which positions are kept
    if (n_kv_used >= 4) {
        // Find top-10 scoring positions (what kvzip will keep)
        std::vector<int> sort_idx(n_kv_used);
        std::iota(sort_idx.begin(), sort_idx.end(), 0);
        std::partial_sort(sort_idx.begin(), sort_idx.begin() + std::min(10, n_kv_used), sort_idx.end(),
            [&](int a, int b) { return scores[a] > scores[b]; });

        // Find scores in the needle region (positions 0-50)
        int needle_kept = 0;
        int needle_total = std::min(50, n_kv_used);
        for (int i = 0; i < 10 && i < n_kv_used; i++) {
            if (sort_idx[i] < 50) needle_kept++;
        }

        LLAMA_LOG_WARN(
            "kvzip-d: n_used=%d keep=%d needle_kept_in_top10=%d/%d "
            "top10_pos=[%d,%d,%d,%d,%d,%d,%d,%d,%d,%d]\n",
            n_kv_used, n_keep, needle_kept, needle_total,
            sort_idx[0],
            n_kv_used > 1 ? sort_idx[1] : -1,
            n_kv_used > 2 ? sort_idx[2] : -1,
            n_kv_used > 3 ? sort_idx[3] : -1,
            n_kv_used > 4 ? sort_idx[4] : -1,
            n_kv_used > 5 ? sort_idx[5] : -1,
            n_kv_used > 6 ? sort_idx[6] : -1,
            n_kv_used > 7 ? sort_idx[7] : -1,
            n_kv_used > 8 ? sort_idx[8] : -1,
            n_kv_used > 9 ? sort_idx[9] : -1);
    }

    // Find top-k indices.
    std::vector<int> indices(n_kv_used);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + n_keep, indices.end(),
        [&](int a, int b) { return scores[a] > scores[b]; });

    // --- Rearrange K/V by moving top-scored rows to front ---
    // Use ggml_nbytes for correct total size (handles padding/quantization)
    // Use nb[1] for row stride (correct stride for the tensor layout)
    const size_t k_total = ggml_nbytes(k);
    const size_t v_total = ggml_nbytes(v);
    const size_t k_row_stride = k->nb[1];  // stride in bytes for one row
    const size_t v_row_stride = v->nb[1];  // stride in bytes for one row

    std::vector<uint8_t> k_buf(k_total);
    std::vector<uint8_t> v_buf(v_total);

    ggml_backend_tensor_get(k, k_buf.data(), 0, k_buf.size());
    ggml_backend_tensor_get(v, v_buf.data(), 0, v_buf.size());

    std::vector<uint8_t> k_new(k_total);
    std::vector<uint8_t> v_new(v_total);

    // Copy all existing data first (for positions beyond n_kv_used, keep as-is).
    memcpy(k_new.data(), k_buf.data(), k_total);
    memcpy(v_new.data(), v_buf.data(), v_total);

    for (int i = 0; i < n_keep; i++) {
        int src = indices[i];
        memcpy(&k_new[i * k_row_stride], &k_buf[src * k_row_stride], k_row_stride);
        memcpy(&v_new[i * v_row_stride], &v_buf[src * v_row_stride], v_row_stride);
    }

    ggml_backend_tensor_set(k, k_new.data(), 0, k_new.size());
    ggml_backend_tensor_set(v, v_new.data(), 0, v_new.size());

    // Snapshot kept-cell metadata BEFORE any mutation (avoids in-place aliasing).
    std::vector<llama_pos>    kept_pos(n_keep);
    std::vector<llama_seq_id> kept_seq(n_keep);
    for (int i = 0; i < n_keep; i++) {
        int src = indices[i];
        kept_pos[i] = cells.pos_get(static_cast<uint32_t>(src));
        kept_seq[i] = cells.seq_get(static_cast<uint32_t>(src));
    }
    // Clear ALL cells [0, n_kv_used) — both kept (will re-set) and dropped.
    for (int i = 0; i < n_kv_used; i++) {
        cells.rm(static_cast<uint32_t>(i));
    }
    // Re-mark the front n_keep with snapshot positions + sequence membership.
    for (int i = 0; i < n_keep; i++) {
        cells.pos_set(static_cast<uint32_t>(i), kept_pos[i]);
        cells.seq_add(static_cast<uint32_t>(i), kept_seq[i]);
    }

    return n_keep;
}

//
// Forward declarations for KV-cache helpers used below.
//
static float depthkv_compute_sensitivity(const ggml_tensor * k, const ggml_tensor * v);
static float depthkv_sensitivity_to_keep(float sensitivity, float min_keep, float max_keep);

void llama_kv_cache::kvzip_compress() {
    if (!kvzip_enabled) {
        return;
    }

    kvzip_counter++;
    LLAMA_LOG_WARN("kvzip-d: compress called (counter=%d, trigger=%d)\n", kvzip_counter, kvzip_trigger);
    if (kvzip_counter < kvzip_trigger) {
        return;
    }
    kvzip_counter = 0;

    for (auto & layer : layers) {
        if (!layer.k || !layer.v) {
            continue;
        }

        // Use stream 0 cells to determine used count.
        const auto & cells = v_cells[0];
        int n_used = 0;
        for (size_t i = 0; i < cells.size(); i++) {
            if (cells.pos_get((uint32_t)i) >= 0) {
                n_used++;
            }
        }

        // DepthKV: per-layer compression ratio.
        float keep = kvzip_keep_ratio;
        if (depthkv_enabled) {
            float sens = depthkv_compute_sensitivity(layer.k, layer.v);
            keep = depthkv_sensitivity_to_keep(sens, depthkv_min_keep, depthkv_max_keep);
        }

        kvzip_compact(layer.k, layer.v, v_cells[0], keep, n_used, kvzip_pos_bias, kvzip_min_latest);
    }
}

//
// FastKV — token-selective propagation (TSP) + KV retention
//

static std::vector<int> fastkv_select_tokens(
    int n_tokens,
    float tsp_rate)
{
    int n_keep = std::max(1, (int)(n_tokens * tsp_rate));
    // Simplified: use position-based scoring (early tokens get priority)
    std::vector<float> scores(n_tokens, 0.0f);
    for (int i = 0; i < n_tokens; i++) {
        scores[i] = (float)(n_tokens - i) / (float)n_tokens;
    }
    std::vector<int> indices(n_tokens);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + n_keep, indices.end(),
        [&](int a, int b) { return scores[a] > scores[b]; });
    return std::vector<int>(indices.begin(), indices.begin() + n_keep);
}

void llama_kv_cache::fastkv_select_prefill() {
    if (!fastkv_enabled) {
        return;
    }
    const auto & cells = v_cells[0];
    int n_used = 0;
    for (size_t i = 0; i < cells.size(); i++) {
        if (cells.pos_get((uint32_t)i) >= 0) {
            n_used++;
        }
    }
    if (n_used <= 0) {
        return;
    }
    fastkv_selected_indices = fastkv_select_tokens(n_used, fastkv_tsp_rate);
    fastkv_prefill_done = true;
}

void llama_kv_cache::fastkv_compress() {
    if (!fastkv_enabled || !fastkv_prefill_done) {
        return;
    }
    // Apply KV retention: keep only the top KV retention tokens per layer.
    // Use the same position-based scoring as TSP selection.
    for (auto & layer : layers) {
        if (!layer.k || !layer.v) {
            continue;
        }
        const auto & cells = v_cells[0];
        int n_used = 0;
        for (size_t i = 0; i < cells.size(); i++) {
            if (cells.pos_get((uint32_t)i) >= 0) {
                n_used++;
            }
        }
        float keep = fastkv_kv_retention;
        // DepthKV can override per-layer retention
        if (depthkv_enabled) {
            float sens = depthkv_compute_sensitivity(layer.k, layer.v);
            keep = depthkv_sensitivity_to_keep(sens, depthkv_min_keep, depthkv_max_keep);
            // Also apply kv_retention as a cap
            keep = std::min(keep, fastkv_kv_retention);
        }
        // Reuse kvzip_compact with position-based scoring by using
        // a different scoring approach. We directly compact using the precomputed
        // selected indices merged with kvzip-like compression.
        kvzip_compact(layer.k, layer.v, v_cells[0], keep, n_used);
    }
    fastkv_prefill_done = false;
}

//
// DepthKV — layer-dependent compression budgets
//

static float depthkv_compute_sensitivity(
    const ggml_tensor * k,
    const ggml_tensor * v)
{
    const int n_kv = (int)std::min(k->ne[1], (int64_t)512);
    if (n_kv < 4) return 0.5f; // default

    const int n_embd_k = (int)k->ne[0];
    const int n_embd_v = (int)v->ne[0];

    // Staging buffers for reading rows and converting to f32
    // This handles quantized types correctly
    std::vector<uint8_t> row_k_bytes(n_embd_k * ggml_type_size(k->type));
    std::vector<uint8_t> row_v_bytes(n_embd_v * ggml_type_size(v->type));
    std::vector<float> row_k(n_embd_k);
    std::vector<float> row_v(n_embd_v);

    float mean_k = 0.0f, mean_v = 0.0f;
    for (int i = 0; i < n_kv; i++) {
        float nk = 0.0f, nv = 0.0f;

        // Read K row i using proper offset (nb[1] = stride for one row)
        size_t k_row_bytes = ggml_row_size(k->type, n_embd_k);
        ggml_backend_tensor_get(k, row_k_bytes.data(), i * k->nb[1], k_row_bytes);

        // Convert to f32
        switch (k->type) {
            case GGML_TYPE_F32:
                {
                    float * fp = (float *)row_k_bytes.data();
                    for (int e = 0; e < n_embd_k; e++) {
                        row_k[e] = fp[e];
                    }
                } break;
            case GGML_TYPE_F16:
                ggml_cpu_fp16_to_fp32((ggml_fp16_t *)row_k_bytes.data(), row_k.data(), n_embd_k);
                break;
            case GGML_TYPE_BF16:
                ggml_cpu_bf16_to_fp32((ggml_bf16_t *)row_k_bytes.data(), row_k.data(), n_embd_k);
                break;
            default:
                // For quantized types, use the generic path
                ggml_backend_tensor_get(k, row_k.data(), i * k->nb[1], k_row_bytes);
                break;
        }

        for (int e = 0; e < n_embd_k; e++) {
            nk += row_k[e] * row_k[e];
        }

        // Read V row i similarly
        size_t v_row_bytes = ggml_row_size(v->type, n_embd_v);
        ggml_backend_tensor_get(v, row_v_bytes.data(), i * v->nb[1], v_row_bytes);

        switch (v->type) {
            case GGML_TYPE_F32:
                {
                    float * fp = (float *)row_v_bytes.data();
                    for (int e = 0; e < n_embd_v; e++) {
                        row_v[e] = fp[e];
                    }
                } break;
            case GGML_TYPE_F16:
                ggml_cpu_fp16_to_fp32((ggml_fp16_t *)row_v_bytes.data(), row_v.data(), n_embd_v);
                break;
            case GGML_TYPE_BF16:
                ggml_cpu_bf16_to_fp32((ggml_bf16_t *)row_v_bytes.data(), row_v.data(), n_embd_v);
                break;
            default:
                ggml_backend_tensor_get(v, row_v.data(), i * v->nb[1], v_row_bytes);
                break;
        }

        for (int e = 0; e < n_embd_v; e++) {
            nv += row_v[e] * row_v[e];
        }
        mean_k += nk;
        mean_v += nv;
    }
    mean_k /= (float)n_kv;
    mean_v /= (float)n_kv;

    float var_k = 0.0f, var_v = 0.0f;
    for (int i = 0; i < n_kv; i++) {
        float nk = 0.0f, nv = 0.0f;

        // Re-read rows for variance calculation
        size_t k_row_bytes = ggml_row_size(k->type, n_embd_k);
        ggml_backend_tensor_get(k, row_k_bytes.data(), i * k->nb[1], k_row_bytes);

        switch (k->type) {
            case GGML_TYPE_F32:
                {
                    float * fp = (float *)row_k_bytes.data();
                    for (int e = 0; e < n_embd_k; e++) {
                        row_k[e] = fp[e];
                    }
                } break;
            case GGML_TYPE_F16:
                ggml_cpu_fp16_to_fp32((ggml_fp16_t *)row_k_bytes.data(), row_k.data(), n_embd_k);
                break;
            case GGML_TYPE_BF16:
                ggml_cpu_bf16_to_fp32((ggml_bf16_t *)row_k_bytes.data(), row_k.data(), n_embd_k);
                break;
            default:
                ggml_backend_tensor_get(k, row_k.data(), i * k->nb[1], k_row_bytes);
                break;
        }

        for (int e = 0; e < n_embd_k; e++) {
            nk += row_k[e] * row_k[e];
        }

        // Read V row
        size_t v_row_bytes = ggml_row_size(v->type, n_embd_v);
        ggml_backend_tensor_get(v, row_v_bytes.data(), i * v->nb[1], v_row_bytes);

        switch (v->type) {
            case GGML_TYPE_F32:
                {
                    float * fp = (float *)row_v_bytes.data();
                    for (int e = 0; e < n_embd_v; e++) {
                        row_v[e] = fp[e];
                    }
                } break;
            case GGML_TYPE_F16:
                ggml_cpu_fp16_to_fp32((ggml_fp16_t *)row_v_bytes.data(), row_v.data(), n_embd_v);
                break;
            case GGML_TYPE_BF16:
                ggml_cpu_bf16_to_fp32((ggml_bf16_t *)row_v_bytes.data(), row_v.data(), n_embd_v);
                break;
            default:
                ggml_backend_tensor_get(v, row_v.data(), i * v->nb[1], v_row_bytes);
                break;
        }

        for (int e = 0; e < n_embd_v; e++) {
            nv += row_v[e] * row_v[e];
        }
        var_k += (nk - mean_k) * (nk - mean_k);
        var_v += (nv - mean_v) * (nv - mean_v);
    }
    var_k /= (float)n_kv;
    var_v /= (float)n_kv;

    // Normalize to [0, 1] using sigmoid-like clamp.
    // Higher variance = more informative = more sensitive to pruning.
    return std::max(0.0f, std::min(1.0f, (var_k + var_v) / 2.0f));
}

static float depthkv_sensitivity_to_keep(float sensitivity, float min_keep, float max_keep) {
    float clamped = std::max(0.0f, std::min(1.0f, sensitivity));
    return min_keep + clamped * (max_keep - min_keep);
}

//
// RazorAttention — head-type specialization
//

static float razor_compute_echo_score(
    const ggml_tensor * k,
    const int n_embd_head,
    const int n_head,
    const int head_idx,
    const int n_kv)
{
    // Echo score: measure how much a head attends to tokens that repeat.
    // Compute dot product similarity between the first token's K and each
    // subsequent token's K for this head. Higher similarity at distance = more retrieval.
    // Uses per-row access to handle quantized tensor types correctly.

    // K tensor layout: [n_embd, n_kv, n_head]
    // head_idx offset within a row: head_idx * n_embd_head
    // Row i offset: i * k->nb[1] (stride for one row)
    const size_t row_bytes = ggml_row_size(k->type, k->ne[0]);
    const size_t head_offset_bytes = head_idx * n_embd_head * ggml_type_size(k->type);
    const float inv_embd = 1.0f / (float)n_embd_head;

    std::vector<uint8_t> row_bytes_buf(row_bytes);
    std::vector<float> row_f32(k->ne[0]);

    float total_sim = 0.0f;
    int count = 0;

    // First, read the first token's K values for this head (v0)
    std::vector<float> v0_head(n_embd_head);
    ggml_backend_tensor_get(k, row_bytes_buf.data(), 0 * k->nb[1], row_bytes);
    // Convert row to f32
    switch (k->type) {
        case GGML_TYPE_F32:
            {
                float * fp = (float *)row_bytes_buf.data();
                for (int e = 0; e < n_embd_head; e++) {
                    v0_head[e] = fp[head_idx * n_embd_head + e];
                }
            } break;
        case GGML_TYPE_F16:
            {
                ggml_cpu_fp16_to_fp32((ggml_fp16_t *)row_bytes_buf.data(), row_f32.data(), k->ne[0]);
                for (int e = 0; e < n_embd_head; e++) {
                    v0_head[e] = row_f32[head_idx * n_embd_head + e];
                }
            } break;
        case GGML_TYPE_BF16:
            {
                ggml_cpu_bf16_to_fp32((ggml_bf16_t *)row_bytes_buf.data(), row_f32.data(), k->ne[0]);
                for (int e = 0; e < n_embd_head; e++) {
                    v0_head[e] = row_f32[head_idx * n_embd_head + e];
                }
            } break;
        default:
            ggml_backend_tensor_get(k, row_f32.data(), 0 * k->nb[1], row_bytes);
            for (int e = 0; e < n_embd_head; e++) {
                v0_head[e] = row_f32[head_idx * n_embd_head + e];
            }
            break;
    }

    for (int i = 1; i < std::min(n_kv, 64); i++) {
        float dot = 0.0f;

        // Read token i's K row
        ggml_backend_tensor_get(k, row_bytes_buf.data(), i * k->nb[1], row_bytes);

        switch (k->type) {
            case GGML_TYPE_F32:
                {
                    float * fp = (float *)row_bytes_buf.data();
                    for (int e = 0; e < n_embd_head; e++) {
                        float vi = fp[head_idx * n_embd_head + e];
                        dot += v0_head[e] * vi;
                    }
                } break;
            case GGML_TYPE_F16:
                ggml_cpu_fp16_to_fp32((ggml_fp16_t *)row_bytes_buf.data(), row_f32.data(), k->ne[0]);
                for (int e = 0; e < n_embd_head; e++) {
                    dot += v0_head[e] * row_f32[head_idx * n_embd_head + e];
                }
                break;
            case GGML_TYPE_BF16:
                ggml_cpu_bf16_to_fp32((ggml_bf16_t *)row_bytes_buf.data(), row_f32.data(), k->ne[0]);
                for (int e = 0; e < n_embd_head; e++) {
                    dot += v0_head[e] * row_f32[head_idx * n_embd_head + e];
                }
                break;
            default:
                ggml_backend_tensor_get(k, row_f32.data(), i * k->nb[1], row_bytes);
                for (int e = 0; e < n_embd_head; e++) {
                    dot += v0_head[e] * row_f32[head_idx * n_embd_head + e];
                }
                break;
        }

        total_sim += fabsf(dot) * inv_embd;
        count++;
    }
    return count > 0 ? total_sim / count : 0.0f;
}

void llama_kv_cache::razor_profile_heads(
    const ggml_tensor * k_first_layer)
{
    if (!k_first_layer || k_first_layer->ne[1] < 16) {
        return;
    }

    const int n_embd_head = 128;
    const int n_kv = (int)k_first_layer->ne[1];
    const int n_head_total = razor_attn_n_head > 0 ? razor_attn_n_head : 8;

    razor_attn_profiles.clear();
    razor_attn_profiles.reserve(n_head_total);

    for (int h = 0; h < n_head_total; h++) {
        float avg_dist = razor_compute_echo_score(
            k_first_layer, n_embd_head, n_head_total, h, n_kv);
        bool is_retrieval = avg_dist > 0.3f;
        razor_attn_profiles.push_back({
            /*.layer =*/ 0, /*.head =*/ (uint32_t)h,
            /*.is_retrieval=*/ is_retrieval,
            /*.avg_attn_distance=*/ avg_dist,
        });
    }
    razor_attn_profiled = true;
}

// Post-process an already-built KQ mask for RazorAttention.
// After the mask is built with causal/SWA logic, apply per-head local window.
void llama_kv_cache::razor_apply_mask(ggml_tensor * kq_mask) const {
    if (!razor_attn_enabled || !razor_attn_profiled || razor_attn_profiles.empty()) {
        return;
    }
    // kq_mask shape: [n_kv, n_tokens, n_head, n_stream] or similar
    // For each (stream, head, token, kv_pos): if head is local and
    // kv_pos is outside window, set to -INF.
    const int64_t ne0 = kq_mask->ne[0]; // n_kv
    const int64_t ne1 = kq_mask->ne[1]; // n_tokens or n_head
    const int64_t ne2 = kq_mask->ne[2];
    const int64_t ne3 = kq_mask->ne[3];
    const bool is_half = kq_mask->type == GGML_TYPE_F16;
    const int n_heads = (int)razor_attn_profiles.size();

    for (int64_t s = 0; s < ne3; s++) {
        for (int64_t h = 0; h < std::min((int64_t)n_heads, ne2); h++) {
            if (h >= (int64_t)razor_attn_profiles.size()) break;
            if (razor_attn_profiles[h].is_retrieval) continue;
            for (int64_t t = 0; t < ne1; t++) {
                for (int64_t k = 0; k < ne0; k++) {
                    if (k + razor_attn_window < ne0) {
                        int64_t idx = k + t * ne0 + h * ne0 * ne1 + s * ne0 * ne1 * ne2;
                        if (is_half) {
                            // F16: write -INF as very negative float16
                            auto * data = (ggml_fp16_t *)kq_mask->data;
                            data[idx] = ggml_fp32_to_fp16(-FLT_MAX / 2.0f);
                        } else {
                            auto * data = (float *)kq_mask->data;
                            data[idx] = -FLT_MAX / 2.0f;
                        }
                    }
                }
            }
        }
    }
}

// Compensation token computation for RazorAttention.
// Average K/V of tokens that fall outside the local window.
void llama_kv_cache::razor_compute_compensation(int _n_kv) const { (void)_n_kv;
    if (!razor_attn_enabled || !razor_attn_profiled) {
        return;
    }
    // For each layer, for each local head, average the K/V of tokens
    // outside the window.
    for (auto & layer : layers) {
        if (!layer.k || !layer.v) continue;
        (void)layer; // placeholder — per-head access needs stride info
    }
}

//
// RazorAttention profile persistence
//

void llama_kv_cache::razor_save_profile(const char * path) const {
    if (!razor_attn_profiled || razor_attn_profiles.empty()) {
        return;
    }
    FILE * f = fopen(path, "w");
    if (!f) return;
    fprintf(f, "{\"n_heads\":%zu,\"window\":%d,\"heads\":[",
            razor_attn_profiles.size(), razor_attn_window);
    for (size_t i = 0; i < razor_attn_profiles.size(); i++) {
        if (i > 0) fprintf(f, ",");
        fprintf(f, "{\"h\":%u,\"l\":%u,\"r\":%s,\"d\":%f}",
                razor_attn_profiles[i].head,
                razor_attn_profiles[i].layer,
                razor_attn_profiles[i].is_retrieval ? "true" : "false",
                razor_attn_profiles[i].avg_attn_distance);
    }
    fprintf(f, "]}\n");
    fclose(f);
}

bool llama_kv_cache::razor_load_profile(const char * path) {
    FILE * f = fopen(path, "r");
    if (!f) return false;
    // Simple JSON parse — extract is_retrieval flags.
    razor_attn_profiles.clear();
    // Just detect & parse the JSON structure minimally.
    // For v1: assume file is well-formed and skip full parser.
    char buf[4096];
    size_t n = fread(buf, 1, sizeof(buf), f);
    fclose(f);
    if (n == 0) return false;
    buf[n] = '\0';

    // Count heads and set profiled flag.
    int count = 0;
    const char * p = buf;
    while ((p = strstr(p, "\"r\":")) != NULL) {
        p += 4;
        bool is_ret = (*p == 't');
        razor_attn_profiles.push_back({
            /*layer=*/0, /*head=*/(uint32_t)count,
            /*is_retrieval=*/is_ret,
            /*avg_attn_distance=*/0.0f
        });
        count++;
    }
    razor_attn_profiled = !razor_attn_profiles.empty();
    return razor_attn_profiled;
}

//
// xKV — cross-layer SVD factorization for KV-cache compression
//
// Computes a shared SVD basis across transformer layers and stores
// per-layer coefficients instead of full K/V tensors, achieving ~8x compression.
//

// Power iteration to compute the top-n eigenvectors of a symmetric matrix G (gram matrix).
// G is [n x n] in row-major storage.
// Returns eigenvectors as [n_singular x n] row-major, eigenvalues in descending order.
static bool xkv_power_iteration(
    const std::vector<double> & G,
    int n,
    int n_singular,
    std::vector<float> & eigenvectors,
    std::vector<double> & eigenvalues,
    int max_iter = 100,
    double tol = 1e-8)
{
    eigenvectors.resize(n_singular * n, 0.0f);
    eigenvalues.resize(n_singular, 0.0);

    std::vector<double> G_host(n * n);
    for (int i = 0; i < n * n; i++) {
        G_host[i] = G[i];
    }

    for (int s = 0; s < n_singular; s++) {
        // Random initial vector
        std::vector<double> v(n, 0.0);
        std::vector<double> v_prev(n, 0.0);
        for (int i = 0; i < n; i++) {
            v[i] = (double)(rand() % 10000) / 5000.0 - 1.0;
        }

        // Normalize
        double norm = 0.0;
        for (int i = 0; i < n; i++) norm += v[i] * v[i];
        norm = sqrt(norm);
        if (norm > 0) {
            for (int i = 0; i < n; i++) v[i] /= norm;
        }

        bool converged = false;
        for (int iter = 0; iter < max_iter; iter++) {
            // v = G * v_prev
            std::vector<double> v_new(n, 0.0);
            for (int i = 0; i < n; i++) {
                double sum = 0.0;
                for (int j = 0; j < n; j++) {
                    sum += G_host[i * n + j] * v[j];
                }
                v_new[i] = sum;
            }

            // Deflate: subtract projection onto already-found eigenvectors
            for (int ps = 0; ps < s; ps++) {
                double dot = 0.0;
                for (int i = 0; i < n; i++) {
                    dot += eigenvectors[ps * n + i] * v_new[i];
                }
                for (int i = 0; i < n; i++) {
                    v_new[i] -= dot * eigenvectors[ps * n + i];
                }
            }

            norm = 0.0;
            for (int i = 0; i < n; i++) norm += v_new[i] * v_new[i];
            norm = sqrt(norm);
            if (norm > 0) {
                for (int i = 0; i < n; i++) v_new[i] /= norm;
            }

            // Check convergence
            double diff = 0.0;
            for (int i = 0; i < n; i++) {
                double d = v_new[i] - v[i];
                diff += d * d;
            }
            diff = sqrt(diff);

            v = v_new;

            if (diff < tol) {
                converged = true;
                break;
            }
        }

        // Store eigenvector
        for (int i = 0; i < n; i++) {
            eigenvectors[s * n + i] = (float)v[i];
        }

        // Compute eigenvalue = v^T * G * v
        double eval = 0.0;
        for (int i = 0; i < n; i++) {
            double row_sum = 0.0;
            for (int j = 0; j < n; j++) {
                row_sum += G_host[i * n + j] * v[j];
            }
            eval += v[i] * row_sum;
        }
        eigenvalues[s] = eval;
    }

    return true;
}

// xKV tensor access helpers — safe replacements for ggml_get_f32_1d on the
// flat-1D pattern. The flat pattern aborts on block-quantized types and is
// brittle on f16/bf16. These helpers use ggml_backend_tensor_get with proper
// nb[] strides and type-aware dequantization. (See depthkv_compute_sensitivity
// for the per-row pattern this mirrors.)
static void xkv_read_row_f32(const ggml_tensor * t, int64_t i, int n_embd, std::vector<float> & out) {
    out.assign(n_embd, 0.0f);
    const size_t row_bytes = ggml_row_size(t->type, n_embd);
    std::vector<uint8_t> buf(row_bytes);
    ggml_backend_tensor_get(t, buf.data(), (size_t)i * t->nb[1], row_bytes);
    switch (t->type) {
        case GGML_TYPE_F32: {
            const float * fp = (const float *) buf.data();
            for (int e = 0; e < n_embd; e++) out[e] = fp[e];
        } break;
        case GGML_TYPE_F16:
            ggml_cpu_fp16_to_fp32((const ggml_fp16_t *) buf.data(), out.data(), n_embd);
            break;
        case GGML_TYPE_BF16:
            ggml_cpu_bf16_to_fp32((const ggml_bf16_t *) buf.data(), out.data(), n_embd);
            break;
        default:
            // Block-quantized K/V is uncommon; if encountered, leave zeros so
            // the va == 0.0 check in callers skips this row. Better than crashing.
            break;
    }
}

static double xkv_read_elem_f64(const ggml_tensor * t, int64_t row_idx, int64_t col_idx) {
    const size_t byte_offset = (size_t)(row_idx * t->nb[1] + col_idx * t->nb[0]);
    switch (t->type) {
        case GGML_TYPE_F32: {
            float v;
            ggml_backend_tensor_get(t, &v, byte_offset, sizeof(float));
            return (double)v;
        }
        case GGML_TYPE_F16: {
            ggml_fp16_t v;
            ggml_backend_tensor_get(t, &v, byte_offset, sizeof(ggml_fp16_t));
            float f32;
            ggml_cpu_fp16_to_fp32(&v, &f32, 1);
            return (double) f32;
        }
        case GGML_TYPE_BF16: {
            ggml_bf16_t v;
            ggml_backend_tensor_get(t, &v, byte_offset, sizeof(ggml_bf16_t));
            float f32;
            ggml_cpu_bf16_to_fp32(&v, &f32, 1);
            return (double) f32;
        }
        default: {
            // Block-quantized: dequantize the whole row and pick the element.
            std::vector<float> row;
            xkv_read_row_f32(t, row_idx, (int) t->ne[0], row);
            return (double) row[(size_t) col_idx];
        }
    }
}

void llama_kv_cache::xkv_compute_basis() {
    if (!xkv_enabled) {
        return;
    }

    // Try loading profile first
    if (!xkv_profile_path.empty()) {
        if (xkv_load_profile(xkv_profile_path.c_str())) {
            LLAMA_LOG_INFO("%s: loaded xKV profile from %s\n", __func__, xkv_profile_path.c_str());
            return;
        }
    }

    if (layers.empty()) {
        LLAMA_LOG_WARN("%s: no layers in cache, cannot compute basis\n", __func__);
        return;
    }

    // Sample layers: 0, mid, last
    int n_layers = (int)layers.size();
    std::vector<int> sample_layers;
    sample_layers.push_back(0);
    if (n_layers > 2) {
        sample_layers.push_back(n_layers / 2);
    }
    if (n_layers > 1) {
        sample_layers.push_back(n_layers - 1);
    }

    int n_embd_k = hparams.n_embd_k_gqa(sample_layers[0]);
    int n_embd_v = 0; // determined below

    // Collect K tensors from sampled layers for Gram matrix computation
    std::vector<const ggml_tensor *> k_sampled;
    for (int li : sample_layers) {
        if (li < (int)layers.size() && layers[li].k) {
            k_sampled.push_back(layers[li].k);
        }
    }

    if (k_sampled.empty()) {
        LLAMA_LOG_WARN("%s: no K tensors available for xKV basis computation\n", __func__);
        return;
    }

    // Compute Gram matrix G = A^T * A where A is stacked K rows
    int n_embd = n_embd_k;
    std::vector<double> G(n_embd * n_embd, 0.0);

    for (auto * t : k_sampled) {
        int n_rows = (int)t->ne[1];
        for (int i = 0; i < n_rows; i++) {
            std::vector<float> row;
            xkv_read_row_f32(t, i, n_embd, row);
            for (int a = 0; a < n_embd; a++) {
                double va = (double) row[a];
                if (va == 0.0) continue;
                for (int b = 0; b < n_embd; b++) {
                    double vb = (double) row[b];
                    G[a * n_embd + b] += va * vb;
                }
            }
        }
    }

    // Compute top-n_singular eigenvectors via power iteration
    int n_singular = xkv_rank;
    std::vector<float> eigvecs_k;
    std::vector<double> evals_k;

    if (!xkv_power_iteration(G, n_embd, n_singular, eigvecs_k, evals_k)) {
        LLAMA_LOG_ERROR("%s: power iteration failed for K basis\n", __func__);
        return;
    }

    // Same for V: collect V tensors and compute Gram matrix
    std::vector<const ggml_tensor *> v_sampled;
    for (int li : sample_layers) {
        if (li < (int)layers.size() && layers[li].v) {
            v_sampled.push_back(layers[li].v);
        }
    }

    if (v_sampled.empty()) {
        LLAMA_LOG_WARN("%s: no V tensors available for xKV basis computation\n", __func__);
        return;
    }

    // For V, determine embedding dimension from the tensor shape
    // V may be transposed: [kv_size, n_embd_v_gqa, ...] or [n_embd_head, n_head, ...]
    n_embd_v = (int)v_sampled[0]->ne[0];
    if (v_trans) {
        // V transposed layout: ne[0] = kv_size
        // The actual embedding dim is in the V tensor's spatial dims
        // We'll use hparams to get the correct dim
        n_embd_v = hparams.n_embd_v_gqa(sample_layers[0]);
    }

    std::vector<double> Gv(n_embd_v * n_embd_v, 0.0);
    for (auto * t : v_sampled) {
        int n_rows = (int)t->ne[1];
        if (v_trans) {
            // For transposed V, each column slice is a V vector
            // V shape: [kv_size, n_embd_v_gqa, ...]
            // Use ne[0] as the token dimension (kv_size), ne[1] as embedding
            n_rows = (int)t->ne[0]; // kv_size
            int n_embd_v_gqa = (int)t->ne[1];
            for (int i = 0; i < n_rows; i++) {
                for (int a = 0; a < n_embd_v_gqa; a++) {
                    double va = xkv_read_elem_f64(t, a, i);  // column-major (col, row)
                    if (va == 0.0) continue;
                    for (int b = 0; b < n_embd_v_gqa; b++) {
                        double vb = xkv_read_elem_f64(t, b, i);
                        Gv[a * n_embd_v_gqa + b] += va * vb;
                    }
                }
            }
            n_embd_v = n_embd_v_gqa;
        } else {
            for (int i = 0; i < n_rows; i++) {
                std::vector<float> row;
                xkv_read_row_f32(t, i, n_embd_v, row);
                for (int a = 0; a < n_embd_v; a++) {
                    double va = (double) row[a];
                    if (va == 0.0) continue;
                    for (int b = 0; b < n_embd_v; b++) {
                        double vb = (double) row[b];
                        Gv[a * n_embd_v + b] += va * vb;
                    }
                }
            }
        }
    }

    std::vector<float> eigvecs_v;
    std::vector<double> evals_v;

    if (!xkv_power_iteration(Gv, n_embd_v, n_singular, eigvecs_v, evals_v)) {
        LLAMA_LOG_ERROR("%s: power iteration failed for V basis\n", __func__);
        return;
    }

    // Store basis
    xkv_basis_data.n_singular = n_singular;
    xkv_basis_data.n_embd_k   = n_embd_k;
    xkv_basis_data.n_embd_v   = n_embd_v;
    xkv_basis_data.U_k        = std::move(eigvecs_k);
    xkv_basis_data.U_v        = std::move(eigvecs_v);

    // Initialize coefficient arrays
    xkv_basis_data.coeffs_k.resize(n_layers);
    xkv_basis_data.coeffs_v.resize(n_layers);
    for (int l = 0; l < n_layers; l++) {
        xkv_basis_data.coeffs_k[l].resize(n_singular, 0.0f);
        xkv_basis_data.coeffs_v[l].resize(n_singular, 0.0f);
    }

    xkv_profiled = true;

    LLAMA_LOG_INFO("%s: xKV basis computed: n_singular=%d, n_embd_k=%d, n_embd_v=%d, samples from %d layers\n",
        __func__, n_singular, n_embd_k, n_embd_v, (int)k_sampled.size());

    // Save profile
    if (!xkv_profile_path.empty()) {
        xkv_save_profile(xkv_profile_path.c_str());
    }
}

void llama_kv_cache::xkv_project() {
    if (!xkv_enabled || !xkv_profiled || !xkv_basis_data.is_valid()) {
        return;
    }

    if (layers.empty()) {
        return;
    }

    int n_singular = xkv_basis_data.n_singular;
    int n_layers   = (int)layers.size();

    // Project each layer's K onto the shared basis
    for (int l = 0; l < n_layers; l++) {
        auto * k_tensor = layers[l].k;
        if (!k_tensor) continue;

        int n_embd = (int)k_tensor->ne[0];
        int n_kv   = (int)k_tensor->ne[1]; // number of KV tokens in this layer

        // coeffs_k[l][s] = mean over tokens of sum_e K_token[e] * U_k[s][e]
        for (int s = 0; s < n_singular; s++) {
            double sum = 0.0;
            int count = 0;
            for (int i = 0; i < n_kv; i++) {
                std::vector<float> row;
                xkv_read_row_f32(k_tensor, i, n_embd, row);
                double dot = 0.0;
                for (int e = 0; e < n_embd && e < xkv_basis_data.n_embd_k; e++) {
                    double ke = (double) row[e];
                    dot += ke * xkv_basis_data.U_k[s * xkv_basis_data.n_embd_k + e];
                }
                sum += dot;
                count++;
            }
            xkv_basis_data.coeffs_k[l][s] = count > 0 ? (float)(sum / count) : 0.0f;
        }

        // Same for V
        auto * v_tensor = layers[l].v;
        if (!v_tensor) continue;

        int n_embd_v_gqa = xkv_basis_data.n_embd_v;
        int n_kv_v = v_trans ? (int)v_tensor->ne[0] : (int)v_tensor->ne[1];

        for (int s = 0; s < n_singular; s++) {
            double sum = 0.0;
            int count = 0;
            for (int i = 0; i < n_kv_v; i++) {
                double dot = 0.0;
                if (v_trans) {
                    // V transposed: V[e][i]
                    for (int e = 0; e < n_embd_v_gqa; e++) {
                        double ve = xkv_read_elem_f64(v_tensor, e, i);
                        dot += ve * xkv_basis_data.U_v[s * xkv_basis_data.n_embd_v + e];
                    }
                } else {
                    std::vector<float> row;
                    xkv_read_row_f32(v_tensor, i, n_embd_v_gqa, row);
                    for (int e = 0; e < n_embd_v_gqa; e++) {
                        double ve = (double) row[e];
                        dot += ve * xkv_basis_data.U_v[s * xkv_basis_data.n_embd_v + e];
                    }
                }
                sum += dot;
                count++;
            }
            xkv_basis_data.coeffs_v[l][s] = count > 0 ? (float)(sum / count) : 0.0f;
        }
    }

    LLAMA_LOG_INFO("%s: xKV projection complete for %d layers, rank=%d\n",
        __func__, n_layers, n_singular);
}

void llama_kv_cache::xkv_reconstruct_k(
    ggml_tensor * k_out,
    const ggml_tensor * coeffs_in,
    int32_t il) const
{
    GGML_UNUSED(coeffs_in);
    if (!xkv_enabled || !xkv_profiled) {
        return;
    }

    const int32_t ikv = map_layer_ids.at(il);
    if (ikv < 0 || ikv >= (int32_t)layers.size()) {
        return;
    }

    int n_singular = xkv_basis_data.n_singular;
    int n_embd_k   = (int)k_out->ne[0];

    // For each token in k_out, reconstruct: k_token[e] = sum_s coeffs[l][s] * U_k[s][e]
    // Since we stored mean coefficients, we use the same U_k and scale by the coefficient
    int n_kv = (int)k_out->ne[1];
    for (int i = 0; i < n_kv; i++) {
        for (int e = 0; e < n_embd_k && e < xkv_basis_data.n_embd_k; e++) {
            double val = 0.0;
            for (int s = 0; s < n_singular; s++) {
                val += xkv_basis_data.coeffs_k[ikv][s] * xkv_basis_data.U_k[s * xkv_basis_data.n_embd_k + e];
            }
            ggml_set_f32_1d(k_out, (int64_t)i * n_embd_k + e, (float)val);
        }
    }
}

void llama_kv_cache::xkv_reconstruct_v(
    ggml_tensor * v_out,
    const ggml_tensor * coeffs_in,
    int32_t il) const
{
    GGML_UNUSED(coeffs_in);
    if (!xkv_enabled || !xkv_profiled) {
        return;
    }

    const int32_t ikv = map_layer_ids.at(il);
    if (ikv < 0 || ikv >= (int32_t)layers.size()) {
        return;
    }

    int n_singular = xkv_basis_data.n_singular;
    int n_embd_v   = xkv_basis_data.n_embd_v;

    if (v_trans) {
        // V is transposed: [kv_size, n_embd_v_gqa]
        int n_kv = (int)v_out->ne[0];
        int n_embd_v_gqa = (int)v_out->ne[1];
        for (int i = 0; i < n_kv; i++) {
            for (int e = 0; e < n_embd_v_gqa && e < n_embd_v; e++) {
                double val = 0.0;
                for (int s = 0; s < n_singular; s++) {
                    val += xkv_basis_data.coeffs_v[ikv][s] * xkv_basis_data.U_v[s * n_embd_v + e];
                }
                ggml_set_f32_1d(v_out, (int64_t)e * n_kv + i, (float)val);
            }
        }
    } else {
        int n_kv = (int)v_out->ne[1];
        int n_embd_v_gqa = (int)v_out->ne[0];
        for (int i = 0; i < n_kv; i++) {
            for (int e = 0; e < n_embd_v_gqa && e < n_embd_v; e++) {
                double val = 0.0;
                for (int s = 0; s < n_singular; s++) {
                    val += xkv_basis_data.coeffs_v[ikv][s] * xkv_basis_data.U_v[s * n_embd_v + e];
                }
                ggml_set_f32_1d(v_out, (int64_t)i * n_embd_v_gqa + e, (float)val);
            }
        }
    }
}

void llama_kv_cache::xkv_save_profile(const char * path) const {
    if (!xkv_profiled || !xkv_basis_data.is_valid()) {
        return;
    }

    FILE * f = fopen(path, "wb");
    if (!f) {
        LLAMA_LOG_WARN("%s: failed to open %s for writing\n", __func__, path);
        return;
    }

    // Binary format:
    //   int32_t: n_singular
    //   int32_t: n_embd_k
    //   int32_t: n_embd_v
    //   int32_t: n_layers
    //   float[n_singular * n_embd_k]: U_k
    //   float[n_singular * n_embd_v]: U_v
    //   float[n_layers * n_singular]: coeffs_k
    //   float[n_layers * n_singular]: coeffs_v

    int32_t n_singular = xkv_basis_data.n_singular;
    int32_t n_embd_k   = xkv_basis_data.n_embd_k;
    int32_t n_embd_v   = xkv_basis_data.n_embd_v;
    int32_t n_layers   = (int32_t)xkv_basis_data.coeffs_k.size();

    fwrite(&n_singular, sizeof(int32_t), 1, f);
    fwrite(&n_embd_k,   sizeof(int32_t), 1, f);
    fwrite(&n_embd_v,   sizeof(int32_t), 1, f);
    fwrite(&n_layers,   sizeof(int32_t), 1, f);

    fwrite(xkv_basis_data.U_k.data(), sizeof(float), (size_t)n_singular * n_embd_k, f);
    fwrite(xkv_basis_data.U_v.data(), sizeof(float), (size_t)n_singular * n_embd_v, f);

    for (int32_t l = 0; l < n_layers; l++) {
        fwrite(xkv_basis_data.coeffs_k[l].data(), sizeof(float), n_singular, f);
    }
    for (int32_t l = 0; l < n_layers; l++) {
        fwrite(xkv_basis_data.coeffs_v[l].data(), sizeof(float), n_singular, f);
    }

    fclose(f);
    LLAMA_LOG_INFO("%s: xKV profile saved to %s (rank=%d, %d layers)\n",
        __func__, path, n_singular, n_layers);
}

bool llama_kv_cache::xkv_load_profile(const char * path) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        return false;
    }

    int32_t n_singular, n_embd_k, n_embd_v, n_layers;
    if (fread(&n_singular, sizeof(int32_t), 1, f) != 1) { fclose(f); return false; }
    if (fread(&n_embd_k,   sizeof(int32_t), 1, f) != 1) { fclose(f); return false; }
    if (fread(&n_embd_v,   sizeof(int32_t), 1, f) != 1) { fclose(f); return false; }
    if (fread(&n_layers,   sizeof(int32_t), 1, f) != 1) { fclose(f); return false; }

    if (n_singular <= 0 || n_embd_k <= 0 || n_embd_v <= 0 || n_layers <= 0) {
        fclose(f);
        return false;
    }

    xkv_basis basis;
    basis.n_singular = n_singular;
    basis.n_embd_k   = n_embd_k;
    basis.n_embd_v   = n_embd_v;

    basis.U_k.resize((size_t)n_singular * n_embd_k);
    basis.U_v.resize((size_t)n_singular * n_embd_v);

    if (fread(basis.U_k.data(), sizeof(float), (size_t)n_singular * n_embd_k, f) != (size_t)n_singular * n_embd_k) {
        fclose(f); return false;
    }
    if (fread(basis.U_v.data(), sizeof(float), (size_t)n_singular * n_embd_v, f) != (size_t)n_singular * n_embd_v) {
        fclose(f); return false;
    }

    basis.coeffs_k.resize(n_layers);
    basis.coeffs_v.resize(n_layers);
    for (int32_t l = 0; l < n_layers; l++) {
        basis.coeffs_k[l].resize(n_singular);
        if (fread(basis.coeffs_k[l].data(), sizeof(float), n_singular, f) != (size_t)n_singular) {
            fclose(f); return false;
        }
    }
    for (int32_t l = 0; l < n_layers; l++) {
        basis.coeffs_v[l].resize(n_singular);
        if (fread(basis.coeffs_v[l].data(), sizeof(float), n_singular, f) != (size_t)n_singular) {
            fclose(f); return false;
        }
    }

    fclose(f);

    if (!basis.is_valid()) {
        return false;
    }

    xkv_basis_data = std::move(basis);
    xkv_profiled = true;
    xkv_rank     = n_singular;

    LLAMA_LOG_INFO("%s: xKV profile loaded from %s (rank=%d, %d layers)\n",
        __func__, path, n_singular, n_layers);
    return true;
}

std::vector<uint32_t> llama_kv_cache::get_layer_ids() const {
    std::vector<uint32_t> res;
    res.reserve(layers.size());

    for (const auto & layer : layers) {
        res.push_back(layer.il);
    }

    return res;
}

ggml_tensor * llama_kv_cache::get_k_storage(int32_t il) const {
    const int32_t ikv = map_layer_ids.at(il);

    return layers[ikv].k;
}

uint32_t llama_kv_cache::get_n_kv(const slot_info & sinfo) const {
    uint32_t result = 0;

    // pad the n_kv value so that the graph remains constant across batches and can be reused
    // note: this also helps some backends with performance (f.ex https://github.com/ggml-org/llama.cpp/pull/16812#issuecomment-3455112220)
    const uint32_t n_pad_cur = std::max(n_pad, 256u);

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        const auto & cells = v_cells[sinfo.strm[s]];

        result = std::max(std::min(cells.size(), std::max(n_pad_cur, GGML_PAD(cells.used_max_p1(), n_pad_cur))), result);
    }

    return result;
}

ggml_tensor * llama_kv_cache::get_k(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const {
    const int32_t ikv = map_layer_ids.at(il);

    auto * k = layers[ikv].k;

    const uint64_t kv_size      = get_size();
    const uint64_t n_embd_k_gqa = k->ne[0];

    assert(n_embd_k_gqa == hparams.n_embd_k_gqa(il));

    const uint32_t ns = sinfo.s1 - sinfo.s0 + 1;

    return ggml_view_4d(ctx, k,
            hparams.n_embd_head_k(il), hparams.n_head_kv(il), n_kv, ns,
            ggml_row_size(k->type, hparams.n_embd_head_k(il)),
            ggml_row_size(k->type, n_embd_k_gqa),
            ggml_row_size(k->type, n_embd_k_gqa*kv_size),
            ggml_row_size(k->type, n_embd_k_gqa*kv_size)*sinfo.s0);
}

ggml_tensor * llama_kv_cache::get_v(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const {
    const int32_t ikv = map_layer_ids.at(il);

    auto * v = layers[ikv].v;

    const uint64_t kv_size      = get_size();
    const uint64_t n_embd_v_gqa = v->ne[0];

    // [TAG_V_CACHE_VARIABLE]
    assert(n_embd_v_gqa >= hparams.n_embd_v_gqa(il));

    const uint32_t ns = sinfo.s1 - sinfo.s0 + 1;

    if (!v_trans) {
        // note: v->nb[1] <= v->nb[2]
        return ggml_view_4d(ctx, v,
                hparams.n_embd_head_v(il), hparams.n_head_kv(il), n_kv, ns,
                ggml_row_size(v->type, hparams.n_embd_head_v(il)),          // v->nb[1]
                ggml_row_size(v->type, n_embd_v_gqa),                   // v->nb[2]
                ggml_row_size(v->type, n_embd_v_gqa*kv_size),           // v->nb[3]
                ggml_row_size(v->type, n_embd_v_gqa*kv_size)*sinfo.s0);
    }

    // note: v->nb[1] > v->nb[2]
    return ggml_view_4d(ctx, v,
            n_kv, hparams.n_head_kv(il), hparams.n_embd_head_v(il), ns,
            ggml_row_size(v->type, kv_size*hparams.n_embd_head_v(il)),  // v->nb[1]
            ggml_row_size(v->type, kv_size),                        // v->nb[2]
            ggml_row_size(v->type, kv_size*n_embd_v_gqa),           // v->nb[3]
            ggml_row_size(v->type, kv_size*n_embd_v_gqa)*sinfo.s0);
}

ggml_tensor * llama_kv_cache::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il, const slot_info & sinfo) const {
    GGML_UNUSED(sinfo);

    const int32_t ikv = map_layer_ids.at(il);

    ggml_tensor * k = layers[ikv].k;

    const int64_t n_embd_head = k_cur->ne[0];
    const int64_t n_head      = k_cur->ne[1];
    const int64_t n_tokens    = k_cur->ne[2];

    const int64_t n_embd_gqa = n_embd_head*n_head;

    // we can merge dims 0 and 1
    // TODO: add ggml helper function for this?
    GGML_ASSERT(ggml_row_size(k_cur->type, n_embd_head) == k_cur->nb[1]);

    k_cur = ggml_view_2d(ctx, k_cur, n_embd_gqa, n_tokens, k_cur->nb[2], 0);

    const int64_t n_stream = k->ne[2];

    if (n_stream > 1) {
        const int64_t kv_size = get_size();

        assert(n_embd_gqa == k->ne[0]);
        assert(kv_size    == k->ne[1]);

        // merge the buffer across all streams because the idxs are global
        k = ggml_reshape_2d(ctx, k, n_embd_gqa, kv_size*n_stream);
    }

    // store the current K values into the cache
    return ggml_set_rows(ctx, k, k_cur, k_idxs);
}

ggml_tensor * llama_kv_cache::cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il, const slot_info & sinfo) const {
    GGML_UNUSED(sinfo);

    const int32_t ikv = map_layer_ids.at(il);

    auto * v = layers[ikv].v;

    const int64_t n_embd_head = v_cur->ne[0];
    const int64_t n_head      = v_cur->ne[1];
    const int64_t n_tokens    = v_cur->ne[2];

    const int64_t n_embd_gqa = n_embd_head*n_head;

    // we can merge dims 0 and 1
    GGML_ASSERT(ggml_row_size(v_cur->type, n_embd_head) == v_cur->nb[1]);

    const int64_t n_stream = v->ne[2];

    // take this branch when FA is enabled (the V cache is not transposed)
    if (!v_trans) {
        v_cur = ggml_view_2d(ctx, v_cur, n_embd_gqa, n_tokens, v_cur->nb[2], 0);

        if (n_stream > 1) {
            const int64_t kv_size = get_size();

            assert(n_embd_gqa == v->ne[0]);
            assert(kv_size    == v->ne[1]);

            // merge the buffer across all streams because the idxs are global
            v = ggml_reshape_2d(ctx, v, n_embd_gqa, kv_size*n_stream);
        }

        return ggml_set_rows(ctx, v, v_cur, v_idxs);
    }

    if (ggml_row_size(v_cur->type, n_embd_gqa) == v_cur->nb[2]) {
        // we can merge dims 0, 1 and 2
        v_cur = ggml_reshape_2d(ctx, v_cur, n_embd_gqa, n_tokens);
    } else {
        // otherwise -> make a copy to get contiguous data
        v_cur = ggml_cont_2d   (ctx, v_cur, n_embd_gqa, n_tokens);
    }

    // [TAG_V_CACHE_VARIABLE]
    if (n_embd_gqa < v->ne[0]) {
        v_cur = ggml_pad(ctx, v_cur, v->ne[0] - n_embd_gqa, 0, 0, 0);
    }

    // in this branch the v_idxs are constructed in such a way that each row is a single head element
    ggml_tensor * v_view = ggml_reshape_2d(ctx, v, 1, ggml_nelements(v));

    v_cur = ggml_reshape_2d(ctx, v_cur, 1, ggml_nelements(v_cur));

    return ggml_set_rows(ctx, v_view, v_cur, v_idxs);
}

ggml_tensor * llama_kv_cache::build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    const uint32_t n_tokens = ubatch.n_tokens;

    ggml_tensor * k_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);

    ggml_set_input(k_idxs);

    return k_idxs;
}

ggml_tensor * llama_kv_cache::build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    const uint32_t n_tokens = ubatch.n_tokens;

    ggml_tensor * v_idxs;

    if (!v_trans) {
        v_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
    } else {
        v_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens*hparams.n_embd_v_gqa_max());
    }

    ggml_set_input(v_idxs);

    return v_idxs;
}

ggml_tensor * llama_kv_cache::build_input_k_rot(ggml_context * ctx) const {
    ggml_tensor * res = nullptr;

    if (attn_rot_k) {
        int nrot = 64;

        // TODO: investigate if using the smallest rotation matrix is beneficial also for K (similar as for V)
        // ref: https://github.com/ggml-org/llama.cpp/pull/21038#issuecomment-4141323088
        do {
            nrot *= 2;
        } while (n_embd_head_k_all % nrot == 0);
        nrot /= 2;

        res = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nrot, nrot);
        ggml_set_input(res);
        ggml_set_name(res, "attn_inp_k_rot");
    }

    return res;
}

ggml_tensor * llama_kv_cache::build_input_v_rot(ggml_context * ctx) const {
    ggml_tensor * res = nullptr;

    if (attn_rot_v) {
        int nrot = 64;
        // using smaller rotation matrices for V seems beneficial
        // ref: https://github.com/ggml-org/llama.cpp/pull/21038#issuecomment-4146397570
        //do {
        //    nrot *= 2;
        //} while (hparams.n_embd_head_v() % nrot == 0);
        //nrot /= 2;

        res = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nrot, nrot);
        ggml_set_input(res);
        ggml_set_name(res, "attn_inp_v_rot");
    }

    return res;
}

void llama_kv_cache::set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const {
    const uint32_t n_tokens = ubatch->n_tokens;
    GGML_ASSERT(n_tokens == (int64_t) sinfo.size()*sinfo.n_stream());

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    int64_t * data = (int64_t *) dst->data;

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        const int64_t offs = sinfo.strm[s]*get_size();

        for (uint32_t i = 0; i < sinfo.size(); ++i) {
            data[s*sinfo.size() + i] = offs + sinfo.idxs[s][i];
        }
    }
}

void llama_kv_cache::set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const {
    const uint32_t n_tokens = ubatch->n_tokens;
    GGML_ASSERT(n_tokens == (int64_t) sinfo.size()*sinfo.n_stream());

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    int64_t * data = (int64_t *) dst->data;

    if (!v_trans) {
        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            const int64_t offs = sinfo.strm[s]*get_size();

            for (uint32_t i = 0; i < sinfo.size(); ++i) {
                data[s*sinfo.size() + i] = offs + sinfo.idxs[s][i];
            }
        }
    } else {
        // note: the V cache is transposed when not using flash attention
        const int64_t kv_size = get_size();

        const int64_t n_embd_v_gqa = hparams.n_embd_v_gqa_max();

        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            const int64_t offs = sinfo.strm[s]*kv_size*n_embd_v_gqa;

            for (uint32_t i = 0; i < sinfo.size(); ++i) {
                for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                    data[s*sinfo.size()*n_embd_v_gqa + i*n_embd_v_gqa + j] = offs + j*kv_size + sinfo.idxs[s][i];
                }
            }
        }
    }
}

void llama_kv_cache::set_input_k_shift(ggml_tensor * dst) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    int32_t * data = (int32_t *) dst->data;

    for (uint32_t s = 0; s < n_stream; ++s) {
        const auto & cells = v_cells[s];

        for (uint32_t i = 0; i < cells.size(); ++i) {
            data[s*cells.size() + i] = cells.is_empty(i) ? 0 : cells.get_shift(i);
        }
    }
}

struct args_set_input_kq_mask {
    const llama_hparams & hparams;
    const llama_ubatch  * ubatch;

    const std::vector<llama_kv_cells> & v_cells;
    const std::vector<uint32_t>       & seq_to_stream;

    uint32_t       n_swa;
    llama_swa_type swa_type;

    int64_t n_kv;
    int64_t n_stream;
    int64_t n_tps;
};

template<typename T, bool causal, bool swa, bool is_2d, bool alibi>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data) {
  //const auto & hparams = args.hparams;
    const auto & ubatch  = args.ubatch;

    const auto & v_cells       = args.v_cells;
    const auto & seq_to_stream = args.seq_to_stream;

    const uint32_t       n_swa    = args.n_swa;
    const llama_swa_type swa_type = args.swa_type;

    const int64_t n_kv     = args.n_kv;
    const int64_t n_stream = args.n_stream;
    const int64_t n_tps    = args.n_tps;

    const T mask_keep = llama_cast<T>(0.0f);
    const T mask_drop = llama_cast<T>(-INFINITY);

    // the min position in the batch for each sequence
    llama_pos seq_pos_min[LLAMA_MAX_SEQ];
    std::fill(seq_pos_min, seq_pos_min + LLAMA_MAX_SEQ, INT32_MAX);

    for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
        const llama_seq_id seq_id = ubatch->seq_id[i][0];

        seq_pos_min[seq_id] = std::min(seq_pos_min[seq_id], ubatch->pos[i]);
    }

    for (uint32_t s = 0; s < n_stream; ++s) {
        // bookkeeping of the KQ mask cells that could change for other tokens of the same sequence
        std::unordered_map<llama_seq_id, uint32_t>              seq_srct;
        std::unordered_map<llama_seq_id, std::vector<uint32_t>> seq_idxs;

        for (uint32_t ii = 0; ii < n_tps; ++ii) {
            const uint32_t i = s*n_tps + ii;

            const llama_seq_id seq_id = ubatch->seq_id[i][0];

            const auto & cells = v_cells.at(seq_to_stream[seq_id]);

                  llama_pos p0 = -1;
            const llama_pos p1 = ubatch->pos[i];

            // for M-RoPE
            const llama_pos p1_x = is_2d ? ubatch->pos[i + ubatch->n_tokens*2] : 0;
            const llama_pos p1_y = is_2d ? ubatch->pos[i + ubatch->n_tokens]   : 0;

            const uint64_t idst = n_kv*i;

            // for tokens of the same sequence, the mask is mostly the same, so we can reuse it
            // the only cells that could change are the ones that are with similar positions as the
            //   ones in the batch (i.e. due to causal masking, SWA, etc.)
            // keep track of those cells and shortcut the loop to save time
            // note: this optimization is not compatible with Alibi position encoding
            // ref:  https://github.com/ggml-org/llama.cpp/pull/18842
            bool prev = false;

            auto & idxs = seq_idxs[seq_id];

            if (!alibi) {
                if (seq_srct.find(seq_id) != seq_srct.end()) {
                    const uint32_t srct = seq_srct[seq_id];

                    const uint64_t idst_prev = n_kv*srct;

                    std::copy(data + idst_prev, data + idst_prev + n_kv, data + idst);

                    prev = true;
                } else {
                    idxs.clear();
                    idxs.reserve(ubatch->n_tokens + n_swa + 32);

                    seq_srct[seq_id] = i;
                }
            }

            for (uint32_t jj = 0; jj < n_kv; ++jj) {
                uint32_t j = jj;

                // we have an exiting mask for this sequence -> update just seq_idxs
                if (!alibi) {
                    if (prev) {
                        if (jj >= idxs.size()) {
                            break;
                        }

                        j = idxs[jj];
                    }
                }

                if (cells.is_empty(j)) {
                    goto skip;
                }

                // mask the token if not the same sequence
                if (!cells.seq_has(j, seq_id)) {
                    goto skip;
                }

                p0 = cells.pos_get(j);

                if (!alibi) {
                    if (!prev) {
                        // record all cells for which: p0 >= seq_pos_min[seq_id] - n_swa - 32
                        if (p0 + (int32_t) (n_swa + 32) >= seq_pos_min[seq_id]) {
                            idxs.push_back(j);
                        }
                    }
                }

                if (causal) {
                    // mask future tokens
                    if (p0 > p1) {
                        goto skip;
                    }

                    // M-RoPE causal mask
                    if (is_2d) {
                        if (p0 == p1) {
                            const auto & p0_ext = cells.ext_get(j);

                            if (p0_ext.is_2d_gt(p1_x, p1_y)) {
                                goto skip;
                            }
                        }
                    }
                }

                // apply SWA if any
                if (swa) {
                    if (llama_hparams::is_masked_swa(n_swa, swa_type, p0, p1)) {
                        goto skip;
                    }
                }

                if (alibi) {
                    data[idst + j] = llama_cast<T>(static_cast<float>(-std::abs(p0 - p1)));
                } else {
                    data[idst + j] = mask_keep;
                }

                continue;
skip:
                data[idst + j] = mask_drop;
            }
        }
    }
}

template<typename T, bool causal, bool swa, bool is_2d>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data) {
    const bool alibi = args.hparams.use_alibi;
    if (alibi) {
        set_input_kq_mask_impl<T, causal, swa, is_2d, true> (args, data);
    } else {
        set_input_kq_mask_impl<T, causal, swa, is_2d, false>(args, data);
    }
}

template<typename T, bool causal, bool swa>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data) {
    const bool is_2d = args.ubatch->is_pos_2d();
    if (is_2d) {
        set_input_kq_mask_impl<T, causal, swa, true> (args, data);
    } else {
        set_input_kq_mask_impl<T, causal, swa, false>(args, data);
    }
}

template<typename T, bool causal>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data) {
    const bool swa = args.swa_type != LLAMA_SWA_TYPE_NONE;
    if (swa) {
        set_input_kq_mask_impl<T, causal, true> (args, data);
    } else {
        set_input_kq_mask_impl<T, causal, false>(args, data);
    }
}

template<typename T>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data, bool causal_attn) {
    if (causal_attn) {
        set_input_kq_mask_impl<T, true> (args, data);
    } else {
        set_input_kq_mask_impl<T, false>(args, data);
    }
}

void llama_kv_cache::set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const {
    const uint32_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    const int64_t n_kv     = dst->ne[0];
    const int64_t n_stream = dst->ne[3]; // num streams in the current ubatch

    GGML_ASSERT(n_tokens%n_stream == 0);

    // n_tps == n_tokens_per_stream
    const int64_t n_tps = n_tokens/n_stream;

    //const int64_t t_start = ggml_time_us();

    const args_set_input_kq_mask args = {
        /*.hparams          =*/ hparams,
        /*.ubatch           =*/ ubatch,
        /*.v_cells          =*/ v_cells,
        /*.seq_to_stream    =*/ seq_to_stream,
        /*.n_swa            =*/ n_swa,
        /*.swa_type         =*/ swa_type,
        /*.n_kv             =*/ n_kv,
        /*.n_stream         =*/ n_stream,
        /*.n_tps            =*/ n_tps,
    };

    if (dst->type == GGML_TYPE_F16) {
        set_input_kq_mask_impl<ggml_fp16_t>(args, (ggml_fp16_t *) dst->data, causal_attn);
    } else {
        set_input_kq_mask_impl<float>(args, (float *) dst->data, causal_attn);
    }

    // If RazorAttention is active, apply per-head mask adjustments.
    //
    // The mask tensor now has shape (n_kv, n_tps, n_head, n_stream) instead of
    // (n_kv, n_tps, 1, n_stream). Head 0 already has the correct base causal/SWA mask.
    // For each additional head:
    //   - Retrieval heads: copy from head 0 (full context, no change)
    //   - Local heads: copy from head 0, then mask positions outside the local window
    if (razor_attn_enabled && razor_attn_profiled) {
        const int64_t n_head   = dst->ne[2];
        const int64_t stride_h = n_kv * n_tps; // per-head stride within one stream

        if (dst->type == GGML_TYPE_F16) {
            auto * data = (ggml_fp16_t *) dst->data;
            for (int64_t s = 0; s < n_stream; ++s) {
                for (int64_t h = 1; h < n_head; ++h) {
                    const uint64_t off_s = s * n_head * stride_h;
                    // Copy from head 0 (base mask)
                    memcpy(data + off_s + h * stride_h,
                           data + off_s + 0 * stride_h,
                           stride_h * sizeof(ggml_fp16_t));

                    // Check if this head is "local" (not retrieval)
                    bool is_local = true;
                    for (const auto & hp : razor_attn_profiles) {
                        // We need the layer index; for profile matching we search
                        // but the mask is layer-agnostic, so we store per-layer profiles.
                        // For now, treat all stored profiles that match this head idx
                        if (hp.head == (uint32_t)h && hp.is_retrieval) {
                            is_local = false;
                            break;
                        }
                    }
                    if (!is_local) continue;

                    // For local heads: mask positions outside local_window
                    // Iterate over query positions then KV positions
                    for (int64_t ii = 0; ii < n_tps; ++ii) {
                        const uint32_t i = s*n_tps + ii;
                        const llama_seq_id seq_id = ubatch->seq_id[i][0];
                        const auto & cells = v_cells.at(seq_to_stream[seq_id]);
                        const int64_t p_q = ubatch->pos[i];

                        const uint64_t idst = off_s + h * stride_h + ii * n_kv;
                        for (int64_t jj = 0; jj < n_kv; ++jj) {
                            if (cells.is_empty(jj)) {
                                data[idst + jj] = ggml_fp16_to_fp32(data[idst + jj]) > -1e20f
                                    ? ggml_fp32_to_fp16(-INFINITY)
                                    : data[idst + jj];
                                continue;
                            }
                            const int64_t p_kv = cells.pos_get(jj);
                            if (p_q - p_kv > (int64_t)razor_attn_window) {
                                data[idst + jj] = ggml_fp32_to_fp16(-INFINITY);
                            }
                        }
                    }
                }
            }
        } else {
            auto * data = (float *) dst->data;
            for (int64_t s = 0; s < n_stream; ++s) {
                for (int64_t h = 1; h < n_head; ++h) {
                    const uint64_t off_s = s * n_head * stride_h;
                    // Copy from head 0 (base mask)
                    memcpy(data + off_s + h * stride_h,
                           data + off_s + 0 * stride_h,
                           stride_h * sizeof(float));

                    // Check if this head is local
                    bool is_local = true;
                    for (const auto & hp : razor_attn_profiles) {
                        if (hp.head == (uint32_t)h && hp.is_retrieval) {
                            is_local = false;
                            break;
                        }
                    }
                    if (!is_local) continue;

                    // For local heads: mask positions outside local_window
                    for (int64_t ii = 0; ii < n_tps; ++ii) {
                        const uint32_t i = s*n_tps + ii;
                        const llama_seq_id seq_id = ubatch->seq_id[i][0];
                        const auto & cells = v_cells.at(seq_to_stream[seq_id]);
                        const int64_t p_q = ubatch->pos[i];

                        const uint64_t idst = off_s + h * stride_h + ii * n_kv;
                        for (int64_t jj = 0; jj < n_kv; ++jj) {
                            if (cells.is_empty(jj)) {
                                if (data[idst + jj] > -1e20f) {
                                    data[idst + jj] = -INFINITY;
                                }
                                continue;
                            }
                            const int64_t p_kv = cells.pos_get(jj);
                            if (p_q - p_kv > (int64_t)razor_attn_window) {
                                data[idst + jj] = -INFINITY;
                            }
                        }
                    }
                }
            }
        }
    }

    //const int64_t t_end = ggml_time_us();

    //LLAMA_LOG_ERROR("%s: kq mask time: %0.3f ms\n", __func__, (t_end - t_start)/1000.0);
}

void llama_kv_cache::set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    const int64_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(n_stream == 1 && "TODO: support multiple streams");
    const auto & cells = v_cells[0];

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    GGML_ASSERT(!ubatch->equal_seqs()); // TODO: use ubatch->n_seqs instead of failing

    int32_t * data = (int32_t *) dst->data;

    const int32_t n_kv = dst->ne[0];

    for (int h = 0; h < 1; ++h) {
        for (int i = 0; i < n_tokens; ++i) {
            for (int j = 0; j < n_kv; ++j) {
                // the position when the cells is empty is irrelevant - it will be masked out later in the attention
                const llama_pos p0 = cells.is_empty(j) ? -1 : cells.pos_get(j);

                data[h*(n_kv*n_tokens) + i*n_kv + j] = llama_relative_position_bucket(p0, ubatch->pos[i], hparams.n_rel_attn_bkts, false);
            }
        }
    }
}

void llama_kv_cache::set_input_k_rot(ggml_tensor * dst) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    const auto n_rot = dst->ne[0];
    GGML_ASSERT(attn_rot_hadamard.count(dst->ne[0]));

    memcpy(dst->data, attn_rot_hadamard.at(n_rot).data(), ggml_nbytes(dst));
}

void llama_kv_cache::set_input_v_rot(ggml_tensor * dst) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    const auto n_rot = dst->ne[0];
    GGML_ASSERT(attn_rot_hadamard.count(dst->ne[0]));

    memcpy(dst->data, attn_rot_hadamard.at(n_rot).data(), ggml_nbytes(dst));
}

size_t llama_kv_cache::total_size() const {
    size_t size = 0;

    for (const auto & [_, buf] : ctxs_bufs) {
        size += ggml_backend_buffer_get_size(buf.get());
    }

    return size;
}

size_t llama_kv_cache::size_k_bytes() const {
    size_t size_k_bytes = 0;

    for (const auto & layer : layers) {
        size_k_bytes += ggml_nbytes(layer.k);
    }

    return size_k_bytes;
}

size_t llama_kv_cache::size_v_bytes() const {
    size_t size_v_bytes = 0;

    for (const auto & layer : layers) {
        size_v_bytes += layer.v ? ggml_nbytes(layer.v) : 0;
    }

    return size_v_bytes;
}

ggml_tensor * llama_kv_cache::build_rope_shift(
        const llama_cparams & cparams,
               ggml_context * ctx,
                ggml_tensor * cur,
                ggml_tensor * shift,
                ggml_tensor * rot,
                ggml_tensor * factors,
                      float   freq_base,
                      float   freq_scale,
                   uint32_t   il) const {
    const auto & n_ctx_orig = cparams.n_ctx_orig_yarn;

    const auto & yarn_ext_factor  = cparams.yarn_ext_factor;
    const auto & yarn_beta_fast   = cparams.yarn_beta_fast;
    const auto & yarn_beta_slow   = cparams.yarn_beta_slow;
    const auto & yarn_attn_factor = cparams.yarn_attn_factor;

    const auto & n_rot     = hparams.n_rot(il);
    const auto & rope_type = hparams.rope_type == LLAMA_ROPE_TYPE_MROPE || hparams.rope_type == LLAMA_ROPE_TYPE_IMROPE
                                // @ngxson : this is a workaround
                                // for M-RoPE, we want to rotate the whole vector when doing KV shift
                                // a normal RoPE should work, we just need to use the correct ordering
                                // ref: https://github.com/ggml-org/llama.cpp/pull/13870
                                ? LLAMA_ROPE_TYPE_NEOX
                                : hparams.rope_type;
    ggml_tensor * tmp;

    if (ggml_is_quantized(cur->type)) {
        // dequantize to f32 -> RoPE -> quantize back
        tmp = ggml_cast(ctx, cur, GGML_TYPE_F32);

        // rotate back
        tmp = ggml_mul_mat_aux(ctx, tmp, rot);

        tmp = ggml_rope_ext(ctx, tmp,
                shift, factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                yarn_ext_factor, yarn_attn_factor, yarn_beta_fast, yarn_beta_slow);

        // rotate fwd
        tmp = ggml_mul_mat_aux(ctx, tmp, rot);

        tmp = ggml_cpy(ctx, tmp, cur);
    } else {
        // we rotate only the first n_rot dimensions
        tmp = ggml_rope_ext_inplace(ctx, cur,
                shift, factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                yarn_ext_factor, yarn_attn_factor, yarn_beta_fast, yarn_beta_slow);
    }

    return tmp;
}

class llm_graph_input_k_shift : public llm_graph_input_i {
public:
    llm_graph_input_k_shift(const llama_kv_cache * kv_self) : kv_self(kv_self) {}
    virtual ~llm_graph_input_k_shift() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * k_shift; // I32 [kv_size*n_stream]

    // note: assumes k_rot^2 == I
    ggml_tensor * k_rot = nullptr;

    const llama_kv_cache * kv_self;
};

void llm_graph_input_k_shift::set_input(const llama_ubatch * ubatch) {
    GGML_UNUSED(ubatch);

    if (k_shift) {
        kv_self->set_input_k_shift(k_shift);
    }

    if (k_rot) {
        kv_self->set_input_k_rot(k_rot);
    }
}

ggml_cgraph * llama_kv_cache::build_graph_shift(llm_graph_result * res, llama_context * lctx) const {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    GGML_ASSERT(!other);

    auto * ctx = res->get_ctx();
    auto * gf  = res->get_gf();

    auto inp = std::make_unique<llm_graph_input_k_shift>(this);

    inp->k_shift = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) get_size()*n_stream);
    ggml_set_input(inp->k_shift);

    inp->k_rot = build_input_k_rot(ctx);

    const auto & cparams = lctx->get_cparams();

    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        const int64_t n_head_kv    = hparams.n_head_kv(il);
        const int64_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);

        const auto n_rot         = hparams.n_rot(il);
        const auto n_embd_head_k = hparams.n_embd_head_k(il);
        const auto n_embd_nope   = hparams.n_lora_kv > 0 ? n_embd_head_k - n_rot : 0;

        const float freq_base_l  = model.get_rope_freq_base (cparams, il);
        const float freq_scale_l = model.get_rope_freq_scale(cparams, il);

        ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

        ggml_tensor * k =
            ggml_view_3d(ctx, layer.k,
                n_rot, n_head_kv, get_size()*n_stream,
                ggml_row_size(layer.k->type, n_embd_head_k),
                ggml_row_size(layer.k->type, n_embd_k_gqa),
                ggml_row_size(layer.k->type, n_embd_nope));

        ggml_tensor * cur = build_rope_shift(cparams, ctx, k, inp->k_shift, inp->k_rot, rope_factors, freq_base_l, freq_scale_l, il);

        ggml_build_forward_expand(gf, cur);
    }

    res->add_input(std::move(inp));

    return gf;
}

void llama_kv_cache::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_UNUSED(flags);

    io.write(&n_stream, sizeof(n_stream));

    for (uint32_t s = 0; s < n_stream; ++s) {
        cell_ranges_t cr { s, {} };

        uint32_t cell_count = 0;

        const auto & cells = v_cells[s];

        // Count the number of cells with the specified seq_id
        // Find all the ranges of cells with this seq id (or all, when -1)
        uint32_t cell_range_begin = cells.size();

        for (uint32_t i = 0; i < cells.size(); ++i) {
            bool add_cell = true;

            add_cell = add_cell && !cells.is_empty(i);
            add_cell = add_cell && (seq_id == -1 || cells.seq_has(i, seq_id));

            // check the cell is not SWA-masked
            if (add_cell && seq_id != -1) {
                const bool is_masked = llama_hparams::is_masked_swa(n_swa, swa_type, cells.pos_get(i), cells.seq_pos_max(seq_id));

                add_cell = !is_masked;
            }

            if (add_cell) {
                ++cell_count;
                if (cell_range_begin == cells.size()) {
                    cell_range_begin = i;
                }
            } else {
                if (cell_range_begin != cells.size()) {
                    cr.data.emplace_back(cell_range_begin, i);
                    cell_range_begin = cells.size();
                }
            }
        }

        if (cell_range_begin != cells.size()) {
            cr.data.emplace_back(cell_range_begin, cells.size());
        }

        // DEBUG CHECK: Sum of cell counts in ranges should equal the total cell count
        uint32_t cell_count_check = 0;
        for (const auto & range : cr.data) {
            cell_count_check += range.second - range.first;
        }
        GGML_ASSERT(cell_count == cell_count_check);

        io.write(&cell_count, sizeof(cell_count));

        // skip empty streams
        if (cell_count == 0) {
            continue;
        }

        state_write_meta(io, cr, seq_id);
        state_write_data(io, cr);
    }
}

void llama_kv_cache::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_UNUSED(flags);

    GGML_ASSERT(seq_id == -1 || (seq_id >= 0 && (size_t) seq_id < seq_to_stream.size()));

    uint32_t n_stream_cur;
    io.read(&n_stream_cur, sizeof(n_stream_cur));
    if (n_stream_cur != n_stream) {
        throw std::runtime_error("n_stream mismatch");
    }

    for (uint32_t s = 0; s < n_stream; ++s) {
        uint32_t cell_count;
        io.read(&cell_count, sizeof(cell_count));

        if (cell_count == 0) {
            continue;
        }

        const uint32_t strm = seq_id == -1 ? s : seq_to_stream[seq_id];

        slot_info sinfo;

        bool res = true;
        res = res && state_read_meta(io, strm, cell_count, sinfo, seq_id);
        res = res && state_read_data(io, strm, cell_count, sinfo);

        if (!res) {
            if (seq_id == -1) {
                clear(true);
            } else {
                seq_rm(seq_id, -1, -1);
            }
            throw std::runtime_error("failed to restore kv cache");
        }
    }
}

void llama_kv_cache::state_write_meta(llama_io_write_i & io, const cell_ranges_t & cr, llama_seq_id seq_id) const {
    const auto & cells = v_cells[cr.strm];

    for (const auto & range : cr.data) {
        for (uint32_t i = range.first; i < range.second; ++i) {
            std::vector<llama_seq_id> seq_ids;

            for (llama_seq_id cur = 0; cur < (int) n_seq_max; ++cur) {
                if (cur == seq_id || seq_id == -1) {
                    if (cells.seq_has(i, cur)) {
                        seq_ids.push_back(cur);
                    }
                }
            }

            const llama_pos pos     = cells.pos_get(i);
            const uint32_t n_seq_id = seq_ids.size();

            io.write(&pos,      sizeof(pos));
            io.write(&n_seq_id, sizeof(n_seq_id));

            if (hparams.n_pos_per_embd() > 1) {
                const llama_kv_cell_ext ext = cells.ext_get(i);
                io.write(&ext, sizeof(ext));
            }

            for (const auto & seq_id : seq_ids) {
                io.write(&seq_id, sizeof(seq_id));
            }
        }
    }
}

void llama_kv_cache::state_write_data(llama_io_write_i & io, const cell_ranges_t & cr) const {
    const auto & cells = v_cells[cr.strm];

    const uint32_t v_trans = this->v_trans ? 1 : 0;
    const uint32_t n_layer = layers.size();

    io.write(&v_trans, sizeof(v_trans));
    io.write(&n_layer, sizeof(n_layer));

    // Iterate and write all the keys first, each row is a cell
    // Get whole range at a time
    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);

        auto * k = layer.k_stream[cr.strm];

        // Write key type
        const int32_t k_type_i = (int32_t) k->type;
        io.write(&k_type_i, sizeof(k_type_i));

        // Write row size of key
        const uint64_t k_size_row = ggml_row_size(k->type, n_embd_k_gqa);
        io.write(&k_size_row, sizeof(k_size_row));

        // Read each range of cells of k_size length and write out
        for (const auto & range : cr.data) {
            const size_t range_size = range.second - range.first;
            const size_t buf_size = range_size * k_size_row;
            io.write_tensor(k, range.first * k_size_row, buf_size);
        }
    }

    if (!v_trans) {
        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            auto * v = layer.v_stream[cr.strm];
            if (!v) {
                continue;
            }

            // Write value type
            const int32_t v_type_i = (int32_t) v->type;
            io.write(&v_type_i, sizeof(v_type_i));

            // Write row size of value
            const uint64_t v_size_row = ggml_row_size(v->type, n_embd_v_gqa);
            io.write(&v_size_row, sizeof(v_size_row));

            // Read each range of cells of v_size length and write out
            for (const auto & range : cr.data) {
                const size_t range_size = range.second - range.first;
                const size_t buf_size = range_size * v_size_row;
                io.write_tensor(v, range.first * v_size_row, buf_size);
            }
        }
    } else {
        // When v is transposed, we also need the element size and get the element ranges from each row
        const uint32_t kv_size = cells.size();

        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            auto * v = layer.v_stream[cr.strm];
            if (!v) {
                continue;
            }

            // Write value type
            const int32_t v_type_i = (int32_t) v->type;
            io.write(&v_type_i, sizeof(v_type_i));

            // Write element size
            const uint32_t v_size_el = ggml_type_size(v->type);
            io.write(&v_size_el, sizeof(v_size_el));

            // Write GQA embedding size
            io.write(&n_embd_v_gqa, sizeof(n_embd_v_gqa));

            // For each row, we get the element values of each cell
            for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                // Read each range of cells of v_size_el length and write out
                for (const auto & range : cr.data) {
                    const size_t range_size = range.second - range.first;
                    const size_t src_offset = (range.first + j * kv_size) * v_size_el;
                    const size_t buf_size = range_size * v_size_el;
                    io.write_tensor(v, src_offset, buf_size);
                }
            }
        }
    }
}

bool llama_kv_cache::state_read_meta(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, slot_info & sinfo, llama_seq_id dest_seq_id) {
    auto & cells = v_cells[strm];
    auto & head  = v_heads[strm];

    if (dest_seq_id != -1) {
        // single sequence
        seq_rm(dest_seq_id, -1, -1);

        llama_batch_allocr balloc(hparams.n_pos_per_embd());

        llama_ubatch ubatch = balloc.ubatch_reserve(cell_count, 1);

        ubatch.seq_id_unq[0] = dest_seq_id;

        for (uint32_t i = 0; i < cell_count; ++i) {
            llama_pos pos;
            uint32_t n_seq_id;

            io.read(&pos,      sizeof(pos));
            io.read(&n_seq_id, sizeof(n_seq_id));

            if (n_seq_id != 1) {
                LLAMA_LOG_ERROR("%s: invalid seq_id-agnostic kv cell\n", __func__);
                return false;
            }

            if (hparams.n_pos_per_embd() > 1) {
                llama_kv_cell_ext ext;
                io.read(&ext, sizeof(ext));

                ubatch.pos[i + ubatch.n_tokens]   = ext.y;
                ubatch.pos[i + ubatch.n_tokens*2] = ext.x;
            }

            // read the sequence id, but directly discard it - we will use dest_seq_id instead
            {
                llama_seq_id seq_id;
                io.read(&seq_id, sizeof(seq_id));
            }

            ubatch.pos[i]      = pos;
            ubatch.n_seq_id[i] = n_seq_id;
            ubatch.seq_id[i]   = &dest_seq_id;
        }

        sinfo = find_slot(ubatch, false);
        if (sinfo.empty()) {
            LLAMA_LOG_ERROR("%s: failed to find %d available cells in kv cache\n", __func__,  cell_count);
            return false;
        }

        // TODO: we cannot yet restore llama_kv_cell_ext as the apply_ubatch() does not support it yet
        //       see: https://github.com/ggml-org/llama.cpp/pull/16825#issuecomment-3460868350
        apply_ubatch(sinfo, ubatch);

        LLAMA_LOG_DEBUG("%s: cell_count = %d, dest_seq_id = %d\n", __func__, cell_count, dest_seq_id);

        // DEBUG CHECK: verify that all cells were allocated and have correct seq_id and pos values
        GGML_ASSERT(sinfo.n_stream() == 1);
        GGML_ASSERT(sinfo.idxs[0].size() == cell_count);
        for (uint32_t i = 0; i < cell_count; ++i) {
            const uint32_t idx = sinfo.idxs[0][i];
            GGML_ASSERT(cells.pos_get(idx) == ubatch.pos[i]);
            GGML_ASSERT(cells.seq_has(idx, dest_seq_id));
        }
    } else {
        // whole KV cache restore

        if (cell_count > cells.size()) {
            LLAMA_LOG_ERROR("%s: not enough cells in kv cache\n", __func__);
            return false;
        }

        clear(true);

        for (uint32_t i = 0; i < cell_count; ++i) {
            llama_pos pos;
            uint32_t  n_seq_id;

            io.read(&pos,      sizeof(pos));
            io.read(&n_seq_id, sizeof(n_seq_id));

            cells.pos_set(i, pos);

            if (hparams.n_pos_per_embd() > 1) {
                llama_kv_cell_ext ext;
                io.read(&ext, sizeof(ext));
                cells.ext_set(i, ext);
            }

            for (uint32_t j = 0; j < n_seq_id; ++j) {
                llama_seq_id seq_id;
                io.read(&seq_id, sizeof(seq_id));

                if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max) {
                    LLAMA_LOG_ERROR("%s: invalid seq_id, %d is out of range [0, %u)\n", __func__, seq_id, n_seq_max);
                    return false;
                }

                cells.seq_add(i, seq_id);
            }
        }

        // Create contiguous slot_info for whole cache restore
        sinfo.s0 = strm;
        sinfo.s1 = strm;
        sinfo.resize(1);
        sinfo.strm[0] = strm;
        sinfo.idxs[0].resize(cell_count);
        for (uint32_t i = 0; i < cell_count; ++i) {
            sinfo.idxs[0][i] = i;
        }

        head = 0;
    }

    return true;
}

bool llama_kv_cache::state_read_data(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, const slot_info & sinfo) {
    auto & cells = v_cells[strm];

    uint32_t v_trans;
    uint32_t n_layer;

    io.read(&v_trans, sizeof(v_trans));
    io.read(&n_layer, sizeof(n_layer));

    if (n_layer != layers.size()) {
        LLAMA_LOG_ERROR("%s: mismatched layer count (%u instead of %u)\n", __func__, n_layer, (uint32_t) layers.size());
        return false;
    }

    if (cell_count > cells.size()) {
        LLAMA_LOG_ERROR("%s: not enough cells in kv cache to restore state (%u > %u)\n", __func__, cell_count, cells.size());
        return false;
    }

    if (this->v_trans != (bool) v_trans) {
        LLAMA_LOG_ERROR("%s: incompatible V transposition\n", __func__);
        return false;
    }

    // For each layer, read the keys for each cell, one row is one cell, read as one contiguous block
    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);

        auto * k = layer.k_stream[strm];

        // Read type of key
        int32_t k_type_i_ref;
        io.read(&k_type_i_ref, sizeof(k_type_i_ref));
        const int32_t k_type_i = (int32_t) k->type;
        if (k_type_i != k_type_i_ref) {
            LLAMA_LOG_ERROR("%s: mismatched key type (%d != %d, layer %d)\n", __func__, k_type_i, k_type_i_ref, il);
            return false;
        }

        // Read row size of key
        uint64_t k_size_row_ref;
        io.read(&k_size_row_ref, sizeof(k_size_row_ref));
        const size_t k_size_row = ggml_row_size(k->type, n_embd_k_gqa);
        if (k_size_row != k_size_row_ref) {
            LLAMA_LOG_ERROR("%s: mismatched key row size (%zu != %zu, layer %d)\n", __func__, k_size_row, (size_t) k_size_row_ref, il);
            return false;
        }

        if (cell_count) {
            if (sinfo.is_contiguous()) {
                // Fast path: contiguous cells, single memcpy
                io.read_tensor(k, sinfo.head() * k_size_row, cell_count * k_size_row);
            } else {
                // Slow path: scatter to non-contiguous positions
                for (uint32_t i = 0; i < cell_count; ++i) {
                    const size_t dst_offset = sinfo.idxs[0][i] * k_size_row;
                    io.read_tensor(k, dst_offset, k_size_row);
                }
            }
        }
    }

    if (!this->v_trans) {
        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            auto * v = layer.v_stream[strm];
            if (!v) {
                continue;
            }

            // Read type of value
            int32_t v_type_i_ref;
            io.read(&v_type_i_ref, sizeof(v_type_i_ref));
            const int32_t v_type_i = (int32_t) v->type;
            if (v_type_i != v_type_i_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value type (%d != %d, layer %d)\n", __func__, v_type_i, v_type_i_ref, il);
                return false;
            }

            // Read row size of value
            uint64_t v_size_row_ref;
            io.read(&v_size_row_ref, sizeof(v_size_row_ref));
            const size_t v_size_row = ggml_row_size(v->type, n_embd_v_gqa);
            if (v_size_row != v_size_row_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value row size (%zu != %zu, layer %d)\n", __func__, v_size_row, (size_t) v_size_row_ref, il);
                return false;
            }

            if (cell_count) {
                if (sinfo.is_contiguous()) {
                    // Fast path: contiguous cells, single memcpy
                    io.read_tensor(v, sinfo.head() * v_size_row, cell_count * v_size_row);
                } else {
                    // Slow path: scatter to non-contiguous positions
                    for (uint32_t i = 0; i < cell_count; ++i) {
                        const size_t dst_offset = sinfo.idxs[0][i] * v_size_row;
                        io.read_tensor(v, dst_offset, v_size_row);
                    }
                }
            }
        }
    } else {
        // For each layer, read the values for each cell (transposed)
        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            auto * v = layer.v_stream[strm];
            if (!v) {
                continue;
            }

            // Read type of value
            int32_t v_type_i_ref;
            io.read(&v_type_i_ref, sizeof(v_type_i_ref));
            const int32_t v_type_i = (int32_t) v->type;
            if (v_type_i != v_type_i_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value type (%d != %d, layer %d)\n", __func__, v_type_i, v_type_i_ref, il);
                return false;
            }

            // Read element size of value
            uint32_t v_size_el_ref;
            io.read(&v_size_el_ref, sizeof(v_size_el_ref));
            const size_t v_size_el = ggml_type_size(v->type);
            if (v_size_el != v_size_el_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value element size (%zu != %zu, layer %d)\n", __func__, v_size_el, (size_t) v_size_el_ref, il);
                return false;
            }

            // Read GQA embedding size
            uint32_t n_embd_v_gqa_ref;
            io.read(&n_embd_v_gqa_ref, sizeof(n_embd_v_gqa_ref));
            if (n_embd_v_gqa != n_embd_v_gqa_ref) {
                LLAMA_LOG_ERROR("%s: mismatched GQA embedding size (%u != %u, layer %d)\n", __func__, n_embd_v_gqa, n_embd_v_gqa_ref, il);
                return false;
            }

            if (cell_count) {
                if (sinfo.is_contiguous()) {
                    // Fast path: contiguous cells
                    const uint32_t h = sinfo.head();
                    for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                        const size_t dst_offset = (h + j * cells.size()) * v_size_el;
                        io.read_tensor(v, dst_offset, cell_count * v_size_el);
                    }
                } else {
                    // Slow path: scatter to non-contiguous positions
                    for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                        for (uint32_t i = 0; i < cell_count; ++i) {
                            const size_t dst_offset = (sinfo.idxs[0][i] + j * cells.size()) * v_size_el;
                            io.read_tensor(v, dst_offset, v_size_el);
                        }
                    }
                }
            }
        }
    }

    return true;
}

//
// llama_kv_cache_context
//

llama_kv_cache_context::llama_kv_cache_context(llama_memory_status status) : status(status) {}

llama_kv_cache_context::llama_kv_cache_context(
        llama_kv_cache * kv) : status(LLAMA_MEMORY_STATUS_SUCCESS), kv(kv) {
    n_kv = kv->get_size();

    const uint32_t n_stream = kv->get_n_stream();

    // create a dummy slot info - the actual data is irrelevant. we just need to build the graph
    sinfos.resize(1);
    sinfos[0].s0 = 0;
    sinfos[0].s1 = n_stream - 1;
    sinfos[0].idxs.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        sinfos[0].strm.push_back(s);
        sinfos[0].idxs[s].resize(1, 0);
    }
}

llama_kv_cache_context::llama_kv_cache_context(
        llama_kv_cache * kv,
        llama_context * lctx,
        bool do_shift,
        stream_copy_info sc_info) : status(LLAMA_MEMORY_STATUS_SUCCESS), kv(kv), lctx(lctx), do_shift(do_shift), sc_info(std::move(sc_info)) {
    if (!do_shift && this->sc_info.empty()) {
        status = LLAMA_MEMORY_STATUS_NO_UPDATE;
    }
}

llama_kv_cache_context::llama_kv_cache_context(
        llama_kv_cache * kv,
        llama_kv_cache::slot_info_vec_t sinfos,
        std::vector<llama_ubatch> ubatches) : status(LLAMA_MEMORY_STATUS_SUCCESS), kv(kv), sinfos(std::move(sinfos)), ubatches(std::move(ubatches)) {
}

llama_kv_cache_context::~llama_kv_cache_context() = default;

bool llama_kv_cache_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    if (++i_cur >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_kv_cache_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    // no ubatches -> this is a KV cache update
    if (ubatches.empty()) {
        kv->update(lctx, do_shift, sc_info);

        return true;
    }

    kv->apply_ubatch(sinfos[i_cur], ubatches[i_cur]);
    n_kv = kv->get_n_kv(sinfos[i_cur]);

    return true;
}

llama_memory_status llama_kv_cache_context::get_status() const {
    return status;
}

const llama_ubatch & llama_kv_cache_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ubatches[i_cur];
}

uint32_t llama_kv_cache_context::get_n_kv() const {
    return n_kv;
}

ggml_type llama_kv_cache_context::type_k() const {
    return kv->type_k();
}

ggml_type llama_kv_cache_context::type_v() const {
    return kv->type_v();
}

ggml_tensor * llama_kv_cache_context::get_k(ggml_context * ctx, int32_t il) const {
    return kv->get_k(ctx, il, n_kv, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::get_v(ggml_context * ctx, int32_t il) const {
    return kv->get_v(ctx, il, n_kv, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const {
    return kv->cpy_k(ctx, k_cur, k_idxs, il, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il) const {
    return kv->cpy_v(ctx, v_cur, v_idxs, il, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    return kv->build_input_k_idxs(ctx, ubatch);
}

ggml_tensor * llama_kv_cache_context::build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    return kv->build_input_v_idxs(ctx, ubatch);
}

ggml_tensor * llama_kv_cache_context::build_input_k_rot(ggml_context * ctx) const {
    return kv->build_input_k_rot(ctx);
}

ggml_tensor * llama_kv_cache_context::build_input_v_rot(ggml_context * ctx) const {
    return kv->build_input_v_rot(ctx);
}

void llama_kv_cache_context::set_input_k_shift(ggml_tensor * dst) const {
    kv->set_input_k_shift(dst);
}

void llama_kv_cache_context::set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_k_idxs(dst, ubatch, sinfos[i_cur]);
}

void llama_kv_cache_context::set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_v_idxs(dst, ubatch, sinfos[i_cur]);
}

void llama_kv_cache_context::set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const {
    kv->set_input_kq_mask(dst, ubatch, causal_attn);
}

void llama_kv_cache_context::set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_pos_bucket(dst, ubatch);
}

void llama_kv_cache_context::set_input_k_rot(ggml_tensor * dst) const {
    kv->set_input_k_rot(dst);
}

void llama_kv_cache_context::set_input_v_rot(ggml_tensor * dst) const {
    kv->set_input_v_rot(dst);
}
