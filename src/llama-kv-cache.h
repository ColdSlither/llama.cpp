#pragma once

#include "llama-batch.h"
#include "llama-graph.h"
#include "llama-kv-cells.h"
#include "llama-memory.h"

#include <string>
#include <unordered_map>
#include <vector>

//
// xKV basis — cross-layer SVD factorization for KV-cache compression
//

struct xkv_basis {
    int32_t n_singular = 32;   // number of singular vectors (rank)
    int32_t n_embd_k   = 0;    // K embedding dimension
    int32_t n_embd_v   = 0;    // V embedding dimension

    // Shared basis matrices (n_singular x n_embd), row-major
    // Each row is one singular vector
    std::vector<float> U_k;  // [n_singular * n_embd_k]
    std::vector<float> U_v;  // [n_singular * n_embd_v]

    // Per-layer coefficients (n_layers x n_singular)
    // coeffs_k[l][s] = projection of layer l's K onto singular vector s
    std::vector<std::vector<float>> coeffs_k;
    std::vector<std::vector<float>> coeffs_v;

    bool is_valid() const {
        return n_singular > 0 && n_embd_k > 0 && n_embd_v > 0
            && !U_k.empty() && !U_v.empty()
            && (int32_t)U_k.size() == n_singular * n_embd_k
            && (int32_t)U_v.size() == n_singular * n_embd_v
            && coeffs_k.size() == coeffs_v.size()
            && !coeffs_k.empty();
    }
};

struct llama_cparams;
struct llama_hparams;
struct llama_model;
struct llama_context;

//
// llama_kv_cache
//

class llama_kv_cache : public llama_memory_i {
public:
    struct stream_copy_info {
        bool empty() const {
            assert(ssrc.size() == sdst.size());
            return ssrc.empty();
        }

        std::vector<uint32_t> ssrc;
        std::vector<uint32_t> sdst;
    };

    // for each ubatch, create a slot_info that contains information about where the ubatch should be inserted in the
    //   KV cells. for example, cell indices for each token, such that: token[i] -> goes to cells[idxs[i]]
    struct slot_info {
        // data for ggml_set_rows
        using idx_vec_t = std::vector<uint32_t>;

        // number of streams: ns = s1 - s0 + 1
        uint32_t s0;
        uint32_t s1;

        std::vector<llama_seq_id> strm; // [ns]
        std::vector<idx_vec_t>    idxs; // [ns]

        uint32_t head() const {
            GGML_ASSERT(idxs.size() == 1);
            GGML_ASSERT(!idxs[0].empty());

            return idxs[0][0];
        }

        void resize(size_t n) {
            strm.resize(n);
            idxs.resize(n);
        }

        size_t size() const {
            GGML_ASSERT(idxs.size() == strm.size());
            GGML_ASSERT(!idxs.empty());

            return idxs[0].size();
        }

        size_t n_stream() const {
            return strm.size();
        }

        bool empty() const {
            return idxs.empty();
        }

        void clear() {
            idxs.clear();
        }

        // check if indices are contiguous starting from head()
        bool is_contiguous() const {
            if (idxs.empty() || idxs[0].empty()) {
                return true;
            }
            if (idxs.size() > 1) {
                return false;
            }
            const uint32_t h = idxs[0][0];
            for (size_t i = 0; i < idxs[0].size(); ++i) {
                if (idxs[0][i] != h + i) {
                    return false;
                }
            }
            return true;
        }
    };

    using slot_info_vec_t = std::vector<slot_info>;

    // TODO: refactor the memory instances to not depend on `llama_model`
    //       instead pass all necessary info (e.g. hparams, dev layers, arch, etc.) directly
    //       likely through `struct llama_memory_params`
    llama_kv_cache(
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
        const  layer_share_cb & share);

    ~llama_kv_cache() = default;

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    bool get_can_shift() const override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    //
    // llama_kv_cache specific API
    //

    // KVzip compression
    void kvzip_compress();
    // Setter methods for KVzip/DepthKV/RazorAttention/FastKV/xKV params
    void set_kvzip_enabled(bool v)        { kvzip_enabled = v; }
    void set_kvzip_keep_ratio(float v)    { kvzip_keep_ratio = v; }
    void set_kvzip_trigger(int v)         { kvzip_trigger = v; }
    void set_kvzip_min_latest(int v)      { kvzip_min_latest = v; }
    void set_kvzip_pos_bias(float v)      { kvzip_pos_bias = v; }
    void set_depthkv_enabled(bool v)      { depthkv_enabled = v; }
    void set_depthkv_min_keep(float v)    { depthkv_min_keep = v; }
    void set_depthkv_max_keep(float v)    { depthkv_max_keep = v; }
    void set_fastkv_enabled(bool v)       { fastkv_enabled = v; }
    void set_fastkv_tsp_rate(float v)     { fastkv_tsp_rate = v; }
    void set_fastkv_tsp_layer(int v)      { fastkv_tsp_layer = v; }
    void set_fastkv_kv_retention(float v) { fastkv_kv_retention = v; }
    void set_xkv_enabled(bool v)          { xkv_enabled = v; }
    void set_xkv_rank(int v)              { xkv_rank = v; }
    void set_xkv_profile_path(const std::string & v) { xkv_profile_path = v; }
    void set_razor_attn_enabled(bool v)   { razor_attn_enabled = v; }
    void set_razor_attn_window(int v)     { razor_attn_window = v; }

    // RazorAttention — head-type specialization
    void razor_apply_mask(ggml_tensor * kq_mask) const;
    void razor_compute_compensation(int _n_kv) const;
    void razor_save_profile(const char * path) const;
    bool razor_load_profile(const char * path);
    void razor_profile_heads(const ggml_tensor * k_first_layer);

    // FastKV — token-selective propagation + KV retention
    void fastkv_select_prefill();
    void fastkv_compress();
    const std::vector<int> & fastkv_get_selected() const { return fastkv_selected_indices; }

    // xKV — cross-layer SVD factorization
    void xkv_compute_basis();
    void xkv_project();
    void xkv_reconstruct_k(ggml_tensor * k_out, const ggml_tensor * coeffs_in, int32_t il) const;
    void xkv_reconstruct_v(ggml_tensor * v_out, const ggml_tensor * coeffs_in, int32_t il) const;
    void xkv_save_profile(const char * path) const;
    bool xkv_load_profile(const char * path);

    uint32_t get_size()     const;
    uint32_t get_n_stream() const;

    bool get_has_shift() const;

    ggml_type type_k() const;
    ggml_type type_v() const;

    std::vector<uint32_t> get_layer_ids() const;
    ggml_tensor * get_k_storage(int32_t il) const;

    //
    // graph_build API
    //

    uint32_t get_n_kv(const slot_info & sinfo) const;

    // get views of the current state of the cache
    ggml_tensor * get_k(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const;
    ggml_tensor * get_v(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const;

    // store k_cur and v_cur in the cache based on the provided head location
    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il, const slot_info & sinfo) const;
    ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il, const slot_info & sinfo) const;

    //
    // preparation API
    //

    // find places for the provided ubatches in the cache, returns the slot infos
    // return empty vector on failure
    slot_info_vec_t prepare(const std::vector<llama_ubatch> & ubatches);

    bool update(llama_context * lctx, bool do_shift, const stream_copy_info & sc_info);

    // find a slot of kv cells that can hold the ubatch
    // if cont == true, then the slot must be continuous
    // return empty slot_info on failure
    slot_info find_slot(const llama_ubatch & ubatch, bool cont) const;

    // emplace the ubatch context into slot: [sinfo.idxs[0...ubatch.n_tokens - 1]]
    void apply_ubatch(const slot_info & sinfo, const llama_ubatch & ubatch);

    //
    // input API
    //

    ggml_tensor * build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;
    ggml_tensor * build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;

    ggml_tensor * build_input_k_rot(ggml_context * ctx) const;
    ggml_tensor * build_input_v_rot(ggml_context * ctx) const;

    void set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const;
    void set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const;

    void set_input_k_shift(ggml_tensor * dst) const;

    void set_input_kq_mask   (ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const;
    void set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const;

    void set_input_k_rot(ggml_tensor * dst) const;
    void set_input_v_rot(ggml_tensor * dst) const;

private:
    const llama_model & model;
    const llama_hparams & hparams;

    struct kv_layer {
        // layer index in the model
        // note: can be different from the layer index in the KV cache
        uint32_t il;

        ggml_tensor * k;
        ggml_tensor * v;

        std::vector<ggml_tensor *> k_stream;
        std::vector<ggml_tensor *> v_stream;
    };

    bool v_trans = true;  // the value tensor is transposed

    const uint32_t n_seq_max = 1;
    const uint32_t n_stream  = 1;

    // required padding
    const uint32_t n_pad = 1;

    // SWA
    const uint32_t n_swa = 0;

    // env: LLAMA_ATTN_ROT_DISABLE
    bool attn_rot_k = false;
    bool attn_rot_v = false;

    // if all layers participating in the cache have constant head size, the value is stored here
    // otherwise the value is -1
    int32_t n_embd_head_k_all = 0;
    int32_t n_embd_head_v_all = 0;

    // pre-computed hadamard martrices
    std::unordered_map<int64_t, std::vector<float>> attn_rot_hadamard;

    // env: LLAMA_KV_CACHE_DEBUG
    int debug = 0;

    // KVzip compression state
    bool     kvzip_enabled        = false;
    int      kvzip_counter        = 0;
    float    kvzip_keep_ratio     = 0.33f;
    int      kvzip_trigger        = 512;
    int      kvzip_min_latest     = 0;
    float    kvzip_pos_bias       = 0.1f;

    // DepthKV — layer-dependent budgets
    bool     depthkv_enabled      = false;
    float    depthkv_min_keep     = 0.20f;
    float    depthkv_max_keep     = 0.80f;

    // RazorAttention state
    bool     razor_attn_enabled   = false;
    int      razor_attn_window    = 2048;
    bool     razor_attn_profiled  = false;
    int32_t  razor_attn_n_head    = 0;   // n_head_kv, set during profiling

    // FastKV state
    bool     fastkv_enabled       = false;
    float    fastkv_tsp_rate      = 0.20f;
    int      fastkv_tsp_layer     = -1;
    float    fastkv_kv_retention  = 0.10f;
    bool     fastkv_prefill_done  = false;
    std::vector<int> fastkv_selected_indices;

    // xKV state
    bool     xkv_enabled          = false;
    int32_t  xkv_rank             = 32;
    bool     xkv_profiled         = false;
    std::string xkv_profile_path;
    xkv_basis xkv_basis_data;

    struct head_profile {
        uint32_t layer;
        uint32_t head;
        bool     is_retrieval;
        float    avg_attn_distance;
    };
    std::vector<head_profile> razor_attn_profiles;

    // Persistence: file path for profile storage
    std::string razor_attn_profile_path;

    // this is the SWA type of the cache - not to be confused with the model SWA type
    const llama_swa_type swa_type = LLAMA_SWA_TYPE_NONE;

    // ggml contexts for the KV cache along with the allocated backend buffers:
    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> ctxs_bufs;

    // the current index from where we start searching for a free slot in the ring buffer of KV cells (see find_slot())
    // note: this is not part of the KV state and it's only used to speed-up the find_slot() method
    std::vector<uint32_t> v_heads;

    // TODO: temporary until we refactor to be able to share the same cells between 2 kv caches [TAG_KV_CACHE_SHARE_CELLS]
    llama_kv_cache * other;

    std::shared_ptr<llama_kv_cells_vec> v_cells_impl;

    llama_kv_cells_vec & v_cells;

    // maps from a sequence id to a stream id
    std::vector<uint32_t> seq_to_stream;

    // pending stream copies that will be applied during the next update
    stream_copy_info sc_info;

    std::vector<kv_layer> layers;

    // model layer id -> KV cache layer id
    std::unordered_map<int32_t, int32_t> map_layer_ids;

    size_t total_size() const;

    size_t size_k_bytes() const;
    size_t size_v_bytes() const;

    ggml_tensor * build_rope_shift(
            const llama_cparams & cparams,
                   ggml_context * ctx,
                    ggml_tensor * cur,
                    ggml_tensor * shift,
                    ggml_tensor * rot,
                    ggml_tensor * factors,
                          float   freq_base,
                          float   freq_scale,
                       uint32_t   il) const;

    ggml_cgraph * build_graph_shift(
               llm_graph_result * res,
                  llama_context * lctx) const;

    struct cell_ranges_t {
        uint32_t strm;

        std::vector<std::pair<uint32_t, uint32_t>> data; // ranges, from inclusive, to exclusive
    };

    void state_write_meta(llama_io_write_i & io, const cell_ranges_t & cr, llama_seq_id seq_id = -1) const;
    void state_write_data(llama_io_write_i & io, const cell_ranges_t & cr) const;

    bool state_read_meta(llama_io_read_i & io, uint32_t strm, uint32_t cell_count,       slot_info & sinfo, llama_seq_id dest_seq_id = -1);
    bool state_read_data(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, const slot_info & sinfo);
};

class llama_kv_cache_context : public llama_memory_context_i {
public:
    // some shorthands
    using slot_info_vec_t  = llama_kv_cache::slot_info_vec_t;
    using stream_copy_info = llama_kv_cache::stream_copy_info;

    // used for errors
    llama_kv_cache_context(llama_memory_status status);

    // used to create a full-cache context
    llama_kv_cache_context(
            llama_kv_cache * kv);

    // used to create an update context
    llama_kv_cache_context(
            llama_kv_cache * kv,
            llama_context * lctx,
            bool do_shift,
            stream_copy_info sc_info);

    // used to create a batch processing context from a batch
    llama_kv_cache_context(
            llama_kv_cache * kv,
            slot_info_vec_t sinfos,
            std::vector<llama_ubatch> ubatches);

    virtual ~llama_kv_cache_context();

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    //
    // llama_kv_cache_context specific API
    //

    uint32_t get_n_kv() const;

    ggml_type type_k() const;
    ggml_type type_v() const;

    // get views of the current state of the cache
    ggml_tensor * get_k(ggml_context * ctx, int32_t il) const;
    ggml_tensor * get_v(ggml_context * ctx, int32_t il) const;

    // store k_cur and v_cur in the cache based on the provided head location
    // note: the heads in k_cur and v_cur should be laid out contiguously in memory
    //   - k_cur  [n_embd_head_k, n_head_k, n_tokens]
    //   - k_idxs [n_tokens]
    //   - v_cur  [n_embd_head_v, n_head_v, n_tokens]
    //   - v_idxs [n_tokens] or [n_tokens*n_embd_v_gqa] depending if V cache is transposed
    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const;
    ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il) const;

    // create destination indices for each head of the current batch for where it would be written in the KV cache
    // the indices address the global KV cache (not per stream) - this is not relevant for the user of this API, but
    //   helps understand the implementation logic of cpy_k and cpy_v
    ggml_tensor * build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;
    ggml_tensor * build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;

    ggml_tensor * build_input_k_rot(ggml_context * ctx) const;
    ggml_tensor * build_input_v_rot(ggml_context * ctx) const;

    void set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const;
    void set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const;

    void set_input_k_shift   (ggml_tensor * dst) const;
    void set_input_kq_mask   (ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const;
    void set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const;

    void set_input_k_rot(ggml_tensor * dst) const;
    void set_input_v_rot(ggml_tensor * dst) const;

private:
    llama_memory_status status;

    llama_kv_cache * kv;
    llama_context * lctx;

    //
    // update context
    //

    bool do_shift = false;

    stream_copy_info sc_info;

    //
    // batch processing context
    //

    // the index of the cur ubatch to process
    size_t i_cur = 0;

    slot_info_vec_t sinfos;

    std::vector<llama_ubatch> ubatches;

    //
    // data needed for building the compute graph for the current ubatch:
    //

    // a heuristic, to avoid attending the full cache if it is not yet utilized
    // as the cache gets filled, the benefit from this heuristic disappears
    int32_t n_kv;
};
