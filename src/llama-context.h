#pragma once

#include "llama.h"
#include "llama-ext.h"
#include "llama-cparams.h"
#include "llama-graph.h"
#include "llama-adapter.h"
#include "llama-impl.h"
#include "llama-memory.h"
#include "llama-moe-route-stats.h"
#include "llama-moe-cache.h"

#include "ggml-cpp.h"
#include "ggml-opt.h"

#include <array>
#include <map>
#include <array>
#include <atomic>
#include <functional>
#include <deque>
#include <string>
#include <vector>

struct llama_model;
class llama_batch_allocr;
class llama_moe_cache;

class llama_io_read_i;
class llama_io_write_i;

// "memory" as in abstract memory for the context
struct llama_memory_i;
struct llama_memory_context_i;

// stores copy of the memory in device buffer. used for fast state save/load
struct llama_memory_buffer {
    int n_tensors = 0;
    size_t total_size = 0;

    ggml_backend_buffer_ptr buf;

    ggml_context_ptr ctx;

    std::vector<ggml_tensor *> org;
    std::vector<ggml_tensor *> cpy;
};

using llama_memory_buffers = std::map<ggml_backend_buffer_type_t, llama_memory_buffer>;

struct llama_context {
    // init scheduler and compute buffers, reserve worst-case graphs
    llama_context(
            const llama_model & model,
                  llama_context_params params,
                  std::shared_ptr<llama_memory_i> shared_memory = {});

    // Private compute graph, outputs and backend streams over the same KV pool.
    // Memory planning/mutation remains coordinator-owned; this is not a
    // thread-safe llama_decode API until the execution-wave contract is applied.
    std::unique_ptr<llama_context> create_execution_lane(uint32_t batch_capacity, uint32_t logical_capacity = 0);

    // Caller fences cache mutation and owns each context until all results return.
    // Completion runs on the coordinator after the lane has synchronized.
    // It holds shared-memory planning and may continue the private tail with
    // one token without joining other lanes. wait_idle quiesces metadata readers.
    using execution_completion = std::function<bool(llama_context *, llama_batch &)>;
    using execution_wait = std::function<void(const std::function<void()> &)>;
    std::vector<int> decode_execution_wave(const std::vector<std::pair<llama_context *, llama_batch>> & work,
            execution_completion completed = {}, execution_wait wait_idle = {});

    int decode_with_execution_lanes(const llama_batch & batch, const std::vector<llama_context *> & lanes,
        const std::vector<std::pair<llama_context *, llama_batch>> & additional = {},
            execution_completion completed = {}, execution_wait wait_idle = {});
    const std::string & get_execution_fallback() const { return execution_fallback; }

    // Cumulative worker wall times; relaxed snapshots may straddle a completion.
    // No GPU operations or shared-memory inspection are performed by readers.
    std::array<uint64_t, 8 + GGML_SCHED_TRANSPORT_PROFILE_FIELDS> get_execution_profile() const {
        std::array<uint64_t, 8 + GGML_SCHED_TRANSPORT_PROFILE_FIELDS> result{};
        for (size_t i = 0; i < result.size(); ++i) result[i] = execution_profile[i].load(std::memory_order_relaxed);
        return result;
    }

    std::vector<std::string> get_execution_backend_names() const {
        std::vector<std::string> names;
        for (auto * backend : backend_ptrs) names.emplace_back(ggml_backend_name(backend));
        return names;
    }

    std::pair<const char *, size_t> get_route_capture_memory() const {
        return buf_moe_routes ? std::make_pair(ggml_backend_buffer_name(buf_moe_routes.get()),
            ggml_backend_buffer_get_size(buf_moe_routes.get())) : std::make_pair("none", size_t(0));
    }

    size_t get_last_execution_wave_lanes() const { return last_execution_wave_lanes; }

    ~llama_context();

    // reserve a new backend scheduler (if needed)
    // for example, when:
    //   - changing loras
    //   - changing samplers
    //   - changing attention type
    //   - etc.
    void sched_reserve();

    void synchronize();

    const llama_model   & get_model()   const;
    const llama_cparams & get_cparams() const;

    ggml_backend_sched_t get_sched() const;
    void set_execution_input_validator(std::function<void(const llm_graph_result &)> validator) {
        execution_input_validator = std::move(validator);
    }

    uint32_t n_ctx()     const;
    uint32_t n_ctx_seq() const;
    uint32_t n_batch()   const;
    uint32_t n_ubatch()  const;
    uint32_t n_seq_max() const;

    uint32_t n_threads()       const;
    uint32_t n_threads_batch() const;

    llama_memory_t get_memory() const;

    // return true if the memory was updated
    bool memory_update(bool optimize);

    enum llama_pooling_type pooling_type() const;

    float * get_logits();
    float * get_logits_ith(int32_t i);

    float * get_embeddings();
    float * get_embeddings_ith(int32_t i);
    float * get_embeddings_seq(llama_seq_id seq_id);
    const std::map<int32_t, std::vector<float>> & get_hidden_states() const;

    float * get_embeddings_nextn();
    float * get_embeddings_nextn_ith(int32_t i);

    // Queue a device-resident hidden-state row range from the most recent
    // NextN output of src to the next native-MTP decode input. Returns false
    // when the source rows are not resident in the current graph result.
    bool set_nextn_input_from(
            const llama_context & src,
                       int32_t     src_row,
                       int32_t     dst_row,
                       int32_t     n_rows);

    void set_nextn_host_output(bool enabled);
    bool preserve_nextn_ith(int32_t i);
    bool copy_nextn_ith(int32_t i, float * dst, size_t n_floats);
    bool copy_nextn_carry(float * dst, size_t n_floats);

    float * get_embeddings_layer_inp(uint32_t lid);

    llama_token * get_sampled_tokens() const;
    llama_token   get_sampled_token_ith(int32_t idx);

    float * get_sampled_logits_ith(int32_t idx);
    size_t  get_sampled_logits_count(int32_t idx);

    float * get_sampled_probs_ith(int32_t idx);
    size_t  get_sampled_probs_count(int32_t idx);

    const llama_token * get_sampled_candidates_ith(int32_t idx);
    size_t get_sampled_candidates_count(int32_t idx);

    bool get_causal_attn() const;

    void attach_threadpool(
            ggml_threadpool_t threadpool,
            ggml_threadpool_t threadpool_batch);

    void detach_threadpool();

    void set_n_threads(int32_t n_threads, int32_t n_threads_batch);

    void set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data);

    void set_embeddings (bool value);
    void set_hidden_state_layers(const std::set<int32_t> & layers);
    void set_embeddings_nextn(bool value, bool masked);
    void set_embeddings_layer_inp(uint32_t lid, bool enable);
    void set_nextn_layer_offset(int32_t offset);
    void set_causal_attn(bool value);
    void set_warmup(bool value);

    void set_adapters_lora(llama_adapter_lora ** adapters, size_t n_adapters, float * scales);

    bool adapters_lora_are_same(llama_adapter_lora ** adapters, size_t n_adapters, float * scales);

    bool set_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end);

    // process a single ubatch with a specific graph type
    // if memory_context is provided, it will be applied first to the context's memory
    // ret contains the status of the graph computation
    // returns nullptr only if ret != GGML_STATUS_SUCCESS
    llm_graph_result * process_ubatch(
                const llama_ubatch & ubatch,
                    llm_graph_type   gtype,
            llama_memory_context_i * mctx,
                       ggml_status & ret);

    int encode(const llama_batch_ext & batch_inp);
    int decode(const llama_batch_ext & batch_inp);

    // compat version
    int encode(const llama_batch & batch_inp);
    int decode(const llama_batch & batch_inp);

    //
    // state save/load
    //

    size_t state_get_size();
    size_t state_get_data(      uint8_t * dst, size_t size);
    size_t state_set_data(const uint8_t * src, size_t size);

    size_t state_seq_get_size(llama_seq_id seq_id, llama_state_seq_flags flags);
    size_t state_seq_get_size_range(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_state_seq_flags flags);

    size_t state_seq_get_data(llama_seq_id seq_id,       uint8_t * dst, size_t size, llama_state_seq_flags flags);
    size_t state_seq_get_data_range(llama_seq_id seq_id, llama_pos p0, llama_pos p1, uint8_t * dst, size_t size, llama_state_seq_flags flags);
    size_t state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size, llama_state_seq_flags flags);

    bool state_load_file(
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    bool state_save_file(
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    size_t state_seq_load_file(
          llama_seq_id   seq_id,
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    size_t state_seq_save_file(
          llama_seq_id   seq_id,
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    //
    // perf
    //

    llama_perf_context_data perf_get_data() const;
    void perf_reset();

    llama_memory_breakdown memory_breakdown() const;

    //
    // training
    //

    void opt_init(struct llama_model * model, struct llama_opt_params lopt_params);

    // TODO: more flexible combinations of logical/physical batch size and context size
    void opt_epoch(
            ggml_opt_dataset_t      dataset,
            ggml_opt_result_t       result_train,
            ggml_opt_result_t       result_eval,
            int64_t                 idata_split,
            ggml_opt_epoch_callback callback_train,
            ggml_opt_epoch_callback callback_eval);

    void opt_epoch_iter(
            ggml_opt_dataset_t               dataset,
            ggml_opt_result_t                result,
            const std::vector<llama_token> & tokens,
            const std::vector<llama_token> & labels_sparse,
            llama_batch                    & batch,
            ggml_opt_epoch_callback          callback,
            bool                             train,
            int64_t                          idata_in_loop,
            int64_t                          ndata_in_loop,
            int64_t                          t_loop_start);

private:
    //
    // output
    //

    // Make sure enough space is available for outputs.
    // Returns max number of outputs for which space was reserved.
    uint32_t output_reserve(int32_t n_outputs);

    void output_reorder();

    // map the output row index `i` to batch index
    int64_t output_resolve_row(int32_t i) const;

    // async-copy enabled layer-input tensors (per cparams.output_layer_inp)
    // from backend into host-side embd_layer_inp buffers
    bool extract_layer_inputs(const llm_graph_result * res, size_t token_offset, size_t n_tokens);
    void extract_moe_routes(const llm_graph_result * res);
    void consume_moe_routes();
    void report_moe_routes();

    //
    // graph
    //

public:
    uint32_t graph_max_nodes(uint32_t n_tokens) const;

    // can reuse the llm_graph_result instance of the context (for example to update a memory module)
    llm_graph_result * get_gf_res_reserve() const;

    // returns the result of ggml_backend_sched_graph_compute_async execution
    ggml_status graph_compute(ggml_cgraph * gf, bool batched);

    // reserve a graph with a dummy ubatch of the specified size
    ggml_cgraph * graph_reserve(
        uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, const llama_memory_context_i * mctx, bool split_only = false, size_t * sizes = nullptr);

    bool set_sampler(llama_seq_id seq_id, llama_sampler * sampler);

private:
    llm_graph_result * get_gf_res_prev();

    llm_graph_params graph_params(
                        llm_graph_result * res,
                      const llama_ubatch & ubatch,
            const llama_memory_context_i * mctx,
                          llm_graph_type   gtype) const;

    llm_graph_cb graph_get_cb() const;

    // create a new backend scheduler for a graph of the specified size and register the callbacks below
    void sched_new(size_t max_nodes, bool parallel);

    // ggml_backend_sched copy callback, copies only the experts used by MUL_MAT_ID and updates the MoE cache
    static bool sched_copy_experts(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph, void * user_data);

    // ggml_backend_sched realloc callback, counts and traces the graph reallocations of this context
    static void sched_graph_realloc(ggml_backend_sched_t sched, ggml_cgraph * gf, bool backend_ids_changed, bool graph_size_changed, void * user_data);

    // disable auto fused ops (Flash Attention, Gated Delta Net) whose op lands on a device
    // that differs from the layer it belongs to (usually due to missing backend support)
    void resolve_fused_ops(const llama_memory_context_i * mctx, uint32_t n_seqs);

    void record_nextn_transport(ggml_backend_t backend, size_t bytes, int64_t submit_start_us);
    void apply_nextn_device_inputs(llm_graph_result * res);
    int decode_impl(const llama_batch_ext & batch_inp);

    // TODO: read/write lora adapters and cvec
    size_t state_write_data(llama_io_write_i & io);
    size_t state_read_data (llama_io_read_i  & io);

    size_t state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags);
    size_t state_seq_write_data_range(llama_io_write_i & io, llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_state_seq_flags flags);
    size_t state_seq_read_data (llama_io_read_i  & io, llama_seq_id seq_id, llama_state_seq_flags flags);

    //
    // members
    //

    const llama_model & model;

    llama_cparams cparams;

    llama_adapter_cvec_ptr  cvec;
    llama_adapter_loras_ptr loras;

    llama_cross cross; // TODO: tmp for handling cross-attention - need something better probably

    std::function<void(const llm_graph_result &)> execution_input_validator;
    std::function<void()> execution_dispatch_begin;
    std::function<void()> execution_dispatch_end;
    // calls, planning wait/held, graph submission, outputs, final sync,
    // input population, graph preparation. The last two overlap planning held.
    std::array<std::atomic<uint64_t>, 8 + GGML_SCHED_TRANSPORT_PROFILE_FIELDS> execution_profile{};
    bool execution_wave_active = false;
    size_t last_execution_wave_lanes = 0;
    uint64_t execution_lane_id = 0;
    std::string execution_fallback;
    const llama_context_params params_init;
    std::shared_ptr<llama_memory_i> memory;
    llama_moe_cache_ptr moe_cache;

    // decode output (2-dimensional array: [n_outputs][n_vocab])
    buffer_view<float> logits = {nullptr, 0};

    // embeddings output (2-dimensional array: [n_outputs][n_embd])
    // populated only when pooling_type == LLAMA_POOLING_TYPE_NONE
    buffer_view<float> embd = {nullptr, 0};

    // hidden state required by the nextn layers (2-dimensional array: [n_outputs][n_embd])
    // populated only when cparams.embeddings_nextn is enabled and the model graph
    // sets llm_graph_result::t_h_nextn
    buffer_view<float> embd_nextn = {nullptr, 0};

    struct nextn_transport_info {
        uint64_t copies    = 0;
        uint64_t bytes     = 0;
        int64_t  submit_us = 0;
        std::string backend;
    } nextn_transport;

    struct nextn_device_output_info {
        ggml_tensor * tensor     = nullptr;
        ggml_backend_t backend   = nullptr;
        int32_t row_offset       = 0;
        int32_t n_rows           = 0;
        uint64_t generation      = 0;
    } nextn_device_output;

    struct nextn_device_input_span {
        const llama_context * src = nullptr;
        int32_t src_row            = 0;
        int32_t dst_row            = 0;
        int32_t n_rows             = 0;
        uint64_t generation        = 0;
        bool use_carry             = false;
    };
    std::vector<nextn_device_input_span> nextn_device_inputs;

    bool validate_nextn_device_input(
            const nextn_device_input_span & span,
            ggml_tensor * dst,
            ggml_backend_t backend_dst) const;
    bool nextn_device_inputs_cover(llm_graph_result * res, int32_t n_rows) const;
    bool prepare_nextn_device_inputs_for_reset(const llama_ubatch & ubatch);
    bool recover_nextn_self_inputs_to_host(const llama_ubatch & ubatch);
    bool preserve_nextn_output_row(int32_t row);
    void synchronize_nextn_device_carry();

    struct nextn_device_input_transport_info {
        uint64_t copies    = 0;
        uint64_t aliases   = 0;
        uint64_t bytes     = 0;
        int64_t  submit_us = 0;
        std::string routes;
    } nextn_device_input_transport;

    struct nextn_device_carry_info {
        std::vector<uint8_t> meta;
        ggml_context_ptr ctx;
        ggml_backend_buffer_ptr buf;
        nextn_device_output_info output;
        bool valid = false;
        bool pending = false;
    } nextn_device_carry;

    bool nextn_host_output = true;

    // host buffers for output layer input embeddings, per layer
    // populated when cparams.output_layer_inp[il] is true
    std::vector<buffer_view<float>> embd_layer_inp;
    std::vector<int32_t> embd_batch_idxs; // extracted index -> original batch index

    struct moe_route_snapshot {
        int32_t layer = -1;
        size_t n_used = 0;
        size_t n_tokens = 0;
        buffer_view<int32_t> ids = {nullptr, 0};
        buffer_view<float> weights = {nullptr, 0};
    };
    static constexpr size_t MOE_ROUTE_PENDING_LIMIT = 16u * 1024u * 1024u;
    ggml_backend_buffer_ptr buf_moe_routes;
    std::deque<moe_route_snapshot> moe_route_pending;
    size_t moe_route_pending_bytes = 0;
    uint64_t moe_route_next_report = 4096;
    int64_t moe_route_next_snapshot_us = 0;
    llama_moe_route_stats moe_route_stats;

    struct sampling_info {
        // !samplers.empty() to check if any samplers are active
        std::map<llama_seq_id, llama_sampler *> samplers;

        buffer_view<float>       logits     = {nullptr, 0};
        buffer_view<llama_token> sampled    = {nullptr, 0};
        buffer_view<float>       probs      = {nullptr, 0};
        buffer_view<llama_token> candidates = {nullptr, 0};

        std::vector<uint32_t> logits_count;
        std::vector<uint32_t> probs_count;
        std::vector<uint32_t> candidates_count;

        // optimization
        std::vector<llama_token> token_ids_full_vocab;
    };

    sampling_info sampling;

    // sequence embeddings output (map of [n_embd] vectors)
    // populated only when pooling_type != LLAMA_POOLING_TYPE_NONE
    std::map<llama_seq_id, std::vector<float>> embd_seq;

    std::set<int32_t> hidden_state_layers;
    std::map<int32_t, std::vector<float>> hidden_states;

    // reuse the batch_allocr to avoid unnecessary memory allocations
    std::unique_ptr<llama_batch_allocr> balloc;

    uint32_t n_input_tensors = 0; // number of tensors marked as input during the last graph reserve
    uint32_t n_outputs = 0; // number of actually-used outputs in the current ubatch or last logical batch

    std::vector<int32_t> output_ids; // map batch token positions to ids of the logits and embd buffers

    struct swap_info {
        uint32_t i0;
        uint32_t i1;
    };

    std::vector<swap_info> output_swaps;

    ggml_backend_sched_ptr sched;

    bool sched_need_reserve = true;

    // whether the reserve graph has a logits tensor, set in sched_reserve()
    // true until then, because the first output_reserve() runs before any graph is reserved
    bool graph_has_logits = true;

    // state of sched_copy_experts, reset before each graph compute
    struct copy_experts_info {
        const ggml_tensor *  ids = nullptr;
        std::vector<int32_t> ids_data;
        std::vector<bool>    used;

        void reset() {
            ids = nullptr;
            ids_data.clear();
            used.clear();
        }
    };

    copy_experts_info copy_experts;

    // graph reallocations during inference where the backend assignments did not change, split by whether the
    // graph size is the same as the previously scheduled one (unexpected) or different
    int32_t n_reallocs_same = 0;
    int32_t n_reallocs_diff = 0;

    ggml_backend_t backend_cpu = nullptr;
    std::vector<ggml_backend_ptr> backends;

    // training
    ggml_opt_context_t opt_ctx = nullptr;

    ggml_threadpool_t threadpool       = nullptr;
    ggml_threadpool_t threadpool_batch = nullptr;

    ggml_abort_callback abort_callback      = nullptr;
    void *              abort_callback_data = nullptr;

    std::vector<std::pair<ggml_backend_t, ggml_backend_set_n_threads_t>> set_n_threads_fns;

    // pointers and buffer types used for the compute buffer of each backend
    std::vector<ggml_backend_t>             backend_ptrs;
    std::vector<ggml_backend_buffer_type_t> backend_buft;
    std::vector<size_t>                     backend_buf_exp_size; // expected buffer sizes

    // Separate arenas give batches with and without outputs distinct CUDA graph cache keys.
    std::array<llm_graph_result_ptr, 2> gf_res_prev;
    llm_graph_result_ptr gf_res_reserve;

    llm_graph_result * gf_res_prev_active = nullptr;

    // host buffer for the model output (logits and embeddings)
    ggml_backend_buffer_ptr buf_output;

    // keep copies of the per-sequence memory on the device
    std::map<llama_seq_id, llama_memory_buffers> mem_storage;

    bool has_evaluated_once = false;

    // env: LLAMA_GRAPH_REUSE_DISABLE
    bool graph_reuse_disable = false;

    // true after the prism.hadamard coverage check passes on a reserved graph
    bool hadamard_verified = false;

    // perf
    mutable int64_t t_start_us  = 0;
    mutable int64_t t_load_us   = 0;
    mutable int64_t t_p_eval_us = 0;
    mutable int64_t t_eval_us   = 0;

    mutable int64_t t_compute_start_us = 0;
    mutable int64_t n_queued_tokens    = 0;

    mutable int32_t n_p_eval = 0; // number of tokens in eval calls for the prompt (with batch size > 1)
    mutable int32_t n_eval   = 0; // number of eval calls

    mutable int32_t n_reused = 0; // number of times the previous graph was reused
};
