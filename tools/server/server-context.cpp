#include "server-context.h"
#include <atomic>
#include <thread>
#include "base64.hpp"
#include "server-chat.h"
#include "server-common.h"
#include "server-decision.h"
#include "server-http.h"
#include "server-task.h"
#include "server-queue.h"
#include "server-schema.h"
#include "server-stream.h"
#include "server-page-ram.h"
#include "server-responses-control.h"
#include "server-responses-leases.h"

#include "build-info.h"
#include "common.h"
#include "fit.h"
#include "llama.h"
#include "../../src/llama-context.h"
#include "../../src/llama-ext.h"
#include "../../src/llama-model.h"
#include "vendor/hash/hash.h"
#include "log.h"
#include "sampling.h"
#include "speculative.h"
#include "moe-route-trace.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#pragma push_macro("LOG_DBG")
#undef LOG_DBG
#pragma push_macro("LOG_INF")
#undef LOG_INF
#pragma push_macro("LOG_WRN")
#undef LOG_WRN
#pragma push_macro("LOG_ERR")
#undef LOG_ERR
#include "mtmd-helper-common.h"
#pragma pop_macro("LOG_ERR")
#pragma pop_macro("LOG_WRN")
#pragma pop_macro("LOG_INF")
#pragma pop_macro("LOG_DBG")
#include "base64.hpp"

#include <future>
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstddef>
#include <cinttypes>
#include <csignal>
#include <cstring>
#include <deque>
#include <exception>
#include <fstream>
#include <memory>
#include <filesystem>
#include <list>
#include <random>
#include <unordered_map>
#include <utility>
#include <fstream>
#include <mutex>
#include <thread>

#if defined(__linux__)
#include <dirent.h>
#include <dlfcn.h>
#include <execinfo.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>
#endif

// fix problem with std::min and std::max
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

constexpr int HTTP_POLLING_SECONDS = 1;

// Optional profiler annotations are resolved at runtime so ordinary server
// builds do not acquire a ROCm profiler dependency. ROCtx ranges are
// thread-local, so construct the scope inside the queue callback that performs
// the actual decode work.
struct server_roctx_scope {
    using push_fn = int (*)(const char *);
    using pop_fn  = int (*)();

    explicit server_roctx_scope(const char * name) {
#if defined(__linux__)
        if (!enabled()) {
            return;
        }

        static void * handle = dlopen("libroctx64.so", RTLD_LAZY | RTLD_LOCAL);
        static auto push = handle ? reinterpret_cast<push_fn>(dlsym(handle, "roctxRangePushA")) : nullptr;
        static auto pop  = handle ? reinterpret_cast<pop_fn>(dlsym(handle, "roctxRangePop")) : nullptr;
        pop_ = pop;
        if (push && pop_) {
            push(name);
            active_ = true;
        }
#else
        (void) name;
#endif
    }

    ~server_roctx_scope() {
        if (active_) {
            pop_();
        }
    }

    server_roctx_scope(const server_roctx_scope &) = delete;
    server_roctx_scope & operator=(const server_roctx_scope &) = delete;

private:
    static bool enabled() {
        static const bool value = []() {
            const char * raw = std::getenv("LLAMA_SERVER_ROCTX_RANGES");
            return raw && raw[0] != '\0' && std::strcmp(raw, "0") != 0;
        }();
        return value;
    }

    pop_fn pop_ = nullptr;
    bool active_ = false;
};

static common_speculative_output_limits server_output_limits(const common_params & params) {
    if (!params.mmproj.path.empty()) {
        const auto mcaps = mtmd_get_cap_from_file(params.mmproj.path.c_str());
        if (mcaps.gen_audio) {
            // some TTS models decode with embeddings output for the whole batch
            return { params.n_batch, params.n_batch };
        }
    }

    if (params.embedding ||
            (params.pooling_type != LLAMA_POOLING_TYPE_UNSPECIFIED && params.pooling_type != LLAMA_POOLING_TYPE_NONE)) {
        return { params.n_batch, 1 };
    }

    auto result = common_speculative_get_output_limits(
            params.n_batch, params.n_parallel, common_speculative_n_max(&params.speculative));

    result.total   = std::max<int32_t>(1, result.total);
    result.per_seq = std::max<int32_t>(1, result.per_seq);
    return result;
}

// a checkpoint restore dropped tokens the target had accepted - re-accept them rather than verify again
static std::vector<llama_token> server_accept_replay(
        common_sampler * smpl,
        llama_context * ctx,
        const std::vector<int32_t> & idxs,
        const llama_tokens & draft) {
    GGML_ASSERT(idxs.size() == draft.size() + 1);

    std::vector<llama_token> result;
    result.reserve(idxs.size());

    for (size_t i = 0; i < draft.size(); ++i) {
        // the token is discarded - the call is what advances the sampler over this position
        common_sampler_sample(smpl, ctx, idxs[i]);
        common_sampler_accept(smpl, draft[i], true);
        result.push_back(draft[i]);
    }

    const llama_token id = common_sampler_sample(smpl, ctx, idxs[draft.size()]);
    common_sampler_accept(smpl, id, true);
    result.push_back(id);

    return result;
}

// synthetic draft verification for benchmarking - accept draft tokens at random instead of by match with the target
// on replay the draft was already accepted before a context checkpoint restore, so repeat the same decisions
static std::vector<llama_token> server_sample_and_accept_synth(
        common_sampler * smpl,
        llama_context * ctx,
        const std::vector<int32_t> & idxs,
        const llama_tokens & draft,
        const std::vector<double> & synth_probs,
        std::mt19937 & rng,
        bool is_replay) {
    GGML_ASSERT(idxs.size() == draft.size() + 1);
    GGML_ASSERT(synth_probs.size() >= draft.size());

    std::vector<llama_token> result;
    result.reserve(idxs.size());

    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (size_t i = 0; i < draft.size(); ++i) {
        const llama_token id = common_sampler_sample(smpl, ctx, idxs[i]);
        const bool accept = is_replay || dist(rng) < synth_probs[i];
        // do not accept a drafted EOG token - it would end the generation early
        // on replay the last token is from the target and can be EOG, so skip this check
        if (accept && (is_replay || !llama_vocab_is_eog(vocab, draft[i]))) {
            // synthetic draft tokens do not advance grammar or reasoning state
            // the last replay token is from the target and must advance both
            const bool is_replay_target = is_replay && i + 1 == draft.size();
            common_sampler_accept(smpl, draft[i], is_replay_target);
            result.push_back(draft[i]);
            continue;
        }

        common_sampler_accept(smpl, id, true);
        result.push_back(id);
        return result;
    }

    const llama_token id = common_sampler_sample(smpl, ctx, idxs[draft.size()]);
    common_sampler_accept(smpl, id, true);
    result.push_back(id);

    return result;
}

static int64_t server_load_watchdog_timeout_ms() {
    const char * raw = std::getenv("LLAMA_SERVER_LOAD_PROGRESS_STALL_TIMEOUT_S");
    if (raw == nullptr || raw[0] == 0) {
        return 0;
    }
    char * end = nullptr;
    const double seconds = std::strtod(raw, &end);
    if (end == raw || seconds <= 0.0) {
        return 0;
    }
    return (int64_t) (seconds * 1000.0);
}

#if defined(__linux__)
static void server_load_watchdog_signal_handler(int signo) {
    void * frames[128];
    const int n_frames = backtrace(frames, 128);
    dprintf(STDERR_FILENO, "\n[load-watchdog] signal %d thread %ld stack:\n", signo, (long) syscall(SYS_gettid));
    backtrace_symbols_fd(frames, n_frames, STDERR_FILENO);
}

static void server_load_watchdog_install_signal_handler() {
    static std::once_flag once;
    std::call_once(once, []() {
        struct sigaction action;
        memset(&action, 0, sizeof(action));
        action.sa_handler = server_load_watchdog_signal_handler;
        sigemptyset(&action.sa_mask);
        action.sa_flags = SA_RESTART;
        sigaction(SIGUSR2, &action, nullptr);
    });
}

static void server_load_watchdog_dump_process_state(const char * model_path, float progress, int64_t idle_ms) {
    SRV_ERR("llama.cpp load watchdog: model load stalled for %.3f s at progress %.6f for %s\n",
            idle_ms / 1000.0, progress, model_path ? model_path : "");

    std::ifstream status("/proc/self/status");
    if (status.good()) {
        SRV_ERR("%s", "llama.cpp load watchdog: /proc/self/status follows\n");
        std::string line;
        while (std::getline(status, line)) {
            SRV_ERR("  %s\n", line.c_str());
        }
    }

    SRV_ERR("%s", "llama.cpp load watchdog: skipping in-process stack signal before forced exit\n");
}
#endif

struct server_load_progress_watchdog {
    llama_progress_callback previous_callback = nullptr;
    void * previous_user_data = nullptr;
    const char * model_path = nullptr;
    int64_t timeout_ms = 0;
    std::atomic<int64_t> last_progress_ms{0};
    std::atomic<float> last_progress{0.0f};
    std::atomic<bool> active{false};
    std::atomic<bool> done{false};
    std::thread thread;

    ~server_load_progress_watchdog() {
        stop();
    }

    static bool callback(float progress, void * user_data) {
        auto * self = static_cast<server_load_progress_watchdog *>(user_data);
        self->active.store(true, std::memory_order_relaxed);
        self->last_progress.store(progress, std::memory_order_relaxed);
        self->last_progress_ms.store(ggml_time_ms(), std::memory_order_relaxed);
        if (self->previous_callback != nullptr) {
            return self->previous_callback(progress, self->previous_user_data);
        }
        return true;
    }

    void start() {
        if (timeout_ms <= 0) {
            return;
        }
        last_progress_ms.store(ggml_time_ms(), std::memory_order_relaxed);
#if defined(__linux__)
        server_load_watchdog_install_signal_handler();
#endif
        thread = std::thread([this]() {
            while (!done.load(std::memory_order_relaxed)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(250));
                if (done.load(std::memory_order_relaxed)) {
                    break;
                }
                if (!active.load(std::memory_order_relaxed)) {
                    continue;
                }
                const int64_t now = ggml_time_ms();
                const int64_t idle_ms = now - last_progress_ms.load(std::memory_order_relaxed);
                if (idle_ms < timeout_ms) {
                    continue;
                }
#if defined(__linux__)
                dprintf(STDERR_FILENO,
                        "llama.cpp load watchdog: forced exit after %.3f s stall at progress %.6f for %s\n",
                        idle_ms / 1000.0, last_progress.load(std::memory_order_relaxed), model_path ? model_path : "");
                _exit(134);
#else
                SRV_ERR("llama.cpp load watchdog: model load stalled for %.3f s at progress %.6f for %s\n",
                        idle_ms / 1000.0, last_progress.load(std::memory_order_relaxed), model_path ? model_path : "");
                std::abort();
#endif
            }
        });
    }

    void stop() {
        done.store(true, std::memory_order_relaxed);
        if (thread.joinable()) {
            thread.join();
        }
    }
};

static std::vector<int32_t> parse_hidden_state_layers(const json & body, int32_t n_layer) {
    std::vector<int32_t> layers;

    if (body.contains("layers") && body.at("layers").is_array()) {
        for (const auto & item : body.at("layers")) {
            if (!item.is_number_integer()) {
                throw std::runtime_error("layers must contain only integers");
            }
            layers.push_back(item.get<int32_t>());
        }
    } else if (body.contains("layer_start") && body.contains("layer_end")) {
        const int32_t layer_start = body.at("layer_start").get<int32_t>();
        const int32_t layer_end   = body.at("layer_end").get<int32_t>();
        if (layer_start > layer_end) {
            throw std::runtime_error("layer_start must be less than or equal to layer_end");
        }
        for (int32_t il = layer_start; il <= layer_end; ++il) {
            layers.push_back(il);
        }
    } else {
        throw std::runtime_error("layers or layer_start/layer_end is required");
    }

    if (layers.empty()) {
        throw std::runtime_error("at least one hidden-state layer is required");
    }
    for (int32_t il : layers) {
        if (il < 0 || il >= n_layer) {
            throw std::runtime_error("hidden-state layer index is outside the model layer range");
        }
    }

    std::sort(layers.begin(), layers.end());
    layers.erase(std::unique(layers.begin(), layers.end()), layers.end());
    return layers;
}

static std::vector<int32_t> parse_hidden_state_token_indices(const json & body, int32_t n_tokens) {
    std::vector<int32_t> indices;
    if (!body.contains("token_indices") || body.at("token_indices").is_null()) {
        indices.reserve(n_tokens);
        for (int32_t i = 0; i < n_tokens; ++i) {
            indices.push_back(i);
        }
        return indices;
    }

    if (!body.at("token_indices").is_array()) {
        throw std::runtime_error("token_indices must be an integer array");
    }
    for (const auto & item : body.at("token_indices")) {
        if (!item.is_number_integer()) {
            throw std::runtime_error("token_indices must contain only integers");
        }
        int32_t idx = item.get<int32_t>();
        if (idx < 0) {
            idx += n_tokens;
        }
        if (idx < 0 || idx >= n_tokens) {
            throw std::runtime_error("token index is outside the prompt token range");
        }
        indices.push_back(idx);
    }
    if (indices.empty()) {
        throw std::runtime_error("token_indices must not be empty");
    }
    return indices;
}

static std::mutex g_sensitive_trace_mutex;
static std::atomic<uint64_t> g_sensitive_trace_next_id{1};

static const char * sensitive_trace_path() {
    const char * path = std::getenv("LLAMA_CPP_SERVER_SENSITIVE_TRACE_PATH");
    if (path == nullptr || path[0] == '\0') {
        return nullptr;
    }
    return path;
}

static void sensitive_trace_append(const std::string & text) {
    const char * path = sensitive_trace_path();
    if (path == nullptr) {
        return;
    }

    std::lock_guard<std::mutex> lock(g_sensitive_trace_mutex);
    std::ofstream file(path, std::ios::out | std::ios::app | std::ios::binary);
    if (!file) {
        return;
    }
    file << text;
    file.flush();

    std::error_code ec;
    std::filesystem::permissions(
        path,
        std::filesystem::perms::owner_read | std::filesystem::perms::owner_write,
        std::filesystem::perm_options::replace,
        ec);
}

static void sensitive_trace_append_json_text(std::string & out, const json & value) {
    if (value.is_string()) {
        out += value.get<std::string>();
        return;
    }
    if (!value.is_array()) {
        return;
    }
    for (const auto & item : value) {
        if (item.is_string()) {
            out += item.get<std::string>();
        } else if (item.is_object()) {
            if (item.contains("text") && item.at("text").is_string()) {
                out += item.at("text").get<std::string>();
            }
        }
    }
}

static std::string sensitive_trace_output_text(const json & res_json) {
    std::string out;

    auto append_key = [&](const json & obj, const char * key) {
        if (obj.contains(key)) {
            sensitive_trace_append_json_text(out, obj.at(key));
        }
    };

    append_key(res_json, "content");
    append_key(res_json, "text");
    append_key(res_json, "delta");

    if (res_json.contains("choices") && res_json.at("choices").is_array()) {
        for (const auto & choice : res_json.at("choices")) {
            if (!choice.is_object()) {
                continue;
            }
            append_key(choice, "text");
            if (choice.contains("delta") && choice.at("delta").is_object()) {
                const auto & delta = choice.at("delta");
                append_key(delta, "reasoning_content");
                append_key(delta, "content");
                append_key(delta, "text");
                if (delta.contains("tool_calls") && delta.at("tool_calls").is_array()) {
                    for (const auto & tool_call : delta.at("tool_calls")) {
                        if (!tool_call.is_object()) {
                            continue;
                        }
                        if (tool_call.contains("function") && tool_call.at("function").is_object()) {
                            append_key(tool_call.at("function"), "arguments");
                        }
                    }
                }
            }
            if (choice.contains("message") && choice.at("message").is_object()) {
                append_key(choice.at("message"), "reasoning_content");
                append_key(choice.at("message"), "content");
            }
        }
    }

    return out;
}

// state diagram: https://github.com/ggml-org/llama.cpp/pull/9283
enum slot_state {
    SLOT_STATE_IDLE,
    SLOT_STATE_WAIT_OTHER, // after assigning a task, but waiting for parent slot to process prompt
    SLOT_STATE_STARTED,    // after assigning a task and about to process prompt
    SLOT_STATE_PROCESSING_PROMPT,
    SLOT_STATE_DONE_PROMPT,
    SLOT_STATE_GENERATING,
};

struct server_slot; // forward declaration

struct server_batch {
    common_batch view; // the rendered sub-batch [off, off + n_tokens), see render()
    std::vector<llama_token> lane_tokens;
    std::vector<llama_pos> lane_positions;
    std::vector<llama_seq_id> lane_sequences;
    std::vector<llama_seq_id *> lane_sequence_ptrs;
    std::vector<int32_t> lane_sequence_counts;
    std::vector<int8_t> lane_outputs;

    llama_batch get_legacy_view(int32_t off, int32_t count) {
        lane_tokens.resize(count);
        lane_positions.resize(count);
        lane_sequences.resize(count);
        lane_sequence_ptrs.resize(count);
        lane_sequence_counts.assign(count, 1);
        lane_outputs.resize(count);
        for (int32_t i = 0; i < count; ++i) {
            const auto & token = tokens.at(off + i);
            lane_tokens[i] = token.token;
            lane_positions[i] = token.pos[0];
            lane_sequences[i] = token.id_slot;
            lane_sequence_ptrs[i] = &lane_sequences[i];
            lane_outputs[i] = token.output;
        }
        return { count, lane_tokens.data(), nullptr, lane_positions.data(),
                 lane_sequence_counts.data(), lane_sequence_ptrs.data(), lane_outputs.data() };
    }

    struct token {
        int32_t id_slot;
        llama_token token;
        std::array<llama_pos, GGML_MROPE_SECTIONS> pos; // only pos[0] is used for text tokens
        bool output;
        bool is_prompt; // for stats tracking
        int32_t decision_order = 0;
        int32_t i_embd = -1; // row in embd, -1 if this is a token
    };
    std::vector<token> tokens;
    int32_t n_tokens_alloc = 0;
    int32_t n_embd = 0;

    // track if given slot can be batched with slots already in the batch
    server_slot * slot_batched = nullptr;

    // embedding entries, they can be mixed with tokens if the context supports it
    std::vector<float> embd; // n_embd per row

    float  alora_scale       = -1.0f;
    size_t alora_disabled_id = 0;

    void init(llama_context * ctx, int32_t n_tokens_alloc, int32_t n_embd) {
        this->n_tokens_alloc = n_tokens_alloc;
        this->n_embd = n_embd;
        view = common_batch(ctx);
        tokens.reserve(n_tokens_alloc);
    }

    bool add(int32_t id_slot, llama_token token, llama_pos pos, bool output, bool is_prompt) {
        if ((int32_t)tokens.size() >= n_tokens_alloc) {
            return false;
        }
        tokens.push_back({ id_slot, token, { pos, 0, 0, 0 }, output, is_prompt });
        return true;
    }

    // embd_in has n_embd values, pos has GGML_MROPE_SECTIONS values
    bool add_embd(int32_t id_slot, const float * embd_in, const llama_pos * pos, bool output, bool is_prompt) {
        if ((int32_t)tokens.size() >= n_tokens_alloc) {
            return false;
        }
        tokens.push_back({ id_slot, LLAMA_TOKEN_NULL, { pos[0], pos[1], pos[2], pos[3] }, output, is_prompt });
        tokens.back().i_embd = (int32_t) (embd.size() / n_embd);
        embd.insert(embd.end(), embd_in, embd_in + n_embd);
        return true;
    }

    bool has_embd() const {
        return !embd.empty();
    }

    void clear() {
        tokens.clear();
        embd.clear();
        view.clear();
        slot_batched      = nullptr;
        alora_scale       = -1.0f;
        alora_disabled_id = 0;
    }

    int32_t size() const {
        return (int32_t)tokens.size();
    }

    // remove the entries after the first n
    void truncate(int32_t n) {
        GGML_ASSERT(n >= 0 && n <= size());
        for (int32_t i = n; i < size(); i++) {
            if (tokens[i].i_embd >= 0) {
                embd.resize((size_t) tokens[i].i_embd * n_embd);
                break;
            }
        }
        tokens.resize(n);
    }

    void set_output(int32_t idx, bool output) {
        GGML_ASSERT(idx >= 0 && idx < (int32_t)tokens.size());
        tokens[idx].output = output;
    }

    void set_decision_order(int32_t idx, int32_t order) {
        GGML_ASSERT(idx >= 0 && idx < (int32_t)tokens.size());
        tokens[idx].decision_order = order;
    }

    // render the sub-batch [off, off + n_tokens) into view, index i in view is index off + i here
    void render(int32_t off, int32_t n_tokens) {
        GGML_ASSERT(off >= 0 && off < size());
        GGML_ASSERT(n_tokens > 0 && off + n_tokens <= size());

        view.clear();
        for (int32_t i = off; i < off + n_tokens; i++) {
            const auto & t = tokens[i];
            if (t.i_embd >= 0) {
                view.add_embd({ embd.data() + (size_t) t.i_embd * n_embd, 1, (size_t) n_embd }, t.pos.data(), t.id_slot, t.output);
            } else {
                view.add(t.token, t.pos[0], t.id_slot, t.output);
            }
            view.tokens.back().decision_order = t.decision_order;
        }
    }
};

class server_prompt_page_cache {
public:
    struct counters {
        uint64_t capture_attempts = 0;
        uint64_t capture_us       = 0;
        uint64_t capture_last_us  = 0;
        uint64_t restore_attempts = 0;
        uint64_t restore_us       = 0;
        uint64_t restore_last_us  = 0;
        uint64_t captures         = 0;
        uint64_t hits             = 0;
        uint64_t misses           = 0;
        uint64_t restored_tokens  = 0;
        uint64_t evictions        = 0;
        uint64_t failures         = 0;
        uint64_t verify_attempts  = 0;
        uint64_t verify_failures  = 0;
        uint64_t branch_releases  = 0;
        uint64_t live_prefix_adoptions = 0;
        uint64_t resident_prefix_restores = 0, resident_prefix_positions = 0;
        uint64_t capture_bridges = 0;
        uint64_t superseded_pruned = 0;
        uint64_t manifest_node_builds = 0, manifest_node_reuses = 0;
        uint64_t persist_attempts = 0, persist_us = 0, persist_build_us = 0, persist_write_us = 0;
        uint64_t capture_sync_us = 0, capture_alloc_us = 0, capture_copy_us = 0, capture_seal_us = 0;
        uint64_t ssd_maintenance_runs = 0;
        uint64_t ram_demotions = 0, ssd_restores = 0, ssd_restored_bytes = 0, ssd_restore_us = 0, ssd_restore_failures = 0, capture_waiting_backup = 0;
    };

    server_prompt_page_cache(size_t limit_size, size_t page_tokens) :
        limit_size(server_page_ram_store::effective_limit(limit_size)), page_tokens(page_tokens), ram_store(this->limit_size) {
        GGML_ASSERT(limit_size > 0);
        GGML_ASSERT(page_tokens > 0);
        recover();
        maintain_ssd({});
    }

    void maintenance(const std::unordered_set<int> & active_tasks) {
        const auto now = server_cache_now_ms();
        if (now < next_ssd_maintenance_ms) return;
        next_ssd_maintenance_ms = now + 1000;
        std::unordered_set<std::string> protected_keys;
        for (const auto & node : pages) if (active_tasks.count(node.usage.last_request)) protected_keys.insert(node.key);
        maintain_ssd(protected_keys);
    }

    json tier_stats() const {
        const auto * disk = ram_store.disk_store();
        json result = disk ? json::parse(disk->stats().dump()) : json{{"enabled", false}};
        result["ram_bytes"] = size_cur; result["ram_limit_bytes"] = limit_size;
        result["demotions"] = stats_cur.ram_demotions; result["ssd_restores"] = stats_cur.ssd_restores;
        result["ssd_restored_bytes"] = stats_cur.ssd_restored_bytes; result["ssd_restore_us"] = stats_cur.ssd_restore_us;
        result["ssd_restore_failures"] = stats_cur.ssd_restore_failures;
        result["capture_waiting_backup"] = stats_cur.capture_waiting_backup;
        result["maintenance_runs"] = stats_cur.ssd_maintenance_runs;
        return result;
    }

    json persistence_stats() const {
        return {{"enabled", ram_store.active()}, {"tiers", tier_stats()}, {"recovered_pages", ram_store.recovered},
            {"rejected_pages", ram_store.rejected}, {"commits", ram_store.commits},
            {"failures", ram_store.failures}, {"orphan_files", ram_store.orphan_files},
            {"stale_namespaces", ram_store.stale_namespaces}, {"stale_bytes", ram_store.stale_bytes}};
    }

    json replacement_stats() const {
        const uint64_t now = server_cache_now_ms();
        size_t known = 0, cold = 0, retrievals = 0;
        uint64_t oldest_ms = 0;
        for (const auto & node : pages) {
            if (node.usage.last_access_ms) {
                known++;
                oldest_ms = std::max(oldest_ms, now > node.usage.last_access_ms ? now - node.usage.last_access_ms : 0);
            } else { cold++; }
            retrievals += node.usage.retrievals;
        }
        return {{"name", "age_frequency_density_v1"}, {"history_known_pages", known},
            {"legacy_cold_pages", cold}, {"bounded_retrievals", retrievals}, {"oldest_access_age_s", oldest_ms / 1000},
            {"incidental_interactive_half_life_s", 1800}, {"retained_interactive_half_life_s", 3600},
            {"retained_background_half_life_s", 28800}, {"ephemeral_half_life_s", 900},
            {"retrieval_count_cap", 15}, {"release_grace_s", 30},
            {"superseded_partial_idle_s", 1800}, {"superseded_pruned", stats_cur.superseded_pruned},
            {"recent_evictions", json::parse(eviction_history.dump())}};
    }

    size_t page_size_tokens() const {
        return page_tokens;
    }

    size_t size() const {
        return size_cur;
    }

    size_t limit() const {
        return limit_size;
    }

    size_t target_size() const {
        return size_tgt_cur;
    }

    size_t draft_size() const {
        return size_dft_cur;
    }

    size_t metadata_size() const {
        return size_cur - size_tgt_cur - size_dft_cur;
    }

    size_t count() const {
        return pages.size();
    }

    const counters & stats() const {
        return stats_cur;
    }

    bool prepare_lease(const std::string & scope, const std::string & revision,
                       const std::string & expected, uint64_t generation, uint64_t ttl) {
        return mutate_lease([&] { return response_leases.prepare(scope, revision, expected,
            server_cache_now_ms(), ttl, generation); });
    }

    bool observe_lease(const std::string & scope, const std::string & revision,
                       const std::string & branch, size_t minimum_tokens, bool complete) {
        // Validate the actual reachable endpoint, including sparse tail pages.
        // A branch label by itself is not evidence of a complete checkpoint.
        auto endpoint = pages.end();
        for (auto node = pages.begin(); node != pages.end(); ++node) {
            if (node->branches.count(branch) && node->p1_tokens >= minimum_tokens &&
                    (endpoint == pages.end() || node->p1_tokens > endpoint->p1_tokens)) endpoint = node;
        }
        if (endpoint == pages.end()) return false;
        server_responses_leases::page_sizes ancestry;
        auto node = endpoint;
        while (true) {
            const size_t bytes = node->data_tgt.size() + node->data_dft.size() + node->data_tgt_reference.size();
            if (!bytes || !ancestry.emplace(node->key, bytes).second) return false;
            if (node->parent_key.empty()) {
                if (node->p0_tokens != 0 || node->p0_pos != 0) return false;
                break;
            }
            const auto parent = by_key.find(node->parent_key);
            if (parent == by_key.end() || parent->second->p1_tokens != node->p0_tokens ||
                    parent->second->p1_pos != node->p0_pos) return false;
            node = parent->second;
        }
        return mutate_lease([&] { return response_leases.observe(scope, revision, ancestry,
            complete, server_cache_now_ms()); });
    }

    bool commit_lease(const std::string & scope, const std::string & revision,
                      const std::string & expected, uint64_t ttl) {
        const auto found = response_leases.snapshot().find(scope);
        if (found == response_leases.snapshot().end()) return false;
        for (const auto & page : found->second.pending.pages) {
            const auto node = by_key.find(page.first);
            if (node == by_key.end() || page.second != node->second->data_tgt.size() +
                    node->second->data_dft.size() + node->second->data_tgt_reference.size()) return false;
        }
        return mutate_lease([&] { return response_leases.commit(scope, revision, expected,
            server_cache_now_ms(), ttl); });
    }

    bool release_lease(const std::string & scope, const std::string & expected) {
        return mutate_lease([&] { return response_leases.release(scope, expected, server_cache_now_ms()); });
    }

    bool lease_prepared(const std::string & scope, const std::string & revision, uint64_t generation) const {
        const auto found = response_leases.snapshot().find(scope);
        return found != response_leases.snapshot().end() && found->second.generation == generation &&
            found->second.pending.revision == revision && found->second.pending.expires > server_cache_now_ms();
    }

    json context_control(const json & body) {
        const auto scope = body.at("scope").get<std::string>();
        const auto action = body.at("action").get<std::string>();
        const auto revision = json_value(body, "revision", std::string{});
        const auto expected = json_value(body, "expected", std::string{});
        const auto ttl = json_value(body, "lease_ms", uint64_t(7200000));
        bool success = action == "state";
        if (action == "prepare") success = prepare_lease(scope, revision, expected, body.at("generation").get<uint64_t>(), ttl);
        else if (action == "observe") success = observe_lease(scope, revision, body.at("branch").get<std::string>(),
            body.at("minimum_tokens").get<size_t>(), true);
        else if (action == "commit") {
            const auto found = response_leases.snapshot().find(scope);
            const auto old_head = found == response_leases.snapshot().end() ? std::string{} : found->second.head;
            success = (found != response_leases.snapshot().end() && old_head == revision &&
                found->second.expires > server_cache_now_ms()) || commit_lease(scope, revision, expected, ttl);
            if (success && old_head != revision) {
                // Drop only the old response's named claim. Shared ancestor and
                // sibling ownership survives; old sparse tails become evictable.
                const auto old_branch = json_value(body, "old_branch", std::string{});
                if (!old_branch.empty()) release_branch(old_branch);
            }
        } else if (action == "release") success = release_lease(scope, expected);
        else if (action == "abandon") success = mutate_lease([&] { return response_leases.abandon(scope, revision); });
        json result = {{"success", success}, {"scope", scope}, {"head", ""}, {"generation", 0},
            {"expires", 0}, {"pending_revision", ""}, {"pending_complete", false},
            {"protected_state_bytes", response_leases.protected_bytes(server_cache_now_ms())},
            {"protected_state_byte_limit", response_leases.limit()}, {"lease_ms", 7200000}, {"persistent", ram_store.active()},
            {"namespace", std::getenv("LLAMA_SERVER_PAGE_CACHE_NAMESPACE") ? std::getenv("LLAMA_SERVER_PAGE_CACHE_NAMESPACE") : ""}};
        const auto found = response_leases.snapshot().find(scope);
        if (found != response_leases.snapshot().end()) {
            const auto & state = found->second;
            result["head"] = state.head;
            result["generation"] = state.generation;
            result["expires"] = state.expires;
            result["pending_revision"] = state.pending.revision;
            result["pending_complete"] = state.pending.endpoint_complete;
            result["pending_expires"] = state.pending.expires;
            result["pending_pages"] = state.pending.pages.size();
        }
        return result;
    }

    size_t release_branch(const std::string & branch) {
        size_t released_nodes = 0;
        std::unordered_set<std::string> affected;
        for (auto & node : pages) {
            const auto owner = node.branches.find(branch);
            if (owner == node.branches.end()) {
                continue;
            }
            size_cur -= owner->first.size() + sizeof(owner->second);
            node.descriptor_dirty = true;
            node.branches.erase(owner);
            affected.insert(node.key);
            released_nodes++;
        }
        // A fingerprint rollover may release the old branch immediately before
        // submitting its replacement. Keep its now-unowned pages reusable for
        // 30 seconds; adoption restores ownership, otherwise pressure GC can
        // reclaim them once the grace expires. Existing shared claims survive.
        const uint64_t deadline = server_cache_now_ms() + 30000;
        for (auto & node : pages) {
            if (affected.count(node.key) && !node.unowned_retained &&
                    !node.unowned_ephemeral && node.branches.empty()) {
                node.release_after_ms = deadline;
            }
        }
        stats_cur.branch_releases++;
        persist();
        return released_nodes;
    }

    size_t match_tokens(const server_tokens & tokens) const {
        return find_longest_chain(tokens, nullptr).tokens;
    }

    bool capture(
            const server_tokens & tokens,
                  llama_context * ctx_tgt,
                  llama_context * ctx_dft,
                   llama_seq_id   seq_id,
            const server_cache_control & control,
                           bool   exact_endpoint = false) {
        if (control.lifecycle == server_cache_lifecycle::one_shot) {
            return false;
        }
        const auto retained = find_longest_chain(tokens, ctx_dft);
        maintain_ssd({retained.keys.begin(), retained.keys.end()});
        prune_superseded({retained.keys.begin(), retained.keys.end()}, server_cache_now_ms());
        const size_t p1_tokens = tokens.size();
        const auto boundaries = tokens.page_boundaries(page_tokens);
        const bool is_coarse = !boundaries.empty() && boundaries.back().token_count == p1_tokens;
        if (!exact_endpoint && !is_coarse) {
            return false;
        }
        if (p1_tokens == 0) {
            return false;
        }
        const int64_t t_start = ggml_time_us();

        const server_tokens::page_boundary end = is_coarse
            ? boundaries.back()
            : server_tokens::page_boundary{p1_tokens, tokens.pos_next()};
        server_tokens::page_boundary begin;
        for (const auto & boundary : boundaries) {
            if (boundary.token_count >= end.token_count) {
                break;
            }
            begin = boundary;
        }
        const auto nominal_begin = begin;
        begin = capture_begin(tokens, ctx_dft, nominal_begin);
        const bool bridging = begin.token_count != nominal_begin.token_count;
        page value;
        value.key       = tokens.cache_prefix_hash(end.token_count);
        value.parent_key = begin.token_count > 0 ? tokens.cache_prefix_hash(begin.token_count) : std::string();
        value.p0_tokens = begin.token_count;
        value.p1_tokens = end.token_count;
        value.p0_pos    = begin.position;
        value.p1_pos    = end.position;
        value.coarse    = is_coarse;

        // Exact prefix identity alone determines whether two requests share a
        // node. Branch names are retention hints and never part of the key.
        if (const auto found = by_key.find(value.key); found != by_key.end()) {
            mark_control(*found->second, control);
            pages.splice(pages.end(), pages, found->second);
            note_superseded(tokens, value.key, server_cache_now_ms());
            persist();
            record_capture(t_start);
            return true;
        }
        value.record_control(control);

        const size_t tgt_size = llama_state_seq_get_size_range_ext(ctx_tgt, seq_id, begin.position, end.position, LLAMA_STATE_SEQ_FLAGS_NONE);
        const size_t dft_size = ctx_dft ? llama_state_seq_get_size_range_ext(ctx_dft, seq_id, begin.position, end.position, LLAMA_STATE_SEQ_FLAGS_NONE) : 0;
        const size_t ref_size = tokens.has_media() && verify_restore_enabled()
            ? llama_state_seq_get_size_ext(ctx_tgt, seq_id, LLAMA_STATE_SEQ_FLAGS_NONE) : 0;
        const size_t value_size = value.size() + tgt_size + dft_size + ref_size;
        if (value_size > limit_size) {
            record_capture(t_start);
            SRV_WRN("range-state page %.3f MiB exceeds cache limit %.3f MiB, skipping\n",
                    value_size / (1024.0 * 1024.0), limit_size / (1024.0 * 1024.0));
            return false;
        }

        const uint64_t rank_now = server_cache_now_ms();
        // Keep the entire parent chain while making room for this child.
        std::unordered_set<std::string> ancestors;
        for (std::string key = value.parent_key; !key.empty();) {
            const auto parent = by_key.find(key);
            if (parent == by_key.end() || !ancestors.insert(key).second) {
                record_capture(t_start);
                stats_cur.failures++;
                return false;
            }
            key = parent->second->parent_key;
        }

        bool reclaimed = false;
        while (!pages.empty() && size_cur + value_size > limit_size) {
            auto * disk = ram_store.disk_store();
            if (disk && disk->active() && demote_ram()) { reclaimed = true; continue; }
            if (disk && disk->active() && !disk->degraded() && disk->free_bytes() >= disk->reserve() &&
                    value_size <= disk->free_bytes() - disk->reserve()) {
                // Copies run independently; retaining a useful checkpoint must
                // never turn the generation thread into an SSD write waiter.
                stats_cur.capture_waiting_backup++;
                persist(); record_capture(t_start); return false;
            }
            auto leaf = eviction_candidate(ancestors, rank_now);
            if (disk && disk->active()) {
                // A degraded writer must not delete cold graph leaves which
                // reclaim no RAM. Preserve them and choose a resident leaf.
                leaf = pages.end();
                for (auto it = pages.begin(); it != pages.end(); ++it) {
                    if (!(it->data_tgt.resident_size() + it->data_dft.resident_size() + it->data_tgt_reference.resident_size()) ||
                            it->children || ancestors.count(it->key) || disk->pending(it->key) ||
                            it->release_after_ms > rank_now || response_leases.protects(it->key, rank_now)) continue;
                    if (leaf == pages.end() || it->eviction_rank(rank_now).less_than(leaf->eviction_rank(rank_now))) leaf = it;
                }
            }
            if (leaf == pages.end()) {
                persist();
                record_capture(t_start);
                SRV_WRN("%s", "range-state graph has no evictable leaf below cache limit, skipping node\n");
                return false;
            }
            evict(leaf);
            reclaimed = true;
        }

        // Publish reclamation before allocation so tmpfs has space and a crash cannot revive removed pages.
        if (reclaimed) persist();
        if (!capture_state(ctx_tgt, seq_id, begin.position, end.position, value.data_tgt) ||
                (ctx_dft && !capture_state(ctx_dft, seq_id, begin.position, end.position, value.data_dft))) {
            persist();
            record_capture(t_start);
            stats_cur.failures++;
            return false;
        }
        if (tokens.has_media() && verify_restore_enabled() &&
                !capture_full_state(ctx_tgt, seq_id, value.data_tgt_reference)) {
            persist();
            record_capture(t_start);
            stats_cur.failures++;
            return false;
        }

        pages.push_back(std::move(value));
        const auto added = std::prev(pages.end());
        by_key.emplace(added->key, added);
        endpoint_counts[added->p1_tokens]++;
        if (!added->parent_key.empty()) {
            const auto parent = by_key.find(added->parent_key);
            GGML_ASSERT(parent != by_key.end());
            parent->second->children++;
        }
        size_cur += value_size;
        size_tgt_cur += added->data_tgt.size() + added->data_tgt_reference.size();
        size_dft_cur += added->data_dft.size();
        record_capture(t_start);
        stats_cur.captures++;
        ancestors.insert(added->key);
        maintain_ssd(ancestors);
        note_superseded(tokens, added->key, server_cache_now_ms());
        if (bridging) {
            stats_cur.capture_bridges++;
            SRV_TRC("range-state capture bridged missing boundary %zu from %zu to %zu tokens\n",
                    nominal_begin.token_count, begin.token_count, end.token_count);
        }
        persist();
        return true;
    }

    size_t restore(
            const server_tokens & tokens,
                         size_t   current_lcp,
                         bool     live_requires_rewind,
                  llama_context * ctx_tgt,
                  llama_context * ctx_dft,
                   llama_seq_id   seq_id,
            const server_cache_control & control,
                           bool & failed,
            std::function<bool(size_t, llama_pos)> seed_prefix = {},
            std::function<bool(llama_pos)> reserve_restore = {}) {
        failed = false;
        // A complete restored prompt has no logits and its hybrid recurrent
        // endpoint cannot be rewound. Save that endpoint for continuations,
        // but replay at least one last token for identical-prompt requests.
        const chain_match match = find_longest_chain(tokens, ctx_dft, tokens.empty() ? 0 : tokens.size() - 1);
        const auto & chain = match.keys;
        const size_t restored_tokens = match.tokens;
        // A hybrid live prefix can be longer than a saved coarse node while
        // still being unusable: truncation of its recurrent endpoint fails
        // later and would force a full prefill. Restore the exact graph node
        // instead when the new prompt diverges and live rewind is unavailable.
        if (restored_tokens == 0) {
            if (tokens.size() >= page_tokens) stats_cur.misses++;
            return 0;
        }
        if (restored_tokens <= current_lcp && !live_requires_rewind) {
            // A real live-slot hit must refresh graph history even though no
            // checkpoint import is necessary. One-shot retains its no-touch rule.
            if (control.lifecycle != server_cache_lifecycle::one_shot) {
                touch_chain(chain, control);
                stats_cur.live_prefix_adoptions++;
                persist();
            }
            return 0;
        }
        if (control.lifecycle != server_cache_lifecycle::one_shot) {
            maintain_ssd({match.keys.begin(), match.keys.end()});
            prune_superseded({match.keys.begin(), match.keys.end()}, server_cache_now_ms());
        }
        const int64_t t_start = ggml_time_us();

        if (!clear_sequence(ctx_tgt, seq_id) || !clear_sequence(ctx_dft, seq_id)) {
            record_restore(t_start);
            stats_cur.failures++;
            failed = true;
            return 0;
        }

        // Each range carries its own recurrent endpoint. Borrow immutable
        // resident rows through a range start, then import that range and all
        // successors. Always import at least the last page, even on an exact hit.
        size_t first_range = 0;
        llama_pos seeded_positions = 0;
        if (!ctx_dft && seed_prefix && !verify_restore_enabled()) {
            for (size_t i = chain.size(); i-- > 1;) {
                const auto & value = *by_key.at(chain[i]);
                if (seed_prefix(value.p0_tokens, value.p0_pos)) {
                    first_range = i;
                    seeded_positions = value.p0_pos;
                    break;
                }
            }
        }
        for (size_t i = first_range; i < chain.size(); ++i) {
            const auto & key = chain[i];
            const auto found = by_key.find(key);
            GGML_ASSERT(found != by_key.end());
            const auto & value = *found->second;
            if (reserve_restore && !reserve_restore(value.p1_pos)) {
                clear_sequence(ctx_tgt, seq_id);
                clear_sequence(ctx_dft, seq_id);
                record_restore(t_start);
                return 0; // Capacity pressure does not invalidate checkpoint contents.
            }
            const uint64_t cold_bytes = (value.data_tgt.on_disk() ? value.data_tgt.size() : 0) +
                (value.data_dft.on_disk() ? value.data_dft.size() : 0);
            const auto cold_start = ggml_time_us();
            if (!restore_state(ctx_tgt, seq_id, value.data_tgt) ||
                    (ctx_dft && !restore_state(ctx_dft, seq_id, value.data_dft))) {
                clear_sequence(ctx_tgt, seq_id);
                clear_sequence(ctx_dft, seq_id);
                if (cold_bytes) stats_cur.ssd_restore_failures++;
                evict_graph(key, false);
                persist();
                record_restore(t_start);
                stats_cur.failures++;
                failed = true;
                return 0;
            }
            if (cold_bytes) {
                stats_cur.ssd_restored_bytes += cold_bytes;
                stats_cur.ssd_restore_us += uint64_t(ggml_time_us() - cold_start);
            }
        }
        if (seeded_positions) {
            ++stats_cur.resident_prefix_restores;
            stats_cur.resident_prefix_positions += seeded_positions;
        }
        if (std::any_of(chain.begin() + first_range, chain.end(), [&](const std::string & key) {
                return by_key.at(key)->data_tgt.on_disk() || by_key.at(key)->data_dft.on_disk(); })) stats_cur.ssd_restores++;

        const auto verify_found = by_key.find(chain.back());
        const bool has_reference = verify_found != by_key.end() &&
            !verify_found->second->data_tgt_reference.empty();
        if (verify_restore_enabled() && has_reference) {
            stats_cur.verify_attempts++;
            const auto found = verify_found;
            std::vector<uint8_t> restored_data;
            const server_page_bytes * expected_data = nullptr;
            if (found != by_key.end()) {
                expected_data = !found->second->data_tgt_reference.empty()
                    ? &found->second->data_tgt_reference
                    : nullptr;
            }
            if (found == by_key.end() ||
                    expected_data == nullptr ||
                    !capture_full_state(ctx_tgt, seq_id, restored_data) ||
                    !(*expected_data == restored_data)) {
                if (expected_data != nullptr && !restored_data.empty()) {
                    const size_t shared = std::min(expected_data->size(), restored_data.size());
                    const auto mismatch = std::mismatch(
                            expected_data->begin(), expected_data->begin() + shared, restored_data.begin());
                    const size_t offset = mismatch.first == expected_data->begin() + shared
                        ? shared
                        : size_t(mismatch.first - expected_data->begin());
                    SRV_ERR("page-cache restore verification diverged at byte %zu "
                            "(reference %zu bytes, restored %zu bytes, pages %zu)\n",
                            offset, expected_data->size(), restored_data.size(), chain.size());
                }
                clear_sequence(ctx_tgt, seq_id);
                clear_sequence(ctx_dft, seq_id);
                record_restore(t_start);
                stats_cur.verify_failures++;
                stats_cur.failures++;
                failed = true;
                return 0;
            }
        }

        touch_chain(chain, control);

        record_restore(t_start);
        if (control.lifecycle != server_cache_lifecycle::one_shot) { persist(); }
        stats_cur.hits++;
        stats_cur.restored_tokens += restored_tokens;
        return restored_tokens;
    }

private:
    struct page {
        std::string key;
        std::string parent_key;
        size_t p0_tokens = 0;
        size_t p1_tokens = 0;
        llama_pos p0_pos = 0;
        llama_pos p1_pos = 0;
        bool coarse = false;
        size_t children = 0;
        bool unowned_retained = false;
        bool unowned_interactive = false;
        bool unowned_ephemeral = false;
        bool unowned_ephemeral_interactive = false;
        server_cache_usage usage;
        uint64_t release_after_ms = 0;
        uint64_t superseded_at_ms = 0;
        std::string superseded_by;
        uint64_t ssd_retry_after_ms = 0;
        std::map<std::string, std::pair<server_cache_priority, server_cache_lifecycle>> branches;
        server_page_bytes data_tgt;
        // Diagnostic-only uninterrupted state for attributing range composition.
        server_page_bytes data_tgt_reference;
        server_page_bytes data_dft;
        // CPU-only descriptor cache; state buffers and persisted format are unchanged.
        server_page_ram_store::json descriptor;
        bool descriptor_dirty = true;

        static size_t hint_index(server_cache_priority priority, server_cache_lifecycle lifecycle, bool named) {
            const bool interactive = priority == server_cache_priority::interactive;
            if (lifecycle != server_cache_lifecycle::retained) return interactive ? 4 : 5;
            return interactive ? (named ? 1 : 0) : (named ? 2 : 3);
        }

        void record_control(const server_cache_control & control, bool reused = false) {
            descriptor_dirty = true;
            release_after_ms = 0;
            superseded_at_ms = 0; // Returning to an old edit point renews its usefulness.
            usage.touch(hint_index(control.priority, control.lifecycle, !control.branch.empty()),
                        server_cache_now_ms(), control.request_id, reused);
            if (!control.branch.empty()) {
                branches[control.branch] = {control.priority, control.lifecycle};
            } else if (control.lifecycle == server_cache_lifecycle::retained) {
                unowned_retained = true;
                unowned_interactive |= control.priority == server_cache_priority::interactive;
            } else if (control.lifecycle == server_cache_lifecycle::ephemeral) {
                unowned_ephemeral = true;
                unowned_ephemeral_interactive |= control.priority == server_cache_priority::interactive;
            }
        }

        size_t policy_size() const {
            size_t result = 0;
            for (const auto & branch : branches) {
                result += branch.first.size() + sizeof(branch.second);
            }
            return result;
        }

        server_cache_eviction_rank eviction_rank(uint64_t now) const {
            double weight = unowned_retained ? usage.weight(3, now) : 0;
            if (unowned_retained && unowned_interactive) weight = std::max(weight, usage.weight(0, now));
            if (unowned_ephemeral) weight = std::max(weight, usage.weight(5, now));
            if (unowned_ephemeral && unowned_ephemeral_interactive) weight = std::max(weight, usage.weight(4, now));
            // An old interactive hint is not refreshed by a background access.
            // Shared pages take the strongest currently useful ownership claim.
            for (const auto & branch : branches) {
                weight = std::max(weight, usage.weight(hint_index(branch.second.first, branch.second.second, true), now));
            }
            return {weight, p1_tokens - p0_tokens, stored_size(), usage.last_access_ms};
        }

        bool backed() const {
            return data_tgt.backed() && (data_dft.empty() || data_dft.backed()) &&
                (data_tgt_reference.empty() || data_tgt_reference.backed());
        }
        size_t stored_size() const {
            return size() - data_tgt.resident_size() - data_dft.resident_size() - data_tgt_reference.resident_size() +
                data_tgt.size() + data_dft.size() + data_tgt_reference.size();
        }
        size_t size() const {
            return sizeof(usage) + sizeof(release_after_ms) + sizeof(superseded_at_ms) + superseded_by.size() + key.size() + parent_key.size() + data_tgt.resident_size() + data_tgt_reference.resident_size() +
                data_dft.resident_size() + policy_size();
        }
    };

    using page_list = std::list<page>;

    page_list::iterator eviction_candidate(const std::unordered_set<std::string> & ancestors, uint64_t now) {
        auto leaf = pages.end();
        for (auto it = pages.begin(); it != pages.end(); ++it) {
            if ((ram_store.disk_store() && ram_store.disk_store()->pending(it->key)) ||
                    it->children || ancestors.count(it->key) || it->release_after_ms > now ||
                    response_leases.protects(it->key, now)) continue;
            if (leaf == pages.end() || it->eviction_rank(now).less_than(leaf->eviction_rank(now))) leaf = it;
        }
        return leaf;
    }


    // A larger sibling is not necessarily an extension. Verify its exact
    // shorter prefix while tokens are available, after state publication.
    void note_superseded(const server_tokens & tokens, const std::string & replacement, uint64_t now) {
        for (auto & node : pages) {
            if (node.coarse || node.children || node.p1_tokens >= tokens.size() ||
                    tokens.size() - node.p1_tokens <= 1 || !tokens.complete_boundary(node.p1_tokens)) continue;
            if (node.superseded_at_ms && by_key.count(node.superseded_by)) continue;
            if (tokens.cache_prefix_hash(node.p1_tokens) != node.key) continue;
            const size_t extra = replacement.size() > node.superseded_by.size()
                ? replacement.size() - node.superseded_by.size() : 0;
            if (extra > limit_size - size_cur) continue;
            size_cur -= node.superseded_by.size();
            node.descriptor_dirty = true;
            node.superseded_by = replacement;
            size_cur += node.superseded_by.size();
            if (!node.superseded_at_ms) node.superseded_at_ms = now;
        }
    }

    size_t prune_superseded(const std::unordered_set<std::string> & protected_keys, uint64_t now) {
        size_t removed = 0;
        for (auto it = pages.begin(); it != pages.end();) {
            auto node = it++;
            const auto stamp = std::max(node->superseded_at_ms, node->usage.last_access_ms);
            if ((ram_store.disk_store() && ram_store.disk_store()->pending(node->key)) ||
                    !node->superseded_at_ms || node->coarse || node->children ||
                    now < stamp || now - stamp < 1800000 || protected_keys.count(node->key) ||
                    node->release_after_ms > now || response_leases.protects(node->key, now)) continue;
            const auto replacement = by_key.find(node->superseded_by);
            if (replacement == by_key.end() || replacement->second->p1_tokens <= node->p1_tokens + 1) continue;
            // Explicitly retained periodic warmers must keep their pre-logits
            // endpoints even if another request extends the base prompt.
            bool warmer = false;
            for (const auto & branch : node->branches) {
                warmer |= branch.second.first == server_cache_priority::background &&
                    branch.second.second == server_cache_lifecycle::retained;
            }
            if (warmer) continue;
            evict(node, true, "superseded_partial");
            stats_cur.superseded_pruned++;
            removed++;
        }
        if (removed) persist();
        return removed;
    }

    bool demote_ram() {
#if defined(__linux__)
        auto * disk = ram_store.disk_store();
        if (!disk || !disk->active()) return false;
        auto selected = pages.end();
        const auto now = server_cache_now_ms();
        // Prefer unleased pages, but disk-backed lease pages remain valid when
        // RAM capacity requires their demotion; only logical deletion is fenced.
        for (bool include_leased : {false, true}) {
            for (auto it = pages.begin(); it != pages.end(); ++it) {
                const size_t bytes = it->data_tgt.resident_size() + it->data_dft.resident_size() + it->data_tgt_reference.resident_size();
                if (!bytes || !it->backed() || (!include_leased && response_leases.protects(it->key, now))) continue;
                if (selected == pages.end() || it->eviction_rank(now).less_than(selected->eviction_rank(now))) selected = it;
            }
            if (selected != pages.end()) break;
        }
        if (selected == pages.end()) return false;
        selected->descriptor_dirty = true;
        const auto before = selected->size();
        const auto target = selected->data_tgt.resident_size() + selected->data_tgt_reference.resident_size();
        const auto draft = selected->data_dft.resident_size();
        try {
            for (const auto * part : {&selected->data_tgt, &selected->data_dft, &selected->data_tgt_reference}) {
                if (part->empty()) continue;
                struct stat st{};
                const auto file = disk->path(part->metadata().at("file").get<std::string>());
                if (lstat(file.c_str(), &st) || !S_ISREG(st.st_mode) || uint64_t(st.st_size) != part->size())
                    throw std::runtime_error("SSD backup disappeared");
            }
            for (auto * part : {&selected->data_tgt, &selected->data_dft, &selected->data_tgt_reference}) {
                if (!part->empty() && !part->on_disk()) part->demote(disk->path(part->metadata().at("file").get<std::string>()));
            }
        } catch (...) {
            for (auto * part : {&selected->data_tgt, &selected->data_dft, &selected->data_tgt_reference}) part->forget_backup();
            return false;
        }
        size_cur -= before - selected->size(); size_tgt_cur -= target; size_dft_cur -= draft;
        stats_cur.ram_demotions++; return true;

#else
        return false;
#endif
    }

    void maintain_ssd(const std::unordered_set<std::string> & protected_keys) {
        auto * disk = ram_store.disk_store();
        if (!disk || !disk->active()) return;
        stats_cur.ssd_maintenance_runs++;
        bool changed = false;
        for (const auto & complete : disk->poll()) {
            const auto found = by_key.find(complete.key);
            if (found == by_key.end()) continue;
            if (!complete.success) { found->second->ssd_retry_after_ms = server_cache_now_ms() + 60000; continue; }
            found->second->descriptor_dirty = true;
            for (auto * part : {&found->second->data_tgt, &found->second->data_dft, &found->second->data_tgt_reference}) part->mark_backed();
            changed = true;
        }
        // Do not spend SSD bandwidth archiving tails already known obsolete.
        changed |= prune_superseded(protected_keys, server_cache_now_ms()) > 0;
        std::vector<std::pair<size_t, std::string>> ordered;
        for (const auto & node : pages) ordered.emplace_back(node.p1_tokens, node.key);
        std::sort(ordered.begin(), ordered.end()); // Parent backups precede their children.
        for (const auto & entry : ordered) {
            const auto found = by_key.find(entry.second);
            if (found == by_key.end()) continue;
            auto & node = *found->second;
            if (node.backed() || disk->pending(node.key) || node.ssd_retry_after_ms > server_cache_now_ms()) continue;
            ram_json parts = ram_json::array(); uint64_t bytes = 0;
            for (const auto * part : {&node.data_tgt, &node.data_dft, &node.data_tgt_reference}) {
                if (part->empty() || part->backed()) continue;
                auto descriptor = part->metadata(); parts.push_back(descriptor);
                if (!std::filesystem::exists(disk->path(descriptor.at("file").get<std::string>()))) bytes += part->size();
            }
            if (!disk->queue_space(bytes) || !disk->free_bytes()) break;
            if (!disk->can_fit(bytes)) disk->reclaim_dormant(bytes);
            while (!disk->can_fit(bytes)) {
                auto victims_protected = protected_keys; victims_protected.insert(node.key);
                auto victim = pages.end();
                // Only deletion with an actual backing object frees SSD budget.
                // Pick directly from disk-backed graph-safe leaves.
                for (auto it = pages.begin(); it != pages.end(); ++it) {
                    if (!it->backed() || it->children || victims_protected.count(it->key) || disk->pending(it->key) ||
                            it->release_after_ms > server_cache_now_ms() || response_leases.protects(it->key, server_cache_now_ms())) continue;
                    if (victim == pages.end() || it->eviction_rank(server_cache_now_ms()).less_than(victim->eviction_rank(server_cache_now_ms()))) victim = it;
                }
                if (victim == pages.end()) break;
                evict(victim, true, "ssd_pressure"); persist(); changed = true;
            }
            if (!disk->can_fit(bytes)) break;
            if (!disk->submit(node.key, ram_store.ram_directory(), parts)) break;
        }
        while (size_cur > limit_size - limit_size / 10 && demote_ram()) changed = true;
        if (changed) persist();
    }

    struct chain_match {
        std::vector<std::string> keys;
        size_t tokens = 0;
    };

    void mark_control(page & node, const server_cache_control & control) {
        size_t extra = 0;
        if (!control.branch.empty() && node.branches.find(control.branch) == node.branches.end()) {
            extra = control.branch.size() + sizeof(std::pair<server_cache_priority, server_cache_lifecycle>);
        }
        if (extra > limit_size - size_cur) {
            SRV_WRN("%s", "range-state branch metadata exceeds cache limit, ignoring retention hint\n");
            return;
        }
        node.record_control(control, true);
        size_cur += extra;
    }

    void touch_chain(const std::vector<std::string> & chain, const server_cache_control & control) {
        if (control.lifecycle == server_cache_lifecycle::one_shot) return;
        for (const auto & key : chain) {
            const auto found = by_key.find(key);
            if (found == by_key.end()) continue;
            mark_control(*found->second, control);
            pages.splice(pages.end(), pages, found->second);
        }
    }

    chain_match find_longest_chain(
            const server_tokens & tokens, const llama_context * ctx_dft, size_t max_tokens = SIZE_MAX) const {
        for (auto endpoint = endpoint_counts.rbegin(); endpoint != endpoint_counts.rend(); ++endpoint) {
            if (endpoint->first > tokens.size() || endpoint->first > max_tokens ||
                    !tokens.complete_boundary(endpoint->first)) {
                continue;
            }

            const std::string leaf_key = tokens.cache_prefix_hash(endpoint->first);
            auto current = by_key.find(leaf_key);
            if (current == by_key.end() ||
                    current->second->p1_tokens != endpoint->first ||
                    current->second->p1_pos != tokens.pos_next(endpoint->first)) {
                continue;
            }

            chain_match result;
            std::unordered_set<std::string> seen;
            while (current != by_key.end() && seen.insert(current->first).second) {
                const page & node = *current->second;
                if ((ctx_dft != nullptr) != !node.data_dft.empty()) {
                    result.keys.clear();
                    break;
                }
                if (tokens.cache_prefix_hash(node.p1_tokens) != node.key) {
                    result.keys.clear();
                    break;
                }
                result.keys.push_back(node.key);
                if (node.p0_tokens == 0 && node.p0_pos == 0 && node.parent_key.empty()) {
                    std::reverse(result.keys.begin(), result.keys.end());
                    result.tokens = endpoint->first;
                    return result;
                }
                if (node.parent_key.empty()) {
                    result.keys.clear();
                    break;
                }
                const auto parent = by_key.find(node.parent_key);
                if (parent == by_key.end() ||
                        parent->second->p1_tokens != node.p0_tokens ||
                        parent->second->p1_pos != node.p0_pos) {
                    result.keys.clear();
                    break;
                }
                current = parent;
            }
        }
        return {};
    }

    server_tokens::page_boundary capture_begin(
            const server_tokens & tokens, const llama_context * ctx_dft,
            const server_tokens::page_boundary & nominal) const {
        if (!nominal.token_count) return nominal;
        const auto parent = by_key.find(tokens.cache_prefix_hash(nominal.token_count));
        if (parent != by_key.end() && parent->second->p1_tokens == nominal.token_count &&
                parent->second->p1_pos == nominal.position &&
                (ctx_dft != nullptr) == !parent->second->data_dft.empty()) {
            return nominal;
        }
        // Live-prefix reuse can skip coarse captures. Its current tail is still
        // capturable: attach a wider range to a rooted matching older endpoint,
        // or capture from zero if no compatible ancestor survives the budget.
        const auto match = find_longest_chain(tokens, ctx_dft, nominal.token_count);
        return {match.tokens, tokens.pos_next(match.tokens)};
    }

    static bool verify_restore_enabled() {
        static const bool enabled = []() {
            const char * value = getenv("LLAMA_SERVER_PAGE_CACHE_VERIFY_RESTORE");
            return value && atoi(value) > 0;
        }();
        return enabled;
    }

    bool capture_state(
            llama_context * ctx,
             llama_seq_id   seq_id,
                   size_t   p0,
                   size_t   p1,
            server_page_bytes & data) {
        const size_t size = llama_state_seq_get_size_range_ext(ctx, seq_id, p0, p1, LLAMA_STATE_SEQ_FLAGS_NONE);
        if (size == 0) {
            return false;
        }
        auto phase_start = ggml_time_us();
        try {
            ram_store.allocate(data, size);
        } catch (const std::exception &) { return false; }
        stats_cur.capture_alloc_us += ggml_time_us() - phase_start;
        phase_start = ggml_time_us();
        if (ctx) llama_synchronize(ctx);
        stats_cur.capture_sync_us += ggml_time_us() - phase_start;
        phase_start = ggml_time_us();
        const bool ok = llama_state_seq_get_data_range_ext(
                ctx, data.data(), data.size(), seq_id, p0, p1, LLAMA_STATE_SEQ_FLAGS_NONE) == size;
        stats_cur.capture_copy_us += ggml_time_us() - phase_start;
        phase_start = ggml_time_us();
        if (ok && ram_store.active()) { data.seal(); }
        stats_cur.capture_seal_us += ggml_time_us() - phase_start;
        return ok;
    }

    bool capture_full_state(llama_context * ctx, llama_seq_id seq_id, server_page_bytes & data) {
        const size_t size = llama_state_seq_get_size_ext(ctx, seq_id, LLAMA_STATE_SEQ_FLAGS_NONE);
        if (!size) return false;
        try { ram_store.allocate(data, size); } catch (const std::exception &) { return false; }
        const bool ok = llama_state_seq_get_data_ext(ctx, data.data(), size, seq_id, LLAMA_STATE_SEQ_FLAGS_NONE) == size;
        if (ok && ram_store.active()) { data.seal(); }
        return ok;
    }

    static bool capture_full_state(
            llama_context * ctx,
             llama_seq_id   seq_id,
            std::vector<uint8_t> & data) {
        const size_t size = llama_state_seq_get_size_ext(ctx, seq_id, LLAMA_STATE_SEQ_FLAGS_NONE);
        if (size == 0) {
            return false;
        }
        try {
            data.resize(size);
        } catch (const std::bad_alloc &) {
            return false;
        }
        return llama_state_seq_get_data_ext(
                ctx, data.data(), data.size(), seq_id, LLAMA_STATE_SEQ_FLAGS_NONE) == size;
    }

    static bool restore_state(llama_context * ctx, llama_seq_id seq_id, const server_page_bytes & data) {
        try {
            const bool ok = llama_state_seq_set_data_ext(ctx, data.data(), data.size(), seq_id, LLAMA_STATE_SEQ_FLAGS_NONE) == data.size();
            data.release_disk_mapping(); return ok;
        } catch (...) { data.release_disk_mapping(); return false; }
    }

    static bool clear_sequence(llama_context * ctx, llama_seq_id seq_id) {
        return ctx == nullptr || llama_memory_seq_rm(llama_get_memory(ctx), seq_id, -1, -1);
    }

    void record_capture(int64_t t_start) {
        stats_cur.capture_last_us = uint64_t(ggml_time_us() - t_start);
        stats_cur.capture_us += stats_cur.capture_last_us;
        stats_cur.capture_attempts++;
    }

    void record_restore(int64_t t_start) {
        stats_cur.restore_last_us = uint64_t(ggml_time_us() - t_start);
        stats_cur.restore_us += stats_cur.restore_last_us;
        stats_cur.restore_attempts++;
    }

    void evict(page_list::iterator it, bool count_eviction = true, const char * reason = "pressure") {
        if (count_eviction) {
            const auto now = server_cache_now_ms();
            ram_json branches = ram_json::array();
            for (const auto & branch : it->branches) branches.push_back(branch.first);
            eviction_history.push_back({{"at_ms", now}, {"reason", reason}, {"key", it->key},
                {"p0_tokens", it->p0_tokens}, {"p1_tokens", it->p1_tokens}, {"coarse", it->coarse},
                {"bytes", it->stored_size()}, {"ram_bytes", it->size()}, {"last_access_ms", it->usage.last_access_ms},
                {"retrievals", it->usage.retrievals}, {"weight", it->eviction_rank(now).weight},
                {"branches", branches}, {"superseded_by", it->superseded_by}});
            if (eviction_history.size() > 64) eviction_history.erase(eviction_history.begin());
        }
        GGML_ASSERT(it->children == 0);
        response_leases.invalidate(it->key, server_cache_now_ms());
        if (!it->parent_key.empty()) {
            const auto parent = by_key.find(it->parent_key);
            if (parent != by_key.end()) {
                GGML_ASSERT(parent->second->children > 0);
                parent->second->children--;
            }
        }
        const auto endpoint = endpoint_counts.find(it->p1_tokens);
        GGML_ASSERT(endpoint != endpoint_counts.end() && endpoint->second > 0);
        if (--endpoint->second == 0) {
            endpoint_counts.erase(endpoint);
        }
        size_cur -= it->size();
        size_tgt_cur -= it->data_tgt.resident_size() + it->data_tgt_reference.resident_size();
        size_dft_cur -= it->data_dft.resident_size();
        by_key.erase(it->key);
        pages.erase(it);
        if (count_eviction) {
            stats_cur.evictions++;
        }
    }

    void evict_graph(const std::string & key, bool count_eviction) {
        std::vector<std::string> children;
        for (const auto & item : pages) {
            if (item.parent_key == key) {
                children.push_back(item.key);
            }
        }
        for (const auto & child : children) {
            evict_graph(child, count_eviction);
        }
        const auto found = by_key.find(key);
        if (found != by_key.end()) {
            evict(found->second, count_eviction);
        }
    }

    template<typename Operation> bool mutate_lease(Operation operation) {
        const auto previous = response_leases.snapshot();
        if (!operation()) return false;
        if (persist()) return true;
        server_responses_leases::page_sizes available;
        for (const auto & node : pages) available.emplace(node.key,
            node.data_tgt.size() + node.data_dft.size() + node.data_tgt_reference.size());
        response_leases.restore(previous, available, server_cache_now_ms());
        persist();
        return false; // never acknowledge a lease publication whose manifest failed
    }

    using ram_json = server_page_ram_store::json;

    bool persist() {
        if (!ram_store.active()) return true;
        const auto started = ggml_time_us();
        stats_cur.persist_attempts++;
        ram_json graph = ram_json::array();
        for (auto & node : pages) {
            if (node.descriptor_dirty) {
                ram_json branches = ram_json::array();
                for (const auto & branch : node.branches) branches.push_back({branch.first, int(branch.second.first), int(branch.second.second)});
                node.descriptor = {{"key", node.key}, {"parent", node.parent_key},
                    {"p0_tokens", node.p0_tokens}, {"p1_tokens", node.p1_tokens}, {"p0_pos", node.p0_pos}, {"p1_pos", node.p1_pos},
                    {"coarse", node.coarse}, {"unowned_retained", node.unowned_retained}, {"unowned_interactive", node.unowned_interactive},
                    {"unowned_ephemeral", node.unowned_ephemeral}, {"unowned_ephemeral_interactive", node.unowned_ephemeral_interactive},
                    {"branches", branches}, {"page_tokens", page_tokens},
                    {"release_after_ms", node.release_after_ms}, {"last_access_ms", node.usage.last_access_ms}, {"retrievals", node.usage.retrievals},
                    {"hint_ms", node.usage.hint_ms},
                    {"superseded_at_ms", node.superseded_at_ms}, {"superseded_by", node.superseded_by},
                    {"eviction_metadata_bytes", node.size() - node.data_tgt.resident_size() - node.data_dft.resident_size() - node.data_tgt_reference.resident_size()},
                    {"target", node.data_tgt.metadata()}, {"draft", node.data_dft.metadata()}, {"reference", node.data_tgt_reference.metadata()}};
                node.descriptor_dirty = false;
                stats_cur.manifest_node_builds++;
            } else {
                stats_cur.manifest_node_reuses++;
            }
            graph.push_back(node.descriptor);
        }
        ram_json leases = ram_json::array();
        for (const auto & entry : response_leases.snapshot()) {
            const auto & state = entry.second;
            const auto & pending = state.pending;
            leases.push_back({{"scope", entry.first}, {"head", state.head}, {"generation", state.generation},
                {"expires", state.expires}, {"touched", state.touched}, {"pages", state.pages},
                {"pending", {{"revision", pending.revision}, {"generation", pending.generation},
                    {"expires", pending.expires}, {"pages", pending.pages}, {"complete", pending.endpoint_complete}}}});
        }
        const auto built = ggml_time_us();
        const bool ok = ram_store.write(std::move(graph), {{"response_leases", leases}, {"eviction_history", eviction_history}, {"tiers", server_page_ram_store::json::parse(tier_stats().dump())}});
        const auto finished = ggml_time_us();
        stats_cur.persist_build_us += built - started;
        stats_cur.persist_write_us += finished - built;
        stats_cur.persist_us += finished - started;
        return ok;
    }

    void recover() {
        if (!ram_store.active()) return;
        const auto graph = ram_store.read();
        const auto history = ram_store.metadata().value("eviction_history", ram_json::array());
        if (history.is_array()) {
            for (size_t i = history.size() > 64 ? history.size() - 64 : 0; i < history.size(); ++i) {
                if (history[i].is_object()) eviction_history.push_back(history[i]);
            }
        }
        if (!graph.is_array()) { ram_store.rejected++; persist(); return; }
        // Manifest order is LRU, not topology. Load parents before their descendants.
        std::unordered_map<std::string, size_t> lru_order;
        for (size_t i = 0; i < graph.size(); ++i) {
            if (graph[i].contains("key") && graph[i]["key"].is_string()) lru_order.emplace(graph[i]["key"].get<std::string>(), i);
        }
        std::vector<ram_json> pending(graph.begin(), graph.end());
        std::stable_sort(pending.begin(), pending.end(), [](const ram_json & a, const ram_json & b) {
            const auto endpoint = [](const ram_json & value) {
                try { return value.at("p1_tokens").get<size_t>(); } catch (...) { return SIZE_MAX; }
            };
            return endpoint(a) < endpoint(b);
        });
        for (const auto & entry : pending) {
            try {
                page node;
                node.key = entry.at("key").get<std::string>(); node.parent_key = entry.at("parent").get<std::string>();
                node.p0_tokens = entry.at("p0_tokens"); node.p1_tokens = entry.at("p1_tokens");
                node.p0_pos = entry.at("p0_pos"); node.p1_pos = entry.at("p1_pos");
                const uint64_t now = server_cache_now_ms();
                const auto unsigned_value = [](const ram_json & value) -> uint64_t {
                    if (value.is_number_unsigned()) return value.get<uint64_t>();
                    if (value.is_number_integer() && value.get<int64_t>() >= 0) return uint64_t(value.get<int64_t>());
                    return 0;
                };
                // Invalid optional policy history must neither fabricate activity
                // nor reject otherwise valid, checksum-verified model state.
                node.release_after_ms = std::min(unsigned_value(entry.value("release_after_ms", ram_json())), now + 30000);
                node.usage.last_access_ms = std::min(unsigned_value(entry.value("last_access_ms", ram_json())), now);
                node.superseded_at_ms = std::min(unsigned_value(entry.value("superseded_at_ms", ram_json())), now);
                if (entry.contains("superseded_by") && entry["superseded_by"].is_string() &&
                        entry["superseded_by"].get<std::string>().size() == 64) {
                    node.superseded_by = entry["superseded_by"].get<std::string>();
                } else { node.superseded_at_ms = 0; }
                node.usage.retrievals = uint32_t(std::min<uint64_t>(unsigned_value(entry.value("retrievals", ram_json())), 15));
                if (entry.contains("hint_ms") && entry.at("hint_ms").is_array() && entry.at("hint_ms").size() == 6) {
                    for (size_t i = 0; i < 6; ++i) node.usage.hint_ms[i] = std::min(unsigned_value(entry.at("hint_ms")[i]), now);
                }
                node.coarse = entry.at("coarse"); node.unowned_retained = entry.at("unowned_retained");
                node.unowned_interactive = entry.at("unowned_interactive");
                node.unowned_ephemeral = entry.contains("unowned_ephemeral") && entry.at("unowned_ephemeral").is_boolean() && entry.at("unowned_ephemeral").get<bool>();
                node.unowned_ephemeral_interactive = entry.contains("unowned_ephemeral_interactive") && entry.at("unowned_ephemeral_interactive").is_boolean() && entry.at("unowned_ephemeral_interactive").get<bool>();
                if (entry.at("page_tokens") != page_tokens || node.key.empty() || by_key.count(node.key) ||
                    node.p1_tokens <= node.p0_tokens || node.p1_pos <= node.p0_pos || node.p0_pos < 0)
                    throw std::runtime_error("invalid RAM page coordinates");
                const auto parent = by_key.find(node.parent_key);
                if (node.parent_key.empty() ? (node.p0_tokens != 0 || node.p0_pos != 0) :
                    (parent == by_key.end() || parent->second->p1_tokens != node.p0_tokens || parent->second->p1_pos != node.p0_pos))
                    throw std::runtime_error("incomplete RAM page chain");
                for (const auto & branch : entry.at("branches")) {
                    const auto priority = branch.at(1).get<int>(); const auto lifecycle = branch.at(2).get<int>();
                    if (priority < 0 || priority > 1 || lifecycle < 0 || lifecycle > 2) throw std::runtime_error("invalid RAM policy");
                    node.branches.emplace(branch.at(0).get<std::string>(), std::make_pair(server_cache_priority(priority), server_cache_lifecycle(lifecycle)));
                }
                if (node.size() > limit_size - size_cur) throw std::runtime_error("RAM metadata exceeds limit");
                size_t budget = limit_size - size_cur - node.size();
                ram_store.load(node.data_tgt, entry.at("target"), budget);
                ram_store.load(node.data_dft, entry.at("draft"), budget);
                ram_store.load(node.data_tgt_reference, entry.at("reference"), budget);
                if (node.data_tgt.empty()) throw std::runtime_error("empty RAM target state");
                size_cur += node.size(); size_tgt_cur += node.data_tgt.resident_size() + node.data_tgt_reference.resident_size(); size_dft_cur += node.data_dft.resident_size();
                if (parent != by_key.end()) parent->second->children++;
                endpoint_counts[node.p1_tokens]++;
                pages.push_back(std::move(node)); by_key.emplace(pages.back().key, std::prev(pages.end())); ram_store.recovered++;
            } catch (const std::exception &) { ram_store.rejected++; }
        }
        try {
            std::map<std::string, server_responses_leases::scope> recovered;
            for (const auto & entry : ram_store.metadata().value("response_leases", ram_json::array())) {
                server_responses_leases::scope state;
                state.head = entry.at("head").get<std::string>();
                state.generation = entry.at("generation").get<uint64_t>();
                state.expires = entry.at("expires").get<uint64_t>();
                state.touched = entry.at("touched").get<uint64_t>();
                state.pages = entry.at("pages").get<server_responses_leases::page_sizes>();
                const auto & pending = entry.at("pending");
                state.pending.revision = pending.at("revision").get<std::string>();
                state.pending.generation = pending.at("generation").get<uint64_t>();
                state.pending.expires = pending.at("expires").get<uint64_t>();
                state.pending.pages = pending.at("pages").get<server_responses_leases::page_sizes>();
                state.pending.endpoint_complete = pending.at("complete").get<bool>();
                if (!recovered.emplace(entry.at("scope").get<std::string>(), std::move(state)).second)
                    throw std::runtime_error("duplicate recovered response scope");
            }
            server_responses_leases::page_sizes available;
            for (const auto & node : pages) available.emplace(node.key,
                node.data_tgt.size() + node.data_dft.size() + node.data_tgt_reference.size());
            if (!response_leases.restore(std::move(recovered), available, server_cache_now_ms())) ram_store.rejected++;
        } catch (const std::exception &) { ram_store.rejected++; }
        // Loading topology must not turn old parents into recent cache entries.
        pages.sort([&](const page & a, const page & b) { return lru_order.at(a.key) < lru_order.at(b.key); });
        persist();
    }

    size_t limit_size;
    size_t page_tokens;
    size_t size_cur = 0;
    size_t size_tgt_cur = 0;
    size_t size_dft_cur = 0;
    page_list pages;
    std::unordered_map<std::string, page_list::iterator> by_key;
    std::map<size_t, size_t> endpoint_counts;
    counters stats_cur;
    uint64_t next_ssd_maintenance_ms = 0;
    ram_json eviction_history = ram_json::array();
    server_page_ram_store ram_store;
    server_responses_leases response_leases{limit_size / 2};
};

struct server_slot {
    int id;

    llama_context * ctx_tgt = nullptr;
    llama_context * ctx_dft = nullptr;

    common_memory mem;

    // multimodal
    mtmd_context * mctx = nullptr;
    mtmd::batch_ptr mbatch = nullptr;
    std::future<mtmd::batch_ptr> media_preparation;

    struct tts_ctx {
        mtmd_helper::gen_audio ctx;
        std::vector<float> h_state_prompt; // the first h_state after step_prompt done
        const float * h_state;
        llama_token   sampled;
        int32_t       n_decoded;
        bool          stop;
        bool is_supported() const {
            return ctx.valid();
        }
        void reset() {
            ctx.reset();
            h_state_prompt.clear();
            h_state   = nullptr;
            sampled   = LLAMA_TOKEN_NULL;
            n_decoded = 0;
            stop      = false;
        }
    };
    tts_ctx tts;

    // speculative decoding
    common_speculative * spec;

    std::vector<server_tokens::page_boundary> prompt_page_boundaries;

    llama_tokens spec_draft;

    // draft candidates per token in spec_draft; only draft-simple and draft-mtp fill it
    std::vector<std::vector<llama_token_data>> spec_draft_q;
    llama_tokens spec_prompt;
    std::vector<int32_t> spec_i_batch;
    common_prompt_checkpoint spec_ckpt;
    bool spec_is_replay = false;
    std::mt19937 spec_synth_rng;

    // TODO: move members that belong to the task (such as `generated`, `has_new_line`) to task_results_state
    //       see https://github.com/ggml-org/llama.cpp/pull/18283#issuecomment-3710175837
    std::shared_ptr<const server_task> task;
    std::shared_ptr<const server_task> task_prev; // used for debugging

    // used to determine the slot that has been used the longest
    int64_t t_last_used = -1;

    // generation props
    int32_t n_ctx   = 0;  // context size per slot
    int32_t n_keep  = 0;
    int32_t i_batch = -1;

    // effective generation limit for the current task, -1 means unlimited
    int32_t n_predict_max = -1;

    size_t last_nl_pos = 0;

    common_chat_input generated;
    std::string  debug_generated_text;
    llama_tokens generated_tokens;
    size_t n_sent_text = 0; // number of sent text character (i.e. handle partial UTF-8 on streaming)

    std::vector<completion_token_output> generated_token_probs;

    bool has_next_token = true;
    bool has_new_line   = false;
    bool truncated      = false;

    stop_type stop;

    std::string stopping_word;

    // state
    slot_state state = SLOT_STATE_IDLE;
    std::string capacity_file;
    bool waiting_for_prefill_workspace = false;
    int execution_worker = -1;
    size_t media_offset = 0;
    std::string cache_restore_status = "not_attempted";
    bool capacity_parked() const { return !capacity_file.empty(); }


    server_prompt prompt;

    bool prompt_save(server_prompt_cache & prompt_cache) const {
        if (prompt.tokens.size() == 0) {
            return false;
        }

        const size_t cur_size_tgt =           llama_state_seq_get_size_ext(ctx_tgt, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        const size_t cur_size_dft = ctx_dft ? llama_state_seq_get_size_ext(ctx_dft, id, LLAMA_STATE_SEQ_FLAGS_NONE) : 0;

        const size_t cur_size = cur_size_tgt + cur_size_dft;

        SRV_TRC(" - saving prompt with length %d, total state size = %.3f MiB (draft: %.3f MiB)\n",
                (int) prompt.tokens.size(), cur_size / (1024.0 * 1024.0), cur_size_dft / (1024.0 * 1024.0));

        auto * cur = prompt_cache.alloc(prompt, cur_size_tgt, cur_size_dft);
        if (cur == nullptr) {
            return false;
        }

        llama_state_seq_get_data_ext(ctx_tgt, cur->data.main.data(), cur_size_tgt, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        if (ctx_dft) {
            llama_state_seq_get_data_ext(ctx_dft, cur->data.drft.data(), cur_size_dft, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        }

        prompt_cache.save_shared(*cur);

        return true;
    }

    bool prompt_load(server_prompt_cache & prompt_cache, const server_tokens & tokens) {
        bool res = prompt_cache.load(prompt, tokens, ctx_tgt, ctx_dft, id);
        prompt.tokens.has_mtmd = mctx != nullptr;
        if (!res) {
            SLT_WRN(*this, "%s", "failed to load prompt from cache\n");
        }

        return res;
    }

    void prompt_save_shared() const {
        if (!server_shared_prompt_cache_enabled() || prompt.tokens.empty()) {
            return;
        }

        server_prompt_cache_state state {
            /*.prompt =*/ prompt.clone(),
            /*.data   =*/ {},
        };
        const size_t size_tgt = llama_state_seq_get_size_ext(ctx_tgt, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        const size_t size_dft = ctx_dft ? llama_state_seq_get_size_ext(ctx_dft, id, LLAMA_STATE_SEQ_FLAGS_NONE) : 0;
        try {
            state.data.main.resize(size_tgt);
            state.data.drft.resize(size_dft);
        } catch (const std::bad_alloc & e) {
            SLT_ERR(*this, "failed to allocate shared prompt state: %s\n", e.what());
            return;
        }
        llama_state_seq_get_data_ext(ctx_tgt, state.data.main.data(), size_tgt, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        if (ctx_dft) {
            llama_state_seq_get_data_ext(ctx_dft, state.data.drft.data(), size_dft, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        }
        server_shared_prompt_cache_save(state);
    }

    bool prompt_load_shared(const server_tokens & tokens) {
        bool failed_restore = false;
        server_shared_prompt_cache_load(prompt, tokens, ctx_tgt, ctx_dft, id, &failed_restore);
        prompt.tokens.has_mtmd = mctx != nullptr;
        return !failed_restore;
    }

    void prompt_clear() {
        SLT_TRC(*this, "clearing prompt with %zu tokens\n", prompt.tokens.size());

        mem.seq_rm(id, -1, -1);

        prompt.clear();
    }

    std::vector<common_adapter_lora_info> lora;
    int32_t alora_invocation_start = -1;

    // sampling
    json json_schema;

    common_sampler_ptr smpl;

    llama_token sampled; // in speculative mode, this is the last accepted token

    // for TTS models, this is the embd generated from prev step, decode this to generate next hidden state
    // corresponding to one token position (size = n_embd)
    // for realtime transcription, these are the rows from the last audio stream step (size = n_rows * n_embd)
    std::vector<float> inp_embd;

    // realtime transcription, see SERVER_TASK_TYPE_TRANSCRIBE_RT
    mtmd::audio_stream_ptr audio_stream;
    bool audio_wait = false; // the stream needs more audio before it can produce rows
    std::deque<int64_t> audio_commit_rows; // end row of each non-final commit not yet completed

    server_slot_stats stats;

    // accepted tokens per draft position
    // not in server_slot_stats to avoid copying to every task result
    std::vector<uint64_t> n_accepted_per_pos;

    std::function<void(int /* id_slot */)>   callback_on_release;
    std::function<void(const server_slot &)> callback_on_reset; // called before reset()

    // this is for printing timings with slot progress, not part of metrics
    int64_t t_print_last = 0;
    int32_t n_gen_last = 0;

    void reset() {
        if (media_preparation.valid()) {
            media_preparation.wait();
            media_preparation = {};
        }
        SLT_DBG(*this, "%s", "\n");

        if (capacity_parked()) {
            std::remove(capacity_file.c_str());
            capacity_file.clear();
            prompt.clear();
        }
        media_offset = 0;
        cache_restore_status = "not_attempted";
        spec_is_replay = false;

        last_nl_pos    = 0;
        generated      = {};
        has_new_line   = false;
        truncated      = false;
        stop           = STOP_TYPE_NONE;
        stopping_word  = "";
        n_sent_text    = 0;

        if (can_speculate()) {
            spec_draft.clear();
            spec_i_batch.clear();
            spec_ckpt.clear();
        }
        generated_tokens.clear();
        generated_token_probs.clear();
        prompt_page_boundaries.clear();
        json_schema = json();

        task_prev = std::move(task);
        task.reset();

        // note: callback_on_reset() must have run before this, see release()
        stats = {};
        n_accepted_per_pos.clear();

        n_predict_max = -1;

        llama_set_sampler(ctx_tgt, id, nullptr);

        // clear alora start
        alora_invocation_start = -1;

        // clear multimodal state
        mbatch.reset();

        tts.reset();

        audio_stream.reset();
        audio_wait = false;
        audio_commit_rows.clear();
        inp_embd.clear();
    }

    void init_sampler() const {
        common_sampler_reset(smpl.get());

        if (!task->need_sampling()) {
            return;
        }

        const int64_t t_start = ggml_time_us();

        int n_text = 0;

        for (int i = 0; i < (int) prompt.tokens.size(); i++) {
            const llama_token id = prompt.tokens[i];

            if (id != LLAMA_TOKEN_NULL) {
                common_sampler_accept(smpl.get(), id, false);
                n_text++;
            }
        }

        SLT_TRC(*this, "init sampler, took %0.2f ms, tokens: text = %d, total = %d\n",
                (ggml_time_us() - t_start) / 1000.0, n_text, (int) prompt.tokens.size());
    }

    bool need_embd() const {
        GGML_ASSERT(task);
        return task->need_embd();
    }

    bool can_split() const {
        GGML_ASSERT(task);
        // a decision over a non-causal context reads the whole prompt in one batch
        if (task->type == SERVER_TASK_TYPE_DECISION && !llama_get_causal_attn(ctx_tgt)) {
            return false;
        }
        // MTP supports splitting - uses task->need_embd() not need_embd()
        if (!task->need_embd()) {
            return true;
        }
        // if the context does not have a memory module then all embeddings have to be computed within a single ubatch
        if (!llama_get_memory(ctx_tgt)) {
            return false;
        }
        // context can be chunked/split if the pooling type is LAST
        const auto pooling = llama_pooling_type(ctx_tgt);
        if (pooling == LLAMA_POOLING_TYPE_LAST) {
            return true;
        }
        // causal rerankers read the last token and have a KV cache, so they can also be chunked/split.
        if (pooling == LLAMA_POOLING_TYPE_RANK && llama_get_causal_attn(ctx_tgt)) {
            return true;
        }
        // a decision task reads its outputs from the last batch
        if (task->type == SERVER_TASK_TYPE_DECISION) {
            return true;
        }
        return false;
    }

    bool can_batch_with(server_slot & other_slot) const {
        GGML_ASSERT(task);

        // a joint decision head reads the whole batch
        if (!task->decision.order.empty() || !other_slot.task->decision.order.empty()) {
            return false;
        }

        if (task->type != other_slot.task->type ||
            inp_embd.size() != other_slot.inp_embd.size() ||
            can_speculate() != other_slot.can_speculate() ||
            !are_lora_equal(lora, other_slot.lora)) {
            return false;
        }
        if (task->type == SERVER_TASK_TYPE_HIDDEN_STATE) {
            // The context exposes one row-major capture buffer per layer, with
            // no sequence-to-row map. Keep extraction requests isolated so a
            // slot can never consume another slot's rows.
            return false;
        }
        return true;
    }

    // returns -1 if the generation is limitless
    int32_t n_remaining() const {
        return n_predict_max == -1 ? -1 : n_predict_max - (int32_t) stats.n_gen;
    }

    bool has_budget() const {
        return n_predict_max == -1 || n_remaining() > 0;
    }

    bool is_processing() const {
        return state != SLOT_STATE_IDLE;
    }

    bool can_speculate() const {
        return !!spec;
    }

    // at temp 0 both p and q are point masses, so rejection is the same as sample-and-match
    bool use_spec_rejection() const {
        return task && task->params.sampling.temp > 0.0f;
    }

    void add_token(const completion_token_output & token) {
        if (!is_processing()) {
            SLT_WRN(*this, "%s", "slot is not processing\n");
            return;
        }

        generated_token_probs.push_back(token);
    }

    int get_n_draft_max() const {
        GGML_ASSERT(task);

        if (!can_speculate()) {
            return 0;
        }

        // determine the max draft that fits the current slot state
        // note: slot.prompt is not yet expanded with the `id` token sampled above
        //       also, need to leave space for 1 extra token to allow context shifts
        int n_draft_max = n_ctx - prompt.n_tokens() - 2;

        if (n_remaining() > 0) {
            n_draft_max = std::min(n_draft_max, n_remaining() - 1);
        }

        SLT_DBG(*this, "max possible draft: %d\n", n_draft_max);

        return n_draft_max;
    }

    // add sampled token of this slot to the batch, optionally add the speculative draft tokens if any
    void handle_last_sampled_token(server_batch & batch) {
        bool add_ok = true;
        // the audio stream has no token before the first step, track the position with a dummy token
        const llama_token tok_track = sampled == LLAMA_TOKEN_NULL ? 0 : sampled;
        if (spec_draft.empty()) {
            // no speculative decoding
            i_batch = batch.size();

            if (!inp_embd.empty()) {
                const int32_t n_rows = (int32_t) inp_embd.size() / batch.n_embd;
                GGML_ASSERT((int32_t) inp_embd.size() == n_rows * batch.n_embd);
                i_batch = batch.size() + n_rows - 1;
                for (int32_t i = 0; i < n_rows; i++) {
                    // text embeddings broadcast the same position across the M-RoPE sections
                    const llama_pos p = prompt.tokens.pos_next();
                    const llama_pos pos[GGML_MROPE_SECTIONS] = { p, p, p, 0 };
                    add_ok &= batch.add_embd(id, inp_embd.data() + (size_t) i * batch.n_embd, pos, i == n_rows - 1, false);
                    if (i < n_rows - 1) {
                        prompt.tokens.push_back(tok_track);
                    }
                }
                if (audio_stream) {
                    inp_embd.clear(); // rows are consumed, the next step produces new ones
                }
            } else {
                add_ok &= batch.add(id, sampled, prompt.tokens.pos_next(), true, false);
            }

            SLT_DBG(*this, "slot decode token, id=%d, n_ctx = %d, n_tokens = %d, truncated = %d\n",
                    sampled, n_ctx, prompt.n_tokens(), truncated);
        } else {
            SLT_DBG(*this, "generate_draft: id=%d, #tokens=%zu, #draft=%zu, pos_next=%d\n",
                    sampled, prompt.tokens.size(), spec_draft.size(), prompt.tokens.pos_next());

            GGML_ASSERT(spec_i_batch.empty());

            spec_i_batch.push_back(batch.size());
            for (size_t i = 0; i < spec_draft.size(); i++) {
                spec_i_batch.push_back(batch.size() + i + 1);
            }

            auto pos0 = prompt.tokens.pos_next();

            add_ok &= batch.add(id, sampled, pos0++, true, false);
            for (auto token : spec_draft) {
                add_ok &= batch.add(this->id, token, pos0++, true, false);
            }
        }

        GGML_ASSERT(add_ok && "batch must be large enough to hold the sampled and draft tokens");

        prompt.tokens.push_back(tok_track);
        prompt.tokens.insert(spec_draft);
    }

    void release() {
        if (is_processing()) {
            GGML_ASSERT(task);

            SLT_INF(*this, "stop processing: n_tokens = %d, truncated = %d\n", prompt.n_tokens(), truncated);

            t_last_used = ggml_time_us();

            state = SLOT_STATE_IDLE;

            // do not keep context of the child slots - the parent's context is enough
            // nor of the stateless slots - no task ever reuses it
            if (task->is_child() || task->is_stateless() || audio_stream) {
                prompt_clear();
            }

            // mtmd_helper always re-eval the whole prompt
            if (task->type == SERVER_TASK_TYPE_TTS) {
                prompt_clear();
            }

            callback_on_reset(*this);

            reset();

            callback_on_release(id);
        }
    }

    size_t find_stopping_strings(const std::string & text, const size_t last_token_size, bool is_full_stop) {
        GGML_ASSERT(task);

        size_t stop_pos = std::string::npos;

        for (const std::string & word : task->params.antiprompt) {
            size_t pos;

            if (is_full_stop) {
                const size_t tmp      = word.size() + last_token_size;
                const size_t from_pos = text.size() > tmp ? text.size() - tmp : 0;

                pos = text.find(word, from_pos);
            } else {
                // otherwise, partial stop
                pos = string_find_partial_stop(text, word);
            }

            if (pos != std::string::npos && (stop_pos == std::string::npos || pos < stop_pos)) {
                if (is_full_stop) {
                    stop           = STOP_TYPE_WORD;
                    stopping_word  = word;
                    has_next_token = false;
                }
                stop_pos = pos;
            }
        }

        return stop_pos;
    }

    void print_timings_tg() {
        if (stats.n_gen < 100) {
            return;
        }

        const int64_t t_now = ggml_time_us();

        if (t_now - t_print_last < 3*1000*1000) {
            return;
        }

        const double n_gen_second     = stats.n_gen_tps();
        const double n_gen_second_win = 1e6 / (t_now - t_print_last) * (stats.n_gen - n_gen_last);

        t_print_last = t_now;
        n_gen_last = stats.n_gen;

        SLT_INF(*this, "n_gen = %6d, tg = %6.2f t/s, tg_3s = %6.2f t/s\n", (int) stats.n_gen, n_gen_second, n_gen_second_win);
    }

    void print_timings_pp() const {
        const double t_prompt_total = stats.t_prompt_ms();

        if (t_prompt_total < 3000.0) {
            return;
        }

        const double n_prompt_second = stats.n_prompt_tps();
        const double f_progress = task->n_tokens() > 0 ? (double) prompt.n_tokens() / task->n_tokens() : 0.0;

        SLT_INF(*this, "prompt processing, n_tokens = %6d, progress = %.2f, t = %6.2f s / %.2f tokens per second\n",
                (int) stats.n_prompt_processed, f_progress, t_prompt_total / 1e3, n_prompt_second);
    }

    void print_timings() const {
        const double t_prompt_total = stats.t_prompt_ms();
        const double t_gen_total    = stats.t_gen_ms();

        const double t_prompt        = stats.t_prompt_per_token_ms();
        const double n_prompt_second = stats.n_prompt_tps();

        const double t_gen        = stats.t_gen_per_token_ms();
        const double n_gen_second = stats.n_gen_tps();

        SLT_INF(*this,
                "prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                t_prompt_total, (int) stats.n_prompt_processed, t_prompt, n_prompt_second);

        SLT_INF(*this,
                "       eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                t_gen_total, (int) stats.n_gen, t_gen, n_gen_second);

        SLT_INF(*this,
                "      total time = %10.2f ms / %5d tokens\n",
                t_prompt_total + t_gen_total, (int) (stats.n_prompt_processed + stats.n_gen));

        SLT_INF(*this,
                "   graphs reused = %10d\n",
                llama_perf_context(ctx_tgt).n_reused);

        SLT_INF(*this,
                "  graph reallocs = %10d    / %5d        (same size) / (changed size)\n",
                llama_perf_context(ctx_tgt).n_reallocs_same, llama_perf_context(ctx_tgt).n_reallocs_diff);

        const int32_t n_draft_total       = stats.n_draft_tokens;
        const int32_t n_draft_accepted    = stats.n_draft_accepted;
        const int32_t n_draft_verif_steps = stats.n_draft_verif_steps;

        if (n_draft_total > 0) {
            const float  draft_ratio  = (float) n_draft_accepted / n_draft_total;
            const double mean_acc_len = n_draft_verif_steps > 0 ? 1.0 + (double) n_draft_accepted / (double) n_draft_verif_steps : 1.0;

            std::string acceptance_rates_per_pos;
            if (n_draft_verif_steps > 0) {
                for (size_t i = 0; i < n_accepted_per_pos.size(); ++i) {
                    if (i > 0) {
                        acceptance_rates_per_pos += ", ";
                    }
                    acceptance_rates_per_pos += string_format("%.3f", (double) n_accepted_per_pos[i] / (double) n_draft_verif_steps);
                }
            }

            SLT_INF(*this,
                    "draft acceptance = %0.5f (%5d accepted / %5d generated), mean len = %5.2f\n",
                    draft_ratio, n_draft_accepted, n_draft_total, mean_acc_len);
            SLT_TRC(*this,
                    "     acc per pos = (%s)\n", acceptance_rates_per_pos.c_str());
        }

        common_speculative_print_stats(spec);
    }

    json to_json(bool only_metrics = false) const {
        json res;

        res = {
            {"id",            id},
            {"n_ctx",         n_ctx},
            {"speculative",   can_speculate()},
            {"is_processing", is_processing()},
        };

        const auto & ptask = task ? task : task_prev;

        if (ptask) {
            res["id_task"] = ptask->id;
            res["capacity_parked"] = capacity_parked();
            res["waiting_for_prefill_workspace"] = waiting_for_prefill_workspace;
            res["execution_worker"] = execution_worker;
            res["execution_priority"] = ptask->params.execution_priority;
            res["cache_restore_status"] = cache_restore_status;
            res["media_tokens_processed"] = media_offset;
            res["media_preparation_pending"] = media_preparation.valid();
            res["n_prompt_tokens_total"] = ptask->n_tokens();
            res["n_prompt_tokens"]           = (int32_t) prompt.tokens.size();
            res["n_prompt_tokens_processed"] = stats.n_prompt_processed;
            res["n_prompt_tokens_cache"]     = stats.n_prompt_cached;
            res["params"] = ptask->params.to_json(only_metrics);
            res["next_token"] = json::array({
                {
                    {"has_next_token", has_next_token},
                    {"has_new_line",   has_new_line},
                    {"n_remain",       n_remaining()},
                    {"n_decoded",      stats.n_gen},
                }
            });

            if (!only_metrics) {
                res["prompt"] = ptask->tokens.detokenize(ctx_tgt, true);
                res["generated"] = generated.empty() ? debug_generated_text : generated.text;
            }
        }

        return res;
    }

    // the other slot continues from the tokens processed so far
    void copy_prompt_to(server_slot & other) const {
        mem.seq_rm(other.id,     -1, -1);
        mem.seq_cp(id, other.id, -1, -1);

        other.prompt = prompt.clone();
    }

    void copy_state_to(server_slot & other) const {
        GGML_ASSERT(state == SLOT_STATE_DONE_PROMPT);

        mem.seq_rm(other.id,     -1, -1);
        mem.seq_cp(id, other.id, -1, -1);

        other.i_batch = i_batch;

        other.stats = stats;

        other.prompt = prompt.clone();
        other.init_sampler();
    }
};

// encode the mtmd chunk at idx, batched with as many of the next chunks as possible
// returns 0 on success
static int encode_mtmd_chunk(const server_slot & slot, mtmd::batch_ptr & mbatch, size_t idx, bool batch_more = true) {
    const auto & mctx = slot.mctx;
    const auto & input_tokens = slot.task->tokens;
    const auto & chunk = input_tokens.find_chunk(idx);

    mbatch.reset(mtmd_batch_init(mctx));
    int32_t res = mtmd_batch_add_chunk(mbatch.get(), chunk.get());
    GGML_ASSERT(res == 0); // we should never have an empty batch

    // try batching as much as possible
    int n_added = 1;
    size_t idx_cur = idx;
    while (res == 0 && batch_more) {
        auto [next_chunk, next_idx] = input_tokens.find_next_media_chunk(idx_cur);
        if (next_chunk == nullptr) {
            break;
        }
        res = mtmd_batch_add_chunk(mbatch.get(), next_chunk->get());
        n_added += (res == 0 ? 1 : 0);
        idx_cur = next_idx;
        SLT_DBG(slot, "try adding media chunk idx = %zu to batch, res = %d\n", next_idx, res);
        // if res != 0, batch is full or chunk is not compatible -> this loop breaks
    }

    // TODO @ngxson : move this log line to debug when it become more stable
    SLT_TRC(slot, "encoding mtmd batch from idx = %zu, n_chunks = %d\n", idx, n_added);

    res = mtmd_batch_encode(mbatch.get());
    if (res != 0) {
        SLT_ERR(slot, "failed to encode mtmd batch for chunk idx = %zu, res = %d\n", idx, res);
        return -1;
    }

    return 0;
}

struct pending_media_decode {
    int slot_id;
    size_t chunk_index;
    size_t total_tokens;
    std::unique_ptr<decode_embd_batch> storage;
    std::vector<int32_t> sequence_counts;
    std::vector<llama_seq_id *> sequences;
    std::vector<int8_t> outputs;
    llama_batch slice;
};

// The GLM execution lanes currently support causal, contiguous one-lane
// positions. Preserve the existing helper for every other media geometry.
static std::unique_ptr<pending_media_decode> prepare_media_wave(
        const server_slot & slot, size_t index, int quantum) {
    const auto & chunk = slot.task->tokens.find_chunk(index);
    if (!slot.mbatch || slot.can_speculate() ||
            mtmd_decode_use_non_causal(slot.mctx, chunk.get()) ||
            mtmd_decode_use_mrope(slot.mctx)) return {};
    float * encoded = mtmd_batch_get_output_embd(slot.mbatch.get(), chunk.get());
    if (!encoded) return {};
    const float * external = mtmd_input_chunk_get_embd(chunk.get());
    const auto * positions = mtmd_input_chunk_get_pos(chunk.get());
    const bool ext = external && positions;
    const int width = llama_model_n_embd_inp(llama_get_model(slot.ctx_tgt));
    if (ext && (mtmd_input_chunk_get_n_pos_per_embd(chunk.get()) != 1 ||
                mtmd_input_chunk_get_n_embd(chunk.get()) != (size_t) width)) return {};
    const size_t count = mtmd_input_chunk_get_n_tokens(chunk.get());
    if (!count || slot.media_offset >= count) return {};
    auto result = std::make_unique<pending_media_decode>();
    result->slot_id = slot.id;
    result->chunk_index = index;
    result->total_tokens = count;
    result->storage = std::make_unique<decode_embd_batch>(
        ext ? const_cast<float *>(external) : encoded, count, 1, width);
    const auto base = slot.prompt.tokens.pos_next();
    if (ext) result->storage->set_position_external(positions, base, slot.id);
    else result->storage->set_position_normal(base, slot.id);
    for (size_t i = 0; i < count; ++i) {
        if (result->storage->pos[i] != base + (llama_pos) i) return {};
    }
    const auto view = result->storage->get_view(slot.media_offset,
        std::min<size_t>(quantum, count - slot.media_offset));
    result->sequence_counts.assign(view.n_tokens, 1);
    result->sequences.assign(view.n_tokens, &result->storage->seq_id);
    result->outputs.assign(view.n_tokens, false);
    result->slice = { view.n_tokens, nullptr, const_cast<float *>(view.embd),
        const_cast<llama_pos *>(view.pos), result->sequence_counts.data(),
        result->sequences.data(), result->outputs.data() };
    return result;
}

// returns 0 on success
// caller need to update prompt.tokens after a successful call to keep track of the processing progress
// note: this is not a member of server_slot because we want to run it inside yield_to_queue
//       slot is passed as const to avoid accidental modification of the slot state
//       some pointers are allowed to be used, they are not used by to_json()
static int process_mtmd_chunk(const server_slot & slot, mtmd::batch_ptr & mbatch, size_t idx, size_t & n_tokens_out, int quantum, size_t & media_offset) {
    GGML_ASSERT(slot.mctx);
    const auto & mctx = slot.mctx;
    const auto & input_tokens = slot.task->tokens;
    const auto & chunk = input_tokens.find_chunk(idx);
    int32_t res = 0;

    auto try_decode = [&]() -> int32_t {
        if (mbatch) {
            float * embd = mtmd_batch_get_output_embd(mbatch.get(), chunk.get());
            if (embd) {
                if (quantum > 0 && !slot.can_speculate() && !mtmd_decode_use_non_causal(mctx, chunk.get())) {
                    const float * external = mtmd_input_chunk_get_embd(chunk.get());
                    const llama_pos * external_pos = mtmd_input_chunk_get_pos(chunk.get());
                    const int external_lanes = mtmd_input_chunk_get_n_pos_per_embd(chunk.get());
                    const bool ext = external && external_pos && external_lanes > 0;
                    const int count = mtmd_input_chunk_get_n_tokens(chunk.get());
                    const int width = llama_model_n_embd_inp(llama_get_model(slot.ctx_tgt));
                    if (ext && mtmd_input_chunk_get_n_embd(chunk.get()) != (size_t) width) return -1;
                    decode_embd_batch view(ext ? const_cast<float *>(external) : embd,
                        count, ext ? external_lanes : (mtmd_decode_use_mrope(mctx) ? 4 : 1), width);
                    const auto pos = slot.prompt.tokens.pos_next();
                    if (ext) view.set_position_external(external_pos, pos, slot.id);
                    else if (mtmd_decode_use_mrope(mctx)) {
                        if (mtmd_input_chunk_get_type(chunk.get()) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
                            std::vector<mtmd_decoder_pos> positions(count);
                            mtmd_helper_image_get_decoder_pos(mtmd_input_chunk_get_tokens_image(chunk.get()), pos, positions.data());
                            view.set_position_mrope_2d(positions, slot.id);
                        } else view.set_position_mrope_1d(pos, slot.id);
                    } else view.set_position_normal(pos, slot.id);
                    const int take = std::min<int>(quantum, count - media_offset);
                    auto * slice = view.render(slot.ctx_tgt, media_offset, take);
                    const int result = llama_process(slot.ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, slice);
                    if (result) return -1;
                    n_tokens_out = take;
                    media_offset += take;
                    if (media_offset < (size_t) count) return 2; // yield with exact partial state
                    media_offset = 0;
                    return 0;
                }
                struct cb_data_t {
                    common_speculative * spec;
                    llama_context * ctx;
                } cb_data = { slot.spec, slot.ctx_tgt };

                static auto cb = [](const mtmd_helper_embd_batch * b, void * user_data) {
                    const auto * data = static_cast<cb_data_t *>(user_data);

                    common_batch batch(data->ctx);
                    for (int32_t i = 0; i < b->n_tokens; ++i) {
                        llama_pos pos[GGML_MROPE_SECTIONS] = { 0, 0, 0, 0 };
                        for (int32_t j = 0; j < b->n_pos; ++j) {
                            pos[j] = b->pos[j * b->n_tokens + i];
                        }
                        batch.add_embd({ b->embd + (size_t) i * b->n_embd, 1, (size_t) b->n_embd }, pos, b->seq_id, false);
                    }

                    return common_speculative_process(data->spec, batch) ? 0 : 1;
                };

                llama_pos new_n_past; // unused for now
                res = mtmd_helper_decode_image_chunk(
                    mctx,
                    slot.ctx_tgt,
                    chunk.get(),
                    embd,
                    slot.prompt.tokens.pos_next(),
                    slot.id,
                    llama_n_batch(slot.ctx_tgt),
                    &new_n_past,
                    cb,
                    &cb_data
                );
                if (res != 0) {
                    SLT_ERR(slot, "failed to decode mtmd chunk, idx = %zu, res = %d\n", idx, res);
                    return -1;
                }
                n_tokens_out = mtmd_input_chunk_get_n_tokens(chunk.get());
                return 0; // success
            }
        }
        return 1; // (non-error) need to create & encode batch
    };

    // if the batch is already exist, try searching & encode
    res = try_decode();
    if (res == 0 || res == 2) {
        return res;
    }
    if (res < 0) {
        // fatal error
        return res;
    }

    // otherwise, the batch is either uninitialized or is used up
    // we need to create & encode a new batch
    if (encode_mtmd_chunk(slot, mbatch, idx, quantum == 0) != 0) {
        return -1;
    }

    return try_decode();
}

// add an encoded mtmd chunk to the batch of the text tokens, instead of decoding it on its own like process_mtmd_chunk()
// embd is the output of the encoder for this chunk
// returns false if the batch is full
static bool add_mtmd_chunk(const server_slot & slot, const mtmd_input_chunk * chunk, const float * embd, server_batch & batch) {
    // positions are the ones of mtmd_helper_decode_image_chunk()
    const auto * image    = mtmd_input_chunk_get_tokens_image(chunk);
    const bool   is_mrope = mtmd_decode_use_mrope(slot.mctx);
    const size_t n_tokens = mtmd_input_chunk_get_n_tokens(chunk);
    const size_t n_embd   = batch.n_embd;
    const llama_pos pos_0 = slot.prompt.tokens.pos_next();

    for (size_t i = 0; i < n_tokens; i++) {
        const llama_pos p = pos_0 + (llama_pos) i;
        llama_pos pos[GGML_MROPE_SECTIONS] = { p, p, p, p };
        if (is_mrope && image) {
            const mtmd_decoder_pos rel = mtmd_image_tokens_get_decoder_pos(image, pos_0, i);
            pos[0] = rel.t;
            pos[1] = rel.y;
            pos[2] = rel.x;
            pos[3] = rel.z;
        }
        if (!batch.add_embd(slot.id, embd + i * n_embd, pos, slot.need_embd(), /* is_prompt */ true)) {
            return false;
        }
    }
    return true;
}

//
// server_context_impl (private implementation)
//

struct server_context_impl {
    friend struct server_context;

public:
    // only use these pointers outside of this class:
    //  - when not in sleeping state
    //  - and, with thread-safe APIs (e.g., tokenizer calls)
    llama_model * model_tgt = nullptr;

    mtmd_context * mctx = nullptr;
    // note: video_params.ffmpeg_bin_dir points into params_base, which outlives this struct
    mtmd_helper_init_opt init_opt = mtmd_helper_init_opt_default();
    const llama_vocab * vocab = nullptr;

    server_responses_registry & request_registry() { return response_registry; } // CPU-only, safe while sleeping
    bool has_cap_tts() const {
        return mctx != nullptr && mtmd_gen_audio_get_info(mctx).type != MTMD_GEN_AUDIO_TYPE_NONE;
    }

    bool has_cap_chat() const {
        return mctx == nullptr || mtmd_helper_model_can_chat(ctx_tgt, mctx);
    }

    server_decision_context decision;

    server_queue    queue_tasks;
    server_response queue_results;

    // note: chat_params must not be refreshed upon existing sleeping state
    server_chat_params chat_params;

    server_state_callback_t callback_state = [](server_state, json) -> void {};

    server_context_impl() {
        mtmd_helper_log_set(common_log_default_callback, nullptr);
    }

    ~server_context_impl() {
        if (!sleeping) {
            // destroy() is already called when entering sleeping state
            // we don't call it again here to avoid double free
            destroy();
        }
    }

    server_metrics get_metrics() const {
        return metrics;
    }

    void reset_metrics_bucket() {
        metrics.reset_bucket();
    }

    json probe_prompt_cache(const server_tokens & tokens_new) const {
        json slots_json = json::array();
        int32_t best_slot_id = -1;
        int32_t best_common_prefix_tokens = 0;
        int32_t best_cached_tokens = 0;
        int32_t busy_slots = 0;
        int64_t cached_tokens_total = 0;

        for (const server_slot & slot : slots) {
            const bool busy = slot.is_processing();
            if (busy) {
                busy_slots++;
            }

            const int32_t cached_tokens = (int32_t) slot.prompt.tokens.size();
            const int32_t common_prefix_tokens = cached_tokens > 0
                ? slot.prompt.tokens.get_common_prefix(tokens_new)
                : 0;
            cached_tokens_total += cached_tokens;

            if (!busy && common_prefix_tokens > best_common_prefix_tokens) {
                best_slot_id = slot.id;
                best_common_prefix_tokens = common_prefix_tokens;
                best_cached_tokens = cached_tokens;
            }

            const char * state = "unknown";
            switch (slot.state) {
                case SLOT_STATE_IDLE:              state = "idle";              break;
                case SLOT_STATE_WAIT_OTHER:        state = "wait_other";        break;
                case SLOT_STATE_STARTED:           state = "started";           break;
                case SLOT_STATE_PROCESSING_PROMPT: state = "processing_prompt"; break;
                case SLOT_STATE_DONE_PROMPT:       state = "done_prompt";       break;
                case SLOT_STATE_GENERATING:        state = "generating";        break;
                default: break;
            }

            slots_json.push_back(json {
                {"id_slot", slot.id},
                {"state", state},
                {"busy", busy},
                {"cached_tokens", cached_tokens},
                {"common_prefix_tokens", common_prefix_tokens},
                {"prompt_match_fraction", tokens_new.empty() ? 0.0 : double(common_prefix_tokens) / double(tokens_new.size())},
                {"slot_keep_fraction", cached_tokens == 0 ? 0.0 : double(common_prefix_tokens) / double(cached_tokens)},
                {"last_used_us", slot.t_last_used},
            });
        }

        json page_cache_json = {
            {"enabled", prompt_page_cache != nullptr},
        };
        if (prompt_page_cache) {
            const auto & page_stats = prompt_page_cache->stats();
            page_cache_json["replacement_policy"] = prompt_page_cache->replacement_stats();
            page_cache_json["page_tokens"]    = prompt_page_cache->page_size_tokens();
            page_cache_json["matched_tokens"] = prompt_page_cache->match_tokens(tokens_new);
            page_cache_json["entries"]        = prompt_page_cache->count();
            page_cache_json["bytes"]          = prompt_page_cache->size();
            page_cache_json["target_bytes"]   = prompt_page_cache->target_size();
            page_cache_json["draft_bytes"]    = prompt_page_cache->draft_size();
            page_cache_json["metadata_bytes"] = prompt_page_cache->metadata_size();
            page_cache_json["limit_bytes"]    = prompt_page_cache->limit();
            page_cache_json["persist_attempts"] = page_stats.persist_attempts;
            page_cache_json["persist_us"] = page_stats.persist_us;
            page_cache_json["manifest_node_builds"] = page_stats.manifest_node_builds;
            page_cache_json["manifest_node_reuses"] = page_stats.manifest_node_reuses;
            page_cache_json["persist_build_us"] = page_stats.persist_build_us;
            page_cache_json["persist_write_us"] = page_stats.persist_write_us;
            page_cache_json["capture_sync_us"] = page_stats.capture_sync_us;
            page_cache_json["capture_alloc_us"] = page_stats.capture_alloc_us;
            page_cache_json["capture_copy_us"] = page_stats.capture_copy_us;
            page_cache_json["capture_seal_us"] = page_stats.capture_seal_us;
            page_cache_json["capture_attempts"] = page_stats.capture_attempts;
            page_cache_json["capture_us"]       = page_stats.capture_us;
            page_cache_json["capture_last_us"]  = page_stats.capture_last_us;
            page_cache_json["restore_attempts"] = page_stats.restore_attempts;
            page_cache_json["restore_us"]       = page_stats.restore_us;
            page_cache_json["restore_last_us"]  = page_stats.restore_last_us;
            page_cache_json["captures"]       = page_stats.captures;
            page_cache_json["capture_bridges"] = page_stats.capture_bridges;
            page_cache_json["hits"]           = page_stats.hits;
            page_cache_json["misses"]         = page_stats.misses;
            page_cache_json["restored_tokens"] = page_stats.restored_tokens;
            page_cache_json["evictions"]      = page_stats.evictions;
            page_cache_json["failures"]       = page_stats.failures;
            page_cache_json["verify_attempts"] = page_stats.verify_attempts;
            page_cache_json["verify_failures"] = page_stats.verify_failures;
            page_cache_json["branch_releases"] = page_stats.branch_releases;
            page_cache_json["live_prefix_adoptions"] = page_stats.live_prefix_adoptions;
            page_cache_json["resident_prefix_restores"] = page_stats.resident_prefix_restores;
            page_cache_json["resident_prefix_positions"] = page_stats.resident_prefix_positions;
            page_cache_json["persistence"] = prompt_page_cache->persistence_stats();
        }

        return {
            {"input_tokens", (int32_t) tokens_new.size()},
            {"best_slot_id", best_slot_id},
            {"best_common_prefix_tokens", best_common_prefix_tokens},
            {"best_cached_tokens", best_cached_tokens},
            {"best_prompt_match_fraction", tokens_new.empty() ? 0.0 : double(best_common_prefix_tokens) / double(tokens_new.size())},
            {"best_slot_keep_fraction", best_cached_tokens == 0 ? 0.0 : double(best_common_prefix_tokens) / double(best_cached_tokens)},
            {"cached_tokens_total", cached_tokens_total},
            {"cache_utilization", n_ctx <= 0 ? 0.0 : double(cached_tokens_total) / double(n_ctx)},
            {"n_ctx", n_ctx},
            {"slot_count", (int32_t) slots.size()},
            {"busy_slots", busy_slots},
            {"shared_prompt_cache_enabled", server_shared_prompt_cache_enabled()},
            {"page_cache", std::move(page_cache_json)},
            {"execution_memory", execution_memory_snapshot},
            {"execution_timing", json{
                {"worker_profiles", [&]() {
                    json result = json::array();
                    static const char * names[] = {"calls", "planning_wait_us", "planning_held_us",
                        "graph_submit_us", "output_us", "synchronize_us", "input_us", "graph_prepare_us",
                        "transport_dispatches", "transport_copies", "transport_bytes", "transport_copy_submit_us",
                        "transport_dependency_submit_us", "transport_blocking_wait_us",
                        "transport_sync_copy_us", "transport_compute_submit_us"};
                    for (size_t i = 0; i < execution_lanes.size(); ++i) {
                        json row{{"worker", i}};
                        const auto values = execution_lanes[i]->get_execution_profile();
                        for (size_t j = 0; j < 16; ++j) row[names[j]] = values[j];
                        const auto backends = execution_lanes[i]->get_execution_backend_names();
                        for (size_t j = 0; j < std::min(backends.size(), size_t(GGML_SCHED_TRANSPORT_PROFILE_BACKENDS)); ++j) {
                            row["transport_ready_wait_" + backends[j] + "_us"] = values[16 + j];
                            row["transport_ready_wait_" + backends[j] + "_calls"] =
                                values[16 + GGML_SCHED_TRANSPORT_PROFILE_BACKENDS + j];
                        }
                        result.push_back(std::move(row));
                    }
                    return result;
                }()},
                {"independent_contexts", execution_lanes.size()},
                {"execution_waves_with_media", execution_media_waves},
                {"execution_prompt_handoffs", execution_prompt_handoffs},
                {"execution_prompt_handoffs_while_prefill_pending", execution_prompt_handoffs_while_prefill_pending},
                {"execution_decode_completions", execution_decode_completions},
                {"execution_decode_completions_while_prefill_pending", execution_decode_while_prefill},
                {"execution_decode_resubmissions", execution_decode_resubmissions},
                {"media_preparations_started", media_preparations_started.load()},
                {"media_preparations_completed", media_preparations_completed.load()},
                {"execution_waves_by_lanes", snapshot_counters(execution_waves)},
                {"execution_prefill_waves_by_sequences", snapshot_counters(execution_prefill_waves)},
                {"execution_mixed_prefill_decode_waves", execution_mixed_waves.load()},
                {"serial_single_sequence", execution_fallback_single.load()},
                {"serial_lane_capacity", execution_fallback_capacity.load()},
                {"serial_pooled_rebuild", execution_fallback_rebuild.load()},
                {"serial_unsupported", execution_fallback_other.load()},
                {"pre_decode_wall_us", wall_pre_us.load()}, {"pre_decode_calls", wall_pre_calls.load()},
                {"post_decode_wall_us", wall_post_us.load()}, {"post_decode_calls", wall_post_calls.load()},
                {"prefill_decode_wall_us", wall_prefill_us.load()}, {"prefill_decode_calls", wall_prefill_calls.load()},
                {"decode_only_wall_us_by_sequences", snapshot_counters(wall_decode_us)},
                {"decode_only_calls_by_sequences", snapshot_counters(wall_decode_calls)},
            }},
            {"slots", slots_json},
        };
    }

private:
    // note: accessing these fields outside of this class is not thread-safe
    // use server_context methods instead

    common_params params_base;

    // Outlives both contexts, whose callbacks retain its role-specific userdata.
    std::unique_ptr<common_moe_trace> moe_trace;

    // note: keep these alive - they determine the lifetime of the model, context, etc.
    common_init_result_ptr llama_init;

    llama_context * ctx_tgt = nullptr;
    std::vector<std::unique_ptr<llama_context>> execution_lanes;
    std::vector<llama_context *> execution_lane_ptrs;
    int prefill_workers = 0;
    uint32_t prefill_worker_batch = 64;
    std::vector<uint32_t> execution_capacities;
    std::vector<std::unique_ptr<pending_media_decode>> pending_media;
    uint64_t execution_media_waves = 0;
    uint64_t execution_prompt_handoffs = 0;
    uint64_t execution_prompt_handoffs_while_prefill_pending = 0;
    uint64_t execution_decode_completions = 0;
    uint64_t execution_decode_while_prefill = 0;
    uint64_t execution_decode_resubmissions = 0;
    std::mutex media_preparation_mutex;
    std::atomic<uint64_t> media_preparations_started{0}, media_preparations_completed{0};

    server_batch batch;

    llama_model   * model_dft = nullptr;
    llama_context * ctx_dft   = nullptr;

    common_speculative_init_result_ptr spec_init;

    common_context_seq_rm_type ctx_tgt_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_NO;
    common_context_seq_rm_type ctx_dft_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_NO;

    common_speculative_ptr spec;

    bool add_bos_token = true;

    int32_t n_ctx; // total context for all clients / slots

    // set to llama_model_n_swa(model)
    // if swa_full is enabled, this is set to 0 to simulate a non-SWA model
    int32_t n_swa;

    // slots / clients
    std::vector<server_slot> slots;

    int trace = 0;        // env: LLAMA_TRACE
    int slots_debug = 0;  // env: LLAMA_SERVER_SLOTS_DEBUG
    int slots_n_diff = 0; // env: LLAMA_SERVER_SLOTS_N_DIFF

    int n_empty_consecutive = 0;

    std::unique_ptr<server_prompt_cache>      prompt_cache;
    std::unique_ptr<server_prompt_page_cache> prompt_page_cache;
    server_responses_registry response_registry;

    server_metrics metrics;

    // queued prompt stats - llama_decode() is async, so the timing is only valid after a sync
    // note: kept out of server_metrics, which is copied as-is into the task result
    int64_t  t_decode_start  = 0; // start of the last submitted decode
    int64_t  t_prompt_start  = 0; // start of the oldest queued prompt decode
    uint64_t n_prompt_queued = 0;

    json json_ui_settings = json::object();

    // Necessary similarity of prompt for slot selection
    float slot_prompt_similarity = 0.0f;

    std::string model_name; // name of the loaded model, to be used by API
    std::set<std::string> model_aliases; // additional names for the model
    std::set<std::string> model_tags;    // informational tags

    bool sleeping = false;

    int64_t t_last_load_progress_ms = 0;

    void destroy() {
        for (auto & slot : slots) {
            if (slot.media_preparation.valid()) {
                slot.media_preparation.wait();
                slot.media_preparation = {};
            }
        }
        spec.reset();
        spec_init.reset();

        ctx_dft   = nullptr;
        model_dft = nullptr;

        execution_lane_ptrs.clear();
        execution_lanes.clear();
        llama_init.reset();

        ctx_tgt = nullptr;
        model_tgt = nullptr;

        mtmd_free(mctx);
        mctx = nullptr;
        if (moe_trace) { moe_trace->finish("context_closed"); }
    }

    void handle_sleeping_state(bool new_state) {
        GGML_ASSERT(sleeping != new_state);
        if (new_state) {
            if (callback_state) {
                callback_state(SERVER_STATE_SLEEPING, {});
                // note: for sleeping == false, event is emitted by load_model()
            }
            SRV_INF("%s", "server is entering sleeping state\n");
            destroy();
        } else {
            SRV_INF("%s", "server is exiting sleeping state\n");
            if (!load_model(params_base)) {
                GGML_ABORT("failed to reload model after sleeping");
            }
        }
        sleeping = new_state;
    }

    struct load_progress_data {
        server_context_impl * ctx;
        std::string stage;
        std::vector<std::string> stages;
        int64_t t_last_load_progress_ms = 0;
        load_progress_data(server_context_impl * ctx, const std::string & stage) : ctx(ctx), stage(stage) {}
    };
    static bool load_progress_callback(float progress, void * user_data) {
        auto * d = static_cast<load_progress_data *>(user_data);
        GGML_ASSERT(d);
        // always emit the first and final sample; throttle the rest to one per 200ms
        {
            auto & t_last = d->t_last_load_progress_ms;
            const int64_t t_now = ggml_time_ms();
            const bool first = t_last == 0;
            const bool done  = progress >= 1.0f;
            const bool throttled = !first && !done && (t_now - t_last) < 200;
            if (throttled) {
                return true;
            }
            t_last = t_now;
        }
        if (d->ctx->callback_state) {
            d->ctx->callback_state(SERVER_STATE_LOADING, {
                {"stages", d->stages},
                {"current", d->stage},
                {"value", progress},
            });
        }
        return true;
    }

    // load the model and initialize llama_context
    // this may also be called to resume from sleeping state
    bool load_model(common_params & params) {
        load_progress_data load_progress_text  (this, "text_model");
        load_progress_data load_progress_mmproj(this, "mmproj_model");
        load_progress_data load_progress_spec  (this, "spec_model");

        const bool is_resume = sleeping;

        if (!is_resume) {
            const char * trace_path = std::getenv("LLAMA_MOE_TRACE_FILE");
            if (trace_path && *trace_path && params.cb_eval) {
                throw std::runtime_error("route capture cannot replace an existing evaluation callback");
            }
            moe_trace = common_moe_trace::from_env(params.n_parallel);
        }

        params_base = params;
        if (moe_trace && moe_trace->routes_enabled()) {
            params_base.cb_eval = common_moe_trace::callback;
            params_base.cb_eval_user_data = moe_trace->callback_data(common_moe_trace_role::target);
        }
        const auto output_limits = server_output_limits(params_base);
        params_base.n_outputs_max = output_limits.total;
        params_base.n_outputs_max_per_seq = output_limits.per_seq;

        const bool has_mmproj = !params.mmproj.path.empty();
        const bool has_visiontower = !params.visiontower_url.empty();
        const bool has_mtmd = has_mmproj || has_visiontower;
        const bool has_draft = params.speculative.has_dft();
        const bool spec_mtp = std::find(params_base.speculative.types.begin(),
                                        params_base.speculative.types.end(),
                                        COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params_base.speculative.types.end();
        const bool has_spec = has_draft || spec_mtp;

        if (callback_state) {
            std::vector<std::string> stages = {"text_model"};
            if (has_spec) {
                stages.push_back("spec_model");
            }
            if (has_mmproj) {
                stages.push_back("mmproj_model");
            }
            load_progress_text.stages   = stages;
            load_progress_mmproj.stages = stages;
            load_progress_spec.stages   = stages;

            // trigger 0% progress
            load_progress_callback(0.0f, &load_progress_text);
        }


        SRV_INF("loading model '%s'\n", params.model.get_name().c_str());
        SRV_TRC("local path '%s'\n", params.model.path.c_str());

        std::string & mmproj_path = params_base.mmproj.path;
        mtmd_context_params mparams = mtmd_context_params_default();
        if (has_mtmd) {
            mparams.use_gpu          = params_base.mmproj_use_gpu;
            mparams.device           = params_base.mmproj_device;
            mparams.print_timings    = false;
            mparams.n_threads        = params_base.cpuparams.n_threads;
            mparams.flash_attn_type  = params_base.flash_attn_type;
            mparams.warmup           = params_base.warmup;
            mparams.image_min_tokens = params_base.image_min_tokens;
            mparams.image_max_tokens = params_base.image_max_tokens;
            mparams.batch_max_tokens = params_base.mtmd_batch_max_tokens;
            mparams.media_marker     = get_media_marker();
            // progress callback
            mparams.progress_callback           = load_progress_callback;
            mparams.progress_callback_user_data = &load_progress_mmproj;
        }

        // get the memory usage of mmproj, also used to check image_max_tokens against n_ubatch
        mtmd_memory_usage mmproj_usage = {{}, -1, false};
        int64_t mmproj_usage_t_us = 0;
        if (has_mmproj) {
            int64_t t_start = ggml_time_us();
            mmproj_usage = mtmd_get_memory_usage(mmproj_path.c_str(), mparams);
            mmproj_usage_t_us = ggml_time_us() - t_start;
        }

        // optionally fit mmproj memory usage
        // on resume params_base already holds the margin from the first load
        if (has_mmproj && params_base.fit_params && !is_resume) {
            const auto & mmproj_mem = mmproj_usage.backend_mem_usage;
            const int64_t t_elapsed = mmproj_usage_t_us;
            if (!mmproj_mem.empty()) {
                size_t total = 0;
                for (auto & [dev, size] : mmproj_mem) {
                    total += size;
                }
                SRV_TRC("[mtmd] estimated worst-case memory usage of mmproj is %.2f MiB (took %.2f ms)\n", total / (1024.0 * 1024.0), t_elapsed / 1000.0);
                GGML_ASSERT(!params_base.fit_params_target.empty());
                for (auto & [dev, size] : mmproj_mem) {
                    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
                        if (ggml_backend_dev_get(i) == dev) {
                            if (i < params_base.fit_params_target.size()) {
                                SRV_DBG("[mtmd] adding %.2f MiB to fit_params_target for device %s\n", size / (1024.0 * 1024.0), ggml_backend_dev_name(dev));
                                params_base.fit_params_target[i] += size;
                            }
                            break;
                        }
                    }
                }
            } else {
                SRV_ERR("%s", "[mtmd] failed to get memory usage of mmproj\n");
            }
        }

        // note: the draft / MTP context is fitted together with the target model, see common_fit_extra_model

        // attach a progress callback
        {
            params_base.load_progress_callback = load_progress_callback;
            params_base.load_progress_callback_user_data = &load_progress_text;
        }

        server_load_progress_watchdog load_watchdog;
        load_watchdog.timeout_ms = server_load_watchdog_timeout_ms();
        if (load_watchdog.timeout_ms > 0) {
            load_watchdog.previous_callback = params_base.load_progress_callback;
            load_watchdog.previous_user_data = params_base.load_progress_callback_user_data;
            load_watchdog.model_path = params_base.model.path.c_str();
            params_base.load_progress_callback = server_load_progress_watchdog::callback;
            params_base.load_progress_callback_user_data = &load_watchdog;
            SRV_INF("load progress watchdog enabled: %.3f s stall timeout\n", load_watchdog.timeout_ms / 1000.0);
            load_watchdog.start();
        }

        llama_init = common_init_from_params(params_base);

        if (load_watchdog.timeout_ms > 0) {
            load_watchdog.stop();
            params_base.load_progress_callback = load_watchdog.previous_callback;
            params_base.load_progress_callback_user_data = load_watchdog.previous_user_data;
        }

        params.sampling = params_base.sampling;

        model_tgt = llama_init->model();
        ctx_tgt   = llama_init->context();

        if (model_tgt == nullptr) {
            SRV_ERR("failed to load model, '%s'\n", params_base.model.path.c_str());
            return false;
        }

        if (ctx_tgt == nullptr) {
            SRV_ERR("failed to create_context with model '%s'\n", params_base.model.path.c_str());
            return false;
        }

        vocab = llama_model_get_vocab(model_tgt);

        try {
            decision.init(model_tgt);
        } catch (const std::exception & e) {
            SRV_ERR("failed to init decision model: %s\n", e.what());
            return false;
        }

        n_ctx = llama_n_ctx(ctx_tgt);
        if (moe_trace) {
            moe_trace->metadata(llama_commit(), params_base.model.path, llm_arch_name(model_tgt->arch),
                                model_tgt->hparams.n_expert, model_tgt->hparams.n_expert_used_max());
        }

        add_bos_token = llama_vocab_get_add_bos(vocab);

        if (has_spec) {
            // spec_mtp doesn't use load a model internally, so we report 0.0 and 1.0 manually
            load_progress_callback(0.0f, &load_progress_spec);
            load_progress_spec.t_last_load_progress_ms = 0;  // reset so internal cbs aren't delayed

            {
                common_params params_dft = common_base_params_to_speculative(params_base);
                if (moe_trace && moe_trace->routes_enabled()) {
                    params_dft.cb_eval_user_data = moe_trace->callback_data(common_moe_trace_role::draft);
                }

                // progress callback
                params_dft.load_progress_callback           = load_progress_callback;
                params_dft.load_progress_callback_user_data = &load_progress_spec;

                spec_init = common_speculative_init_from_params(params_dft, model_tgt, ctx_tgt);
                model_dft = spec_init->model();
                ctx_dft   = spec_init->context();

                if (has_draft && model_dft == nullptr) {
                    SRV_ERR("failed to load draft model, '%s'\n", params_dft.model.path.c_str());
                    return false;
                }

                if (ctx_dft == nullptr) {
                    SRV_ERR("%s", "failed to create MTP context\n");
                    return false;
                }

                params_base.speculative.draft.ctx_tgt = ctx_tgt;
                params_base.speculative.draft.ctx_dft = ctx_dft;
            }

            load_progress_callback(1.0f, &load_progress_spec);
        }

        if (has_mtmd) {
            if (callback_state) {
                callback_state(SERVER_STATE_LOADING, {{"stage", has_mmproj ? "mmproj_model" : "visiontower_client"}});
            }

            if (!is_resume) {
                mtmd_helper_log_set(common_log_default_callback, nullptr);
            }

            // the image must fit in one ubatch if the model needs non-causal attention on it
            // a non-causal model also has the text of the prompt in that ubatch, leave half of it to the text
            {
                const int n_ubatch = llama_n_ubatch(ctx_tgt);
                const bool is_mixed = use_mixed_batch();
                if (is_mixed || mmproj_usage.use_non_causal) {
                    const int n_max = is_mixed ? n_ubatch / 2 : n_ubatch;
                    if (mmproj_usage.image_max_tokens > n_max) {
                        SRV_WRN("cap image_max_tokens (original=%d) to %d (n_ubatch = %d) because %s\n", mmproj_usage.image_max_tokens, n_max, n_ubatch,
                                is_mixed ? "the model processes the prompt in one ubatch" : "model needs non-causal attention on image");
                        SRV_WRN("%s\n", "increase n_ubatch (-ub) to increase vision token budget");
                        mparams.image_max_tokens = n_max;
                        mparams.image_min_tokens = std::min(mparams.image_min_tokens, n_max);
                    }
                }
            }

            mctx = has_mmproj
                ? mtmd_init_from_file(mmproj_path.c_str(), model_tgt, mparams)
                : mtmd_init_from_text_model(model_tgt, mparams);
            if (mctx == nullptr) {
                SRV_ERR("%s", "failed to initialize multimodal context\n");
                return false;
            }
            if (has_mmproj) {
                SRV_INF("loaded multimodal model, '%s'\n", mmproj_path.c_str());
            }
            mtmd_set_visiontower_url(mctx, params_base.visiontower_url.c_str());
            if (has_visiontower) {
                SRV_INF("configured external visiontower '%s'\n", params_base.visiontower_url.c_str());
            }

            init_opt.video_params.fps_target = params_base.video_fps;
            init_opt.video_params.timestamp_interval_ms = params_base.video_timestamp_interval_ms;
            init_opt.video_params.ffmpeg_bin_dir = params_base.video_ffmpeg_bin_dir.empty()
                                ? nullptr : params_base.video_ffmpeg_bin_dir.c_str();

            if (params_base.ctx_shift) {
                params_base.ctx_shift = false;
                SRV_WRN("%s\n", "ctx_shift is not supported by multimodal, it will be disabled");
            }

            if (params_base.n_cache_reuse) {
                params_base.n_cache_reuse = 0;
                SRV_WRN("%s\n", "cache_reuse is not supported by multimodal, it will be disabled");
            }
        }

        if (!llama_memory_can_shift(llama_get_memory(ctx_tgt))) {
            if (params_base.ctx_shift) {
                params_base.ctx_shift = false;
                SRV_WRN("%s\n", "ctx_shift is not supported by this context, it will be disabled");
            }

            if (params_base.n_cache_reuse) {
                params_base.n_cache_reuse = 0;
                SRV_WRN("%s\n", "cache_reuse is not supported by this context, it will be disabled");
            }
        }

        if (llama_model_n_swa(model_tgt) == 0) {
            if (params_base.swa_full) {
                params_base.swa_full = false;
                SRV_WRN("%s\n", "swa_full is not supported by this model, it will be disabled");
            }
        }

        n_swa = params_base.swa_full ? 0 : llama_model_n_swa(model_tgt);

        // Necessary similarity of prompt for slot selection
        slot_prompt_similarity = params_base.slot_prompt_similarity;

        const int n_ctx_train = llama_model_n_ctx_train(model_tgt);

        {
            // note: the capping itself is done in n_ctx_slot(), here we only report it
            const int n_ctx_seq = llama_n_ctx_seq(ctx_tgt);

            if (params_base.kv_unified_per_slot > 0) {
                if (n_ctx_seq > params_base.kv_unified_per_slot) {
                    SRV_INF("capping per-slot context (%d) to --kv-unified-per-slot (%d)\n",
                            n_ctx_seq, params_base.kv_unified_per_slot);
                } else if (params_base.kv_unified_per_slot > n_ctx_seq) {
                    // cap is above the per-slot pool capacity, so it can never bind
                    SRV_WRN(
                        "--kv-unified-per-slot (%d) exceeds the per-slot pool capacity (%d) - cap has no effect, "
                        "slots are limited to %d (raise the KV pool with -c, or unset -c to size it to "
                        "n_parallel * kv_unified_per_slot)\n",
                        params_base.kv_unified_per_slot, n_ctx_seq, n_ctx_seq);
                }
            }

            const int n_ctx_capped = params_base.kv_unified_per_slot > 0 ?
                std::min(n_ctx_seq, params_base.kv_unified_per_slot) : n_ctx_seq;

            if (n_ctx_capped > n_ctx_train) {
                SRV_WRN("the slot context (%d) exceeds the training context of the model (%d) - capping\n",
                        n_ctx_capped, n_ctx_train);
            }
        }

        slots.clear();

        ctx_tgt_seq_rm_type = common_context_can_seq_rm(ctx_tgt);
        if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
            SRV_WRN("%s", "speculative decoding not supported by this context\n");
        }

        if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
            SRV_TRC("%s", "speculative decoding will use checkpoints\n");
        }

        // setup slots
        SRV_INF("initializing, n_slots = %d, n_ctx_slot = %d, kv_unified = '%s'\n",
                params_base.n_parallel, n_ctx_slot(), params_base.kv_unified ? "true" : "false");

        // initialize slots
        for (int i = 0; i < params_base.n_parallel; i++) {
            slots.emplace_back();
        }

        // try speculative decoding
        if (ctx_tgt_seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
            try {
                spec.reset(common_speculative_init(params_base.speculative, params_base.n_parallel));
            } catch (const std::exception & e) {
                SRV_ERR("failed to initialize speculative decoding context: %s\n", e.what());
                if (params_base.speculative.has_synth()) {
                    return false;
                }
            }
        }

        if (ctx_dft) {
            ctx_dft_seq_rm_type = common_context_can_seq_rm(ctx_dft);
        }

        if (spec) {
            SRV_TRC("%s", "speculative decoding context initialized\n");
        } else {
            spec_init.reset();
            ctx_dft   = nullptr;
            model_dft = nullptr;
        }

        if (!spec && params_base.speculative.has_synth()) {
            SRV_ERR("%s", "synthetic acceptance requires an initialized speculative decoding context\n");
            return false;
        }

        if (const char * enabled = std::getenv("LLAMA_GLM5NEXT_EXECUTION_LANES")) {
            const int count = std::atoi(enabled);
            if (count != params_base.n_parallel || count < 2 || count > 8 ||
                    model_tgt->arch != LLM_ARCH_GLM5NEXT || !params_base.kv_unified || spec) {
                throw std::runtime_error("independent execution requires 2 through 8 unified GLM slots without speculation");
            }
            if (const char * value = std::getenv("LLAMA_GLM5NEXT_WORKER_BATCHES")) {
                std::string list(value);
                size_t offset = 0;
                while (offset < list.size()) {
                    const size_t end = list.find(',', offset);
                    const auto part = list.substr(offset, end == std::string::npos ? end : end - offset);
                    if (part.empty() || part.find_first_not_of("0123456789") != std::string::npos) {
                        throw std::runtime_error("worker batches require positive comma-separated integers");
                    }
                    const auto capacity = std::stoul(part);
                    if (!capacity || capacity > llama_n_batch(ctx_tgt)) {
                        throw std::runtime_error("worker batch exceeds logical batch");
                    }
                    execution_capacities.push_back(capacity);
                    if (end == std::string::npos) break;
                    offset = end + 1;
                    if (offset == list.size()) throw std::runtime_error("empty worker batch");
                }
                if (execution_capacities.size() != (size_t) count ||
                        !std::is_sorted(execution_capacities.rbegin(), execution_capacities.rend())) {
                    throw std::runtime_error("worker batches must match slot count in descending order");
                }
                prefill_workers = std::count_if(execution_capacities.begin(), execution_capacities.end(),
                        [](uint32_t capacity) { return capacity > 1; });
                if (!prefill_workers || prefill_workers == count) {
                    throw std::runtime_error("worker batches require both prefill and decode workers");
                }
                prefill_worker_batch = execution_capacities.front();
            } else {
                if (const char * value = std::getenv("LLAMA_GLM5NEXT_PREFILL_WORKERS")) {
                    prefill_workers = std::atoi(value);
                    if (prefill_workers < 1 || prefill_workers >= count) {
                        throw std::runtime_error("prefill workers must leave at least one decode worker");
                    }
                    const char * batch = std::getenv("LLAMA_GLM5NEXT_PREFILL_BATCH");
                    const int capacity = batch ? std::atoi(batch) : 256;
                    if (capacity < 1 || capacity > (int) llama_n_batch(ctx_tgt)) {
                        throw std::runtime_error("prefill workspace batch exceeds logical batch");
                    }
                    prefill_worker_batch = capacity;
                }
                for (int i = 0; i < count; ++i) {
                    execution_capacities.push_back(prefill_workers ?
                        (i < prefill_workers ? prefill_worker_batch : 1) :
                        std::min<uint32_t>(64, llama_n_ubatch(ctx_tgt)));
                }
            }
            for (const auto capacity : execution_capacities) {
                execution_lanes.push_back(ctx_tgt->create_execution_lane(capacity, capacity > 1 ? llama_n_batch(ctx_tgt) : 1));
                execution_lane_ptrs.push_back(execution_lanes.back().get());
            }
            SRV_INF("created %d independent execution contexts over shared model and KV\n", count);
        }

        refresh_execution_memory_snapshot();

        for (int i = 0; i < params_base.n_parallel; i++) {
            server_slot & slot = slots[i];

            slot.id      = i;
            slot.ctx_tgt = ctx_tgt;
            slot.ctx_dft = ctx_dft;
            slot.mem.init(ctx_tgt, ctx_dft);
            slot.spec    = spec.get();
            slot.n_ctx   = n_ctx_slot();

            slot.mctx                   = mctx;
            slot.prompt.tokens.has_mtmd = mctx != nullptr;

            if (has_cap_tts()) {
                slot.tts.ctx.init(ctx_tgt, mctx);
            }

            SLT_TRC(slot, "new slot, n_ctx = %d\n", slot.n_ctx);

            slot.callback_on_release = [this](int id_slot) {
                queue_tasks.pop_deferred_task(id_slot);
            };

            slot.callback_on_reset = [this](const server_slot & slot) {
                if (slot.task && !slot.task->params.response_scope.empty() && prompt_page_cache) {
                    auto & mutable_slot = const_cast<server_slot &>(slot);
                    capture_prompt_page(mutable_slot, true);
                    prompt_page_cache->observe_lease(slot.task->params.response_scope, slot.task->params.response_revision,
                        slot.task->params.cache_control.branch, slot.task->n_tokens(), slot.stats.n_gen > 0);
                }
                if (slot.task) response_registry.finish(slot.task->id);
                // flush the generated token stats before reset()
                if (slot.stats.n_gen > 0) {
                    metrics_on_prediction(slot);
                }
            };

            slot.reset();
        }

        {
            const char * LLAMA_TRACE = getenv("LLAMA_TRACE");
            trace = LLAMA_TRACE ? atoi(LLAMA_TRACE) : 0;

            if (trace) {
                SRV_WRN("LLAMA_TRACE = %d\n", trace);
            }
        }

        {
            const char * LLAMA_SERVER_SLOTS_DEBUG = getenv("LLAMA_SERVER_SLOTS_DEBUG");
            slots_debug = LLAMA_SERVER_SLOTS_DEBUG ? atoi(LLAMA_SERVER_SLOTS_DEBUG) : 0;

            if (slots_debug) {
                SRV_WRN("LLAMA_SERVER_SLOTS_DEBUG = %d\n", slots_debug);
            }
        }

        {
            const char * LLAMA_SERVER_SLOTS_N_DIFF = getenv("LLAMA_SERVER_SLOTS_N_DIFF");
            slots_n_diff = LLAMA_SERVER_SLOTS_N_DIFF ? atoi(LLAMA_SERVER_SLOTS_N_DIFF) : 0;

            if (slots_n_diff) {
                SRV_WRN("LLAMA_SERVER_SLOTS_N_DIFF = %d\n", slots_n_diff);
            }
        }

        // the update_slots() logic will always submit a maximum of n_batch or n_parallel tokens
        // note that n_batch can be > n_ctx (e.g. for non-causal attention models such as BERT where the KV cache is not used)
        {
            const int32_t n_batch = llama_n_batch(ctx_tgt);
            const int32_t n_embd  = llama_model_n_embd_inp(model_tgt);
            batch.init(ctx_tgt, std::max(n_batch, params_base.n_parallel), n_embd);
        }

        // decision requests rarely repeat a state, the prompt cache is opt-in for decision models
        if (decision.type != COMMON_DECISION_TYPE_NONE && !params_base.cache_ram_explicit) {
            SRV_INF("%s", "decision model: prompt cache disabled by default, use `--cache-ram N` to enable it\n");
            params_base.cache_ram_mib    = 0;
            params_base.cache_idle_slots = false;
        }

        if (params_base.cache_ram_mib != 0) {
            if (params_base.cache_ram_mib < 0) {
                SRV_TRC("prompt cache is enabled, size limit: %s\n", "no limit");
            } else {
                SRV_TRC("prompt cache is enabled, size limit: %d MiB\n", params_base.cache_ram_mib);
            }
            SRV_TRC("%s", "use `--cache-ram 0` to disable the prompt cache\n");

            prompt_cache = std::make_unique<server_prompt_cache>(params_base.cache_ram_mib, n_ctx);
        } else {
            SRV_TRC("%s", "prompt cache is disabled - use `--cache-ram N` to enable it\n");
        }
        SRV_TRC("%s", "for more info see https://github.com/ggml-org/llama.cpp/pull/16391\n");

        if (params_base.cache_page_ram_mib > 0) {
            if (uint32_t(params_base.cache_page_tokens) > llama_n_ctx_seq(ctx_tgt)) {
                SRV_WRN("--cache-page-tokens %d exceeds per-sequence context %u, disabling range-state page cache\n",
                        params_base.cache_page_tokens, llama_n_ctx_seq(ctx_tgt));
            } else {
                prompt_page_cache = std::make_unique<server_prompt_page_cache>(
                        size_t(params_base.cache_page_ram_mib) * 1024ull * 1024ull,
                        size_t(params_base.cache_page_tokens));
                SRV_TRC("range-state page cache enabled, page = %d tokens, limit = %d MiB\n",
                        params_base.cache_page_tokens, params_base.cache_page_ram_mib);
            }
        } else {
            SRV_TRC("%s", "range-state page cache disabled - use `--cache-page-ram N` to enable it\n");
        }

        if (params_base.n_ctx_checkpoints > 0) {
            SRV_TRC("context checkpoints enabled, max = %d, min spacing = %d\n",
                    params_base.n_ctx_checkpoints, params_base.checkpoint_min_step);
        } else {
            SRV_TRC("%s", "context checkpoints disabled\n");
        }
        if (server_shared_prompt_cache_enabled()) {
            SRV_WRN("%s", "shared prompt cache is enabled independently of local prompt cache\n");
        }

        if (!params_base.model_alias.empty()) {
            // backward compat: use first alias as model name
            model_name = *params_base.model_alias.begin();
        } else if (!params_base.model.get_name().empty()) {
            model_name = params_base.model.get_name();
        } else {
            // fallback: derive model name from file name
            auto model_path = std::filesystem::path(params_base.model.path);
            model_name = model_path.filename().string();
        }

        model_aliases = params_base.model_alias;
        model_tags    = params_base.model_tags;

        // propagate new defaults back to caller
        params = params_base;

        if (!is_resume) {
            return init();
        }

        if (callback_state) {
            callback_state(SERVER_STATE_READY, {});
        }

        return true;
    }

    // unlike load_model(), this is only called once during initialization
    bool init() {
        GGML_ASSERT(ctx_tgt   != nullptr);
        GGML_ASSERT(model_tgt != nullptr);

        GGML_ASSERT(!sleeping);

        // wiring up server queues
        queue_tasks.on_new_task([this](server_task && task, bool is_yielding) {
            return process_single_task(std::move(task), is_yielding);
        });
        queue_tasks.on_update_slots([this]() {
            update_slots();
        });
        queue_tasks.on_sleeping_state([this](bool sleeping) {
            handle_sleeping_state(sleeping);
        });

        metrics.init();

        if (params_base.cache_idle_slots) {
            if (params_base.cache_ram_mib == 0) {
                SRV_WRN("%s", "--cache-idle-slots requires --cache-ram, disabling\n");
                params_base.cache_idle_slots = false;
            } else {
                if (params_base.kv_unified) {
                    SRV_TRC("%s", "idle slots will be saved to prompt cache and cleared upon starting a new task\n");
                } else {
                    // without a unified KV cache, clearing a slot frees no reusable room, so we only
                    // publish a RAM-cache copy of idle slots (their KV stays in VRAM) [TAG_IDLE_SLOT_CLEAR]
                    SRV_TRC("%s", "idle slots will be saved to prompt cache upon starting a new task\n");
                }
                SRV_DBG("%s", "__TEST_TAG_CACHE_IDLE_SLOTS_ENABLED__\n");
            }
        }

        {
            const std::string & cfg = params_base.ui_config_json;
            if (!cfg.empty()) {
                try {
                    json json_settings = json::parse(cfg);
                    json_ui_settings = json_settings;
                } catch (const std::exception & e) {
                    SRV_ERR("%s: failed to parse UI config: %s\n", __func__, e.what());
                    return false;
                }
            }
        }

        // populate chat template params
        {
            common_chat_templates_ptr chat_templates;
            bool enable_thinking = false;

            try {
                chat_templates = common_chat_templates_init(model_tgt, params_base.chat_template);

                SRV_TRC("%s: chat template, example_format: '%s'\n", __func__,
                    common_chat_format_example(chat_templates.get(), params_base.use_jinja, params_base.default_template_kwargs).c_str());

                // thinking is enabled if:
                // 1. It's not explicitly disabled via --reasoning off
                // 2. The chat template supports it
                const bool template_supports_thinking = params_base.use_jinja && common_chat_templates_support_enable_thinking(chat_templates.get());
                enable_thinking = params_base.enable_reasoning != 0 && template_supports_thinking;
                SRV_TRC("%s: chat template, thinking = %d\n", __func__, enable_thinking);
            } catch (const std::exception & e) {
                SRV_ERR("%s: chat template parsing error: %s\n", __func__, e.what());
                SRV_ERR("%s: please consider disabling jinja via --no-jinja, or use a custom chat template via --chat-template\n", __func__);
                SRV_ERR("%s: for example: --no-jinja --chat-template chatml\n", __func__);
                return false;
            }

            // IMPORTANT: chat_params is reused across sleeping / resuming states,
            //            never store llama_context/llama_model pointers in chat_params,
            //            as they may be invalidated after sleeping
            chat_params = {
                /* use_jinja             */ params_base.use_jinja,
                /* prefill_assistant     */ params_base.prefill_assistant,
                /* reasoning_format      */ params_base.reasoning_format,
                /* chat_template_kwargs  */ params_base.default_template_kwargs,
                /* tmpls                 */ std::move(chat_templates),
                /* allow_image           */ mctx ? (mtmd_support_vision(mctx) || !params_base.visiontower_url.empty()) : false,
                /* allow_audio           */ mctx ? mtmd_support_audio (mctx) : false,
                /* allow_video           */ mctx ? (mtmd_helper_support_video(mctx) || !params_base.visiontower_url.empty()) : false,
                /* enable_thinking       */ enable_thinking,
                /* reasoning_budget      */ params_base.sampling.reasoning_budget_tokens,
                /* reasoning_budget_msg  */ params_base.sampling.reasoning_budget_message,
                /* media_path            */ params_base.media_path,
                /* force_pure_content    */ params_base.force_pure_content_parser
            };

            {
                auto caps = common_chat_templates_get_caps(chat_params.tmpls.get());
                auto it = params_base.default_template_kwargs.find("preserve_reasoning");
                bool supported = caps.at("supports_preserve_reasoning");
                bool specified = params_base.preserve_reasoning_specified;
                // note: the kwarg is enabled by default if not specified explicitly, so check the value
                bool enabled = it != params_base.default_template_kwargs.end() && it->second == "true";
                if (supported) {
                    SRV_TRC("preserve_reasoning kwarg: %s\n",
                            it == params_base.default_template_kwargs.end() ? "unset (template default)" : it->second.c_str());
                } else {
                    SRV_TRC("%s", "preserve_reasoning kwarg: not supported by template\n");
                }
                if (supported && !specified) {
                    SRV_WRN("%s", "chat template supports preserving reasoning, it is enabled by default (may use more tokens, disable via --no-reasoning-preserve)\n");
                }
                if (supported && !enabled) {
                    SRV_INF("%s", "chat template supports preserving reasoning, consider enabling it via --reasoning-preserve\n");
                }
                if (!supported && specified && enabled) {
                    SRV_WRN("%s", "chat template does NOT support preserving reasoning, --reasoning-preserve has no effect\n");
                }
            }
        }

        return true;
    }

    server_slot * get_slot_by_id(int id_slot) {
        // note: allow id_slot to be out of bounds (wrap around)
        id_slot = id_slot % slots.size();

        for (server_slot & slot : slots) {
            if (slot.id == id_slot) {
                return &slot;
            }
        }

        return nullptr;
    }

    server_slot * get_slot_by_cmpl_id(const std::string & cmpl_id) {
        if (cmpl_id.empty()) {
            return nullptr;
        }

        for (server_slot & slot : slots) {
            if (slot.is_processing() && slot.task && slot.task->params.oaicompat_cmpl_id == cmpl_id) {
                return &slot;
            }
        }

        return nullptr;
    }

    server_slot * get_available_slot(const server_task & task) {
        server_slot * ret = nullptr;

        bool update_cache = false;

        // if a specific slot is requested, use it (still goes through cache update logic below)
        if (task.id_slot != -1) {
            ret = get_slot_by_id(task.id_slot);
            if (ret) {
                // a busy slot is returned untouched, the caller defers the task
                if (ret->is_processing()) {
                    return ret;
                }

                SLT_INF(*ret, "selected slot by id (%d)\n", task.id_slot);
            }
        }

        // find the slot that has at least n% prompt similarity
        if (slot_prompt_similarity != 0.0f) {
            float f_sim_best = 0;

            for (server_slot & slot : slots) {
                if (task.id_slot != -1 && slot.id != task.id_slot) {
                    continue;
                }

                // skip the slot if it is not available
                if (slot.is_processing()) {
                    SLT_TRC(slot, " - skipping, is_processing = %d\n", slot.is_processing());
                    continue;
                }

                const auto & tokens = slot.prompt.tokens;

                // skip the slot if it does not contains cached tokens
                if (tokens.empty()) {
                    SLT_TRC(slot, "%s", " - skipping, slot is empty\n");
                    continue;
                }

                // fraction of the Longest Common Prefix length with respect to the input prompt length
                const size_t lcp_len = tokens.get_common_prefix(task.tokens);
                const float f_sim_cur = float(lcp_len) / task.tokens.size();

                SLT_TRC(slot, " - checking sim = %.3f (%zu/%zu) > %.3f\n", f_sim_cur, lcp_len, task.tokens.size(), slot_prompt_similarity);

                // select the current slot if the criteria match
                if (f_sim_cur > f_sim_best && f_sim_cur > slot_prompt_similarity) {
                    f_sim_best = f_sim_cur;

                    ret = &slot;
                }
            }

            if (ret != nullptr) {
                const float f_keep = (f_sim_best*task.tokens.size()) / ret->prompt.tokens.size();

                if (task.id_slot == -1) {
                    SLT_INF(*ret, "selected slot by LCP similarity, f_sim_best = %.3f (> %.3f thold), f_keep = %.3f\n",
                            f_sim_best, slot_prompt_similarity, f_keep);
                }

                // if we are about to lose a large portion of the existing context - save it in the prompt cache
                if (f_keep < 0.5f) {
                    update_cache = true;
                }
            }
        }

        // find an empty slot, or the one that has been least recently used
        if (ret == nullptr) {
            const auto free_slots = get_free_slots(1, -1);

            if (!free_slots.empty()) {
                ret = free_slots[0];

                SLT_INF(*ret, "selected slot by LRU, n_tokens = %d, t_last = %" PRId64 "\n", ret->prompt.n_tokens(), ret->t_last_used);

                update_cache = true;
            }
        }

        if (ret) {
            const auto & tokens = ret->prompt.tokens;

            // cache prompts only for completion tasks
            update_cache = update_cache && !task.is_stateless() && task.type != SERVER_TASK_TYPE_HIDDEN_STATE;
            update_cache = update_cache && (prompt_cache || server_shared_prompt_cache_enabled());

            if (update_cache) {
                SRV_TRC("%s", "updating prompt cache\n");

                const int64_t t_start = ggml_time_us();

                // don't save the slot's state if its context is empty
                if (tokens.size() > 0) {
                    if (prompt_cache) {
                        ret->prompt_save(*prompt_cache);
                    } else {
                        ret->prompt_save_shared();
                    }
                }

                const bool loaded_prompt = prompt_cache
                    ? ret->prompt_load(*prompt_cache, task.tokens)
                    : ret->prompt_load_shared(task.tokens);
                if (!loaded_prompt) {
                    ret->prompt_clear();
                }

                if (prompt_cache) {
                    prompt_cache->update();
                }

                SRV_TRC("prompt cache update took %.2f ms\n", (ggml_time_us() - t_start) / 1000.0);
            }
        }

        return ret;
    }

    // return true if at least one slot has been cleared
    // TODO: improve logic
    //       - smarter decision which slot to clear (LRU or longest prompt?)
    //       - move slot to level 2 cache instead of removing?
    //       - instead of purging, try to store and resume later?
    bool try_clear_idle_slots(int except_slot = -1) {
        bool res = false;

        if (!params_base.kv_unified) {
            return res;
        }

        for (auto & slot : slots) {
            if (slot.is_processing() || slot.id == except_slot) {
                continue;
            }

            if (slot.prompt.n_tokens() > 0) {
                SRV_WRN("purging slot %d with %zu tokens\n", slot.id, slot.prompt.tokens.size());

                slot.prompt_clear();

                res = true;

                // clear slots one by one
                break;
            }
        }

        return res;
    }

    std::vector<common_adapter_lora_info> construct_lora_list(const std::map<int, float> & config) const {
        std::vector<common_adapter_lora_info> output = params_base.lora_adapters; // copy
        for (size_t i = 0; i < output.size(); ++i) {
            auto it = config.find(i);
            if (it != config.end()) {
                output[i].scale = it->second;
            } else {
                output[i].scale = 0.0f;
            }
        }
        return output;
    }

    bool prompt_page_cache_eligible(const server_slot & slot, const server_task & task) const {
        return prompt_page_cache &&
            task.type == SERVER_TASK_TYPE_COMPLETION &&
            !task.is_parent() &&
            !task.is_child() &&
            lora_get_enabled_ids(slot.lora).empty() &&
            task.tokens.has_stable_media_identity();
    }

    bool prompt_page_cache_boundary(const server_slot & slot) const {
        if (!slot.task || !prompt_page_cache_eligible(slot, *slot.task)) {
            return false;
        }
        const size_t n_tokens = slot.prompt.n_tokens();
        const llama_pos position = slot.prompt.tokens.pos_next();
        return std::any_of(
                slot.prompt_page_boundaries.begin(), slot.prompt_page_boundaries.end(),
                [&](const auto & boundary) {
                    return boundary.token_count == n_tokens && boundary.position == position;
                });
    }

    const server_tokens::page_boundary * next_prompt_page_boundary(const server_slot & slot) const {
        const size_t n_tokens = slot.prompt.n_tokens();
        for (const auto & boundary : slot.prompt_page_boundaries) {
            if (boundary.token_count > n_tokens) {
                return &boundary;
            }
        }
        return nullptr;
    }

    bool capture_prompt_page(server_slot & slot, bool exact_endpoint = false) {
        if (slot.capacity_parked() || slot.media_offset || !slot.task || slot.task->params.cache_control.lifecycle == server_cache_lifecycle::one_shot ||
                (!exact_endpoint && !prompt_page_cache_boundary(slot)) ||
                !prompt_page_cache_eligible(slot, *slot.task) ||
                slot.prompt.tokens.empty() || slot.prompt.tokens.pos_next() <= 0) {
            return false;
        }
        llama_context * page_ctx_dft = slot.can_speculate() ? ctx_dft : nullptr;
        if (!prompt_page_cache->capture(
                    slot.prompt.tokens, ctx_tgt, page_ctx_dft, slot.id, slot.task->params.cache_control, exact_endpoint)) {
            SLT_WRN(slot, "failed to capture range-state page ending at token %zu / position %d\n",
                    slot.prompt.tokens.size(), slot.prompt.tokens.pos_next());
            return false;
        }
        if (!slot.task->params.response_scope.empty()) {
            prompt_page_cache->observe_lease(slot.task->params.response_scope, slot.task->params.response_revision,
                slot.task->params.cache_control.branch, 0, false);
        }
        SLT_TRC(slot, "captured range-state page ending at token %zu / position %d, cache = %zu pages / %.3f MiB\n",
                slot.prompt.tokens.size(), slot.prompt.tokens.pos_next(), prompt_page_cache->count(),
                prompt_page_cache->size() / (1024.0 * 1024.0));
        return true;
    }

    uint32_t execution_continuation_budget() const {
        return std::min<uint32_t>(512, llama_n_ctx(ctx_tgt) / 64);
    }

    uint32_t rolling_decode_reserve() const {
        return execution_lane_ptrs.empty() ? 128 :
            std::max<uint32_t>(128, 2 * execution_continuation_budget() + 16);
    }

    bool unified_glm_capacity(const server_slot & candidate, const server_task & task) const {
        if (!params_base.kv_unified || model_tgt->arch != LLM_ARCH_GLM5NEXT) { return true; }
        for (const auto & active : slots) if (active.capacity_parked()) return false;
#if defined(__linux__)
        const bool rolling = !spec; // SSD parking currently preserves non-speculative state only
#else
        const bool rolling = false;
#endif
        const auto budget = [&](const server_task & request, int cap) -> uint32_t {
            const int predict = request.params.n_predict != -1 ? request.params.n_predict : params_base.n_predict;
            const uint64_t prompt = std::max<int64_t>(request.n_tokens(), request.tokens.pos_next());
            const uint64_t output = predict < 0 ? uint64_t(cap) : uint64_t(predict);
            return (uint32_t) std::min<uint64_t>(cap,
                prompt + (request.params.prefill_only ? 0 : (rolling ? std::min<uint64_t>(rolling_decode_reserve(), output) : output)));
        };
        std::vector<std::pair<llama_seq_id, uint32_t>> budgets;
        budgets.emplace_back(candidate.id, budget(task, candidate.n_ctx));
        for (const auto & active : slots) {
            if (active.id != candidate.id && active.is_processing() && !active.capacity_parked() && active.task) {
                budgets.emplace_back(active.id, budget(*active.task, active.n_ctx));
            }
        }
        return llama_glm5next_reserved_cells(ctx_tgt, budgets) <= llama_n_ctx(ctx_tgt);
    }

    void share_restored_prefix(server_slot & slot, const server_task & task) {
        // Page restore supplies the destination recurrent endpoint. Only the
        // immutable attention/indexer prefix can be adopted from a diverged peer.
        if (!params_base.kv_unified || slot.can_speculate() ||
                !prompt_page_cache_eligible(slot, task)) { return; }
        for (const auto & donor : slots) {
            if (donor.id == slot.id || donor.can_speculate() ||
                    !are_lora_equal(donor.lora, slot.lora)) { continue; }
            const size_t common = std::min(
                donor.prompt.tokens.get_common_prefix(task.tokens),
                slot.prompt.tokens.get_common_prefix(task.tokens));
            if (!common || !task.tokens.complete_boundary(common)) { continue; }
            const llama_pos end = task.tokens.pos_next(common);
            if (llama_glm5next_share_prefix(ctx_tgt, donor.id, slot.id, end)) {
                SLT_INF(slot, "shared %d restored prefix positions with slot %d; recurrent endpoint retained\n", end, donor.id);
                return;
            }
        }
    }

    void restore_prompt_pages(server_slot & slot, const server_task & task) {
        // Task token containers describe an individual request and plain-text
        // tasks intentionally have MTMD disabled. The persistent slot instead
        // inherits the server capability. Reassert it before every exit path,
        // including a page-cache miss followed by a media continuation.
        slot.prompt.tokens.has_mtmd = mctx != nullptr;

        if (!prompt_page_cache_eligible(slot, task)) {
            slot.cache_restore_status = "ineligible";
            return;
        }

        const size_t current_lcp = slot.prompt.tokens.get_common_prefix(task.tokens);
        // Hybrid/recurrent state can report that memory supports shifting
        // while still rejecting truncation inside its recurrent endpoint.
        // The later prompt-rewind guard would then discard the entire prefix.
        const bool live_requires_rewind = current_lcp < slot.prompt.tokens.size() &&
            (llama_model_is_hybrid(model_tgt) || llama_model_is_recurrent(model_tgt));
        bool failed = false;
        llama_context * page_ctx_dft = slot.can_speculate() ? ctx_dft : nullptr;
        std::vector<std::pair<llama_seq_id, size_t>> resident_donors;
        if (params_base.kv_unified && !slot.can_speculate()) {
            for (const auto & donor : slots) {
                if (donor.id != slot.id && !donor.can_speculate() && are_lora_equal(donor.lora, slot.lora)) {
                    resident_donors.emplace_back(donor.id, donor.prompt.tokens.get_common_prefix(task.tokens));
                }
            }
        }
        auto seed_prefix = [&](size_t count, llama_pos end) {
            if (!count || !task.tokens.complete_boundary(count)) return false;
            for (const auto & donor : resident_donors) {
                if (donor.second >= count && llama_glm5next_seed_prefix(ctx_tgt, donor.first, slot.id, end)) return true;
            }
            return false;
        };
        bool capacity_blocked = false;
        auto reserve_restore = [&](llama_pos end) {
            if (!params_base.kv_unified || model_tgt->arch != LLM_ARCH_GLM5NEXT || spec) return true;
            std::vector<std::pair<llama_seq_id, uint32_t>> budgets;
            for (const auto & resident : slots) budgets.emplace_back(resident.id, resident.id == slot.id ? end : 0);
            while (llama_glm5next_reserved_cells(ctx_tgt, budgets) > llama_n_ctx(ctx_tgt)) {
                if (!try_clear_idle_slots(slot.id)) { capacity_blocked = true; return false; }
            }
            return true;
        };
        const size_t restored = prompt_page_cache->restore(
                task.tokens, current_lcp, live_requires_rewind, ctx_tgt, page_ctx_dft,
                slot.id, task.params.cache_control, failed, seed_prefix, reserve_restore);
        if (capacity_blocked) {
            slot.cache_restore_status = "capacity_wait";
            slot.prompt.clear();
            return;
        }
        if (failed) {
            slot.cache_restore_status = "restore_failed";
            slot.prompt.clear();
            SLT_WRN(slot, "%s", "range-state page restore failed; destination sequence was cleared\n");
            return;
        }
        if (restored == 0) {
            slot.cache_restore_status = current_lcp ? "live_prefix" : "no_matching_checkpoint";
            return;
        }
        slot.cache_restore_status = "restored";
        slot.prompt.tokens = task.tokens.clone();
        // Text tasks intentionally tokenize into a non-MTMD container so they
        // can use text/speculative helpers. Once adopted by a persistent slot,
        // restore the slot capability or a later image continuation will abort
        // when it appends the first media placeholder.
        slot.prompt.tokens.has_mtmd = mctx != nullptr;
        slot.prompt.tokens.keep_first(restored);
        slot.prompt.checkpoints.clear();
        SLT_TRC(slot, "restored %zu prompt tokens from range-state pages\n", restored);
    }

    bool launch_slot_with_task(server_slot & slot, server_task && task) {
        if (!task.params.response_scope.empty() && (!prompt_page_cache ||
                !prompt_page_cache->lease_prepared(task.params.response_scope, task.params.response_revision,
                    task.params.response_generation))) {
            response_registry.finish(task.id); // no slot/GPU work was admitted
            send_error(task, "primary context lease was released or expired", ERROR_TYPE_SERVER);
            return false;
        }

        // Adoption and later capture must share one native request identity.
        task.params.cache_control.request_id = task.id;
        // process per-request lora adapters
        if (!task.params.lora.empty()) {
            auto task_loras = construct_lora_list(task.params.lora);
            if (!are_lora_equal(task_loras, slot.lora)) {
                // if lora has changed, check to see if the cache should be cleared
                if (lora_should_clear_cache(slot.lora, task_loras)) {
                    SLT_TRC(slot, "clearing cache for lora change. %zu loras -> %zu loras\n", slot.lora.size(), task.params.lora.size());
                    slot.prompt.clear();
                } else {
                    SLT_TRC(slot, "keeping cache for alora. %zu target loras\n", task_loras.size());
                }
                slot.lora = task_loras;
            }
        } else {
            slot.lora = params_base.lora_adapters;
        }

        // if using alora, make sure it's only a single one requested and active
        size_t alora_invocation_start = task.tokens.size();
        if (lora_all_alora(slot.lora)) {
            const auto & enabled_ids = lora_get_enabled_ids(slot.lora);
            // TODO: This will error out if a user requests two aloras, but only
            // provides the activation string for one. We could, instead search
            // for all requested alora activation strings and then either keep
            // only the last one, or reject if multiple are found.
            if (enabled_ids.size() != 1) {
                send_error(task, "Cannot run multiple aLoRAs in a single request", ERROR_TYPE_INVALID_REQUEST);
                return false;
            }
            const auto & lora = slot.lora[enabled_ids[0]].ptr;

            // get the pointer and count for the invocation tokens
            const uint64_t      n_invocation_tokens = llama_adapter_get_alora_n_invocation_tokens(lora);
            const llama_token * invocation_tokens   = llama_adapter_get_alora_invocation_tokens  (lora);

            // scan backwards through the prompt tokens to find the last
            // occurrence of the invocation sequence
            int match_idx = static_cast<int>(n_invocation_tokens) - 1;
            for (int i = task.tokens.size() - 1; i >= 0; --i) {
                // the token in this position matches the next token to find in
                // the invocation sequence
                if (task.tokens[i] == invocation_tokens[match_idx]) {
                    // if it's a full match, we've found the start
                    if (match_idx == 0) {
                        alora_invocation_start = i;
                        break;
                    }
                    // otherwise, check the next token in the sequence
                    --match_idx;
                } else {
                    // no match in this position, so start looking over again
                    match_idx = static_cast<int>(n_invocation_tokens) - 1;
                }
            }

            // if the activation string is not found, disable the alora
            if (alora_invocation_start == task.tokens.size()) {
                SLT_DBG(slot, "alora %zu requested, but not found. deactivating\n", enabled_ids[0]);
                slot.lora[enabled_ids[0]].scale = 0.0f;
            } else {
                SLT_DBG(slot, "alora %zu activated starting at %zu\n", enabled_ids[0], alora_invocation_start);
                slot.alora_invocation_start = alora_invocation_start;
            }
        }

        if (!task.tokens.validate(ctx_tgt)) {
            send_error(task, "Prompt contains invalid tokens", ERROR_TYPE_INVALID_REQUEST);
            return false;
        }

        // MTP cannot mirror image-embedding batches into its draft context. A
        // media task therefore leaves the draft at the text prefix before the
        // first image. When this slot returns to text-only work, replay from
        // position zero so both draft KV and MTP's pending hidden-state carry
        // are reconstructed; a target/draft page restore alone does not restore
        // that carry.
        const bool replay_mtp_after_media = spec && slot.spec == nullptr && !task.tokens.has_media();
        slot.prompt_page_boundaries = prompt_page_cache_eligible(slot, task)
            ? task.tokens.page_boundaries(prompt_page_cache->page_size_tokens())
            : std::vector<server_tokens::page_boundary>{};
        if (prompt_page_cache_eligible(slot, task)) {
            // The final prompt token is replayed to produce logits on a
            // same-prompt cache hit. Split the decode at n-1 so the exact
            // predecessor state is also captured, even if no coarse page
            // boundary falls near the endpoint.
            const size_t penultimate = task.tokens.size() > 1 ? task.tokens.size() - 1 : 0;
            if (penultimate > 0 && task.tokens.complete_boundary(penultimate)) {
                const server_tokens::page_boundary boundary{
                    penultimate, task.tokens.pos_next(penultimate),
                };
                const auto at = std::lower_bound(
                    slot.prompt_page_boundaries.begin(), slot.prompt_page_boundaries.end(), penultimate,
                    [](const auto & item, size_t count) { return item.token_count < count; });
                if (at == slot.prompt_page_boundaries.end() || at->token_count != penultimate) {
                    slot.prompt_page_boundaries.insert(at, boundary);
                }
            }
        }

        if (replay_mtp_after_media) {
            slot.prompt_clear();
        }
        slot.spec = task.tokens.has_media() ? nullptr : spec.get();

        if (task.is_child()) {
            // The parent's exact endpoint is copied after its prompt completes.
            // Restoring a duplicate child checkpoint now wastes physical cells.
            slot.prompt_clear();
        } else if (!replay_mtp_after_media) {
            restore_prompt_pages(slot, task);
            if (slot.cache_restore_status == "capacity_wait") {
                queue_tasks.defer(std::move(task));
                return true;
            }
            share_restored_prefix(slot, task);
        }
        if (!task.is_child() && !task.is_parent() && !unified_glm_capacity(slot, task)) {
            SLT_INF(slot, "deferring task %d until unified physical context budget is available\n", task.id);
            slot.prompt_clear();
            queue_tasks.defer(std::move(task));
            return true; // handled by the deferred queue; no slot was started
        }

        SLT_DBG(slot, "launching slot : %s\n", safe_json_to_str(slot.to_json()).c_str());

        if (task.type == SERVER_TASK_TYPE_TTS) {
            GGML_ASSERT(has_cap_tts()); // should already checked in route handler
            if (!slot.tts.is_supported()) {
                slot.tts.ctx.init(ctx_tgt, slot.mctx);
            }
            // mtmd_helper always re-eval the whole prompt
            slot.prompt_clear();
            task.tts_inp.data.seq_id = slot.id;
            if (slot.tts.ctx.set_input(task.tts_inp.get()) != 0) {
                send_error(task, "failed to process TTS prompt", ERROR_TYPE_SERVER);
                return false;
            }
        }

        // initialize samplers
        if (task.need_sampling()) {
            try {
                slot.smpl.reset(common_sampler_init(model_tgt, task.params.sampling));
            } catch (std::exception & e) {
                std::string err_msg = std::string("Failed to initialize samplers: ") + e.what();
                send_error(task, err_msg, ERROR_TYPE_INVALID_REQUEST);
                return false;
            }

            const bool need_pre_sample_logits = task.params.sampling.n_probs > 0 && !task.params.post_sampling_probs;

            bool use_backend_sampling = task.params.sampling.backend_sampling && !task.params.prefill_only;

            // TODO: getting pre sampling logits is not yet supported with backend sampling
            use_backend_sampling &= !need_pre_sample_logits;

            // TODO: check verify if this actually works with TTS
            use_backend_sampling &= task.type != SERVER_TASK_TYPE_TTS;

            // TODO: tmp until backend sampling is fully implemented
            if (use_backend_sampling) {
                llama_set_sampler(ctx_tgt, slot.id, common_sampler_get(slot.smpl.get()));
            } else {
                llama_set_sampler(ctx_tgt, slot.id, nullptr);
            }

            SLT_TRC(slot, "sampler chain: %s\n", common_sampler_print(slot.smpl.get()).c_str());
            SLT_TRC(slot, "sampler params: \n%s\n", task.params.sampling.print().c_str());

            if (spec && !common_speculative_get_synth_probs(spec.get()).empty()) {
                const uint32_t seed = task.params.sampling.seed == LLAMA_DEFAULT_SEED
                    ? std::random_device{}()
                    : task.params.sampling.seed;
                slot.spec_synth_rng.seed(seed);
            }
        } else {
            slot.smpl.reset();
        }

        // the per-request limit takes priority over the global one
        slot.n_predict_max = task.params.n_predict != -1 ? task.params.n_predict : params_base.n_predict;

        slot.task = std::make_unique<const server_task>(std::move(task));

        if (slot.task->type == SERVER_TASK_TYPE_TTS) {
            slot.state = SLOT_STATE_PROCESSING_PROMPT;
        } else {
            slot.state = slot.task->is_child()
                ? SLOT_STATE_WAIT_OTHER // wait for the parent to process prompt
                : SLOT_STATE_STARTED;
        }

        // no prompt, the rows come from the audio stream, which audio controls reach from now on
        if (slot.task->type == SERVER_TASK_TYPE_TRANSCRIBE_RT) {
            slot.audio_stream.reset(mtmd_audio_stream_init(mctx, mtmd_audio_stream_params_default()));
            if (!slot.audio_stream) {
                send_error(slot, "failed to initialize the audio stream", ERROR_TYPE_SERVER);
                slot.release();
                return false;
            }
            slot.prompt_clear();
            slot.init_sampler();
            slot.sampled = LLAMA_TOKEN_NULL;
            slot.i_batch = -1;
            slot.stats.update_prompt_start();
            slot.state = SLOT_STATE_GENERATING;
            send_partial_response(slot, {}, false, true);
        }

        // reset server kill-switch counter
        n_empty_consecutive = 0;

        SLT_INF(slot, "processing task, is_child = %d\n", slot.task->is_child());
        return true;
    }

    bool process_token(completion_token_output & result, server_slot & slot) {
        // remember which tokens were sampled - used for repetition penalties during sampling
        const std::string token_str = result.text_to_send;
        slot.sampled = result.tok;

        slot.generated.append(token_str, result.tok);
        if (slot.task->params.return_tokens) {
            slot.generated_tokens.push_back(result.tok);
        }
        slot.has_next_token = true;

        // check if there is incomplete UTF-8 character at the end
        bool incomplete = validate_utf8(slot.generated.text) < slot.generated.text.size();

        // search stop word and delete it
        if (!incomplete) {
            size_t pos = std::min(slot.n_sent_text, slot.generated.text.size());

            const std::string str_test = slot.generated.text.substr(pos);
            bool send_text = true;

            size_t stop_pos = slot.find_stopping_strings(str_test, token_str.size(), true);
            if (stop_pos != std::string::npos) {
                slot.generated.truncate(pos + stop_pos);
                pos = std::min(slot.n_sent_text, slot.generated.text.size());
            } else if (slot.has_next_token && !llama_vocab_is_eog(vocab, result.tok) ) {
                stop_pos = slot.find_stopping_strings(str_test, token_str.size(), false);
                send_text = stop_pos == std::string::npos;
            }

            // check if there is any token to predict
            if (send_text) {
                // no send the stop word in the response
                result.text_to_send = slot.generated.text.substr(pos, std::string::npos);
                slot.n_sent_text += result.text_to_send.size();
                // add the token to slot queue and cache
            } else {
                result.text_to_send = "";
            }

            slot.add_token(result);
            if (slot.task->params.stream) {
                send_partial_response(slot, result, false);
            }
        }

        if (incomplete) {
            slot.has_next_token = true;
            if (slot.task->params.stream && slot.task->params.res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
                // Token provenance must not wait for a complete UTF-8 character.
                auto token_only = result;
                token_only.text_to_send.clear();
                send_partial_response(slot, token_only, false);
            }
        }

        // if context shifting is disabled, make sure that we don't run out of context
        if (!params_base.ctx_shift && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
            slot.truncated      = true;
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped due to running out of context capacity, prompt.n_tokens() = %d, task.n_tokens = %d, n_gen = %d, n_ctx = %d\n",
                    slot.prompt.n_tokens(), slot.task->n_tokens(), (int) slot.stats.n_gen, slot.n_ctx);
        }

        // check the limits
        if (slot.stats.n_gen > 0 && slot.has_next_token && !slot.has_budget()) {
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped by limit, n_gen = %d, n_predict = %d\n", (int) slot.stats.n_gen, slot.task->params.n_predict);
        }

        if (slot.has_new_line) {
            // require that each new line has a whitespace prefix (i.e. indentation) of at least slot.params.n_indent
            if (slot.task->params.n_indent > 0) {
                // check the current indentation
                // TODO: improve by not doing it more than once for each new line
                if (slot.last_nl_pos > 0) {
                    size_t pos = slot.last_nl_pos;

                    int n_indent = 0;
                    while (pos < slot.generated.text.size() && (slot.generated.text[pos] == ' ' || slot.generated.text[pos] == '\t')) {
                        n_indent++;
                        pos++;
                    }

                    if (pos < slot.generated.text.size() && n_indent < slot.task->params.n_indent) {
                        slot.stop           = STOP_TYPE_LIMIT;
                        slot.has_next_token = false;

                        // cut the last line
                        slot.generated.truncate(pos);

                        SLT_DBG(slot, "stopped by indentation limit, n_gen = %d, n_indent = %d\n", (int) slot.stats.n_gen, n_indent);
                    }
                }

                // find the next new line
                {
                    const size_t pos = slot.generated.text.find('\n', slot.last_nl_pos);

                    if (pos != std::string::npos) {
                        slot.last_nl_pos = pos + 1;
                    }
                }
            }
        }

        // check if there is a new line in the generated text
        if (result.text_to_send.find('\n') != std::string::npos) {
            slot.has_new_line = true;

            // if we have seen a new line, we stop after a certain time limit, but only upon another new line
            if (slot.task->params.t_max_predict_ms > 0 && slot.stats.t_gen_ms() > slot.task->params.t_max_predict_ms) {
                slot.stop           = STOP_TYPE_LIMIT;
                slot.has_next_token = false;

                SLT_DBG(slot, "stopped by time limit, n_gen = %d, t_max_predict_ms = %d ms\n", (int) slot.stats.n_gen, (int) slot.task->params.t_max_predict_ms);
            }
        }

        if (llama_vocab_is_eog(vocab, result.tok)) {
            slot.stop           = STOP_TYPE_EOS;
            slot.has_next_token = false;

            SLT_DBG(slot, "%s", "stopped by EOS\n");
        }

        SLT_DBG(slot, "n_gen = %d, n_remaining = %d, next token: %5d '%s'\n", (int) slot.stats.n_gen, slot.n_remaining(), result.tok, token_str.c_str());

        return slot.has_next_token; // continue
    }

    void populate_token_probs(const server_slot & slot, completion_token_output & result, bool post_sampling, bool special, int idx) const {
        const size_t n_probs_request = slot.task->params.sampling.n_probs;

        if (post_sampling) {
            const auto * cur_p = common_sampler_get_candidates(slot.smpl.get(), true);
            const size_t max_probs = cur_p->size;
            const size_t n_probs = std::min(max_probs, n_probs_request);

            // set probability for sampled token
            for (size_t i = 0; i < max_probs; i++) {
                if (cur_p->data[i].id == result.tok) {
                    result.prob = cur_p->data[i].p;
                    break;
                }
            }

            // set probability for top n_probs tokens
            result.probs.reserve(n_probs);
            for (size_t i = 0; i < n_probs; i++) {
                // Some samplers do return 0.0 probabilities, others don't.
                // Filter 0.0 probailities, to ensure the behavior is consistent.
                if (cur_p->data[i].p == 0.0) {
                    break;
                }

                result.probs.push_back({
                    cur_p->data[i].id,
                    common_token_to_piece(ctx_tgt, cur_p->data[i].id, special),
                    cur_p->data[i].p
                });
            }
        } else {
            std::vector<llama_token_data> cur = get_token_probabilities(slot.ctx_tgt, idx, n_probs_request);
            const size_t max_probs = cur.size();
            const size_t n_probs = std::min(max_probs, n_probs_request);

            // set probability for sampled token
            for (size_t i = 0; i < max_probs; i++) {
                // set probability for sampled token
                if (cur[i].id == result.tok) {
                    result.prob = cur[i].p;
                    break;
                }
            }

            // set probability for top n_probs tokens
            result.probs.reserve(n_probs);
            for (size_t i = 0; i < n_probs; i++) {
                result.probs.push_back({
                    cur[i].id,
                    common_token_to_piece(ctx_tgt, cur[i].id, special),
                    cur[i].p
                });
            }
        }
    }

    void send_error(const server_task & task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(task.id, error, type);
    }

    void send_error(const server_slot & slot, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(slot.task->id, error, type, slot.task->n_tokens(), slot.n_ctx);
    }

    void send_error(const int id_task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER, const int32_t n_prompt_tokens = 0, const int32_t n_ctx = 0) {
        SRV_ERR("task id = %d, error: %s\n", id_task, error.c_str());

        if (type == ERROR_TYPE_EXCEED_CONTEXT_SIZE) {
            GGML_ASSERT(n_ctx > 0 && n_prompt_tokens > 0);
        }

        auto res = std::make_unique<server_task_result_error>();
        res->id              = id_task;
        res->err_type        = type;
        res->err_msg         = error;
        res->n_prompt_tokens = n_prompt_tokens;
        res->n_ctx           = n_ctx;

        queue_results.send(std::move(res));
    }

    void send_partial_response(server_slot & slot, const completion_token_output & tkn, bool is_progress, bool is_begin = false) {
        auto res = std::make_unique<server_task_result_cmpl_partial>();

        res->id    = slot.task->id;
        res->index = slot.task->index;

        if (is_progress) {
            res->is_progress        = true;
            res->progress.waiting_for_prefill_workspace = slot.waiting_for_prefill_workspace;
            res->progress.total     = slot.task->n_tokens();
            res->progress.cache     = slot.stats.n_prompt_cached;
            res->progress.processed = slot.prompt.tokens.size();
            res->progress.time_ms   = slot.stats.t_elapsed_us() / 1000;
        }
        if (is_begin) {
            res->is_begin = true;
        } else {
            res->content = slot.generated.substr(slot.n_sent_text - tkn.text_to_send.size(), tkn.text_to_send.size());
            res->tokens.assign(1, tkn.tok);
        }

        res->n_decoded             = slot.stats.n_gen;
        res->n_prompt_tokens       = slot.task->n_tokens();
        res->n_prompt_tokens_cache = slot.stats.n_prompt_cached;
        res->post_sampling_probs   = slot.task->params.post_sampling_probs;

        res->verbose           = slot.task->params.verbose;
        res->res_type          = slot.task->params.res_type;
        res->oaicompat_model   = slot.task->params.oaicompat_model;
        res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;

        // populate res.probs_output
        if (slot.task->params.sampling.n_probs > 0) {
            res->prob_output = tkn; // copy the token probs
        }

        // populate timings if this is final response or timings_per_token is enabled
        if (slot.stop != STOP_TYPE_NONE || slot.task->params.timings_per_token) {
            res->stats = slot.stats;
        }

        queue_results.send(std::move(res));
    }

    void send_tts_result(server_slot & slot, int32_t sample_rate, const char * data, size_t data_len, bool final) {
        auto res = std::make_unique<server_task_result_tts>();

        res->id          = slot.task->id;
        res->index       = slot.task->index;
        res->sample_rate = sample_rate;
        res->audio.assign(data, data_len);
        res->final       = final;

        queue_results.send(std::move(res));
    }

    void send_final_response(server_slot & slot) {
        if (moe_trace) { moe_trace->finish("request_complete"); }
        auto res = std::make_unique<server_task_result_cmpl_final>();

        res->id      = slot.task->id;
        res->id_slot = slot.id;

        res->index = slot.task->index;

        // keep copy of last generated text for debugging purposes
        if (slots_debug) {
            slot.debug_generated_text = slot.generated.text;
        }

        // in stream mode, content and tokens are already in last partial chunk
        if (slot.task->params.stream) {
            res->content     = {};
            res->tokens      = llama_tokens{};
        } else {
            res->content     = std::move(slot.generated);
            res->tokens      = std::move(slot.generated_tokens);
        }
        res->stats           = slot.stats;
        res->prompt          = slot.task->tokens.detokenize(ctx_tgt, true);
        res->response_fields = std::move(slot.task->params.response_fields);

        res->truncated             = slot.truncated;
        res->n_decoded             = slot.stats.n_gen;
        res->n_prompt_tokens       = slot.task->n_tokens();
        res->n_prompt_tokens_cache = slot.stats.n_prompt_cached;
        res->n_tokens_cached       = slot.prompt.n_tokens();
        res->has_new_line          = slot.has_new_line;
        res->stopping_word         = slot.stopping_word;
        res->stop                  = slot.stop;
        res->post_sampling_probs   = slot.task->params.post_sampling_probs;

        res->verbose           = slot.task->params.verbose;
        res->stream            = slot.task->params.stream;
        res->include_usage     = slot.task->params.include_usage;
        res->res_type          = slot.task->params.res_type;
        res->oaicompat_model   = slot.task->params.oaicompat_model;
        res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;

        // populate res.probs_output
        if (slot.task->params.sampling.n_probs > 0) {
            if (!slot.task->params.stream && slot.stop == STOP_TYPE_WORD) {
                const llama_tokens stop_word_toks = common_tokenize(ctx_tgt, slot.stopping_word, false);

                size_t safe_offset = std::min(slot.generated_token_probs.size(), stop_word_toks.size());
                res->probs_output = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin(),
                        slot.generated_token_probs.end() - safe_offset);
            } else {
                res->probs_output = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin(),
                        slot.generated_token_probs.end());
            }
        }

        res->generation_params = slot.task->params; // copy the parameters

        queue_results.send(std::move(res));
    }

    void send_embedding(const server_slot & slot, const common_batch & batch) {
        auto res = std::make_unique<server_task_result_embd>();
        res->id        = slot.task->id;
        res->index     = slot.task->index;
        res->n_tokens  = slot.task->n_tokens();
        res->res_type  = slot.task->params.res_type;

        const int n_embd_out = llama_model_n_embd_out(model_tgt);

        std::vector<float> embd_res(n_embd_out, 0.0f);

        for (int i = 0; i < batch.size(); ++i) {
            if (!batch.tokens[i].output || batch.tokens[i].seq_id != slot.id) {
                continue;
            }

            const float * embd = nullptr;
            if (llama_pooling_type(slot.ctx_tgt) == LLAMA_POOLING_TYPE_NONE) {
                embd = llama_get_embeddings_ith(slot.ctx_tgt, i);
            } else {
                embd = llama_get_embeddings_seq(slot.ctx_tgt, batch.tokens[i].seq_id);
            }

            if (embd == nullptr) {
                SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.tokens[i].id, batch.tokens[i].seq_id);

                res->embedding.push_back(std::vector<float>(n_embd_out, 0.0f));
                continue;
            }

            // normalize only when there is pooling
            if (llama_pooling_type(slot.ctx_tgt) != LLAMA_POOLING_TYPE_NONE) {
                common_embd_normalize(embd, embd_res.data(), n_embd_out, slot.task->params.embd_normalize);
                res->embedding.push_back(embd_res);
                break;
            }

            res->embedding.emplace_back(embd, embd + n_embd_out);
        }

        SLT_DBG(slot, "%s", "sending embeddings\n");

        queue_results.send(std::move(res));
    }

    void send_hidden_states(const server_slot & slot) {
        ctx_tgt->synchronize();

        auto res = std::make_unique<server_task_result_hidden_state>();
        res->id              = slot.task->id;
        res->index           = slot.task->index;
        res->n_tokens        = slot.task->n_tokens();
        res->n_embd          = llama_model_n_embd(model_tgt);
        res->pooling         = slot.task->hidden_state_pooling;
        res->encoding_format = slot.task->hidden_state_encoding_format;
        res->include_tokens  = slot.task->hidden_state_include_tokens;
        res->tokens          = slot.task->tokens.get_text_tokens();

        const auto & hidden_by_layer = ctx_tgt->get_hidden_states();
        const int32_t n_embd = res->n_embd;
        const auto & selected_indices = slot.task->hidden_state_token_indices;

        if (slot.task->hidden_state_ltx25) {
            constexpr int32_t sequence_length = 1024;
            constexpr int32_t state_count = 49;
            const int32_t token_count = slot.task->n_tokens();
            const size_t state_values = (size_t) token_count*n_embd;
            const size_t payload_size = (size_t) state_count*state_values*sizeof(uint16_t);
            const llama_token pad_token = llama_vocab_pad(vocab);
            if (pad_token == LLAMA_TOKEN_NULL) {
                send_error(slot, "LTX-2.5 tokenizer has no padding token", ERROR_TYPE_SERVER);
                return;
            }

            json header {
                {"schema", "serverless-host.ltx25.hidden-states-wire.v1"},
                {"byte_order", "little"},
                {"dtype", "BF16"},
                {"sequence_length", sequence_length},
                {"token_count", token_count},
                {"position_offset", slot.task->hidden_state_position_offset},
                {"hidden_size", n_embd},
                {"state_count", state_count},
                {"payload_bytes", payload_size},
                {"pad_token_id", pad_token},
                {"input_ids", res->tokens},
            };
            const std::string header_data = safe_json_to_str(header);
            const uint64_t header_size = header_data.size();

            res->binary_data.resize(sizeof(header_size) + header_size + payload_size);
            for (size_t i = 0; i < sizeof(header_size); ++i) {
                res->binary_data[i] = (char) ((header_size >> (8*i)) & 0xff);
            }
            std::memcpy(res->binary_data.data() + sizeof(header_size), header_data.data(), header_size);

            char * dst = res->binary_data.data() + sizeof(header_size) + header_size;
            for (int32_t state = 0; state < state_count; ++state) {
                const int32_t key = state == 0
                    ? LLAMA_HIDDEN_STATE_KEY_SCALED_INPUT
                    : state == state_count - 1
                        ? LLAMA_HIDDEN_STATE_KEY_FINAL_NORM
                        : state - 1;
                auto it = hidden_by_layer.find(key);
                if (it == hidden_by_layer.end() || it->second.size() != state_values) {
                    SLT_ERR(slot, "missing or malformed LTX-2.5 hidden state %d (capture key %d)\n", state, key);
                    send_error(slot, "missing or malformed LTX-2.5 hidden state", ERROR_TYPE_SERVER);
                    return;
                }
                for (float value : it->second) {
                    const uint16_t bits = ggml_fp32_to_bf16(value).bits;
                    *dst++ = (char) (bits & 0xff);
                    *dst++ = (char) (bits >> 8);
                }
            }

            SLT_DBG(slot, "%s", "sending LTX-2.5 hidden states\n");
            queue_results.send(std::move(res));
            return;
        }

        auto vector_to_json = [](const float * data, int32_t n) {
            json values = json::array();
            for (int32_t i = 0; i < n; ++i) {
                values.push_back(data[i]);
            }
            return values;
        };

        auto vector_to_base64 = [](const float * data, int32_t n) {
            return base64::encode(reinterpret_cast<const char *>(data), (size_t) n*sizeof(float));
        };

        json hidden_states = json::array();
        for (int32_t il : slot.task->hidden_state_layers) {
            res->layers.push_back(il);
            auto it = hidden_by_layer.find(il);
            if (it == hidden_by_layer.end()) {
                SLT_ERR(slot, "missing captured hidden state for layer %d\n", il);
                continue;
            }

            const auto & values = it->second;
            json layer_obj {
                {"layer", il},
                {"dtype", "float32"},
            };

            if (slot.task->hidden_state_pooling == "none") {
                json rows = json::array();
                std::vector<float> packed;
                if (slot.task->hidden_state_encoding_format == "base64") {
                    packed.reserve((size_t) selected_indices.size()*n_embd);
                }
                for (int32_t idx : selected_indices) {
                    const float * row = values.data() + (size_t) idx*n_embd;
                    if (slot.task->hidden_state_encoding_format == "base64") {
                        packed.insert(packed.end(), row, row + n_embd);
                    } else {
                        rows.push_back(vector_to_json(row, n_embd));
                    }
                }
                layer_obj["shape"] = json::array({ (int32_t) selected_indices.size(), n_embd });
                if (slot.task->hidden_state_encoding_format == "base64") {
                    layer_obj["data"] = vector_to_base64(packed.data(), (int32_t) packed.size());
                } else {
                    layer_obj["data"] = rows;
                }
            } else {
                std::vector<float> pooled(n_embd, 0.0f);
                if (slot.task->hidden_state_pooling == "mean") {
                    for (int32_t idx : selected_indices) {
                        const float * row = values.data() + (size_t) idx*n_embd;
                        for (int32_t j = 0; j < n_embd; ++j) {
                            pooled[j] += row[j];
                        }
                    }
                    const float denom = (float) selected_indices.size();
                    for (float & value : pooled) {
                        value /= denom;
                    }
                } else {
                    const int32_t idx = slot.task->hidden_state_pooling == "cls"
                        ? selected_indices.front()
                        : selected_indices.back();
                    const float * row = values.data() + (size_t) idx*n_embd;
                    std::copy(row, row + n_embd, pooled.begin());
                }

                layer_obj["shape"] = { n_embd };
                if (slot.task->hidden_state_encoding_format == "base64") {
                    layer_obj["data"] = vector_to_base64(pooled.data(), n_embd);
                } else {
                    layer_obj["data"] = vector_to_json(pooled.data(), n_embd);
                }
            }

            hidden_states.push_back(layer_obj);
        }

        res->hidden_states = hidden_states;
        SLT_DBG(slot, "%s", "sending hidden states\n");
        queue_results.send(std::move(res));
    }

    void send_decision(const server_slot & slot, const common_batch & batch, int32_t i_batch) {
        auto res = std::make_unique<server_task_result_decision>();
        res->id       = slot.task->id;
        res->index    = slot.task->index;
        res->n_tokens = slot.task->n_tokens();

        const auto & decision = slot.task->decision;

        if (!decision.labels.empty()) {
            const float * logits = llama_get_logits_ith(slot.ctx_tgt, i_batch);
            if (logits == nullptr) {
                send_error(slot, "failed to get logits", ERROR_TYPE_SERVER);
                return;
            }
            const int32_t n_vocab = llama_vocab_n_tokens(vocab);
            for (const llama_token label : decision.labels) {
                GGML_ASSERT(label >= 0 && label < n_vocab);
                res->scores.push_back(logits[label]);
            }
            if (!decision.label_groups.empty()) {
                std::vector<float> scores;
                size_t i = 0;
                for (const int32_t n : decision.label_groups) {
                    GGML_ASSERT(n > 0 && i + n <= res->scores.size());
                    scores.push_back(*std::max_element(res->scores.begin() + i, res->scores.begin() + i + n));
                    i += n;
                }
                res->scores = std::move(scores);
            }
        } else {
            // the outputs of this slot in this batch are the last tokens of the prompt
            std::vector<int32_t> idx;
            for (int i = 0; i < batch.size(); ++i) {
                if (batch.tokens[i].output && batch.tokens[i].seq_id == slot.id) {
                    idx.push_back(i);
                }
            }
            const int32_t pos_first = slot.prompt.n_tokens() - (int32_t) idx.size();
            auto get_embd = [&](int32_t pos) -> const float * {
                const int32_t i = pos - pos_first;
                return i >= 0 && i < (int32_t) idx.size() ? llama_get_embeddings_ith(slot.ctx_tgt, idx[i]) : nullptr;
            };

            // joint head (decision model): the scores are the first rows
            for (int32_t i = 0; i < decision.n_scores; i++) {
                const float * embd = i < (int32_t) idx.size() ? llama_get_embeddings_ith(slot.ctx_tgt, idx[i]) : nullptr;
                if (embd == nullptr) {
                    send_error(slot, "failed to get embeddings", ERROR_TYPE_SERVER);
                    return;
                }
                res->scores.push_back(embd[0]);
            }

            const int32_t n_embd_out = llama_model_n_embd_out(model_tgt);
            const int32_t n_pointer  = n_embd_out / 2;
            const float * embd_q = decision.pointer >= 0 ? get_embd(decision.pointer) : nullptr;
            GGML_ASSERT(decision.column >= 0 && decision.column < n_embd_out);

            for (const int32_t marker : decision.markers) {
                const float * embd = get_embd(marker);
                if (embd == nullptr || (decision.pointer >= 0 && embd_q == nullptr)) {
                    send_error(slot, "failed to get embeddings, the question and its options must fit in one batch", ERROR_TYPE_SERVER);
                    return;
                }
                if (decision.pointer < 0) {
                    res->scores.push_back(embd[decision.column]);
                    continue;
                }
                float dot = 0.0f;
                for (int32_t i = 0; i < n_pointer; i++) {
                    dot += embd_q[i] * embd[n_pointer + i];
                }
                res->scores.push_back(dot / sqrtf((float) n_pointer));
            }
        }

        SLT_DBG(slot, "%s", "sending decision result\n");

        queue_results.send(std::move(res));
    }

    void send_rerank(const server_slot & slot, const common_batch & batch) {
        auto res = std::make_unique<server_task_result_rerank>();
        res->id       = slot.task->id;
        res->index    = slot.task->index;
        res->n_tokens = slot.task->n_tokens();

        for (int i = 0; i < batch.size(); ++i) {
            if (!batch.tokens[i].output || batch.tokens[i].seq_id != slot.id) {
                continue;
            }

            const float * embd = llama_get_embeddings_seq(ctx_tgt, batch.tokens[i].seq_id);
            if (embd == NULL) {
                embd = llama_get_embeddings_ith(ctx_tgt, i);
            }

            if (embd == NULL) {
                SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.tokens[i].id, batch.tokens[i].seq_id);

                res->score = -1e6;
                continue;
            }

            res->score = embd[0];
        }

        SLT_DBG(slot, "sending rerank result, res.score = %f\n", res->score);

        queue_results.send(std::move(res));
    }

    //
    // Functions to process the task
    //

    // tokenize the input if it's set by CLI, return false on error
    bool tokenize_cli_input(server_task & task) {
        try {
            auto & prompt = task.cli_prompt;
            if (mctx != nullptr) {
                task.tokens = process_mtmd_prompt(mctx, prompt, task.cli_files, init_opt);
            } else {
                task.tokens = std::move(tokenize_input_prompts(vocab, mctx, prompt, true, true, init_opt)[0]);
            }
            task.cli_prompt.clear();
            task.cli_files.clear();
        } catch (const std::exception & e) {
            send_error(task, std::string("Failed to format input: ") + e.what(), ERROR_TYPE_INVALID_REQUEST);
            return false;
        }
        return true;
    }

    // idle slots that lose the least by being reused: empty slots first, then the least recently used ones
    std::vector<server_slot *> get_free_slots(size_t n_slots_needed, int exclude_id_slot) {
        std::vector<server_slot *> free_slots;
        for (auto & slot : slots) {
            if (!slot.is_processing() && slot.id != exclude_id_slot) {
                free_slots.push_back(&slot);
            }
        }

        std::stable_sort(free_slots.begin(), free_slots.end(), [](const server_slot * a, const server_slot * b) {
            if ((a->prompt.n_tokens() == 0) != (b->prompt.n_tokens() == 0)) {
                return a->prompt.n_tokens() == 0;
            }
            return a->t_last_used < b->t_last_used;
        });

        free_slots.resize(std::min(free_slots.size(), n_slots_needed));

        return free_slots;
    }

    // launch multiple slots for parent + child tasks
    bool launch_slots_with_parent_task(server_slot & parent_slot, std::vector<server_slot *> & child_slots, server_task && parent_task) {
        GGML_ASSERT(!parent_slot.is_processing());
        GGML_ASSERT(parent_task.is_parent());
        GGML_ASSERT(child_slots.size() == parent_task.child_tasks.size());

        if (params_base.kv_unified && model_tgt->arch == LLM_ARCH_GLM5NEXT) {
            const uint64_t prompt = std::max<int64_t>(parent_task.n_tokens(), parent_task.tokens.pos_next());
            const auto tail_budget = [&](const server_task & request) -> uint64_t {
                const int predict = request.params.n_predict != -1 ? request.params.n_predict : params_base.n_predict;
                const uint64_t available = prompt < (uint64_t) parent_slot.n_ctx ? parent_slot.n_ctx - prompt : 0;
                return request.params.prefill_only ? 0 : (predict < 0 ? available :
                        std::min<uint64_t>(available, predict));
            };
            uint64_t need = prompt + tail_budget(parent_task);
            for (const auto & child : parent_task.child_tasks) { need += tail_budget(child); }
            if (need > llama_n_ctx(ctx_tgt)) {
                response_registry.finish(parent_task.id);
                send_error(parent_task, "parallel completion budgets exceed unified context capacity", ERROR_TYPE_INVALID_REQUEST);
                return false;
            }
            std::vector<std::pair<llama_seq_id, uint32_t>> budgets;
            for (const auto & active : slots) {
                if (active.is_processing() && active.task) {
                    // Full per-slot cap is a conservative bound for unrelated work.
                    budgets.emplace_back(active.id, active.n_ctx);
                }
            }
            if (!budgets.empty() && need + llama_glm5next_reserved_cells(ctx_tgt, budgets) > llama_n_ctx(ctx_tgt)) {
                queue_tasks.defer(std::move(parent_task));
                return true;
            }
        }

        int id_parent = parent_task.id;

        SRV_TRC("launching slots for parent task id_task = %d with %zu child tasks\n", id_parent, parent_task.child_tasks.size());

        // to be called in case of failure to release all launched slots
        auto release_slots = [this, id_parent]() {
            for (auto & slot : slots) {
                if (slot.is_processing() && (
                        slot.task->id == id_parent ||
                        slot.task->id_parent == id_parent
                )) {
                    slot.release();
                }
            }
        };

        // launch all child tasks first
        size_t idx = 0;
        GGML_ASSERT(child_slots.size() == parent_task.child_tasks.size());
        for (auto * slot : child_slots) {
            // keep the state that the child task is about to overwrite
            if (prompt_cache && slot->prompt_save(*prompt_cache)) {
                prompt_cache->update();
            }

            int id_child = parent_task.child_tasks[idx].id;
            if (!launch_slot_with_task(*slot, std::move(parent_task.child_tasks[idx]))) {
                SRV_ERR("failed to launch slot with child task, id_task = %d\n", id_child);
                release_slots();
                return false;
            }
            idx++;
        }

        // finally, launch the parent task
        if (!launch_slot_with_task(parent_slot, std::move(parent_task))) {
            SRV_ERR("failed to launch slot with task, id_task = %d\n", id_parent);
            release_slots();
            return false;
        }

        return true;
    }

    // n_tokens_cur: the number of tokens added to the batch for the current slot
    void create_checkpoint(server_slot & slot, const int64_t n_tokens_cur, llama_pos pos_min, llama_pos pos_max) {
        const int id_task = slot.task->id;

        // evict checkpoints within min-step of a previous checkpoint, unless they were
        // created by the current task
        // only when the list is full, otherwise short prompts keep just the oldest checkpoint
        int64_t last = -1;
        for (auto it = slot.prompt.checkpoints.begin();
                slot.prompt.checkpoints.size() + 1 >= (size_t) params_base.n_ctx_checkpoints &&
                it != slot.prompt.checkpoints.end(); ) {
            if (it->id_task != id_task && last >= 0 && it->n_tokens <= last + params_base.checkpoint_min_step) {
                SLT_TRC(slot, "erasing context checkpoint too close to an earlier one (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                        it->pos_min, it->pos_max, it->n_tokens, (float) it->size() / 1024 / 1024);

                it = slot.prompt.checkpoints.erase(it);
                continue;
            }

            last = it->n_tokens;
            ++it;
        }

        while (slot.prompt.checkpoints.size() >= (size_t) params_base.n_ctx_checkpoints) {
            // make room for the new checkpoint, if needed
            const auto & cur = slot.prompt.checkpoints.front();

            SLT_WRN(slot, "erasing old context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                    cur.pos_min, cur.pos_max, cur.n_tokens, (float) cur.size() / 1024 / 1024);

            slot.prompt.checkpoints.erase(slot.prompt.checkpoints.begin());
        }

        // replace an existing checkpoint at the same n_tokens instead of appending a duplicate
        {
            const int64_t n_tokens_new = slot.prompt.n_tokens() - n_tokens_cur;
            for (auto it = slot.prompt.checkpoints.begin(); it != slot.prompt.checkpoints.end(); ) {
                if (it->n_tokens == n_tokens_new) {
                    SLT_TRC(slot, "superseding context checkpoint at n_tokens = %" PRId64 "\n", it->n_tokens);
                    it = slot.prompt.checkpoints.erase(it);
                } else {
                    ++it;
                }
            }
        }

        auto & cur = slot.prompt.checkpoints.emplace_back();

        cur.id_task = id_task;

        // [TAG_CHECKPOINTS_FIX_POS_MIN]
        // TODO: here we incorrectly deterimne that the saved checkpoint data covers the [pos_min, pos_max] range
        //       this is not true for SWA models: https://github.com/ggml-org/llama.cpp/pull/24411#issuecomment-4677983225
        cur.update_pos(slot.prompt.n_tokens() - n_tokens_cur, pos_min, pos_max);

        cur.update_tgt(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        cur.update_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        // stash the draft's speculative state with the checkpoint
        common_speculative_get_state(spec.get(), slot.id, cur.data_spec);

        SLT_TRC(slot,
                "created context checkpoint %d of %d (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                (int) slot.prompt.checkpoints.size(), params_base.n_ctx_checkpoints, cur.pos_min,
                cur.pos_max, cur.n_tokens, (float) cur.size() / 1024 / 1024);
    }

    // checkpoints are appended to the slot save file, after the llama state payload
    // they cannot be recreated from the final state alone (a recurrent state cannot be rewound)
    static constexpr uint32_t SLOT_CKPT_MAGIC   = 0x504b4353; // "SCKP"
    static constexpr uint32_t SLOT_CKPT_VERSION = 1;

    static bool ckpt_read(std::ifstream & ifs, void * dst, size_t size, size_t & n_read) {
        if (!ifs.read((char *) dst, size)) {
            return false;
        }
        n_read += size;
        return true;
    }

    static bool ckpt_read_buf(std::ifstream & ifs, std::vector<uint8_t> & buf, size_t n_avail, size_t & n_read) {
        uint64_t n = 0;
        // check the size against the bytes left in the file before allocating, the size field may be corrupted
        if (!ckpt_read(ifs, &n, sizeof(n), n_read) || n > n_avail - n_read) {
            return false;
        }
        buf.resize(n);
        return n == 0 || ckpt_read(ifs, buf.data(), n, n_read);
    }

    static void ckpt_write(std::ofstream & ofs, const void * src, size_t size, size_t & n_written) {
        ofs.write((const char *) src, size);
        n_written += size;
    }

    static void ckpt_write_buf(std::ofstream & ofs, const std::vector<uint8_t> & buf, size_t & n_written) {
        const uint64_t n = buf.size();
        ckpt_write(ofs, &n, sizeof(n), n_written);
        if (n > 0) {
            ckpt_write(ofs, buf.data(), n, n_written);
        }
    }

    // returns false if the appendix could not be written completely
    bool save_slot_checkpoints(const std::string & filepath, const server_slot & slot, size_t & n_written) const {
        n_written = 0;
        if (slot.prompt.checkpoints.empty()) {
            return true;
        }
        std::ofstream ofs(std::filesystem::u8path(filepath), std::ios::binary | std::ios::app);
        if (!ofs) {
            SLT_WRN(slot, "failed to append context checkpoints to '%s'\n", filepath.c_str());
            return false;
        }
        const uint32_t magic   = SLOT_CKPT_MAGIC;
        const uint32_t version = SLOT_CKPT_VERSION;
        const uint32_t count   = (uint32_t) slot.prompt.checkpoints.size();
        ckpt_write(ofs, &magic,   sizeof(magic),   n_written);
        ckpt_write(ofs, &version, sizeof(version), n_written);
        ckpt_write(ofs, &count,   sizeof(count),   n_written);
        for (const auto & cur : slot.prompt.checkpoints) {
            ckpt_write(ofs, &cur.n_tokens, sizeof(cur.n_tokens), n_written);
            ckpt_write(ofs, &cur.pos_min,  sizeof(cur.pos_min),  n_written);
            ckpt_write(ofs, &cur.pos_max,  sizeof(cur.pos_max),  n_written);
            ckpt_write_buf(ofs, cur.data_tgt,  n_written);
            ckpt_write_buf(ofs, cur.data_dft,  n_written);
            ckpt_write_buf(ofs, cur.data_spec, n_written);
        }
        ofs.flush();
        if (!ofs) {
            SLT_WRN(slot, "failed to append context checkpoints to '%s' - the appendix is incomplete\n", filepath.c_str());
            return false;
        }
        SLT_INF(slot, "appended %u context checkpoint(s) (%.3f MiB) to '%s'\n",
                count, (float) n_written / 1024 / 1024, filepath.c_str());
        return true;
    }

    // returns the number of bytes consumed, 0 if there is no usable appendix
    size_t load_slot_checkpoints(const std::string & filepath, size_t offset, server_slot & slot) const {
        std::ifstream ifs(std::filesystem::u8path(filepath), std::ios::binary | std::ios::ate);
        const size_t file_size = ifs ? (size_t) ifs.tellg() : 0;
        if (!ifs || file_size < offset || !ifs.seekg(offset)) {
            return 0;
        }
        const size_t n_avail = file_size - offset; // bytes after the llama state payload
        size_t n_read = 0;
        uint32_t magic   = 0;
        uint32_t version = 0;
        uint32_t count   = 0;
        if (!ckpt_read(ifs, &magic, sizeof(magic), n_read) || magic != SLOT_CKPT_MAGIC) {
            return 0;
        }
        if (!ckpt_read(ifs, &version, sizeof(version), n_read) || version != SLOT_CKPT_VERSION ||
            !ckpt_read(ifs, &count,   sizeof(count),   n_read)) {
            SLT_WRN(slot, "invalid context checkpoint appendix in '%s' - ignored\n", filepath.c_str());
            return 0;
        }
        std::list<common_prompt_checkpoint> checkpoints;
        for (uint32_t i = 0; i < count; ++i) {
            common_prompt_checkpoint cur;
            cur.id_task = -1; // not created by a task - marks a checkpoint restored from a slot file
            if (!ckpt_read(ifs, &cur.n_tokens, sizeof(cur.n_tokens), n_read) ||
                !ckpt_read(ifs, &cur.pos_min,  sizeof(cur.pos_min),  n_read) ||
                !ckpt_read(ifs, &cur.pos_max,  sizeof(cur.pos_max),  n_read) ||
                !ckpt_read_buf(ifs, cur.data_tgt,  n_avail, n_read) ||
                !ckpt_read_buf(ifs, cur.data_dft,  n_avail, n_read) ||
                !ckpt_read_buf(ifs, cur.data_spec, n_avail, n_read)) {
                SLT_WRN(slot, "truncated context checkpoint appendix in '%s' - ignored\n", filepath.c_str());
                return 0;
            }
            // a saved checkpoint always holds a target state - an empty blob would roll back without restoring anything
            if (cur.data_tgt.empty()) {
                SLT_WRN(slot, "invalid context checkpoint appendix in '%s' - ignored\n", filepath.c_str());
                return 0;
            }
            checkpoints.push_back(std::move(cur));
            if (checkpoints.size() > (size_t) params_base.n_ctx_checkpoints) {
                checkpoints.pop_front();
            }
        }
        // the slot file does not check the draft context - test-load one draft checkpoint, drop the draft data if it does not fit
        if (ctx_dft != nullptr && !checkpoints.empty() && !checkpoints.back().data_dft.empty()) {
            const bool ok = checkpoints.back().load_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            llama_memory_seq_rm(llama_get_memory(ctx_dft), slot.id, -1, -1);
            if (!ok) {
                SLT_WRN(slot, "draft context checkpoint data in '%s' does not match the draft context - dropped\n", filepath.c_str());
                for (auto & cur : checkpoints) {
                    cur.clear_dft();
                }
            }
        }
        slot.prompt.checkpoints = std::move(checkpoints);
        SLT_INF(slot, "restored %zu context checkpoint(s) from '%s'\n", slot.prompt.checkpoints.size(), filepath.c_str());
        return n_read;
    }

    // the live draft state follows the checkpoints: the full draft context and the speculative state
    // the llama state payload holds the target context only
    static constexpr uint32_t SLOT_DRAFT_MAGIC   = 0x54464453; // "SDFT"
    static constexpr uint32_t SLOT_DRAFT_VERSION = 1;

    // returns false if the draft state could not be written completely
    bool save_slot_draft(const std::string & filepath, const server_slot & slot, size_t & n_written) const {
        n_written = 0;
        std::vector<uint8_t> data_dft;
        std::vector<uint8_t> data_spec;
        if (ctx_dft != nullptr) {
            data_dft.resize(llama_state_seq_get_size_ext(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_NONE));
            llama_state_seq_get_data_ext(ctx_dft, data_dft.data(), data_dft.size(), slot.id, LLAMA_STATE_SEQ_FLAGS_NONE);
        }
        common_speculative_get_state(spec.get(), slot.id, data_spec);
        if (data_dft.empty() && data_spec.empty()) {
            return true;
        }
        std::ofstream ofs(std::filesystem::u8path(filepath), std::ios::binary | std::ios::app);
        if (!ofs) {
            SLT_WRN(slot, "failed to append the draft state to '%s'\n", filepath.c_str());
            return false;
        }
        const uint32_t magic   = SLOT_DRAFT_MAGIC;
        const uint32_t version = SLOT_DRAFT_VERSION;
        ckpt_write(ofs, &magic,   sizeof(magic),   n_written);
        ckpt_write(ofs, &version, sizeof(version), n_written);
        ckpt_write_buf(ofs, data_dft,  n_written);
        ckpt_write_buf(ofs, data_spec, n_written);
        ofs.flush();
        if (!ofs) {
            SLT_WRN(slot, "failed to append the draft state to '%s' - the draft state is incomplete\n", filepath.c_str());
            return false;
        }
        SLT_INF(slot, "appended the draft state (%.3f MiB) to '%s'\n", (float) n_written / 1024 / 1024, filepath.c_str());
        return true;
    }

    // returns the number of bytes consumed, 0 if there is no usable draft state
    // the draft starts empty unless the file holds a draft state that fits the draft context
    size_t load_slot_draft(const std::string & filepath, size_t offset, server_slot & slot) const {
        if (ctx_dft != nullptr) {
            llama_memory_seq_rm(llama_get_memory(ctx_dft), slot.id, -1, -1);
        }
        std::ifstream ifs(std::filesystem::u8path(filepath), std::ios::binary | std::ios::ate);
        const size_t file_size = ifs ? (size_t) ifs.tellg() : 0;
        if (!ifs || file_size < offset || !ifs.seekg(offset)) {
            return 0;
        }
        const size_t n_avail = file_size - offset;
        size_t n_read = 0;
        uint32_t magic   = 0;
        uint32_t version = 0;
        if (!ckpt_read(ifs, &magic, sizeof(magic), n_read) || magic != SLOT_DRAFT_MAGIC) {
            return 0;
        }
        std::vector<uint8_t> data_dft;
        std::vector<uint8_t> data_spec;
        if (!ckpt_read(ifs, &version, sizeof(version), n_read) || version != SLOT_DRAFT_VERSION ||
            !ckpt_read_buf(ifs, data_dft,  n_avail, n_read) ||
            !ckpt_read_buf(ifs, data_spec, n_avail, n_read)) {
            SLT_WRN(slot, "invalid draft state in '%s' - ignored\n", filepath.c_str());
            return 0;
        }
        if (!data_dft.empty()) {
            if (ctx_dft == nullptr ||
                llama_state_seq_set_data_ext(ctx_dft, data_dft.data(), data_dft.size(), slot.id, LLAMA_STATE_SEQ_FLAGS_NONE) != data_dft.size()) {
                if (ctx_dft != nullptr) {
                    llama_memory_seq_rm(llama_get_memory(ctx_dft), slot.id, -1, -1);
                }
                SLT_WRN(slot, "draft state in '%s' does not match the draft context - dropped\n", filepath.c_str());
                return n_read;
            }
        }
        if (!data_spec.empty()) {
            common_speculative_set_state(spec.get(), slot.id, data_spec);
        }
        SLT_INF(slot, "restored the draft state (%.3f MiB) from '%s'\n", (float) n_read / 1024 / 1024, filepath.c_str());
        return n_read;
    }

    int execution_priority(const server_slot & slot) {
        return queue_tasks.prefix_priority(slot.task->id, slot.task->params.execution_priority);
    }

    void send_prefix_progress(server_task & task, const server_slot * producer, size_t target, size_t published) {
        const auto now = ggml_time_ms();
        if (producer && now - task.prefix_progress_ms < 1000 && task.prefix_producer == producer->task->id) return;
        task.prefix_progress_ms = now;
        if (!task.params.stream || !task.params.return_progress) return;
        auto res = std::make_unique<server_task_result_cmpl_partial>();
        res->id = task.id;
        res->index = task.index;
        res->res_type = task.params.res_type;
        res->oaicompat_model = task.params.oaicompat_model;
        res->oaicompat_cmpl_id = task.params.oaicompat_cmpl_id;
        res->is_progress = true;
        res->progress.total = task.n_tokens();
        res->progress.waiting_for_prefix = producer != nullptr;
        res->progress.prefix_producer = producer ? producer->task->params.serverless_request_id : "";
        res->progress.prefix_target_tokens = target;
        res->progress.prefix_published_tokens = published;
        queue_results.send(std::move(res));
    }

    bool defer_shared_prefix(server_task & task) {
        if (!prompt_page_cache || spec || task.type != SERVER_TASK_TYPE_COMPLETION ||
                task.is_parent() || task.is_child() || !task.params.cache_prompt ||
                task.params.cache_control.lifecycle == server_cache_lifecycle::one_shot ||
                !task.tokens.has_stable_media_identity() || task.tokens.size() < 2 ||
                !lora_get_enabled_ids(construct_lora_list(task.params.lora)).empty()) return false;
        server_slot * producer = nullptr;
        size_t target = 0;
        for (auto & slot : slots) {
            if (!slot.is_processing() || !slot.task || slot.capacity_parked() ||
                    (slot.state != SLOT_STATE_STARTED && slot.state != SLOT_STATE_PROCESSING_PROMPT) ||
                    !prompt_page_cache_eligible(slot, *slot.task) || !slot.task->params.cache_prompt ||
                    slot.task->params.cache_control.lifecycle == server_cache_lifecycle::one_shot) continue;
            const size_t common = std::min(task.tokens.size() - 1,
                slot.task->tokens.get_common_prefix(task.tokens));
            size_t boundary = 0;
            for (const auto & page : slot.prompt_page_boundaries) {
                if (page.token_count <= common) boundary = page.token_count;
                else break;
            }
            if (boundary <= slot.prompt.tokens.size() || boundary <= target) continue;
            target = boundary;
            producer = &slot;
        }
        if (producer) {
            const size_t published = prompt_page_cache->match_tokens(task.tokens);
            if (published < target) {
                send_prefix_progress(task, producer, target, published);
                task.prefix_producer = producer->task->id;
                queue_tasks.defer_prefix(std::move(task));
                return true;
            }
        }
        if (task.prefix_producer >= 0) {
            send_prefix_progress(task, nullptr, 0, 0);
            task.prefix_producer = -1;
        }
        return false;
    }

    // returns false to decline the task, it is offered again after the decode is done
    bool process_single_task(server_task && task, bool is_yielding) {
        // while yielding, an encode / decode is running and only reading the server state is safe
        if (is_yielding && task.type != SERVER_TASK_TYPE_METRICS && task.type != SERVER_TASK_TYPE_SLOT_GET &&
                task.type != SERVER_TASK_TYPE_RESPONSES_CONTEXT) {
            SRV_DBG("decoding, decline task, id_task = %d\n", task.id);
            return false;
        }

        switch (task.type) {
            case SERVER_TASK_TYPE_COMPLETION:
            case SERVER_TASK_TYPE_INFILL:
            case SERVER_TASK_TYPE_TRANSCRIBE_RT:
            case SERVER_TASK_TYPE_EMBEDDING:
            case SERVER_TASK_TYPE_HIDDEN_STATE:
            case SERVER_TASK_TYPE_RERANK:
            case SERVER_TASK_TYPE_TTS:
            case SERVER_TASK_TYPE_DECISION:
                {
                    // special case: if input is provided via CLI, tokenize it first
                    // otherwise, no need to tokenize as it's already done inside the HTTP thread
                    if (task.cli) {
                        if (!tokenize_cli_input(task)) {
                            break;
                        }
                    }

                    const int id_task = task.id;

                    if (defer_shared_prefix(task)) break;
                    server_slot * slot = get_available_slot(task);

                    //
                    // slot scheduling logic
                    //

                    if (slot == nullptr) {
                        // if no slot is available, we defer this task for processing later
                        SRV_DBG("no slot is available, defer task, id_task = %d\n", id_task);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", id_task);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (task.is_parent()) {
                        // try getting free slots for all child tasks
                        size_t n_child_tasks = task.child_tasks.size();
                        std::vector<server_slot *> child_slots = get_free_slots(n_child_tasks, slot->id);
                        if (child_slots.size() < n_child_tasks) {
                            SRV_DBG("not enough free slots for child tasks, n_free = %zu, n_children = %zu, defer task, id_task = %d\n", child_slots.size(), n_child_tasks, id_task);
                            queue_tasks.defer(std::move(task));
                            break;
                        }
                        if (!launch_slots_with_parent_task(*slot, child_slots, std::move(task))) {
                            SRV_ERR("failed to launch slot with parent task, id_task = %d\n", id_task);
                            break; // drop the task
                        }
                    } else if (!launch_slot_with_task(*slot, std::move(task))) {
                        SRV_ERR("failed to launch slot with task, id_task = %d\n", id_task);
                        break; // drop the task
                    }

                    if (params_base.cache_idle_slots) {
                        for (auto & slot : slots) {
                            if (!slot.is_processing()) {
                                SLT_TRC(slot, "%s", "saving idle slot to prompt cache\n");

                                if (slot.prompt_save(*prompt_cache)) {
                                    SLT_DBG(slot, "%s", "__TEST_TAG_CACHE_IDLE_SLOT__\n");
                                    prompt_cache->update();
                                }

                                if (params_base.kv_unified) {
                                    // [TAG_IDLE_SLOT_CLEAR]
                                    slot.prompt_clear();
                                }
                            }
                        }
                    }
                } break;
            case SERVER_TASK_TYPE_CANCEL:
                {
                    // Responses cancellation is acknowledged only after model
                    // work is synchronized and the exact queued/active task ends.
                    const bool tracked = response_registry.owns(task.id_target);
                    if (tracked && ctx_tgt) llama_synchronize(ctx_tgt);
                    if (tracked && ctx_dft) llama_synchronize(ctx_dft);
                    // release slot linked with the task id
                    for (auto & slot : slots) {
                        if (slot.task && slot.task->id == task.id_target) {
                            slot.release();
                            break;
                        }
                    }
                    response_registry.finish(task.id_target); // also covers removed queued/deferred tasks
                } break;
            case SERVER_TASK_TYPE_RESPONSES_CONTEXT:
                {
                    auto result = std::make_unique<server_task_result_responses_context>();
                    result->id = task.id;
                    result->proof = prompt_page_cache ? prompt_page_cache->context_control(task.response_context)
                        : json{{"success", false}, {"reason", "page cache unavailable"}};
                    queue_results.send(std::move(result));
                } break;
            case SERVER_TASK_TYPE_RESPONSES_SUSPEND:
                {
                    auto reply = std::make_unique<server_task_result_control>();
                    reply->id = task.id;
                    reply->success = false;
                    reply->message = "no safe prefill request";
                    for (auto & slot : slots) {
                        if (!slot.task || slot.task->id != task.id_target) continue;
                        if (slot.stats.n_gen != 0 || slot.state != SLOT_STATE_PROCESSING_PROMPT) {
                            reply->message = "decode suspend is not enabled yet";
                            break;
                        }
                        if (ctx_tgt) llama_synchronize(ctx_tgt);
                        if (!slot.prompt.tokens.empty()) capture_prompt_page(slot, true);
                        auto marker = std::make_unique<server_task_result_responses_suspended>();
                        marker->id = slot.task->id;
                        marker->index = slot.task->index;
                        marker->request_id = slot.task->params.serverless_request_id;
                        queue_results.send(std::move(marker));
                        slot.release();
                        reply->success = true;
                        reply->message = "prefill suspended";
                        break;
                    }
                    queue_results.send(std::move(reply));
                } break;
            case SERVER_TASK_TYPE_CONTROL:
                {
                    auto res = std::make_unique<server_task_result_control>();
                    res->id = task.id;

                    server_slot * slot = get_slot_by_cmpl_id(task.params.control_cmpl_id);
                    if (slot == nullptr) {
                        SRV_WRN("control %d on unknown completion id=%s, no live slot\n",
                                (int) task.params.control_action, task.params.control_cmpl_id.c_str());
                        res->success = false;
                        res->message = "no active completion for this id";
                        queue_results.send(std::move(res));
                        break;
                    }

                    if (task.params.control_action == SERVER_CONTROL_ACTION_REASONING_END) {
                        // the budget sampler only exists when reasoning control was armed
                        if (!slot->task->params.sampling.reasoning_control) {
                            res->success = false;
                            res->message = "reasoning control not enabled for this completion";
                            queue_results.send(std::move(res));
                            break;
                        }
                        // act on the live slot mid generation, never defer
                        common_sampler_reasoning_budget_force(slot->smpl.get());
                        res->success = true;
                    } else if (task.params.control_action == SERVER_CONTROL_ACTION_AUDIO_APPEND
                            || task.params.control_action == SERVER_CONTROL_ACTION_AUDIO_COMMIT
                            || task.params.control_action == SERVER_CONTROL_ACTION_AUDIO_END) {
                        if (!slot->audio_stream) {
                            res->success = false;
                            res->message = "not a realtime transcription slot";
                        } else {
                            auto * stream = slot->audio_stream.get();
                            if (task.params.control_action == SERVER_CONTROL_ACTION_AUDIO_APPEND) {
                                mtmd_audio_stream_push(stream, task.params.control_audio.data(), task.params.control_audio.size());
                            } else {
                                const int32_t ret = mtmd_audio_stream_flush(stream, task.params.control_action == SERVER_CONTROL_ACTION_AUDIO_END);
                                // a non-final commit completes its item once the rows of its audio are decoded
                                if (ret == 0 && task.params.control_action == SERVER_CONTROL_ACTION_AUDIO_COMMIT) {
                                    slot->audio_commit_rows.push_back(mtmd_audio_stream_get_n_rows_flushed(stream));
                                }
                            }
                            slot->audio_wait = false;
                            res->success = true;
                        }
                    } else {
                        res->success = false;
                        res->message = "unknown control action";
                    }

                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_CACHE_RELEASE:
                {
                    auto res = std::make_unique<server_task_result_cache_release>();
                    res->id = task.id;
                    res->success = prompt_page_cache != nullptr;
                    if (prompt_page_cache) {
                        res->released_nodes = prompt_page_cache->release_branch(task.cache_branch);
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_NEXT_RESPONSE:
                {
                    // do nothing
                } break;
            case SERVER_TASK_TYPE_METRICS:
                {
                    int n_processing_slots = 0;

                    for (server_slot & slot : slots) {
                        if (slot.is_processing()) {
                            n_processing_slots++;
                        }
                    }
                    SRV_DBG("n_processing_slots = %d\n", n_processing_slots);

                    auto res = std::make_unique<server_task_result_metrics>();
                    res->id                  = task.id;
                    res->n_processing_slots  = n_processing_slots;
                    res->n_tasks_deferred    = queue_tasks.queue_tasks_deferred_size();
                    res->metrics             = metrics;

                    if (task.metrics_reset_bucket) {
                        metrics.reset_bucket();
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_GET:
                {
                    json slots_data = json::array();

                    int n_idle_slots = 0;

                    for (server_slot & slot : slots) {
                        if (!slot.is_processing()) {
                            n_idle_slots++;
                        }

                        slots_data.push_back(slot.to_json(slots_debug == 0));
                    }
                    SRV_DBG("n_idle_slots = %d\n", n_idle_slots);

                    auto res = std::make_unique<server_task_result_slots>();
                    res->id           = task.id;
                    res->slots_data   = std::move(slots_data);
                    res->n_idle_slots = n_idle_slots;

                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_SAVE:
                {
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.slot_action.filename;
                    std::string filepath = task.slot_action.filepath;

                    std::vector<char> packed;
                    try {
                        packed = slot->prompt.tokens.serialize();
                    } catch (const std::exception & err) {
                        send_error(task, err.what(), ERROR_TYPE_NOT_SUPPORTED);
                        break;
                    }

                    GGML_ASSERT(packed.size() % sizeof(llama_token) == 0);
                    const size_t nwrite = llama_state_seq_save_file(
                        ctx_tgt, filepath.c_str(), slot->id,
                        reinterpret_cast<const llama_token *>(packed.data()), packed.size() / sizeof(llama_token));
                    if (nwrite == 0) {
                        send_error(task, "Unable to save slot", ERROR_TYPE_SERVER);
                        break;
                    }

                    size_t nwrite_ckpt = 0;
                    if (!save_slot_checkpoints(filepath, *slot, nwrite_ckpt)) {
                        send_error(task, "Unable to save slot: incomplete context checkpoints", ERROR_TYPE_SERVER);
                        break;
                    }

                    size_t nwrite_dft = 0;
                    if (!save_slot_draft(filepath, *slot, nwrite_dft)) {
                        send_error(task, "Unable to save slot: incomplete draft state", ERROR_TYPE_SERVER);
                        break;
                    }

                    const int64_t t_end = ggml_time_us();
                    const double t_save_ms = (t_end - t_start) / 1000.0;

                    auto res = std::make_unique<server_task_result_slot_save_load>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->filename = filename;
                    res->is_save  = true;
                    res->n_tokens = slot->prompt.tokens.size();
                    res->n_bytes  = nwrite + nwrite_ckpt + nwrite_dft;
                    res->t_ms     = t_save_ms;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_RESTORE:
                {
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.slot_action.filename;
                    std::string filepath = task.slot_action.filepath;

                    size_t nread = 0;
                    try {
                        size_t n_packed = 0;
                        llama_tokens packed;
                        nread = llama_state_seq_load_file(ctx_tgt, filepath.c_str(), slot->id, nullptr, 0, &n_packed);
                        if (nread != 0) {
                            packed.resize(std::max<size_t>(1, n_packed));
                            nread = llama_state_seq_load_file(ctx_tgt, filepath.c_str(), slot->id, packed.data(), packed.size(), &n_packed);
                        }
                        if (nread == 0) {
                            throw std::runtime_error("No available space in KV cache or invalid slot save file");
                        }
                        packed.resize(n_packed);

                        server_tokens restored = server_tokens::deserialize(packed, mctx != nullptr);

                        if (restored.size() > (size_t) slot->n_ctx) {
                            throw std::runtime_error("Restored prompt does not fit in the slot context");
                        }

                        if (!restored.validate(ctx_tgt)) {
                            throw std::runtime_error("Invalid tokens in slot save file");
                        }

                        slot->prompt.clear();
                        slot->prompt.tokens = std::move(restored);
                    } catch (const std::exception & err) {
                        slot->prompt_clear();
                        send_error(task, std::string("Unable to restore slot: ") + err.what(), ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }

                    // nread is the end offset of the llama state payload within the file
                    const size_t nread_ckpt = load_slot_checkpoints(filepath, nread, *slot);
                    const size_t nread_dft  = load_slot_draft(filepath, nread + nread_ckpt, *slot);

                    const int64_t t_end = ggml_time_us();
                    const double t_restore_ms = (t_end - t_start) / 1000.0;

                    auto res = std::make_unique<server_task_result_slot_save_load>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->filename = filename;
                    res->is_save  = false;
                    res->n_tokens = slot->prompt.tokens.size();
                    res->n_bytes  = nread + nread_ckpt + nread_dft;
                    res->t_ms     = t_restore_ms;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_ERASE:
                {
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    // Erase token cache
                    const size_t n_erased = slot->prompt.tokens.size();

                    slot->prompt_clear();

                    auto res = std::make_unique<server_task_result_slot_erase>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->n_erased = n_erased;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_GET_LORA:
                {
                    // TODO @ngxson : make lora_adapters a dedicated member of server_context
                    auto & loras = params_base.lora_adapters;
                    auto res = std::make_unique<server_task_result_get_lora>();
                    res->id = task.id;
                    for (size_t i = 0; i < loras.size(); ++i) {
                        auto & lora = loras[i];
                        std::string alora_invocation_string = "";
                        const uint64_t n_alora_tokens = llama_adapter_get_alora_n_invocation_tokens(lora.ptr);
                        llama_tokens alora_invocation_tokens;
                        if (n_alora_tokens) {
                            const llama_token * alora_tokens = llama_adapter_get_alora_invocation_tokens(lora.ptr);
                            for (uint64_t j = 0; j < n_alora_tokens; ++j) {
                                alora_invocation_string += common_token_to_piece(vocab, alora_tokens[j]);
                                alora_invocation_tokens.push_back(alora_tokens[j]);
                            }
                        }
                        res->loras.push_back(server_task_result_get_lora::lora{
                            lora,
                            alora_invocation_string,
                            alora_invocation_tokens,
                        });
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SET_LORA:
                {
                    auto new_loras = construct_lora_list(task.set_lora);
                    // logging
                    for (size_t i = 0; i < new_loras.size(); ++i) {
                        SRV_TRC("set lora adapter idx=%zu scale=%f\n", i, new_loras[i].scale);
                    }
                    // TODO @ngxson : make lora_adapters a dedicated member of server_context
                    params_base.lora_adapters = new_loras;
                    auto res = std::make_unique<server_task_result_apply_lora>();
                    res->id = task.id;
                    queue_results.send(std::move(res));
                } break;
        }

        return true;
    }

    void iterate(std::vector<server_slot> & slots, std::function<void(server_slot &)> callback) {
        for (auto & slot : slots) {
            try {
                callback(slot);
            } catch (const std::exception & e) {
                SLT_ERR(slot, "got exception: %s\n", e.what());
                send_error(slot, std::string("got exception: ") + e.what(), ERROR_TYPE_SERVER);
                slot.release();
            }
        }
    }

    void iterate(std::vector<server_slot *> & slots, std::function<void(server_slot &)> callback) {
        for (auto & slot : slots) {
            try {
                callback(*slot);
            } catch (const std::exception & e) {
                SLT_ERR(*slot, "got exception: %s\n", e.what());
                send_error(*slot, std::string("got exception: ") + e.what(), ERROR_TYPE_SERVER);
                slot->release();
            }
        }
    }

    void abort_all_slots(const std::string & reason) {
        for (auto & slot : slots) {
            if (slot.is_processing()) {
                send_error(slot, reason, ERROR_TYPE_SERVER);
                slot.release();
            }
        }
    }

    template<class Counts>
    static std::vector<uint64_t> snapshot_counters(const Counts & counters) {
        std::vector<uint64_t> result;
        for (const auto & counter : counters) result.push_back(counter.load());
        return result;
    }

    // Host wall time, not device kernel time. Last histogram bucket is >=8 sequences.
    std::atomic<uint64_t> wall_pre_us{0}, wall_pre_calls{0}, wall_post_us{0}, wall_post_calls{0};
    std::atomic<uint64_t> wall_prefill_us{0}, wall_prefill_calls{0};
    json execution_memory_snapshot = json::object();
    uint64_t execution_memory_sample = 0;
    json execution_kv_occupancy = json::object();
    int64_t execution_kv_occupancy_us = 0;
    std::atomic<uint64_t> execution_waves[8]{};
    std::atomic<uint64_t> execution_prefill_waves[8]{};
    std::atomic<uint64_t> execution_mixed_waves{0};
    std::atomic<uint64_t> execution_fallback_single{0}, execution_fallback_capacity{0};
    std::atomic<uint64_t> execution_fallback_rebuild{0}, execution_fallback_other{0};
    std::array<std::atomic<uint64_t>, 8> wall_decode_us{}, wall_decode_calls{};
    struct execution_wall_timer {
        std::atomic<uint64_t> & elapsed;
        std::atomic<uint64_t> & calls;
        const int64_t start = ggml_time_us();
        execution_wall_timer(std::atomic<uint64_t> & elapsed, std::atomic<uint64_t> & calls)
            : elapsed(elapsed), calls(calls) {}
        ~execution_wall_timer() {
            elapsed.fetch_add(ggml_time_us() - start, std::memory_order_relaxed);
            calls.fetch_add(1, std::memory_order_relaxed);
        }
    };

    // @ngxson : for debugging only
    int64_t t_pre_decode  = 0;
    int64_t t_decode      = 0;
    int64_t t_post_decode = 0;
    int64_t t_sampl       = 0;
    int64_t n_pre_decode  = 0;
    int64_t n_decode      = 0;
    int64_t n_post_decode = 0;
    int64_t n_sampl       = 0;
// #define DEBUG_TIMINGS
#ifdef DEBUG_TIMINGS
    struct scoped_timer {
        int64_t & t;
        int64_t & n;
        int64_t t_start;
        scoped_timer(int64_t & t_, int64_t & n_) : t(t_), n(n_) {
            t_start = ggml_time_us();
        }
        ~scoped_timer() {
            t += ggml_time_us() - t_start;
            n++;
        }
    };
#else
    struct scoped_timer {
        scoped_timer(int64_t &, int64_t &) {}
        ~scoped_timer() {}
    };
#endif

    // fill slot.inp_embd with the next rows of the audio stream
    // returns false if the slot has nothing to decode now
    bool slot_audio_step(server_slot & slot) {
        if (!slot.inp_embd.empty()) {
            return true; // rows from a previous step, not decoded yet
        }

        // every produced row is decoded and its delta queued, the commits ending there are complete
        const int64_t n_rows = mtmd_audio_stream_get_n_rows(slot.audio_stream.get());
        while (!slot.audio_commit_rows.empty() && slot.audio_commit_rows.front() <= n_rows) {
            slot.audio_commit_rows.pop_front();
            auto res = std::make_unique<server_task_result_cmpl_partial>();
            res->id                   = slot.task->id;
            res->index                = slot.task->index;
            res->n_decoded            = slot.stats.n_gen;
            res->n_prompt_tokens      = slot.task->n_tokens();
            res->res_type             = slot.task->params.res_type;
            res->oaicompat_model      = slot.task->params.oaicompat_model;
            res->oaicompat_cmpl_id    = slot.task->params.oaicompat_cmpl_id;
            res->audio_commit_drained = true;
            queue_results.send(std::move(res));
        }

        const int32_t n = mtmd_audio_stream_step(slot.audio_stream.get(), slot.sampled);
        if (n > 0) {
            const float * rows = mtmd_audio_stream_get_output_embd(slot.audio_stream.get());
            slot.inp_embd.assign(rows, rows + (size_t) n * batch.n_embd);
            slot.audio_wait = false;
            return true;
        }
        if (n == 0) {
            slot.audio_wait = true;
            return false;
        }
        if (n == -2) {
            send_error(slot, "audio stream error", ERROR_TYPE_SERVER);
            slot.release();
            return false;
        }

        // end of stream
        slot.stop           = STOP_TYPE_EOS;
        slot.has_next_token = false;
        slot.print_timings();
        send_final_response(slot);
        slot.release();
        return false;
    }

    void update_slots() {
        queue_tasks.poll_prefix();
#ifdef DEBUG_TIMINGS
        static int64_t t_prev = 0;
        int64_t t_start = ggml_time_us();
        if (t_start - t_prev > 5 * 1000 * 1000) { // every 5 seconds
            t_prev = t_start;
            SRV_INF("n_pre_decode      = %" PRId64 "\n", n_pre_decode);
            SRV_INF("avg t_pre_decode  = %f ms\n", (double) t_pre_decode / n_pre_decode / 1000.0);
            SRV_INF("avg t_decode      = %f ms\n", (double) t_decode / n_decode / 1000.0);
            SRV_INF("avg t_post_decode = %f ms\n", (double) t_post_decode / n_post_decode / 1000.0);
            SRV_INF("avg t_sampl       = %f ms\n", (double) t_sampl / n_sampl / 1000.0);
        }
#endif

        if (prompt_page_cache) {
            std::unordered_set<int> active_tasks;
            for (const auto & slot : slots) if (slot.task && slot.is_processing()) active_tasks.insert(slot.task->id);
            prompt_page_cache->maintenance(active_tasks);
        }

        // check if all slots are idle
        // a realtime slot waiting for audio has no work, the next audio control task wakes the loop
        {
            bool all_idle = true;

            for (auto & slot : slots) {
                if (slot.is_processing() && !slot.audio_wait) {
                    all_idle = false;
                    break;
                }
            }

            if (all_idle) {
                SRV_TRC("%s", "all slots are idle\n");

                metrics_flush_idle();

                return; // skip further processing

            } else {
                SRV_DBG("%s", "posting NEXT_RESPONSE\n");

                server_task task(SERVER_TASK_TYPE_NEXT_RESPONSE);
                task.id = queue_tasks.get_new_id();
                queue_tasks.post(std::move(task));
            }
        }

        try {
            scoped_timer t(t_pre_decode, n_pre_decode);
            execution_wall_timer wall(wall_pre_us, wall_pre_calls);
            pre_decode();
        } catch (const std::exception & e) {
            SRV_ERR("pre_decode() failed: %s\n", e.what());
            abort_all_slots("pre_decode() failed: " + std::string(e.what()));

            // the batch is half-built and not rendered, skip now to avoid UB
            return;
        }

        // note: TTS slots bypass the shared batch entirely
        try {
            process_tts_slots();
        } catch (const std::exception & e) {
            SRV_ERR("process_tts_slots() failed: %s\n", e.what());
            abort_all_slots("process_tts_slots() failed: " + std::string(e.what()));
        }

        GGML_ASSERT(batch.slot_batched || batch.size() == 0);
        if (batch.size() == 0 && !pending_media.empty()) {
            try {
                metrics_pre_decode();
                execution_wall_timer wall(wall_prefill_us, wall_prefill_calls);
                int ret = 0;
                llama_batch empty = {};
                auto media = media_wave_batches();
                queue_tasks.yield_to_queue([&]() {
                    ret = ctx_tgt->decode_with_execution_lanes(empty, execution_lane_ptrs, media);
                    record_execution_wave(empty, 0);
                });
                if (ret) throw std::runtime_error("media execution wave failed");
                commit_media_wave();
            } catch (const std::exception & e) {
                pending_media.clear();
                abort_all_slots(std::string("media execution wave failed: ") + e.what());
            }
            return;
        }
        if (batch.size() == 0) {
            for (auto & slot : slots) {
                if (slot.media_preparation.valid()) {
                    queue_tasks.yield_to_queue([&]() {
                        slot.media_preparation.wait_for(std::chrono::milliseconds(5));
                    });
                    break;
                }
            }
        }

        if (batch.slot_batched) {
            auto & slot_batched      = batch.slot_batched;
            auto & alora_scale       = batch.alora_scale;
            auto & alora_disabled_id = batch.alora_disabled_id;

            // TODO @ngxson : alora handling is too messy, need to refactor it to be more clear and maintainable
            // apply lora, only need to do it once per batch
            common_set_adapter_lora(ctx_tgt, slot_batched->lora);

            // if the lora is temporarily disabled for an alora, re-enable it
            // for next time
            if (alora_scale > 0.0f) {
                SRV_DBG("re-enabling alora with scale %f\n", alora_scale);
                slot_batched->lora[alora_disabled_id].scale = alora_scale;
            }

            llama_set_embeddings(ctx_tgt, slot_batched->need_embd());
            ctx_tgt->set_hidden_state_layers(slot_batched->task->hidden_state_layers);
        }

        int32_t off_next = 0;
        int32_t n_batch = llama_n_batch(ctx_tgt);
        for (int32_t off = 0; off < batch.size(); off = off_next) {
            const int32_t n_tokens = std::min(n_batch, batch.size() - off);
            llama_batch batch_view = {};
            try {
                scoped_timer t(t_decode, n_decode);
                // TODO @ngxson : maybe handle n_batch == 1 here instead of inside decode()

                batch.render(off, n_tokens);
                batch_view = batch.get_legacy_view(off, n_tokens);
                bool prompt = false;
                std::unordered_set<llama_seq_id> sequences;
                for (int32_t i = off; i < off + n_tokens; ++i) {
                    prompt |= batch.tokens[i].is_prompt;
                    sequences.insert(batch.tokens[i].id_slot);
                }
                const size_t bucket = std::min<size_t>(8, sequences.size()) - 1;
                execution_wall_timer wall(prompt ? wall_prefill_us : wall_decode_us[bucket],
                                          prompt ? wall_prefill_calls : wall_decode_calls[bucket]);
                bool ok = decode(n_batch, off, batch_view);
#ifdef DEBUG_TIMINGS
                llama_synchronize(ctx_tgt);
#endif

                if (ok) {
                    // move the head of the batch forward with the number of tokens we just processed
                    off_next = off + n_tokens;

                    // on successful decode, restore the original batch size
                    n_batch = llama_n_batch(ctx_tgt);
                } else {
                    // try again with the updated n_batch
                    continue;
                }
            } catch (const std::exception & e) {
                SRV_ERR("decode() failed: %s\n", e.what());
                abort_all_slots("decode() failed: " + std::string(e.what()));
                break; // stop any further processing
            }

            try {
                scoped_timer t(t_post_decode, n_post_decode);
                execution_wall_timer wall(wall_post_us, wall_post_calls);
                post_decode(n_tokens, off);
            } catch (const std::exception & e) {
                SRV_ERR("post_decode() failed: %s\n", e.what());
                abort_all_slots("post_decode() failed: " + std::string(e.what()));
                break; // stop any further processing
            }
        }
    }

    void process_tts_slots() {
        iterate(slots, [&](server_slot & slot) {
            if (!slot.is_processing() || slot.task->type != SERVER_TASK_TYPE_TTS) {
                return;
            }

            llama_set_embeddings(ctx_tgt, true);

            if (slot.state == SLOT_STATE_PROCESSING_PROMPT) {
                const int32_t ret = slot.tts.ctx.step_prompt(llama_n_batch(ctx_tgt));
                if (ret < 0) {
                    send_error(slot, "TTS prompt processing failed", ERROR_TYPE_SERVER);
                    slot.release();
                } else if (ret == 0) {
                    // done prompt, do sample and save h_state_prompt
                    slot.tts.sampled = common_sampler_sample(slot.smpl.get(), ctx_tgt, -1);
                    common_sampler_accept(slot.smpl.get(), slot.tts.sampled, true);
                    const float * h_embd = llama_get_embeddings_ith(ctx_tgt, -1);
                    slot.tts.h_state_prompt.assign(h_embd, h_embd + llama_model_n_embd(model_tgt));
                    slot.tts.h_state = slot.tts.h_state_prompt.data();
                    slot.state = SLOT_STATE_GENERATING;
                }
                return;
            }

            const int32_t n_predict = slot.n_predict_max > 0 ? slot.n_predict_max : 512;
            if (slot.tts.stop || slot.tts.n_decoded >= n_predict) {
                int32_t      sample_rate = 0;
                const char * data        = nullptr;
                size_t       data_len    = 0;
                // generation truly ends here: force out any sub-window remainder still buffered
                if (slot.tts.ctx.flush() != 0 || slot.tts.ctx.get_output(&sample_rate, &data, &data_len) != 0) {
                    send_error(slot, "failed to finalize TTS output", ERROR_TYPE_SERVER);
                } else {
                    send_tts_result(slot, sample_rate, data, data_len, true);
                }
                slot.release();
                return;
            }

            const float * h_state_next = nullptr;
            bool          stop         = false;
            if (slot.tts.ctx.step_gen(slot.tts.sampled, slot.tts.h_state, &h_state_next, &stop) != 0) {
                send_error(slot, "TTS generation failed", ERROR_TYPE_SERVER);
                slot.release();
                return;
            }
            if (h_state_next == nullptr) {
                // end-of-speech without a new frame (e.g. pocket-tts eos head)
                slot.tts.stop = true;
                return;
            }
            slot.tts.h_state = h_state_next;
            slot.tts.n_decoded++;
            slot.tts.stop = stop;

            if (!stop) {
                slot.tts.sampled = common_sampler_sample(slot.smpl.get(), ctx_tgt, -1);
                common_sampler_accept(slot.smpl.get(), slot.tts.sampled, true);
            }

            if (slot.task->params.stream) {
                int32_t      sample_rate = 0;
                const char * data        = nullptr;
                size_t       data_len    = 0;
                if (slot.tts.ctx.get_output(&sample_rate, &data, &data_len) != 0) {
                    send_error(slot, "TTS streaming output failed", ERROR_TYPE_SERVER);
                    slot.release();
                } else if (data_len > 0) {
                    send_tts_result(slot, sample_rate, data, data_len, false);
                }
            }
        });
    }

    uint64_t prefill_round = 0;

    void manage_unified_capacity() {
        if (!params_base.kv_unified || model_tgt->arch != LLM_ARCH_GLM5NEXT || spec) return;
        std::vector<server_slot *> resident, parked;
        std::vector<std::pair<llama_seq_id, uint32_t>> budgets;
        for (auto & slot : slots) {
            if (!slot.is_processing() || !slot.task || slot.state == SLOT_STATE_WAIT_OTHER) continue;
            if (slot.capacity_parked()) { parked.push_back(&slot); continue; }
            resident.push_back(&slot);
            const uint64_t end = std::max<int64_t>(slot.task->tokens.pos_next(),
                slot.prompt.tokens.pos_next() + slot.media_offset);
            budgets.emplace_back(slot.id, std::min<uint64_t>(slot.n_ctx, end + rolling_decode_reserve()));
        }
        if (resident.empty() && !parked.empty()) {
            // Preserve all sampler/output/media cursors. Only device state moved.
            auto & slot = **std::min_element(parked.begin(), parked.end(),
                [](auto a, auto b) { return a->task->id < b->task->id; });
            while (try_clear_idle_slots()) {}
            llama_token dummy = 0; size_t count = 0;
            if (!llama_state_seq_load_file(ctx_tgt, slot.capacity_file.c_str(), slot.id,
                    &dummy, 0, &count)) {
                send_error(slot, "failed to restore capacity checkpoint", ERROR_TYPE_SERVER);
                slot.release();
                return;
            }
            std::remove(slot.capacity_file.c_str());
            slot.capacity_file.clear();
            SLT_INF(slot, "%s", "resumed exact sequence after unified capacity pressure\n");
            return;
        }
        // Private lanes cannot retry a wave after another lane has committed.
        // Reserve active growth together with all idle physical cells first.
        if (!budgets.empty()) {
            auto physical_budgets = budgets;
            for (const auto & slot : slots) {
                if (std::none_of(budgets.begin(), budgets.end(), [&](const auto & entry) {
                        return entry.first == slot.id;
                    })) physical_budgets.emplace_back(slot.id, 0);
            }
            while (llama_glm5next_reserved_cells(ctx_tgt, physical_budgets) > llama_n_ctx(ctx_tgt) &&
                    try_clear_idle_slots()) {}
        }
        if (resident.size() <= 1 || llama_glm5next_reserved_cells(ctx_tgt, budgets) <= llama_n_ctx(ctx_tgt)) return;
        std::sort(resident.begin(), resident.end(),
            [](auto a, auto b) { return a->task->id < b->task->id; });
        for (size_t i = 1; i < resident.size(); ++i) {
            auto & slot = *resident[i];
#if defined(__linux__)
            std::string path = (std::filesystem::temp_directory_path() / "llama-capacity-XXXXXX").string();
            int fd = mkstemp(path.data());
            if (fd < 0) throw std::runtime_error("cannot create private capacity checkpoint");
            close(fd);
            const size_t saved = llama_state_seq_save_file(ctx_tgt, path.c_str(), slot.id, nullptr, 0);
            if (!saved) { std::remove(path.c_str()); throw std::runtime_error("cannot save capacity checkpoint"); }
            // No state is discarded until the complete file has been written.
            slot.capacity_file = path;
            if (!llama_memory_seq_rm(llama_get_memory(ctx_tgt), slot.id, -1, -1))
                throw std::runtime_error("cannot release parked sequence");
            SLT_INF(slot, "parked exact sequence for unified capacity, bytes = %zu\n", saved);
#else
            throw std::runtime_error("capacity parking requires private temporary files");
#endif
        }
    }

    void pre_decode() {
        pending_media.clear();
        manage_unified_capacity();

        // apply context-shift if needed
        // TODO: simplify and improve
        iterate(slots, [&](server_slot & slot) {
            if (slot.task && slot.task->type == SERVER_TASK_TYPE_TTS) {
                // TTS slots drive their own decode loop in process_tts_slots(), never enter the shared batch
                return;
            }
            if (slot.state == SLOT_STATE_GENERATING && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
                if (!params_base.ctx_shift) {
                    // this check is redundant (for good)
                    // we should never get here, because generation should already stopped in process_token()
                    send_error(slot, "context shift is disabled", ERROR_TYPE_SERVER);
                    slot.release();
                    return;
                }

                if (mctx) {
                    // we should never reach this because params_base.ctx_shift is automatically disabled if mmproj is loaded
                    // we don't support ctx_shift because an image chunk may contains multiple tokens
                    GGML_ABORT("not supported by multimodal");
                }

                if (slot.task->is_parent() || slot.task->is_child()) {
                    send_error(slot, "context shift cannot be used for shared prompt", ERROR_TYPE_SERVER);
                    slot.release();
                    return;
                }

                // Shift context
                int n_keep = slot.task->params.n_keep < 0 ? slot.task->n_tokens() : slot.task->params.n_keep;

                if (add_bos_token) {
                    n_keep += 1;
                }

                n_keep = std::min(slot.n_ctx - 4, n_keep);

                const int n_left    = slot.prompt.n_tokens() - n_keep;
                int       n_discard = slot.task->params.n_discard ? slot.task->params.n_discard : (n_left / 2);

                // ref: https://github.com/ggml-org/llama.cpp/pull/24786
                n_discard = std::clamp(n_discard, 0, std::max(0, n_left - 1));

                SLT_WRN(slot, "slot context shift, n_keep = %d, n_left = %d, n_discard = %d\n", n_keep, n_left, n_discard);

                slot.mem.seq_rm (slot.id, n_keep            , n_keep + n_discard);
                slot.mem.seq_add(slot.id, n_keep + n_discard, slot.prompt.tokens.pos_next(), -n_discard);

                // add generated tokens to cache
                // ref: https://github.com/ggml-org/llama.cpp/pull/16818#discussion_r2473269481
                {
                    GGML_ASSERT(!slot.prompt.tokens.has_mtmd);

                    llama_tokens new_tokens = slot.prompt.tokens.get_tokens(); // copy
                    for (size_t i = n_keep + n_discard; i < new_tokens.size(); i++) {
                        new_tokens[i - n_discard] = new_tokens[i];
                    }

                    new_tokens.resize(slot.prompt.tokens.size() - n_discard);

                    slot.prompt.clear();
                    slot.prompt.tokens.insert(new_tokens);
                }

                slot.truncated = true;
            }
        });

        // start populating the batch for this iteration
        batch.clear();

        // Previous wave has joined. Rebind workers without moving shared sequence state.
        if (prefill_workers) {
            GGML_ASSERT(pending_media.empty());
            std::fill(execution_lane_ptrs.begin(), execution_lane_ptrs.end(), nullptr);
            std::vector<server_slot *> prefilling, decoding;
            for (auto & slot : slots) {
                slot.waiting_for_prefill_workspace = false;
                slot.execution_worker = -1;
                if (!slot.is_processing() || slot.capacity_parked()) continue;
                if (slot.state == SLOT_STATE_GENERATING) decoding.push_back(&slot);
                else if (slot.state == SLOT_STATE_STARTED || slot.state == SLOT_STATE_PROCESSING_PROMPT) {
                    prefilling.push_back(&slot);
                    slot.waiting_for_prefill_workspace = true;
                }
            }
            std::stable_sort(prefilling.begin(), prefilling.end(), [&](auto a, auto b) {
                if (execution_priority(*a) != execution_priority(*b))
                    return execution_priority(*a) > execution_priority(*b);
                return a->task->n_tokens() - a->prompt.n_tokens() <
                       b->task->n_tokens() - b->prompt.n_tokens();
            });
            std::vector<bool> used(execution_lanes.size(), false);
            auto assign = [&](server_slot & slot, size_t worker) {
                used[worker] = true;
                execution_lane_ptrs.at(slot.id) = execution_lanes[worker].get();
                slot.execution_worker = worker;
                slot.waiting_for_prefill_workspace = false;
            };
            std::vector<server_slot *> waiting = prefilling;
            for (size_t worker = 0; worker < (size_t) prefill_workers && !waiting.empty(); ++worker) {
                auto selected = waiting.begin();
                // Reserve the wide tier for the largest miss at the highest priority.
                // Equal-size tiers retain shortest-tail ordering.
                if (worker + 1 < (size_t) prefill_workers &&
                        execution_capacities[worker] > execution_capacities[worker + 1]) {
                    const auto priority = execution_priority(*waiting.front());
                    for (auto it = waiting.begin(); it != waiting.end() &&
                            execution_priority(**it) == priority; ++it) {
                        if ((*it)->task->n_tokens() - (*it)->prompt.n_tokens() >
                                (*selected)->task->n_tokens() - (*selected)->prompt.n_tokens()) selected = it;
                    }
                }
                assign(**selected, worker);
                waiting.erase(selected);
            }
            for (auto * slot : decoding) {
                // Prefer tiny decode workspaces; unused prefill workers can decode too.
                size_t worker = prefill_workers;
                while (worker < used.size() && used[worker]) ++worker;
                if (worker == used.size()) {
                    worker = 0;
                    while (worker < used.size() && used[worker]) ++worker;
                }
                GGML_ASSERT(worker < used.size());
                assign(*slot, worker);
            }
            for (auto * slot : prefilling) {
                if (slot->task->params.stream && slot->task->params.return_progress) {
                    send_partial_response(*slot, {}, true);
                }
            }
        }

        // track if given slot can be batched with slots already in the batch
        auto & slot_batched = batch.slot_batched;

        std::vector<server_slot *> generating;
        std::vector<server_slot *> drafting;

        // determine which slots are generating and drafting
        iterate(slots, [&](server_slot & slot) {
            if (slot.capacity_parked() || slot.state != SLOT_STATE_GENERATING || slot.task->type == SERVER_TASK_TYPE_TTS) {
                return;
            }

            if (slot.audio_stream && !slot_audio_step(slot)) {
                return; // waiting for audio, or the stream ended
            }

            // check if we can batch this slot with the previous one
            if (!slot_batched) {
                slot_batched = &slot;
            } else if (!slot_batched->can_batch_with(slot)) {
                return;
            }

            generating.push_back(&slot);

            if (spec) {
                common_speculative_get_draft_params(spec.get(), slot.id).drafting = false;

                const bool use_ckpt_tgt = ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
                const bool use_ckpt_dft = ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

                const int n_draft_max = slot.get_n_draft_max();

                if (n_draft_max > 0) {
                    GGML_ASSERT(slot.can_speculate());

                    // stale candidates: a replay never reads them, a new draft refills them
                    slot.spec_draft_q.clear();

                    if (!slot.spec_draft.empty()) {
                        // we have a previous (partial) draft to reuse
                        if (use_ckpt_tgt) {
                            GGML_ASSERT(!slot.spec_ckpt.empty());
                        }
                    } else {
                        GGML_ASSERT(slot.spec_i_batch.empty());

                        slot.spec_ckpt.update_pos(
                                slot.prompt.n_tokens(),
                                llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id),
                                llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id));

                        if (use_ckpt_dft) {
                            slot.spec_ckpt.update_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                        }

                        slot.spec_prompt = slot.prompt.tokens.get_text_tokens();

                        const bool spec_reject = slot.use_spec_rejection();

                        common_speculative_get_draft_params(spec.get(), slot.id) = {
                            /* .drafting = */ true,
                            /* .n_max    = */ n_draft_max,
                            /* .pos0     = */ slot.prompt.tokens.pos_next(),
                            /* .id_last  = */ slot.sampled,
                            /* .prompt   = */ &slot.spec_prompt,
                            /* .result   = */ &slot.spec_draft,
                            /* .result_q = */ spec_reject ? &slot.spec_draft_q : nullptr,
                            /* .temp     = */ slot.task->params.sampling.temp,
                            /* .seed     = */ slot.task->params.sampling.seed,
                        };

                        drafting.push_back(&slot);
                    }
                }
            }
        });

        // generate the actual drafts (if any)
        if (!drafting.empty()) {
            queue_tasks.yield_to_queue([&]() {
                server_roctx_scope range("glm53.mtp.draft");
                common_moe_trace_scope capture(moe_trace.get(), common_moe_trace_phase::draft);
                common_speculative_draft(spec.get());
                capture.end(0);
            });
        }

        // make checkpoints if needed
        iterate(drafting, [&](server_slot & slot) {
            auto & draft = slot.spec_draft;
            auto & ckpt  = slot.spec_ckpt;

            slot.stats.n_draft_tokens += draft.size();
            if (moe_trace) { moe_trace->proposed(draft.size()); }

            // TODO: avoid restoring the draft context and re-evaluating the drafted tokens when not needed [TAG_SPEC_AVOID_DRAFT_REEVAL]
            const bool use_ckpt_dft = ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

            if (ctx_dft) {
                if (use_ckpt_dft) {
                    GGML_ASSERT(ckpt.load_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
                }

                if (!llama_memory_seq_rm(llama_get_memory(ctx_dft), slot.id, ckpt.pos_max + 1, -1)) {
                    GGML_ABORT("failed to remove sequence %d\n", slot.id);
                }
            }

            if (!draft.empty()) {
                const bool use_ckpt_tgt =
                    ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                   (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && draft.size() > llama_n_rs_seq(ctx_tgt));

                const bool use_ckpt_dft =
                   (ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && draft.size() > llama_n_rs_seq(ctx_dft));

                if (use_ckpt_tgt) {
                    //const int64_t t_start = ggml_time_us();

                    ckpt.update_tgt(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                    //const int64_t t_total = ggml_time_us() - t_start;
                    //printf("checkpoint total: %f ms\n", t_total / 1000.0);

                    SLT_DBG(slot, "created speculative checkpoint (pos_min = %d, pos_max = %d, n_tokens = %d, size = %.3f MiB, draft = %.3f MiB)\n",
                            ckpt.pos_min, ckpt.pos_max, slot.prompt.n_tokens(),
                            (float) ckpt.size() / 1024 / 1024,
                            (float) ckpt.data_dft.size() / 1024 / 1024);
                }

                if (use_ckpt_dft) {
                    ckpt.update_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                }
            }
        });

        // update the batch with the sampled/drafted tokens
        iterate(generating, [&](server_slot & slot) {
            slot.handle_last_sampled_token(batch);
        });

        // process in chunks of params.n_batch
        int32_t n_batch  = llama_n_batch(ctx_tgt);
        int32_t n_ubatch = llama_n_ubatch(ctx_tgt);

        auto & alora_scale       = batch.alora_scale;
        auto & alora_disabled_id = batch.alora_disabled_id;

        // Keep cold prefill from holding decode behind a whole 4096-token pass.
        // Three decode-only iterations alternate with one bounded prefill slice.
        const bool responsive = model_tgt->arch == LLM_ARCH_GLM5NEXT && !generating.empty();
        const bool competing = model_tgt->arch == LLM_ARCH_GLM5NEXT &&
            std::count_if(slots.begin(), slots.end(), [](const auto & slot) {
                return slot.is_processing() && !slot.capacity_parked();
            }) > 1;
        const bool independent = !execution_lane_ptrs.empty();
        const bool prefill_turn = independent || !responsive || (++prefill_round % 4 == 0);
        // Bound first-token latency too, before any slot has started decoding.
        if (responsive || competing) {
            const int prompt_budget = prefill_workers ? prefill_workers * prefill_worker_batch :
                64 * (independent ? (int) execution_lane_ptrs.size() : 1);
            n_batch = std::min(n_batch, batch.size() + prompt_budget);
        }
        // next, batch any pending prompts without exceeding n_batch
        if (prefill_turn && (params_base.cont_batching || batch.size() == 0)) {
            bool add_ok = true; // false means the batch is full, skip remaining slots

            // Finish a cached tail before filling the batch with a cold history.
            // A slot number must not determine time-to-first-token.
            std::vector<server_slot *> prompt_order;
            for (auto & slot : slots) {
                if (slot.is_processing() && !slot.capacity_parked()) prompt_order.push_back(&slot);
            }
            std::stable_sort(prompt_order.begin(), prompt_order.end(), [&](auto a, auto b) {
                if (execution_priority(*a) != execution_priority(*b))
                    return execution_priority(*a) > execution_priority(*b);
                return a->task->n_tokens() - a->prompt.n_tokens() <
                       b->task->n_tokens() - b->prompt.n_tokens();
            });
            iterate(prompt_order, [&](server_slot & slot) {
                if (!add_ok || batch.size() >= n_batch) {
                    return; // batch is full, skip remaining slots
                }

                if (!slot.is_processing() || slot.capacity_parked() || slot.waiting_for_prefill_workspace || slot.task->type == SERVER_TASK_TYPE_TTS) {
                    return;
                }

                // check if we can batch this slot with the previous one
                if (slot_batched && !slot_batched->can_batch_with(slot)) {
                    return;
                }

                // check if this is a child slot
                if (slot.state == SLOT_STATE_WAIT_OTHER) {
                    SLT_DBG(slot, "%s", "waiting for parent slot to complete\n");
                    return;
                }

                // no prompt, the rows come from the audio stream
                if (slot.task->type == SERVER_TASK_TYPE_TRANSCRIBE_RT) {
                    return;
                }

                // this slot still has a prompt to be processed
                if (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_STARTED) {
                    const auto & input_tokens = slot.task->tokens;

                    // used to determine the number of tokens added to the batch for the current slot
                    const auto n_tokens_prev = batch.size();

                    // TODO: maybe move branch to outside of this loop in the future
                    if (slot.state == SLOT_STATE_STARTED) {
                        slot.stats.update_prompt_start();

                        slot.state = SLOT_STATE_PROCESSING_PROMPT;

                        // A text-only task (including a restored prompt page) can replace the
                        // slot's token container with one that has_mtmd=false. Re-enable the
                        // media-aware cache path before comparing prefixes or appending image
                        // placeholders for a later multimodal task on the same slot.
                        if (input_tokens.has_media()) {
                            GGML_ASSERT(mctx != nullptr);
                            GGML_ASSERT(slot.prompt.tokens.has_mtmd || !slot.prompt.tokens.has_media());
                            slot.prompt.tokens.has_mtmd = true;
                        }

                        SLT_TRC(slot, "new prompt, n_ctx_slot = %d, n_keep = %d, task.n_tokens = %d\n",
                                slot.n_ctx, slot.task->params.n_keep, slot.task->n_tokens());

                        // print prompt tokens (for debugging)
                        /*if (1) {
                            // first 16 tokens (avoid flooding logs)
                            for (int i = 0; i < std::min<int>(16, input_tokens.size()); i++) {
                                SLT_DBG(slot, "prompt token %3d: %6d '%s'\n", i, input_tokens[i], common_token_to_piece(ctx_tgt, input_tokens[i]).c_str());
                            }
                        } else {
                            // all
                            for (int i = 0; i < (int) input_tokens.size(); i++) {
                                SLT_DBG(slot, "prompt token %3d: %6d '%s'\n", i, input_tokens[i], common_token_to_piece(ctx_tgt, input_tokens[i]).c_str());
                            }
                        }*/

                        // keep track how many tokens we can reuse from the previous state
                        int n_past = 0;

                        // empty prompt passed -> release the slot and send empty response
                        if (input_tokens.empty()) {
                            SLT_WRN(slot, "%s", "empty prompt - releasing slot\n");

                            slot.print_timings();
                            send_final_response(slot);
                            slot.release();

                            return;
                        }

                        // TODO: support memory-less logits computation
                        if (slot.task->need_logits() && !llama_get_memory(ctx_tgt)) {
                            send_error(slot, "the current context does not support logits computation. skipping", ERROR_TYPE_SERVER);
                            slot.release();
                            return;
                        }

                        if (!slot.can_split()) {
                            if (slot.task->n_tokens() > n_ubatch) {
                                send_error(slot,
                                           string_format(
                                               "input (%d tokens) is too large to process. increase the physical batch "
                                               "size (current batch size: %d)",
                                               slot.task->n_tokens(), n_ubatch),
                                           ERROR_TYPE_SERVER);
                                slot.release();
                                return;
                            }

                            if (slot.task->n_tokens() > slot.n_ctx) {
                                send_error(
                                    slot,
                                    string_format(
                                        "input (%d tokens) is larger than the max context size (%d tokens). skipping",
                                        slot.task->n_tokens(), slot.n_ctx),
                                    ERROR_TYPE_EXCEED_CONTEXT_SIZE);
                                slot.release();
                                return;
                            }
                        } else {
                            if (slot.task->n_tokens() >= slot.n_ctx) {
                                send_error(slot,
                                           string_format("request (%d tokens) exceeds the available context size (%d "
                                                         "tokens), try increasing it",
                                                         slot.task->n_tokens(), slot.n_ctx),
                                           ERROR_TYPE_EXCEED_CONTEXT_SIZE);
                                slot.release();
                                return;
                            }

                            // the outputs of a decision are read from one batch
                            const int32_t n_decision_first = slot.task->type == SERVER_TASK_TYPE_DECISION ? slot.task->decision.pos_first() : -1;
                            if (n_decision_first >= 0 && slot.task->n_tokens() - n_decision_first > n_batch) {
                                send_error(slot,
                                           string_format("the question and its options (%d tokens) are too large to process. "
                                                         "increase the batch size (current batch size: %d)",
                                                         slot.task->n_tokens() - n_decision_first, n_batch),
                                           ERROR_TYPE_INVALID_REQUEST);
                                slot.release();
                                return;
                            }

                            if (slot.task->params.cache_prompt && !slot.task->is_stateless()) {
                                // reuse any previously computed tokens that are common with the new prompt
                                n_past = slot.prompt.tokens.get_common_prefix(input_tokens);

                                // the children start from the shared prefix, do not go past it
                                if (slot.task->n_tokens_shared > 0) {
                                    n_past = std::min(n_past, slot.task->n_tokens_shared);
                                }

                                // the outputs of a decision are read from the batch, so its first position must be evaluated
                                if (n_decision_first >= 0) {
                                    n_past = std::min(n_past, n_decision_first);
                                }

                                // if there is an alora invoked, don't cache after the invocation start
                                if (slot.alora_invocation_start > 0) {
                                    SLT_DBG(slot, "only caching to alora invocation start (n_past = %d, alora_invocation_start = %d)\n", n_past, slot.alora_invocation_start);
                                    n_past = std::min(n_past, slot.alora_invocation_start - 1);
                                }

                                const auto n_cache_reuse = slot.task->params.n_cache_reuse;

                                const bool can_cache_reuse =
                                    llama_memory_can_shift(llama_get_memory(ctx_tgt)) &&
                                    !slot.prompt.tokens.has_mtmd;

                                if (!can_cache_reuse && n_cache_reuse > 0) {
                                    SLT_WRN(slot, "cache reuse is not supported - ignoring n_cache_reuse = %d\n", n_cache_reuse);
                                }

                                // reuse chunks from the cached prompt by shifting their KV cache in the new position
                                if (can_cache_reuse && n_cache_reuse > 0) {
                                    GGML_ASSERT(!slot.prompt.tokens.has_mtmd);

                                    size_t head_c = n_past; // cache
                                    size_t head_p = n_past; // current prompt

                                    if (mctx) {
                                        // we should never reach this
                                        GGML_ABORT("not supported by multimodal");
                                    }

                                    SLT_DBG(slot, "trying to reuse chunks with size > %d, n_past = %d\n", n_cache_reuse, n_past);

                                    while (head_c < slot.prompt.tokens.size() &&
                                           head_p < input_tokens.size()) {

                                        size_t n_match = 0;
                                        while (head_c + n_match < slot.prompt.tokens.size() &&
                                               head_p + n_match < input_tokens.size()       &&
                                               slot.prompt.tokens[head_c + n_match] == input_tokens[head_p + n_match]) {
                                            n_match++;
                                        }

                                        if (n_match >= (size_t) n_cache_reuse) {
                                            SLT_TRC(slot, "reusing chunk with size %zu, shifting KV cache [%zu, %zu) -> [%zu, %zu)\n", n_match, head_c, head_c + n_match, head_p, head_p + n_match);
                                            //for (size_t i = head_p; i < head_p + n_match; i++) {
                                            //    SLT_DBG(slot, "cache token %3zu: %6d '%s'\n", i, prompt_tokens[i], common_token_to_piece(ctx_tgt, prompt_tokens[i]).c_str());
                                            //}

                                            const int64_t kv_shift = (int64_t) head_p - (int64_t) head_c;

                                            slot.mem.seq_rm (slot.id, head_p, head_c);
                                            slot.mem.seq_add(slot.id, head_c, head_c + n_match, kv_shift);

                                            for (size_t i = 0; i < n_match; i++) {
                                                slot.prompt.tokens.set_token(head_p + i, slot.prompt.tokens[head_c + i]);
                                                n_past++;
                                            }

                                            head_c += n_match;
                                            head_p += n_match;
                                        } else {
                                            head_c += 1;
                                        }
                                    }

                                    SLT_DBG(slot, "after context reuse, new n_past = %d\n", n_past);
                                }
                            } else {
                                // if we don't cache the prompt, we have to remove all previous tokens
                                n_past = 0;
                            }

                            llama_pos pos_next = slot.prompt.tokens.pos_next(n_past);

                            // number of cached tokens shared with the new prompt, before any checkpoint is restored
                            const int n_prefix = n_past;

                            // ref: https://github.com/ggml-org/llama.cpp/pull/24110
                            const bool has_new_tokens = (n_past < slot.task->n_tokens());

                            // the first evaluated token attends to the positions > pos_eval - n_swa, so a SWA state
                            // starting at pos_eval - n_swa + 1 is enough, while a state without SWA must end before pos_eval
                            const llama_pos pos_eval = pos_next - (has_new_tokens ? 0 : 1);

                            // a state or a checkpoint is useful when its pos_min is below this threshold
                            const auto pos_min_thold = std::max(0, n_swa > 0 ? pos_eval - n_swa + 2 : pos_eval);

                            if (n_past > 0 && n_past <= slot.prompt.n_tokens()) {
                                const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id);
                                if (pos_min == -1) {
                                    SLT_ERR(slot, "n_past = %d, slot.prompt.tokens.size() = %d, seq_id = %d, pos_min = %d\n", n_past, (int) slot.prompt.tokens.size(), slot.id, pos_min);
                                    GGML_ABORT("pos_min == -1, but n_past > 0 - should not happen: https://github.com/ggml-org/llama.cpp/pull/13833#discussion_r2116181237");
                                }

                                // when the prompt prefix does not match, print the tokens around the mismatch
                                // this is useful for debugging prompt caching
                                if (slots_debug) {
                                    const int np0 = std::max<int>(n_past - slots_n_diff, 0);
                                    const int np1 = std::min<int>(n_past + slots_n_diff + 2, std::min(slot.prompt.tokens.size(), slot.task->tokens.size()));

                                    std::stringstream ss0;
                                    std::stringstream ss1;

                                    std::stringstream st0;
                                    std::stringstream st1;

                                    ss0 << "old: ... ";
                                    ss1 << "new: ... ";

                                    for (int i = np0; i < np1; i++) {
                                        if (i == n_past) {
                                            ss0 << " | ";
                                            ss1 << " | ";
                                        }

                                        {
                                            const auto token = slot.prompt.tokens[i];
                                            const auto piece = token != LLAMA_TOKEN_NULL ? common_token_to_piece(ctx_tgt, token) : "[mtmd]";
                                            ss0 << piece;
                                            st0 << std::setw(8) << token;
                                        }

                                        {
                                            const auto token = slot.task->tokens[i];
                                            const auto piece = token != LLAMA_TOKEN_NULL ? common_token_to_piece(ctx_tgt, token) : "[mtmd]";
                                            ss1 << piece;
                                            st1 << std::setw(8) << token;
                                        }
                                    }

                                    SLT_WRN(slot, "%s\n", ss0.str().c_str());
                                    SLT_WRN(slot, "%s\n", ss1.str().c_str());

                                    SLT_WRN(slot, "%s\n", st0.str().c_str());
                                    SLT_WRN(slot, "%s\n", st1.str().c_str());
                                }

                                if (pos_min >= pos_min_thold) {
                                    // search for a context checkpoint
                                    const auto it = std::find_if(
                                        slot.prompt.checkpoints.rbegin(),
                                        slot.prompt.checkpoints.rend(),
                                        [&](const auto & cur) {
                                            // guarantee that a checkpoint will result in at least one token being processed [TAG_PROMPT_LOGITS]
                                            SLT_TRC(slot, "checking checkpoint with [%d, %d] against %d...\n", cur.pos_min, cur.pos_max, pos_min_thold);
                                            // workaround for [TAG_CHECKPOINTS_FIX_POS_MIN]
                                            if (cur.pos_max > pos_next) {
                                                return false;
                                            }
                                            return cur.pos_min < pos_min_thold || cur.pos_min == 0;
                                        }
                                    );

                                    bool do_reset = it == slot.prompt.checkpoints.rend();

                                    if (!do_reset) {
                                        // restore the context checkpoint
                                        auto restore = [&](bool draft) {
                                            auto load = [&] { return draft ? it->load_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) : it->load_tgt(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY); };
                                            bool ok = load();
                                            while (!ok && try_clear_idle_slots()) ok = load();
                                            return ok;
                                        };

                                        const bool restored = restore(false) && restore(true);
                                        if (!restored) {
                                            SLT_ERR(slot, "failed to restore context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                                                    it->pos_min, it->pos_max, it->n_tokens, (float) it->size() / 1024 / 1024);
                                            do_reset = true;
                                        } else {
                                            common_speculative_set_state(spec.get(), slot.id, it->data_spec);
                                            pos_next = std::min(pos_next, std::max(it->pos_min + 1, it->pos_max));
                                            n_past = std::min(slot.prompt.tokens.size_up_to_pos(pos_next), (size_t) it->n_tokens);
                                            SLT_TRC(slot, "restored context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", n_past = %d, size = %.3f MiB)\n",
                                                    it->pos_min, it->pos_max, it->n_tokens, n_past, (float) it->size() / 1024 / 1024);
                                        }
                                    }

                                    if (do_reset) {
                                        SLT_TRC(slot, "forcing full prompt re-processing due to lack of cache data (likely due to SWA or hybrid/recurrent memory, see %s)\n",
                                                "https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055");
                                        pos_next = 0;
                                        n_past = 0;
                                    }
                                }
                            }

                            {
                                // erase any checkpoints with pos_max > pos_next, or that cover tokens not shared with the new prompt
                                for (auto it = slot.prompt.checkpoints.begin(); it != slot.prompt.checkpoints.end();) {
                                    const auto & cur = *it;
                                    if (cur.pos_max > pos_next || cur.n_tokens > n_prefix) {
                                        SLT_TRC(slot, "erased invalidated context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", n_swa = %d, pos_next = %d, size = %.3f MiB)\n", cur.pos_min, cur.pos_max, cur.n_tokens, n_swa, pos_next, (float) cur.size() / 1024 / 1024);
                                        it = slot.prompt.checkpoints.erase(it);
                                    } else {
                                        ++it;
                                    }
                                }
                            }
                        }

                        // [TAG_PROMPT_LOGITS]
                        if (n_past == slot.task->n_tokens() && n_past > 0) {
                            SLT_WRN(slot, "need to evaluate at least 1 token for each active slot (n_past = %d, task.n_tokens() = %d)\n", n_past, slot.task->n_tokens());
                            n_past--;
                            SLT_WRN(slot, "n_past was set to %d\n", n_past);
                        }

                        slot.stats.n_prompt_cached    = n_past;
                        slot.stats.n_prompt_processed = 0;

                        metrics.add_prompt_cached(n_past);

                        slot.prompt.tokens.keep_first(n_past);

                        // this is to signal the client that the request has started processing
                        if (slot.task->params.stream) {
                            if (slot.task->params.return_progress) {
                                // send initial 0% progress update if needed
                                send_partial_response(slot, {}, true);
                            } else {
                                // otherwise, for streaming without progress, signal HTTP to send the headers (i.e. 200 status)
                                send_partial_response(slot, {}, false, true);
                            }
                        }
                    } // end of SLOT_STATE_STARTED

                    if (!slot.can_split()) {
                        // cannot fit the prompt in the current batch - will try next iter
                        if (batch.size() + slot.task->n_tokens() > n_batch) {
                            return;
                        }
                    }

                    // note: the prompt timing is advanced in post_decode(), so it does not cover
                    //       the tokens added to the batch below
                    slot.print_timings_pp();

                    // truncate any tokens that are beyond n_past for this slot
                    const llama_pos p0 = slot.prompt.tokens.pos_next();

                    SLT_TRC(slot, "cached n_tokens = %d, memory_seq_rm [%d, end)\n", slot.prompt.n_tokens(), p0);

                    if (!slot.media_offset) slot.mem.seq_rm(slot.id, p0, -1);

                    // shared prompt prefix: once it is processed, the children continue from it with their own prompt
                    bool wait_shared = false;
                    if (slot.task->n_tokens_shared > 0) {
                        const bool is_shared_done = slot.prompt.n_tokens() == slot.task->n_tokens_shared;
                        for (auto & other : slots) {
                            if (other.state != SLOT_STATE_WAIT_OTHER || other.task->id_parent != slot.task->id) {
                                continue;
                            }
                            if (is_shared_done) {
                                SLT_TRC(slot, " - copying shared prompt (%d tokens) to child %d\n", slot.prompt.n_tokens(), other.id);
                                slot.copy_prompt_to(other);
                                other.state = SLOT_STATE_STARTED;
                            } else {
                                wait_shared = true;
                            }
                        }
                    }

                    // If using an alora, there may be uncached tokens that come
                    // before the invocation sequence. When this happens, the
                    // tokens before the invocation sequence need to be
                    // processed without the adapter in a separate batch, then
                    // the adapter needs to be enabled for the remaining tokens.
                    if (lora_all_alora(slot.lora) && slot.alora_invocation_start - 1 > slot.prompt.n_tokens()) {
                        SLT_DBG(slot, "processing pre-alora tokens without the adapter (n_tokens = %d, alora_invocation_start = %d)\n", slot.prompt.n_tokens(), slot.alora_invocation_start);
                        const auto & enabled_loras = lora_get_enabled_ids(slot.lora);
                        GGML_ASSERT(enabled_loras.size() == 1);
                        alora_scale = slot.lora[enabled_loras[0]].scale;
                        slot.lora[enabled_loras[0]].scale = 0.0f;
                        alora_disabled_id = enabled_loras[0];
                    }

                    bool do_checkpoint = params_base.n_ctx_checkpoints > 0;

                    // make checkpoints only for completion tasks
                    do_checkpoint = do_checkpoint && slot.task->type == SERVER_TASK_TYPE_COMPLETION;

                    // make a checkpoint of the parts of the memory that cannot be rolled back.
                    // checkpoints are created only if:
                    // - the model does not support partial sequence removal
                    // - the model uses SWA (and we are not using `swa_full`)
                    // - the model supports partial sequence removal but only up to a fixed bound
                    do_checkpoint = do_checkpoint && (
                            ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                            ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS ||
                            n_swa > 0);

                    // TODO: do the same for all models, then remove process_mtmd_chunk()
                    if (use_mixed_batch() && !slot.can_split() && input_tokens.has_mtmd) {
                        if (!add_prompt_mixed(slot)) {
                            return;
                        }
                    }

                    bool has_mtmd = false;

                    // check if we should process the mtmd chunk
                    while (true) {
                        auto cur_token_idx = slot.prompt.n_tokens();
                        if (
                            cur_token_idx >= slot.task->n_tokens() ||
                            input_tokens[cur_token_idx] != LLAMA_TOKEN_NULL // encountered a text token
                        ) {
                            break;
                        }

                        if (independent && (!slot.mbatch ||
                                !mtmd_batch_get_output_embd(slot.mbatch.get(), input_tokens.find_chunk(cur_token_idx).get()))) {
                            if (!slot.media_preparation.valid()) {
                                auto held_task = slot.task;
                                auto * projector = slot.mctx;
                                slot.media_preparation = std::async(std::launch::async,
                                    [this, held_task, projector, cur_token_idx]() {
                                        // One projector owner; at most one pending preparation per active slot.
                                        media_preparations_started++;
                                        std::lock_guard<std::mutex> lock(media_preparation_mutex);
                                        mtmd::batch_ptr prepared(mtmd_batch_init(projector));
                                        const auto & chunk = held_task->tokens.find_chunk(cur_token_idx);
                                        if (mtmd_batch_add_chunk(prepared.get(), chunk.get()) != 0 ||
                                                mtmd_batch_encode(prepared.get()) != 0) {
                                            throw std::runtime_error("media preparation failed");
                                        }
                                        media_preparations_completed++;
                                        return prepared;
                                    });
                                return;
                            }
                            if (slot.media_preparation.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
                                return;
                            }
                            try {
                                slot.mbatch = slot.media_preparation.get();
                            } catch (const std::exception &) {
                                send_error(slot, "media preparation failed", ERROR_TYPE_SERVER);
                                slot.release();
                                return;
                            }
                        }

                        if (independent && competing) {
                            auto media = prepare_media_wave(slot, cur_token_idx, llama_n_ubatch(execution_lane_ptrs.at(slot.id)));
                            if (media) {
                                pending_media.push_back(std::move(media));
                                return; // dispatched together with the other slots below
                            }
                        }

                        // process the mtmd chunk
                        // note: it submits its own decode, potentially be async
                        //       so the timing is queued and flushed on the next sync
                        metrics_pre_decode();

                        // encode on the worker thread, so we can still handle metrics tasks
                        size_t n_tokens_out = 0;
                        int32_t res = 0;
                        size_t media_offset = slot.media_offset;
                        queue_tasks.yield_to_queue([&]() {
                            res = process_mtmd_chunk(slot, slot.mbatch, cur_token_idx, n_tokens_out,
                                model_tgt->arch == LLM_ARCH_GLM5NEXT ? ((responsive || competing) ? 64 : llama_n_ubatch(ctx_tgt)) : 0, media_offset);
                        });
                        slot.media_offset = media_offset;

                        if (res != 0 && res != 2) {
                            SLT_ERR(slot, "failed to process mtmd chunk, res = %d\n", res);
                            send_error(slot, "failed to process mtmd chunk", ERROR_TYPE_SERVER);
                            slot.release();
                            return; // the slot is done, skip it entirely
                        }

                        metrics_queue_prompt(n_tokens_out);
                        slot.stats.n_prompt_processed += n_tokens_out;
                        slot.stats.update_prompt_last();

                        if (res == 2) return; // decode slots get a turn before the next media slice

                        // add the mtmd chunk to cache
                        {
                            const auto & chunk = input_tokens.find_chunk(cur_token_idx);
                            // the chunk is already in the KV cache at this point, so we don't need to keep its data around
                            slot.prompt.tokens.push_back_placeholder(chunk.get());
                        }

                        // Media chunks are atomic and may cross the nominal page span.
                        // Capture immediately while this is still the hybrid-memory tail.
                        capture_prompt_page(slot);

                        has_mtmd = true;
                        if (responsive || competing) return; // one media slice, never a whole image history
                    }

                    const auto & spans = slot.task->params.message_spans;
                    const auto last_user_pos = spans.last_user_message_pos();
                    const auto * next_page_boundary = next_prompt_page_boundary(slot);

                    const int32_t n_decision_first = slot.task->type == SERVER_TASK_TYPE_DECISION ? slot.task->decision.pos_first() : -1;

                    // Each private lane gets a bounded prompt slice in the same dispatch.
                    const int32_t slot_limit = independent ? std::min(n_batch, batch.size() + (int) (prefill_workers ? prefill_worker_batch : llama_n_ubatch(execution_lane_ptrs.at(slot.id)))) : n_batch;
                    while (slot.prompt.n_tokens() < slot.task->n_tokens() && batch.size() < slot_limit) {
                        // get next token to process
                        llama_token cur_tok = input_tokens[slot.prompt.n_tokens()];
                        if (cur_tok == LLAMA_TOKEN_NULL) {
                            break; // end of text chunk
                        }

                        // stop at the end of the shared prefix, the children are started from this state
                        if (wait_shared && slot.prompt.n_tokens() == slot.task->n_tokens_shared) {
                            break;
                        }

                        // the outputs of a decision are read from one batch, do not split them
                        if (slot.prompt.n_tokens() == n_decision_first && batch.size() + slot.task->n_tokens() - n_decision_first > n_batch) {
                            break;
                        }

                        // if this is an alora request with pre-invocation
                        // tokens that are not cached, we need to stop filling
                        // this batch at those pre-invocation tokens.
                        if (alora_scale > 0 && slot.prompt.n_tokens() == slot.alora_invocation_start - 1) {
                            SLT_DBG(slot, "stop prompt batch filling at (n_tokens = %d, alora_invocation_start = %d)\n", slot.prompt.n_tokens(), slot.alora_invocation_start);
                            break;
                        }

                        // embedding requires all tokens in the batch to be output;
                        // MTP also wants logits at every prompt position so the
                        // streaming hook can mirror t_h_nextn into ctx_dft.
                        add_ok &= batch.add(slot.id,
                            cur_tok,
                            /* pos       = */ slot.prompt.tokens.pos_next() + slot.task->hidden_state_position_offset,
                            /* output    = */ slot.task->need_embd(),
                            /* is_prompt = */ true);
                        if (!slot.task->decision.order.empty()) {
                            batch.set_decision_order(batch.size() - 1, slot.task->decision.order[slot.prompt.n_tokens()]);
                        }
                        slot.prompt.tokens.push_back(cur_tok);

                        // A hybrid range-state page must be captured at its exact recurrent tail.
                        // Stop the prompt batch here so post_decode() can serialize that boundary.
                        if (next_page_boundary && (size_t) slot.prompt.n_tokens() == next_page_boundary->token_count) {
                            break;
                        }

                        // break at the last user message, or at user messages at least min step past the last checkpoint
                        if (do_checkpoint && spans.is_user_start(slot.prompt.n_tokens())) {
                            const auto pos = slot.prompt.n_tokens();
                            const auto & checkpoints = slot.prompt.checkpoints;

                            if (pos == last_user_pos || checkpoints.empty() || pos > checkpoints.back().n_tokens + params_base.checkpoint_min_step) {
                                break;
                            }
                        }

                        // process the last few tokens of the prompt separately in order to allow for a checkpoint to be created.
                        // create checkpoints that many tokens before the end of the prompt:
                        //  - 4 + n_ubatch
                        //  - 4
                        // ref: https://github.com/ggml-org/llama.cpp/pull/20288
                        if (do_checkpoint) {
                            static const int checkpoint_offsets[] = {4 + n_ubatch, 4};

                            bool should_break = false;
                            for (int offset : checkpoint_offsets) {
                                const int n_last = std::min(n_batch, offset);
                                if (slot.task->n_tokens() == slot.prompt.n_tokens() + n_last) {
                                    should_break = true;
                                    break;
                                }
                            }
                            if (should_break) {
                                break;
                            }
                        }
                    }

                    // the number of tokens added to the batch for the current slot
                    const auto n_tokens_cur = batch.size() - n_tokens_prev;

                    // Exact cache checkpoints split the last few tokens across
                    // iterations. Do not fill those short-tail batches with cold work.
                    if (!independent && model_tgt->arch == LLM_ARCH_GLM5NEXT && n_tokens_cur > 0 &&
                            slot.task->n_tokens() - slot.prompt.n_tokens() <= 64) {
                        add_ok = false;
                    }

                    const auto n_tokens_start = slot.prompt.n_tokens() - n_tokens_cur;

                    const bool near_prompt_end = slot.task->n_tokens() < slot.prompt.n_tokens() + n_ubatch;

                    const bool is_user_start = spans.is_user_start(n_tokens_start);
                    const bool is_last_user_message = n_tokens_start == last_user_pos;

                    // entire prompt has been processed
                    if (slot.prompt.n_tokens() == slot.task->n_tokens()) {
                        slot.state = SLOT_STATE_DONE_PROMPT;

                        GGML_ASSERT(batch.size() > 0);

                        // extract the logits only for the last token
                        batch.set_output(batch.size() - 1, true);

                        slot.stats.n_gen = 0;
                        slot.i_batch     = batch.size() - 1;

                        slot.init_sampler();

                        // Submit these logits now; another prompt's text, image
                        // evaluation or cache restore must not hold the first token.
                        if (!independent && model_tgt->arch == LLM_ARCH_GLM5NEXT) add_ok = false;
                    } else {
                        // skip ordinary mid-prompt checkpoints, unless the batch starts a user
                        // message or we are near the end of the prompt
                        if (!is_user_start && !near_prompt_end) {
                            do_checkpoint = false;
                        }
                    }

                    const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id);
                    const auto pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id);

                    // nothing to checkpoint yet
                    // TODO: is this check needed?
                    if (do_checkpoint && pos_min < 0) {
                        do_checkpoint = false;
                    }

                    // do not checkpoint after mtmd chunks
                    do_checkpoint = do_checkpoint && !has_mtmd;

                    // no need to create checkpoints that are too close together, unless it's the last user message
                    do_checkpoint = do_checkpoint && (
                            slot.prompt.checkpoints.empty() ||
                            is_last_user_message || near_prompt_end ||
                            n_tokens_start > slot.prompt.checkpoints.back().n_tokens + params_base.checkpoint_min_step);
                    SLT_DBG(slot, "main/do_checkpoint = %s, pos_min = %d, pos_max = %d\n", do_checkpoint ? "yes" : "no", pos_min, pos_max);

                    // note: we create the checkpoint before calling llama_decode(), so the current batch is not
                    //       yet processed and therefore it is not part of the checkpoint.
                    if (do_checkpoint) {
                        create_checkpoint(slot, n_tokens_cur, pos_min, pos_max);
                    }
                }

                if (!slot_batched) {
                    slot_batched = &slot;
                }
            });
        }
    }

    // see https://github.com/ggml-org/llama.cpp/pull/29969
    // TODO @ngxson : maybe remove this once we use "mixed" batch everywhere
    bool use_mixed_batch() const {
        return !llama_get_memory(ctx_tgt) || !llama_get_causal_attn(ctx_tgt);
    }

    // add the rest of the prompt to the batch, the mtmd chunks are added as embeddings next to the text tokens
    // the caller makes sure that it fits in the batch
    // returns false on error, the slot is then released
    bool add_prompt_mixed(server_slot & slot) {
        const auto & input_tokens = slot.task->tokens;
        const auto n_tokens_prev = batch.size();

        while (slot.prompt.n_tokens() < slot.task->n_tokens()) {
            const auto cur_token_idx = slot.prompt.n_tokens();
            const llama_token cur_tok = input_tokens[cur_token_idx];

            if (cur_tok != LLAMA_TOKEN_NULL) {
                const bool add_ok = batch.add(slot.id,
                    cur_tok,
                    /* pos       = */ slot.prompt.tokens.pos_next(),
                    /* output    = */ slot.need_embd(),
                    /* is_prompt = */ true);
                GGML_ASSERT(add_ok);
                if (!slot.task->decision.order.empty()) {
                    batch.set_decision_order(batch.size() - 1, slot.task->decision.order[cur_token_idx]);
                }
                slot.prompt.tokens.push_back(cur_tok);
                continue;
            }

            const auto & chunk = input_tokens.find_chunk(cur_token_idx);

            float * embd = slot.mbatch ? mtmd_batch_get_output_embd(slot.mbatch.get(), chunk.get()) : nullptr;
            if (!embd) {
                // encode on the worker thread, so we can still handle metrics tasks
                int32_t res = 0;
                queue_tasks.yield_to_queue([&]() {
                    res = encode_mtmd_chunk(slot, slot.mbatch, cur_token_idx);
                });
                embd = res == 0 ? mtmd_batch_get_output_embd(slot.mbatch.get(), chunk.get()) : nullptr;
            }

            if (!embd || !add_mtmd_chunk(slot, chunk.get(), embd, batch)) {
                SLT_ERR(slot, "%s", "failed to process mtmd chunk\n");
                // the batch must not keep the entries of a released slot
                batch.truncate(n_tokens_prev);
                send_error(slot, "failed to process mtmd chunk", ERROR_TYPE_SERVER);
                slot.release();
                return false;
            }

            slot.prompt.tokens.push_back_placeholder(chunk.get());
        }

        return true;
    }

    std::vector<std::pair<llama_context *, llama_batch>> media_wave_batches() {
        std::vector<std::pair<llama_context *, llama_batch>> result;
        for (const auto & media : pending_media) {
            result.emplace_back(execution_lane_ptrs.at(media->slot_id), media->slice);
        }
        return result;
    }

    void commit_media_wave() {
        for (const auto & media : pending_media) {
            auto & slot = slots.at(media->slot_id);
            slot.media_offset += media->slice.n_tokens;
            slot.stats.n_prompt_processed += media->slice.n_tokens;
            slot.stats.update_prompt_last();
            metrics_queue_prompt(media->slice.n_tokens);
            if (slot.media_offset == media->total_tokens) {
                const auto & chunk = slot.task->tokens.find_chunk(media->chunk_index);
                slot.prompt.tokens.push_back_placeholder(chunk.get());
                slot.media_offset = 0;
                capture_prompt_page(slot);
            }
            if (slot.task->params.stream && slot.task->params.return_progress) {
                send_partial_response(slot, {}, true);
            }
        }
        pending_media.clear();
    }

    // Called only on the coordinator, after workers/metadata readers have joined.
    // Metadata requests read this cached value; they never race lane allocation.
    void refresh_execution_memory_snapshot() {
        const auto rows = [](const llama_context * context, bool shared) {
            json result = json::array();
            for (const auto & [buft, bytes] : llama_get_memory_breakdown(context)) {
                json row{{"buffer_type", ggml_backend_buft_name(buft)}, {"compute_bytes", bytes.compute}};
                if (shared) {
                    row["model_bytes"] = bytes.model;
                    row["context_bytes"] = bytes.context;
                }
                result.push_back(std::move(row));
            }
            return result;
        };
        const auto route_capture = [](const llama_context * context) {
            const auto memory = context->get_route_capture_memory();
            return json{{"buffer_type", memory.first}, {"bytes", memory.second}};
        };
        json lanes = json::array();
        for (size_t i = 0; i < execution_lanes.size(); ++i) {
            lanes.push_back({{"lane", i}, {"batch_capacity", llama_n_ubatch(execution_lanes[i].get())},
                {"logical_batch_capacity", llama_n_batch(execution_lanes[i].get())},
                {"buffers", rows(execution_lanes[i].get(), false)},
                {"route_capture_host", route_capture(execution_lanes[i].get())}});
        }
        const int64_t now = ggml_time_us();
        if (model_tgt->arch == LLM_ARCH_GLM5NEXT && params_base.kv_unified &&
                (!execution_kv_occupancy_us || now - execution_kv_occupancy_us >= 2000000)) {
            std::vector<std::pair<llama_seq_id, uint32_t>> all, active;
            uint64_t logical_cells = 0, active_logical_cells = 0;
            uint32_t logical_sequences = 0;
            for (int i = 0; i < params_base.n_parallel; ++i) {
                all.emplace_back(i, 0); // zero budgets count only physically present cells
                logical_cells += slots[i].prompt.n_tokens();
                logical_sequences += slots[i].prompt.n_tokens() > 0;
                if (slots[i].is_processing() && !slots[i].capacity_parked()) {
                    active.emplace_back(i, 0);
                    active_logical_cells += slots[i].prompt.n_tokens();
                }
            }
            execution_kv_occupancy = {
                {"capacity_positions", llama_n_ctx(ctx_tgt)},
                {"logical_cells", logical_cells},
                {"logical_sequences", logical_sequences},
                {"active_sequences", active.size()},
                {"active_logical_cells", active_logical_cells},
                {"physical_cells", llama_glm5next_reserved_cells(ctx_tgt, all)},
                {"active_physical_cells", active.empty() ? 0 : llama_glm5next_reserved_cells(ctx_tgt, active)},
                {"sample_monotonic_us", now},
            };
            execution_kv_occupancy_us = now;
        }
        execution_memory_snapshot = {
            {"kv_occupancy", execution_kv_occupancy},
            {"sample", ++execution_memory_sample}, {"logical_slots", params_base.n_parallel},
            {"owner", rows(ctx_tgt, true)}, {"route_capture_host", route_capture(ctx_tgt)},
            {"lanes", std::move(lanes)},
            {"includes_mmproj", false}, {"includes_driver", false}, {"includes_output_buffers", false},
        };
    }

    void record_execution_wave(const llama_batch & batch_view, int off) {
        refresh_execution_memory_snapshot();
        const size_t count = ctx_tgt->get_last_execution_wave_lanes();
        if (count) {
            execution_waves[std::min<size_t>(8, count) - 1]++;
            std::unordered_set<llama_seq_id> prompt_sequences;
            for (int i = off; i < off + batch_view.n_tokens; ++i) {
                if (batch.tokens[i].is_prompt) prompt_sequences.insert(batch.tokens[i].id_slot);
            }
            for (const auto & media : pending_media) prompt_sequences.insert(media->slot_id);
            if (count > 1 && !pending_media.empty()) execution_media_waves++;
            if (!prompt_sequences.empty()) {
                execution_prefill_waves[std::min<size_t>(8, prompt_sequences.size()) - 1]++;
                if (prompt_sequences.size() < count) execution_mixed_waves++;
            }
        } else {
            const auto & reason = ctx_tgt->get_execution_fallback();
            if (reason == "single_sequence") execution_fallback_single++;
            else if (reason == "lane_batch_capacity") execution_fallback_capacity++;
            else if (reason == "pooled_key_rebuild") execution_fallback_rebuild++;
            else execution_fallback_other++;
        }
    }

    // returns true = success ; false = retry with smaller batch size
    // throw std::runtime_error on fatal error
    bool decode(int32_t & n_batch, int32_t off, llama_batch & batch_view) {
        SRV_DBG("n_batch (effective) = %d, off = %d\n", n_batch, off);

        metrics_pre_decode();

        if (batch.size() == 0) {
            SRV_WRN("%s", "no tokens to decode\n");

            if (++n_empty_consecutive > 3) {
                GGML_ABORT("fatal error - please provide logs and repro in %s\n", "https://github.com/ggml-org/llama.cpp/pull/20277");
            }

            return true; // nothing to decode
        } else {
            n_empty_consecutive = 0;
        }

        // TODO @ngxson : dft model may have different n_embd than the tgt model, so we check & reject if that's the case
        // this case is not currently used by any models, but may need to be supported in the future
        if (spec && batch.has_embd()) {
            if (llama_model_n_embd_inp(model_dft) != llama_model_n_embd_inp(model_tgt)) {
                SRV_ERR("%s", "unsupported batch.has_embd + spec case\n");
                throw std::runtime_error("unsupported batch.has_embd + spec case");
            }
        }

        bool has_output = false;
        for (int i = off; i < off + batch.view.size(); ++i) {
            has_output |= batch.tokens[i].output;
        }

        // yield to the queue, so we can still handle metrics tasks while decoding
        // note: the sync is done here too, so that the wait is also covered by the yield
        int ret = 0;
        bool capture_prompt = false, capture_replay = false;
        if (moe_trace) {
            for (int i = off; i < off + batch.view.size(); ++i) {
                capture_prompt |= batch.tokens[i].is_prompt;
                capture_replay |= slots[batch.tokens[i].id_slot].spec_is_replay;
            }
        }
        if (!pending_media.empty() && (off != 0 || batch_view.n_tokens != batch.size())) {
            throw std::runtime_error("media wave requires one bounded text batch");
        }
        auto media = media_wave_batches();
        std::set<llama_seq_id> pending_prefill;
        std::set<llama_seq_id> decoders;
        std::set<llama_seq_id> stopped;
        std::set<llama_seq_id> completed_prefills;
        std::map<llama_seq_id, uint32_t> completed_steps;
        for (int i = off; i < off + batch_view.n_tokens; ++i) {
            const auto & token = batch.tokens[i];
            const auto & slot = slots[token.id_slot];
            if (token.is_prompt) pending_prefill.insert(slot.id);
            else if (slot.state == SLOT_STATE_GENERATING && !slot.can_speculate()) decoders.insert(slot.id);
        }
        for (const auto & item : media) pending_prefill.insert(item.second.seq_id[0][0]);
        const bool contains_prefill = !pending_prefill.empty();
        bool closing_window = false;
        auto complete = [&](llama_context * lane, llama_batch & next) {
            const auto id = next.seq_id[0][0];
            auto & slot = slots[id];
            int output_row = 0;
            if (!decoders.count(id)) {
                pending_prefill.erase(id);
                if (slot.state != SLOT_STATE_DONE_PROMPT || !next.token || next.embd ||
                        !slot.task->need_sampling() || slot.task->params.prefill_only ||
                        slot.task->is_parent() || slot.task->is_child() || slot.can_speculate()) return false;
                // Preserve the prompt endpoint before this worker starts decoding.
                // The executor holds shared-memory planning while calling us.
                capture_prompt_page(slot, true);
                slot.stats.n_prompt_processed += next.n_tokens;
                slot.stats.set_prompt_last(ggml_time_us());
                completed_prefills.insert(id);
                ++execution_prompt_handoffs;
                if (!pending_prefill.empty()) ++execution_prompt_handoffs_while_prefill_pending;
                slot.state = SLOT_STATE_GENERATING;
                decoders.insert(id);
                output_row = next.n_tokens - 1;
            } else if (next.n_tokens != 1 || !next.token) {
                throw std::runtime_error("decode continuation requires a single token");
            }
            auto * original = slot.ctx_tgt;
            slot.ctx_tgt = lane;
            bool keep_going;
            try { keep_going = sample_completed_slot(slot, output_row); }
            catch (...) { slot.ctx_tgt = original; throw; }
            slot.ctx_tgt = original;
            ++completed_steps[id];
            ++execution_decode_completions;
            if (!pending_prefill.empty()) ++execution_decode_while_prefill;
            if (!keep_going) {
                stopped.insert(id);
                return false;
            }
            // The sampled token remains pending when closing this window.
            // Ordinary pre_decode will append it after global cache maintenance.
            closing_window |= completed_steps[id] >= execution_continuation_budget() ||
                    (contains_prefill && pending_prefill.empty()) ||
                    queue_tasks.has_pending_control_tasks() ||
                    slot.prompt.tokens.pos_next() + 1 >= slot.n_ctx;
            if (closing_window) return false;
            ++execution_decode_resubmissions;
            next.n_tokens = 1;
            next.token[0] = slot.sampled;
            next.pos[0] = slot.prompt.tokens.pos_next();
            next.logits[0] = true;
            slot.prompt.tokens.push_back(slot.sampled);
            return true;
        };
        auto compute = [&]() {
            server_roctx_scope range("glm53.target");
            common_moe_trace_scope capture(moe_trace.get(), common_moe_trace_phase::target,
                                           batch_view.n_tokens, capture_prompt, capture_replay);
            const bool modern_batch = batch.has_embd() || std::any_of(
                batch.view.tokens.begin(), batch.view.tokens.end(),
                [](const common_batch::token & token) { return token.decision_order != 0; });
            if (execution_lane_ptrs.empty() || modern_batch) {
                for (const auto & item : media) {
                    ret = llama_decode(ctx_tgt, item.second);
                    llama_synchronize(ctx_tgt);
                    if (ret) break;
                }
                if (!ret) ret = llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch.view.get());
                if (ret && !media.empty()) ret = -3;
            } else {
                ret = ctx_tgt->decode_with_execution_lanes(batch_view, execution_lane_ptrs, media,
                    complete, [&](const std::function<void()> & wait) { queue_tasks.yield_to_queue([&]() { wait(); }); });
                record_execution_wave(batch_view, off);
            }
            if (ret == 0 && has_output) llama_synchronize(ctx_tgt);
            capture.end(ret, ret == 0 && has_output);
        };
        // Completion samples/streams on this thread. The read-only queue worker
        // runs only during waits, so it cannot observe a half-mutated slot.
        if (execution_lane_ptrs.empty()) queue_tasks.yield_to_queue(compute);
        else compute();

        if (ret == 0) {
            for (const auto id : stopped) {
                auto & slot = slots[id];
                slot.print_timings();
                send_final_response(slot);
                slot.release();
            }
        }

        if (ret == 0) commit_media_wave();
        else pending_media.clear();

        if (ret != 0) {
            {
                std::string err;

                if (n_batch == 1 && ret == 1) {
                    // TODO: try to terminate only the largest active slot/sequence and continue with the rest
                    //       need to remove the tokens from the current batch too
                    err = "Context size has been exceeded.";
                }

                if (ret == -1) {
                    err = "Invalid input batch.";
                }

                if (ret < -1) {
                    // TODO: update slot state based on llama_memory_seq_pos_min() and llama_memory_seq_pos_max()
                    err = "Compute error.";
                }

                // TODO: handle ret == 2 (abort) when we start aborting

                if (!err.empty()) {
                    SRV_ERR("%s off = %d, n_batch = %d, ret = %d\n", err.c_str(), off, n_batch, ret);

                    for (auto & slot : slots) {
                        if (slot.is_processing()) {
                            send_error(slot, err);
                            slot.release();

                            // note: it's complicated to keep track of how much of the current batch has been
                            //       processed before the error occurred, so we simply clear the entire context
                            slot.prompt_clear();
                        }
                    }

                    // stop, do not retry with smaller batch size
                    throw std::runtime_error(err);
                }
            }

            // retry with half the batch size to try to find a free slot in the KV cache
            if (!try_clear_idle_slots()) {
                n_batch /= 2;
            }

            SRV_WRN("failed to find free space in the KV cache, retrying with smaller batch size, off = %d, n_batch = %d, ret = %d\n", off, n_batch, ret);

            return false; // retry with the updated n_batch
        } else {
            // success, apply batch metrics
            metrics_post_decode(off, batch_view.n_tokens, has_output, completed_prefills);
        }

        // TODO: avoid restoring the draft context and re-evaluating the drafted tokens when not needed [TAG_SPEC_AVOID_DRAFT_REEVAL]
        //       for now, always re-evaluate for simplicity
        //       ref: https://github.com/ggml-org/llama.cpp/pull/22728#issuecomment-4400925384
        if (spec && batch.slot_batched && batch.slot_batched->can_speculate()) {
            bool ok = true;
            queue_tasks.yield_to_queue([&]() {
                server_roctx_scope range("glm53.mtp.process");
                common_moe_trace_scope capture(moe_trace.get(), common_moe_trace_phase::process,
                                               batch.view.size(), capture_prompt, capture_replay);
                ok = common_speculative_process(spec.get(), batch.view);
                capture.end(ok ? 0 : -1);
            });

            if (!ok) {
                SRV_ERR("%s", "failed to process speculative batch\n");

                // TODO: handle error
                throw std::runtime_error("failed to process speculative batch");
            }
        }

        // handle `n_cmpl > 1` tasks - when the main prompt is processed, activate all child tasks too
        for (auto & slot : slots) {
            if (slot.state == SLOT_STATE_DONE_PROMPT && slot.task->is_parent()) {
                std::vector<server_slot *> children;
                for (auto & other : slots) {
                    if (other.state == SLOT_STATE_WAIT_OTHER && slot.task->id == other.task->id_parent) {
                        children.push_back(&other);
                    }
                }

                // all children slots should already launched by launch_slots_with_parent_task()
                // copy state to the child slots
                for (auto & child : children) {
                    SLT_TRC(slot, " - copying state to child %d\n", child->id);

                    GGML_ASSERT(child->state == SLOT_STATE_WAIT_OTHER);
                    // children with their own prompt are started at the end of the shared prefix
                    GGML_ASSERT(slot.task->n_tokens_shared == 0);

                    slot.copy_state_to(*child);
                    child->state = SLOT_STATE_DONE_PROMPT;
                }
            }
        }

        return true;
    }

    // Caller owns slot mutation. Releasing a stopped slot is deliberately
    // separate because it can mutate shared KV and must wait for all lanes.
    bool sample_completed_slot(server_slot & slot, int tok_idx) {
        llama_token id;
        {
            scoped_timer timer(t_sampl, n_sampl);
            id = common_sampler_sample(slot.smpl.get(), slot.ctx_tgt, tok_idx);
        }

        slot.i_batch = -1;

        common_sampler_accept(slot.smpl.get(), id, true);

        // here we have synchronized the llama_context (due to the sampling above), so we can do time measurement
        const int64_t t_now = ggml_time_us();

        slot.stats.n_gen += 1;

        if (slot.stats.n_gen == 1) {
            slot.stats.update_prompt_last();
            slot.t_print_last = t_now;
            slot.n_gen_last = 0;
        }

        slot.stats.update_gen_last();

        completion_token_output result;
        result.tok          = id;
        result.text_to_send = common_token_to_piece(slot.ctx_tgt, result.tok, (params_base.special || slot.task->params.sampling.preserved_tokens.count(result.tok)));
        result.prob         = 1.0f; // TODO: set it here instead of doing inside populate_token_probs

        if (slot.task->params.sampling.n_probs > 0) {
            populate_token_probs(slot, result, slot.task->params.post_sampling_probs, params_base.special, tok_idx);
        }

        if (!process_token(result, slot)) {
            return false;
        }

        slot.print_timings_tg();
        return true;
    }

    void post_decode(int32_t n_batch_tokens, int32_t off) {
        // for checking if a given batch index is inside batch_view
        auto is_inside_view = [&](int32_t idx) {
            return idx >= off && idx < off + n_batch_tokens;
        };

        // TODO @ngxson : it's tricky to make sub-batch compatible with common_sampler_sample_and_accept_n,
        // so for now we will throw an error in this case: https://github.com/ggml-org/llama.cpp/issues/24840
        iterate(slots, [&](server_slot & slot) {
            for (auto & i : slot.spec_i_batch) {
                if (!is_inside_view(i)) {
                    throw std::runtime_error(string_format("speculative batch index %d is not inside the current sub-batch [%d, %d)", i, off, off + n_batch_tokens));
                }
            }
        });

        iterate(slots, [&](server_slot & slot) {
            if (!slot.task || !prompt_page_cache_eligible(slot, *slot.task) ||
                    slot.prompt.tokens.empty() || slot.prompt.tokens.pos_next() <= 0) {
                return;
            }

            bool decoded_prompt_tail = false;
            for (int32_t i = off; i < off + n_batch_tokens; ++i) {
                const auto & token = batch.tokens[i];
                if (token.id_slot == slot.id && token.is_prompt &&
                        token.pos[0] + 1 == slot.prompt.tokens.pos_next()) {
                    decoded_prompt_tail = true;
                    break;
                }
            }
            if (!decoded_prompt_tail) {
                return;
            }

            const bool penultimate = slot.prompt.tokens.size() + 1 == slot.task->tokens.size();
            if (prompt_page_cache_boundary(slot)) {
                capture_prompt_page(slot, penultimate);
            } else if (penultimate || slot.state == SLOT_STATE_DONE_PROMPT) {
                capture_prompt_page(slot, true);
            }
        });

        auto accept_special_token = [&](server_slot & slot, llama_token token) {
            return params_base.special ||
                slot.task->params.sampling.preserved_tokens.find(token) != slot.task->params.sampling.preserved_tokens.end();
        };

        iterate(slots, [&](server_slot & slot) {
            if (slot.capacity_parked()) return;
            // optionally send prompt processing progress
            if (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_DONE_PROMPT) {
                if (slot.task->params.stream && slot.task->params.return_progress) {
                    send_partial_response(slot, {}, true);
                }
            }

            if (!is_inside_view(slot.i_batch)) {
                // the required token not in this sub-batch, skip
                return;
            }

            if (slot.state == SLOT_STATE_DONE_PROMPT) {
                if (slot.task->type == SERVER_TASK_TYPE_HIDDEN_STATE) {
                    send_hidden_states(slot);
                    slot.release();
                    slot.i_batch = -1;
                    return;
                }

                if (slot.task->type == SERVER_TASK_TYPE_EMBEDDING) {
                    // prompt evaluated for embedding
                    send_embedding(slot, batch.view);
                    slot.release();
                    slot.i_batch = -1;
                    return;
                }

                if (slot.task->type == SERVER_TASK_TYPE_RERANK) {
                    send_rerank(slot, batch.view);
                    slot.release();
                    slot.i_batch = -1;
                    return;
                }

                if (slot.task->type == SERVER_TASK_TYPE_DECISION) {
                    send_decision(slot, batch.view, slot.i_batch - off);
                    slot.release();
                    slot.i_batch = -1;
                    return;
                }

                if (slot.task->params.prefill_only) {
                    llama_synchronize(slot.ctx_tgt);
                    slot.stats.update_prompt_last();
                    slot.stop = STOP_TYPE_EOS;
                    slot.has_next_token = false;
                    slot.print_timings();
                    send_final_response(slot);
                    slot.release();
                    slot.i_batch = -1;
                    return;
                }

                GGML_ASSERT(slot.task->need_sampling());

                // prompt evaluated for next-token prediction
                slot.state = SLOT_STATE_GENERATING;

                if (slot.can_speculate()) {
                    common_speculative_begin(spec.get(), slot.id, slot.prompt.tokens.get_text_tokens());
                }
            } else if (slot.state != SLOT_STATE_GENERATING) {
                return;
            }

            if (slot.can_speculate() && !slot.spec_draft.empty()) {
                return; // sample using speculative decoding
            }

            // shifted according to the current sub-batch
            const int tok_idx = slot.i_batch - off;

            if (!sample_completed_slot(slot, tok_idx)) {
                slot.print_timings();
                send_final_response(slot);
                slot.release();
            }
        });

        // speculative decoding - main model sample and accept
        iterate(slots, [&](server_slot & slot) {
            if (slot.state != SLOT_STATE_GENERATING || !slot.can_speculate() ||
                    slot.spec_draft.empty() || slot.spec_i_batch.empty()) {
                return;
            }

            // save the original draft size
            const size_t n_draft = slot.spec_draft.size();

            GGML_ASSERT(n_draft > 0);

            // batch indices of the draft tokens, used to get the token probabilities below
            auto spec_i_batch = std::move(slot.spec_i_batch);

            // verify and try to accept the draft
            {
                common_sampler_ptr smpl_save(common_sampler_clone(slot.smpl.get()));

                GGML_ASSERT(spec_i_batch.size() == n_draft + 1);
                GGML_ASSERT(slot.spec_draft_q.empty() || (slot.spec_draft_q.size() == slot.spec_draft.size()));
                const auto & synth_probs = common_speculative_get_synth_probs(spec.get());

                // drafters that fill no distribution fall back here
                const bool use_rejection = slot.use_spec_rejection() && !slot.spec_draft_q.empty();

                std::vector<llama_token> accepted;
                if (!synth_probs.empty()) {
                    // synthetic acceptance replaces verification entirely, so it comes first
                    accepted = server_sample_and_accept_synth(
                            slot.smpl.get(), slot.ctx_tgt, spec_i_batch, slot.spec_draft,
                            synth_probs, slot.spec_synth_rng, slot.spec_is_replay);
                } else if (slot.spec_is_replay && slot.use_spec_rejection()) {
                    accepted = server_accept_replay(slot.smpl.get(), slot.ctx_tgt, spec_i_batch, slot.spec_draft);
                } else if (use_rejection) {
                    accepted = common_sampler_sample_and_accept_n_rejection(slot.smpl.get(), slot.ctx_tgt, spec_i_batch, slot.spec_draft, slot.spec_draft_q);
                } else {
                    accepted = common_sampler_sample_and_accept_n(slot.smpl.get(), slot.ctx_tgt, spec_i_batch, slot.spec_draft);
                }

                GGML_ASSERT(accepted.size() >= 1);

                const uint32_t n_rollback = slot.spec_draft.size() + 1 - accepted.size();

                const bool use_ckpt_tgt =
                    ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                    (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && n_rollback > llama_n_rs_seq(ctx_tgt));
                if (moe_trace) {
                    moe_trace->accepted(n_draft, accepted.size() - 1, slot.spec_is_replay,
                                        n_rollback > 0 && use_ckpt_tgt);
                }

                // check for partial draft acceptance
                if (n_rollback > 0) {
                    if (use_ckpt_tgt) {
                        if (trace > 0) {
                            SLT_INF(slot, "accepted %2zu/%2zu draft tokens (restore checkpoint)\n", accepted.size() - 1, slot.spec_draft.size());
                        }

                        // partial acceptance is not supported by the context -> truncate the draft and restore the state
                        slot.spec_is_replay = true;
                        slot.spec_draft = std::move(accepted);

                        const auto & ckpt = slot.spec_ckpt;

                        SLT_DBG(slot, "restoring speculative checkpoint (pos_min = %d, pos_max = %d, size = %zu)\n", ckpt.pos_min, ckpt.pos_max, ckpt.size());

                        GGML_ASSERT(ckpt.load_tgt(slot.ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));

                        if (slot.ctx_dft) {
                            GGML_ASSERT(ckpt.load_dft(slot.ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
                        }

                        slot.mem.seq_rm(slot.id, ckpt.pos_max + 1, -1);

                        slot.prompt.tokens.keep_first(ckpt.n_tokens);
                        common_sampler_copy(smpl_save.get(), slot.smpl.get());

                        return;
                    }
                }

                if (trace > 0) {
                    SLT_INF(slot, "accepted %2zu/%2zu draft tokens\n", accepted.size() - 1, n_draft);
                }

                common_speculative_accept(spec.get(), slot.id, accepted.size() - 1);

                slot.spec_draft = std::move(accepted);
            }

            const auto ids = std::move(slot.spec_draft);

            size_t n_accepted = ids.size() - 1;
            if (slot.spec_is_replay && n_accepted > 0) {
                n_accepted--;
            }
            slot.spec_is_replay = false;

            slot.stats.update_gen_last();

            // update how many tokens out of those tested were accepted
            slot.stats.n_draft_accepted += n_accepted;
            slot.stats.n_draft_verif_steps += 1;

            auto & n_accepted_per_pos = slot.n_accepted_per_pos;
            if (n_accepted_per_pos.empty()) {
                n_accepted_per_pos.resize(common_speculative_n_max(spec.get()), 0);
            }
            for (size_t i = 0; i < n_accepted && i < n_accepted_per_pos.size(); ++i) {
                n_accepted_per_pos[i]++;
            }

            // add accepted tokens to the prompt
            slot.prompt.tokens.keep_first(slot.prompt.n_tokens() - n_draft);
            slot.prompt.tokens.insert({ids.begin(), ids.end() - 1});

            slot.sampled = ids.back(); // last accepted token
            SLT_DBG(slot, "add accepted tokens: sampled=%d, ids.size=%zu, n_draft=%zu\n", slot.sampled, ids.size(), n_draft);

            slot.mem.seq_rm(slot.id, slot.prompt.tokens.pos_next(), -1);

            for (size_t i = 0; i < ids.size(); ++i) {
                completion_token_output result;

                result.tok          = ids[i];
                result.text_to_send = common_token_to_piece(slot.ctx_tgt, result.tok, accept_special_token(slot, result.tok));
                result.prob         = 1.0f; // set later

                // post_sampling_probs is not supported with speculative decoding
                if (slot.task->params.sampling.n_probs > 0 && !slot.task->params.post_sampling_probs) {
                    populate_token_probs(slot, result, false, params_base.special, spec_i_batch[i]);
                }

                slot.stats.n_gen += 1;

                if (!process_token(result, slot)) {
                    slot.print_timings();
                    send_final_response(slot);
                    slot.release();

                    return;
                }
            }

            slot.print_timings_tg();

            SLT_DBG(slot, "accepted %d/%d draft tokens, new n_tokens = %d\n", (int) n_accepted, (int) n_draft, slot.prompt.n_tokens());
        });
    }

    // context size of a single slot, capped by --kv-unified-per-slot and by the training context of the model
    int n_ctx_slot() const {
        int res = llama_n_ctx_seq(ctx_tgt);

        if (params_base.kv_unified_per_slot > 0) {
            res = std::min(res, params_base.kv_unified_per_slot);
        }

        return std::min(res, llama_model_n_ctx_train(model_tgt));
    }

    server_response_reader get_response_reader() {
        return server_response_reader(queue_tasks, queue_results, HTTP_POLLING_SECONDS);
    }

    //
    // metrics helpers
    //

    // call before submitting a decode, so that the queued prompt stats can be timed
    void metrics_pre_decode() {
        t_decode_start = ggml_time_us();
    }

    // the batch is submitted, but its compute may not be done yet
    void metrics_queue_prompt(uint64_t n_tokens) {
        if (n_tokens == 0) {
            return;
        }
        if (n_prompt_queued == 0) {
            t_prompt_start = t_decode_start;
        }
        n_prompt_queued += n_tokens;
    }

    // call only after the context is synchronized, otherwise the time is meaningless
    void metrics_flush_prompt() {
        if (n_prompt_queued == 0) {
            return;
        }
        metrics.add_prompt(n_prompt_queued, ggml_time_us() - t_prompt_start);
        n_prompt_queued = 0;
    }

    // has_output is computed by the caller, which also already synchronized the context if it is set
    void metrics_post_decode(int32_t off, int32_t n_tokens, bool has_output, const std::set<llama_seq_id> & accounted) {
        metrics.n_decode++;
        for (const auto & slot : slots) {
            if (slot.is_processing()) {
                metrics.n_busy_slots++;
            }
            metrics.n_tokens_max = std::max(metrics.n_tokens_max, (uint64_t) slot.prompt.n_tokens());
        }

        // apply enqueued prompt tokens stats
        // note: a slot can be released before we get here, which clears its stats
        //       the tokens were still computed, counted in the global metrics, not in slot
        uint64_t n_prompt_tokens = 0;

        for (int i = off; i < off + n_tokens; ++i) {
            const auto & t = batch.tokens[i];

            if (!t.is_prompt) {
                continue; // generated tokens are handled after sampling
            }

            n_prompt_tokens++;

            auto & slot = slots[t.id_slot];
            if (slot.stats.is_set() && !accounted.count(slot.id)) {
                slot.stats.n_prompt_processed++;
            }
        }

        metrics_queue_prompt(n_prompt_tokens);

        if (has_output) {
            // the context is already synchronized, so the timings are correct
            metrics_flush_prompt();
        }

        // advance the prompt timing of the slots that had tokens in this batch
        // note: a second pass, it must run after the sync to reflect the compute
        const int64_t t_now = ggml_time_us();
        for (int i = off; i < off + n_tokens; ++i) {
            const auto & t = batch.tokens[i];
            auto & slot = slots[t.id_slot];
            if (t.is_prompt && slot.stats.is_set() && !accounted.count(slot.id)) {
                slot.stats.set_prompt_last(t_now);
            }
        }
    }

    // flush any queued prompt metrics if all slots are now idle
    void metrics_flush_idle() {
        if (n_prompt_queued == 0) {
            return;
        }

        llama_synchronize(ctx_tgt);
        metrics_flush_prompt();
    }

    void metrics_on_prediction(const server_slot & slot) {
        const uint64_t t_us    = slot.stats.t_gen_us();
        const uint64_t n       = slot.stats.n_gen;
        const uint64_t n_steps = slot.stats.n_gen_steps();

        metrics.predict       .add(n, n_steps, t_us);
        metrics.predict_bucket.add(n, n_steps, t_us);

        metrics.n_draft_tokens      += slot.stats.n_draft_tokens;
        metrics.n_draft_accepted    += slot.stats.n_draft_accepted;
        metrics.n_draft_verif_steps += slot.stats.n_draft_verif_steps;

        auto & dst = metrics.n_accepted_per_pos;
        const auto & src = slot.n_accepted_per_pos;

        if (dst.size() < src.size()) {
            dst.resize(src.size(), 0);
        }
        for (size_t i = 0; i < src.size(); i++) {
            dst[i] += src[i];
        }
    }
};

//
// server_context (public API)
//

server_context::server_context() : impl(new server_context_impl()) {}
server_context::~server_context() = default;

bool server_context::load_model(common_params & params) {
    return impl->load_model(params);
}

void server_context::start_loop() {
    auto & params = impl->params_base;
    impl->queue_tasks.start_loop(params.sleep_idle_seconds * 1000);
}

void server_context::terminate() {
    impl->queue_tasks.terminate();
}

llama_context * server_context::get_llama_context() const {
    return impl->ctx_tgt;
}

server_response_reader server_context::get_response_reader() {
    return impl->get_response_reader();
}

server_context_meta server_context::get_meta() const {
    auto bos_id = llama_vocab_bos(impl->vocab);
    auto eos_id = llama_vocab_eos(impl->vocab);
    auto bos_token_str = bos_id != LLAMA_TOKEN_NULL ? common_token_to_piece(impl->ctx_tgt, bos_id, true) : "";
    auto eos_token_str = eos_id != LLAMA_TOKEN_NULL ? common_token_to_piece(impl->ctx_tgt, eos_id, true) : "";

    const char * ftype_name = llama_ftype_name(llama_model_ftype(impl->model_tgt));

    return server_context_meta {
        /* build_info              */ std::string(llama_build_info()),
        /* model_name              */ impl->model_name,
        /* model_aliases           */ impl->model_aliases,
        /* model_tags              */ impl->model_tags,
        /* model_path              */ impl->params_base.model.path,
        /* model_output_modalities */ server_model_output_modalities(impl->model_tgt, impl->has_cap_chat(), impl->has_cap_tts()),
        /* has_mtmd                */ impl->mctx != nullptr,
        /* has_inp_image           */ impl->chat_params.allow_image,
        /* has_inp_audio           */ impl->chat_params.allow_audio,
        /* has_inp_video           */ impl->chat_params.allow_video,
        /* has_cap_chat            */ impl->has_cap_chat(),
        /* has_cap_tts             */ impl->has_cap_tts(),
        /* has_audio_stream        */ impl->mctx != nullptr && mtmd_support_audio_stream(impl->mctx),
        /* json_ui_settings        */ impl->json_ui_settings,
        /* slot_n_ctx              */ impl->n_ctx_slot(),
        /* pooling_type            */ llama_pooling_type(impl->ctx_tgt),

        /* chat_params             */ impl->chat_params,
        /* chat_template_caps      */ common_chat_templates_get_caps(impl->chat_params.tmpls.get()),

        /* bos_token_str           */ bos_token_str,
        /* eos_token_str           */ eos_token_str,
        /* fim_pre_token           */ llama_vocab_fim_pre(impl->vocab),
        /* fim_sub_token           */ llama_vocab_fim_suf(impl->vocab),
        /* fim_mid_token           */ llama_vocab_fim_mid(impl->vocab),
        /* fim_pad_token           */ llama_vocab_fim_pad(impl->vocab),
        /* fim_rep_token           */ llama_vocab_fim_rep(impl->vocab),
        /* fim_sep_token           */ llama_vocab_fim_sep(impl->vocab),

        /* logit_bias_eog          */ impl->params_base.sampling.logit_bias_eog,

        /* model_vocab_type        */ llama_vocab_type(impl->vocab),
        /* model_vocab_n_tokens    */ llama_vocab_n_tokens(impl->vocab),
        /* model_n_ctx_train       */ llama_model_n_ctx_train(impl->model_tgt),
        /* model_n_embd_inp        */ llama_model_n_embd(impl->model_tgt),
        /* model_n_params          */ llama_model_n_params(impl->model_tgt),
        /* model_size              */ llama_model_size(impl->model_tgt),
        /* model_ftype             */ ftype_name,
    };
}

// generator-like API for HTTP response generation
// may have bypass_sleep = true if the task does not use ctx_server
struct server_res_generator : server_res_spipe {
    server_response_reader rd;
    server_res_generator(server_queue & queue_tasks, server_response & queue_results, int sleep_idle_seconds, bool bypass_sleep = false)
            : rd(queue_tasks, queue_results, HTTP_POLLING_SECONDS) {
        // fast path in case sleeping is disabled
        bypass_sleep |= sleep_idle_seconds < 0;
        if (!bypass_sleep) {
            queue_tasks.wait_until_no_sleep();
        }
    }
    void ok(const json & response_data) {
        status = 200;
        data = safe_json_to_str(response_data);
    }
    void error(const json & error_data) {
        status = json_value(error_data, "code", 500);
        data = safe_json_to_str({{ "error", error_data }});
    }
};

void server_context::set_state_callback(server_state_callback_t callback) {
    impl->callback_state = std::move(callback);
}

//
// server_routes
//

std::unique_ptr<server_res_generator> server_routes::handle_completions_impl(
            const server_http_req & req,
            server_task_type type,
            const json & data,
            const std::vector<raw_buffer> & files,
            task_response_type res_type,
            const common_chat_session & chat_session) {
    GGML_ASSERT(type == SERVER_TASK_TYPE_COMPLETION || type == SERVER_TASK_TYPE_INFILL);

    auto res = create_response();
    auto completion_id = gen_chatcmplid();
    auto & rd = res->rd;
    auto & params = this->params;

    res->set_req(&req); // will also set spipe if needed

    if (!ctx_server.has_cap_chat()) {
        res->error(format_error_response("this server does not support chat/completions", ERROR_TYPE_NOT_SUPPORTED));
        return res;
    }

    int32_t sse_ping_interval = params.sse_ping_interval;
    const uint64_t sensitive_trace_id = sensitive_trace_path() == nullptr ? 0 : g_sensitive_trace_next_id.fetch_add(1);
    if (sensitive_trace_id != 0) {
        sensitive_trace_append(
            "\n===== llama.cpp request " + std::to_string(sensitive_trace_id) + " begin =====\n" +
            req.body +
            "\n----- raw output chunks -----\n");
    }

    try {
        std::vector<server_task> tasks;

        const auto & prompt = data.at("prompt");
        // TODO: this log can become very long, put it behind a flag or think about a more compact format
        //SRV_DBG("Prompt: %s\n", prompt.is_string() ? prompt.get<std::string>().c_str() : prompt.dump(2).c_str());

        if (!params.path_prompts_log_dir.empty()) {
            const auto file_path = std::filesystem::path(params.path_prompts_log_dir) / string_format("%012" PRId64 ".txt", ggml_time_ms());
            std::ofstream f(file_path);
            if (f) {
                f << (prompt.is_string() ? prompt.get<std::string>().c_str() : prompt.dump(2).c_str());
            } else {
                SRV_ERR("failed to create %s\n", file_path.string().c_str());
            }
        }

        // process prompt
        std::vector<server_tokens> inputs;

        if (res_type != TASK_RESPONSE_TYPE_NONE && ctx_server.mctx != nullptr) {
            // This is the case used by OAI compatible chat path with MTMD. TODO It can be moved to the path below.
            inputs.push_back(process_mtmd_prompt(ctx_server.mctx, prompt.get<std::string>(), files, ctx_server.init_opt));
        } else {
            // Everything else, including multimodal completions.
            inputs = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, true, true, ctx_server.init_opt);
        }

        // tasks.reserve(inputs.size()); // TODO: this is inaccurate due to child tasks

        for (size_t i = 0; i < inputs.size(); i++) {
            server_task task = server_task(type);

            task.id = rd.get_new_id();

            task.tokens = std::move(inputs[i]);
            task.params = server_schema::eval_llama_cmpl_schema(
                    ctx_server.vocab,
                    params,
                    meta->logit_bias_eog,
                    data);

            task.apply_chat_session(chat_session);
            task.params.serverless_request_id = json_value(data, "serverless_request_id", std::string{});
            task.params.execution_priority = std::clamp(json_value(data, "serverless_execution_priority", 2), 0, 3);
            if (data.contains("serverless_context")) {
                const auto & context = data.at("serverless_context");
                task.params.response_scope = context.at("scope").get<std::string>();
                task.params.response_revision = context.at("revision").get<std::string>();
                task.params.response_generation = context.at("generation").get<uint64_t>();
                if (task.params.serverless_request_id.empty() ||
                        task.params.response_revision != "resp_" + task.params.serverless_request_id.substr(4, 32))
                    throw std::invalid_argument("context revision does not match response request identity");
            }

            if (!task.params.serverless_request_id.empty() && (inputs.size() != 1 || task.params.n_cmpl != 1)) {
                throw std::invalid_argument("serverless Responses require one native completion");
            }

            task.id_slot = json_value(data, "id_slot", -1);
            sse_ping_interval = task.params.sse_ping_interval;

            // OAI-compat
            task.params.res_type          = res_type;
            task.params.oaicompat_cmpl_id = completion_id;
            task.params.oaicompat_model   = meta->model_name;

            // prepare child tasks
            if (task.params.n_cmpl > 1) {
                int n_children = task.params.n_cmpl - 1;
                for (int j = 0; j < n_children; j++) {
                    task.add_child(task.id, rd.get_new_id());
                }
            }

            tasks.push_back(std::move(task));
        }

        const std::string request_id = json_value(data, "serverless_request_id", std::string{});
        if (!request_id.empty()) {
            std::vector<int> ids;
            for (const auto & task : tasks) ids.push_back(task.id);
            if (!ctx_server.request_registry().post(request_id, ids, [&] { rd.post_tasks(std::move(tasks), chat_session); })) {
                throw std::invalid_argument("serverless response was cancelled during preparation");
            }
        } else {
            rd.post_tasks(std::move(tasks), chat_session);
        }
    } catch (const std::exception & e) {
        if (sensitive_trace_id != 0) {
            sensitive_trace_append(
                "\n===== llama.cpp request " + std::to_string(sensitive_trace_id) + " end: invalid request =====\n");
        }
        res->error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    bool stream = json_value(data, "stream", false);

    if (!stream) {
        // non-stream, wait for the results
        auto all_results = rd.wait_for_all(req.should_stop);
        if (all_results.is_terminated) {
            return res; // connection is closed
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        } else {
            json arr = json::array();
            for (auto & res : all_results.results) {
                GGML_ASSERT(dynamic_cast<server_task_result_cmpl_final*>(res.get()) != nullptr);
                arr.push_back(res->to_json());
            }
            GGML_ASSERT(!arr.empty() && "empty results");
            if (arr.size() == 1) {
                // if single request, return single object instead of array
                if (sensitive_trace_id != 0) {
                    sensitive_trace_append(
                        sensitive_trace_output_text(arr[0]) +
                        "\n===== llama.cpp request " + std::to_string(sensitive_trace_id) + " end =====\n");
                }
                res->ok(arr[0]);
            } else if (res_type == TASK_RESPONSE_TYPE_OAI_CHAT || res_type == TASK_RESPONSE_TYPE_OAI_CMPL) {
                // if multiple results in OAI format, we need to re-format them
                json & usage   = arr[0]["usage"];
                json & choices = arr[0]["choices"];
                for (size_t i = 1; i < arr.size(); i++) {
                    choices.push_back(std::move(arr[i]["choices"][0]));
                }
                if (sensitive_trace_id != 0) {
                    sensitive_trace_append(
                        sensitive_trace_output_text(arr[0]) +
                        "\n===== llama.cpp request " + std::to_string(sensitive_trace_id) + " end =====\n");
                }

                // OAI counts generated tokens across all choices
                int64_t n_completion_tokens = 0;
                for (auto & item : arr) {
                    n_completion_tokens += item["usage"]["completion_tokens"].get<int64_t>();
                }
                usage["completion_tokens"] = n_completion_tokens;
                usage["total_tokens"]      = usage["prompt_tokens"].get<int64_t>() + n_completion_tokens;

                res->ok(arr[0]);
            } else {
                // multi-results, non-OAI compat
                if (sensitive_trace_id != 0) {
                    std::string trace_text;
                    for (const auto & item : arr) {
                        trace_text += sensitive_trace_output_text(item);
                    }
                    sensitive_trace_append(
                        trace_text +
                        "\n===== llama.cpp request " + std::to_string(sensitive_trace_id) + " end =====\n");
                }
                res->ok(arr);
            }
        }
    } else {
        // in streaming mode, the first error must be treated as non-stream response
        // this is to match the OAI API behavior
        // ref: https://github.com/ggml-org/llama.cpp/pull/16486#discussion_r2419657309
        auto first_result = rd.next(req.should_stop);
        if (first_result == nullptr) {
            GGML_ASSERT(req.should_stop());
            if (sensitive_trace_id != 0) {
                sensitive_trace_append(
                    "\n===== llama.cpp request " + std::to_string(sensitive_trace_id) + " end: client stopped =====\n");
            }
            return res; // connection is closed
        }

        if (first_result->is_error()) {
            if (sensitive_trace_id != 0) {
                sensitive_trace_append(
                    "\n===== llama.cpp request " + std::to_string(sensitive_trace_id) + " end: error =====\n");
            }
            res->error(first_result->to_json());
            return res;
        }

        GGML_ASSERT(server_task_result_is_completion_stream(first_result.get()));

        // next responses are streamed
        // to be sent immediately
        json first_result_json = first_result->to_json();
        if (sensitive_trace_id != 0) {
            sensitive_trace_append(sensitive_trace_output_text(first_result_json));
        }
        if (first_result_json == nullptr) {
            res->data = ""; // simply send HTTP headers and status code
        } else if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
            res->data = format_anthropic_sse(first_result_json);
        } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
            res->data = format_oai_resp_sse(first_result_json);
        } else {
            res->data = format_oai_sse(first_result_json);
        }
        res->status = 200;
        res->content_type = "text/event-stream";
        res->set_next([res_this = res.get(), res_type, sse_ping_interval, sensitive_trace_id](std::string & output) -> bool {
            static auto format_error = [](task_response_type res_type, const json & res_json) {
                if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                    return format_anthropic_sse({
                        {"event", "error"},
                        {"data", res_json},
                    });
                } else {
                    return format_oai_sse(json {{ "error", res_json }});
                }
            };

            auto effective_should_stop = [&res_this]() {
                return res_this->should_stop();
            };

            try {
                if (effective_should_stop()) {
                    SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
                    if (sensitive_trace_id != 0) {
                        sensitive_trace_append(
                            "\n===== llama.cpp request " + std::to_string(sensitive_trace_id) + " end: client stopped =====\n");
                    }
                    return false; // should_stop condition met
                }

                if (!res_this->data.empty()) {
                    // flush the first chunk
                    output = std::move(res_this->data);
                    res_this->data.clear();
                    return true;
                }

                server_response_reader & rd = res_this->rd;

                // check if there is more data
                if (!rd.has_next()) {
                    switch (res_type) {
                        case TASK_RESPONSE_TYPE_NONE:
                        case TASK_RESPONSE_TYPE_OAI_RESP:
                        case TASK_RESPONSE_TYPE_ANTHROPIC:
                            output = "";
                            break;

                        default:
                            output = "data: [DONE]\n\n";
                            break;
                    }
                    SRV_DBG("%s", "all results received, terminating stream\n");
                    if (sensitive_trace_id != 0) {
                        sensitive_trace_append(
                            "\n===== llama.cpp request " + std::to_string(sensitive_trace_id) + " end =====\n");
                    }
                    return false; // no more data, terminate
                }

                // receive subsequent results
                bool timeout = false;
                int64_t start_time = ggml_time_ms();
                auto result = rd.next([&timeout, &start_time, sse_ping_interval, &effective_should_stop]() {
                    if (effective_should_stop()) {
                        return true; // should_stop condition met
                    } else if (sse_ping_interval > 0 && ggml_time_ms() - start_time > (int64_t)sse_ping_interval * 1000) {
                        timeout = true;
                        return true; // timeout
                    }
                    return false;
                });

                if (timeout) {
                    // some clients may time out (e.g. undici) will time out if no data is received for a while, so we need to send a ping to keep the connection alive
                    SRV_DBG("%s", "sending SSE ping\n");
                    output = ":\n\n";
                    return true;
                }

                if (result == nullptr) {
                    SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
                    GGML_ASSERT(effective_should_stop());
                    if (sensitive_trace_id != 0) {
                        sensitive_trace_append(
                            "\n===== llama.cpp request " + std::to_string(sensitive_trace_id) + " end: client stopped =====\n");
                    }
                    return false; // should_stop condition met
                }

                // send the results
                if (result->is_error()) {
                    json res_json = result->to_json();
                    output = format_error(res_type, res_json);
                    SRV_DBG("%s", "error received during streaming, terminating stream\n");
                    if (sensitive_trace_id != 0) {
                        sensitive_trace_append(
                            "\n===== llama.cpp request " + std::to_string(sensitive_trace_id) + " end: error =====\n");
                    }
                    return false; // terminate on error
                } else {
                    GGML_ASSERT(server_task_result_is_completion_stream(result.get()));
                    json res_json = result->to_json();
                    if (sensitive_trace_id != 0) {
                        sensitive_trace_append(sensitive_trace_output_text(res_json));
                    }
                    if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                        output = format_anthropic_sse(res_json);
                    } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
                        output = format_oai_resp_sse(res_json);
                    } else {
                        output = format_oai_sse(res_json);
                    }
                }

                // has next data, continue
                return true;

            } catch (const std::exception & e) {
                json error_json = format_error_response(e.what(), ERROR_TYPE_SERVER);
                output = format_error(res_type, error_json);

                // terminate on exception
                return false;
            }
        });
    }

    return res;
}

std::unique_ptr<server_res_generator> server_routes::create_response(bool bypass_sleep) {
    return std::make_unique<server_res_generator>(queue_tasks, queue_results, params.sleep_idle_seconds, bypass_sleep);
}

server_routes::server_routes(const common_params & params, server_context & ctx_server)
        : params(params),
          ctx_server(*ctx_server.impl),
          queue_tasks(ctx_server.impl->queue_tasks),
          queue_results(ctx_server.impl->queue_results) {
    init_routes();

    // note: this must be registered before load_model()
    //       so that on sleep phase, the callback is called before ctx is destroyed
    queue_tasks.on_sleeping_state([this](bool is_sleeping) {
        update_cached_responses(is_sleeping);
    });
}

static json get_res_model_info(const server_context_meta & meta) {
    // note: do NOT use ctx_server here, otherwise it's not possible to use this during sleep

    return {
        {"id",       meta.model_name},
        {"aliases",  meta.model_aliases},
        {"tags",     meta.model_tags},
        {"object",   "model"},
        {"architecture", server_model_architecture_json(
            meta.has_inp_image,
            meta.has_inp_audio,
            meta.has_inp_video,
            meta.model_output_modalities)},
        {"created",  std::time(0)},
        {"owned_by", "llamacpp"},
        {"meta",     {
            {"vocab_type",  meta.model_vocab_type},
            {"n_vocab",     meta.model_vocab_n_tokens},
            {"n_ctx",       meta.slot_n_ctx},
            {"n_ctx_train", meta.model_n_ctx_train},
            {"n_embd",      meta.model_n_embd_inp},
            {"n_params",    meta.model_n_params},
            {"size",        meta.model_size},
            {"ftype",       meta.model_ftype},
        }},
    };
}

static json get_res_models(const server_context_meta & meta) {
    // note: do NOT use ctx_server here, otherwise it's not possible to use this during sleep

    return json{
        {"models", json::array({
            {
                {"name",  meta.model_name},
                {"model", meta.model_name},
                {"modified_at", ""},
                {"size", ""},
                {"digest", ""}, // dummy value, llama.cpp does not support managing model file's hash
                {"type", "model"},
                {"description", ""},
                {"tags", json::array({""})},
                {"capabilities", meta.has_mtmd ? json::array({"completion","multimodal"}) : json::array({"completion"})},
                {"parameters", ""},
                {"details", {
                    {"parent_model", ""},
                    {"format", "gguf"},
                    {"family", ""},
                    {"families", json::array({""})},
                    {"parameter_size", ""},
                    {"quantization_level", ""}
                }}
            }
        })},
        {"object", "list"},
        {"data", json::array({
            get_res_model_info(meta),
        })}
    };
}

static json get_res_props(const server_context_meta & meta, const common_params & params, bool is_sleeping) {
    // note: do NOT use ctx_server here, otherwise it's not possible to use this during sleep

    task_params tparams;
    tparams.sampling = params.sampling;
    json default_generation_settings_for_props = json {
        { "params", tparams.to_json(true) },
        { "n_ctx",  meta.slot_n_ctx },
    };

    std::string tmpl_default = common_chat_templates_source(meta.chat_params.tmpls.get(), "");
    std::string tmpl_tools   = common_chat_templates_source(meta.chat_params.tmpls.get(), "tool_use");

    json props = {
        { "default_generation_settings", default_generation_settings_for_props },
        { "total_slots",                 params.n_parallel },
        { "model_alias",                 meta.model_name },
        { "model_ftype",                 meta.model_ftype },
        { "model_path",                  meta.model_path },
        { "modalities",                  json {
            {"vision", meta.has_inp_image},
            {"video",  meta.has_inp_video},
            {"audio",  meta.has_inp_audio},
        } },
        { "media_marker",                get_media_marker() },
        { "endpoint_slots",              params.endpoint_slots },
        { "endpoint_props",              params.endpoint_props },
        { "endpoint_metrics",            params.endpoint_metrics },
        { "ui",                          params.ui },
        { "ui_settings",                 meta.json_ui_settings },
        { "chat_template",               tmpl_default },
        { "chat_template_caps",          meta.chat_template_caps },
        { "bos_token",                   meta.bos_token_str },
        { "eos_token",                   meta.eos_token_str },
        { "build_info",                  meta.build_info },
        { "is_sleeping",                 is_sleeping },
        { "cors_proxy_enabled",          params.ui_mcp_proxy },
    };

    if (!params.server_connect_code.empty()) {
        props["connect_code"] = params.server_connect_code;
    }
    if (params.use_jinja) {
        if (!tmpl_tools.empty()) {
            props["chat_template_tool_use"] = tmpl_tools;
        }
    }

    return props;
}

json server_routes::get_model_info() const {
    return get_res_model_info(*meta);
}

void server_routes::init_routes() {
    // IMPORTANT: all lambda functions must start with create_response()
    // this is to ensure that the server_res_generator can handle sleeping case correctly

    this->get_health = [this](const server_http_req &) {
        // error and loading states are handled by middleware
        auto res = create_response(true);

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server; // do NOT delete this line
        GGML_UNUSED(ctx_server);

        res->ok({{"status", "ok"}});
        return res;
    };

    this->get_metrics = [this](const server_http_req & req) {
        auto res = create_response(true);
        if (!params.endpoint_metrics) {
            res->error(format_error_response("This server does not support metrics endpoint. Start it with `--metrics`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // render response using cached_metrics
        auto use_cached_metrics = [&]() {
            std::unique_lock<std::mutex> lock(mutex_cache);
            res->headers["Process-Start-Time-Unix"] = std::to_string(cached_metrics.t_start);
            server_task_result_metrics tmp;
            tmp.metrics = cached_metrics;
            res->content_type = "text/plain; version=0.0.4";
            res->status = 200;
            res->data = tmp.to_metrics();
            // the gauges are averaged over the window between two scrapes
            cached_metrics.reset_bucket();
            should_reset_buckets = true;
        };

        if (queue_tasks.is_sleeping()) {
            use_cached_metrics();

        } else {
            // request slots data using task queue
            {
                server_task task(SERVER_TASK_TYPE_METRICS);
                task.id = res->rd.get_new_id();
                // the gauges are averaged over the window between two scrapes
                task.metrics_reset_bucket = true;
                res->rd.post_task(std::move(task), true); // high-priority task
            }

            // a task posted right before sleeping is never processed, do not wait for it
            auto result = res->rd.next([&]{
                return req.should_stop() || queue_tasks.is_sleeping();
            });
            if (!result) {
                if (!req.should_stop()) {
                    use_cached_metrics();
                }
                return res;
            }

            if (result->is_error()) {
                res->error(result->to_json());
                return res;
            }

            auto res_task = dynamic_cast<server_task_result_metrics*>(result.get());
            GGML_ASSERT(res_task != nullptr);

            res->headers["Process-Start-Time-Unix"] = std::to_string(res_task->metrics.t_start);
            res->content_type = "text/plain; version=0.0.4";
            res->status = 200;
            res->data = res_task->to_metrics();
        }

        return res;
    };

    this->get_slots = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.endpoint_slots) {
            res->error(format_error_response("This server does not support slots endpoint. Start it with `--slots`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // request slots data using task queue
        {
            server_task task(SERVER_TASK_TYPE_SLOT_GET);
            task.id = res->rd.get_new_id();
            res->rd.post_task(std::move(task), true); // high-priority task
        }

        // get the result
        auto result = res->rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        auto * res_task = dynamic_cast<server_task_result_slots*>(result.get());
        GGML_ASSERT(res_task != nullptr);

        // optionally return "fail_on_no_slot" error
        if (!req.get_param("fail_on_no_slot").empty()) {
            if (res_task->n_idle_slots == 0) {
                res->error(format_error_response("no slot available", ERROR_TYPE_UNAVAILABLE));
                return res;
            }
        }

        res->ok(res_task->to_json());
        return res;
    };

    this->post_slots = [this](const server_http_req & req) {
        auto res = create_response();
        if (params.slot_save_path.empty()) {
            res->error(format_error_response("This server does not support slots action. Start it with `--slot-save-path`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        std::string id_slot_str = req.get_param("id_slot");

        int id_slot;
        try {
            id_slot = std::stoi(id_slot_str);
        } catch (const std::exception &) {
            res->error(format_error_response("Invalid slot ID", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::string action = req.get_param("action");

        if (action == "save") {
            return handle_slots_save(req, id_slot);
        }
        if (action == "restore") {
            return handle_slots_restore(req, id_slot);
        }
        if (action == "erase") {
            return handle_slots_erase(req, id_slot);
        }

        res->error(format_error_response("Invalid action", ERROR_TYPE_INVALID_REQUEST));
        return res;
    };

    this->get_props = [this](const server_http_req &) {
        auto res = create_response(true);
        // note: do NOT use ctx_server here, this endpoint must be accessible during sleep
        if (queue_tasks.is_sleeping()) {
            std::unique_lock<std::mutex> lock(mutex_cache);
            res->ok(cached_props);
        } else {
            res->ok(get_res_props(*meta, params, false));
        }
        return res;
    };

    this->post_props = [this](const server_http_req &) {
        auto res = create_response();
        if (!params.endpoint_props) {
            res->error(format_error_response("This server does not support changing global properties. Start it with `--props`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }
        // update any props here

        res->ok({{ "success", true }});
        return res;
    };

    this->post_infill = [this](const server_http_req & req) {
        auto res = create_response();
        // check model compatibility
        std::string err;
        if (llama_vocab_fim_pre(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "prefix token is missing. ";
        }
        if (llama_vocab_fim_suf(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "suffix token is missing. ";
        }
        if (llama_vocab_fim_mid(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "middle token is missing. ";
        }
        if (!err.empty()) {
            res->error(format_error_response(string_format("Infill is not supported by this model: %s", err.c_str()), ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // validate input
        json data = json::parse(req.body);
        if (data.contains("prompt") && !data.at("prompt").is_string()) {
            // prompt is optional
            res->error(format_error_response("\"prompt\" must be a string", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_prefix")) {
            res->error(format_error_response("\"input_prefix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_suffix")) {
            res->error(format_error_response("\"input_suffix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (data.contains("input_extra") && !data.at("input_extra").is_array()) {
            // input_extra is optional
            res->error(format_error_response("\"input_extra\" must be an array of {\"filename\": string, \"text\": string}", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        json input_extra = json_value(data, "input_extra", json::array());
        for (const auto & chunk : input_extra) {
            // { "text": string, "filename": string }
            if (!chunk.contains("text") || !chunk.at("text").is_string()) {
                res->error(format_error_response("extra_context chunk must contain a \"text\" field with a string value", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            // filename is optional
            if (chunk.contains("filename") && !chunk.at("filename").is_string()) {
                res->error(format_error_response("extra_context chunk's \"filename\" field must be a string", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        }
        data["input_extra"] = input_extra; // default to empty array if it's not exist

        std::string prompt = json_value(data, "prompt", std::string());
        std::vector<server_tokens> tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, false, true, ctx_server.init_opt);
        SRV_DBG("creating infill tasks, n_prompts = %d\n", (int) tokenized_prompts.size());
        data["prompt"] = format_prompt_infill(
            ctx_server.vocab,
            data.at("input_prefix"),
            data.at("input_suffix"),
            data.at("input_extra"),
            params.n_batch,
            params.n_predict,
            meta->slot_n_ctx,
            params.spm_infill,
            tokenized_prompts[0].get_tokens() // TODO: this could maybe be multimodal.
        );

        std::vector<raw_buffer> files; // dummy
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_INFILL,
            data,
            files,
            TASK_RESPONSE_TYPE_NONE); // infill is not OAI compatible
    };

    this->post_completions = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy
        json body = json::parse(req.body);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body,
            files,
            TASK_RESPONSE_TYPE_NONE);
    };

    this->post_completions_oai = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy
        json body = json::parse(req.body);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body,
            files,
            TASK_RESPONSE_TYPE_OAI_CMPL);
    };

    this->post_chat_completions = [this](const server_http_req & req) {
        json body = json::parse(req.body);
        const std::string id = json_value(body, "serverless_request_id", std::string{});
        std::unique_ptr<server_responses_preparing_guard> preparing;
        if (!id.empty()) preparing = std::make_unique<server_responses_preparing_guard>(ctx_server.request_registry(), id);
        auto res = create_response();
        std::vector<raw_buffer> files;
        common_chat_session session;
        json body_parsed = oaicompat_chat_params_parse(
            ctx_server.vocab,
            body,
            meta->chat_params,
            files,
            session);
        if (!id.empty()) body_parsed["serverless_request_id"] = id;
        body_parsed["serverless_execution_priority"] = json_value(body, "serverless_execution_priority", 2);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_CHAT,
            session);
    };

    this->post_chat_completions_tok = [this](const server_http_req & req) {
        return handle_count_tokens(req, TASK_RESPONSE_TYPE_OAI_CHAT);
    };

    this->post_control = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);

        const std::string cmpl_id = json_value(body, "id", std::string());
        const std::string action  = json_value(body, "action", std::string());
        if (cmpl_id.empty()) {
            res->error(format_error_response("missing completion id", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        if (action != "reasoning_end") {
            res->error(format_error_response("unknown control action", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_CONTROL);
            task.id              = rd.get_new_id();
            task.params.control_cmpl_id = cmpl_id;
            task.params.control_action  = SERVER_CONTROL_ACTION_REASONING_END;
            rd.post_task(std::move(task));
        }

        auto result = rd.next(req.should_stop);
        if (!result) {
            GGML_ASSERT(req.should_stop());
            return res;
        }
        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }
        res->ok(result->to_json());
        return res;
    };

    this->get_responses_capabilities = [this](const server_http_req &) {
        auto res = create_response(true); // pure control metadata; never wakes a model
        res->ok({{"interface_version", 1}, {"response_format", true}, {"request_registry", true}, {"chat_request_registry", true},
                 {"partial_prefill", true}, {"prefill_suspend", true}, {"decode_suspend", false}, {"cache_ownership", true}, {"renderer_profile", "glm53-native-responses-v1"}});
        return res;
    };

    this->post_responses_context = [this](const server_http_req & req) {
        auto res = create_response(true); // control never wakes/loads the model
        const json body = json::parse(req.body);
        const auto scope = json_value(body, "scope", std::string{});
        const auto action = json_value(body, "action", std::string{});
        if (scope.size() != 64 || scope.find_first_not_of("0123456789abcdef") != std::string::npos)
            throw std::invalid_argument("invalid context scope identity");
        if (body.contains("expected_namespace")) {
            const auto expected_namespace = body.at("expected_namespace").get<std::string>();
            const std::string current_namespace = std::getenv("LLAMA_SERVER_PAGE_CACHE_NAMESPACE")
                ? std::getenv("LLAMA_SERVER_PAGE_CACHE_NAMESPACE") : "";
            if (expected_namespace != current_namespace) throw std::invalid_argument("context namespace changed");
        }
        const std::set<std::string> actions{"state", "prepare", "observe", "commit", "release", "abandon"};
        if (!actions.count(action)) throw std::invalid_argument("invalid context action");
        for (const auto & item : body.items()) {
            const std::set<std::string> allowed{"scope", "action", "revision", "expected", "generation", "branch", "old_branch", "minimum_tokens", "lease_ms", "expected_namespace"};
            if (!allowed.count(item.key())) throw std::invalid_argument("unknown context field");
        }
        for (const auto field : {"revision", "expected"}) if (body.contains(field)) {
            const auto value = body.at(field).get<std::string>();
            if (!value.empty() && (value.size() != 37 || value.substr(0,5) != "resp_" ||
                    value.substr(5).find_first_not_of("0123456789abcdef") != std::string::npos))
                throw std::invalid_argument("invalid context revision");
        }
        for (const auto field : {"generation", "minimum_tokens", "lease_ms"}) if (body.contains(field)) {
            const auto & value = body.at(field);
            if (!value.is_number_integer() || value.is_boolean() || value.get<int64_t>() < 0)
                throw std::invalid_argument("invalid context numeric field");
        }
        if (body.contains("lease_ms") && (body.at("lease_ms").get<uint64_t>() == 0 || body.at("lease_ms").get<uint64_t>() > 7200000))
            throw std::invalid_argument("invalid context lease duration");
        if (action == "prepare" && (!body.contains("revision") || !body.contains("generation")))
            throw std::invalid_argument("context prepare needs revision and generation");
        if (action == "observe" && (!body.contains("revision") || !body.contains("branch") || !body.contains("minimum_tokens")))
            throw std::invalid_argument("context observe needs revision, branch and endpoint");
        for (const auto field : {"branch", "old_branch"}) if (body.contains(field))
            server_schema::parse_cache_control(json{{"branch", body.at(field)}});
        server_task task(SERVER_TASK_TYPE_RESPONSES_CONTEXT);
        task.id = res->rd.get_new_id(); task.response_context = body;
        if (!res->rd.post_task_if_awake(std::move(task), true)) {
            res->ok({{"success", false}, {"reason", "worker sleeping or stopped"}});
            return res;
        }
        const auto result = res->rd.next(req.should_stop);
        if (!result || result->is_error()) throw std::runtime_error("native context control did not acknowledge");
        res->ok(result->to_json());
        return res;
    };

    this->post_responses_suspend = [this](const server_http_req & req) {
        auto res = create_response(true);
        const json body = json::parse(req.body);
        const std::string id = json_value(body, "request_id", std::string{});
        if (!server_responses_registry::valid_id(id)) throw std::invalid_argument("invalid suspend identity");
        const auto targets = ctx_server.request_registry().targets(id);
        if (targets.size() != 1) {
            res->ok({{"request_id", id}, {"suspended", false}, {"released", false}});
            return res;
        }
        server_task task(SERVER_TASK_TYPE_RESPONSES_SUSPEND);
        task.id = res->rd.get_new_id();
        task.id_target = targets.front();
        res->rd.post_task(std::move(task), true);
        const auto result = res->rd.next(req.should_stop);
        if (!result || result->is_error()) throw std::runtime_error("native suspend did not acknowledge");
        const auto * reply = dynamic_cast<server_task_result_control *>(result.get());
        res->ok({{"request_id", id}, {"suspended", reply && reply->success},
                 {"released", ctx_server.request_registry().released(id)}});
        return res;
    };

    this->post_responses_cancel = [this](const server_http_req & req) {
        auto res = create_response(true);
        const json body = json::parse(req.body);
        if (!body.is_object() || body.size() != 1 || !body.contains("request_id") || !body.at("request_id").is_string()) {
            throw std::invalid_argument("cancel requires only a request_id string");
        }
        const std::string id = body.at("request_id");
        const auto targets = ctx_server.request_registry().cancel(id);
        std::vector<server_task> tasks;
        for (int target : targets) {
            server_task task(SERVER_TASK_TYPE_CANCEL);
            task.id_target = target;
            tasks.push_back(std::move(task));
        }
        if (!tasks.empty()) queue_tasks.post(std::move(tasks), true);
        res->ok({{"request_id", id}, {"released", ctx_server.request_registry().released(id)}});
        return res;
    };

    this->post_responses_oai = [this](const server_http_req & req) {
        json incoming = json::parse(req.body);
        const std::string id = json_value(incoming, "serverless_request_id", std::string{});
        std::unique_ptr<server_responses_preparing_guard> preparing;
        if (!id.empty()) preparing = std::make_unique<server_responses_preparing_guard>(ctx_server.request_registry(), id);
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = server_chat_convert_responses_to_chatcmpl(incoming);
        SRV_DBG("%s\n", "Request converted: OpenAI Responses -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        common_chat_session session;
        json body_parsed = oaicompat_chat_params_parse(
            ctx_server.vocab,
            body,
            meta->chat_params,
            files,
            session);
        if (!id.empty()) body_parsed["serverless_request_id"] = id;
        body_parsed["serverless_execution_priority"] = json_value(incoming, "serverless_execution_priority", 2);
        if (incoming.contains("serverless_context")) body_parsed["serverless_context"] = incoming.at("serverless_context");
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_RESP,
            session);
    };

    this->post_responses_tok_oai = [this](const server_http_req & req) {
        return handle_count_tokens(req, TASK_RESPONSE_TYPE_OAI_RESP);
    };

    this->post_transcriptions_oai = [this](const server_http_req & req) {
        auto res = create_response();

        if (!meta->has_mtmd || !meta->chat_params.allow_audio) {
            res->error(format_error_response("The current model does not support audio input.", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        std::vector<raw_buffer> files;
        json body = convert_transcriptions_to_chatcmpl(
            json::parse(req.body),
            meta->chat_params.tmpls.get(),
            req.files,
            files);
        SRV_DBG("%s\n", "Request converted: OpenAI Transcriptions -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        common_chat_session session;
        json body_parsed = oaicompat_chat_params_parse(
            ctx_server.vocab,
            body,
            meta->chat_params,
            files,
            session);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_ASR,
            session);
    };

    // realtime transcription over websocket, modeled after the OpenAI realtime API
    // audio is PCM16 mono at the model sample rate, sent as base64 in input_audio_buffer.append events
    // input_audio_buffer.commit with "final": true ends the input, closing the socket also ends it
    this->ws_realtime = [this](const server_http_req &, server_http_context::ws_conn & conn) {
        auto send_event = [&conn](json ev) {
            ev["event_id"] = "event_" + random_string();
            conn.send(safe_json_to_str(ev));
        };
        auto send_error_event = [&send_event](const std::string & message, const std::string & type) {
            send_event(json {
                {"type", "error"},
                {"error", {
                    {"type",    type},
                    {"code",    nullptr},
                    {"message", message},
                    {"param",   nullptr},
                }},
            });
        };

        if (!meta->has_audio_stream) {
            send_error_event("The current model does not support realtime transcription.", "invalid_request_error");
            conn.close();
            return;
        }

        const std::string session_id = "sess_" + random_string();
        const json session_config = json {
            {"id",     session_id},
            {"object", "realtime.transcription_session"},
            {"type",   "transcription"},
            {"audio",  {{"input", {
                {"format",          {{"type", "audio/pcm"}, {"rate", mtmd_get_audio_sample_rate(ctx_server.mctx)}}},
                {"transcription",   {{"model", meta->model_name}}},
                {"turn_detection",  nullptr},
                {"noise_reduction", nullptr},
            }}}},
        };

        auto res = create_response();
        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_TRANSCRIBE_RT);
            task.id = rd.get_new_id();
            task.params.sampling          = params.sampling;
            task.params.stream            = true;
            task.params.oaicompat_cmpl_id = session_id;
            task.params.oaicompat_model   = meta->model_name;
            rd.post_task(std::move(task));
        }

        // the slot answers once its audio stream is live, the session starts there
        {
            auto result = rd.next([&conn]() { return !conn.is_open(); });
            if (!result || result->is_error()) {
                if (result) {
                    send_error_event(json_value(result->to_json(), "message", std::string("server error")), "server_error");
                }
                conn.close();
                return;
            }
        }
        send_event(json {{"type", "session.created"}, {"session", session_config}});

        // one item per committed audio buffer, deltas are attributed to the current item
        std::mutex  mutex_item;
        std::string item_id = "item_" + random_string();
        std::string item_prev; // the committed item before item_id, empty for the first one
        std::string item_text;

        // called with mutex_item held
        auto send_completed = [&]() {
            send_event(json {
                {"type",          "conversation.item.input_audio_transcription.completed"},
                {"item_id",       item_id},
                {"content_index", 0},
                {"transcript",    item_text},
            });
            item_text.clear();
        };

        auto post_control = [&](server_control_action action, std::vector<float> && audio) {
            server_response_reader ctl(queue_tasks, queue_results, HTTP_POLLING_SECONDS);
            server_task task(SERVER_TASK_TYPE_CONTROL);
            task.id                     = ctl.get_new_id();
            task.params.control_action  = action;
            task.params.control_cmpl_id = session_id;
            task.params.control_audio   = std::move(audio);
            ctl.post_task(std::move(task));
            auto result = ctl.next([&conn]() { return !conn.is_open(); });
            auto * control = result ? dynamic_cast<server_task_result_control *>(result.get()) : nullptr;
            return control && control->success;
        };

        // client events are read on this thread, the transcript is sent from the thread below
        std::atomic<bool> input_ended{false};
        std::thread reader([&]() {
            std::string msg;
            while (conn.read(msg) != 0) {
                json ev;
                try {
                    ev = json::parse(msg);
                } catch (const std::exception &) {
                    send_error_event("invalid JSON", "invalid_request_error");
                    continue;
                }
                const std::string type = json_value(ev, "type", std::string());
                if (type == "session.update" || type == "transcription_session.update") {
                    send_event(json {{"type", "session.updated"}, {"session", session_config}});
                } else if (type == "input_audio_buffer.append") {
                    std::string pcm;
                    try {
                        pcm = base64::decode(json_value(ev, "audio", std::string()));
                    } catch (const std::exception &) {
                        send_error_event("invalid base64 audio", "invalid_request_error");
                        continue;
                    }
                    std::vector<float> samples(pcm.size() / 2);
                    for (size_t i = 0; i < samples.size(); i++) {
                        int16_t v;
                        memcpy(&v, pcm.data() + 2 * i, sizeof(v));
                        samples[i] = v / 32768.0f;
                    }
                    if (!post_control(SERVER_CONTROL_ACTION_AUDIO_APPEND, std::move(samples))) {
                        break;
                    }
                } else if (type == "input_audio_buffer.commit") {
                    // "final": true ends the input, the last item is completed when the stream drains
                    if (json_value(ev, "final", false)) {
                        input_ended = true;
                        post_control(SERVER_CONTROL_ACTION_AUDIO_END, {});
                        continue;
                    }
                    if (!post_control(SERVER_CONTROL_ACTION_AUDIO_COMMIT, {})) {
                        break;
                    }
                    // completed is sent once the rows of the committed audio are decoded (audio_commit_drained below)
                    std::lock_guard<std::mutex> lock(mutex_item);
                    send_event(json {
                        {"type",             "input_audio_buffer.committed"},
                        {"previous_item_id", item_prev.empty() ? json(nullptr) : json(item_prev)},
                        {"item_id",          item_id},
                    });
                } else if (type == "input_audio_buffer.clear") {
                    send_event(json {{"type", "input_audio_buffer.cleared"}});
                } else {
                    send_error_event("unknown event type: " + type, "invalid_request_error");
                }
            }
            if (!input_ended.exchange(true)) {
                post_control(SERVER_CONTROL_ACTION_AUDIO_END, {});
            }
            // the result loop waits on the socket, nobody reads it from here on
            conn.close();
        });

        while (true) {
            auto result = rd.next([&conn]() { return !conn.is_open(); });
            if (!result) {
                break; // connection closed
            }
            if (result->is_error()) {
                send_error_event(json_value(result->to_json(), "message", std::string("server error")), "server_error");
                break;
            }
            if (auto * partial = dynamic_cast<server_task_result_cmpl_partial *>(result.get())) {
                if (partial->audio_commit_drained) {
                    // complete the committed item and open the next one
                    std::lock_guard<std::mutex> lock(mutex_item);
                    send_completed();
                    item_prev = item_id;
                    item_id   = "item_" + random_string();
                } else if (!partial->content.text.empty()) {
                    std::lock_guard<std::mutex> lock(mutex_item);
                    item_text += partial->content.text;
                    send_event(json {
                        {"type",          "conversation.item.input_audio_transcription.delta"},
                        {"item_id",       item_id},
                        {"content_index", 0},
                        {"delta",         partial->content.text},
                    });
                }
            } else if (result->is_stop()) {
                std::lock_guard<std::mutex> lock(mutex_item);
                send_completed();
                break;
            }
        }

        input_ended = true;
        conn.close();
        reader.join();
    };

    this->post_anthropic_messages = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = server_chat_convert_anthropic_to_oai(json::parse(req.body));
        SRV_DBG("%s\n", "Request converted: Anthropic -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        common_chat_session session;
        json body_parsed = oaicompat_chat_params_parse(
            ctx_server.vocab,
            body,
            meta->chat_params,
            files,
            session);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_ANTHROPIC,
            session);
    };

    this->post_anthropic_count_tokens = [this](const server_http_req & req) {
        return handle_count_tokens(req, TASK_RESPONSE_TYPE_ANTHROPIC);
    };

    // same with handle_chat_completions, but without inference part
    this->post_apply_template = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy, unused
        common_chat_session session;   // dummy, unused
        json body = json::parse(req.body);
        json data = oaicompat_chat_params_parse(
            ctx_server.vocab,
            body,
            meta->chat_params,
            files,
            session);
        res->ok({{ "prompt", std::move(data.at("prompt")) }});
        return res;
    };

    this->post_cache_release = [this](const server_http_req & req) {
        auto res = create_response();
        std::string branch;
        try {
            const json body = json::parse(req.body);
            if (!body.is_object() || body.size() != 1 || !body.contains("branch")) {
                throw std::invalid_argument("cache release requires only a branch string");
            }
            branch = server_schema::parse_cache_control(body).branch;
        } catch (const std::exception & e) {
            res->error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        auto & rd = res->rd;
        server_task task(SERVER_TASK_TYPE_CACHE_RELEASE);
        task.id = rd.get_new_id();
        task.cache_branch = std::move(branch);
        rd.post_task(std::move(task));
        auto result = rd.next(req.should_stop);
        if (!result) {
            GGML_ASSERT(req.should_stop());
            return res;
        }
        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }
        res->ok(result->to_json());
        return res;
    };

    this->post_cache_probe = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = json::parse(req.body);

        json prompt;
        if (body.contains("messages")) {
            common_chat_session session;
            json data = oaicompat_chat_params_parse(
                ctx_server.vocab, body, meta->chat_params, files, session);
            prompt = std::move(data.at("prompt"));
        } else if (body.contains("prompt")) {
            prompt = body.at("prompt");
        } else if (body.contains("content")) {
            prompt = body.at("content");
        } else {
            throw std::runtime_error("cache probe requires messages, prompt, or content");
        }

        auto tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, true, true, ctx_server.init_opt);
        if (tokenized_prompts.size() != 1) {
            throw std::runtime_error("cache probe expects exactly one prompt");
        }

        res->ok(ctx_server.probe_prompt_cache(tokenized_prompts[0]));
        return res;
    };

    this->get_models = [this](const server_http_req &) {
        auto res = create_response(true);
        // note: do NOT use ctx_server here, this endpoint must be accessible during sleep
        if (queue_tasks.is_sleeping()) {
            std::unique_lock<std::mutex> lock(mutex_cache);
            res->ok(cached_models);
        } else {
            res->ok(get_res_models(*meta));
        }
        return res;
    };

    this->post_tokenize = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        json tokens_response = json::array();
        if (body.count("content") != 0) {
            const bool add_special = json_value(body, "add_special", false);
            const bool parse_special = json_value(body, "parse_special", true);
            const bool with_pieces = json_value(body, "with_pieces", false);

            llama_tokens tokens = tokenize_mixed(ctx_server.vocab, body.at("content"), add_special, parse_special);

            if (with_pieces) {
                for (const auto& token : tokens) {
                    std::string piece = common_token_to_piece(ctx_server.vocab, token);
                    json piece_json;

                    // Check if the piece is valid UTF-8
                    if (is_valid_utf8(piece)) {
                        piece_json = piece;
                    } else {
                        // If not valid UTF-8, store as array of byte values
                        piece_json = json::array();
                        for (unsigned char c : piece) {
                            piece_json.push_back(static_cast<int>(c));
                        }
                    }

                    tokens_response.push_back({
                        {"id", token},
                        {"piece", piece_json}
                    });
                }
            } else {
                tokens_response = tokens;
            }
        }

        res->ok(json{{"tokens", std::move(tokens_response)}});
        return res;
    };

    this->post_detokenize = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);

        std::string content;
        if (body.count("tokens") != 0) {
            const llama_tokens tokens = body.at("tokens").get<llama_tokens>();
            content = tokens_to_str(ctx_server.vocab, tokens);
        }

        res->ok(json{{"content", std::move(content)}});
        return res;
    };

    this->post_embeddings = [this](const server_http_req & req) {
        return handle_embeddings_impl(req, TASK_RESPONSE_TYPE_NONE);
    };

    this->post_embeddings_oai = [this](const server_http_req & req) {
        return handle_embeddings_impl(req, TASK_RESPONSE_TYPE_OAI_EMBD);
    };

    this->post_hidden_states = [this](const server_http_req & req) {
        return handle_hidden_states_impl(req);
    };

    this->post_rerank = [this](const server_http_req & req) {
        return handle_rerank_impl(req);
    };

    this->post_tts = [this](const server_http_req & req) {
        auto res = create_response();
        res->set_req(&req); // will also set spipe if needed

        if (!ctx_server.has_cap_tts()) {
            res->error(format_error_response("this server does not support audio generation", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        const auto info = mtmd_gen_audio_get_info(ctx_server.mctx);

        json body = json::parse(req.body);

        // multipart form fields arrive as strings, correct the types of the numeric and boolean params
        for (const char * k : { "top_k", "n_predict", "max_tokens", "seed" }) {
            if (body.contains(k) && body[k].is_string()) {
                body[k] = std::stoll(body[k].get<std::string>());
            }
        }
        for (const char * k : { "top_p", "repeat_penalty" }) {
            if (body.contains(k) && body[k].is_string()) {
                body[k] = std::stof(body[k].get<std::string>());
            }
        }
        if (body.contains("stream") && body["stream"].is_string()) {
            body["stream"] = body["stream"] == "true";
        }

        std::string prompt = json_value(body, "input", json_value(body, "prompt", std::string()));
        if (prompt.empty()) {
            res->error(format_error_response("\"input\" must be a non-empty string", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        const std::string response_format = json_value(body, "response_format", std::string("wav"));

        server_task task(SERVER_TASK_TYPE_TTS);
        server_schema::eval_tts_schema(params, task, body);
        const bool stream = task.params.stream;
        task.tts_inp.set_prompt(prompt);
        task.tts_inp.set_lang(json_value(body, "lang", std::string()));
        task.tts_inp.data.stream    = stream;
        task.tts_inp.data.out_type  = response_format == "pcm"
            ? MTMD_HELPER_GEN_AUDIO_OUTTYPE_PCM
            : MTMD_HELPER_GEN_AUDIO_OUTTYPE_WAV;
        // note: -1 is clamped to 0 by the sampler, it will disable the penalty
        task.params.sampling.penalty_last_n = task.params.n_predict > 0 ? task.params.n_predict : 512;
        task.params.sampling.seed = task.tts_inp.data.seed; // same seed for the codec/vocoder RNG
        if (task.tts_inp.data.top_k > 0) {
            task.params.sampling.top_k = task.tts_inp.data.top_k;
        }
        if (task.tts_inp.data.top_p > 0) {
            task.params.sampling.top_p = task.tts_inp.data.top_p;
        }

        // speaker reference: either an uploaded form file ("speaker_ref") or a base64 JSON field ("speaker_ref_b64")
        const unsigned char * speaker_ref_data = nullptr;
        size_t speaker_ref_len  = 0;
        std::string speaker_ref_b64_decoded;

        auto speaker_ref_file = req.files.find("speaker_ref");
        if (speaker_ref_file != req.files.end()) {
            speaker_ref_data = speaker_ref_file->second.data.data();
            speaker_ref_len  = speaker_ref_file->second.data.size();
        } else {
            std::string speaker_ref_b64 = json_value(body, "speaker_ref_b64", std::string());
            if (!speaker_ref_b64.empty()) {
                try {
                    speaker_ref_b64_decoded = base64::decode(speaker_ref_b64);
                } catch (const std::exception &) {
                    res->error(format_error_response("\"speaker_ref_b64\" is not valid base64", ERROR_TYPE_INVALID_REQUEST));
                    return res;
                }
                speaker_ref_data = (const unsigned char *) speaker_ref_b64_decoded.data();
                speaker_ref_len  = speaker_ref_b64_decoded.size();
            }
        }

        if (speaker_ref_len > 0) {
            auto wrapper = mtmd_helper_bitmap_init_from_buf(ctx_server.mctx, speaker_ref_data, speaker_ref_len, false, ctx_server.init_opt);
            if (!wrapper.bitmap) {
                res->error(format_error_response("failed to decode \"speaker_ref\"", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            task.tts_inp.set_speaker_ref(mtmd::bitmap_ptr(wrapper.bitmap));
        } else {
            SRV_WRN("%s", "no speaker reference provided, the model may behave randomly\n");
        }

        auto & rd = res->rd;
        task.id = rd.get_new_id();
        rd.post_task(std::move(task));

        // raw float32 LE mono samples; audio/L16 would imply 16-bit big-endian (RFC 2586)
        const std::string content_type = response_format == "pcm"
            ? "audio/pcm;rate=" + std::to_string(info.sample_rate) + ";encoding=float;bits=32"
            : "audio/wav";

        if (!stream) {
            auto result = rd.next(req.should_stop);
            if (!result) {
                GGML_ASSERT(req.should_stop());
                return res; // connection is closed
            }
            if (result->is_error()) {
                res->error(result->to_json());
                return res;
            }
            auto * tts_res = dynamic_cast<server_task_result_tts *>(result.get());
            GGML_ASSERT(tts_res != nullptr);
            res->status       = 200;
            res->content_type = content_type;
            res->data         = std::move(tts_res->audio);
            return res;
        } else {
            auto first_result = rd.next(req.should_stop);
            if (!first_result) {
                GGML_ASSERT(req.should_stop());
                return res; // connection is closed
            }
            if (first_result->is_error()) {
                res->error(first_result->to_json());
                return res;
            }
            auto * first_tts_res = dynamic_cast<server_task_result_tts *>(first_result.get());
            GGML_ASSERT(first_tts_res != nullptr);

            res->status       = 200;
            res->content_type = content_type;
            res->data          = std::move(first_tts_res->audio);
            bool is_done       = first_tts_res->final;

            res->set_next([res_this = res.get(), is_done](std::string & output) mutable -> bool {
                // flush buffered audio before is_done - the first result can already be final
                if (!res_this->data.empty()) {
                    output = std::move(res_this->data);
                    res_this->data.clear();
                    return true;
                }
                if (is_done) {
                    return false;
                }
                if (res_this->should_stop()) {
                    return false;
                }

                server_response_reader & rd = res_this->rd;
                if (!rd.has_next()) {
                    return false;
                }
                auto result = rd.next([&res_this]() { return res_this->should_stop(); });
                if (!result || result->is_error()) {
                    return false;
                }
                auto * tts_res = dynamic_cast<server_task_result_tts *>(result.get());
                GGML_ASSERT(tts_res != nullptr);
                output  = std::move(tts_res->audio);
                is_done = tts_res->final;
                return true;
            });
        }

        return res;
    };

    this->post_systemone = [this](const server_http_req & req) {
        auto res = create_response();
        const auto & decision = ctx_server.decision;
        if (decision.type == COMMON_DECISION_TYPE_NONE) {
            res->error(format_error_response("This model is not a decision model", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        const json body = json::parse(req.body);
        const auto questions = decision.parse_questions(body);

        std::vector<raw_buffer> files;
        const json state = decision.prepare_state(decision.parse_state(body, files));
        if (!files.empty() && (!decision.can_use_images() || !meta->has_inp_image)) {
            res->error(format_error_response("This server does not support image input for decisions. For a model that supports it, start it with `--mmproj`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // one task per variant of each question, or one task for all the questions
        auto & rd = res->rd;
        {
            std::vector<server_task> tasks;
            if (decision.is_joint()) {
                server_task task = server_task(SERVER_TASK_TYPE_DECISION);
                task.id = rd.get_new_id();
                decision.fill_task_joint(state, questions, files, ctx_server.mctx, ctx_server.init_opt, task);
                tasks.push_back(std::move(task));
            } else {
                for (const auto & question : questions) {
                    for (size_t variant = 0; variant < decision.n_variants(question); variant++) {
                        server_task task = server_task(SERVER_TASK_TYPE_DECISION);
                        task.id = rd.get_new_id();
                        decision.fill_task(state, questions, question, variant, files, ctx_server.mctx, ctx_server.init_opt, task);
                        tasks.push_back(std::move(task));
                    }
                }
            }
            if (decision.can_share_prompt()) {
                tasks = server_decision_group_tasks(std::move(tasks), params.n_parallel);
            }
            rd.post_tasks(std::move(tasks));
        }

        auto all_results = rd.wait_for_all(req.should_stop);

        if (all_results.is_terminated) {
            return res; // connection is closed
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        }

        json answers = json::object();
        int32_t n_tokens = 0;
        size_t i_result = 0;
        size_t i_score  = 0;
        for (const auto & question : questions) {
            std::vector<std::vector<float>> scores;
            if (decision.is_joint()) {
                // one result with the scores of all the questions, in order
                auto * result = dynamic_cast<server_task_result_decision *>(all_results.results[0].get());
                GGML_ASSERT(result != nullptr && i_score + question.options.size() <= result->scores.size());
                scores.emplace_back(result->scores.begin() + i_score, result->scores.begin() + i_score + question.options.size());
                i_score += question.options.size();
                n_tokens = result->n_tokens;
            }
            for (size_t variant = 0; !decision.is_joint() && variant < decision.n_variants(question); variant++) {
                auto * result = dynamic_cast<server_task_result_decision *>(all_results.results[i_result++].get());
                GGML_ASSERT(result != nullptr);
                scores.push_back(result->scores);
                n_tokens += result->n_tokens;
            }
            answers[question.id] = decision.format_answer(question, scores, !files.empty());
        }

        res->ok(json{
            {"model",   meta->model_name},
            {"answers", answers},
            {"usage",   {
                {"input_tokens",  n_tokens},
                {"output_tokens", 0},
            }},
        });
        return res;
    };

    this->get_lora_adapters = [this](const server_http_req & req) {
        auto res = create_response();

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_GET_LORA);
            task.id = rd.get_new_id();
            rd.post_task(std::move(task));
        }

        // get the result
        auto result = rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_get_lora*>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    };

    this->post_lora_adapters = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        if (!body.is_array()) {
            res->error(format_error_response("Request body must be an array", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_SET_LORA);
            task.id = rd.get_new_id();
            task.set_lora = parse_lora_request(body);
            rd.post_task(std::move(task));
        }

        // get the result
        auto result = rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_apply_lora*>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    };
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_save(const server_http_req & req, int id_slot) {
    auto res = create_response();
    const json request_data = json::parse(req.body);
    std::string filename = request_data.at("filename");
    if (!fs_validate_filename(filename)) {
        res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    std::string filepath = params.slot_save_path + filename;

    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_SAVE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot  = id_slot;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_restore(const server_http_req & req, int id_slot) {
    auto res = create_response();
    const json request_data = json::parse(req.body);
    std::string filename = request_data.at("filename");
    if (!fs_validate_filename(filename)) {
        res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    std::string filepath = params.slot_save_path + filename;

    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_RESTORE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot  = id_slot;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_slot_save_load*>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_erase(const server_http_req & req, int id_slot) {
    auto res = create_response();
    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_ERASE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot = id_slot;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_slot_erase*>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_embeddings_impl(const server_http_req & req, task_response_type res_type) {
    auto res = create_response();
    if (!params.embedding) {
        res->error(format_error_response("This server does not support embeddings. Start it with `--embeddings`", ERROR_TYPE_NOT_SUPPORTED));
        return res;
    }

    if (res_type != TASK_RESPONSE_TYPE_NONE && meta->pooling_type == LLAMA_POOLING_TYPE_NONE) {
        res->error(format_error_response("Pooling type 'none' is not OAI compatible. Please use a different pooling type", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    const json body = json::parse(req.body);

    // for the shape of input/content, see tokenize_input_prompts()
    json prompt;
    if (body.count("input") != 0) {
        prompt = body.at("input");
    } else if (body.contains("content")) {
        res_type = TASK_RESPONSE_TYPE_NONE; // "content" field is not OAI compatible
        prompt = body.at("content");
    } else {
        res->error(format_error_response("\"input\" or \"content\" must be provided", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    bool use_base64 = false;
    if (body.count("encoding_format") != 0) {
        const std::string & format = body.at("encoding_format");
        if (format == "base64") {
            use_base64 = true;
        } else if (format != "float") {
            res->error(format_error_response("The format to return the embeddings in. Can be either float or base64", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    // same shapes as tokenize_input_prompts(), plus OAI content: { "content": [ { "type": "text"|"image_url"|"input_audio"|"input_video", ... } ] }
    auto tokenize_entry = [&](const json & p) {
        if (p.is_object() && p.contains("content")) {
            return tokenize_oai_content_array(ctx_server.vocab, ctx_server.mctx, meta->chat_params, p.at("content"), true, true, ctx_server.init_opt);
        }
        return tokenize_input_subprompt(ctx_server.vocab, ctx_server.mctx, p, true, true, ctx_server.init_opt);
    };

    // dispatch to the appropriate tokenizer based on the shape of the input
    std::vector<server_tokens> tokenized_prompts;
    if (prompt.is_array() && !json_is_array_and_contains_numbers(prompt)) {
        for (const auto & p : prompt) {
            tokenized_prompts.push_back(tokenize_entry(p));
        }
    } else {
        tokenized_prompts.push_back(tokenize_entry(prompt));
    }
    if (tokenized_prompts.empty()) {
        res->error(format_error_response("\"input\" must not be empty", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    for (const auto & tokens : tokenized_prompts) {
        // this check is necessary for models that do not add BOS token to the input
        if (tokens.empty()) {
            res->error(format_error_response("Input content cannot be empty", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    int embd_normalize = params.embd_normalize;
    if (body.count("embd_normalize") != 0) {
        embd_normalize = body.at("embd_normalize").get<int>();
        if (meta->pooling_type == LLAMA_POOLING_TYPE_NONE) {
            SRV_DBG("embd_normalize is not supported by pooling type %d, ignoring it\n", meta->pooling_type);
        }
    }

    // create and queue the task
    json responses = json::array();
    auto & rd = res->rd;
    {
        std::vector<server_task> tasks;
        for (size_t i = 0; i < tokenized_prompts.size(); i++) {
            server_task task = server_task(SERVER_TASK_TYPE_EMBEDDING);

            task.id     = rd.get_new_id();
            task.tokens = std::move(tokenized_prompts[i]);

            // OAI-compat
            task.params.res_type = res_type;
            task.params.embd_normalize = embd_normalize;

            tasks.push_back(std::move(task));
        }
        rd.post_tasks(std::move(tasks));
    }

    // wait for the results
    auto all_results = rd.wait_for_all(req.should_stop);

    // collect results
    if (all_results.is_terminated) {
        return res; // connection is closed
    } else if (all_results.error) {
        res->error(all_results.error->to_json());
        return res;
    } else {
        for (auto & res : all_results.results) {
            GGML_ASSERT(dynamic_cast<server_task_result_embd*>(res.get()) != nullptr);
            responses.push_back(res->to_json());
        }
    }

    // write JSON response
    json root = res_type == TASK_RESPONSE_TYPE_OAI_EMBD
        ? format_embeddings_response_oaicompat(body, meta->model_name, responses, use_base64)
        : json(responses);
    res->ok(root);
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_rerank_impl(const server_http_req & req) {
    auto res = create_response();
    if (!params.embedding || params.pooling_type != LLAMA_POOLING_TYPE_RANK) {
        res->error(format_error_response("This server does not support reranking. Start it with `--reranking`", ERROR_TYPE_NOT_SUPPORTED));
        return res;
    }

    const json body = json::parse(req.body);

    // if true, use TEI API format, otherwise use Jina API format
    // Jina: https://jina.ai/reranker/
    // TEI: https://huggingface.github.io/text-embeddings-inference/#/Text%20Embeddings%20Inference/rerank
    bool is_tei_format = body.contains("texts");

    // expand one rerank input item: plain string or {"text": str|[str], "image": url|data-uri|raw-base64|[urls]}
    auto parse_item = [&](const json & item, std::string & out_text, std::vector<raw_buffer> & out_files) {
        if (item.is_string()) {
            out_text = item.get<std::string>();
            return;
        }
        if (!item.is_object()) {
            throw std::invalid_argument("rerank input must be a string or an object of the form {\"text\": ..., \"image\": ...}");
        }
        if (item.contains("video") || item.contains("fps") || item.contains("max_frames")) {
            throw std::invalid_argument("video/fps/max_frames are not supported for rerank input");
        }
        json text = item.value("text", json());
        if (text.is_string()) {
            out_text += text.get<std::string>();
        } else if (text.is_array()) {
            for (const auto & t : text) {
                if (!t.is_string()) {
                    throw std::invalid_argument("\"text\" array elements must be strings");
                }
                out_text += t.get<std::string>();
            }
        } else if (!text.is_null()) {
            throw std::invalid_argument("\"text\" must be a string or an array of strings");
        }
        json image = item.value("image", json());
        auto add_image = [&](const std::string & url) {
            // marker is appended by format_prompt_rerank, one per decoded file
            try {
                handle_media(out_files, url, params.media_path);
            } catch (const std::runtime_error & e) {
                throw std::invalid_argument(e.what());
            }
        };
        if (image.is_string()) {
            add_image(image.get<std::string>());
        } else if (image.is_array()) {
            for (const auto & u : image) {
                if (!u.is_string()) {
                    throw std::invalid_argument("\"image\" array elements must be strings");
                }
                add_image(u.get<std::string>());
            }
        } else if (!image.is_null()) {
            throw std::invalid_argument("\"image\" must be a string or an array of strings");
        }
    };

    // Parse query
    std::string query_str;
    std::vector<raw_buffer> query_files;
    if (body.count("query") == 1) {
        try {
            parse_item(body.at("query"), query_str, query_files);
        } catch (const std::invalid_argument & e) {
            res->error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    } else {
        res->error(format_error_response("\"query\" must be provided", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    // Resolve Jina/TEI format ambiguity
    json documents_json = json_value(body, "documents",
        json_value(body, "texts", json::array()));
    if (!documents_json.is_array() || documents_json.empty()) {
        res->error(format_error_response("\"documents\" must be a non-empty array of strings or objects", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    // Parse documents
    std::vector<std::string>          documents(documents_json.size());
    std::vector<std::vector<raw_buffer>> document_files(documents_json.size());
    for (size_t i = 0; i < documents_json.size(); i++) {
        try {
            parse_item(documents_json[i], documents[i], document_files[i]);
        } catch (const std::invalid_argument & e) {
            res->error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    int top_n = json_value(body, "top_n", (int)documents.size());
    if (top_n < 0) {
        res->error(format_error_response("\"top_n\" must be >= 0", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    // create and queue the task
    json responses = json::array();
    auto & rd = res->rd;
    {
        std::vector<server_task> tasks;
        tasks.reserve(documents.size());
        for (size_t i = 0; i < documents.size(); i++) {
            auto tmp = format_prompt_rerank(ctx_server.model_tgt, ctx_server.vocab, ctx_server.mctx, query_str, documents[i], query_files, document_files[i], ctx_server.init_opt);
            server_task task = server_task(SERVER_TASK_TYPE_RERANK);
            task.id     = rd.get_new_id();
            task.tokens = std::move(tmp);
            tasks.push_back(std::move(task));
        }
        rd.post_tasks(std::move(tasks));
    }

    // wait for the results
    auto all_results = rd.wait_for_all(req.should_stop);

    // collect results
    if (all_results.is_terminated) {
        return res; // connection is closed
    } else if (all_results.error) {
        res->error(all_results.error->to_json());
        return res;
    } else {
        for (auto & res : all_results.results) {
            GGML_ASSERT(dynamic_cast<server_task_result_rerank*>(res.get()) != nullptr);
            responses.push_back(res->to_json());
        }
    }

    // write JSON response
    json root = format_response_rerank(
        body,
        meta->model_name,
        responses,
        is_tei_format,
        documents,
        top_n);

    res->ok(root);
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_count_tokens(const server_http_req & req, task_response_type res_type) {
    auto res = create_response();
    std::vector<raw_buffer> files;
    json body = json::parse(req.body);
    bool is_oai = false;

    switch (res_type) {
        case TASK_RESPONSE_TYPE_OAI_CHAT:
            {
                is_oai = true;
            } break;
        case TASK_RESPONSE_TYPE_OAI_RESP:
            {
                is_oai = true;
                body = server_chat_convert_responses_to_chatcmpl(body);
            } break;
        case TASK_RESPONSE_TYPE_ANTHROPIC:
            {
                body = server_chat_convert_anthropic_to_oai(body);
            } break;
        default:
            res->error(format_error_response("invalid res_type", ERROR_TYPE_INVALID_REQUEST));
            return res;
    }

    common_chat_session session; // dummy, unused
    json body_parsed = oaicompat_chat_params_parse(
            ctx_server.vocab,
            body,
            meta->chat_params,
            files,
            session);
    json prompt = body_parsed.at("prompt");
    // SRV_DBG("prompt = %s\n", prompt.dump().c_str());

    // TODO @ngxson : refactor this code block, move this to server-common and reuse it in other places
    size_t n_tokens;
    if (ctx_server.mctx != nullptr) {
        if (!prompt.is_string()) {
            throw std::runtime_error("for mtmd, input prompt must be a string.");
        }
        n_tokens = process_mtmd_prompt(ctx_server.mctx, prompt.get<std::string>(), files, ctx_server.init_opt, true).size();
    } else {
        n_tokens = tokenize_mixed(ctx_server.vocab, prompt, true, true).size();
    }

    json response = {{"input_tokens", static_cast<int64_t>(n_tokens)}};
    if (is_oai) {
        response["object"] = "response.input_tokens";
    }
    res->ok(response);
    return res;
}

void server_routes::update_cached_responses(bool is_sleeping) {
    // caller is task_queue, so ctx_server can be accessed without holding locks
    std::unique_lock<std::mutex> lock(mutex_cache);

    if (is_sleeping) {
        cached_models  = get_res_models(*meta);
        cached_props   = get_res_props(*meta, params, true);
        cached_metrics = ctx_server.get_metrics();

        should_reset_buckets = false;

        SRV_DBG("%s\n", "cached responses updated");

    } else if (should_reset_buckets) {
        // a scrape during sleep already reported these buckets
        ctx_server.reset_metrics_bucket();

        should_reset_buckets = false;
    }
}

std::unique_ptr<server_res_generator> server_routes::handle_hidden_states_impl(const server_http_req & req) {
    auto res = create_response();
    const json body = json::parse(req.body);

    if (!body.contains("prompt")) {
        res->error(format_error_response("\"prompt\" is required", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    const bool ltx25_layout = json_value(body, "layout", std::string()) == "ltx25";
    if (!ltx25_layout && !body.at("prompt").is_string()) {
        res->error(format_error_response("\"prompt\" must be provided as a string", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    if (ltx25_layout && !json_is_array_of_numbers(body.at("prompt"))) {
        res->error(format_error_response("LTX-2.5 layout requires prompt as an explicit token ID array", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    if (ltx25_layout && (ctx_server.model_tgt->arch != LLM_ARCH_GEMMA4 ||
            llama_model_n_layer(ctx_server.model_tgt) != 48 || llama_model_n_embd(ctx_server.model_tgt) != 3840)) {
        res->error(format_error_response("LTX-2.5 layout requires the 48-layer, 3840-wide Gemma4 model", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    if (ltx25_layout && meta->slot_n_ctx < 1024) {
        res->error(format_error_response("LTX-2.5 layout requires a context size of at least 1024", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    std::string pooling = json_value(body, "pooling", std::string("none"));
    if (pooling != "none" && pooling != "mean" && pooling != "cls" && pooling != "last") {
        res->error(format_error_response("pooling must be one of none, mean, cls, or last", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    std::string encoding_format = json_value(body, "encoding_format", std::string("json"));
    if (encoding_format != "json" && encoding_format != "base64") {
        res->error(format_error_response("encoding_format must be either json or base64", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    std::vector<int32_t> layers;
    if (!ltx25_layout) {
        try {
            layers = parse_hidden_state_layers(body, llama_model_n_layer(ctx_server.model_tgt));
        } catch (const std::exception & err) {
            res->error(format_error_response(err.what(), ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    } else {
        layers.push_back(LLAMA_HIDDEN_STATE_KEY_SCALED_INPUT);
        for (int32_t il = 0; il < 47; ++il) {
            layers.push_back(il);
        }
        layers.push_back(LLAMA_HIDDEN_STATE_KEY_FINAL_NORM);
    }

    json prompt = body.at("prompt");
    auto tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, true, true, ctx_server.init_opt);
    if (tokenized_prompts.size() != 1) {
        res->error(format_error_response("hidden-state extraction supports exactly one prompt per request", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    if (tokenized_prompts[0].empty()) {
        res->error(format_error_response("prompt cannot be empty", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    if (ltx25_layout && tokenized_prompts[0].size() > 1024) {
        res->error(format_error_response("LTX-2.5 token input exceeds 1024 tokens", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    std::vector<int32_t> token_indices;
    if (!ltx25_layout) {
        try {
            token_indices = parse_hidden_state_token_indices(body, (int32_t) tokenized_prompts[0].size());
        } catch (const std::exception & err) {
            res->error(format_error_response(err.what(), ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    auto & rd = res->rd;
    {
        server_task task = server_task(SERVER_TASK_TYPE_HIDDEN_STATE);
        task.id     = rd.get_new_id();
        task.tokens = std::move(tokenized_prompts[0]);
        task.hidden_state_layers = std::set<int32_t>(layers.begin(), layers.end());
        task.hidden_state_token_indices = std::move(token_indices);
        task.hidden_state_pooling = pooling;
        task.hidden_state_encoding_format = encoding_format;
        task.hidden_state_include_tokens = json_value(body, "include_tokens", false);
        task.hidden_state_ltx25 = ltx25_layout;
        if (ltx25_layout) {
            task.params.cache_prompt = false;
            task.hidden_state_include_tokens = true;
            task.hidden_state_position_offset = 1024 - task.n_tokens();
        }

        rd.post_task(std::move(task));
    }

    auto all_results = rd.wait_for_all(req.should_stop);

    if (all_results.is_terminated) {
        return res;
    } else if (all_results.error) {
        res->error(all_results.error->to_json());
        return res;
    } else {
        GGML_ASSERT(all_results.results.size() == 1);
        auto * hidden_res = dynamic_cast<server_task_result_hidden_state*>(all_results.results[0].get());
        GGML_ASSERT(hidden_res != nullptr);
        if (ltx25_layout) {
            res->status = 200;
            res->content_type = "application/vnd.serverless-host.ltx25-hidden-states";
            res->data = std::move(hidden_res->binary_data);
        } else {
            json root = hidden_res->to_json();
            root["model"] = meta->model_name;
            res->ok(root);
        }
    }

    return res;
}
