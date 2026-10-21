#include "moe-route-trace.h"

#include <nlohmann/json.hpp>

#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <vector>

#if defined(__unix__)
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace {
using json = nlohmann::ordered_json;
using trace_clock = std::chrono::steady_clock;
const char * phase_name(common_moe_trace_phase phase) {
    switch (phase) {
        case common_moe_trace_phase::target:  return "target";
        case common_moe_trace_phase::draft:   return "draft";
        case common_moe_trace_phase::process: return "process";
    }
    return "invalid";
}
size_t env_count(const char * key, size_t fallback, size_t limit) {
    const char * raw = std::getenv(key);
    if (!raw) { return fallback; }
    if (!*raw) { throw std::runtime_error(std::string(key) + " must be a positive count"); }
    size_t value = 0;
    for (const char * p = raw; *p; ++p) {
        if (*p < '0' || *p > '9' || value > limit / 10) {
            throw std::runtime_error(std::string(key) + " is out of range");
        }
        value = value * 10 + size_t(*p - '0');
        if (value > limit) { throw std::runtime_error(std::string(key) + " is out of range"); }
    }
    if (!value) { throw std::runtime_error(std::string(key) + " must be positive"); }
    return value;
}
bool route_name(const char * name, std::string & stage, int & layer) {
    const char * names[] = {"ffn_moe_topk", "ffn_moe_weights", "ffn_moe_weights_softmax",
                           "ffn_moe_weights_norm", "ffn_moe_weights_scaled"};
    for (const char * prefix : names) {
        const size_t n = std::strlen(prefix);
        if (std::strncmp(name, prefix, n) != 0 || name[n] != '-' || !name[n + 1]) { continue; }
        int value = 0;
        for (const char * p = name + n + 1; *p; ++p) {
            if (*p < '0' || *p > '9' || value > 10000) { return false; }
            value = value * 10 + (*p - '0');
        }
        stage = prefix;
        layer = value;
        return true;
    }
    return false;
}
}

struct common_moe_trace::impl {
    struct callback_state { impl * trace; common_moe_trace_role role; };
    common_moe_trace_config config;
    FILE * file = nullptr;
    callback_state target{this, common_moe_trace_role::target};
    callback_state draft{this, common_moe_trace_role::draft};
    size_t records = 0, bytes = 0, skipped_q = 0;
    uint64_t call = 0, last_target = 0, last_draft = 0;
    std::map<int, uint64_t> layer_calls;
    int experts = 0, top_k = 0;
    bool active = false, failed = false;
    common_moe_trace_phase phase = common_moe_trace_phase::target;
    trace_clock::time_point start;
    ~impl() { if (file) { std::fclose(file); } }

    bool write(const json & event) {
        const std::string line = event.dump() + '\n';
        if (std::fwrite(line.data(), 1, line.size(), file) != line.size() || std::fflush(file) != 0) {
            failed = true;
            return false;
        }
        bytes += line.size();
        return true;
    }
    void finish(const char * status) {
        if (!file) { return; }
        write({{"kind", "final"}, {"status", failed ? "io_error" : status}, {"records", records},
               {"bytes_before_final", bytes}, {"skipped_q_tensors", skipped_q}, {"open_phase", active}});
        if (std::fclose(file) != 0 || failed) {
            // Fixed text only. A missing/invalid final record also makes the artifact incomplete.
            std::fputs("moe route capture: artifact write failed; capture is incomplete\n", stderr);
        }
        file = nullptr;
        active = false;
    }
    void emit(json event) {
        if (!file) { return; }
        const size_t event_bytes = event.dump().size() + 1;
        // Reserve space for explicit truncation and final records.
        if (records >= config.max_records || event_bytes + bytes + 2048 > config.max_bytes) {
            write({{"kind", "truncated"}, {"reason", records >= config.max_records ? "record_cap" : "byte_cap"}});
            finish("truncated");
            return;
        }
        ++records;
        event["record"] = records;
        if (!write(event)) { finish("io_error"); }
    }
    void fail() {
        emit({{"kind", "capture_error"}, {"reason", "invalid_routing_tensor"}});
        finish("capture_error");
    }
    bool collect(ggml_tensor * t, bool ask, common_moe_trace_role role) {
        if (!file || !active || !config.routes) { return ask ? false : true; }
        if ((phase == common_moe_trace_phase::target) != (role == common_moe_trace_role::target)) {
            return ask ? false : true;
        }
        std::string stage;
        int layer;
        if (!route_name(t->name, stage, layer)) { return ask ? false : true; }
        const bool ids = stage == "ffn_moe_topk";
        const bool rows_2d = ids || stage == "ffn_moe_weights_norm";
        const int64_t k = rows_2d ? t->ne[0] : t->ne[1];
        const int64_t q = rows_2d ? t->ne[1] : t->ne[2];
        // Count filtered top-k nodes too, so a short tail keeps its true ordinal.
        if (ids && ask) { ++layer_calls[layer]; }
        if (q > config.max_q) { if (ask) { ++skipped_q; } return ask ? false : true; }
        if (ask) { return true; }
        if (k <= 0 || q <= 0 || k > 1024 || top_k <= 0 || experts <= 0 || k != top_k ||
            t->ne[3] != 1 || (ids && (t->type != GGML_TYPE_I32 || t->ne[2] != 1)) ||
            (!ids && t->type != GGML_TYPE_F32) || t->nb[0] != 4 ||
            (rows_2d ? t->ne[2] != 1 : t->ne[0] != 1)) {
            throw std::runtime_error("invalid route geometry");
        }
        if (layer_calls[layer] == 0) { throw std::runtime_error("coefficient without top-k"); }
        // Read only actual selected values, not padding/full expert probability banks.
        json values = json::array();
        std::vector<uint8_t> row(size_t(k) * 4);
        const size_t slot_stride = rows_2d ? t->nb[0] : t->nb[1];
        const size_t row_stride = rows_2d ? t->nb[1] : t->nb[2];
        for (int64_t token = 0; token < q; ++token) {
            if (slot_stride == 4) {
                ggml_backend_tensor_get(t, row.data(), size_t(token) * row_stride, row.size());
            }
            for (int64_t slot = 0; slot < k; ++slot) {
                uint8_t value[4];
                if (slot_stride == 4) { std::memcpy(value, row.data() + slot * 4, 4); }
                else { ggml_backend_tensor_get(t, value, size_t(token) * row_stride + size_t(slot) * slot_stride, 4); }
                if (ids) {
                    int32_t id;
                    std::memcpy(&id, value, 4);
                    if (id < 0 || id >= experts) { throw std::runtime_error("invalid expert ID"); }
                    values.push_back(id);
                } else {
                    float weight;
                    std::memcpy(&weight, value, 4);
                    if (!std::isfinite(weight)) { throw std::runtime_error("invalid coefficient"); }
                    values.push_back(weight);
                }
            }
        }
        emit({{"kind", "route"}, {"call", call}, {"role", role == common_moe_trace_role::target ? "target" : "draft"},
              {"phase", phase_name(phase)}, {"layer", layer}, {"layer_call", layer_calls[layer]},
              {"q", q}, {"top_k", k}, {"stage", stage},
              {"dtype", ids ? "i32" : "f32"}, {"order", "token_then_original_slot"}, {"values", std::move(values)}});
        return true;
    }
};

common_moe_trace::common_moe_trace(const common_moe_trace_config & config, int n_parallel) : pimpl(new impl) {
    if (n_parallel != 1 || config.path.empty() || config.path[0] != '/' || config.max_records == 0 ||
        config.max_records > 1000000 || config.max_bytes < 4096 || config.max_bytes > 1024 * 1024 * 1024 ||
        config.max_q < 1 || config.max_q > 256) {
        throw std::runtime_error("route capture requires one slot, an absolute new path, and finite limits");
    }
    pimpl->config = config;
#if defined(__unix__)
    const int fd = open(config.path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
    if (fd < 0) { throw std::runtime_error("cannot exclusively create route capture artifact"); }
    if (fchmod(fd, 0600) != 0) { close(fd); throw std::runtime_error("cannot restrict route capture permissions"); }
    pimpl->file = fdopen(fd, "w");
    if (!pimpl->file) { close(fd); throw std::runtime_error("cannot open route capture stream"); }
#else
    throw std::runtime_error("exclusive private route capture requires a POSIX host");
#endif
    pimpl->emit({{"kind", "contract"}, {"version", 1}, {"mode", config.routes ? "routes" : "phases"},
        {"max_records", config.max_records}, {"max_bytes", config.max_bytes}, {"max_route_q", config.max_q},
        {"warmup_excluded", true}, {"single_slot", true}, {"timing", config.routes ? "instrumented_not_performance" : "host_scopes_not_DMA_time"},
        {"contents", "expert_IDs_coefficients_and_phase_counts_only"}});
}
common_moe_trace::~common_moe_trace() { try { finish("context_closed"); } catch (...) {} }
std::unique_ptr<common_moe_trace> common_moe_trace::from_env(int n_parallel) {
    const char * path = std::getenv("LLAMA_MOE_TRACE_FILE");
    if (!path || !*path) { return nullptr; }
    common_moe_trace_config config;
    config.path = path;
    const char * mode = std::getenv("LLAMA_MOE_TRACE_MODE");
    if (mode && std::strcmp(mode, "routes") != 0 && std::strcmp(mode, "phases") != 0) {
        throw std::runtime_error("LLAMA_MOE_TRACE_MODE must be routes or phases");
    }
    config.routes = !mode || std::strcmp(mode, "routes") == 0;
    config.max_records = env_count("LLAMA_MOE_TRACE_MAX_RECORDS", config.max_records, 1000000);
    config.max_bytes = env_count("LLAMA_MOE_TRACE_MAX_BYTES", config.max_bytes, 1024 * 1024 * 1024);
    config.max_q = int(env_count("LLAMA_MOE_TRACE_MAX_Q", config.max_q, 256));
    return std::unique_ptr<common_moe_trace>(new common_moe_trace(config, n_parallel));
}
bool common_moe_trace::routes_enabled() const { return pimpl->config.routes; }
void * common_moe_trace::callback_data(common_moe_trace_role role) {
    return role == common_moe_trace_role::target ? &pimpl->target : &pimpl->draft;
}
bool common_moe_trace::callback(ggml_tensor * tensor, bool ask, void * data) {
    if (!data) { return ask ? false : true; }
    auto * state = static_cast<impl::callback_state *>(data);
    try { return state->trace->collect(tensor, ask, state->role); }
    catch (...) { try { state->trace->fail(); } catch (...) {} return ask ? false : true; }
}
void common_moe_trace::metadata(const std::string & revision, const std::string & model_file,
                              const std::string & architecture, int experts, int top_k) {
    pimpl->experts = experts;
    pimpl->top_k = top_k;
    pimpl->emit({{"kind", "metadata"}, {"revision", revision}, {"model_file", model_file},
                 {"architecture", architecture}, {"experts", experts}, {"top_k", top_k}});
}
void common_moe_trace::begin(common_moe_trace_phase phase, int q, bool prompt, bool replay) {
    if (!pimpl->file) { return; }
    if (pimpl->active) { pimpl->finish("nested_phase_error"); return; }
    pimpl->phase = phase;
    pimpl->layer_calls.clear();
    pimpl->active = true;
    ++pimpl->call;
    if (phase == common_moe_trace_phase::target) { pimpl->last_target = pimpl->call; }
    if (phase == common_moe_trace_phase::draft) { pimpl->last_draft = pimpl->call; }
    pimpl->emit({{"kind", "phase_begin"}, {"call", pimpl->call}, {"phase", phase_name(phase)},
                 {"role", phase == common_moe_trace_phase::target ? "target" : "draft"},
                 {"batch_q", q}, {"prompt", prompt}, {"replay", replay}});
    pimpl->start = trace_clock::now();
}
void common_moe_trace::end(int status, bool synchronized) {
    if (!pimpl->file || !pimpl->active) { return; }
    const double ms = std::chrono::duration<double, std::milli>(trace_clock::now() - pimpl->start).count();
    pimpl->emit({{"kind", "phase_end"}, {"call", pimpl->call}, {"phase", phase_name(pimpl->phase)},
                 {"role", pimpl->phase == common_moe_trace_phase::target ? "target" : "draft"},
                 {"status", status}, {"host_interval_ms", ms}, {"explicit_context_sync", synchronized}});
    pimpl->active = false;
}
void common_moe_trace::proposed(size_t count) {
    pimpl->emit({{"kind", "proposed"}, {"draft_call", pimpl->last_draft}, {"count", count}});
}
void common_moe_trace::accepted(size_t proposed, size_t accepted, bool replay, bool needs_replay) {
    pimpl->emit({{"kind", "accept"}, {"target_call", pimpl->last_target}, {"proposed", proposed},
                 {"verified_prefix_drafts", accepted}, {"replay", replay}, {"needs_replay", needs_replay}});
}
void common_moe_trace::finish(const char * status) { pimpl->finish(status); }
common_moe_trace_scope::common_moe_trace_scope(common_moe_trace * trace, common_moe_trace_phase phase,
                                               int q, bool prompt, bool replay) : trace(trace) {
    if (trace) { trace->begin(phase, q, prompt, replay); }
}
common_moe_trace_scope::~common_moe_trace_scope() {
    if (trace) { try { trace->end(-999, false); } catch (...) {} }
}
void common_moe_trace_scope::end(int status, bool synchronized) {
    if (trace) { trace->end(status, synchronized); trace = nullptr; }
}
