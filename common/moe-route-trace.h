#pragma once

#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

struct common_moe_trace_config {
    std::string path;
    bool routes = true;
    size_t max_records = 50000;
    size_t max_bytes = 64 * 1024 * 1024;
    int max_q = 6;
};

enum class common_moe_trace_role { target, draft };
enum class common_moe_trace_phase { target, draft, process };

// Only numeric routing data and fixed-schema phase events belong in this sink.
class common_moe_trace {
public:
    explicit common_moe_trace(const common_moe_trace_config & config, int n_parallel);
    ~common_moe_trace();
    common_moe_trace(const common_moe_trace &) = delete;
    common_moe_trace & operator=(const common_moe_trace &) = delete;

    static std::unique_ptr<common_moe_trace> from_env(int n_parallel);
    bool routes_enabled() const;
    void * callback_data(common_moe_trace_role role);
    static bool callback(ggml_tensor * tensor, bool ask, void * data);

    void metadata(const std::string & revision, const std::string & model_file,
                  const std::string & architecture, int experts, int top_k);
    void begin(common_moe_trace_phase phase, int q, bool prompt, bool replay);
    void end(int status, bool synchronized);
    void proposed(size_t count);
    void accepted(size_t proposed, size_t accepted, bool replay, bool needs_replay);
    void finish(const char * status);

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

class common_moe_trace_scope {
public:
    common_moe_trace_scope(common_moe_trace * trace, common_moe_trace_phase phase,
                           int q = -1, bool prompt = false, bool replay = false);
    ~common_moe_trace_scope();
    void end(int status, bool synchronized = false);
private:
    common_moe_trace * trace;
};
