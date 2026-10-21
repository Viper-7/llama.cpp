#include "moe-route-trace.h"
#include "ggml-cpu.h"
#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <unistd.h>

using json = nlohmann::ordered_json;
static void check(bool value) { if (!value) { throw std::runtime_error("route trace test failed"); } }
static std::vector<json> read(const std::string & path) {
    std::ifstream stream(path);
    check(bool(stream));
    std::vector<json> rows;
    for (std::string line; std::getline(stream, line);) { rows.push_back(json::parse(line)); }
    return rows;
}
struct fixture {
    ggml_context * ctx = ggml_init({1024 * 1024, nullptr, true});
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_buffer_t buffer = nullptr;
    fixture() { check(ctx && cpu); ggml_backend_cpu_set_n_threads(cpu, 2); }
    ~fixture() { ggml_backend_buffer_free(buffer); ggml_backend_free(cpu); ggml_free(ctx); }
    void allocate() { buffer = ggml_backend_alloc_ctx_tensors(ctx, cpu); check(buffer != nullptr); }
};
static void emit_tensor(common_moe_trace & trace, ggml_tensor * tensor, common_moe_trace_role role) {
    void * data = trace.callback_data(role);
    check(common_moe_trace::callback(tensor, true, data));
    check(common_moe_trace::callback(tensor, false, data));
}
static void test_strided(const std::string & path) {
    fixture f;
    auto * id_storage = ggml_new_tensor_2d(f.ctx, GGML_TYPE_I32, 16, 6);
    auto * ids = ggml_view_2d(f.ctx, id_storage, 8, 6, 16 * 4, 0);
    ggml_set_name(ids, "ffn_moe_topk-3");
    auto * weight_storage = ggml_new_tensor_1d(f.ctx, GGML_TYPE_F32, 256);
    auto * weights = ggml_view_3d(f.ctx, weight_storage, 1, 8, 6, 8, 128, 0);
    ggml_set_name(weights, "ffn_moe_weights_scaled-3");
    f.allocate();
    std::vector<int32_t> id_data(16 * 6, -12345);
    std::vector<float> weight_data(256, 98765.0f);
    for (int q = 0; q < 6; ++q) {
        for (int k = 0; k < 8; ++k) {
            id_data[q * 16 + k] = (q + k) % 16;
            weight_data[q * 32 + k * 2] = std::nextafter(float(q * 8 + k) / 49, 1.0f);
        }
    }
    ggml_backend_tensor_set(id_storage, id_data.data(), 0, id_data.size() * 4);
    ggml_backend_tensor_set(weight_storage, weight_data.data(), 0, weight_data.size() * 4);
    common_moe_trace trace({path}, 1);
    trace.metadata("test", "synthetic", "glm5next", 16, 8);
    auto * data = trace.callback_data(common_moe_trace_role::target);
    check(!common_moe_trace::callback(ids, true, data)); // warmup
    trace.begin(common_moe_trace_phase::target, 6, false, false);
    for (const char * name : {"ffn_moe_probs-3", "ffn_moe_weights_sum-3", "ffn_moe_topk-3-copy",
                              "ffn_moe_topk-", "ffn_moe_topk-3.secret", "result_output", "inp_tokens"}) {
        ggml_set_name(ids, name);
        check(!common_moe_trace::callback(ids, true, data));
    }
    ggml_set_name(ids, "ffn_moe_topk-3");
    check(!common_moe_trace::callback(ids, true, trace.callback_data(common_moe_trace_role::draft)));
    emit_tensor(trace, ids, common_moe_trace_role::target);
    emit_tensor(trace, weights, common_moe_trace_role::target);
    trace.end(0, true);
    trace.accepted(5, 3, false, false);
    trace.finish("request_complete");
    const auto rows = read(path);
    std::vector<json> routes;
    for (const auto & row : rows) { if (row["kind"] == "route") { routes.push_back(row); } }
    check(routes.size() == 2 && routes[0]["values"].size() == 48 && routes[1]["values"].size() == 48);
    for (int q = 0; q < 6; ++q) {
        for (int k = 0; k < 8; ++k) {
            check(routes[0]["values"][q * 8 + k].get<int>() == id_data[q * 16 + k]);
            check(routes[1]["values"][q * 8 + k].get<float>() == weight_data[q * 32 + k * 2]);
        }
    }
    check(rows.back()["status"] == "request_complete");
    struct stat info{};
    check(stat(path.c_str(), &info) == 0 && (info.st_mode & 0777) == 0600);
    bool refused = false;
    try { common_moe_trace other({path}, 1); } catch (const std::runtime_error &) { refused = true; }
    check(refused && read(path).size() == rows.size());
}
static void test_scheduler(const std::string & path) {
    common_moe_trace trace({path}, 1);
    trace.metadata("test", "synthetic", "glm5next", 16, 8);
    for (int q : {1, 3, 6, 6}) {
        fixture f;
        auto * probs = ggml_new_tensor_2d(f.ctx, GGML_TYPE_F32, 16, q);
        auto * ids = ggml_argsort_top_k(f.ctx, probs, 8);
        ggml_set_name(ids, "ffn_moe_topk-3");
        auto * weights = ggml_get_rows(f.ctx, ggml_reshape_3d(f.ctx, probs, 1, 16, q), ids);
        ggml_set_name(weights, "ffn_moe_weights-3");
        auto * flat = ggml_reshape_2d(f.ctx, weights, 8, q);
        auto * norm = ggml_div(f.ctx, flat, ggml_sum_rows(f.ctx, flat));
        ggml_set_name(norm, "ffn_moe_weights_norm-3");
        auto * scaled = ggml_scale(f.ctx, ggml_reshape_3d(f.ctx, norm, 1, 8, q), 2.5f);
        ggml_set_name(scaled, "ffn_moe_weights_scaled-3");
        auto * graph = ggml_new_graph(f.ctx);
        ggml_build_forward_expand(graph, scaled);
        f.allocate();
        std::vector<float> input(16 * q);
        for (int token = 0; token < q; ++token) {
            for (int expert = 0; expert < 16; ++expert) { input[token * 16 + expert] = 1 + (expert + token) % 16; }
        }
        ggml_backend_tensor_set(probs, input.data(), 0, input.size() * 4);
        ggml_backend_sched_t sched = ggml_backend_sched_new(&f.cpu, nullptr, 1, 256, false, true);
        check(sched != nullptr);
        ggml_backend_sched_set_eval_callback(sched, common_moe_trace::callback,
            trace.callback_data(common_moe_trace_role::target));
        common_moe_trace_scope scope(&trace, common_moe_trace_phase::target, q, false, q == 3);
        check(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);
        if (q == 6) { check(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS); }
        scope.end(0, true);
        ggml_backend_sched_free(sched);
    }
    trace.begin(common_moe_trace_phase::draft, -1, false, false);
    trace.end(0, false);
    trace.proposed(5);
    trace.accepted(5, 2, false, true);
    trace.finish("request_complete");
    const auto rows = read(path);
    int n_routes = 0, repeated_invocations = 0;
    for (const auto & row : rows) {
        if (row["kind"] != "route") { continue; }
        ++n_routes;
        if (row["layer_call"] == 2) { ++repeated_invocations; }
        const int q = row["q"];
        check(row["values"].size() == size_t(q * 8));
        if (row["stage"] == "ffn_moe_topk") {
            for (int t = 0; t < q; ++t) {
                for (int s = 0; s < 8; ++s) { check(row["values"][t * 8 + s] == (15 - s - t + 16) % 16); }
            }
        }
        if (row["stage"] == "ffn_moe_weights_scaled") {
            for (int t = 0; t < q; ++t) {
                float sum = 0;
                for (int s = 0; s < 8; ++s) { sum += row["values"][t * 8 + s].get<float>(); }
                check(std::abs(sum - 2.5f) < 1e-6f);
            }
        }
        if (row["stage"] != "ffn_moe_topk") {
            for (int t = 0; t < q; ++t) {
                for (int s = 0; s < 8; ++s) {
                    float expected = 16.0f - s;
                    if (row["stage"] != "ffn_moe_weights") { expected /= 100.0f; }
                    if (row["stage"] == "ffn_moe_weights_scaled") { expected *= 2.5f; }
                    check(std::abs(row["values"][t * 8 + s].get<float>() - expected) < 1e-7f);
                }
            }
        }
    }
    check(n_routes == 24 && repeated_invocations == 8 && rows.back()["status"] == "request_complete");
}
static void test_limits(const std::string & dir) {
    fixture f;
    auto * ids = ggml_new_tensor_2d(f.ctx, GGML_TYPE_I32, 1, 7);
    ggml_set_name(ids, "ffn_moe_topk-3");
    auto * tail_ids = ggml_view_2d(f.ctx, ids, 1, 1, ids->nb[1], 0);
    ggml_set_name(tail_ids, "ffn_moe_topk-3");
    f.allocate();
    const int32_t values[] = {0, 0, 0, 0, 0, 0, 0};
    ggml_backend_tensor_set(ids, values, 0, sizeof(values));
    {
        common_moe_trace trace({dir + "/q-filter"}, 1);
        trace.metadata("test", "synthetic", "glm5next", 16, 1);
        trace.begin(common_moe_trace_phase::target, 7, true, false);
        check(!common_moe_trace::callback(ids, true, trace.callback_data(common_moe_trace_role::target)));
        emit_tensor(trace, tail_ids, common_moe_trace_role::target);
        trace.end(0, true);
        trace.finish("request_complete");
        check(read(dir + "/q-filter").back()["skipped_q_tensors"] == 1);
        for (const auto & row : read(dir + "/q-filter")) {
            if (row["kind"] == "route") { check(row["layer_call"] == 2 && row["q"] == 1); }
        }
    }
    {
        common_moe_trace_config config{dir + "/phases"}; config.routes = false;
        common_moe_trace trace(config, 1);
        trace.begin(common_moe_trace_phase::target, 6, false, false);
        check(!trace.routes_enabled());
        check(!common_moe_trace::callback(ids, true, trace.callback_data(common_moe_trace_role::target)));
        trace.end(0, true);
    }
    {
        common_moe_trace_config config{dir + "/cap"}; config.max_records = 3;
        common_moe_trace trace(config, 1);
        trace.begin(common_moe_trace_phase::target, 1, false, false);
        trace.end(0, true);
        trace.proposed(5);
        const auto rows = read(config.path);
        check(rows.back()["status"] == "truncated" && rows[rows.size() - 2]["reason"] == "record_cap");
    }
    {
        common_moe_trace_config config{dir + "/byte-cap"}; config.max_bytes = 4096;
        common_moe_trace trace(config, 1);
        for (int i = 0; i < 100; ++i) { trace.proposed(5); }
        check(read(config.path).back()["status"] == "truncated");
        check(std::filesystem::file_size(config.path) <= config.max_bytes);
    }
    bool refused = false;
    try { common_moe_trace trace({dir + "/multislot"}, 2); } catch (...) { refused = true; }
    check(refused && !std::filesystem::exists(dir + "/multislot"));
    check(symlink((dir + "/q-filter").c_str(), (dir + "/symlink").c_str()) == 0);
    refused = false;
    try { common_moe_trace trace({dir + "/symlink"}, 1); } catch (...) { refused = true; }
    check(refused);
}
static void test_invalid_and_top1(const std::string & dir) {
    fixture f;
    auto * ids = ggml_new_tensor_2d(f.ctx, GGML_TYPE_I32, 1, 3);
    ggml_set_name(ids, "ffn_moe_topk-3");
    auto * norm = ggml_new_tensor_2d(f.ctx, GGML_TYPE_F32, 1, 3);
    ggml_set_name(norm, "ffn_moe_weights_norm-3");
    auto * scaled = ggml_new_tensor_3d(f.ctx, GGML_TYPE_F32, 1, 1, 3);
    ggml_set_name(scaled, "ffn_moe_weights_scaled-3");
    f.allocate();
    const int32_t id_values[] = {0, 1, 2};
    ggml_backend_tensor_set(ids, id_values, 0, sizeof(id_values));
    const float values[] = {1.0f, 1.0f, 1.0f};
    ggml_backend_tensor_set(norm, values, 0, sizeof(values));
    ggml_backend_tensor_set(scaled, values, 0, sizeof(values));
    common_moe_trace trace({dir + "/top1"}, 1);
    trace.metadata("test", "synthetic", "glm5next", 16, 1);
    trace.begin(common_moe_trace_phase::process, 3, false, false);
    emit_tensor(trace, ids, common_moe_trace_role::draft);
    emit_tensor(trace, norm, common_moe_trace_role::draft);
    emit_tensor(trace, scaled, common_moe_trace_role::draft);
    trace.end(0, true);
    trace.finish("request_complete");
    int routes = 0;
    for (const auto & row : read(dir + "/top1")) {
        if (row["kind"] == "route") {
            ++routes;
            check(row["q"] == 3 && row["top_k"] == 1 && row["role"] == "draft");
        }
    }
    check(routes == 3);
    const float bad[] = {1.0f, std::numeric_limits<float>::quiet_NaN(), 1.0f};
    ggml_backend_tensor_set(norm, bad, 0, sizeof(bad));
    common_moe_trace invalid({dir + "/invalid"}, 1);
    invalid.metadata("test", "synthetic", "glm5next", 16, 1);
    invalid.begin(common_moe_trace_phase::target, 3, false, false);
    emit_tensor(invalid, ids, common_moe_trace_role::target);
    // Capture fails closed, but the scheduler callback must not abort generation.
    emit_tensor(invalid, norm, common_moe_trace_role::target);
    check(read(dir + "/invalid").back()["status"] == "capture_error");
}
static void test_environment(const std::string & dir) {
    const std::string path = dir + "/environment";
    setenv("LLAMA_MOE_TRACE_FILE", path.c_str(), 1);
    setenv("LLAMA_MOE_TRACE_MODE", "phases", 1);
    setenv("LLAMA_MOE_TRACE_MAX_Q", "6", 1);
    setenv("LLAMA_MOE_TRACE_MAX_RECORDS", "10", 1);
    setenv("LLAMA_MOE_TRACE_MAX_BYTES", "4096", 1);
    auto trace = common_moe_trace::from_env(1);
    check(trace && !trace->routes_enabled());
    trace->finish("request_complete");
    const auto rows = read(path);
    check(rows.front()["mode"] == "phases" && rows.front()["max_route_q"] == 6);
    check(rows.front()["max_records"] == 10 && rows.front()["max_bytes"] == 4096);
    setenv("LLAMA_MOE_TRACE_MAX_Q", "0", 1);
    bool refused = false;
    try { auto invalid = common_moe_trace::from_env(1); } catch (...) { refused = true; }
    check(refused && read(path).size() == rows.size());
    for (const char * key : {"LLAMA_MOE_TRACE_FILE", "LLAMA_MOE_TRACE_MODE", "LLAMA_MOE_TRACE_MAX_Q",
                             "LLAMA_MOE_TRACE_MAX_RECORDS", "LLAMA_MOE_TRACE_MAX_BYTES"}) { unsetenv(key); }
}
int main() {
    char path[] = "/tmp/llama-moe-route-trace.XXXXXX";
    const char * dir = mkdtemp(path);
    if (!dir) { return 1; }
    try {
        // Do not let an operator's environment redirect this CPU-only unit test.
        unsetenv("LLAMA_MOE_TRACE_FILE");
        check(!common_moe_trace::from_env(1));
        check(!common_moe_trace::callback(nullptr, true, nullptr));
        common_moe_trace_scope disabled(nullptr, common_moe_trace_phase::target);
        disabled.end(0);
        test_strided(std::string(dir) + "/strided");
        test_scheduler(std::string(dir) + "/scheduler");
        test_limits(dir);
        test_invalid_and_top1(dir);
        test_environment(dir);
        if (std::getenv("LLAMA_MOE_TRACE_TEST_KEEP_ARTIFACTS")) {
            std::printf("synthetic CPU trace artifacts retained in %s\n", dir);
        } else {
            std::filesystem::remove_all(dir);
        }
        std::puts("route capture: CPU filtering, strides, scheduler, phases, limits, and file-safety tests passed");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s; test artifacts retained in %s\n", error.what(), dir);
        return 1;
    }
}
