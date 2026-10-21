// Standalone static expert-ownership experiment. This is deliberately not model
// wiring: router/top-k and the shared expert are outside BOTH measured branches.
// Default execution creates CPU backends only; GPU use requires --devices.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>
extern "C" {
#include "../vendor/hash/sha256/sha256.h"
}

namespace {
using clock_type = std::chrono::steady_clock;
double elapsed_ms(clock_type::time_point start) {
    return std::chrono::duration<double, std::milli>(clock_type::now() - start).count();
}
void require(bool condition, const std::string & message) {
    if (!condition) { throw std::runtime_error(message); }
}
std::vector<std::string> split(const std::string & value) {
    std::vector<std::string> out;
    size_t start = 0;
    do {
        size_t end = value.find(',', start);
        out.push_back(value.substr(start, end - start));
        if (end == std::string::npos) { break; }
        start = end + 1;
    } while (true);
    return out;
}
void set_test_mmq_j(int J) {
    const std::string value = J ? std::to_string(J) : "";
#ifdef _WIN32
    require(_putenv_s("GGML_CUDA_TEST_MMQ_FORCE_J", value.c_str()) == 0, "cannot set MMQ J override");
#else
    require(J ? setenv("GGML_CUDA_TEST_MMQ_FORCE_J", value.c_str(), 1) == 0 :
        unsetenv("GGML_CUDA_TEST_MMQ_FORCE_J") == 0, "cannot set MMQ J override");
#endif
}
struct config {
    int d = 32, f = 64, experts = 16, used = 4, layer = 3;
    int repetitions = 3, warmups = 1;
    float clamp = 7.0f;
    bool inspect = false, quant = false;
    bool native_only = false;
    int mmq_force_j = 0;
    bool mmq_force_j_set = false;
    std::vector<int> tokens = {1, 6, 17};
    std::vector<std::string> patterns = {"balanced", "skew", "random"};
    std::vector<std::string> devices = {"CPU", "CPU"};
    std::vector<std::string> models;
    std::string route_capture, sample_route_capture;
    uint32_t sample_seed = 42;
    int route_case = -1;
    std::vector<std::string> coefficient_stages;
    std::string sequence, reference_in, reference_out;
    size_t cache_entries = 128, cache_bytes = 256 * 1024 * 1024;
};
config parse(int argc, char ** argv) {
    config c;
    bool synthetic_top_k = false;
    bool synthetic_cases = false;
    bool explicit_patterns = false;
    bool sample_seed_set = false;
    bool explicit_warm_controls = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help") {
            std::puts("CPU-default complete routed-FFN static expert-parallel fixture.\n"
                "  --devices CPU,CPU | ROCm0,ROCm1  (explicit allocation authorization required)\n"
                "  --model SHARD.gguf   repeat for shards containing the selected layer\n"
                "  --layer N           default 3; read real gate/up/down banks and clamp\n"
                "  --inspect           metadata only, no backend/device initialization\n"
                "  --native-only       one-device native routed FFN, no owner banks\n"
                "  --mmq-force-j J     test build only: compare native J to default before timing\n"
                "  --tokens 1,6,256     default 1,6,17\n"
                "  --patterns balanced,skew,random  deterministic synthetic routes\n"
                "  --repetitions N --warmups N\n"
                "  --quant             synthetic IQ3_S gate/up, IQ4_XS down, D=F=256\n"
                "  --top-k K           synthetic fixture only, default 4\n"
                "  --route-capture FILE  replay target non-prompt routes for --layer, q<=6\n"
                "  --sample-route-capture FILE  sample q from that layer's q<=6 route histogram\n"
                "  --sample-seed N     deterministic histogram sample seed, default 42\n"
                "  --route-case INDEX    select one zero-based eligible capture case\n"
                "  --sequence native|owner|bucket  complete chronological capture, three passes\n"
                "  --reference-out FILE  native-only, exclusive bounded reference artifact\n"
                "  --reference-in FILE   required for owner/bucket, independently produced native\n"
                "  --cache-entries N --cache-bytes N  per-device arena caps, no eviction\n"
                "Outputs JSONL; graph preparation/cold calls are separate from warm samples.\n"
                "Four controls: native, packed-local, flat EP, cardinality-bucket EP.\n"
                "Warm samples include device route readback, host partition/upload, local\n"
                "packing, compute, owner combine, transport, root reduction, synchronization.\n"
                "Activations stay synthetic; capture replay is not a full-model oracle.\n"
                "Capture input is limited to 64MiB, 1MiB/line, 50000 records, 1024 cases.");
            std::exit(0);
        } else if (arg == "--inspect") { c.inspect = true;
        } else if (arg == "--native-only") { c.native_only = true;
        } else if (arg == "--quant") { c.quant = true; c.d = c.f = 256;
        } else {
            require(i + 1 < argc, "missing value for " + arg);
            const std::string value = argv[++i];
            if (arg == "--devices") { c.devices = split(value);
            } else if (arg == "--model") { c.models.push_back(value);
            } else if (arg == "--layer") { c.layer = std::stoi(value);
            } else if (arg == "--mmq-force-j") { c.mmq_force_j = std::stoi(value); c.mmq_force_j_set = true;
            } else if (arg == "--repetitions") { c.repetitions = std::stoi(value); explicit_warm_controls = true;
            } else if (arg == "--warmups") { c.warmups = std::stoi(value); explicit_warm_controls = true;
            } else if (arg == "--top-k") { c.used = std::stoi(value); synthetic_top_k = true;
            } else if (arg == "--route-capture") { c.route_capture = value;
            } else if (arg == "--sample-route-capture") { c.sample_route_capture = value;
            } else if (arg == "--sample-seed") {
                require(!value.empty() && value.find_first_not_of("0123456789") == std::string::npos, "invalid sample seed");
                const unsigned long long seed = std::stoull(value);
                require(seed <= UINT32_MAX, "sample seed out of range");
                c.sample_seed = uint32_t(seed);
                sample_seed_set = true;
            } else if (arg == "--route-case") { c.route_case = std::stoi(value);
            } else if (arg == "--sequence") { c.sequence = value;
            } else if (arg == "--reference-in") { c.reference_in = value;
            } else if (arg == "--reference-out") { c.reference_out = value;
            } else if (arg == "--cache-entries" || arg == "--cache-bytes") {
                require(!value.empty() && value.find_first_not_of("0123456789") == std::string::npos, "invalid cache cap");
                const size_t cap = std::stoull(value);
                if (arg == "--cache-entries") { c.cache_entries = cap; } else { c.cache_bytes = cap; }
            } else if (arg == "--patterns") { c.patterns = split(value); synthetic_cases = true; explicit_patterns = true;
            } else if (arg == "--tokens") {
                synthetic_cases = true;
                c.tokens.clear();
                for (const auto & t : split(value)) { c.tokens.push_back(std::stoi(t)); }
            } else { throw std::runtime_error("unknown option " + arg); }
        }
    }
    require(c.native_only ? c.devices.size() == 1 : (c.devices.size() >= 2 && c.devices.size() <= 6),
        c.native_only ? "native-only requires one device" : "need 2..6 owners");
#ifndef GGML_CUDA_TEST_MMQ_FORCE_J
    require(!c.mmq_force_j_set, "--mmq-force-j requires GGML_CUDA_TEST_MMQ_FORCE_J build");
#endif
    require(!c.mmq_force_j_set || (c.native_only && (c.mmq_force_j == 8 || c.mmq_force_j == 16 ||
        c.mmq_force_j == 24 || c.mmq_force_j == 32 || c.mmq_force_j == 64)),
        "--mmq-force-j requires --native-only and J in 8,16,24,32,64");
    require(!c.mmq_force_j_set || c.devices[0].compare(0, 4, "ROCm") == 0,
        "--mmq-force-j requires an explicit ROCm device");
    require(!c.native_only || c.sequence.empty(), "native-only does not support sequence mode");
    require(!synthetic_top_k || c.models.empty(), "--top-k is synthetic-only; real top-k comes from GGUF");
    require(c.repetitions > 0 && c.warmups >= 0 && c.layer >= 0, "invalid count");
    require(c.route_case >= -1 && (c.route_case == -1 || !c.route_capture.empty()), "--route-case requires --route-capture");
    require(c.route_capture.empty() || (!synthetic_top_k && !synthetic_cases),
        "capture replay cannot use --top-k, --tokens, or --patterns");
    require(c.sample_route_capture.empty() || (c.native_only && c.route_capture.empty() && c.sequence.empty() &&
        c.route_case == -1 && !synthetic_top_k && !explicit_patterns),
        "sampled routes require native-only and cannot combine with replay, sequence, route-case, top-k, or patterns");
    require(!sample_seed_set || !c.sample_route_capture.empty(), "--sample-seed requires --sample-route-capture");
    require(c.cache_entries > 0 && c.cache_entries <= 4096 && c.cache_bytes > 0 &&
        c.cache_bytes <= size_t(4) * 1024 * 1024 * 1024, "cache caps out of range");
    require(c.sequence.empty() || c.sequence == "native" || c.sequence == "owner" || c.sequence == "bucket", "unknown sequence mode");
    require(c.sequence.empty() || !explicit_warm_controls, "sequence fixes zero warmups and three passes; do not pass warm controls");
    require(c.sequence.empty() ? (c.reference_in.empty() && c.reference_out.empty()) :
        (!c.route_capture.empty() && c.route_case == -1 && !synthetic_cases &&
        (c.sequence == "native" ? (!c.reference_out.empty() && c.reference_in.empty()) :
        (!c.reference_in.empty() && c.reference_out.empty()))), "sequence requires complete capture and exactly one matching reference option");
    for (int q : c.tokens) { require(q > 0 && q <= 256, "tokens must be 1..256"); }
    for (const auto & p : c.patterns) {
        require(p == "balanced" || p == "skew" || p == "random", "unknown route pattern");
    }
    if (!c.sample_route_capture.empty()) { c.patterns = {"sampled_histogram"}; }
    return c;
}

using json = nlohmann::ordered_json;
const std::vector<std::string> route_stages = {"ffn_moe_topk", "ffn_moe_weights",
    "ffn_moe_weights_softmax", "ffn_moe_weights_norm", "ffn_moe_weights_scaled"};
uint64_t count_field(const json & j, const char * key, uint64_t minimum, uint64_t maximum) {
    const auto & value = j.at(key);
    require(value.is_number_integer() && value >= 0, std::string("invalid integer ") + key);
    const uint64_t count = value.get<uint64_t>();
    require(count >= minimum && count <= maximum, std::string("out of range ") + key);
    return count;
}
struct replay_case {
    int q = 0;
    uint64_t call = 0, layer_call = 0;
    bool replay = false;
    std::string stage;
    std::vector<int32_t> ids;
    std::vector<float> coeff;
};
struct capture_data {
    int experts = 0, used = 0;
    size_t records = 0, bytes = 0, skipped_groups = 0;
    std::string revision, model_file;
    std::string sha256;
    std::vector<replay_case> cases;
};
capture_data read_capture(std::istream & input, int layer, const std::vector<std::string> & expected_stages) {
    constexpr size_t max_bytes = 64 * 1024 * 1024, max_line = 1024 * 1024;
    capture_data out;
    bool contract = false, metadata = false, final = false, active = false;
    uint64_t call = 0;
    std::string role, phase;
    bool prompt = false, replay = false;
    struct group {
        int q = 0;
        std::map<std::string, json> stages;
    };
    std::map<uint64_t, group> groups;
    std::string line;
    while (std::getline(input, line)) {
        require(!final && line.size() <= max_line && out.bytes + line.size() + 1 <= max_bytes,
            "capture trailing data or byte/line cap exceeded");
        const json event = json::parse(line);
        const std::string kind = event.at("kind").get<std::string>();
        if (kind == "final") {
            require(contract && metadata && !active && event.at("status") == "request_complete" &&
                event.at("open_phase") == false, "capture did not complete successfully");
            require(count_field(event, "records", 0, 50000) == out.records &&
                count_field(event, "bytes_before_final", 0, max_bytes) == out.bytes, "capture footer count mismatch");
            final = true;
        } else {
            require(out.records < 50000 && count_field(event, "record", 1, 50000) == ++out.records,
                "capture record cap/order mismatch");
            if (kind == "contract") {
                require(out.records == 1 && !contract && event.at("version") == 1 && event.at("mode") == "routes" &&
                    event.at("warmup_excluded") == true && event.at("single_slot") == true, "unsupported capture contract");
                count_field(event, "max_route_q", 1, 256);
                contract = true;
            } else if (kind == "metadata") {
                require(contract && !metadata && !active && call == 0 && event.at("architecture") == "glm5next",
                    "unsupported capture metadata");
                out.experts = int(count_field(event, "experts", 1, 1024));
                out.used = int(count_field(event, "top_k", 1, std::min(32, out.experts)));
                out.revision = event.at("revision").get<std::string>();
                out.model_file = event.at("model_file").get<std::string>();
                metadata = true;
            } else if (kind == "phase_begin") {
                require(metadata && !active, "capture phase overlap or missing metadata");
                const uint64_t next_call = count_field(event, "call", 1, 50000);
                require(next_call == call + 1, "capture call sequence mismatch");
                call = next_call;
                role = event.at("role").get<std::string>();
                phase = event.at("phase").get<std::string>();
                require((phase == "target" && role == "target") ||
                    ((phase == "draft" || phase == "process") && role == "draft"), "unsupported capture phase");
                prompt = event.at("prompt").get<bool>();
                replay = event.at("replay").get<bool>();
                active = true;
                groups.clear();
            } else if (kind == "route") {
                require(active && count_field(event, "call", 1, 50000) == call && event.at("role") == role &&
                    event.at("phase") == phase, "route outside matching phase");
                const int route_layer = int(count_field(event, "layer", 0, 4095));
                const uint64_t ordinal = count_field(event, "layer_call", 1, 50000);
                const int q = int(count_field(event, "q", 1, 256));
                const std::string stage = event.at("stage").get<std::string>();
                require(std::find(route_stages.begin(), route_stages.end(), stage) != route_stages.end(), "unknown route stage");
                if (role == "target" && !prompt && route_layer == layer && q <= 6) {
                    require(count_field(event, "top_k", 1, 32) == uint64_t(out.used) &&
                        event.at("order") == "token_then_original_slot" && event.at("values").is_array() &&
                        event.at("values").size() == size_t(q * out.used), "capture routing geometry mismatch");
                    const bool ids = stage == route_stages[0];
                    require(event.at("dtype") == (ids ? "i32" : "f32"), "capture routing type mismatch");
                    require(groups.count(ordinal) || groups.size() < 1024, "capture phase group cap exceeded");
                    auto & g = groups[ordinal];
                    require((g.q == 0 || g.q == q) && !g.stages.count(stage), "duplicate/mismatched route stage");
                    require(ids ? g.stages.empty() : g.stages.count(route_stages[0]) != 0, "coefficient before top-k");
                    g.q = q;
                    g.stages.emplace(stage, event.at("values"));
                } else if (stage == route_stages[0]) { ++out.skipped_groups; }
            } else if (kind == "phase_end") {
                require(active && count_field(event, "call", 1, 50000) == call && event.at("role") == role &&
                    event.at("phase") == phase && event.at("status") == 0, "unsuccessful/unmatched capture phase");
                for (const auto & entry : groups) {
                    const auto & g = entry.second;
                    require(g.stages.count(route_stages[0]) && g.stages.count(route_stages[1]), "incomplete route group");
                    std::vector<std::string> present;
                    for (size_t i = 1; i < route_stages.size(); ++i) {
                        if (g.stages.count(route_stages[i])) { present.push_back(route_stages[i]); }
                    }
                    require(expected_stages.empty() || present == expected_stages, "capture coefficient stages disagree with GGUF");
                    replay_case c;
                    c.q = g.q; c.call = call; c.layer_call = entry.first; c.replay = replay; c.stage = present.back();
                    for (const auto & value : g.stages.at(route_stages[0])) {
                        require(value.is_number_integer() && value >= 0 && value < out.experts, "invalid captured expert ID");
                        c.ids.push_back(value.get<int32_t>());
                    }
                    for (int t = 0; t < c.q; ++t) {
                        for (int s = 0; s < out.used; ++s) {
                            for (int prev = 0; prev < s; ++prev) {
                                require(c.ids[t * out.used + s] != c.ids[t * out.used + prev], "duplicate captured expert within token");
                            }
                        }
                    }
                    for (const auto & name : present) {
                        for (const auto & value : g.stages.at(name)) {
                            require(value.is_number_float() && std::isfinite(value.get<float>()) &&
                                value.get<double>() == double(value.get<float>()), "invalid captured F32 coefficient");
                            if (name == c.stage) { c.coeff.push_back(value.get<float>()); }
                        }
                    }
                    require(out.cases.size() < 1024, "capture eligible case cap exceeded");
                    out.cases.push_back(std::move(c));
                }
                active = false;
            } else {
                require(metadata && !active && (kind == "proposed" || kind == "accept"), "unsupported/error capture record");
            }
        }
        out.bytes += line.size() + 1;
    }
    require(!input.bad() && final && !out.cases.empty(), "incomplete capture or no eligible routes");
    return out;
}

std::vector<uint64_t> route_histogram(const capture_data & capture) {
    std::vector<uint64_t> counts(capture.experts);
    for (const auto & item : capture.cases) {
        for (int32_t id : item.ids) { ++counts[id]; }
    }
    require(std::count_if(counts.begin(), counts.end(), [](uint64_t n) { return n != 0; }) >= capture.used,
        "route histogram has fewer experts than top-k");
    return counts;
}

std::vector<int32_t> sample_route_histogram(const std::vector<uint64_t> & counts, int q, int used, uint32_t seed) {
    require(q > 0 && used > 0 && std::count_if(counts.begin(), counts.end(),
        [](uint64_t n) { return n != 0; }) >= used, "invalid sampled route geometry");
    const uint64_t total = std::accumulate(counts.begin(), counts.end(), uint64_t(0));
    std::mt19937_64 rng(seed);
    std::vector<int32_t> ids;
    ids.reserve(size_t(q) * used);
    for (int t = 0; t < q; ++t) {
        std::vector<uint64_t> remaining = counts;
        uint64_t available = total;
        for (int s = 0; s < used; ++s) {
            uint64_t ticket = std::uniform_int_distribution<uint64_t>(0, available - 1)(rng);
            size_t id = 0;
            while (ticket >= remaining[id]) { ticket -= remaining[id++]; }
            ids.push_back(int32_t(id));
            available -= remaining[id];
            remaining[id] = 0;
        }
    }
    return ids;
}

std::string route_ids_sha256(const std::vector<int32_t> & ids) {
    sha256_t hash; sha256_init(&hash);
    sha256_update(&hash, reinterpret_cast<const unsigned char *>(ids.data()), ids.size() * sizeof(int32_t));
    unsigned char bytes[32]; sha256_final(&hash, bytes);
    std::string result;
    for (auto b : bytes) { char hex[3]; std::snprintf(hex, sizeof(hex), "%02x", b); result += hex; }
    return result;
}

struct bank_source {
    ggml_type type = GGML_TYPE_F32;
    int64_t k = 0, m = 0, e = 0;
    std::string path;
    size_t offset = 0;
};
std::array<bank_source, 3> sources(config & c) {
    // Order is gate, up, down, matching the actual model FFN arithmetic.
    std::array<bank_source, 3> out;
    if (c.models.empty()) {
        for (int b = 0; b < 3; ++b) {
            out[b].type = c.quant ? (b == 2 ? GGML_TYPE_IQ4_XS : GGML_TYPE_IQ3_S) : GGML_TYPE_F32;
            out[b].k = b == 2 ? c.f : c.d;
            out[b].m = b == 2 ? c.d : c.f;
            out[b].e = c.experts;
        }
        return out;
    }
    bool found_clamp = false;
    bool found_used = false;
    const char * suffix[] = {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"};
    for (const auto & path : c.models) {
        gguf_context * raw = gguf_init_from_file(path.c_str(), {true, nullptr});
        require(raw != nullptr, "cannot read GGUF metadata " + path);
        std::unique_ptr<gguf_context, decltype(&gguf_free)> uf(raw, gguf_free);
        for (int b = 0; b < 3; ++b) {
            const std::string name = "blk." + std::to_string(c.layer) + "." + suffix[b];
            const int64_t tid = gguf_find_tensor(raw, name.c_str());
            if (tid < 0) { continue; }
            require(out[b].path.empty(), "duplicate bank " + name);
            const int64_t * ne = gguf_get_tensor_ne(raw, tid);
            require(ne[3] == 1, "expected 3D expert bank");
            out[b].type = gguf_get_tensor_type(raw, tid);
            out[b].k = ne[0]; out[b].m = ne[1]; out[b].e = ne[2];
            out[b].path = path;
            out[b].offset = gguf_get_data_offset(raw) + gguf_get_tensor_offset(raw, tid);
        }
        const int64_t arch_key = gguf_find_key(raw, "general.architecture");
        // Split GGUF metadata normally lives only in the first shard.
        if (arch_key < 0) { continue; }
        const std::string arch = gguf_get_val_str(raw, arch_key);
        require(arch == "glm5next", "real-weight fixture is qualified for glm5next only");
        if (!c.route_capture.empty() || !c.sample_route_capture.empty()) {
            const int64_t norm_key = gguf_find_key(raw, (arch + ".expert_weights_norm").c_str());
            const int64_t scale_key = gguf_find_key(raw, (arch + ".expert_weights_scale").c_str());
            const int64_t gating_key = gguf_find_key(raw, (arch + ".expert_gating_func").c_str());
            require(norm_key >= 0 && scale_key >= 0 && gating_key >= 0, "missing GGUF coefficient-stage metadata");
            require(gguf_get_kv_type(raw, norm_key) == GGUF_TYPE_BOOL &&
                gguf_get_kv_type(raw, scale_key) == GGUF_TYPE_FLOAT32 &&
                gguf_get_kv_type(raw, gating_key) == GGUF_TYPE_UINT32, "invalid GGUF coefficient-stage metadata types");
            const float scale = gguf_get_val_f32(raw, scale_key);
            const uint32_t gating = gguf_get_val_u32(raw, gating_key);
            require(std::isfinite(scale) && gating >= 1 && gating <= 4, "invalid GGUF routing scale/gating function");
            std::vector<std::string> stages{route_stages[1]};
            // SOFTMAX_WEIGHT is 3 in llama-hparams.h; this fixture links ggml only.
            if (gating == 3) { stages.push_back(route_stages[2]); }
            if (gguf_get_val_bool(raw, norm_key)) { stages.push_back(route_stages[3]); }
            if (scale != 0.0f && scale != 1.0f) { stages.push_back(route_stages[4]); }
            require(c.coefficient_stages.empty() || c.coefficient_stages == stages, "conflicting GGUF routing metadata");
            c.coefficient_stages = std::move(stages);
        }
        const int64_t used_key = gguf_find_key(raw, (arch + ".expert_used_count").c_str());
        if (used_key >= 0) {
            c.used = int(gguf_get_val_u32(raw, used_key)); found_used = true;
        }
        const int64_t key = gguf_find_key(raw, (arch + ".swiglu_clamp_exp").c_str());
        if (key >= 0) {
            if (gguf_get_kv_type(raw, key) == GGUF_TYPE_FLOAT32) {
                c.clamp = gguf_get_val_f32(raw, key);
            } else {
                require(gguf_get_kv_type(raw, key) == GGUF_TYPE_ARRAY &&
                    gguf_get_arr_type(raw, key) == GGUF_TYPE_FLOAT32 &&
                    size_t(c.layer) < gguf_get_arr_n(raw, key), "invalid per-layer clamp metadata");
                c.clamp = static_cast<const float *>(gguf_get_arr_data(raw, key))[c.layer];
            }
            found_clamp = true;
        }
    }
    for (const auto & b : out) { require(!b.path.empty(), "missing selected layer bank: pass its shard with --model"); }
    require(found_clamp && found_used, "missing real model clamp or top-k metadata");
    require((c.route_capture.empty() && c.sample_route_capture.empty()) || !c.coefficient_stages.empty(),
        "missing GGUF routing-stage contract");
    require(out[0].k == out[1].k && out[0].m == out[1].m && out[0].e == out[1].e &&
        out[0].k == out[2].m && out[0].m == out[2].k && out[0].e == out[2].e,
        "inconsistent gate/up/down bank shapes");
    c.d = int(out[0].k); c.f = int(out[0].m); c.experts = int(out[0].e);
    return out;
}

struct backend_guard {
    ggml_backend_t ptr;
    explicit backend_guard(ggml_backend_t p) : ptr(p) { require(ptr != nullptr, "backend initialization failed"); }
    ~backend_guard() { ggml_backend_free(ptr); }
};
struct arena {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_cgraph * graph = nullptr;
    explicit arena(bool with_graph = true) {
        ctx = ggml_init({256 * ggml_tensor_overhead() + ggml_graph_overhead_custom(256, false), nullptr, true});
        require(ctx != nullptr, "metadata allocation failed");
        if (with_graph) { graph = ggml_new_graph_custom(ctx, 256, false); }
    }
    ~arena() { ggml_backend_buffer_free(buffer); ggml_free(ctx); }
    void allocate(ggml_backend_t backend) {
        if (graph) {
            for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
                ggml_tensor * node = ggml_graph_node(graph, i);
                require(ggml_backend_supports_op(backend, node),
                    std::string(ggml_backend_name(backend)) + " lacks " + ggml_op_name(node->op));
            }
        }
        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        require(buffer != nullptr, "backend tensor allocation failed");
    }
};
struct banks {
    arena mem{false};
    std::array<ggml_tensor *, 3> tensor;
    banks(const std::array<bank_source, 3> & src, int count, ggml_backend_t backend) {
        for (int b = 0; b < 3; ++b) {
            tensor[b] = ggml_new_tensor_3d(mem.ctx, src[b].type, src[b].k, src[b].m, count);
            ggml_format_name(tensor[b], "ep_bank_%d", b);
        }
        mem.allocate(backend);
        ggml_backend_buffer_set_usage(mem.buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }
};
std::vector<int> boundaries(int e, int n) {
    std::vector<int> out(n + 1);
    for (int p = 0; p <= n; ++p) { out[p] = e * p / n; }
    return out;
}
int owner_of(int id, const std::vector<int> & bounds) {
    require(id >= 0 && id < bounds.back(), "route expert ID out of range");
    return int(std::upper_bound(bounds.begin(), bounds.end(), id) - bounds.begin()) - 1;
}
void load_banks(const std::array<bank_source, 3> & src, banks & full,
        const std::vector<std::unique_ptr<banks>> & owners, const std::vector<int> & bounds,
        std::string * digest = nullptr) {
    sha256_t hash;
    if (digest) { sha256_init(&hash); }
    std::mt19937 rng(12345);
    std::normal_distribution<float> normal(0, 1);
    for (int b = 0; b < 3; ++b) {
        const size_t expert_bytes = full.tensor[b]->nb[2];
        std::vector<uint8_t> bytes(expert_bytes);
        std::ifstream file;
        if (!src[b].path.empty()) {
            file.open(src[b].path, std::ios::binary);
            require(bool(file), "cannot open bank file");
            file.seekg(std::streamoff(src[b].offset));
            require(bool(file), "cannot seek bank file");
        }
        std::vector<float> values;
        if (!file.is_open()) { values.resize(size_t(src[b].k * src[b].m)); }
        for (int e = 0; e < src[b].e; ++e) {
            if (file.is_open()) {
                file.read(reinterpret_cast<char *>(bytes.data()), std::streamsize(bytes.size()));
                require(size_t(file.gcount()) == bytes.size(), "short bank read");
            } else {
                for (float & v : values) { v = normal(rng) / std::sqrt(float(src[b].k)); }
                require(ggml_quantize_chunk(src[b].type, values.data(), bytes.data(), 0,
                    src[b].m, src[b].k, nullptr) == bytes.size(), "quantized bank size mismatch");
            }
            // Identical raw quantized expert bytes, never dequantize/requantize shards.
            if (digest) { sha256_update(&hash, bytes.data(), bytes.size()); }
            ggml_backend_tensor_set(full.tensor[b], bytes.data(), size_t(e) * expert_bytes, bytes.size());
            if (!owners.empty()) {
                const int p = owner_of(e, bounds);
                ggml_backend_tensor_set(owners[p]->tensor[b], bytes.data(),
                    size_t(e - bounds[p]) * expert_bytes, bytes.size());
            }
        }
    }
    if (digest) {
        unsigned char bytes[32]; sha256_final(&hash, bytes);
        for (auto b : bytes) { char hex[3]; std::snprintf(hex, sizeof(hex), "%02x", b); *digest += hex; }
    }
}

ggml_tensor * expert_ffn(arena & a, const banks & bank, ggml_tensor * x,
        ggml_tensor * ids, float clamp) {
    ggml_tensor * gate = ggml_mul_mat_id(a.ctx, bank.tensor[0], x, ids);
    ggml_tensor * up = ggml_mul_mat_id(a.ctx, bank.tensor[1], x, ids);
    ggml_tensor * act = clamp > 1e-6f ? ggml_swiglu_clamp(a.ctx, gate, up, clamp) : ggml_swiglu_split(a.ctx, gate, up);
    return ggml_mul_mat_id(a.ctx, bank.tensor[2], act, ids);
}
struct assignments {
    std::vector<int32_t> expert, token, slot;
};
std::vector<assignments> partition(const std::vector<int32_t> & ids, int q, int k, const std::vector<int> & bounds) {
    require(ids.size() == size_t(q * k), "bad route size");
    std::vector<assignments> out(bounds.size() - 1);
    for (int t = 0; t < q; ++t) {
        for (int s = 0; s < k; ++s) {
            const int32_t id = ids[t * k + s];
            // Top-k must not repeat an expert within a token.
            for (int prev = 0; prev < s; ++prev) {
                require(ids[t * k + prev] != id, "duplicate expert within token");
            }
            const int p = owner_of(id, bounds);
            out[p].expert.push_back(id - bounds[p]);
            out[p].token.push_back(t);
            out[p].slot.push_back(t * k + s);
        }
    }
    size_t total = 0;
    for (const auto & a : out) { total += a.expert.size(); }
    require(total == ids.size(), "lost assignments");
    return out;
}
// Unlike flat assignments.token, a bucket's token vector contains each token
// once. Its expert/slot vectors have r entries per token, in original slot order.
std::vector<assignments> cardinality_buckets(const assignments & a, int k) {
    require(a.expert.size() == a.token.size() && a.expert.size() == a.slot.size(), "invalid flat assignments");
    std::vector<assignments> out(k + 1);
    size_t first = 0;
    while (first < a.expert.size()) {
        size_t end = first + 1;
        while (end < a.expert.size() && a.token[end] == a.token[first]) { ++end; }
        const size_t r = end - first;
        require(r <= size_t(k), "owner token cardinality exceeds top-k");
        require(first == 0 || a.token[first] > a.token[first - 1], "owner assignments must be in token order");
        auto & bucket = out[r];
        bucket.token.push_back(a.token[first]);
        bucket.expert.insert(bucket.expert.end(), a.expert.begin() + first, a.expert.begin() + end);
        bucket.slot.insert(bucket.slot.end(), a.slot.begin() + first, a.slot.begin() + end);
        first = end;
    }
    return out;
}
std::vector<int32_t> routes(const config & c, int q, const std::string & pattern, const std::vector<int> & bounds) {
    std::vector<int32_t> ids(q * c.used);
    std::mt19937 rng(42 + q);
    std::vector<int> perm(c.experts);
    std::iota(perm.begin(), perm.end(), 0);
    for (int t = 0; t < q; ++t) {
        if (pattern == "random") { std::shuffle(perm.begin(), perm.end(), rng); }
        for (int s = 0; s < c.used; ++s) {
            int id;
            if (pattern == "skew") {
                // High IDs; every assignment belongs to the last owner. Covers
                // an empty root owner and every other owner being empty.
                id = c.experts - 1 - ((s + t) % (bounds.back() - bounds[bounds.size() - 2]));
            } else if (pattern == "random") { id = perm[s];
            } else {
                const int p = s % int(c.devices.size());
                id = bounds[p] + ((s / int(c.devices.size()) + t) % (bounds[p + 1] - bounds[p]));
            }
            ids[t * c.used + ((s + t) % c.used)] = id;
        }
    }
    return ids;
}

struct reference {
    arena mem;
    ggml_tensor * x, * coeff, * ids_storage, * ids, * result;
    reference(const config & c, int q, const banks & full, ggml_backend_t root) {
        x = ggml_new_tensor_2d(mem.ctx, GGML_TYPE_F32, c.d, q);
        coeff = ggml_new_tensor_2d(mem.ctx, GGML_TYPE_F32, c.used, q);
        ids_storage = ggml_new_tensor_2d(mem.ctx, GGML_TYPE_I32, c.experts, q);
        ids = ggml_view_2d(mem.ctx, ids_storage, c.used, q, ids_storage->nb[1], 0);
        ggml_tensor * ff = expert_ffn(mem, full, ggml_reshape_3d(mem.ctx, x, c.d, 1, q), ids, c.clamp);
        ff = ggml_mul(mem.ctx, ff, ggml_reshape_3d(mem.ctx, coeff, 1, c.used, q));
        // Match the accepted build's slot-order weighted ADD reduction.
        result = ggml_view_2d(mem.ctx, ff, c.d, q, ff->nb[2], 0);
        for (int s = 1; s < c.used; ++s) {
            result = ggml_add(mem.ctx, result, ggml_view_2d(mem.ctx, ff, c.d, q, ff->nb[2], s * ff->nb[1]));
        }
        ggml_build_forward_expand(mem.graph, result);
        mem.allocate(root);
    }
};
struct packed_reference {
    arena mem;
    ggml_tensor * result;
    packed_reference(const config & c, int q, const banks & full,
            ggml_backend_t root, reference & ref) {
        const int count = c.used * q;
        ggml_tensor * token = ggml_new_tensor_1d(mem.ctx, GGML_TYPE_I32, count);
        // Route IDs stay device-resident. Flatten their logical top-k view,
        // preserving token/slot order despite the full-bank row stride.
        ggml_tensor * ids = ggml_reshape_2d(mem.ctx, ggml_cont(mem.ctx, ref.ids), 1, count);
        ggml_tensor * x = ggml_get_rows(mem.ctx, ref.x, token);
        ggml_tensor * ff = expert_ffn(mem, full,
            ggml_reshape_3d(mem.ctx, x, c.d, 1, count), ids, c.clamp);
        ff = ggml_mul(mem.ctx, ggml_reshape_2d(mem.ctx, ff, c.d, count),
            ggml_reshape_2d(mem.ctx, ref.coeff, 1, count));
        result = ggml_get_rows_back(mem.ctx, ff, token, ref.x);
        ggml_build_forward_expand(mem.graph, result);
        int mmid_count = 0;
        for (int i = 0; i < ggml_graph_n_nodes(mem.graph); ++i) {
            if (ggml_graph_node(mem.graph, i)->op == GGML_OP_MUL_MAT_ID) { ++mmid_count; }
        }
        require(mmid_count == 3, "packed-local graph must contain exactly one routed FFN");
        mem.allocate(root);
        // These indices depend only on q/top-k, never router decisions. Their
        // one-time initialization belongs to graph preparation, just like shapes.
        std::vector<int32_t> token_ids(count);
        for (int i = 0; i < count; ++i) { token_ids[i] = i / c.used; }
        ggml_backend_tensor_set(token, token_ids.data(), 0, token_ids.size() * sizeof(int32_t));
    }
};
struct owner_graph {
    arena mem;
    ggml_tensor * x, * coeff, * expert = nullptr, * token = nullptr, * slot = nullptr, * result = nullptr;
    ggml_tensor * metadata = nullptr;
    std::vector<size_t> bucket_counts;
    std::vector<int32_t> upload_values;
    bool bucketed;
    int top_k;
    size_t count;
    owner_graph(const config & c, int q, const assignments & a, const banks & bank,
            ggml_backend_t backend, reference * root_ref, bool bucketed_ = false)
        : bucketed(bucketed_), top_k(c.used), count(a.expert.size()) {
        require(count > 0, "empty owner should not have a graph");
        x = root_ref ? root_ref->x : ggml_new_tensor_2d(mem.ctx, GGML_TYPE_F32, c.d, q);
        coeff = root_ref ? root_ref->coeff : ggml_new_tensor_2d(mem.ctx, GGML_TYPE_F32, c.used, q);
        if (bucketed) {
            build_bucketed(c, q, a, bank, backend);
            return;
        }
        expert = ggml_new_tensor_2d(mem.ctx, GGML_TYPE_I32, 1, count);
        token = ggml_new_tensor_1d(mem.ctx, GGML_TYPE_I32, count);
        slot = ggml_new_tensor_1d(mem.ctx, GGML_TYPE_I32, count);
        ggml_tensor * packed = ggml_get_rows(mem.ctx, x, token);
        ggml_tensor * weights = ggml_get_rows(mem.ctx, ggml_reshape_2d(mem.ctx, coeff, 1, c.used * q), slot);
        ggml_tensor * ff = expert_ffn(mem, bank, ggml_reshape_3d(mem.ctx, packed, c.d, 1, count), expert, c.clamp);
        ff = ggml_mul(mem.ctx, ggml_reshape_2d(mem.ctx, ff, c.d, count), weights);
        // Native scatter-add sums all of this owner's selected experts per token.
        // There are exactly count real assignments: no padded/dummy expert work.
        result = ggml_get_rows_back(mem.ctx, ff, token, x);
        ggml_build_forward_expand(mem.graph, result);
        mem.allocate(backend);
    }
    void build_bucketed(const config & c, int q, const assignments & a, const banks & bank, ggml_backend_t backend) {
        const auto buckets = cardinality_buckets(a, c.used);
        bucket_counts.resize(c.used + 1);
        size_t n_tokens = 0;
        for (int r = 1; r <= c.used; ++r) {
            bucket_counts[r] = buckets[r].token.size();
            n_tokens += bucket_counts[r];
        }
        metadata = ggml_new_tensor_1d(mem.ctx, GGML_TYPE_I32, 2 * count + n_tokens);
        upload_values.reserve(2 * count + n_tokens);
        ggml_tensor * all_coeff = ggml_reshape_2d(mem.ctx, coeff, 1, c.used * q);
        size_t offset = 0, n_buckets = 0;
        for (int r = 1; r <= c.used; ++r) {
            const size_t nb = bucket_counts[r];
            if (!nb) { continue; }
            ++n_buckets;
            ggml_tensor * ids = ggml_view_2d(mem.ctx, metadata, r, nb, r * sizeof(int32_t), offset);
            offset += r * nb * sizeof(int32_t);
            ggml_tensor * tokens = ggml_view_1d(mem.ctx, metadata, nb, offset);
            offset += nb * sizeof(int32_t);
            ggml_tensor * slots = ggml_view_1d(mem.ctx, metadata, r * nb, offset);
            offset += r * nb * sizeof(int32_t);
            ggml_tensor * inputs = ggml_get_rows(mem.ctx, x, tokens);
            ggml_tensor * weights = ggml_get_rows(mem.ctx, all_coeff, slots);
            // nb is the real number of TOKENS, not flattened assignments. Each
            // token has exactly r selected local experts: no padded fake work.
            ggml_tensor * ff = expert_ffn(mem, bank,
                ggml_reshape_3d(mem.ctx, inputs, c.d, 1, nb), ids, c.clamp);
            ff = ggml_mul(mem.ctx, ff, ggml_reshape_3d(mem.ctx, weights, 1, r, nb));
            ggml_tensor * token_sum = ggml_view_2d(mem.ctx, ff, c.d, nb, ff->nb[2], 0);
            for (int s = 1; s < r; ++s) {
                token_sum = ggml_add(mem.ctx, token_sum,
                    ggml_view_2d(mem.ctx, ff, c.d, nb, ff->nb[2], s * ff->nb[1]));
            }
            // Only one row per token reaches the scatter, avoiding a scan over
            // every expert assignment in GET_ROWS_BACK's native implementation.
            ggml_tensor * partial = ggml_get_rows_back(mem.ctx, token_sum, tokens, x);
            result = result ? ggml_add(mem.ctx, result, partial) : partial;
        }
        require(offset == ggml_nbytes(metadata), "bucket metadata size mismatch");
        ggml_build_forward_expand(mem.graph, result);
        size_t mmid_count = 0;
        for (int i = 0; i < ggml_graph_n_nodes(mem.graph); ++i) {
            if (ggml_graph_node(mem.graph, i)->op == GGML_OP_MUL_MAT_ID) { ++mmid_count; }
        }
        require(mmid_count == 3 * n_buckets, "bucket graph must execute exactly one FFN per real bucket");
        mem.allocate(backend);
    }
    void upload(const assignments & a, ggml_backend_t backend) {
        require(a.expert.size() == count, "route counts changed: rebuild graph explicitly");
        if (bucketed) {
            const auto buckets = cardinality_buckets(a, top_k);
            upload_values.clear();
            for (int r = 1; r <= top_k; ++r) {
                const auto & bucket = buckets[r];
                require(bucket.token.size() == bucket_counts[r], "bucket token counts changed: rebuild graph explicitly");
                upload_values.insert(upload_values.end(), bucket.expert.begin(), bucket.expert.end());
                upload_values.insert(upload_values.end(), bucket.token.begin(), bucket.token.end());
                upload_values.insert(upload_values.end(), bucket.slot.begin(), bucket.slot.end());
            }
            require(upload_values.size() * sizeof(int32_t) == ggml_nbytes(metadata), "bucket upload size mismatch");
            // Member storage remains alive until the run's final synchronization.
            ggml_backend_tensor_set_async(backend, metadata, upload_values.data(), 0, ggml_nbytes(metadata));
            return;
        }
        ggml_backend_tensor_set_async(backend, expert, a.expert.data(), 0, count * sizeof(int32_t));
        ggml_backend_tensor_set_async(backend, token, a.token.data(), 0, count * sizeof(int32_t));
        ggml_backend_tensor_set_async(backend, slot, a.slot.data(), 0, count * sizeof(int32_t));
    }
};
struct ep_graph {
    arena readback;
    ggml_tensor * route_ids;
    std::vector<std::unique_ptr<owner_graph>> owners;
    arena reduction;
    std::vector<ggml_tensor *> partials;
    ggml_tensor * result = nullptr;
    std::vector<int32_t> host_ids;
    const config & c;
    int q;
    const std::vector<int> & bounds;
    const std::vector<ggml_backend_t> & backends;
    reference & ref;
    double route_host_ms = 0, dispatch_submit_ms = 0, compute_submit_ms = 0, collect_wait_ms = 0;
    ep_graph(const config & c_, int q_, const std::vector<int> & bounds_,
            const std::vector<ggml_backend_t> & backends_, reference & ref_,
            const std::vector<std::unique_ptr<banks>> & weights, const std::vector<int32_t> & ids, bool bucketed = false)
        : c(c_), q(q_), bounds(bounds_), backends(backends_), ref(ref_) {
        host_ids.resize(q * c.used);
        route_ids = ggml_cont(readback.ctx, ref.ids);
        ggml_build_forward_expand(readback.graph, route_ids);
        readback.allocate(backends[0]);
        const auto initial = partition(ids, q, c.used, bounds);
        owners.resize(backends.size()); partials.resize(backends.size(), nullptr);
        for (size_t p = 0; p < backends.size(); ++p) {
            if (initial[p].expert.empty()) { continue; }
            owners[p].reset(new owner_graph(c, q, initial[p], *weights[p], backends[p], p == 0 ? &ref : nullptr, bucketed));
            partials[p] = ggml_new_tensor_2d(reduction.ctx, GGML_TYPE_F32, c.d, q);
            if (p == 0) {
                // A materialized leaf, NOT a ggml view: a view would retain the
                // owner graph's ancestors and re-execute its whole FFN during
                // root reduction. The owning arena outlives this reduction.
                require(ggml_is_contiguous(owners[p]->result), "owner partial must be contiguous");
                partials[p]->data = owners[p]->result->data;
                partials[p]->buffer = owners[p]->result->buffer;
                ggml_set_name(partials[p], "ep_owner0_materialized_partial");
            }
            result = result ? ggml_add(reduction.ctx, result, partials[p]) : partials[p];
        }
        // Always materialize a root output, including one active remote owner.
        result = ggml_cont(reduction.ctx, result);
        ggml_build_forward_expand(reduction.graph, result);
        for (int i = 0; i < ggml_graph_n_nodes(reduction.graph); ++i) {
            const auto op = ggml_graph_node(reduction.graph, i)->op;
            require(op == GGML_OP_ADD || op == GGML_OP_CONT || op == GGML_OP_NONE,
                "reduction graph traversed an owner FFN");
        }
        reduction.allocate(backends[0]);
    }
    double run() {
        const auto start = clock_type::now();
        require(ggml_backend_graph_compute_async(backends[0], readback.graph) == GGML_STATUS_SUCCESS, "route gather failed");
        ggml_backend_tensor_get_async(backends[0], route_ids, host_ids.data(), 0, host_ids.size() * sizeof(int32_t));
        ggml_backend_synchronize(backends[0]);
        const auto a = partition(host_ids, q, c.used, bounds);
        route_host_ms = elapsed_ms(start);
        auto phase_start = clock_type::now();
        // Submit every fan-out BEFORE root-owner compute. Otherwise source-
        // stream copies would sit behind root compute and serialize the owners.
        for (size_t p = 0; p < owners.size(); ++p) {
            require(bool(owners[p]) == !a[p].expert.empty(), "active owners changed: rebuild graph explicitly");
            if (!owners[p]) { continue; }
            owners[p]->upload(a[p], backends[p]);
            if (p != 0) {
                ggml_backend_tensor_copy_async(backends[0], backends[p], ref.x, owners[p]->x);
                ggml_backend_tensor_copy_async(backends[0], backends[p], ref.coeff, owners[p]->coeff);
            }
        }
        dispatch_submit_ms = elapsed_ms(phase_start);
        phase_start = clock_type::now();
        for (size_t p = 0; p < owners.size(); ++p) {
            if (owners[p]) {
                require(ggml_backend_graph_compute_async(backends[p], owners[p]->mem.graph) == GGML_STATUS_SUCCESS, "owner compute failed");
            }
        }
        compute_submit_ms = elapsed_ms(phase_start);
        phase_start = clock_type::now();
        for (size_t p = 1; p < owners.size(); ++p) {
            if (owners[p]) { ggml_backend_tensor_copy_async(backends[p], backends[0], owners[p]->result, partials[p]); }
        }
        require(ggml_backend_graph_compute_async(backends[0], reduction.graph) == GGML_STATUS_SUCCESS, "root reduction failed");
        // Wait for all backends before host route vectors die or buffers are
        // reused. Includes asynchronous uploads and no-peer staging completion.
        for (auto backend : backends) { ggml_backend_synchronize(backend); }
        collect_wait_ms = elapsed_ms(phase_start);
        return elapsed_ms(start);
    }
};
double run_reference(reference & ref, ggml_backend_t backend) {
    const auto start = clock_type::now();
    require(ggml_backend_graph_compute_async(backend, ref.mem.graph) == GGML_STATUS_SUCCESS, "baseline compute failed");
    ggml_backend_synchronize(backend);
    return elapsed_ms(start);
}
double run_packed_reference(packed_reference & ref, ggml_backend_t backend) {
    const auto start = clock_type::now();
    require(ggml_backend_graph_compute_async(backend, ref.mem.graph) == GGML_STATUS_SUCCESS, "packed-local compute failed");
    ggml_backend_synchronize(backend);
    return elapsed_ms(start);
}
struct errors { double nmse, max_abs, max_ref; };
errors compare(ggml_tensor * baseline, ggml_tensor * candidate) {
    require(ggml_nelements(baseline) == ggml_nelements(candidate), "output shape mismatch");
    std::vector<float> a(ggml_nelements(baseline)), b(a.size());
    ggml_backend_tensor_get(baseline, a.data(), 0, a.size() * sizeof(float));
    ggml_backend_tensor_get(candidate, b.data(), 0, b.size() * sizeof(float));
    double sq = 0, norm = 0, max_abs = 0, max_ref = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        require(std::isfinite(a[i]) && std::isfinite(b[i]), "non-finite routed output");
        const double diff = double(a[i]) - double(b[i]);
        sq += diff * diff; norm += double(a[i]) * a[i];
        max_abs = std::max(max_abs, std::abs(diff)); max_ref = std::max(max_ref, std::abs(double(a[i])));
    }
    return {sq / std::max(norm, 1e-30), max_abs, max_ref};
}
errors compare_values(const std::vector<float> & baseline, ggml_tensor * candidate) {
    require(baseline.size() == size_t(ggml_nelements(candidate)), "output shape mismatch");
    std::vector<float> actual(baseline.size());
    ggml_backend_tensor_get(candidate, actual.data(), 0, actual.size() * sizeof(float));
    double sq = 0, norm = 0, max_abs = 0, max_ref = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        require(std::isfinite(baseline[i]) && std::isfinite(actual[i]), "non-finite routed output");
        const double diff = double(baseline[i]) - double(actual[i]);
        sq += diff * diff; norm += double(baseline[i]) * baseline[i];
        max_abs = std::max(max_abs, std::abs(diff)); max_ref = std::max(max_ref, std::abs(double(baseline[i])));
    }
    return {sq / std::max(norm, 1e-30), max_abs, max_ref};
}
double median(std::vector<double> a) {
    std::sort(a.begin(), a.end());
    return (a[(a.size() - 1) / 2] + a[a.size() / 2]) / 2;
}
void partition_tests() {
    const std::vector<int> bounds = {0, 3, 7, 11};
    const std::vector<int32_t> ids = {10, 0, 4, 2, 6, 9};
    const auto a = partition(ids, 2, 3, bounds);
    std::vector<int32_t> restored(ids.size(), -1);
    for (size_t p = 0; p < a.size(); ++p) {
        for (size_t j = 0; j < a[p].slot.size(); ++j) {
            require(restored[a[p].slot[j]] == -1, "assignment duplicated");
            restored[a[p].slot[j]] = a[p].expert[j] + bounds[p];
            require(a[p].token[j] == a[p].slot[j] / 3, "wrong token map");
        }
    }
    require(restored == ids, "ownership reconstruction failed");
    // Uneven per-token cardinality, including empty tokens, with an expert
    // recurring on multiple tokens. Reconstruct original slot ownership.
    const assignments ragged{{2, 4, 1, 4, 0, 2}, {0, 0, 2, 3, 3, 3}, {0, 2, 8, 12, 13, 15}};
    const auto buckets = cardinality_buckets(ragged, 4);
    require(buckets[1].token == std::vector<int32_t>{2} &&
            buckets[2].token == std::vector<int32_t>{0} &&
            buckets[3].token == std::vector<int32_t>{3} && buckets[4].token.empty(), "incorrect cardinality buckets");
    std::vector<int32_t> bucket_restore(16, -1);
    for (int r = 1; r <= 4; ++r) {
        const auto & b = buckets[r];
        require(b.expert.size() == b.token.size() * r && b.slot.size() == b.expert.size(), "ragged bucket size");
        for (size_t j = 0; j < b.slot.size(); ++j) {
            require(bucket_restore[b.slot[j]] == -1, "bucket duplicated assignment");
            require(b.slot[j] / 4 == b.token[j / r], "bucket changed token assignment");
            bucket_restore[b.slot[j]] = b.expert[j];
        }
    }
    for (size_t j = 0; j < ragged.slot.size(); ++j) {
        require(bucket_restore[ragged.slot[j]] == ragged.expert[j], "bucket lost assignment");
    }
    for (const auto & bad : std::vector<std::vector<int32_t>>{{-1, 0, 1}, {0, 11, 1}, {1, 1, 2}}) {
        bool rejected = false;
        try { partition(bad, 1, 3, bounds); } catch (const std::runtime_error &) { rejected = true; }
        require(rejected, "invalid route was accepted");
    }
}
// Exercise the production JSONL schema without a backend or temporary files.
std::string capture_fixture(std::vector<json> events) {
    std::string text;
    size_t record = 0;
    for (auto & event : events) {
        event["record"] = ++record;
        text += event.dump() + "\n";
    }
    const json final = {{"kind", "final"}, {"status", "request_complete"}, {"open_phase", false},
        {"records", record}, {"bytes_before_final", text.size()}};
    return text + final.dump() + "\n";
}
void capture_tests() {
    std::vector<json> events = {
        {{"kind", "contract"}, {"version", 1}, {"mode", "routes"}, {"warmup_excluded", true},
            {"single_slot", true}, {"max_route_q", 6}},
        {{"kind", "metadata"}, {"architecture", "glm5next"}, {"experts", 16}, {"top_k", 8},
            {"revision", "fixture"}, {"model_file", "synthetic"}}};
    std::vector<std::vector<int32_t>> ids;
    std::vector<std::vector<float>> coeff;
    for (int call = 1; call <= 2; ++call) {
        const int q = call == 1 ? 1 : 6;
        events.push_back({{"kind", "phase_begin"}, {"call", call}, {"role", "target"}, {"phase", "target"},
            {"batch_q", q}, {"prompt", false}, {"replay", call == 2}});
        ids.emplace_back(); coeff.emplace_back();
        for (int t = 0; t < q; ++t) {
            for (int s = 0; s < 8; ++s) {
                ids.back().push_back((15 - s + (t % 3) * 3) % 16);
                coeff.back().push_back(s == 0 ? -0.0f : std::nextafter(float(s) / 17, 1.0f));
            }
        }
        for (int stage : {0, 1, 3, 4}) {
            events.push_back({{"kind", "route"}, {"call", call}, {"role", "target"}, {"phase", "target"},
                {"layer", 3}, {"layer_call", call}, {"q", q}, {"top_k", 8}, {"stage", route_stages[stage]},
                {"dtype", stage == 0 ? "i32" : "f32"}, {"order", "token_then_original_slot"},
                {"values", stage == 0 ? json(ids.back()) : json(coeff.back())}});
        }
        events.push_back({{"kind", "phase_end"}, {"call", call}, {"role", "target"}, {"phase", "target"}, {"status", 0}});
    }
    const std::vector<std::string> expected = {route_stages[1], route_stages[3], route_stages[4]};
    std::istringstream good(capture_fixture(events));
    const auto parsed = read_capture(good, 3, expected);
    require(parsed.cases.size() == 2 && parsed.experts == 16 && parsed.used == 8, "capture parser lost cases");
    for (size_t i = 0; i < parsed.cases.size(); ++i) {
        const auto & c = parsed.cases[i];
        require(c.ids == ids[i] && c.coeff.size() == coeff[i].size() &&
            std::memcmp(c.coeff.data(), coeff[i].data(), coeff[i].size() * sizeof(float)) == 0,
            "capture changed original slots or F32 bits");
        require(c.call == i + 1 && c.layer_call == i + 1 && c.replay == (i == 1) && c.stage == route_stages[4],
            "capture lost join/phase/final-stage identity");
        partition(c.ids, c.q, parsed.used, {0, 8, 16});
    }
    const auto histogram = route_histogram(parsed);
    require(std::accumulate(histogram.begin(), histogram.end(), uint64_t(0)) == 56,
        "sample histogram lost captured assignments");
    const auto sampled = sample_route_histogram(histogram, 256, parsed.used, 42);
    require(sampled == sample_route_histogram(histogram, 256, parsed.used, 42) && sampled.size() == 256 * 8,
        "sampled routes are not deterministic");
    for (int t = 0; t < 256; ++t) {
        for (int s = 0; s < parsed.used; ++s) {
            const int32_t id = sampled[t * parsed.used + s];
            require(id >= 0 && id < parsed.experts && histogram[id] > 0, "sampled inactive expert");
            for (int prev = 0; prev < s; ++prev) {
                require(sampled[t * parsed.used + prev] != id, "sampled expert repeated within token");
            }
        }
    }
    auto reject = [&](const std::string & text) {
        bool rejected = false;
        try { std::istringstream input(text); read_capture(input, 3, expected); }
        catch (const std::exception &) { rejected = true; }
        require(rejected, "invalid capture was accepted");
    };
    auto bad = events;
    bad[3]["values"][0] = 16; reject(capture_fixture(bad));
    bad = events; bad[3]["values"][0] = bad[3]["values"][1]; reject(capture_fixture(bad));
    bad = events; bad[3]["values"][0] = 1.5; reject(capture_fixture(bad));
    bad = events; bad[3]["values"].erase(0); reject(capture_fixture(bad));
    bad = events; bad[3]["order"] = "expert_sorted"; reject(capture_fixture(bad));
    bad = events; bad[4]["values"][0] = 1e100; reject(capture_fixture(bad));
    bad = events; bad[4]["values"][0] = 1; reject(capture_fixture(bad));
    bad = events; bad[4]["values"][0] = 1.0000000000000002; reject(capture_fixture(bad));
    bad = events; bad[4]["q"] = 2; reject(capture_fixture(bad));
    bad = events; bad[4]["call"] = 2; reject(capture_fixture(bad));
    bad = events; bad[1]["experts"] = 1025; reject(capture_fixture(bad));
    bad = events; bad[1]["top_k"] = 33; reject(capture_fixture(bad));
    bad = events; bad[7]["status"] = -1; reject(capture_fixture(bad));
    bad = events; bad.erase(bad.begin() + 5); reject(capture_fixture(bad));
    bad = events; bad.erase(bad.begin() + 3); reject(capture_fixture(bad));
    bad = events; bad.insert(bad.begin() + 5, bad[4]); reject(capture_fixture(bad));
    const std::string valid = capture_fixture(events);
    reject(valid.substr(0, valid.rfind("{\"kind\":\"final\"")));
    reject(valid + "{}\n");
    reject(std::string(1024 * 1024 + 1, ' ') + "\n");
}
} // namespace

// Reopen the same anonymous namespace for the opt-in fixture extension.
namespace {
#include "test-moe-expert-parallel-sequence.h"
}

int main(int argc, char ** argv) {
    try {
        config c = parse(argc, argv);
        auto src = sources(c);
        capture_data capture;
        std::vector<uint64_t> sampled_histogram;
        const bool sampled = !c.sample_route_capture.empty();
        const std::string capture_path = sampled ? c.sample_route_capture : c.route_capture;
        if (!capture_path.empty()) {
            const std::string capture_hash = sampled || !c.sequence.empty() ? sequence_hash_file(capture_path, 64 * 1024 * 1024) : "";
            std::ifstream input(capture_path, std::ios::binary | std::ios::ate);
            require(input.good() && input.tellg() > 0 && input.tellg() <= 64 * 1024 * 1024,
                "capture must be a readable file of at most 64MiB");
            input.seekg(0);
            capture = read_capture(input, c.layer, c.coefficient_stages);
            if (sampled || !c.sequence.empty()) {
                require(sequence_hash_file(capture_path, 64 * 1024 * 1024) == capture_hash,
                    "capture changed during validation");
                capture.sha256 = capture_hash;
            }
            if (c.models.empty()) {
                c.experts = capture.experts; c.used = capture.used;
                src = sources(c);
            } else {
                require(capture.experts == c.experts && capture.used == c.used, "capture/GGUF expert geometry mismatch");
            }
            if (sampled) {
                sampled_histogram = route_histogram(capture);
            } else {
                require(c.route_case < 0 || size_t(c.route_case) < capture.cases.size(), "capture case index out of range");
            }
        }
        require(c.used > 0 && c.used <= c.experts, "invalid top-k");
        const auto bounds = boundaries(c.experts, int(c.devices.size()));
        for (size_t p = 0; p < c.devices.size(); ++p) {
            require(bounds[p + 1] - bounds[p] >= c.used, "fixture requires each owner to hold at least top-k experts");
        }
        struct case_input { int q; std::string name; size_t capture_index; std::vector<int32_t> sampled_ids; };
        std::vector<case_input> cases;
        if (sampled) {
            for (int q : c.tokens) {
                cases.push_back({q, "sampled_histogram", 0, sample_route_histogram(sampled_histogram, q, c.used, c.sample_seed)});
            }
            const json event = { {"kind", "sampled_route_input"}, {"path", capture_path}, {"sha256", capture.sha256},
                {"revision", capture.revision}, {"layer", c.layer}, {"seed", c.sample_seed},
                {"source_eligible_cases", capture.cases.size()}, {"source_assignments",
                    std::accumulate(sampled_histogram.begin(), sampled_histogram.end(), uint64_t(0))},
                {"source_histogram", sampled_histogram},
                {"selection", "target_non_prompt_q_1_to_6_including_replay"},
                {"method", "weighted_without_replacement_per_token"},
                {"claim", "sampled_from_capture_histogram_not_captured_q"} };
            std::puts(event.dump().c_str());
        } else if (c.route_capture.empty()) {
            for (int q : c.tokens) {
                for (const auto & pattern : c.patterns) { cases.push_back({q, pattern, 0, {}}); }
            }
        } else {
            c.tokens.clear();
            for (size_t i = 0; i < capture.cases.size(); ++i) {
                if (c.route_case >= 0 && size_t(c.route_case) != i) { continue; }
                cases.push_back({capture.cases[i].q, "capture_" + std::to_string(i), i, {}});
                c.tokens.push_back(capture.cases[i].q);
            }
            const json event = {{"kind", "capture_input"}, {"path", c.route_capture}, {"revision", capture.revision},
                {"capture_model_file", capture.model_file}, {"records", capture.records}, {"bytes", capture.bytes},
                {"eligible_cases", capture.cases.size()}, {"selected_cases", cases.size()}, {"skipped_groups", capture.skipped_groups},
                {"layer", c.layer}, {"selection", "target_non_prompt_q_1_to_6_including_replay"},
                {"coefficient_stage_validation", c.models.empty() ? "capture_only_synthetic_weights" : "GGUF_exact_expected_stages"},
                {"weight_identity", "GGUF_paths_and_capture_metadata_are_not_cryptographic_model_identity_archive_external_hashes"}};
            std::puts(event.dump().c_str());
        }
        const bool all_f32 = std::all_of(src.begin(), src.end(), [](const bank_source & b) { return b.type == GGML_TYPE_F32; });
        const char * route_origin = sampled ? "sampled_from_q_le_6_histogram_synthetic_activations" :
            (c.route_capture.empty() ? "synthetic_deterministic" : "captured_routes_synthetic_activations");
        // Fixed before any device timing. Quantized tolerance matches the
        // existing MUL_MAT_ID backend fixture; packing can change its kernel.
        const double tolerance = all_f32 ? 1e-10 : 5e-4;
        size_t bank_bytes = 0;
        for (const auto & b : src) { bank_bytes += ggml_row_size(b.type, b.k) * b.m * b.e; }
        if (c.sequence.empty()) {
        if (c.native_only) {
            std::printf("{\"kind\":\"contract\",\"region\":\"routed_ffn_only\",\"weights\":\"%s\","
                "\"activations_routes\":\"%s\",\"D\":%d,\"F\":%d,\"experts\":%d,\"top_k\":%d,"
                "\"bank_bytes\":%zu,\"nmse_tolerance\":%.9g,\"variants\":[\"native\"],"
                "\"requested_J\":%d,\"weight_read_bytes\":\"theoretical_bank_bytes_per_expert_tile_not_measured_HBM\"}\n",
                c.models.empty() ? "synthetic" : "GGUF_exact_bytes",
                route_origin,
                c.d, c.f, c.experts, c.used, bank_bytes, tolerance, c.mmq_force_j);
        } else {
        std::printf("{\"kind\":\"contract\",\"region\":\"routed_ffn_only\",\"weights\":\"%s\","
            "\"activations_routes\":\"%s\",\"D\":%d,\"F\":%d,\"experts\":%d,"
            "\"top_k\":%d,\"clamp\":%.9g,\"gate_type\":\"%s\",\"up_type\":\"%s\",\"down_type\":\"%s\","
            "\"bank_bytes\":%zu,\"nmse_tolerance\":%.9g,\"transport_accounting\":\"logical_payload_not_DMA_wire_time\","
            "\"variants\":[\"native\",\"packed_local\",\"expert_parallel\",\"bucketed_expert_parallel\"],"
            "\"dynamic_graph_rebuild_in_warm_samples\":false}\n",
            c.models.empty() ? "synthetic" : "GGUF_exact_bytes",
            route_origin,
            c.d, c.f, c.experts, c.used, c.clamp,
            ggml_type_name(src[0].type), ggml_type_name(src[1].type), ggml_type_name(src[2].type), bank_bytes, tolerance);
        std::printf("{\"kind\":\"sample_order_contract\",\"repetitions\":%d,\"order_cycle_length\":24,"
            "\"order_cycle_complete\":%s,\"sample_role\":\"%s\"}\n", c.repetitions,
            c.repetitions % 24 == 0 ? "true" : "false",
            c.repetitions % 24 == 0 ? "balanced_performance_trial" : "partial_order_smoke");
        }
        }
        if (sampled) {
            for (const auto & item : cases) {
                std::vector<int> expert_counts(c.experts);
                for (int32_t id : item.sampled_ids) { ++expert_counts[id]; }
                int active_experts = 0;
                size_t j64_tiles = 0;
                for (int count : expert_counts) {
                    active_experts += count > 0;
                    j64_tiles += (count + 63) / 64;
                }
                const json event = {{"kind", "sampled_route_case"}, {"q", item.q}, {"pattern", item.name},
                    {"seed", c.sample_seed}, {"route_ids_sha256", route_ids_sha256(item.sampled_ids)},
                    {"assignments", item.sampled_ids.size()}, {"active_experts", active_experts},
                    {"j64_executed_cols", j64_tiles * 64}, {"claim", "sampled_not_captured_q"}};
                std::puts(event.dump().c_str());
            }
        } else if (!c.route_capture.empty()) {
            for (const auto & item : cases) {
                const auto & captured = capture.cases[item.capture_index];
                json counts = json::array(), bucket_counts = json::array();
                for (const auto & owner : partition(captured.ids, captured.q, c.used, bounds)) {
                    counts.push_back(owner.expert.size());
                    json buckets = json::array();
                    for (const auto & bucket : cardinality_buckets(owner, c.used)) { buckets.push_back(bucket.token.size()); }
                    bucket_counts.push_back(std::move(buckets));
                }
                const json event = {{"kind", "capture_case"}, {"pattern", item.name}, {"capture_index", item.capture_index},
                    {"call", captured.call}, {"role", "target"}, {"phase", "target"}, {"prompt", false},
                    {"replay", captured.replay}, {"layer", c.layer}, {"layer_call", captured.layer_call},
                    {"q", captured.q}, {"coefficient_stage", captured.stage}, {"activations", "synthetic_deterministic"},
                    {"owner_assignments", counts}, {"owner_bucket_token_counts_by_cardinality", bucket_counts}};
                std::puts(event.dump().c_str());
            }
        }
        std::fflush(stdout);
        if (c.inspect) { return 0; }
        partition_tests();
        capture_tests();
        require(std::getenv("GGML_CUDA_TEST_MMQ_FORCE_J") == nullptr,
            "external MMQ J override is not permitted; use --mmq-force-j");
        bool wants_gpu = std::any_of(c.devices.begin(), c.devices.end(), [](const std::string & d) { return d != "CPU"; });
        const char * staging_env = std::getenv("GGML_HIP_NO_PEER_STAGING");
        const char * staging_max_env = std::getenv("GGML_HIP_NO_PEER_STAGING_MAX_BYTES");
        const bool staging_enabled = staging_env && std::atoi(staging_env) != 0;
        const size_t staging_max = staging_max_env ? std::stoull(staging_max_env) : 1024 * 1024;
        const size_t largest_packet = size_t(c.d) * *std::max_element(c.tokens.begin(), c.tokens.end()) * sizeof(float);
        const bool wants_hip = std::any_of(c.devices.begin(), c.devices.end(), [](const std::string & d) {
            return d.compare(0, 4, "ROCm") == 0 || d.compare(0, 3, "HIP") == 0;
        });
        if (wants_hip && !c.native_only) {
            require(staging_enabled && staging_max_env && staging_max >= largest_packet,
                "HIP fixture requires explicit GGML_HIP_NO_PEER_STAGING=1 and "
                "GGML_HIP_NO_PEER_STAGING_MAX_BYTES covering the largest X/partial packet");
        }
        std::printf("{\"kind\":\"transport_policy\",\"pinned_no_peer_staging_configured\":%s,"
            "\"staging_max_bytes\":%zu,\"largest_X_or_partial_packet_bytes\":%zu,"
            "\"actual_peer_or_staged_path\":\"not_inferred_from_environment_use_backend_trace\"}\n",
            staging_enabled ? "true" : "false", staging_max, largest_packet);
        if (wants_gpu) { ggml_backend_load_all(); }
        std::vector<std::unique_ptr<backend_guard>> guards;
        std::vector<ggml_backend_t> backends;
        for (size_t p = 0; p < c.devices.size(); ++p) {
            ggml_backend_t b;
            if (c.devices[p] == "CPU") {
                b = ggml_backend_cpu_init();
                require(b != nullptr, "CPU initialization failed");
                ggml_backend_cpu_set_n_threads(b, 2);
            } else {
                for (size_t prev = 0; prev < p; ++prev) { require(c.devices[prev] != c.devices[p], "duplicate GPU owner"); }
                ggml_backend_dev_t dev = ggml_backend_dev_by_name(c.devices[p].c_str());
                require(dev != nullptr, "unknown explicit device " + c.devices[p]);
                b = ggml_backend_dev_init(dev, nullptr);
            }
            guards.emplace_back(new backend_guard(b)); backends.push_back(b);
            std::printf("{\"kind\":\"owner\",\"owner\":%zu,\"device\":\"%s\",\"expert_begin\":%d,\"expert_end\":%d}\n",
                p, ggml_backend_name(b), bounds[p], bounds[p + 1]);
        }
        const auto weight_start = clock_type::now();
        banks full(src, c.experts, backends[0]);
        std::vector<std::unique_ptr<banks>> weights;
        if (!c.native_only) {
            for (size_t p = 0; p < backends.size(); ++p) {
                weights.emplace_back(new banks(src, bounds[p + 1] - bounds[p], backends[p]));
            }
        }
        std::string bank_hash;
        load_banks(src, full, weights, bounds, c.sequence.empty() ? nullptr : &bank_hash);
        std::printf("{\"kind\":\"weight_setup\",\"ms\":%.6f}\n", elapsed_ms(weight_start));
        if (!c.sequence.empty()) {
            run_sequence(c, capture, src, bank_hash, bounds, backends, full, weights, tolerance);
            return 0;
        }
        for (const auto & item : cases) {
            const int q = item.q;
            const std::string & pattern = item.name;
            const replay_case * captured = c.route_capture.empty() ? nullptr : &capture.cases[item.capture_index];
            const auto setup_start = clock_type::now();
            const auto ids = sampled ? item.sampled_ids : (captured ? captured->ids : routes(c, q, pattern, bounds));
            reference ref(c, q, full, backends[0]);
            std::mt19937 rng(123 + q);
            std::normal_distribution<float> normal(0, 0.5f);
            std::vector<float> x(c.d * q), coeff(c.used * q);
            for (float & v : x) { v = normal(rng); }
            for (int t = 0; t < q; ++t) {
                for (int s = 0; s < c.used; ++s) {
                    coeff[t * c.used + s] = 2.0f * (s + 1) / (c.used * (c.used + 1));
                }
            }
            if (captured) { coeff = captured->coeff; }
            std::vector<int32_t> padded(c.experts * q, 0);
            for (int t = 0; t < q; ++t) {
                std::copy_n(ids.begin() + t * c.used, c.used, padded.begin() + t * c.experts);
            }
            ggml_backend_tensor_set(ref.x, x.data(), 0, x.size() * sizeof(float));
            ggml_backend_tensor_set(ref.coeff, coeff.data(), 0, coeff.size() * sizeof(float));
            ggml_backend_tensor_set(ref.ids_storage, padded.data(), 0, padded.size() * sizeof(int32_t));
            if (c.native_only) {
                set_test_mmq_j(0);
                const double default_ms = run_reference(ref, backends[0]);
                std::vector<float> baseline(ggml_nelements(ref.result));
                ggml_backend_tensor_get(ref.result, baseline.data(), 0, baseline.size() * sizeof(float));
                if (c.mmq_force_j) {
                    set_test_mmq_j(c.mmq_force_j);
                }
                const double cold_ms = c.mmq_force_j ? run_reference(ref, backends[0]) : default_ms;
                const errors err = compare_values(baseline, ref.result);
                require(err.nmse <= tolerance, "forced-J native numerical gate failed before timing");
                std::printf("{\"kind\":\"native_j_correctness\",\"q\":%d,\"pattern\":\"%s\",\"forced_J\":%d,"
                    "\"default_ms\":%.6f,\"cold_ms\":%.6f,\"nmse\":%.12g,\"max_abs\":%.12g}\n",
                    q, pattern.c_str(), c.mmq_force_j, default_ms, cold_ms, err.nmse, err.max_abs);
                for (int i = 0; i < c.warmups; ++i) { run_reference(ref, backends[0]); }
                std::vector<double> samples;
                for (int i = 0; i < c.repetitions; ++i) {
                    const double ms = run_reference(ref, backends[0]);
                    const errors repeat = compare_values(baseline, ref.result);
                    require(repeat.nmse <= tolerance, "forced-J native numerical gate failed during timing");
                    samples.push_back(ms);
                    std::printf("{\"kind\":\"native_j_sample\",\"q\":%d,\"pattern\":\"%s\",\"forced_J\":%d,"
                        "\"sample\":%d,\"ms\":%.6f,\"nmse\":%.12g}\n",
                        q, pattern.c_str(), c.mmq_force_j, i, ms, repeat.nmse);
                }
                std::vector<int> expert_counts(c.experts);
                for (int32_t id : ids) { ++expert_counts[id]; }
                int active_experts = 0;
                size_t executed_tiles = 0;
                for (int count : expert_counts) {
                    active_experts += count > 0;
                    if (c.mmq_force_j) { executed_tiles += (count + c.mmq_force_j - 1) / c.mmq_force_j; }
                }
                size_t bank_bytes_per_expert = 0;
                for (const auto & bank : src) { bank_bytes_per_expert += ggml_row_size(bank.type, bank.k) * bank.m; }
                std::printf("{\"kind\":\"native_j_result\",\"q\":%d,\"pattern\":\"%s\",\"forced_J\":%d,"
                    "\"median_ms\":%.6f,\"active_experts\":%d,\"assignments\":%zu,\"executed_tiles\":%zu,"
                    "\"executed_cols\":%zu,\"estimated_weight_read_bytes\":%zu,\"useful_tflops\":%.6f}\n",
                    q, pattern.c_str(), c.mmq_force_j, median(samples), active_experts, ids.size(), executed_tiles,
                    executed_tiles * size_t(c.mmq_force_j), executed_tiles * bank_bytes_per_expert,
                    6.0 * c.d * c.f * ids.size() / (median(samples) * 1e9));
                std::fflush(stdout);
                continue;
            }
            packed_reference packed_ref(c, q, full, backends[0], ref);
            ep_graph ep(c, q, bounds, backends, ref, weights, ids);
            ep_graph bucketed_ep(c, q, bounds, backends, ref, weights, ids, true);
            const double prepare_ms = elapsed_ms(setup_start);
            const auto parts = partition(ids, q, c.used, bounds);
            size_t remote = 0, active_owners = 0, bucket_upload_bytes = 0;
            for (size_t p = 0; p < parts.size(); ++p) {
                if (p && !parts[p].expert.empty()) { ++remote; }
                if (!parts[p].expert.empty()) { ++active_owners; }
                std::printf("{\"kind\":\"assignment_shape\",\"q\":%d,\"pattern\":\"%s\",\"owner\":%zu,"
                    "\"native_n_used\":1,\"native_n_tokens\":%zu}\n", q, pattern.c_str(), p, parts[p].expert.size());
                if (bucketed_ep.owners[p]) {
                    bucket_upload_bytes += ggml_nbytes(bucketed_ep.owners[p]->metadata);
                    for (int r = 1; r <= c.used; ++r) {
                        const size_t nb = bucketed_ep.owners[p]->bucket_counts[r];
                        if (!nb) { continue; }
                        std::printf("{\"kind\":\"bucket_shape\",\"q\":%d,\"pattern\":\"%s\",\"owner\":%zu,"
                            "\"native_n_used\":%d,\"native_n_tokens\":%zu,\"real_assignments\":%zu}\n",
                            q, pattern.c_str(), p, r, nb, r * nb);
                    }
                }
            }
            const size_t route_readback_bytes = ids.size() * sizeof(int32_t);
            const size_t index_upload_bytes = 3 * route_readback_bytes;
            const size_t activation_bytes = remote * size_t(c.d) * q * sizeof(float);
            const size_t coeff_bytes = remote * size_t(c.used) * q * sizeof(float);
            std::printf("{\"kind\":\"case\",\"q\":%d,\"pattern\":\"%s\",\"prepare_ms\":%.6f,"
                "\"route_D2H_bytes\":%zu,\"index_H2D_bytes\":%zu,\"fanout_activation_bytes\":%zu,"
                "\"fanout_coefficient_bytes\":%zu,\"return_partial_bytes\":%zu,\"logical_device_copy_messages\":%zu,"
                "\"logical_cross_device_payload_bytes\":%zu,\"logical_all_transit_payload_bytes\":%zu,"
                "\"bucketed_index_H2D_bytes\":%zu,\"flat_index_upload_calls\":%zu,\"bucketed_index_upload_calls\":%zu,"
                "\"bucketed_logical_all_transit_payload_bytes\":%zu}\n",
                q, pattern.c_str(), prepare_ms, route_readback_bytes, index_upload_bytes,
                activation_bytes, coeff_bytes, activation_bytes, remote * 3,
                2 * activation_bytes + coeff_bytes,
                route_readback_bytes + index_upload_bytes + 2 * activation_bytes + coeff_bytes,
                bucket_upload_bytes, 3 * active_owners, active_owners,
                route_readback_bytes + bucket_upload_bytes + 2 * activation_bytes + coeff_bytes);
            const double cold_ref = run_reference(ref, backends[0]);
            const double cold_packed = run_packed_reference(packed_ref, backends[0]);
            const double cold_ep = ep.run();
            const double cold_bucketed = bucketed_ep.run();
            const errors initial = compare(ref.result, ep.result);
            const errors initial_packed = compare(ref.result, packed_ref.result);
            const errors initial_bucketed = compare(ref.result, bucketed_ep.result);
            std::printf("{\"kind\":\"cold_correctness\",\"q\":%d,\"pattern\":\"%s\",\"baseline_ms\":%.6f,"
                "\"packed_local_ms\":%.6f,\"ep_ms\":%.6f,\"nmse\":%.12g,\"max_abs\":%.12g,\"max_ref\":%.12g,"
                "\"packed_local_nmse\":%.12g,\"packed_local_max_abs\":%.12g,\"bucketed_ep_ms\":%.6f,"
                "\"bucketed_ep_nmse\":%.12g,\"bucketed_ep_max_abs\":%.12g}\n",
                q, pattern.c_str(), cold_ref, cold_packed, cold_ep, initial.nmse, initial.max_abs, initial.max_ref,
                initial_packed.nmse, initial_packed.max_abs, cold_bucketed, initial_bucketed.nmse, initial_bucketed.max_abs);
            require(initial.nmse <= tolerance, "numerical gate failed before warm timing");
            require(initial_packed.nmse <= tolerance, "packed-local numerical gate failed before warm timing");
            require(initial_bucketed.nmse <= tolerance, "bucketed EP numerical gate failed before warm timing");
            for (int i = 0; i < c.warmups; ++i) {
                run_reference(ref, backends[0]); run_packed_reference(packed_ref, backends[0]); ep.run(); bucketed_ep.run();
            }
            std::vector<double> ref_ms, packed_ms, ep_ms, bucketed_ms;
            // Cycle every ordering to balance preceding-cache and thermal
            // effects. Multiples of 24 samples give every ordering equal use.
            std::array<int, 4> order = {{0, 1, 2, 3}};
            const char * variant_names[] = {"native", "packed", "flatEP", "bucketEP"};
            for (int i = 0; i < c.repetitions; ++i) {
                double times[4] = {};
                std::string order_name;
                for (int variant : order) {
                    if (!order_name.empty()) { order_name += '_'; }
                    order_name += variant_names[variant];
                    if (variant == 0) { times[0] = run_reference(ref, backends[0]); }
                    else if (variant == 1) { times[1] = run_packed_reference(packed_ref, backends[0]); }
                    else if (variant == 2) { times[2] = ep.run(); }
                    else { times[3] = bucketed_ep.run(); }
                }
                std::next_permutation(order.begin(), order.end());
                const errors err = compare(ref.result, ep.result);
                const errors packed_err = compare(ref.result, packed_ref.result);
                const errors bucketed_err = compare(ref.result, bucketed_ep.result);
                require(err.nmse <= tolerance, "repeated numerical gate failed");
                require(packed_err.nmse <= tolerance, "repeated packed-local numerical gate failed");
                require(bucketed_err.nmse <= tolerance, "repeated bucketed EP numerical gate failed");
                ref_ms.push_back(times[0]); packed_ms.push_back(times[1]); ep_ms.push_back(times[2]);
                bucketed_ms.push_back(times[3]);
                std::printf("{\"kind\":\"sample\",\"q\":%d,\"pattern\":\"%s\",\"sample\":%d,"
                    "\"order\":\"%s\",\"baseline_ms\":%.6f,\"packed_local_ms\":%.6f,\"ep_ms\":%.6f,\"nmse\":%.12g,"
                    "\"packed_local_nmse\":%.12g,\"bucketed_ep_ms\":%.6f,\"bucketed_ep_nmse\":%.12g,"
                    "\"cpu_interval_route_readback_partition_ms\":%.6f,\"cpu_interval_dispatch_submit_ms\":%.6f,"
                    "\"cpu_interval_compute_submit_ms\":%.6f,\"cpu_interval_collect_sync_ms\":%.6f,"
                    "\"bucketed_cpu_interval_route_readback_partition_ms\":%.6f,\"bucketed_cpu_interval_dispatch_submit_ms\":%.6f,"
                    "\"bucketed_cpu_interval_compute_submit_ms\":%.6f,\"bucketed_cpu_interval_collect_sync_ms\":%.6f}\n",
                    q, pattern.c_str(), i, order_name.c_str(), times[0], times[1], times[2], err.nmse, packed_err.nmse,
                    times[3], bucketed_err.nmse,
                    ep.route_host_ms, ep.dispatch_submit_ms, ep.compute_submit_ms, ep.collect_wait_ms,
                    bucketed_ep.route_host_ms, bucketed_ep.dispatch_submit_ms, bucketed_ep.compute_submit_ms, bucketed_ep.collect_wait_ms);
                std::fflush(stdout);
            }
            std::printf("{\"kind\":\"result\",\"q\":%d,\"pattern\":\"%s\",\"baseline_median_ms\":%.6f,"
                "\"packed_local_median_ms\":%.6f,\"ep_median_ms\":%.6f,\"speedup\":%.6f,"
                "\"native_over_packed_local_speedup\":%.6f,\"packed_local_over_ep_speedup\":%.6f,"
                "\"bucketed_ep_median_ms\":%.6f,\"native_over_bucketed_ep_speedup\":%.6f,\"flat_over_bucketed_ep_speedup\":%.6f}\n",
                q, pattern.c_str(), median(ref_ms), median(packed_ms), median(ep_ms), median(ref_ms) / median(ep_ms),
                median(ref_ms) / median(packed_ms), median(packed_ms) / median(ep_ms),
                median(bucketed_ms), median(ref_ms) / median(bucketed_ms), median(ep_ms) / median(bucketed_ms));
        }
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "expert-parallel fixture: %s\n", e.what());
        return 1;
    }
}
