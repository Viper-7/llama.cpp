#include "server-common.h"
#include "server-chat.h"
#include "server-queue.h"
#include "server-schema.h"

#include "ggml.h"
#include "http.h"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

static void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

static std::string binary_string(const void * data, size_t size) {
    return std::string(static_cast<const char *>(data), size);
}

static void require_close(float expected, float actual) {
    require(std::fabs(expected - actual) < 1e-3f, "embedding value mismatch");
}

int main() {
    {
        json request = {{"instructions", "one system block"}, {"input", json::array({
            json{{"type", "message"}, {"role", "user"}, {"content", json::array({
                json{{"type", "input_text"}, {"text", "question"}},
                json{{"type", "input_image"}, {"image_url", "data:image/png;base64,fixture"}}
            })}},
            json{{"type", "reasoning"}, {"summary", json::array()}, {"content", json::array({json{{"type", "reasoning_text"}, {"text", "thought"}}})}},
            json{{"type", "message"}, {"role", "assistant"}, {"content", json::array({json{{"type", "output_text"}, {"text", "answer"}}})}},
            json{{"type", "function_call"}, {"name", "lookup"}, {"call_id", "call_fixed"}, {"arguments", "{}"}},
            json{{"type", "function_call_output"}, {"call_id", "call_fixed"}, {"output", "result"}}
        })}, {"tools", json::array({json{{"type", "function"}, {"name", "lookup"}, {"parameters", json::object()}}})}};
        const auto mapped = server_chat_convert_responses_to_chatcmpl(request);
        const auto & messages = mapped.at("messages");
        require(messages.size() == 4, "Responses conversion duplicated history/configuration");
        require(messages[0].at("content") == "one system block", "Response instructions lost");
        require(messages[1].at("content")[1].at("image_url").at("url") == "data:image/png;base64,fixture", "Response image ordering lost");
        require(messages[2].at("reasoning_content") == "thought" && messages[2].at("tool_calls")[0].at("id") == "call_fixed", "Response reasoning/call lineage lost");
        require(messages[3].at("tool_call_id") == "call_fixed", "Tool result pairing lost");
        server_task_result_cmpl_final final;
        final.res_type = TASK_RESPONSE_TYPE_OAI_RESP;
        final.stop = STOP_TYPE_LIMIT;
        final.n_prompt_tokens = 100;
        final.n_prompt_tokens_cache = 90;
        final.n_decoded = 1;
        final.content = common_chat_input("x");
        const auto body = final.to_json_oaicompat_resp();
        require(body.at("status") == "incomplete", "Response token limit reported completion");
        require(body.at("usage").at("input_tokens_details").at("cached_tokens") == 90, "Native Responses cache counts lost");
        const auto events = final.to_json_oaicompat_resp_stream();
        require(events.back().at("data").at("type") == "response.incomplete", "Streaming token limit reported completion");
    }
    {
        auto first_deferred_id = [](std::initializer_list<std::tuple<int, int, bool>> candidates,
                                    int freed_slot) {
            server_queue queue;
            int selected = -1;
            queue.on_new_task([&](server_task && task, bool) {
                selected = task.id;
                queue.terminate();
                return true;
            });
            queue.on_update_slots([]() {});
            for (const auto & [id, slot, interactive] : candidates) {
                server_task task(SERVER_TASK_TYPE_COMPLETION);
                task.id = id;
                task.id_slot = slot;
                task.params.cache_control.priority = interactive
                    ? server_cache_priority::interactive : server_cache_priority::background;
                queue.defer(std::move(task));
            }
            queue.pop_deferred_task(freed_slot);
            queue.start_loop();
            return selected;
        };
        require(first_deferred_id({{1, -1, false}, {2, -1, true}, {3, -1, true}}, 0) == 2,
                "interactive deferred requests must precede background, FIFO within priority");
        require(first_deferred_id({{1, -1, true}, {2, 1, false}, {3, 1, true}}, 1) == 3,
                "slot-affinity must precede global requests; interactive wins within affinity");
        require(first_deferred_id({{1, -1, false}, {2, -1, false}}, 0) == 1,
                "default background deferred requests must preserve FIFO");
    }
    {
        using namespace server_schema;
        const auto defaults = parse_cache_control(json::object());
        require(defaults.priority == server_cache_priority::background, "cache priority default");
        require(defaults.lifecycle == server_cache_lifecycle::retained, "cache lifecycle default");
        const auto interactive = parse_cache_control(json{
            {"priority", "interactive"}, {"lifecycle", "ephemeral"}, {"branch", "tool-branch"},
        });
        require(interactive.priority == server_cache_priority::interactive, "cache interactive priority");
        require(interactive.lifecycle == server_cache_lifecycle::ephemeral, "cache ephemeral lifecycle");
        require(interactive.branch == "tool-branch", "cache opaque branch");
        require(parse_cache_control(json{{"lifecycle", "one_shot"}}).lifecycle == server_cache_lifecycle::one_shot,
                "cache one-shot lifecycle");
        const server_cache_eviction_rank released{0, 100, 100};
        const server_cache_eviction_rank background_ephemeral{0.5, 100, 100};
        const server_cache_eviction_rank background_retained{2, 100, 100};
        const server_cache_eviction_rank interactive_ephemeral{3, 100, 100};
        const server_cache_eviction_rank interactive_retained{4, 100, 100};
        require(released.less_than(background_ephemeral), "released branch should evict first");
        require(background_ephemeral.less_than(background_retained), "ephemeral branch eviction order");
        require(background_retained.less_than(interactive_ephemeral), "interactive priority eviction order");
        require(interactive_ephemeral.less_than(interactive_retained), "retained interactive eviction order");
        require(server_cache_eviction_rank{2, 100, 200}.less_than({2, 100, 100}),
                "larger state with equal recomputation should evict first");
        require(!server_cache_eviction_rank{2, 100, 100}.less_than({2, 100, 100}),
                "equal eviction ranks must preserve LRU order");
        for (const json & invalid : {
                json(nullptr), json{{"priority", "urgent"}}, json{{"priority", true}},
                json{{"lifecycle", "temporary"}}, json{{"lifecycle", "one_shot"}, {"branch", "x"}},
                json{{"branch", ""}},
                json{{"branch", std::string(129, 'x')}}, json{{"unknown", true}},
            }) {
            bool rejected = false;
            try {
                parse_cache_control(invalid);
            } catch (const std::invalid_argument &) {
                rejected = true;
            }
            require(rejected, "invalid cache control accepted");
        }
    }
    static_assert(sizeof(llama_pos) == sizeof(int32_t));

    const llama_tokens raw_text = { 11, 12, 13 };
    const server_tokens mtmd_capable_text(raw_text, true);
    require(!mtmd_capable_text.has_media(), "plain text unexpectedly contains media");
    require(mtmd_capable_text.get_text_tokens_view() == raw_text,
            "media-free mtmd token view changed plain text tokens");
    const auto text_boundaries = mtmd_capable_text.page_boundaries(2);
    require(text_boundaries.size() == 1 && text_boundaries[0].token_count == 2 &&
            text_boundaries[0].position == 2,
            "plain-text page boundary changed");
    for (size_t count = 0; count <= raw_text.size(); ++count) {
        require(mtmd_capable_text.complete_boundary(count), "text endpoint is not complete");
    }

    mtmd::input_chunks test_chunks(mtmd_test_create_input_chunks());
    server_tokens media_tokens(test_chunks, true);
    require(media_tokens.has_media() && media_tokens.has_stable_media_identity(),
            "test media tokens have no stable cache identity");
    const auto media_boundaries = media_tokens.page_boundaries(6);
    require(!media_boundaries.empty(), "media page boundaries are empty");
    require(media_tokens.complete_boundary(0), "empty media prefix is not complete");
    require(media_tokens.complete_boundary(media_tokens.size()), "complete media prompt is not a boundary");
    for (size_t idx = 0; idx < media_tokens.size(); ++idx) {
        if (media_tokens[idx] != LLAMA_TOKEN_NULL) {
            continue;
        }
        const auto & chunk = media_tokens.find_chunk(idx);
        const size_t end = idx + mtmd_input_chunk_get_n_tokens(chunk.get());
        for (size_t inside = idx + 1; inside < end; ++inside) {
            require(!media_tokens.complete_boundary(inside), "media chunk interior became a graph endpoint");
        }
        require(media_tokens.complete_boundary(idx), "media start is not a complete boundary");
        require(media_tokens.complete_boundary(end), "media end is not a complete boundary");
        auto complete = media_tokens.clone();
        complete.keep_first(end);
        require(complete.size() == end, "complete media chunk boundary could not be retained");
        if (end > idx + 1) {
            auto partial = media_tokens.clone();
            bool rejected = false;
            try { partial.keep_first(idx + 1); } catch (const std::runtime_error &) { rejected = true; }
            require(rejected, "partial media chunk truncation was accepted");
        }
        idx = end - 1;
    }
    for (const auto & boundary : media_boundaries) {
        require(boundary.token_count <= media_tokens.size(), "media page boundary exceeds prompt");
        require(boundary.position == media_tokens.pos_next(boundary.token_count),
                "media page boundary position mismatch");
    }
    const auto media_prefix_hash = media_tokens.cache_prefix_hash(media_boundaries.front().token_count);
    const auto media_clone = media_tokens.clone();
    require(media_clone.cache_prefix_hash(media_boundaries.front().token_count) == media_prefix_hash,
            "cloned media prefix changed cache identity");
    server_tokens placeholder_tokens;
    placeholder_tokens.has_mtmd = true;
    for (size_t i = 0; i < test_chunks.size(); ++i) {
        placeholder_tokens.push_back_placeholder(test_chunks[i]);
    }
    require(placeholder_tokens.cache_prefix_hash(media_boundaries.front().token_count) == media_prefix_hash,
            "processed media placeholder changed cache identity");

    // A page-cache hit on a text task adopts a clone of the task container,
    // which is intentionally not MTMD-capable. The persistent slot must restore
    // its server-level MTMD capability before a later overlapping image prompt
    // reaches push_back_placeholder().
    server_tokens restored_text = server_tokens(raw_text, false).clone();
    require(!restored_text.has_mtmd, "text task unexpectedly carried MTMD capability");
    restored_text.has_mtmd = true;
    for (size_t i = 0; i < test_chunks.size(); ++i) {
        restored_text.push_back_placeholder(test_chunks[i]);
    }
    require(restored_text.has_media(), "text restore could not accept a later media placeholder");

    httplib::Server server;
    std::atomic<bool> mismatch_modality(false);
    std::atomic<bool> request_ok(true);
    std::atomic<int> post_count(0);
    std::atomic<int> pos_count(0);
    std::string base_url;

    server.Post("/v1/vision/encode", [&](const httplib::Request & req, httplib::Response & res) {
        post_count++;
        if (!req.form.has_field("type") || !req.form.has_field("dtype") || !req.form.has_file("file")) {
            request_ok = false;
            res.status = 400;
            return;
        }

        const auto type = req.form.get_field("type");
        const auto dtype = req.form.get_field("dtype");
        const auto file = req.form.get_file("file");
        if (dtype != "fp16") request_ok = false;
        if (type == "image") {
            if (file.filename != "input.png" || file.content != std::string("\x89PNG", 4)) request_ok = false;
            const float values[] = {1.0f, -2.0f, 3.5f, 4.0f, 5.0f, -6.0f};
            ggml_fp16_t values_f16[6];
            ggml_fp32_to_fp16_row(values, values_f16, 6);
            res.set_content(binary_string(values_f16, sizeof(values_f16)), "application/octet-stream");
            res.set_header("X-N-Visual-Tokens", "2");
            res.set_header("X-Hidden-Size", "3");
            res.set_header("X-N-Pos-Per-Embd", "4");
            res.set_header("X-N-Pos", "7");
            res.set_header("X-Requires-Non-Causal-Attn", "1");
            res.set_header("X-Sha256", "image-cache-id");
            res.set_header("X-MM-Type", mismatch_modality ? "video" : "image");
            res.set_header("X-Dtype", "fp16");
            res.set_header("X-Pos-Path", "/positions/image");
        } else if (type == "video") {
            if (file.filename != "input.mp4" || file.content != "MP4") request_ok = false;
            const float values[] = {8.0f, -9.5f};
            res.set_content(binary_string(values, sizeof(values)), "application/octet-stream");
            res.set_header("X-N-Visual-Tokens", "1");
            res.set_header("X-Hidden-Size", "2");
            res.set_header("X-N-Pos-Per-Embd", "2");
            res.set_header("X-N-Pos", "3");
            res.set_header("X-MM-Type", "video");
            res.set_header("X-Dtype", "fp32");
            res.set_header("X-Pos-Path", base_url + "/positions/video");
        } else {
            request_ok = false;
            res.status = 400;
        }
    });

    server.Get("/positions/image", [&](const httplib::Request &, httplib::Response & res) {
        pos_count++;
        const int32_t positions[] = {0, 1, 2, 3, 4, 5, 6, 7};
        res.set_content(binary_string(positions, sizeof(positions)), "application/octet-stream");
    });
    server.Get("/positions/video", [&](const httplib::Request &, httplib::Response & res) {
        pos_count++;
        const int32_t positions[] = {11, 12};
        res.set_content(binary_string(positions, sizeof(positions)), "application/octet-stream");
    });

    const int port = server.bind_to_any_port("127.0.0.1");
    require(port > 0, "failed to bind mock visiontower");
    base_url = "http://127.0.0.1:" + std::to_string(port);
    std::thread server_thread([&server] { server.listen_after_bind(); });

    try {
        const auto image = visiontower_encode(base_url + "/", false, {0x89, 'P', 'N', 'G'});
        require(image.n_tokens == 2 && image.n_embd == 3, "wrong image geometry");
        require(image.n_pos == 7 && image.n_pos_per_embd == 4, "wrong image position geometry");
        require(image.use_non_causal, "non-causal flag was not preserved");
        require(image.sha256 == "image-cache-id", "cache id was not preserved");
        require(image.embd.size() == 6 && image.pos.size() == 8, "wrong image payload size");
        require_close(1.0f, image.embd[0]);
        require_close(-6.0f, image.embd[5]);
        require(image.pos.front() == 0 && image.pos.back() == 7, "wrong image positions");

        const auto video = visiontower_encode(base_url, true, {'M', 'P', '4'});
        require(video.n_tokens == 1 && video.n_embd == 2, "wrong video geometry");
        require(video.embd.size() == 2 && video.pos.size() == 2, "wrong video payload size");
        require_close(8.0f, video.embd[0]);
        require_close(-9.5f, video.embd[1]);
        require(video.pos[0] == 11 && video.pos[1] == 12, "wrong video positions");

        mismatch_modality = true;
        bool rejected = false;
        try {
            (void) visiontower_encode(base_url, false, {0x89, 'P', 'N', 'G'});
        } catch (const std::runtime_error & err) {
            rejected = std::string(err.what()).find("modality") != std::string::npos;
        }
        require(rejected, "mismatched response modality was accepted");
        require(request_ok, "visiontower request contract mismatch");
        require(post_count == 3 && pos_count == 2, "unexpected mock request count");
    } catch (...) {
        server.stop();
        server_thread.join();
        throw;
    }

    server.stop();
    server_thread.join();
    return 0;
}
