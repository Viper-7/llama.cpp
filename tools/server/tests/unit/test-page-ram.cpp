#include "tools/server/server-page-ram.h"
#include <cassert>
#include <sys/wait.h>
#include <iostream>

using ram_json = server_page_ram_store::json;
static const size_t limit = 1024 * 1024;
static std::string identity(64, 'a');
static ram_json empty_bytes() { server_page_bytes bytes; return bytes.metadata(); }
static ram_json make_page(server_page_ram_store & store, const char * key, const char * parent, size_t begin, size_t end, const char * branch) {
    server_page_bytes bytes;
    store.allocate(bytes, 4096);
    std::fill(bytes.data(), bytes.data() + bytes.size(), uint8_t(17));
    bytes.seal();
    return {{"key", key}, {"parent", parent}, {"p0_tokens", begin}, {"p1_tokens", end},
        {"p0_pos", begin}, {"p1_pos", end}, {"coarse", true}, {"unowned_retained", false},
        {"unowned_interactive", false}, {"branches", ram_json::array({ram_json::array({branch, 0, 0})})},
        {"page_tokens", 2048}, {"target", bytes.metadata()}, {"draft", empty_bytes()}, {"reference", empty_bytes()}};
}
#ifdef TEST_PAGE_GRAPH
// Cached publication must equal a freshly rebuilt descriptor for every node.
static void check_descriptors(server_prompt_page_cache & cache) {
    assert(cache.persist());
    const auto cached = cache.ram_store.read();
    const auto builds = cache.stats().manifest_node_builds;
    const auto reuses = cache.stats().manifest_node_reuses;
    assert(cache.persist());
    assert(cache.stats().manifest_node_builds == builds);
    assert(cache.stats().manifest_node_reuses == reuses + cache.count());
    for (auto & node : cache.pages) node.descriptor_dirty = true;
    assert(cache.persist());
    assert(cache.ram_store.read() == cached);
}
// Substitute only raw-state byte extraction. The real graph/capture policy
// runs without a model, GPU allocation or llama context.
static llama_pos captured_p0 = -1, captured_p1 = -1;
size_t llama_state_seq_get_size_range_ext(llama_context *, llama_seq_id,
        llama_pos p0, llama_pos p1, llama_state_seq_flags) {
    assert(p0 >= 0 && p1 > p0);
    return 128;
}
size_t llama_state_seq_get_data_range_ext(llama_context *, uint8_t * dst,
        size_t size, llama_seq_id, llama_pos p0, llama_pos p1, llama_state_seq_flags) {
    assert(size == 128);
    captured_p0 = p0; captured_p1 = p1;
    std::fill(dst, dst + size, uint8_t(31));
    return size;
}
size_t llama_state_seq_set_data_ext(llama_context *, const uint8_t * data, size_t size,
        llama_seq_id, llama_state_seq_flags) {
    assert(data && size == 128 && data[0] == 31);
    return size;
}
#endif
int main() {
    unsetenv("LLAMA_SERVER_PAGE_CACHE_SSD_DIR");
    unsetenv("LLAMA_SERVER_PAGE_CACHE_TIER_STATS_PATH");
    const std::string root = "/dev/shm/llama-page-ram-test-" + std::to_string(getpid());
    setenv("LLAMA_SERVER_PAGE_CACHE_RAM_DIR", root.c_str(), 1);
    setenv("LLAMA_SERVER_PAGE_CACHE_NAMESPACE", identity.c_str(), 1);
    // _exit skips destructors: committed shared pages must survive a killed owner.
    pid_t child = fork();
    if (child == 0) {
        server_page_ram_store store(limit); assert(store.active());
        ram_json graph = ram_json::array(); graph.push_back(make_page(store, "root", "", 0, 2048, "prompt:old"));
        graph.push_back(make_page(store, "leaf", "root", 2048, 4096, "conversation"));
        assert(store.write(graph));
        server_page_bytes incomplete; store.allocate(incomplete, 4096);
        std::ofstream pending(root + "/" + identity + "/graph.pending"); pending << "{partial"; pending.flush();
        _exit(0);
    }
    int status = 0; waitpid(child, &status, 0); assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    {
        server_page_ram_store store(limit); assert(store.active());
        server_page_ram_store locked(limit); assert(!locked.active() && locked.failures == 1);
        auto graph = store.read(); assert(graph.size() == 2);
        server_page_bytes bytes; size_t budget = limit;
        store.load(bytes, graph[0]["target"], budget); assert(bytes.size() == 4096 && bytes.data()[0] == 17);
        assert(store.write(graph)); assert(store.orphan_files == 1);
        size_t insufficient = 1; server_page_bytes too_large;
        bool failed = false; try { store.load(too_large, graph[0]["target"], insufficient); } catch (...) { failed = true; } assert(failed);
    }
#ifdef TEST_PAGE_GRAPH
    {
        server_prompt_page_cache cache(limit, 2048);
        assert(cache.count() == 2);
        assert(cache.persistence_stats().at("recovered_pages").get<uint64_t>() == 2);
        // Releasing an obsolete prompt must preserve the root used by a context leaf.
        assert(cache.prepare_lease("primary", "head", "", 0, 60000));
        assert(!cache.observe_lease("primary", "head", "missing", 4096, true));
        assert(!cache.observe_lease("primary", "head", "conversation", 8192, true));
        assert(cache.observe_lease("primary", "head", "conversation", 4096, true));
        assert(cache.commit_lease("primary", "head", "", 60000));
        const auto pin_now = server_cache_now_ms();
        assert(cache.eviction_candidate({}, pin_now + 30001) == cache.pages.end());
        assert(!cache.release_lease("primary", "stale"));
        assert(cache.release_lease("primary", "head"));
        assert(cache.release_branch("prompt:old") == 1); assert(cache.count() == 2);
        assert(cache.release_branch("conversation") == 1); assert(cache.count() == 2);
        check_descriptors(cache);
        const uint64_t released = server_cache_now_ms();
        const uint64_t deadline = cache.by_key.at("leaf")->release_after_ms;
        assert(deadline > released && deadline <= released + 30000);
        assert(cache.eviction_candidate({}, released) == cache.pages.end());
        // Repeating an already-removed claim must not prolong the grace.
        assert(cache.release_branch("conversation") == 0);
        assert(cache.by_key.at("leaf")->release_after_ms == deadline);
        // Replacement adopts only the common root. Its old suffix remains
        // protected until expiry, then is the first zero-score candidate.
        server_cache_control replacement;
        replacement.branch = "prompt:new";
        replacement.priority = server_cache_priority::background;
        replacement.lifecycle = server_cache_lifecycle::retained;
        cache.touch_chain({"root"}, replacement);
        assert(cache.by_key.at("root")->release_after_ms == 0);
        assert(cache.by_key.at("root")->branches.count("prompt:new"));
        check_descriptors(cache);
        assert(cache.eviction_candidate({}, deadline - 1) == cache.pages.end());
        auto expired = cache.eviction_candidate({}, deadline);
        assert(expired != cache.pages.end() && expired->key == "leaf");
        assert(expired->eviction_rank(deadline).weight == 0);
        assert(cache.prepare_lease("primary", "renew", "head", 1, 60000));
        assert(cache.observe_lease("primary", "renew", "prompt:new", 2048, true));
        assert(cache.commit_lease("primary", "renew", "head", 60000));
        cache.persist();
    }
    {
        server_prompt_page_cache cache(limit, 2048);
        assert(cache.count() == 2);
        auto leaf = cache.by_key.at("leaf");
        const uint64_t deadline = leaf->release_after_ms;
        assert(deadline && cache.by_key.at("root")->release_after_ms == 0);
        auto expired = cache.eviction_candidate({}, deadline);
        assert(expired->key == "leaf");
        cache.evict(expired);
        assert(cache.count() == 1 && cache.by_key.at("root")->branches.count("prompt:new"));
        assert(cache.response_leases.snapshot().at("primary").head == "renew");
        assert(cache.response_leases.protects("root", server_cache_now_ms()));
        const auto pending_path = root + "/" + identity + "/graph.pending";
        std::filesystem::create_directory(pending_path); // manifest write must fail
        assert(!cache.release_lease("primary", "renew"));
        assert(cache.response_leases.protects("root", server_cache_now_ms()));
        std::filesystem::remove(pending_path);
        cache.evict_graph("root", false); // invalid/corrupt graph removal fences ownership
        assert(!cache.response_leases.protects("root", server_cache_now_ms()));
        cache.persist();
    }
    {
        server_page_ram_store store(limit);
        ram_json graph = ram_json::array(); graph.push_back(make_page(store, "leaf", "missing", 2048, 4096, "orphan"));
        assert(store.write(graph));
    }
    { server_prompt_page_cache cache(limit, 2048); assert(cache.count() == 0); assert(cache.persistence_stats().at("rejected_pages").get<uint64_t>() == 1); }
    {
        server_page_ram_store store(limit);
        auto graph = ram_json::array(); graph.push_back(make_page(store, "root", "", 0, 2048, "test"));
        graph.push_back(make_page(store, "bad-child", "root", 2048, 4096, "test"));
        graph.push_back(make_page(store, "bad-grandchild", "bad-child", 4096, 6144, "test"));
        assert(store.write(graph));
        std::ofstream corrupt(root + "/" + identity + "/" + graph[1]["target"]["file"].get<std::string>(), std::ios::binary | std::ios::app);
        corrupt.put('x'); corrupt.close();
    }
    { server_prompt_page_cache cache(limit, 2048); assert(cache.count() == 1); assert(cache.persistence_stats().at("rejected_pages").get<uint64_t>() == 2); }

    // Legacy graphs reopen as cold history. Topological validation must not
    // discard their actual persisted LRU order.
    {
        server_page_ram_store store(limit);
        auto parent = make_page(store, "root", "", 0, 2048, "test");
        auto child = make_page(store, "child", "root", 2048, 4096, "test");
        assert(store.write(ram_json::array({child, parent})));
    }
    { server_prompt_page_cache cache(limit, 2048); assert(cache.count() == 2); }
    {
        server_page_ram_store store(limit);
        auto graph = store.read();
        assert(graph[0]["key"] == "child" && graph[1]["key"] == "root");
        assert(graph[0]["last_access_ms"] == 0 && graph[0]["retrievals"] == 0);
        const auto now = server_cache_now_ms();
        graph[0]["last_access_ms"] = now - 3600000;
        graph[0]["retrievals"] = 7;
        graph[0]["hint_ms"] = std::array<uint64_t, 6>{0, 0, now - 3600000, 0, 0, 0};
        assert(store.write(graph));
    }
    { server_prompt_page_cache cache(limit, 2048); assert(cache.count() == 2); }
    {
        server_page_ram_store store(limit);
        auto graph = store.read();
        assert(graph[0]["retrievals"] == 7 && graph[0]["last_access_ms"].get<uint64_t>() > 0);
        assert(graph[0]["hint_ms"][2] == graph[0]["last_access_ms"]);
        graph[0]["last_access_ms"] = -1;
        graph[0]["retrievals"] = true;
        graph[0]["hint_ms"] = ram_json::array({true, -1, nullptr, "invalid", -1.0, false});
        assert(store.write(graph));
    }
    { server_prompt_page_cache cache(limit, 2048); assert(cache.count() == 2); }
    {
        server_page_ram_store store(limit); const auto graph = store.read();
        assert(graph[0]["last_access_ms"] == 0 && graph[0]["retrievals"] == 0);
        for (const auto & stamp : graph[0]["hint_ms"]) assert(stamp == 0);
    }
    {
        server_page_ram_store store(limit);
        auto old = make_page(store, "stale", "", 0, 2048, "");
        old["branches"] = ram_json::array(); old["unowned_retained"] = true; old["unowned_interactive"] = true;
        auto warm = make_page(store, "warm", "", 0, 2048, "warm");
        const auto now = server_cache_now_ms();
        old["last_access_ms"] = now - 7200000;
        old["hint_ms"] = std::array<uint64_t, 6>{now - 7200000, 0, 0, 0, 0, 0};
        warm["last_access_ms"] = now - 7200000;
        warm["hint_ms"] = std::array<uint64_t, 6>{0, 0, now - 7200000, 0, 0, 0};
        assert(store.write(ram_json::array({warm, old})));
    }
    {
        server_prompt_page_cache cache(limit, 2048);
        const auto now = server_cache_now_ms();
        auto selected = cache.eviction_candidate({}, now);
        assert(selected != cache.pages.end() && selected->key == "stale");
        // Active capture ancestors remain ineligible, even at a lower score.
        assert(cache.eviction_candidate({"stale"}, now)->key == "warm");
        assert(cache.eviction_candidate({"stale", "warm"}, now) == cache.pages.end());
        auto warm = cache.by_key.at("warm");
        server_cache_control control; control.branch = "warm"; control.request_id = 99;
        cache.touch_chain({"warm"}, control); cache.touch_chain({"warm"}, control);
        assert(warm->usage.retrievals == 1);
        const auto before = warm->usage;
        server_cache_control one_shot; one_shot.lifecycle = server_cache_lifecycle::one_shot;
        one_shot.request_id = 100; one_shot.priority = server_cache_priority::interactive;
        cache.touch_chain({"warm"}, one_shot);
        assert(warm->usage.last_access_ms == before.last_access_ms);
        assert(warm->usage.retrievals == before.retrievals && warm->usage.hint_ms == before.hint_ms);
        assert(warm->usage.last_request == before.last_request);
        cache.evict(selected); cache.persist();
        assert(cache.count() == 1);
    }
    {
        server_page_ram_store store(limit); const auto graph = store.read();
        assert(graph.size() == 1 && graph[0]["key"] == "warm" && graph[0]["retrievals"] == 1);
    }
    // A long live hit can bypass the nominal coarse checkpoint. Capturing the
    // tail must bridge to an older rooted exact endpoint instead of dropping it.
    {
        server_page_ram_store store(limit); assert(store.write(ram_json::array()));
    }
    {
        server_prompt_page_cache cache(limit, 2048);
        server_cache_control control;
        const server_tokens root_tokens(llama_tokens(2048, 7), false);
        const server_tokens older_tail(llama_tokens(3000, 7), false);
        const server_tokens tail(llama_tokens(5000, 7), false);
        assert(cache.capture(root_tokens, nullptr, nullptr, 0, control));
        assert(cache.capture(older_tail, nullptr, nullptr, 0, control, true));
        assert(cache.capture_begin(tail, nullptr, {2048, 2048}).token_count == 2048);
        assert(cache.capture_begin(tail, nullptr, {4096, 4096}).token_count == 3000);
        // Incompatible draft-state chains and changed prefixes are not adopted.
        assert(cache.capture_begin(tail, reinterpret_cast<llama_context *>(1), {4096, 4096}).token_count == 0);
        const server_tokens changed(llama_tokens(5000, 8), false);
        assert(cache.capture_begin(changed, nullptr, {4096, 4096}).token_count == 0);
        assert(cache.capture(tail, nullptr, nullptr, 0, control, true));
        assert(captured_p0 == 3000 && captured_p1 == 5000);
        assert(cache.stats().capture_bridges == 1);
        assert(cache.match_tokens(tail) == 5000);
        auto leaf = cache.by_key.at(tail.cache_prefix_hash(5000));
        assert(leaf->p0_tokens == 3000 && leaf->p1_tokens == 5000);
        assert(!leaf->coarse && leaf->parent_key == older_tail.cache_prefix_hash(3000));
        assert(cache.by_key.at(root_tokens.cache_prefix_hash(2048))->children == 1);
        assert(cache.eviction_candidate({leaf->key}, server_cache_now_ms()) == cache.pages.end());
    }
    {
        server_prompt_page_cache cache(limit, 2048);
        const server_tokens continuation(llama_tokens(5001, 7), false);
        assert(cache.match_tokens(continuation) == 5000);
        assert(cache.find_longest_chain(continuation, nullptr, 5000).tokens == 5000);
        // With no matching ancestor, a bounded root snapshot is still valid.
        const server_tokens changed(llama_tokens(5000, 8), false);
        server_cache_control control;
        assert(cache.capture(changed, nullptr, nullptr, 0, control, true));
        assert(captured_p0 == 0 && captured_p1 == 5000);
        assert(cache.match_tokens(changed) == 5000);
    }
    // Supersession needs exact prefix proof, not sibling length. Check the
    // real capture/GC graph with no model/GPU and preserve edit recovery.
    { server_page_ram_store store(limit); assert(store.write(ram_json::array())); }
    std::string obsolete_key, successor_key;
    uint64_t prune_at = 0;
    {
        server_prompt_page_cache cache(limit, 2048);
        server_cache_control control; control.request_id = 201;
        control.priority = server_cache_priority::interactive;
        const server_tokens root_tokens(llama_tokens(2048, 7), false);
        const server_tokens old(llama_tokens(2500, 7), false);
        assert(cache.capture(root_tokens, nullptr, nullptr, 0, control));
        assert(cache.capture(old, nullptr, nullptr, 0, control, true));
        obsolete_key = old.cache_prefix_hash(old.size());
        auto edited_tokens = llama_tokens(2800, 8);
        std::fill(edited_tokens.begin(), edited_tokens.begin() + 2048, 7);
        const server_tokens edited(edited_tokens, false);
        assert(cache.capture(edited, nullptr, nullptr, 0, control, true));
        assert(!cache.by_key.at(obsolete_key)->superseded_at_ms);
        const server_tokens adjacent(llama_tokens(2501, 7), false);
        assert(cache.capture(adjacent, nullptr, nullptr, 0, control, true));
        assert(!cache.by_key.at(obsolete_key)->superseded_at_ms); // logits pair
        const server_tokens extended(llama_tokens(3000, 7), false);
        assert(cache.capture(extended, nullptr, nullptr, 0, control, true));
        successor_key = extended.cache_prefix_hash(extended.size());
        auto node = cache.by_key.at(obsolete_key);
        assert(node->superseded_by == successor_key && node->superseded_at_ms);
        check_descriptors(cache);
        prune_at = node->superseded_at_ms + 1800000;
        assert(cache.prune_superseded({}, prune_at - 1) == 0);
        // Pin the old branch through a real lease, even after supersession.
        cache.mark_control(*node, control); // returning to edit point renews it
        assert(!node->superseded_at_ms);
        check_descriptors(cache);
        cache.note_superseded(extended, successor_key, server_cache_now_ms());
        control.branch = "old-edit"; cache.mark_control(*node, control);
        cache.note_superseded(extended, successor_key, server_cache_now_ms());
        assert(cache.prepare_lease("edit", "revision", "", 0, 3600000));
        assert(cache.observe_lease("edit", "revision", "old-edit", 2500, true));
        assert(cache.commit_lease("edit", "revision", "", 3600000));
        prune_at = node->superseded_at_ms + 1800000;
        assert(cache.prune_superseded({adjacent.cache_prefix_hash(2501)}, prune_at) == 0);
        assert(cache.release_lease("edit", "revision"));
        assert(cache.prune_superseded({obsolete_key, adjacent.cache_prefix_hash(2501)}, prune_at) == 0);
        // Retained background warmers are exempt, even with a longer sibling.
        control.branch = "warm-edit"; control.priority = server_cache_priority::background;
        cache.mark_control(*node, control);
        cache.note_superseded(extended, successor_key, server_cache_now_ms());
        prune_at = node->superseded_at_ms + 1800000;
        assert(cache.prune_superseded({adjacent.cache_prefix_hash(2501)}, prune_at) == 0);
        cache.release_branch("warm-edit");
        cache.persist();
    }
    {
        server_prompt_page_cache cache(limit, 2048);
        auto old = cache.by_key.at(obsolete_key);
        assert(old->superseded_by == successor_key && old->superseded_at_ms);
        // A lost successor cannot justify deletion of the surviving edit point.
        auto successor = cache.by_key.at(successor_key);
        cache.evict(successor);
        assert(cache.prune_superseded({}, prune_at) == 0);
        const server_tokens extended(llama_tokens(3000, 7), false);
        server_cache_control control; control.request_id = 202;
        assert(cache.capture(extended, nullptr, nullptr, 0, control, true));
        prune_at = std::max(old->superseded_at_ms, old->usage.last_access_ms) + 1800000;
        assert(cache.prune_superseded({}, prune_at) >= 1);
        assert(!cache.by_key.count(obsolete_key));
        assert(cache.match_tokens(extended) == 3000);
        assert(cache.by_key.count(extended.cache_prefix_hash(2048))); // shared ancestor
        assert(cache.stats().superseded_pruned >= 1);
        bool attributed = false;
        const auto telemetry = cache.replacement_stats();
        for (const auto & event : telemetry["recent_evictions"]) {
            attributed |= event["key"] == obsolete_key && event["reason"] == "superseded_partial";
        }
        assert(attributed);
        cache.persist();
    }
    {
        server_prompt_page_cache cache(limit, 2048);
        assert(!cache.replacement_stats()["recent_evictions"].empty());
        assert(!cache.by_key.count(obsolete_key));
    }
    // Media need not force a coarse rollback: complete atomic chunks can
    // participate in a bridged exact endpoint, with positions separate from
    // token counts. Only byte serialization is stubbed, never media identity.
    {
        server_page_ram_store store(limit); assert(store.write(ram_json::array()));
    }
    {
        mtmd::input_chunks chunks(mtmd_test_create_input_chunks());
        server_tokens media(chunks, true);
        for (int i = 0; i < 100; ++i) media.push_back(9);
        const auto boundaries = media.page_boundaries(6);
        assert(boundaries.size() > 2);
        auto prefix = media.clone(); prefix.keep_first(boundaries.front().token_count);
        server_prompt_page_cache cache(limit, 6);
        server_cache_control control;
        const server_tokens old_text(llama_tokens(7, 9), false);
        assert(!media.complete_boundary(7));
        assert(cache.capture(old_text, nullptr, nullptr, 0, control, true));
        const auto bridges_before = cache.stats().capture_bridges;
        assert(cache.capture(prefix, nullptr, nullptr, 0, control));
        assert(cache.capture(media, nullptr, nullptr, 0, control, true));
        assert(cache.stats().capture_bridges == bridges_before + 1);
        assert(captured_p0 == prefix.pos_next() && captured_p1 == media.pos_next());
        assert(cache.match_tokens(media) == media.size());
        assert(!cache.by_key.at(old_text.cache_prefix_hash(7))->superseded_at_ms);
        assert(cache.by_key.at(media.cache_prefix_hash(media.size()))->parent_key ==
                prefix.cache_prefix_hash(prefix.size()));
    }
#endif
    {
        server_page_ram_store store(limit);
        auto graph = ram_json::array(); graph.push_back(make_page(store, "root", "", 0, 2048, "test")); assert(store.write(graph));
        const auto file = graph[0]["target"]["file"].get<std::string>();
        std::fstream corrupt(root + "/" + identity + "/" + file, std::ios::binary | std::ios::in | std::ios::out);
        corrupt.put('x'); corrupt.close();
        server_page_bytes bytes; size_t budget = limit;
        bool failed = false; try { store.load(bytes, graph[0]["target"], budget); } catch (...) { failed = true; } assert(failed);
        graph[0]["target"]["file"] = "../escape";
        failed = false; server_page_bytes traversal; try { store.load(traversal, graph[0]["target"], budget); } catch (...) { failed = true; } assert(failed);
    }
    {
        const auto other = std::string(64, 'b'); setenv("LLAMA_SERVER_PAGE_CACHE_NAMESPACE", other.c_str(), 1);
        server_page_ram_store store(limit); assert(store.active()); assert(store.read().empty());
        setenv("LLAMA_SERVER_PAGE_CACHE_NAMESPACE", identity.c_str(), 1);
    }
    {
        server_page_ram_store store(limit);
        auto graph = ram_json::array({make_page(store, "protected", "", 0, 2048, "primary"),
            make_page(store, "cold", "protected", 2048, 4096, "orphan")});
        const auto now = uint64_t(std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
        const auto lease = ram_json{{"head", "response"}, {"expires", now + 60000}, {"generation", 0},
            {"pages", {{"protected", 4096}}}, {"pending", {{"revision", ""}, {"generation", 0}, {"expires", 0}, {"pages", ram_json::object()}}}};
        assert(store.write(graph, {{"response_leases", ram_json::array({lease})}}));
    }
    {
        const auto other = std::string(64, 'b'); setenv("LLAMA_SERVER_PAGE_CACHE_NAMESPACE", other.c_str(), 1);
        server_page_ram_store store(limit); assert(store.active());
        assert(std::filesystem::exists(root + "/" + identity + "/graph.json")); // dormant pin survives GC
        setenv("LLAMA_SERVER_PAGE_CACHE_NAMESPACE", identity.c_str(), 1);
    }
    {
        server_page_ram_store store(limit);
        auto graph = store.read(); assert(!store.metadata().empty());
        assert(graph.size() == 1 && graph[0]["key"] == "protected");
        assert(store.write(graph)); // expired/released metadata permits ordinary namespace GC again
    }
    {
        server_page_ram_store active(limit);
        const auto other = std::string(64, 'b'); setenv("LLAMA_SERVER_PAGE_CACHE_NAMESPACE", other.c_str(), 1);
        {
            server_page_ram_store newer(limit); assert(newer.active());
            assert(std::filesystem::exists(root + "/" + identity + "/lock"));
        }
        setenv("LLAMA_SERVER_PAGE_CACHE_NAMESPACE", identity.c_str(), 1);
    }
    {
        server_page_ram_store store(limit);
        assert(store.stale_namespaces == 1);
        assert(!std::filesystem::exists(root + "/" + std::string(64, 'b')));
        server_page_bytes oversized;
        bool failed = false; try { store.allocate(oversized, limit + 1); } catch (...) { failed = true; } assert(failed);
    }
    {
        server_page_ram_store store(limit);
        std::ofstream corrupt(root + "/" + identity + "/graph.json"); corrupt << "{partial"; corrupt.close();
        assert(store.read().empty() && store.rejected == 1); assert(store.write(ram_json::array()));
    }
    {
        server_page_ram_store pressure(4096);
        server_page_bytes first; pressure.allocate(first, 4096); first.seal();
        auto graph = ram_json::array({{{"target", first.metadata()}, {"draft", empty_bytes()}, {"reference", empty_bytes()}}});
        assert(pressure.write(graph));
        server_page_bytes blocked; bool failed = false;
        try { pressure.allocate(blocked, 4096); } catch (...) { failed = true; } assert(failed);
        // The cache evicts and commits before capture, so replacement can progress at the budget boundary.
        assert(pressure.write(ram_json::array()));
        server_page_bytes replacement; pressure.allocate(replacement, 4096); assert(replacement.size() == 4096);
    }
    setenv("LLAMA_SERVER_PAGE_CACHE_RAM_DIR", "/tmp/llama-page-ram-disk-rejection", 1);
    { server_page_ram_store store(limit); assert(!store.active()); }
    unsetenv("LLAMA_SERVER_PAGE_CACHE_RAM_DIR");
    { server_page_ram_store store(limit); assert(!store.active()); server_page_bytes bytes; store.allocate(bytes, 4); assert(bytes.size() == 4); }
    setenv("LLAMA_SERVER_PAGE_CACHE_RAM_DIR", root.c_str(), 1);
    const std::string disk_root = "/tmp/llama-page-ssd-test-" + std::to_string(getpid());
    setenv("LLAMA_SERVER_PAGE_CACHE_SSD_DIR", disk_root.c_str(), 1);
    setenv("LLAMA_SERVER_PAGE_CACHE_SSD_BYTES", "1048576", 1);
    setenv("LLAMA_SERVER_PAGE_CACHE_SSD_QUEUE_BYTES", "524288", 1);
    setenv("LLAMA_SERVER_PAGE_CACHE_SSD_RESERVE_BYTES", "0", 1);
    ram_json cold_descriptor;
    {
        server_page_ram_store store(limit); auto * disk = store.disk_store();
        assert(disk && disk->active());
        // Read-only telemetry never walks disk; admission still checks fresh capacity.
        const auto usage_scans = disk->stats()["usage_scans"].get<uint64_t>();
        for (int i = 0; i < 10; ++i) assert(disk->stats()["usage_sample_ms"].get<uint64_t>() > 0);
        assert(disk->stats()["usage_scans"] == usage_scans);
        assert(disk->can_fit(1));
        assert(disk->stats()["usage_scans"].get<uint64_t>() > usage_scans);
        disk->clean({});
        const auto cleanup_scans = disk->stats()["cleanup_scans"].get<uint64_t>();
        disk->clean({});
        assert(disk->stats()["cleanup_scans"] == cleanup_scans);
        assert(disk->stats()["cleanup_skips"].get<uint64_t>() > 0);
        server_page_bytes bytes; store.allocate(bytes, 131072);
        std::fill(bytes.data(), bytes.data() + bytes.size(), uint8_t(71)); bytes.seal();
        cold_descriptor = bytes.metadata();
        assert(disk->submit("page", store.ram_directory(), ram_json::array({cold_descriptor})));
        // A pending copy remains protected, even after its writer completes.
        disk->clean({});
        assert(disk->stats()["cleanup_scans"].get<uint64_t>() > cleanup_scans);
        std::vector<server_page_ssd_store::completion> completed;
        for (int i = 0; i < 5000 && completed.empty(); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1)); completed = disk->poll();
        }
        assert(completed.size() == 1 && completed[0].success && completed[0].bytes == bytes.size());
        assert(disk->used() == bytes.size());
        bytes.mark_backed(); bytes.demote(disk->path(cold_descriptor["file"]));
        assert(bytes.on_disk() && bytes.resident_size() == 0 && bytes.size() == 131072);
        assert(bytes.data()[0] == 71); bytes.release_disk_mapping();
        assert(bytes.metadata()["tier"] == "ssd");
        auto page = make_page(store, "disk-root", "", 0, 2048, "retained");
        page["target"] = bytes.metadata();
        auto publication = ram_json::array({page});
        assert(store.write(std::move(publication), {{"publication", "owned-snapshot"}}));
        publication = ram_json::array({{{"key", "caller-reused-input"}}});
        assert(store.read() == ram_json::array({page}));
        for (int i = 0; i < 5000 && !disk->read_graph().is_object(); ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        assert(disk->read_graph().at("pages") == ram_json::array({page}));
        assert(disk->read_graph().at("metadata").at("publication") == "owned-snapshot");
        // Cold graph recovery needs metadata memory, never the full RAM budget.
        std::filesystem::remove(root + "/" + identity + "/graph.json");
        auto recovered = store.read(); assert(recovered.size() == 1);
        size_t budget = 1; server_page_bytes lazy;
        store.load(lazy, recovered[0]["target"], budget);
        assert(lazy.on_disk() && budget == 1 && lazy.data()[0] == 71);
        lazy.release_disk_mapping();
        // Whole-page checksum validation happens before cold data is exposed.
        const auto file = disk->path(cold_descriptor["file"]);
        std::fstream corrupt(file, std::ios::in | std::ios::out | std::ios::binary);
        corrupt.put('x'); corrupt.close();
        bool rejected = false; try { lazy.data(); } catch (...) { rejected = true; }
        assert(rejected);
        assert(!disk->can_fit(1048577));
        assert(store.write(ram_json::array()));
        assert(disk->used() == 0); // Actual deletion removes the backing object.
        server_page_bytes orphan; store.allocate(orphan, 1024);
        std::fill(orphan.data(), orphan.data() + orphan.size(), uint8_t(9)); orphan.seal();
        assert(disk->submit("orphan", store.ram_directory(), ram_json::array({orphan.metadata()})));
        disk->clean({});
        completed.clear();
        for (int i = 0; i < 5000 && completed.empty(); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1)); completed = disk->poll();
        }
        assert(completed.size() == 1 && completed[0].success);
        assert(std::filesystem::exists(disk->path(orphan.metadata()["file"])));
        // Same live set must rescan when the in-flight protection changes.
        disk->clean({});
        assert(!std::filesystem::exists(disk->path(orphan.metadata()["file"])));
    }
    {
        server_page_ram_store store(limit);
        auto * disk = store.disk_store(); assert(disk && disk->active());
        server_page_bytes bytes; store.allocate(bytes, 131072);
        std::fill(bytes.data(), bytes.data() + bytes.size(), uint8_t(33)); bytes.seal();
        auto bad = bytes.metadata(); bad["checksum"] = bad["checksum"].get<uint64_t>() + 1;
        assert(disk->submit("bad", store.ram_directory(), ram_json::array({bad})));
        std::vector<server_page_ssd_store::completion> done;
        for (int i = 0; i < 5000 && done.empty(); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1)); done = disk->poll();
        }
        assert(done.size() == 1 && !done[0].success);
        assert(!std::filesystem::exists(disk->path(bad["file"])));
        assert(bytes.data()[0] == 33); // RAM remains available on failure.
        assert(disk->stats()["write_failures"].get<uint64_t>() > 0);
    }
#ifdef TEST_PAGE_GRAPH
    // Real cross-tier graph, durable parent-before-child backup, lazy recovery,
    // protected ownership and cold restore use synthetic bytes, never a GPU.
    { server_page_ram_store store(limit); assert(store.write(ram_json::array())); }
    {
        server_prompt_page_cache cache(limit, 2048);
        server_cache_control control; control.branch = "cold-context";
        control.priority = server_cache_priority::interactive;
        const server_tokens root_tokens(llama_tokens(2048, 7), false);
        const server_tokens old(llama_tokens(3000, 7), false);
        const server_tokens head(llama_tokens(5000, 7), false);
        assert(cache.capture(root_tokens, nullptr, nullptr, 0, control));
        assert(cache.capture(old, nullptr, nullptr, 0, control, true));
        assert(cache.capture(head, nullptr, nullptr, 0, control, true));
        for (int i = 0; i < 5000; ++i) {
            cache.maintain_ssd({});
            if (std::all_of(cache.pages.begin(), cache.pages.end(), [](const auto & p) { return p.backed(); })) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        assert(std::all_of(cache.pages.begin(), cache.pages.end(), [](const auto & p) { return p.backed(); }));
        check_descriptors(cache);
        assert(cache.prepare_lease("ssd-primary", "head", "", 0, 60000));
        assert(cache.observe_lease("ssd-primary", "head", "cold-context", 5000, true));
        assert(cache.commit_lease("ssd-primary", "head", "", 60000));
        const auto maintenance_runs = cache.stats().ssd_maintenance_runs;
        const auto commits = cache.ram_store.commits;
        server_cache_control adopted = control; adopted.branch = "resident-adopter";
        bool live_failed = false;
        assert(cache.restore(server_tokens(llama_tokens(5001, 7), false), 5000, false,
            nullptr, nullptr, 0, adopted, live_failed) == 0 && !live_failed);
        assert(cache.stats().ssd_maintenance_runs == maintenance_runs);
        assert(cache.stats().live_prefix_adoptions == 1 && cache.ram_store.commits > commits);
        assert(cache.stats().persist_attempts > 0);
        assert(cache.stats().persist_us == cache.stats().persist_build_us + cache.stats().persist_write_us);
        assert(cache.by_key.at(head.cache_prefix_hash(5000))->branches.count(adopted.branch));
        cache.maintenance({});
        assert(cache.stats().ssd_maintenance_runs > maintenance_runs);
        const auto before = cache.size();
        // A vanished SSD backup must invalidate the cached backed flag too.
        auto * backup_store = cache.ram_store.disk_store();
        std::vector<std::pair<std::string, std::string>> backups;
        for (const auto & node : cache.pages) {
            const std::string path = backup_store->path(node.data_tgt.metadata()["file"]);
            backups.emplace_back(path, path + ".test-missing");
            std::filesystem::rename(backups.back().first, backups.back().second);
        }
        assert(!cache.demote_ram());
        for (const auto & paths : backups) std::filesystem::rename(paths.second, paths.first);
        check_descriptors(cache);
        for (auto & node : cache.pages) { node.data_tgt.mark_backed(); node.descriptor_dirty = true; }
        while (cache.demote_ram()) {}
        check_descriptors(cache);
        assert(cache.size() < before && cache.target_size() == 0);
        assert(cache.match_tokens(head) == 5000 && cache.count() == 3);
        assert(cache.response_leases.protects(head.cache_prefix_hash(5000), server_cache_now_ms()));
        const server_tokens continuation(llama_tokens(5001, 7), false);
        bool failed = false;
        assert(cache.restore(continuation, 0, false, nullptr, nullptr, 0, control, failed) == 5000 && !failed);
        assert(cache.stats().ssd_restores == 1 && cache.stats().ssd_restored_bytes == 384);
        assert(cache.tier_stats()["demotions"].get<uint64_t>() == 3);
        cache.persist();
        auto * disk = cache.ram_store.disk_store();
        bool mirrored = false;
        for (int i = 0; i < 5000 && !mirrored; ++i) {
            const auto graph = disk->read_graph();
            mirrored = graph.is_object() && graph["pages"].size() == 3 &&
                std::all_of(graph["pages"].begin(), graph["pages"].end(), [](const auto & p) { return p["target"]["tier"] == "ssd"; });
            if (!mirrored) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        assert(mirrored);
        std::filesystem::remove(root + "/" + identity + "/graph.json");
    }
    {
        server_prompt_page_cache cache(limit, 2048);
        assert(cache.count() == 3 && cache.target_size() == 0);
        const server_tokens continuation(llama_tokens(5001, 7), false);
        assert(cache.match_tokens(continuation) == 5000);
        server_cache_control control; bool failed = false;
        assert(cache.restore(continuation, 0, false, nullptr, nullptr, 0, control, failed) == 5000 && !failed);
        assert(cache.response_leases.protects(continuation.cache_prefix_hash(5000), server_cache_now_ms()));
        // Corruption invalidates its descendants and ownership, not a different
        // namespace or request. No invalid raw state reaches the fake importer.
        const auto root_key = continuation.cache_prefix_hash(2048);
        const auto file = cache.ram_store.disk_store()->path(cache.by_key.at(root_key)->data_tgt.metadata()["file"]);
        std::fstream corrupt(file, std::ios::in | std::ios::out | std::ios::binary); corrupt.put('x'); corrupt.close();
        assert(cache.restore(continuation, 0, false, nullptr, nullptr, 0, control, failed) == 0 && failed);
        assert(cache.count() == 0 && !cache.response_leases.protects(root_key, server_cache_now_ms()));
    }
    setenv("LLAMA_SERVER_PAGE_CACHE_SSD_BYTES", "256", 1);
    {
        server_prompt_page_cache cache(limit, 2048);
        server_cache_control control; control.branch = "pinned";
        control.priority = server_cache_priority::interactive;
        const server_tokens root_tokens(llama_tokens(2048, 7), false);
        const server_tokens old(llama_tokens(3000, 7), false);
        assert(cache.capture(root_tokens, nullptr, nullptr, 0, control));
        assert(cache.capture(old, nullptr, nullptr, 0, control, true));
        for (int i = 0; i < 5000 && !cache.by_key.at(old.cache_prefix_hash(3000))->backed(); ++i) {
            cache.maintain_ssd({}); std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        assert(cache.by_key.at(old.cache_prefix_hash(3000))->backed());
        assert(cache.ram_store.disk_store()->used() == 256);
        assert(cache.prepare_lease("capacity", "head", "", 0, 60000));
        assert(cache.observe_lease("capacity", "head", "pinned", 3000, true));
        assert(cache.commit_lease("capacity", "head", "", 60000));
        const server_tokens head(llama_tokens(3500, 7), false);
        assert(cache.capture(head, nullptr, nullptr, 0, control, true));
        assert(cache.by_key.count(old.cache_prefix_hash(3000)) &&
            !cache.by_key.at(head.cache_prefix_hash(3500))->backed());
        assert(cache.release_lease("capacity", "head"));
        for (int i = 0; i < 5000 && !cache.by_key.at(head.cache_prefix_hash(3500))->backed(); ++i) {
            cache.maintain_ssd({head.cache_prefix_hash(3500)});
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        assert(cache.by_key.at(head.cache_prefix_hash(3500))->backed());
        assert(!cache.by_key.count(old.cache_prefix_hash(3000)));
        assert(cache.by_key.count(root_tokens.cache_prefix_hash(2048)) && cache.match_tokens(head) == 3500);
        assert(cache.ram_store.disk_store()->used() == 256);
        const auto telemetry = cache.replacement_stats(); bool attributed = false;
        for (const auto & event : telemetry["recent_evictions"]) attributed |= event["reason"] == "ssd_pressure";
        assert(attributed);
    }
    // A failed backing writer must not purge cold leaves to free RAM.
    // Simulate ENOTDIR only inside this isolated temporary namespace.
    { server_page_ram_store store(limit); assert(store.write(ram_json::array())); }
    setenv("LLAMA_SERVER_PAGE_CACHE_SSD_BYTES", "1048576", 1);
    const server_tokens cold_tokens(llama_tokens(2048, 7), false);
    const auto cold_key = cold_tokens.cache_prefix_hash(2048);
    {
        server_page_ram_store store(4096); server_page_bytes bytes;
        store.allocate(bytes, 128); std::fill(bytes.data(), bytes.data() + 128, uint8_t(31)); bytes.seal();
        auto * disk = store.disk_store();
        assert(disk->submit(cold_key, store.ram_directory(), ram_json::array({bytes.metadata()})));
        std::vector<server_page_ssd_store::completion> done;
        for (int i = 0; i < 5000 && done.empty(); ++i) {
            done = disk->poll(); std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        assert(done.size() == 1 && done[0].success);
        bytes.mark_backed(); bytes.demote(disk->path(bytes.metadata()["file"]));
        auto page = ram_json{{"key", cold_key}, {"parent", ""}, {"p0_tokens", 0}, {"p1_tokens", 2048},
            {"p0_pos", 0}, {"p1_pos", 2048}, {"coarse", true}, {"page_tokens", 2048},
            {"unowned_retained", false}, {"unowned_interactive", false}, {"branches", ram_json::array()},
            {"target", bytes.metadata()}, {"draft", empty_bytes()}, {"reference", empty_bytes()}};
        assert(store.write(ram_json::array({page})));
    }
    {
        server_prompt_page_cache cache(4096, 2048); assert(cache.by_key.count(cold_key));
        const auto space = std::filesystem::path(disk_root) / identity;
        const auto saved = std::filesystem::path(disk_root) / "isolated-saved-namespace";
        std::filesystem::rename(space, saved);
        { std::ofstream unavailable(space); unavailable << "temporary fixture"; }
        server_cache_control control; control.priority = server_cache_priority::interactive;
        for (int token = 20; token < 45; ++token) {
            const server_tokens fresh(llama_tokens(2048, token), false);
            assert(cache.capture(fresh, nullptr, nullptr, 0, control));
            for (int i = 0; i < 1000 && !cache.ram_store.disk_store()->degraded(); ++i)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            assert(cache.ram_store.disk_store()->degraded());
            cache.maintain_ssd({});
        }
        assert(cache.by_key.count(cold_key));
        std::filesystem::remove(space); std::filesystem::rename(saved, space);
        const server_tokens continuation(llama_tokens(2049, 7), false); bool failed = false;
        assert(cache.restore(continuation, 0, false, nullptr, nullptr, 0, control, failed) == 2048 && !failed);
    }
#endif
    // Retired-namespace pressure respects both storage owners and leases.
    setenv("LLAMA_SERVER_PAGE_CACHE_SSD_BYTES", "1048576", 1);
    { server_page_ram_store store(limit); assert(store.write(ram_json::array())); }
    setenv("LLAMA_SERVER_PAGE_CACHE_SSD_BYTES", "8192", 1);
    const auto retired = std::string(64, 'b');
    {
        server_page_ram_store store(limit);
        const auto namespace_path = std::filesystem::path(disk_root) / retired;
        std::filesystem::create_directory(namespace_path);
        const auto blob = namespace_path / "page-1-1";
        { std::ofstream output(blob, std::ios::binary); output << std::string(4096, 'x'); }
        const auto now = uint64_t(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
        auto graph = ram_json{{"version", 1}, {"namespace", retired},
            {"pages", ram_json::array({{{"last_access_ms", now - 3600000}}})},
            {"metadata", {{"response_leases", ram_json::array({{{"expires", now + 60000}}})}}}};
        const auto manifest = namespace_path / "graph.json";
        { std::ofstream output(manifest); output << graph.dump(); }
        auto * disk = store.disk_store();
        const auto wanted = 8192 - disk->used() + 4096;
        assert(!disk->can_fit(wanted));
        assert(!disk->reclaim_dormant(wanted) && std::filesystem::exists(blob));
        graph["metadata"]["response_leases"] = ram_json::array();
        { std::ofstream output(manifest); output << graph.dump(); }
        {
            server_page_ram_lock owner((namespace_path / "lock").string(), true);
            assert(owner.owned());
            assert(!disk->reclaim_dormant(wanted) && std::filesystem::exists(blob));
        }
        assert(disk->reclaim_dormant(wanted) && !std::filesystem::exists(blob));
        assert(disk->stats()["namespace_evictions"] == 1);
    }
    unsetenv("LLAMA_SERVER_PAGE_CACHE_SSD_DIR");
    std::filesystem::remove_all(disk_root);
    std::filesystem::remove_all(root);
    std::filesystem::remove("/tmp/llama-page-ram-disk-rejection");
    std::cout << "RAM page crash, reopen, lock, namespace, corruption, bounds and GC tests passed\n";
}
