#include "tools/server/server-responses-control.h"
#include <cassert>
#include <thread>

int main() {
    server_responses_registry registry;
    const auto key = std::string("job_") + std::string(32, 'a') + ":0";
    {
        server_responses_preparing_guard preparing(registry, key);
        assert(!registry.released(key));
        assert(registry.attach(key, 123));
        auto targets = registry.cancel(key);
        assert(targets.size() == 1 && targets[0] == 123);
        registry.finish(123);
        assert(!registry.released(key)); // input preparation still owns work
        assert(!registry.attach(key, 124)); // cancellation cannot race late admission
    }
    assert(registry.released(key));
    const auto before = std::string("job_") + std::string(32, 'b') + ":0";
    assert(registry.cancel(before).empty() && registry.released(before));
    bool refused = false;
    try { registry.begin(before); } catch (const std::invalid_argument &) { refused = true; }
    assert(refused);
    const auto active = std::string("job_") + std::string(32, 'c') + ":0";
    registry.begin(active); assert(registry.attach(active, 125));
    assert(registry.targets(active).empty()); // input preparation is not suspendable
    registry.prepared(active);
    assert(registry.targets(active) == std::vector<int>{125});
    std::thread cancel([&] { registry.cancel(active); }); cancel.join();
    assert(registry.owns(125) && !registry.released(active));
    assert(registry.targets(active).empty()); // caller cancel wins over suspend
    registry.finish(125); assert(registry.released(active));
    assert(registry.targets(active).empty());
    const auto atomic = std::string("job_") + std::string(32, 'd') + ":0";
    registry.begin(atomic);
    bool submitted = false;
    assert(registry.post(atomic, {126}, [&] { submitted = true; }));
    assert(submitted && !registry.released(atomic));
    auto targets = registry.cancel(atomic);
    assert(targets.size() == 1 && targets[0] == 126);
    registry.finish(126); assert(registry.released(atomic));
    const auto failed = std::string("job_") + std::string(32, 'e') + ":0";
    {
        server_responses_preparing_guard preparing(registry, failed);
        try { registry.post(failed, {127}, [] { throw std::runtime_error("fixture failure"); }); }
        catch (const std::runtime_error &) {}
        assert(!registry.owns(127));
    }
    assert(registry.released(failed));
    assert(!server_responses_registry::valid_id("job_short:0"));
}
