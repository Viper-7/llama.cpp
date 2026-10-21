#include "tools/server/server-cache-policy.h"
#include <cassert>
#include <iostream>

int main() {
    constexpr uint64_t hour = 3600000, now = 100 * hour;
    const auto rank = [&](const server_cache_usage & usage, size_t hint, size_t tokens = 2048, size_t bytes = 1000) {
        return server_cache_eviction_rank{usage.weight(hint, now), tokens, bytes, usage.last_access_ms};
    };
    server_cache_usage stale, warmer, active;
    stale.touch(0, now - 2 * hour, 1, false);
    warmer.touch(2, now - 2 * hour, 2, false);
    active.touch(0, now, 3, true);
    assert(rank(stale, 0).less_than(rank(warmer, 2))); // old interactive yields
    assert(rank(warmer, 2).less_than(rank(active, 0))); // active tool-loop protected
    auto incidental = warmer;
    incidental.hint_ms[3] = incidental.hint_ms[2];
    assert(rank(incidental, 3).less_than(rank(warmer, 2)));
    server_cache_usage periodic;
    periodic.touch(2, now - 8 * hour, 4, false);
    assert(rank(stale, 0).less_than(rank(periodic, 2)));
    server_cache_usage ephemeral;
    ephemeral.touch(4, now - hour, 5, false);
    assert(rank(ephemeral, 4).less_than(rank(periodic, 2)));
    // Creation does not inflate reuse; repeated checkpoint touches in one task
    // count one request, not one hit per batch or graph edge traversal.
    server_cache_usage frequent;
    frequent.touch(2, now, 6, false); assert(frequent.retrievals == 0);
    frequent.touch(2, now, 6, true); frequent.touch(2, now, 6, true);
    assert(frequent.retrievals == 0);
    frequent.touch(2, now, 7, true); frequent.touch(2, now, 7, true);
    assert(frequent.retrievals == 1);
    for (int i = 7; i < 10000; ++i) frequent.touch(2, now - 24 * hour, i, true);
    assert(frequent.retrievals == 15);
    assert(rank(frequent, 2).less_than(rank(active, 0))); // frequency cannot immortalize
    server_cache_usage legacy;
    assert(rank(legacy, 0).weight == 0); // no invented fresh activity
    // Background retrieval must not resurrect an old interactive claim.
    stale.touch(2, now, 20, true);
    assert(stale.hint_ms[0] == now - 2 * hour);
    assert(rank(stale, 0).less_than(rank(stale, 2)));
    assert(rank(active, 0, 2048, 2000).less_than(rank(active, 0, 2048, 1000)));
    assert(!rank(active, 0).less_than(rank(active, 0)));
    server_cache_eviction_rank cold_a{0, 2048, 1000, 1}, cold_b{0, 2048, 1000, 2};
    assert(cold_a.less_than(cold_b));
    std::cout << "Age, frequency, lifecycle, density and request-dedup policy tests passed\n";
}
