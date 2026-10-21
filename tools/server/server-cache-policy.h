#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>

// Epoch time survives process replacement. Missing legacy history stays cold.
static inline uint64_t server_cache_now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
}

struct server_cache_usage {
    uint64_t last_access_ms = 0;
    uint32_t retrievals = 0;
    // Incidental interactive, retained interactive, retained named background,
    // incidental background, ephemeral interactive, ephemeral background.
    std::array<uint64_t, 6> hint_ms{};
    int last_request = -1; // process-local, deliberately not persisted

    void touch(size_t hint, uint64_t now, int request_id, bool reused) {
        if (reused && (request_id < 0 || request_id != last_request)) {
            retrievals = std::min<uint32_t>(15, retrievals + 1);
        }
        last_request = request_id;
        last_access_ms = now;
        hint_ms.at(hint) = now;
    }

    double weight(size_t hint, uint64_t now) const {
        static constexpr double multiplier[] = {4, 4, 2, 1, 2, 0.5};
        static constexpr double half_life_ms[] = {
            30*60*1000., 60*60*1000., 8*60*60*1000., 60*60*1000., 15*60*1000., 15*60*1000.,
        };
        const uint64_t stamp = hint_ms.at(hint);
        if (!stamp) return 0;
        const double age = now > stamp ? double(now - stamp) : 0;
        return multiplier[hint] * std::exp2(-age / half_life_ms[hint]) *
               (1 + std::log2(1 + std::min<uint32_t>(retrievals, 15)));
    }
};

// Only graph-safe leaves reach this comparator. Priority is a decaying weight,
// never an absolute class that can monopolize RAM indefinitely.
struct server_cache_eviction_rank {
    double weight = 0;
    size_t recompute_tokens = 0;
    size_t resident_bytes = 0;
    uint64_t last_access_ms = 0;

    bool less_than(const server_cache_eviction_rank & other) const {
        const long double left = static_cast<long double>(weight) * recompute_tokens * (other.resident_bytes + 1);
        const long double right = static_cast<long double>(other.weight) * other.recompute_tokens * (resident_bytes + 1);
        if (left != right) return left < right;
        return last_access_ms < other.last_access_ms;
    }
};
