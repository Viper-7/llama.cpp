#include "llama-moe-route-stats.h"

#include <set>

#define CHECK(condition) do { if (!(condition)) { return __LINE__; } } while (0)

int main() {
    llama_moe_route_stats stats;
    CHECK(stats.configure(2, 4));

    const int32_t ids[] = {
        0, 1, 1,
        2, 3, -1,
    };
    CHECK(stats.add(1, ids, 3, 2));

    const auto & layer = stats.layers().at(1);
    CHECK(layer.routed_tokens == 2);
    CHECK(layer.invalid_ids == 1);
    CHECK(layer.expert_hits[0] == 1);
    CHECK(layer.expert_hits[1] == 1);
    CHECK(layer.expert_hits[2] == 1);
    CHECK(layer.expert_hits[3] == 1);
    CHECK(layer.coactivations[llama_moe_route_stats::pair_index(0, 1, 4)] == 1);
    CHECK(layer.coactivations[llama_moe_route_stats::pair_index(2, 3, 4)] == 1);

    const int32_t reuse_ids[] = {
        0, 1, 2,
        0, 2, 3,
    };
    CHECK(stats.add(0, reuse_ids, 3, 2));
    const auto & reuse = stats.layers().at(0);
    CHECK(reuse.small_batch_calls[2] == 1);
    CHECK(reuse.small_batch_distinct_experts[2] == 4);
    CHECK(reuse.small_batch_repeated_hits[2] == 2);
    CHECK(reuse.small_batch_rank_aligned_hits[2] == 1);

    std::set<size_t> pair_indices;
    for (uint32_t a = 0; a < 8; ++a) {
        for (uint32_t b = a + 1; b < 8; ++b) {
            pair_indices.insert(llama_moe_route_stats::pair_index(a, b, 8));
            CHECK(llama_moe_route_stats::pair_index(a, b, 8) ==
                  llama_moe_route_stats::pair_index(b, a, 8));
        }
    }
    CHECK(pair_indices.size() == 28);
    CHECK(*pair_indices.begin() == 0);
    CHECK(*pair_indices.rbegin() == 27);

    CHECK(!stats.add(0, ids, 33, 1));
    stats.note_dropped(7);
    CHECK(stats.dropped_route_rows() == 7);

    CHECK(stats.configure(46, 288));
    CHECK(!stats.configure(64, 512));
    return 0;
}
