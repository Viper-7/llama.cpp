#include "llama-moe-route-stats.h"

#include <algorithm>
#include <limits>

namespace {

constexpr uint32_t ROUTE_STATS_MAX_LAYERS  = 64;
constexpr uint32_t ROUTE_STATS_MAX_EXPERTS = 512;
constexpr size_t   ROUTE_STATS_MAX_USED    = 32;
constexpr size_t   ROUTE_STATS_MAX_BYTES   = 32u * 1024u * 1024u;

}

bool llama_moe_route_stats::configure(uint32_t n_layers, uint32_t n_experts) {
    layers_.clear();
    routed_tokens_ = 0;
    dropped_route_rows_ = 0;
    n_layers_ = 0;
    n_experts_ = 0;

    if (n_layers == 0 || n_layers > ROUTE_STATS_MAX_LAYERS ||
            n_experts == 0 || n_experts > ROUTE_STATS_MAX_EXPERTS) {
        return false;
    }

    const size_t n_pairs = (size_t) n_experts * (n_experts - 1) / 2;
    const size_t counters_per_layer = (size_t) n_experts + n_pairs;
    if (counters_per_layer > ROUTE_STATS_MAX_BYTES / sizeof(uint64_t) / n_layers) {
        return false;
    }

    n_layers_ = n_layers;
    n_experts_ = n_experts;
    return true;
}

void llama_moe_route_stats::add_saturating(uint64_t & value, uint64_t increment) {
    value = increment > std::numeric_limits<uint64_t>::max() - value
        ? std::numeric_limits<uint64_t>::max()
        : value + increment;
}

size_t llama_moe_route_stats::pair_index(uint32_t a, uint32_t b, uint32_t n_experts) {
    if (a > b) {
        std::swap(a, b);
    }
    return (size_t) a * (2u * n_experts - a - 1u) / 2u + (b - a - 1u);
}

bool llama_moe_route_stats::add(
        int32_t layer, const int32_t * ids, size_t n_used, size_t n_tokens) {
    if (!enabled() || layer < 0 || (uint32_t) layer >= n_layers_ || ids == nullptr ||
            n_used == 0 || n_used > ROUTE_STATS_MAX_USED) {
        return false;
    }

    auto & stats = layers_[layer];
    if (stats.expert_hits.empty()) {
        stats.expert_hits.resize(n_experts_);
        stats.coactivations.resize((size_t) n_experts_ * (n_experts_ - 1) / 2);
    }

    std::vector<uint32_t> selected;
    selected.reserve(n_used);
    std::vector<uint8_t> batch_hits;
    std::vector<uint8_t> batch_rank_hits;
    if (n_tokens >= 2 && n_tokens < layer_stats::SMALL_BATCH_BUCKETS) {
        batch_hits.resize(n_experts_);
        batch_rank_hits.resize((size_t) n_experts_ * n_used);
    }
    for (size_t row = 0; row < n_tokens; ++row) {
        selected.clear();
        for (size_t rank = 0; rank < n_used; ++rank) {
            const int32_t id = ids[row * n_used + rank];
            if (id < 0 || (uint32_t) id >= n_experts_) {
                add_saturating(stats.invalid_ids);
                continue;
            }
            if (!batch_hits.empty()) {
                ++batch_hits[id];
                ++batch_rank_hits[(size_t) id * n_used + rank];
            }
            if (std::find(selected.begin(), selected.end(), (uint32_t) id) == selected.end()) {
                selected.push_back((uint32_t) id);
                add_saturating(stats.expert_hits[id]);
            }
        }

        std::sort(selected.begin(), selected.end());
        for (size_t i = 0; i < selected.size(); ++i) {
            for (size_t j = i + 1; j < selected.size(); ++j) {
                add_saturating(stats.coactivations[pair_index(selected[i], selected[j], n_experts_)]);
            }
        }
        add_saturating(stats.routed_tokens);
        add_saturating(routed_tokens_);
    }
    if (!batch_hits.empty()) {
        uint64_t distinct = 0;
        uint64_t repeated = 0;
        uint64_t aligned  = 0;
        for (uint32_t expert = 0; expert < n_experts_; ++expert) {
            const uint8_t hits = batch_hits[expert];
            if (hits == 0) {
                continue;
            }
            ++distinct;
            repeated += hits - 1;

            uint8_t max_rank_hits = 0;
            for (size_t rank = 0; rank < n_used; ++rank) {
                max_rank_hits = std::max(max_rank_hits, batch_rank_hits[(size_t) expert * n_used + rank]);
            }
            aligned += max_rank_hits - 1;
        }
        add_saturating(stats.small_batch_calls[n_tokens]);
        add_saturating(stats.small_batch_distinct_experts[n_tokens], distinct);
        add_saturating(stats.small_batch_repeated_hits[n_tokens], repeated);
        add_saturating(stats.small_batch_rank_aligned_hits[n_tokens], aligned);
    }
    return true;
}

void llama_moe_route_stats::note_dropped(uint64_t route_rows) {
    add_saturating(dropped_route_rows_, route_rows);
}
