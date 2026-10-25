#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

class llama_moe_route_stats {
public:
    struct layer_stats {
        static constexpr size_t SMALL_BATCH_BUCKETS = 7;

        uint64_t routed_tokens = 0;
        uint64_t invalid_ids   = 0;
        std::vector<uint64_t> expert_hits;
        std::vector<uint64_t> q1_hits;
        std::vector<uint64_t> multi_hits;
        std::vector<double> weight_sums;
        uint64_t q1_rows = 0;
        uint64_t multi_rows = 0;
        uint64_t invalid_weights = 0;
        std::vector<uint64_t> coactivations;
        // q2--q6 raw aggregates: calls / distinct experts / repeated hits /
        // repeated hits retaining their most-common rank slot.
        std::array<uint64_t, SMALL_BATCH_BUCKETS> small_batch_calls {};
        std::array<uint64_t, SMALL_BATCH_BUCKETS> small_batch_distinct_experts {};
        std::array<uint64_t, SMALL_BATCH_BUCKETS> small_batch_repeated_hits {};
        std::array<uint64_t, SMALL_BATCH_BUCKETS> small_batch_rank_aligned_hits {};
    };

    bool configure(uint32_t n_layers, uint32_t n_experts);
    bool add(int32_t layer, const int32_t * ids, size_t n_used, size_t n_tokens, const float * weights = nullptr);
    std::string snapshot_json(const char * context) const;

    void note_dropped(uint64_t route_rows);

    bool enabled() const { return n_experts_ != 0; }
    uint32_t n_experts() const { return n_experts_; }
    uint64_t routed_tokens() const { return routed_tokens_; }
    uint64_t dropped_route_rows() const { return dropped_route_rows_; }
    const std::map<int32_t, layer_stats> & layers() const { return layers_; }

    static size_t pair_index(uint32_t a, uint32_t b, uint32_t n_experts);

private:
    static void add_saturating(uint64_t & value, uint64_t increment = 1);

    uint32_t n_layers_  = 0;
    uint32_t n_experts_ = 0;
    uint64_t routed_tokens_ = 0;
    uint64_t dropped_route_rows_ = 0;
    std::map<int32_t, layer_stats> layers_;
};
