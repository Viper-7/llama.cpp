#include "llama-kv-cache-kpool.h"

#include "llama-batch.h"
#include "llama-impl.h"
#include "llama-kv-cache.h"
#include "llama-kv-cells.h"
#include "llama-memory-hybrid.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

uint32_t llama_kpool_n_pools(uint32_t n_kv, uint32_t kpool, uint32_t n_seqs) {
    GGML_ASSERT(kpool > 0);
    GGML_ASSERT(n_seqs > 0);

    // Shared physical cells can belong to a full logical pool run in every sequence.
    const uint64_t count = (uint64_t(n_kv)/kpool + 2)*n_seqs;
    GGML_ASSERT(count <= std::numeric_limits<uint32_t>::max());
    return uint32_t(count);
}

uint32_t llama_kpool_select_k(uint32_t n_pools, uint32_t indexer_top_k, uint32_t kpool) {
    GGML_ASSERT(kpool > 0);
    GGML_ASSERT(n_pools > 0);
    GGML_ASSERT(indexer_top_k % kpool == 0 && "indexer_top_k must be a whole number of pools");

    return std::min(n_pools, indexer_top_k/kpool);
}

// sel_mask and cand_mask hold only 0.0f and -INFINITY, so f16 is exact here
template <typename T> struct kpool_mask_of;

template <> struct kpool_mask_of<float> {
    static float from(float v) { return v; }
};

template <> struct kpool_mask_of<ggml_fp16_t> {
    static ggml_fp16_t from(float v) { return ggml_fp32_to_fp16(v); }
};

template <typename T>
static void kpool_mask_fill(T * dst, int64_t n) {
    std::fill(dst, dst + n, kpool_mask_of<T>::from(-INFINITY));
}

template <typename T>
static void kpool_mask_row(
                T * cur_sel,
                T * cur_cand,
        const llama_pos * pos_at,
        const int32_t   * pool_of,
          int64_t   n_kv,
        llama_pos   q,
        llama_pos   tail_start,
          int64_t   bo_vis) {
    const T v_sel  = kpool_mask_of<T>::from(0.0f);
    const T v_mask = kpool_mask_of<T>::from(-INFINITY);

    for (int64_t j = 0; j < n_kv; ++j) {
        const bool vis    = (uint32_t) pos_at [j] <= (uint32_t) q;
        const bool pooled = (uint32_t) pool_of[j] <  (uint32_t) bo_vis;
        const bool tail   = pos_at[j] >= tail_start;

        cur_sel [j] = vis && tail   ? v_sel : v_mask;
        // the candidate set, which the top-k budget may overrun but must never escape
        cur_cand[j] = vis && (pooled || tail) ? v_sel : v_mask;
    }
}

void llama_kv_cache_set_input_kpool(
        const llama_kv_cache * kv,
              ggml_tensor    * cell_pool,
              ggml_tensor    * pool_cells,
              ggml_tensor    * bias,
              ggml_tensor    * pool_bias,
              ggml_tensor    * sel_mask,
              ggml_tensor    * cand_mask,
              ggml_tensor    * tail_pad_cells,
              ggml_tensor    * tail_pad_valid,
              ggml_tensor    * segment_rows,
              ggml_tensor    * pool_reps,
              ggml_tensor    * new_pool_cells,
              ggml_tensor    * new_pool_reps,
        const uint32_t       * strm_of,
              int64_t          kv_size,
              bool             rebuild,
        const llama_ubatch   * ubatch,
              uint32_t         kpool) {
    GGML_ASSERT(kv != nullptr);
    GGML_ASSERT(kpool > 0);

    GGML_ASSERT(ggml_backend_buffer_is_host(pool_cells->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(pool_bias ->buffer));
    GGML_ASSERT(pool_cells->type == GGML_TYPE_I32);
    GGML_ASSERT(pool_bias ->type == GGML_TYPE_F32);
    GGML_ASSERT((sel_mask->type == GGML_TYPE_F16 || sel_mask->type == GGML_TYPE_F32) &&
            "sel_mask must be f16 or f32");
    GGML_ASSERT(cand_mask->type == sel_mask->type && "both masks must have the KQ mask's type");

    GGML_ASSERT(ggml_is_contiguous(pool_cells));
    GGML_ASSERT(ggml_is_contiguous(pool_bias));
    GGML_ASSERT(ggml_is_contiguous(sel_mask));
    GGML_ASSERT(ggml_is_contiguous(cand_mask));
    GGML_ASSERT(tail_pad_cells && tail_pad_cells->type == GGML_TYPE_I32 && ggml_is_contiguous(tail_pad_cells));
    GGML_ASSERT(tail_pad_valid && tail_pad_valid->type == GGML_TYPE_F32 && ggml_is_contiguous(tail_pad_valid));
    GGML_ASSERT(segment_rows   && segment_rows  ->type == GGML_TYPE_I32 && ggml_is_contiguous(segment_rows));

    const int64_t n_kv     = sel_mask->ne[0];
    const int64_t n_ns     = sel_mask->ne[3];
    const int64_t r        = kpool;
    const int64_t n_tokens = ubatch->n_tokens;

    // [TAG_KPOOL_SEQ_PARTITION] positions are unambiguous only within one sequence, so
    // one pool map per SEQUENCE, not per stream
    GGML_ASSERT(n_ns == 1 || (int64_t) ubatch->n_seqs_unq == n_ns);

    const int64_t n_ps    = (int64_t) ubatch->n_seqs_unq/n_ns;
    const int64_t n_pools = pool_cells->ne[0]/r;

    GGML_ASSERT(n_ps > 0 && (int64_t) ubatch->n_seqs_unq == n_ns*n_ps);
    GGML_ASSERT(pool_cells->ne[0] % r == 0);
    GGML_ASSERT(n_pools >= 2*n_ps);
    GGML_ASSERT(pool_cells->ne[1] == n_ns);
    GGML_ASSERT(sel_mask->ne[2] == 1);
    GGML_ASSERT(ggml_are_same_shape(cand_mask, sel_mask));
    GGML_ASSERT(pool_bias->ne[0] == n_pools && pool_bias->ne[2] == n_ns);
    GGML_ASSERT(n_tokens % n_ns == 0);

    const int64_t n_tps  = n_tokens/n_ns;
    const int64_t n_padq = sel_mask->ne[1];
    const int64_t n_suffix = tail_pad_cells->ne[0];
    const int64_t n_padded = segment_rows->ne[0];

    GGML_ASSERT(pool_bias->ne[1] == n_tps);
    GGML_ASSERT(n_padq >= n_tps);
    GGML_ASSERT(tail_pad_cells->ne[1] == n_tps && tail_pad_cells->ne[2] == n_ns);
    GGML_ASSERT(ggml_are_same_shape(tail_pad_cells, tail_pad_valid));
    GGML_ASSERT(segment_rows->ne[1] == n_tps && segment_rows->ne[2] == n_ns);
    GGML_ASSERT(n_suffix >= (int64_t) kpool - 1 && n_padded > n_suffix);

    // Graph allocation prunes these inputs when a short history takes the dense fallback.
    // Segmented FA uses the compact tail but prunes the legacy block-diagonal row map.
    const bool has_compact_inputs = tail_pad_cells->data != nullptr;
    const bool has_segment_rows   = segment_rows->data != nullptr;
    const bool has_full_masks     = sel_mask->data != nullptr;
    GGML_ASSERT((tail_pad_valid->data != nullptr) == has_compact_inputs);
    GGML_ASSERT((cand_mask     ->data != nullptr) == has_full_masks);
    GGML_ASSERT(n_padded <= std::numeric_limits<int32_t>::max()/n_tps);
    if (has_compact_inputs) {
        GGML_ASSERT(ggml_backend_buffer_is_host(tail_pad_cells->buffer));
        GGML_ASSERT(ggml_backend_buffer_is_host(tail_pad_valid->buffer));
    }
    if (has_segment_rows) {
        GGML_ASSERT(ggml_backend_buffer_is_host(segment_rows  ->buffer));
    }
    if (has_full_masks) {
        GGML_ASSERT(ggml_backend_buffer_is_host(sel_mask ->buffer));
        GGML_ASSERT(ggml_backend_buffer_is_host(cand_mask->buffer));
    }

    if (cell_pool) {
        GGML_ASSERT(ggml_backend_buffer_is_host(cell_pool->buffer));
        GGML_ASSERT(cell_pool->type == GGML_TYPE_I32);
        GGML_ASSERT(ggml_is_contiguous(cell_pool));
        GGML_ASSERT(cell_pool->ne[0] == n_kv && cell_pool->ne[1] == n_ns);

        // one row per stream, so a shared cell has nowhere to put its second pool
        GGML_ASSERT(n_ps == 1 && "the per-cell pool view needs one sequence per stream");
    }

    if (bias) {
        GGML_ASSERT(ggml_backend_buffer_is_host(bias->buffer));
        GGML_ASSERT(bias->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(bias));
        GGML_ASSERT(bias->ne[0] == n_kv && bias->ne[1] == n_tps && bias->ne[2] == n_ns);
    }

    // pooled-key cache inputs travel together or not at all
    const bool kcache = pool_reps != nullptr;

    GGML_ASSERT((new_pool_cells != nullptr) == kcache);
    GGML_ASSERT((new_pool_reps  != nullptr) == kcache);

    int64_t n_new_max = 0;

    if (kcache) {
        GGML_ASSERT(ggml_backend_buffer_is_host(pool_reps     ->buffer));
        GGML_ASSERT(ggml_backend_buffer_is_host(new_pool_cells->buffer));
        GGML_ASSERT(ggml_backend_buffer_is_host(new_pool_reps ->buffer));

        GGML_ASSERT(pool_reps     ->type == GGML_TYPE_I32);
        GGML_ASSERT(new_pool_cells->type == GGML_TYPE_I32);
        GGML_ASSERT(new_pool_reps ->type == GGML_TYPE_I64);

        GGML_ASSERT(ggml_is_contiguous(pool_reps));
        GGML_ASSERT(ggml_is_contiguous(new_pool_cells));
        GGML_ASSERT(ggml_is_contiguous(new_pool_reps));

        GGML_ASSERT(pool_reps->ne[0] == n_pools && pool_reps->ne[1] == n_ns);
        GGML_ASSERT(new_pool_cells->ne[0] % r == 0 && new_pool_cells->ne[1] == n_ns);

        n_new_max = new_pool_cells->ne[0]/r;

        GGML_ASSERT(new_pool_reps->ne[0] == n_new_max*n_ns);
        GGML_ASSERT(strm_of != nullptr && kv_size > 0);
    }

    int32_t * dst_pool_reps = kcache ? (int32_t *) pool_reps     ->data : nullptr;
    int32_t * dst_new_cells = kcache ? (int32_t *) new_pool_cells->data : nullptr;
    int64_t * dst_new_reps  = kcache ? (int64_t *) new_pool_reps ->data : nullptr;

    int32_t * dst_cell_pool  = cell_pool ? (int32_t *) cell_pool->data : nullptr;
    int32_t * dst_pool_cells = (int32_t *) pool_cells->data;
    float   * dst_bias       = bias ? (float *) bias->data : nullptr;
    float   * dst_pool_bias  = (float   *) pool_bias ->data;
    char    * dst_sel_mask   = has_full_masks ? (char *) sel_mask ->data : nullptr;
    char    * dst_cand_mask  = has_full_masks ? (char *) cand_mask->data : nullptr;
    int32_t * dst_tail_cells = has_compact_inputs ? (int32_t *) tail_pad_cells->data : nullptr;
    float   * dst_tail_valid = has_compact_inputs ? (float   *) tail_pad_valid->data : nullptr;
    int32_t * dst_seg_rows   = has_segment_rows   ? (int32_t *) segment_rows  ->data : nullptr;

    const bool   mask_f16 = sel_mask->type == GGML_TYPE_F16;
    const size_t mask_ts  = ggml_type_size(sel_mask->type);

    // -1 marks a cell with no usable pool; host side only, never copied into cell_pool
    std::vector<int32_t>   pool_of(n_kv);
    std::vector<int32_t>   filled(n_pools);
    std::vector<llama_pos> pos_at;

    std::vector<int64_t> run_off(n_ps);
    std::vector<int64_t> run_len(n_ps);

    auto seq_of = [&](int64_t s, int64_t ps) {
        return n_ps == 1 ? ubatch->seq_id[s*n_tps][0] : ubatch->seq_id_unq[ps];
    };

    for (int64_t s = 0; s < n_ns; ++s) {
        int32_t * cur_pool_cells = dst_pool_cells + s*(r*n_pools);
        char    * cur_sel_mask   = has_full_masks ? dst_sel_mask  + s*(n_padq*n_kv)*mask_ts : nullptr;
        char    * cur_cand_mask  = has_full_masks ? dst_cand_mask + s*(n_padq*n_kv)*mask_ts : nullptr;
        float   * cur_pool_bias  = dst_pool_bias  + s*(n_tps*n_pools);
        int32_t * cur_tail_cells = has_compact_inputs ? dst_tail_cells + s*(n_tps*n_suffix) : nullptr;
        float   * cur_tail_valid = has_compact_inputs ? dst_tail_valid + s*(n_tps*n_suffix) : nullptr;
        int32_t * cur_seg_rows   = has_segment_rows   ? dst_seg_rows   + s*(n_tps*n_padded) : nullptr;

        // Padding must reference this batch's own state. Row zero may belong
        // to another execution lane, and an old complete pool may be shared.
        const auto fallback_seq = ubatch->seq_id[s*n_tps][0];
        const auto & fallback_cells = kv->get_cells(fallback_seq);
        int32_t fallback = -1;
        for (int64_t j = 0; j < n_kv; ++j) {
            if (!fallback_cells.is_empty(j) && fallback_cells.seq_has(j, fallback_seq) &&
                    fallback_cells.pos_get(j) == ubatch->pos[s*n_tps]) {
                fallback = (int32_t) j;
                break;
            }
        }
        GGML_ASSERT(fallback >= 0);

        std::fill(cur_pool_cells, cur_pool_cells + r*n_pools, fallback);
        std::fill(cur_pool_bias,  cur_pool_bias  + n_tps*n_pools, -INFINITY);
        if (has_compact_inputs) {
            std::fill(cur_tail_cells, cur_tail_cells + n_tps*n_suffix, fallback);
            std::fill(cur_tail_valid, cur_tail_valid + n_tps*n_suffix, -INFINITY);
        }
        if (has_segment_rows) {
            for (int64_t q = 0; q < n_tps; ++q) {
                for (int64_t p = 0; p < n_padded; ++p) {
                    cur_seg_rows[q*n_padded + p] = (int32_t) (q*n_padded + p);
                }
            }
        }

        int32_t * cur_pool_reps = kcache ? dst_pool_reps + s*n_pools        : nullptr;
        int32_t * cur_new_cells = kcache ? dst_new_cells + s*(r*n_new_max)  : nullptr;
        int64_t * cur_new_reps  = kcache ? dst_new_reps  + s*n_new_max      : nullptr;

        // count of real entries emitted for this stream; the rest is padding
        int64_t n_new = 0;

        if (kcache) {
            std::fill(cur_pool_reps, cur_pool_reps + n_pools, fallback);
            std::fill(cur_new_cells, cur_new_cells + r*n_new_max, fallback);
        }

        // the token loop writes rows < n_tps in full; only the padding rows need clearing
        if (has_full_masks) {
            if (mask_f16) {
                kpool_mask_fill((ggml_fp16_t *) (cur_sel_mask  + n_tps*n_kv*mask_ts), (n_padq - n_tps)*n_kv);
                kpool_mask_fill((ggml_fp16_t *) (cur_cand_mask + n_tps*n_kv*mask_ts), (n_padq - n_tps)*n_kv);
            } else {
                kpool_mask_fill((float *) (cur_sel_mask  + n_tps*n_kv*mask_ts), (n_padq - n_tps)*n_kv);
                kpool_mask_fill((float *) (cur_cand_mask + n_tps*n_kv*mask_ts), (n_padq - n_tps)*n_kv);
            }
        }

        // [TAG_KPOOL_PACK] one packed run per sequence, sized on the pool range it holds.
        // NOT one full-width table per sequence: the indexer scores every slot against
        // every query, so that multiplies the score tensor by n_seq_max.
        // llama_memory_seq_cp can ask for more slots than exist; then a sequence keeps its
        // newest pools, the same cut a large hole already forces.
        {
            int64_t n_want = 0;

            for (int64_t ps = 0; ps < n_ps; ++ps) {
                const llama_seq_id seq = seq_of(s, ps);
                const auto & cells = kv->get_cells(seq);

                int64_t b_min = 0;
                int64_t b_max = 0;
                bool    found = false;

                for (int64_t j = 0; j < n_kv; ++j) {
                    if (cells.is_empty(j) || !cells.seq_has(j, seq)) {
                        continue;
                    }
                    const int64_t b = cells.pos_get(j)/r;
                    b_min = found ? std::min(b_min, b) : b;
                    b_max = found ? std::max(b_max, b) : b;
                    found = true;
                }

                run_len[ps] = found ? b_max - b_min + 1 : 0;
                n_want += run_len[ps];
            }

            if (n_want > n_pools) {
                int64_t rem = n_pools;

                for (int64_t ps = 0; ps < n_ps; ++ps) {
                    run_len[ps] = std::min(run_len[ps], rem/(n_ps - ps));
                    rem -= run_len[ps];
                }
            }

            int64_t off = 0;
            for (int64_t ps = 0; ps < n_ps; ++ps) {
                run_off[ps] = off;
                off += run_len[ps];
            }

            GGML_ASSERT(off <= n_pools);
        }

        int64_t n_done = 0;

        for (int64_t ps = 0; ps < n_ps; ++ps) {
            const llama_seq_id seq_of_pool = seq_of(s, ps);
            const auto & cells = kv->get_cells(seq_of_pool);

            const int64_t n_run = run_len[ps];

            int32_t * cur_cell_pool   = dst_cell_pool ? dst_cell_pool + s*n_kv : nullptr;
            int32_t * part_pool_cells = cur_pool_cells + run_off[ps]*r;

            std::fill(pool_of.begin(), pool_of.end(), -1);
            std::fill(filled.begin(),  filled.end(),   0);

            pos_at.resize(n_kv);
            for (int64_t j = 0; j < n_kv; ++j) {
                pos_at[j] = cells.is_empty(j) || !cells.seq_has(j, seq_of_pool) ? -1 : cells.pos_get(j);
            }

            // anchoring at the absolute p/kpool follows vLLM and SGLang, not HF
            // (valid_keys.argmax(-1)): it is the only anchor that keeps a pool's identity
            // stable from the prefill that built it to the decodes that read it.
            int64_t b_base = 0;
            {
                int64_t b_min = 0;
                int64_t b_max = 0;
                bool    found = false;

                for (int64_t j = 0; j < n_kv; ++j) {
                    if (pos_at[j] < 0) {
                        continue;
                    }
                    const int64_t b = pos_at[j]/r;
                    b_min = found ? std::min(b_min, b) : b;
                    b_max = found ? std::max(b_max, b) : b;
                    found = true;
                }

                b_base = std::max(b_min, b_max - (n_run - 1));
            }

            for (int64_t j = 0; j < n_kv; ++j) {
                if (pos_at[j] < 0) {
                    continue;
                }

                const llama_pos p  = pos_at[j];
                const int64_t   bo = p/r - b_base;

                if (bo < 0 || bo >= n_run) {
                    continue;
                }

                pool_of[j] = (int32_t) bo;
                part_pool_cells[bo*r + (p%r)] = (int32_t) j;
                filled[bo]++;
            }

            // pool_valid = grouped_valid_keys.all(-1): the compressor consumes all r keys
            for (int64_t j = 0; j < n_kv; ++j) {
                // != rather than <: two cells claiming one position overwrite each other
                if (pool_of[j] >= 0 && filled[pool_of[j]] != (int32_t) r) {
                    pool_of[j] = -1;
                }
                if (cur_cell_pool) {
                    cur_cell_pool[j] = pool_of[j] < 0 ? 0 : pool_of[j];
                }
            }

            if (kcache) {
                // the pooled key of a complete pool lives in the row of its LAST member,
                // the cell holding pos % r == r-1. that slot is only meaningful once the
                // pool is complete: for a partial pool it is still 0 from the fill above,
                // and cell 0 is a real cell whose own pooled key we must not overwrite.
                for (int64_t p = 0; p < n_run; ++p) {
                    if (filled[p] == (int32_t) r) {
                        cur_pool_reps[run_off[ps] + p] = part_pool_cells[p*r + (r - 1)];

                    }
                }

                // recompute exactly the complete pools this ubatch wrote into. touched[] is
                // over the run, so the cost is O(tokens), not O(n_kv).
                std::vector<uint8_t> touched(n_run, 0);

                for (int64_t ii = 0; ii < n_tps; ++ii) {
                    const int64_t i = s*n_tps + ii;

                    if (ubatch->seq_id[i][0] != seq_of_pool) {
                        continue;
                    }

                    const int64_t bo = ubatch->pos[i]/r - b_base;

                    if (bo >= 0 && bo < n_run) {
                        touched[bo] = 1;
                    }
                }

                for (int64_t p = 0; p < n_run; ++p) {
                    // in rebuild mode every cached pooled key is stale, so re-emit all of
                    // them, not just the pools this ubatch closed
                    if ((!touched[p] && !rebuild) || filled[p] != (int32_t) r) {
                        continue;
                    }

                    // n_new_max = n_tps/kpool + n_ps bounds this while a sequence's tokens in
                    // one ubatch are a contiguous position run, which llama-batch.cpp
                    // enforces. dropping a completed pool would silently serve a stale key,
                    // so fail loudly instead of clamping.
                    GGML_ASSERT(n_new < n_new_max && "k-pool: more pools completed than the fixed bound");

                    std::copy(part_pool_cells + p*r, part_pool_cells + (p + 1)*r,
                            cur_new_cells + n_new*r);

                    cur_new_reps[n_new] = (int64_t) strm_of[s]*kv_size + part_pool_cells[p*r + (r - 1)];

                    n_new++;
                }
            }

            for (int64_t ii = 0; ii < n_tps; ++ii) {
                const int64_t   i = s*n_tps + ii;

                if (ubatch->seq_id[i][0] != seq_of_pool) {
                    continue;
                }

                const llama_pos q = ubatch->pos[i];

                // q >= 0 is what makes the unsigned range test below a range test
                GGML_ASSERT(q >= 0);

                n_done++;

                // index_kpool_always_select_tail, which lands selection on pool boundaries
                const llama_pos tail_start = (q + 1)/r*r;

                if (has_compact_inputs) {
                    int32_t * q_tail_cells = cur_tail_cells + ii*n_suffix;
                    float   * q_tail_valid = cur_tail_valid + ii*n_suffix;
                    int64_t n_tail = 0;
                    for (llama_pos p = tail_start; p <= q; ++p) {
                        for (int64_t j = 0; j < n_kv; ++j) {
                            if (pos_at[j] == p) {
                                GGML_ASSERT(n_tail < n_suffix);
                                q_tail_cells[n_tail] = (int32_t) j;
                                q_tail_valid[n_tail] = 0.0f;
                                n_tail++;
                            }
                        }
                    }
                    GGML_ASSERT(n_tail <= r - 1);
                }

                // the reference tests visibility at a pool's LAST member, so a pool the
                // query straddles is dropped whole
                const int64_t bo_vis = std::max<int64_t>(0, tail_start/r - b_base);

                float * cur_bias = dst_bias ? dst_bias + i*n_kv : nullptr;
                if (has_full_masks) {
                    char * cur_sel  = cur_sel_mask  + ii*n_kv*mask_ts;
                    char * cur_cand = cur_cand_mask + ii*n_kv*mask_ts;

                    if (mask_f16) {
                        kpool_mask_row((ggml_fp16_t *) cur_sel, (ggml_fp16_t *) cur_cand,
                                pos_at.data(), pool_of.data(), n_kv, q, tail_start, bo_vis);
                    } else {
                        kpool_mask_row((float *) cur_sel, (float *) cur_cand,
                                pos_at.data(), pool_of.data(), n_kv, q, tail_start, bo_vis);
                    }
                }

                if (cur_bias) {
                    for (int64_t j = 0; j < n_kv; ++j) {
                        const bool vis    = (uint32_t) pos_at [j] <= (uint32_t) q;
                        const bool pooled = (uint32_t) pool_of[j] <  (uint32_t) bo_vis;

                        cur_bias[j] = vis && pooled ? 0.0f : -INFINITY;
                    }
                }

                // the query's own sequence run only; every other slot keeps the -INFINITY
                // of the fill above, which is what keeps a foreign pool out of the budget
                float * q_pool_bias = cur_pool_bias + ii*n_pools + run_off[ps];

                for (int64_t p = 0; p < n_run; ++p) {
                    const bool valid   = filled[p] == (int32_t) r;
                    const bool visible = p < bo_vis;

                    q_pool_bias[p] = valid && visible ? 0.0f : -INFINITY;
                }
            }
        }

        // exactly one partition per row, or a query reads another sequence's pools
        GGML_ASSERT(n_done == n_tps && "every query must belong to a sequence of the ubatch");

        if (kcache) {
            // Repeat a pool actually written by this batch, never an immutable
            // shared-prefix pool. With no completed pool, the current token's
            // unused pooled head is private scratch and cannot affect selection.
            for (int64_t p = n_new; p < n_new_max; ++p) {
                if (n_new > 0) {
                    std::copy(cur_new_cells, cur_new_cells + r, cur_new_cells + p*r);
                    cur_new_reps[p] = cur_new_reps[0];
                } else {
                    cur_new_reps[p] = (int64_t) strm_of[s]*kv_size + fallback;
                }
            }
        }
    }
}

void llm_graph_input_kpool::set_input(const llama_ubatch * ubatch) {
    // unconditional: the key and gate STORE runs on the dense path too. gating it the
    // way the scoring is gated would leave every cell below n_select with no indexer
    // state, and the first ubatch to cross n_select would pool cells never written
    mctx_idx->set_input_k_idxs(k_idxs, ubatch);

    if (pool_cells == nullptr) {
        return;
    }

    // the pooled key is written into the INDEXER cache, whose slot layout the attention
    // cache defines, so the stream map and cell count come from the indexer side
    std::vector<uint32_t> strm_of;

    if (pool_reps) {
        strm_of.resize(mctx_idx->get_n_stream());

        for (uint32_t s = 0; s < strm_of.size(); ++s) {
            strm_of[s] = mctx_idx->get_strm(s);
        }
    }

    llama_kv_cache_set_input_kpool(
            mctx_attn->get_kv(),
            /* cell_pool */ nullptr, pool_cells, /* bias */ nullptr, pool_bias,
            sel_mask, cand_mask,
            tail_pad_cells, tail_pad_valid, segment_rows,
            pool_reps, new_pool_cells, new_pool_reps,
            strm_of.empty() ? nullptr : strm_of.data(),
            pool_reps ? (int64_t) mctx_idx->get_kv()->get_size() : 0,
            rebuild,
            ubatch, kpool);

    // every pool has just been re-emitted, so the cache is consistent again. cleared here
    // rather than in build_inp_kpool because a graph that is built but not evaluated must
    // not clear it.
    if (rebuild) {
        mctx_attn->get_kv()->clear_kpool_dirty();
    }
}

bool llm_graph_input_kpool::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_memory_hybrid_context *>(params.mctx);
    const auto * attn = mctx->get_attn();
    const auto * idx  = mctx->get_idx();

    if (idx == nullptr || params.hparams.indexer_kpool != kpool) {
        return false;
    }

    const uint32_t n_select = params.hparams.indexer_top_k + kpool - 1;
    const bool scoring = params.cparams.n_ctx > n_select;

    bool res = true;

    res &= k_idxs != nullptr && k_idxs->ne[0] == params.ubatch.n_tokens;
    res &= (pool_cells != nullptr) == scoring;

    if (scoring) {
        const int64_t n_stream = params.cparams.kv_unified ? 1 : params.ubatch.n_seqs_unq;
        const int64_t n_tps    = params.ubatch.n_tokens/n_stream;
        const int64_t n_ps     = params.ubatch.n_seqs_unq/n_stream;
        const int64_t n_kv     = attn->get_n_kv();
        const int64_t span_cur = params.cparams.independent_execution ? attn->get_n_kv_logical() : n_kv;
        const int64_t n_pools  = llama_kpool_n_pools(span_cur, kpool, n_ps);
        res &= attention_span == span_cur;
        const int64_t n_select = kpool*llama_kpool_select_k(n_pools, params.hparams.indexer_top_k, kpool);
        const int64_t n_padded = GGML_PAD(n_select + kpool - 1, 256);
        const int64_t n_suffix = n_padded - n_select;

        const bool    rebuild_cur = attn->get_kv()->get_kpool_dirty();
        const int64_t n_new_cur   = rebuild_cur ? n_pools : n_tps/kpool + n_ps;

        res &= n_stream > 0 && n_tps > 0 && n_ps > 0;
        res &= idx->get_n_kv() == attn->get_n_kv();
        res &= idx->get_n_stream() == n_stream;
        res &= rebuild == rebuild_cur && n_new_max == n_new_cur;

        res &= pool_cells && pool_cells->ne[0] == (int64_t) kpool*n_pools && pool_cells->ne[1] == n_stream;
        res &= pool_bias  && pool_bias ->ne[0] == n_pools && pool_bias->ne[1] == n_tps && pool_bias->ne[2] == n_stream;
        res &= pool_reps  && pool_reps ->ne[0] == n_pools && pool_reps->ne[1] == n_stream;

        res &= new_pool_cells && new_pool_cells->ne[0] == (int64_t) kpool*n_new_cur && new_pool_cells->ne[1] == n_stream;
        res &= new_pool_reps  && new_pool_reps ->ne[0] == n_new_cur*n_stream;

        res &= sel_mask  && sel_mask ->ne[0] == n_kv && sel_mask ->ne[1] == n_tps && sel_mask ->ne[3] == n_stream;
        res &= cand_mask && cand_mask->ne[0] == n_kv && cand_mask->ne[1] == n_tps && cand_mask->ne[3] == n_stream;
        res &= tail_pad_cells && tail_pad_cells->ne[0] == n_suffix && tail_pad_cells->ne[1] == n_tps && tail_pad_cells->ne[2] == n_stream;
        res &= tail_pad_valid && tail_pad_valid->ne[0] == n_suffix && tail_pad_valid->ne[1] == n_tps && tail_pad_valid->ne[2] == n_stream;
        res &= segment_rows   && segment_rows  ->ne[0] == n_padded && segment_rows  ->ne[1] == n_tps && segment_rows  ->ne[2] == n_stream;

        res &= (pool_bias_f16 != nullptr) == params.cparams.fused_lid;
        if (pool_bias_f16) {
            res &= pool_bias_f16->ne[0] == n_pools && pool_bias_f16->ne[1] == n_tps && pool_bias_f16->ne[3] == n_stream;
        }
    }

    if (res) {
        mctx_attn = attn;
        mctx_idx  = idx;
    }

    if (debug > 1) {
        LLAMA_LOG_DEBUG("%s: can reuse k-pool input = %d\n", __func__, res);
    }

    return res;
}
