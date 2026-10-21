#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "log.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "ggml-cpp.h"
#include "llama.h"
#include "llama-cpp.h"

// TODO: replace with #include "llama-ext.h" in the future
#include "../src/llama-arch.h"
#include "../src/llama-ext.h"
#include "../src/llama-model-saver.h"

#include <cinttypes>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

static void common_batch_add(llama_batch & batch, llama_token id, llama_pos pos, const std::vector<llama_seq_id> & seq_ids, bool logits) {
    const int32_t i = batch.n_tokens++;
    batch.token[i] = id;
    batch.pos[i] = pos;
    batch.n_seq_id[i] = seq_ids.size();
    std::copy(seq_ids.begin(), seq_ids.end(), batch.seq_id[i]);
    batch.logits[i] = logits;
}

// normalized mean squared error = mse(a, b) / mse(a, 0)
static double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    GGML_ASSERT(a.size() == b.size());
    double mse_a_b = 0.0;
    double mse_a_0 = 0.0;

    for (size_t i = 0; i < a.size(); i++) {
        float a_i = a[i];
        float b_i = b[i];

        mse_a_b += (a_i - b_i) * (a_i - b_i);
        mse_a_0 += a_i * a_i;
    }

    return mse_a_b / mse_a_0;
}

static void set_tensor_data(struct ggml_tensor * tensor, void * userdata) {
    size_t seed = *(const size_t *) userdata;
    std::hash<std::string> hasher;
    seed ^= hasher(tensor->name);
    std::mt19937 gen(seed);
    std::normal_distribution<float> dis(0.0f, 1.0e-2f);

    const int64_t ne = ggml_nelements(tensor);
    if (tensor->type == GGML_TYPE_F32) {
        std::vector<float> tmp(ne);
        for (int64_t i = 0; i < ne; i++) {
            tmp[i] = dis(gen);
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(ne);
        for (int64_t i = 0; i < ne; i++) {
            tmp[i] = ggml_fp32_to_fp16(dis(gen));
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else {
        GGML_ABORT("fatal error");
    }
}

class scoped_env_var {
public:
    scoped_env_var(const char * name, const char * value) : name(name) {
        const char * current = std::getenv(name);
        if (current != nullptr) {
            previous = current;
            had_previous = true;
        }
#ifdef _WIN32
        GGML_ASSERT(_putenv_s(name, value) == 0);
#else
        GGML_ASSERT(setenv(name, value, 1) == 0);
#endif
    }

    ~scoped_env_var() {
#ifdef _WIN32
        _putenv_s(name.c_str(), had_previous ? previous.c_str() : "");
#else
        if (had_previous) {
            setenv(name.c_str(), previous.c_str(), 1);
        } else {
            unsetenv(name.c_str());
        }
#endif
    }

private:
    std::string name;
    std::string previous;
    bool had_previous = false;
};

static void usage(char ** argv) {
    printf("Usage: %s [-a/--arch arch] [-s/--seed seed] [-o/--out dir] [-v/--verbose] [-h/--help]\n", argv[0]);
}

static std::vector<llama_token> get_tokens(const uint32_t n_tokens, const uint32_t n_vocab, const size_t seed){
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> dis(0, n_vocab - 1);
    std::vector<llama_token> ret;
    ret.reserve(n_tokens);
    for (uint32_t i = 0; i < n_tokens; i++) {
        ret.push_back(dis(gen));
    }
    return ret;
}

static gguf_context_ptr get_gguf_ctx(const llm_arch arch, const bool moe) {
    gguf_context_ptr ret(gguf_init_empty());
    llama_model_saver ms(arch, ret.get());
    const uint32_t n_ctx = 256;

    uint32_t n_vocab = 128;
    uint32_t n_embd  = 256;
    uint32_t n_head  = 2;
    uint32_t n_ff    = 384;
    uint32_t n_layer = 2;
    if (arch == LLM_ARCH_LLAMA4) {
        n_layer = 4; // hparams.n_no_rope_layer_step is hard-coded to 4
    } else if (arch == LLM_ARCH_GEMMA4) {
        n_embd = 128;
        n_head = 2;
        n_ff   = 192;
        n_layer = 5; // need at least 5 for swa_pattern (every 5th is full_attention)
    } else if (arch == LLM_ARCH_GEMMA3N) {
        n_embd = 64;
        n_head = 1;
        n_ff   = 96;
        n_layer = 22; // hparams.n_layer_kv_from_start = 20 is hardcoded
    } else if (arch == LLM_ARCH_DEEPSEEK4) {
        // head size 64 so that GPU flash attention kernels support the model
        n_embd  = 512;
        n_head  = 8;
        n_ff    = 1024;
        n_layer = 4;
    } else if (arch == LLM_ARCH_STEP35 || arch == LLM_ARCH_LAGUNA) {
        n_embd = 160; // exercise per-head tensor split granularity with head size 80
    } else if (arch == LLM_ARCH_QWEN3 || arch == LLM_ARCH_MUSE_GLIMMER || arch == LLM_ARCH_AFMOE) {
        n_head = 4;
    } else if (arch == LLM_ARCH_DEEPSEEK2
            || arch == LLM_ARCH_DEEPSEEK32
            || arch == LLM_ARCH_GLM_DSA
            || arch == LLM_ARCH_DOTS3NOTE
            || arch == LLM_ARCH_KIMI_LINEAR
            || arch == LLM_ARCH_BAILINGMOE3
            || arch == LLM_ARCH_KIMI_K3
            || arch == LLM_ARCH_MISTRAL4
            || arch == LLM_ARCH_GLM5NEXT) {
        n_embd = 128;
        n_head = 1;
        n_ff   = 192;
        if (arch == LLM_ARCH_GLM5NEXT) {
            // Two trunk blocks plus one DSA NextN block, matching the real
            // model's convention that block_count includes its MTP block.
            n_layer = 3;
        }
    } else if (arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE) {
        n_layer = 3;
    } else if (arch == LLM_ARCH_CHAMELEON) {
        n_vocab = 10240;
    } else if (arch == LLM_ARCH_QWEN3TTS) {
        n_vocab = 4096; // must be >= the hard-coded codec head size (3072)
    }

    uint32_t n_head_kv = n_head;
    if (arch == LLM_ARCH_QWEN3) {
        n_head_kv = 1; // MQA coverage
    } else if (arch == LLM_ARCH_MUSE_GLIMMER || arch == LLM_ARCH_AFMOE) {
        n_head_kv = 2; // GQA coverage
    }
    const uint32_t n_embd_head = n_embd / n_head;

    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,      llm_arch_name(arch));
    ms.add_kv(LLM_KV_VOCAB_SIZE,                n_vocab);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,            n_ctx);
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,          n_embd);
    ms.add_kv(LLM_KV_FEATURES_LENGTH,           n_embd);
    ms.add_kv(LLM_KV_BLOCK_COUNT,               n_layer);
    ms.add_kv(LLM_KV_LEADING_DENSE_BLOCK_COUNT, uint32_t(1));
    if (arch == LLM_ARCH_GLM5NEXT) {
        ms.add_kv(LLM_KV_NEXTN_PREDICT_LAYERS, uint32_t(1));
    }

    if (arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE) {
        std::vector<uint32_t> n_ff_per_layer;
        n_ff_per_layer.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            n_ff_per_layer.push_back(il <= 1 ? 0 : n_ff);
        }
        ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH, n_ff_per_layer);
    } else {
        ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH, n_ff);
    }

    ms.add_kv(LLM_KV_USE_PARALLEL_RESIDUAL,   false);
    ms.add_kv(LLM_KV_LOGIT_SCALE,             1.0f);
    ms.add_kv(LLM_KV_TIME_MIX_EXTRA_DIM,      uint32_t(64));
    ms.add_kv(LLM_KV_TIME_DECAY_EXTRA_DIM,    uint32_t(128));
    ms.add_kv(LLM_KV_FULL_ATTENTION_INTERVAL, uint32_t(2));

    if (arch == LLM_ARCH_PLAMO2 || arch == LLM_ARCH_JAMBA || arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE ||
            arch == LLM_ARCH_GRANITE_HYBRID || arch == LLM_ARCH_LFM2 || arch == LLM_ARCH_LFM2MOE || arch == LLM_ARCH_KIMI_LINEAR ||
            arch == LLM_ARCH_BAILINGMOE3 || arch == LLM_ARCH_KIMI_K3) {
        GGML_ASSERT(n_layer >= 2);
        std::vector<uint32_t> n_head_per_layer;
        n_head_per_layer.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            n_head_per_layer.push_back(il == 1 ? 0 : n_head);
        }
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head_per_layer);
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, n_head_per_layer);
    } else if (arch == LLM_ARCH_GLM5NEXT) {
        // head_count doubles as the KDA head count, so it stays uniform; the kv array is what
        // marks the recurrent layers, and the loader asserts it holds both a zero and a nonzero
        GGML_ASSERT(n_layer >= 2);
        std::vector<uint32_t> n_head_kv_per_layer;
        n_head_kv_per_layer.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            n_head_kv_per_layer.push_back(il == 1 ? 0 : n_head_kv);
        }
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT,    n_head);
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, n_head_kv_per_layer);
    } else {
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head);
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, arch == LLM_ARCH_DEEPSEEK4 ? uint32_t(1) : n_head_kv);
    }

    ms.add_kv(LLM_KV_ATTENTION_MAX_ALIBI_BIAS, 8.0f);
    if (arch == LLM_ARCH_DEEPSEEK4) {
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH,   n_embd_head);
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH, n_embd_head);
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,   n_embd_head/2);
    } else if (arch == LLM_ARCH_DEEPSEEK2
            || arch == LLM_ARCH_DEEPSEEK32
            || arch == LLM_ARCH_GLM_DSA
            || arch == LLM_ARCH_DOTS3NOTE
            || arch == LLM_ARCH_KIMI_LINEAR
            || arch == LLM_ARCH_BAILINGMOE3
            || arch == LLM_ARCH_KIMI_K3
            || arch == LLM_ARCH_MISTRAL4) {
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH,       uint32_t(576));
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH,     uint32_t(512));
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,       uint32_t(64));
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_MLA,   uint32_t(192));
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_MLA, uint32_t(128));
        if (arch == LLM_ARCH_DOTS3NOTE) {
            // SWA layers reuse the same MLA geometry as the full layers in this fixture
            ms.add_kv(LLM_KV_ATTENTION_KV_LORA_RANK_SWA,     uint32_t(512));
            ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_SWA,       uint32_t(576));
            ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_SWA,     uint32_t(512));
            ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_MLA_SWA,   uint32_t(192));
            ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_MLA_SWA, uint32_t(128));
            ms.add_kv(LLM_KV_ROPE_FREQ_BASE_SWA,             10000.0f);
            // indexer on the full-attention layers (inverse of the swa pattern)
            std::vector<uint32_t> indexer_types;
            indexer_types.reserve(n_layer);
            for (uint32_t il = 0; il < n_layer; il++) {
                indexer_types.push_back(il % 2 ? 0 : 1);
            }
            ms.add_kv(LLM_KV_ATTENTION_INDEXER_TYPES, indexer_types);
        }
    } else if (arch == LLM_ARCH_GLM5NEXT) {
        // nope-only MLA: the cache holds the bare latent, so no rope width is added on top of
        // the kv LoRA rank and n_rot has to be an explicit 0, not the head size default
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH,       uint32_t(512));
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH,     uint32_t(512));
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,       uint32_t(0));
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_MLA,   uint32_t(192));
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_MLA, uint32_t(128));
    } else if (arch == LLM_ARCH_MINIMAX_M3) {
        // partial rotary: n_rot must not exceed the indexer key length (64)
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,       uint32_t(64));
    }
    ms.add_kv(LLM_KV_ATTENTION_CLAMP_KQV,              1.0f);
    // glm5next warns on anything but the 1e-6 its indexer k_norm hardcodes
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_EPS,          arch == LLM_ARCH_GLM5NEXT ? 1e-6f : 1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,      1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_EPS,          1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_GROUPS,       uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_Q_LORA_RANK,            arch == LLM_ARCH_DEEPSEEK4 ? uint32_t(64) : uint32_t(512));
    ms.add_kv(LLM_KV_ATTENTION_KV_LORA_RANK,           uint32_t(512));
    ms.add_kv(LLM_KV_ATTENTION_RELATIVE_BUCKETS_COUNT, uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW,         n_ctx/8);

    if (arch == LLM_ARCH_GEMMA4) {
        ms.add_kv(LLM_KV_EMBEDDING_LENGTH_PER_LAYER,      n_embd/2);
        ms.add_kv(LLM_KV_ATTENTION_SHARED_KV_LAYERS,      uint32_t(0));
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_SWA,        n_embd_head);
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_SWA,      n_embd_head);
        ms.add_kv(LLM_KV_ROPE_FREQ_BASE_SWA,              10000.0f);
        // SWA pattern: every 5th layer is full attention (matches E2B layer_types)
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, uint32_t(5));
    } else if (arch == LLM_ARCH_COHERE2MOE || arch == LLM_ARCH_MIMO2 || arch == LLM_ARCH_STEP35 ||
            arch == LLM_ARCH_MUSE_GLIMMER || arch == LLM_ARCH_GRANITE_SWA || arch == LLM_ARCH_DOTS3NOTE) {
        std::vector<uint32_t> pattern;
        pattern.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            pattern.push_back(il % 2);
        }
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, pattern);
    } else {
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, uint32_t(2));
    }

    // MSA requires one indexer head per GQA (KV) head, unlike the DSA archs where the
    // indexer head count is independent of the main attention head count.
    if (arch == LLM_ARCH_QWEN4EXP) {
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,    uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_LOW_RANK, uint32_t(8));
        // without this the QSA layers fall back to dense and go uncovered
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_RATIOS, std::vector<uint32_t>(n_layer, 4));
    }

    // minimax-m3 keeps one indexer head per GQA head; the rest use a fixed 64 to match the fused
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT,   arch == LLM_ARCH_MINIMAX_M3 ? n_head : uint32_t(64));
    // qwen4exp ropes indexer keys with the main rotary width, so its head can't be < n_rot
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH,
              arch == LLM_ARCH_QWEN4EXP ? n_embd_head : uint32_t(128));

    ms.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K,        uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_BLOCK_SIZE,   uint32_t(4));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_LOCAL_BLOCKS, uint32_t(1));
    ms.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS, std::vector<uint32_t>({n_embd_head/4, n_embd_head/4, n_embd_head/4, n_embd_head/4}));

    if (arch == LLM_ARCH_DEEPSEEK4) {
        ms.add_kv(LLM_KV_ATTENTION_OUTPUT_GROUP_COUNT,         uint32_t(8));
        ms.add_kv(LLM_KV_ATTENTION_OUTPUT_LORA_RANK,           uint32_t(32));
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_RATIOS,            std::vector<uint32_t>({0, 0, 4, 128}));
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_ROPE_FREQ_BASE,    160000.0f);
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,               uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, uint32_t(2));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_EPSILON,             1.0e-6f);
        ms.add_kv(LLM_KV_HASH_LAYER_COUNT,                      uint32_t(0));
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP,                      10.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,                  1.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM,                   true);
    } else if (arch == LLM_ARCH_GLM5NEXT) {
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,               uint32_t(4)); // build_hc_pre asserts exactly 4 streams
        ms.add_kv(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, uint32_t(2));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_EPSILON,             1.0e-6f);
        // the only arch that pools indexer keys; top_k must be a whole number of pools, and
        // the resulting selection width has to stay under n_ctx or the sparse path goes unused
        ms.add_kv(LLM_KV_ATTENTION_INDEXER_KPOOL,              uint32_t(4));
        // glm5next reads these unconditionally; the if (moe) block below never sets them
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,                 1.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM,                  true);
    }
    if (arch == LLM_ARCH_GLM5NEXT) {
        std::vector<std::string> token_list;
        std::vector<float> token_scores(n_vocab, 0.0f);
        std::vector<int32_t> token_types(n_vocab, LLAMA_TOKEN_TYPE_NORMAL);
        token_list.reserve(n_vocab);
        for (uint32_t i = 0; i < n_vocab; ++i) {
            token_list.push_back("token_" + std::to_string(i));
        }
        token_list[0] = "<unk>";
        token_list[1] = "<s>";
        token_list[2] = "</s>";
        token_list[3] = "\n";
        token_types[0] = LLAMA_TOKEN_TYPE_UNKNOWN;
        token_types[1] = LLAMA_TOKEN_TYPE_CONTROL;
        token_types[2] = LLAMA_TOKEN_TYPE_CONTROL;
        ms.add_kv(LLM_KV_TOKENIZER_MODEL,      "llama");
        ms.add_kv(LLM_KV_TOKENIZER_LIST,       token_list);
        ms.add_kv(LLM_KV_TOKENIZER_SCORES,     token_scores);
        gguf_set_arr_data(ms.gguf_ctx, ms.llm_kv(LLM_KV_TOKENIZER_TOKEN_TYPE).c_str(),
                GGUF_TYPE_INT32, token_types.data(), token_types.size());
        ms.add_kv(LLM_KV_TOKENIZER_BOS_ID,     uint32_t(1));
        ms.add_kv(LLM_KV_TOKENIZER_EOS_ID,     uint32_t(2));
        ms.add_kv(LLM_KV_TOKENIZER_EOT_ID,     uint32_t(2));
        ms.add_kv(LLM_KV_TOKENIZER_EOM_ID,     uint32_t(2));
        ms.add_kv(LLM_KV_TOKENIZER_UNK_ID,     uint32_t(0));
        ms.add_kv(LLM_KV_TOKENIZER_SEP_ID,     uint32_t(0));
        ms.add_kv(LLM_KV_TOKENIZER_PAD_ID,     uint32_t(0));
        ms.add_kv(LLM_KV_TOKENIZER_FIM_PRE_ID, uint32_t(0));
        ms.add_kv(LLM_KV_TOKENIZER_FIM_SUF_ID, uint32_t(0));
        ms.add_kv(LLM_KV_TOKENIZER_FIM_MID_ID, uint32_t(0));
        ms.add_kv(LLM_KV_TOKENIZER_FIM_PAD_ID, uint32_t(0));
        ms.add_kv(LLM_KV_TOKENIZER_FIM_REP_ID, uint32_t(0));
        ms.add_kv(LLM_KV_TOKENIZER_FIM_SEP_ID, uint32_t(0));
        ms.add_kv(LLM_KV_TOKENIZER_ADD_BOS,    false);
    } else {
        ms.add_kv(LLM_KV_TOKENIZER_MODEL, "no_vocab");
    }
    // ms.add_kv(LLM_KV_DENSE_2_FEAT_OUT,     n_embd);
    // ms.add_kv(LLM_KV_DENSE_3_FEAT_IN,      n_embd);

    if (moe) {
        ms.add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, n_ff);
        ms.add_kv(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, n_ff / 2);  // distinct from n_ff so a saver key-clobber surfaces on reload
        ms.add_kv(LLM_KV_EXPERT_LATENT_LENGTH,       n_ff);
        ms.add_kv(LLM_KV_INTERLEAVE_MOE_LAYER_STEP,  uint32_t(2));
        ms.add_kv(LLM_KV_EXPERT_COUNT,               uint32_t(2));
        ms.add_kv(LLM_KV_EXPERT_USED_COUNT,          uint32_t(1));
        ms.add_kv(LLM_KV_EXPERT_SHARED_COUNT,        uint32_t(1));
        ms.add_kv(LLM_KV_EXPERT_GATING_FUNC,         arch == LLM_ARCH_DEEPSEEK4 ? uint32_t(4) : uint32_t(2)); // sqrtsoftplus : sigmoid
        ms.add_kv(LLM_KV_EXPERT_GROUP_SCALE,         1.0f);
        ms.add_kv(LLM_KV_EXPERTS_PER_GROUP,          uint32_t(1));
    }

    ms.add_kv(LLM_KV_POSNET_EMBEDDING_LENGTH,   n_embd);
    ms.add_kv(LLM_KV_POSNET_BLOCK_COUNT,        n_layer);
    ms.add_kv(LLM_KV_CONVNEXT_EMBEDDING_LENGTH, n_embd);
    ms.add_kv(LLM_KV_CONVNEXT_BLOCK_COUNT,      n_layer);
    ms.add_kv(LLM_KV_XIELU_ALPHA_N,             1.0f);
    ms.add_kv(LLM_KV_XIELU_ALPHA_P,             1.0f);
    ms.add_kv(LLM_KV_XIELU_BETA,                1.0f);
    ms.add_kv(LLM_KV_XIELU_EPS,                 1.0e-7f);
    ms.add_kv(LLM_KV_SSM_INNER_SIZE,            arch == LLM_ARCH_QWEN3NEXT || arch == LLM_ARCH_QWEN35 || arch == LLM_ARCH_QWEN35MOE || arch == LLM_ARCH_QWEN4EXP ? 256 : 2*n_embd);
    // Zaya CCA applies sequential depthwise and grouped kernel-size-2 convolutions.
    ms.add_kv(LLM_KV_SSM_CONV_KERNEL,           arch == LLM_ARCH_ZAYA ? uint32_t(2) : uint32_t(4));
    ms.add_kv(LLM_KV_SSM_STATE_SIZE,            uint32_t(128));
    ms.add_kv(LLM_KV_SSM_TIME_STEP_RANK,        n_head);
    ms.add_kv(LLM_KV_SSM_GROUP_COUNT,           arch == LLM_ARCH_PLAMO2 ? 0 : uint32_t(2));
    ms.add_kv(LLM_KV_KDA_HEAD_DIM,              uint32_t(128));
    ms.add_kv(LLM_KV_KDA_SAFE_GATE,              true);
    ms.add_kv(LLM_KV_KDA_GATE_LOWER_BOUND,       -5.0f);
    if (arch == LLM_ARCH_BAILINGMOE3) {
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP,   std::vector<float>({0.0f, 4.0f}));
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_SHEXP, std::vector<float>({0.0f, 5.0f}));
    }
    ms.add_kv(LLM_KV_WKV_HEAD_SIZE,             n_embd/n_head);
    ms.add_kv(LLM_KV_SHORTCONV_L_CACHE,         uint32_t(3));
    ms.add_kv(LLM_KV_RESIDUAL_SCALE,            3.5565588200778455f);
    ms.add_kv(LLM_KV_ATTN_RES_BLOCK_SIZE,       uint32_t(12));
    ms.add_kv(LLM_KV_ACTIVATION_SITU_BETA,      4.0f);
    ms.add_kv(LLM_KV_ACTIVATION_SITU_LINEAR_BETA, 25.0f);
    ms.add_kv(LLM_KV_KDA_GATE_LOWER_BOUND,      -5.0f);

    for (uint32_t il = 0; il < n_layer; il++) {
        ggml_tensor t;
        memset(&t, 0, sizeof(ggml_tensor));
        t.type = GGML_TYPE_F16;
        ggml_format_name(&t, "conv%" PRIu32 "d.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "posnet.%" PRIu32 ".conv1.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "posnet.%" PRIu32 ".conv2.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "convnext.%" PRIu32 ".dw.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
    }
    return ret;
}

static bool silent_model_load_progress(float /*progress*/, void * /*user_data*/) {
    return true;
}

struct glm5next_chunk_execution_probe {
    bool enabled = false;
    uint64_t count = 0;
};

static bool glm5next_chunk_execution_cb(ggml_tensor * tensor, bool ask, void * user_data) {
    static constexpr char marker[] = "kpool_chunked_out-";
    auto * probe = static_cast<glm5next_chunk_execution_probe *>(user_data);
    const bool matches = std::strncmp(ggml_get_name(tensor), marker, sizeof(marker) - 1) == 0;
    if (ask) {
        return probe->enabled && matches;
    }
    GGML_ASSERT(probe->enabled && matches);
    ++probe->count;
    return true;
}

static std::pair<llama_model_ptr, llama_context_ptr> get_model_and_ctx(
        struct gguf_context * gguf_ctx, FILE * file, const size_t seed, const std::vector<ggml_backend_dev_t> & devs,
        const llama_split_mode split_mode = LLAMA_SPLIT_MODE_LAYER, bool encode = false, bool load_mtp = false,
        bool turbo3_mla = false, uint32_t n_ctx = 0, uint32_t n_ubatch = 0,
        ggml_backend_sched_eval_callback cb_eval = nullptr, void * cb_eval_user_data = nullptr) {
    GGML_ASSERT((gguf_ctx == nullptr) != (file == nullptr));
    llama_model_params model_params = llama_model_default_params();
    model_params.progress_callback = silent_model_load_progress;
    std::vector<ggml_backend_dev_t> devs_copy = devs;
    devs_copy.push_back(nullptr);
    model_params.devices = devs_copy.data();
    model_params.split_mode = split_mode;
    model_params.load_mtp = load_mtp;

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx;
    ctx_params.n_threads = 4;
    ctx_params.n_threads_batch = 4;
    ctx_params.cb_eval = cb_eval;
    ctx_params.cb_eval_user_data = cb_eval_user_data;
    if (!encode) {
        ctx_params.n_ubatch = n_ubatch > 0 ? n_ubatch : 64;
    }
    if (turbo3_mla || (load_mtp && split_mode == LLAMA_SPLIT_MODE_TENSOR)) {
        ctx_params.type_k = GGML_TYPE_F32;
        ctx_params.type_v = GGML_TYPE_F32;
        ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    }

    size_t tmp = seed;
    llama_model_ptr model(gguf_ctx != nullptr ?
        llama_model_init_from_user(gguf_ctx, set_tensor_data, &tmp, model_params) :
        llama_model_load_from_file_ptr(file, model_params));
    if (!model) {
        throw std::runtime_error("failed to create llama model");
    }
    llama_context_ptr lctx(llama_init_from_model(model.get(), ctx_params));
    if (!lctx) {
        throw std::runtime_error("failed to create llama context");
    }
    return std::make_pair(std::move(model), std::move(lctx));
}

static void test_glm5next_fa_cache_types(llama_model * model) {
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 0;
    ctx_params.n_threads = 4;
    ctx_params.n_threads_batch = 4;
    ctx_params.n_ubatch = 64;
    ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;

    llama_context_ptr ctx_f16(llama_init_from_model(model, ctx_params));
    if (!ctx_f16) {
        throw std::runtime_error("GLM5Next rejected flash attention with homogeneous F16 cache");
    }

    ctx_params.type_k = GGML_TYPE_F32;
    ctx_params.type_v = GGML_TYPE_F32;
    llama_context_ptr ctx_f32(llama_init_from_model(model, ctx_params));
    if (!ctx_f32) {
        throw std::runtime_error("GLM5Next rejected flash attention with F32 cache");
    }

    ctx_params.type_k = GGML_TYPE_F16;
    llama_context_ptr ctx_mixed(llama_init_from_model(model, ctx_params));
    if (ctx_mixed) {
        throw std::runtime_error("GLM5Next accepted flash attention with mixed F16/F32 cache");
    }

    ctx_params.type_k = GGML_TYPE_Q8_0;
    ctx_params.type_v = GGML_TYPE_Q8_0;
    llama_context_ptr ctx_quantized(llama_init_from_model(model, ctx_params));
    if (ctx_quantized) {
        throw std::runtime_error("GLM5Next accepted flash attention with unsupported quantized cache");
    }
}

static void test_glm5next_native_f16_chunk_graph(
        llama_model * model, const std::vector<llama_token> & tokens) {
    scoped_env_var turbo3_mla("LLAMA_GLM5NEXT_TURBO3_MLA", "0");
    scoped_env_var segmented("LLAMA_GLM5NEXT_SPARSE_SEGMENTED", "1");
    scoped_env_var query_chunk("LLAMA_GLM5NEXT_SPARSE_QUERY_CHUNK", "14");
    scoped_env_var query_scatter("LLAMA_GLM5NEXT_SPARSE_QUERY_SCATTER", "1");

    llama_context_params params = llama_context_default_params();
    params.n_ctx = 32768;
    params.n_threads = 4;
    params.n_threads_batch = 4;
    params.n_ubatch = 256;
    params.type_k = GGML_TYPE_F16;
    params.type_v = GGML_TYPE_F16;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    glm5next_chunk_execution_probe chunk_probe;
    params.cb_eval = glm5next_chunk_execution_cb;
    params.cb_eval_user_data = &chunk_probe;

    llama_context_ptr chunked(llama_init_from_model(model, params));
    if (!chunked) {
        throw std::runtime_error("failed to reserve GLM5Next native F16 32K query-chunk graph");
    }

    glm5next_chunk_execution_probe dense_f16_probe;
    llama_context_params dense_f16_params = params;
    dense_f16_params.cb_eval_user_data = &dense_f16_probe;
    llama_context_ptr dense_f16;
    llama_context_ptr dense_f32;
    glm5next_chunk_execution_probe dense_f32_probe;
    {
        scoped_env_var chunk_disabled("LLAMA_GLM5NEXT_SPARSE_QUERY_CHUNK", "0");
        scoped_env_var scatter_disabled("LLAMA_GLM5NEXT_SPARSE_QUERY_SCATTER", "0");
        dense_f16.reset(llama_init_from_model(model, dense_f16_params));

        llama_context_params dense_f32_params = dense_f16_params;
        dense_f32_params.type_k = GGML_TYPE_F32;
        dense_f32_params.type_v = GGML_TYPE_F32;
        dense_f32_params.cb_eval_user_data = &dense_f32_probe;
        dense_f32.reset(llama_init_from_model(model, dense_f32_params));
    }
    if (!dense_f16 || !dense_f32) {
        throw std::runtime_error("failed to create GLM5Next native F16 query-chunk controls");
    }

    const auto decode_batch = [&](llama_context * ctx, uint32_t pos0, bool all_logits) {
        constexpr uint32_t count = 256;
        llama_batch batch = llama_batch_init(count, 0, 1);
        for (uint32_t i = 0; i < count; ++i) {
            common_batch_add(batch, tokens[(pos0 + i) % tokens.size()], pos0 + i, { 0 },
                    all_logits || i + 1 == count);
        }
        if (llama_decode(ctx, batch)) {
            llama_batch_free(batch);
            throw std::runtime_error("failed to execute GLM5Next native F16 query-chunk batch");
        }

        std::vector<float> logits;
        if (all_logits) {
            const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
            logits.reserve(count*n_vocab);
            for (uint32_t i = 0; i < count; ++i) {
                const float * row = llama_get_logits_ith(ctx, i);
                logits.insert(logits.end(), row, row + n_vocab);
            }
        }
        llama_batch_free(batch);
        return logits;
    };

    chunk_probe = { true, 0 };
    for (uint32_t pos = 0; pos < 3584; pos += 256) {
        decode_batch(chunked.get(), pos, false);
        decode_batch(dense_f16.get(), pos, false);
        decode_batch(dense_f32.get(), pos, false);
    }
    if (chunk_probe.count != 0) {
        throw std::runtime_error("GLM5Next native F16 query chunks activated below their configured width");
    }

    dense_f16_probe = { true, 0 };
    dense_f32_probe = { true, 0 };
    const std::vector<float> logits_chunked = decode_batch(chunked.get(), 3584, true);
    const std::vector<float> logits_dense_f16 = decode_batch(dense_f16.get(), 3584, true);
    const std::vector<float> logits_dense_f32 = decode_batch(dense_f32.get(), 3584, true);
    if (chunk_probe.count != 1 || dense_f16_probe.count != 0 || dense_f32_probe.count != 0) {
        throw std::runtime_error("GLM5Next native F16 execution probe did not distinguish chunk and dense routes");
    }
    chunk_probe.enabled = false;
    dense_f16_probe.enabled = false;
    dense_f32_probe.enabled = false;

    const double chunk_nmse = nmse(logits_dense_f16, logits_chunked);
    const double f16_nmse = nmse(logits_dense_f32, logits_dense_f16);
    std::fprintf(stderr,
            "GLM5Next native F16 runtime chunk NMSE: %.9g; dense F16 vs F32 NMSE: %.9g (candidate hits: %" PRIu64 ")\n",
            chunk_nmse, f16_nmse, chunk_probe.count);
    if (!std::isfinite(chunk_nmse) || chunk_nmse > 5e-4 ||
            !std::isfinite(f16_nmse) || f16_nmse > 5e-4) {
        throw std::runtime_error("GLM5Next native F16 runtime query-chunk numerical gate failed");
    }
}

static void test_glm5next_turbo3_cpu_guard(llama_model * model) {
    scoped_env_var turbo3_mla("LLAMA_GLM5NEXT_TURBO3_MLA", "1");
    scoped_env_var segmented("LLAMA_GLM5NEXT_SPARSE_SEGMENTED", "1");

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 0;
    ctx_params.n_threads = 4;
    ctx_params.n_threads_batch = 4;
    ctx_params.n_ubatch = 64;
    ctx_params.type_k = GGML_TYPE_F32;
    ctx_params.type_v = GGML_TYPE_F32;
    ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;

    llama_context_ptr ctx(llama_init_from_model(model, ctx_params));
    if (ctx) {
        throw std::runtime_error("GLM5Next accepted Turbo3 MLA without full GPU offload");
    }
}

static void test_glm5next_turbo3_state(
        llama_model * model, llama_context * ctx, const std::vector<llama_token> & tokens);

static void test_glm5next_turbo3_graph(
        struct gguf_context * gguf_ctx, const size_t seed, const std::vector<ggml_backend_dev_t> & devs,
        const llama_split_mode split_mode, const std::vector<llama_token> & tokens) {
    scoped_env_var turbo3_mla("LLAMA_GLM5NEXT_TURBO3_MLA", "1");
    scoped_env_var segmented("LLAMA_GLM5NEXT_SPARSE_SEGMENTED", "1");

    // Context construction reserves both prompt and decode graphs.  The Turbo3
    // graph validator checks their cache-write, gather, forward-WHT, inverse-WHT,
    // type, placement, and backend-support contracts before any decode occurs.
    glm5next_chunk_execution_probe chunk_probe;
    std::pair<llama_model_ptr, llama_context_ptr> model_and_ctx;
    {
        scoped_env_var query_chunk("LLAMA_GLM5NEXT_SPARSE_QUERY_CHUNK", "14");
        scoped_env_var query_scatter("LLAMA_GLM5NEXT_SPARSE_QUERY_SCATTER", "1");
        model_and_ctx = get_model_and_ctx(
                gguf_ctx, nullptr, seed, devs, split_mode, false, true, true, 32768, 256,
                glm5next_chunk_execution_cb, &chunk_probe);
    }

    // Execute the actual late-prompt runtime route, not only its reserve graph.
    // Fourteen 256-token batches grow n_kv to 3,584 while proving that geometry
    // below the configured width stays on the dense fallback. The next batch
    // reaches a legal width of 14 and uses nineteen chunks.
    llama_context_params dense_params = llama_context_default_params();
    dense_params.n_ctx = 32768;
    dense_params.n_threads = 4;
    dense_params.n_threads_batch = 4;
    dense_params.n_ubatch = 256;
    dense_params.type_k = GGML_TYPE_F32;
    dense_params.type_v = GGML_TYPE_F32;
    dense_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    glm5next_chunk_execution_probe dense_probe;
    dense_params.cb_eval = glm5next_chunk_execution_cb;
    dense_params.cb_eval_user_data = &dense_probe;
    llama_context_ptr dense_ctx;
    {
        scoped_env_var query_chunk("LLAMA_GLM5NEXT_SPARSE_QUERY_CHUNK", "0");
        scoped_env_var query_scatter_disabled("LLAMA_GLM5NEXT_SPARSE_QUERY_SCATTER", "0");
        dense_ctx.reset(llama_init_from_model(model_and_ctx.first.get(), dense_params));
    }
    if (!dense_ctx) {
        throw std::runtime_error("failed to create GLM5Next Turbo3 dense-control context");
    }

    const auto decode_batch = [&](llama_context * ctx, uint32_t pos0, uint32_t count, bool all_logits) {
        llama_batch batch = llama_batch_init(count, 0, 1);
        for (uint32_t i = 0; i < count; ++i) {
            common_batch_add(batch, tokens[(pos0 + i) % tokens.size()], pos0 + i, { 0 },
                    all_logits || i + 1 == count);
        }
        if (llama_decode(ctx, batch)) {
            llama_batch_free(batch);
            throw std::runtime_error("failed to decode GLM5Next Turbo3 query-chunk batch");
        }

        std::vector<float> logits;
        if (all_logits) {
            const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model_and_ctx.first.get()));
            logits.reserve(count*n_vocab);
            for (uint32_t i = 0; i < count; ++i) {
                const float * row = llama_get_logits_ith(ctx, i);
                logits.insert(logits.end(), row, row + n_vocab);
            }
        }
        llama_batch_free(batch);
        return logits;
    };

    chunk_probe = { true, 0 };
    for (uint32_t pos = 0; pos < 3584; pos += 256) {
        decode_batch(dense_ctx.get(), pos, 256, false);
        decode_batch(model_and_ctx.second.get(), pos, 256, false);
    }
    if (chunk_probe.count != 0) {
        throw std::runtime_error("GLM5Next Turbo3 query chunks activated below their configured width");
    }
    dense_probe = { true, 0 };
    const std::vector<float> dense_logits = decode_batch(dense_ctx.get(), 3584, 256, true);
    const std::vector<float> chunked_logits = decode_batch(model_and_ctx.second.get(), 3584, 256, true);
    if (dense_probe.count != 0 || chunk_probe.count != 1) {
        throw std::runtime_error("GLM5Next Turbo3 query-chunk execution probe did not observe the expected routes");
    }
    dense_probe.enabled = false;
    chunk_probe.enabled = false;
    const double chunk_nmse = nmse(dense_logits, chunked_logits);
    std::fprintf(stderr, "GLM5Next Turbo3 runtime query-chunk NMSE: %.9g (candidate hits: %" PRIu64 ")\n",
            chunk_nmse, chunk_probe.count);
    if (!std::isfinite(chunk_nmse) || chunk_nmse > 5e-4) {
        throw std::runtime_error("GLM5Next Turbo3 runtime query chunks changed logits (NMSE " +
                                 std::to_string(chunk_nmse) + ")");
    }
    dense_ctx.reset();
    llama_memory_clear(llama_get_memory(model_and_ctx.second.get()), true);

    test_glm5next_turbo3_state(model_and_ctx.first.get(), model_and_ctx.second.get(), tokens);
}

static std::vector<float> get_logits(
        llama_model * model, llama_context * lctx, const std::vector<llama_token> & tokens, bool encode = false) {
    const uint32_t n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const uint32_t n_ctx    = llama_n_ctx(lctx);
    const uint32_t n_tokens = tokens.size();
    llama_batch batch = llama_batch_init(n_ctx, 0, 1);
    GGML_ASSERT(n_tokens <= n_ctx);
    for (uint32_t pos = 0; pos < n_tokens; pos++) {
        common_batch_add(batch, tokens[pos], pos, {0}, true);
    }
    batch.n_tokens = n_tokens;
    if (encode) {
        if (llama_encode(lctx, batch)) {
            llama_batch_free(batch);
            throw std::runtime_error("failed to encode batch");
        }
    }
    if (llama_decode(lctx, batch)) {
        llama_batch_free(batch);
        throw std::runtime_error("failed to decode batch");
    }

    std::vector<float> ret;
    ret.reserve(n_tokens*n_vocab);
    for (uint32_t i = 0; i < n_tokens; i++) {
        const float * logits_ith = llama_get_logits_ith(lctx, i);
        for (uint32_t j = 0; j < n_vocab; j++) {
            ret.push_back(logits_ith[j]);
        }
    }
    llama_batch_free(batch);
    return ret;
}

static void test_glm5next_failed_state_restore_cleanup(
        llama_model * model, const std::vector<llama_token> & tokens) {
    const auto make_context = [&](ggml_type cache_type) {
        llama_context_params params = llama_context_default_params();
        params.n_ctx = 0;
        params.n_threads = 4;
        params.n_threads_batch = 4;
        params.n_ubatch = 64;
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        params.type_k = cache_type;
        params.type_v = cache_type;

        llama_context_ptr ctx(llama_init_from_model(model, params));
        if (!ctx) {
            throw std::runtime_error("failed to create GLM5Next state-cleanup context");
        }
        return ctx;
    };

    const size_t n_prefix = std::min<size_t>(8, tokens.size());
    const std::vector<llama_token> prefix(tokens.begin(), tokens.begin() + n_prefix);

    llama_context_ptr src = make_context(GGML_TYPE_F32);
    get_logits(model, src.get(), prefix);

    const size_t state_size = llama_state_seq_get_size(src.get(), 0);
    std::vector<uint8_t> state(state_size);
    if (state_size == 0 || llama_state_seq_get_data(src.get(), state.data(), state.size(), 0) != state_size) {
        throw std::runtime_error("failed to save GLM5Next mismatched-cache state");
    }

    llama_context_ptr dst = make_context(GGML_TYPE_F16);
    get_logits(model, dst.get(), prefix);
    if (llama_memory_seq_pos_max(llama_get_memory(dst.get()), 0) < 0) {
        throw std::runtime_error("GLM5Next mismatch destination was not dirtied");
    }

    if (llama_state_seq_set_data(dst.get(), state.data(), state.size(), 0) != 0) {
        throw std::runtime_error("GLM5Next accepted state with a mismatched attention-cache type");
    }
    if (llama_memory_seq_pos_max(llama_get_memory(dst.get()), 0) != -1) {
        throw std::runtime_error("failed GLM5Next state restore retained sequence positions");
    }

    // A position check alone can be hidden by the now-empty attention cache.
    // Decode from position zero and compare against a genuinely fresh context
    // to cover stale recurrent or indexer state left by a partial restore.
    llama_context_ptr fresh = make_context(GGML_TYPE_F16);
    const std::vector<llama_token> probe = { tokens.back() };
    const std::vector<float> logits_dst = get_logits(model, dst.get(), probe);
    const std::vector<float> logits_ref = get_logits(model, fresh.get(), probe);
    if (logits_dst != logits_ref) {
        throw std::runtime_error("failed GLM5Next state restore left the destination memory dirty");
    }
}

static void test_glm5next_paged_state_restore(llama_model * model, llama_context * ctx, const std::vector<llama_token> & tokens) {
    const auto decode_range = [&](size_t begin, size_t end, bool logits) {
        llama_batch batch = llama_batch_init(end - begin, 0, 1);
        for (size_t i = begin; i < end; ++i) {
            common_batch_add(batch, tokens[i], i, { 0 }, logits && i + 1 == end);
        }
        const int status = llama_decode(ctx, batch);
        llama_batch_free(batch);
        if (status) {
            throw std::runtime_error("failed to decode GLM5Next state page");
        }
    };

    const auto save_page = [&](llama_pos p0, llama_pos p1) {
        const size_t size = llama_state_seq_get_size_range_ext(ctx, 0, p0, p1, LLAMA_STATE_SEQ_FLAGS_NONE);
        std::vector<uint8_t> state(size);
        if (size == 0 || llama_state_seq_get_data_range_ext(ctx, state.data(), state.size(), 0, p0, p1, LLAMA_STATE_SEQ_FLAGS_NONE) != size) {
            throw std::runtime_error("failed to save GLM5Next sequence-state page");
        }
        return state;
    };

    const auto save_full = [&] {
        const size_t size = llama_state_seq_get_size(ctx, 0);
        std::vector<uint8_t> state(size);
        if (size == 0 || llama_state_seq_get_data(ctx, state.data(), state.size(), 0) != size) {
            throw std::runtime_error("failed to save complete GLM5Next sequence state");
        }
        return state;
    };

    const auto decode_next = [&] {
        llama_token token = tokens.back();
        llama_pos pos = tokens.size();
        llama_batch batch = llama_batch_get_one(&token, 1);
        batch.pos = &pos;
        if (llama_decode(ctx, batch)) {
            throw std::runtime_error("failed to decode GLM5Next state-page continuation");
        }
    };

    decode_range(0, 64, false);
    const std::vector<uint8_t> page_0 = save_page(0, 64);
    decode_range(64, 128, false);
    const std::vector<uint8_t> page_1 = save_page(64, 128);
    const std::vector<uint8_t> full_ref = save_full();
    decode_next();

    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const float * logits = llama_get_logits_ith(ctx, -1);
    std::vector<float> logits_ref(logits, logits + n_vocab);

    llama_memory_clear(llama_get_memory(ctx), true);
    if (llama_state_seq_set_data(ctx, page_0.data(), page_0.size(), 0) != page_0.size() ||
            llama_state_seq_set_data(ctx, page_1.data(), page_1.size(), 0) != page_1.size()) {
        throw std::runtime_error("failed to restore GLM5Next sequence-state pages");
    }
    const std::vector<uint8_t> full_restored = save_full();
    if (full_ref != full_restored) {
        const size_t shared = std::min(full_ref.size(), full_restored.size());
        const auto mismatch = std::mismatch(full_ref.begin(), full_ref.begin() + shared, full_restored.begin());
        const size_t offset = mismatch.first == full_ref.begin() + shared
            ? shared
            : size_t(mismatch.first - full_ref.begin());
        throw std::runtime_error("GLM5Next sequence-state pages changed serialized state at byte " +
                std::to_string(offset) + " (reference " + std::to_string(full_ref.size()) +
                " bytes, restored " + std::to_string(full_restored.size()) + " bytes)");
    }
    decode_next();

    logits = llama_get_logits_ith(ctx, -1);
    if (!std::equal(logits_ref.begin(), logits_ref.end(), logits)) {
        throw std::runtime_error("GLM5Next sequence-state page restore changed logits");
    }
    llama_memory_clear(llama_get_memory(ctx), true);
}

static void test_glm5next_native_f16_state(
        llama_model * model, const std::vector<llama_token> & tokens) {
    scoped_env_var turbo3_mla("LLAMA_GLM5NEXT_TURBO3_MLA", "0");
    scoped_env_var query_chunk("LLAMA_GLM5NEXT_SPARSE_QUERY_CHUNK", "0");
    scoped_env_var query_scatter("LLAMA_GLM5NEXT_SPARSE_QUERY_SCATTER", "0");

    const auto make_context = [&](llama_context_type ctx_type, ggml_type cache_type) {
        llama_context_params params = llama_context_default_params();
        params.n_ctx = 0;
        params.n_threads = 4;
        params.n_threads_batch = 4;
        params.n_ubatch = 64;
        params.type_k = cache_type;
        params.type_v = cache_type;
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        params.ctx_type = ctx_type;

        llama_context_ptr ctx(llama_init_from_model(model, params));
        if (!ctx) {
            throw std::runtime_error(ctx_type == LLAMA_CONTEXT_TYPE_MTP ?
                    "failed to create GLM5Next native F16 draft context" :
                    "failed to create GLM5Next native F16 target context");
        }
        return ctx;
    };

    llama_context_ptr target = make_context(LLAMA_CONTEXT_TYPE_DEFAULT, GGML_TYPE_F16);
    const std::vector<float> logits_f16 = get_logits(model, target.get(), tokens);
    if (!std::all_of(logits_f16.begin(), logits_f16.end(), [](float value) { return std::isfinite(value); })) {
        throw std::runtime_error("GLM5Next native F16 target produced non-finite logits");
    }
    llama_context_ptr control = make_context(LLAMA_CONTEXT_TYPE_DEFAULT, GGML_TYPE_F32);
    const std::vector<float> logits_f32 = get_logits(model, control.get(), tokens);
    const double native_f16_nmse = nmse(logits_f32, logits_f16);
    std::fprintf(stderr, "GLM5Next native F16 cache NMSE vs F32: %.9g\n", native_f16_nmse);
    if (!std::isfinite(native_f16_nmse) || native_f16_nmse > 5e-4) {
        throw std::runtime_error("GLM5Next native F16 cache exceeded F32 control tolerance (NMSE " +
                                 std::to_string(native_f16_nmse) + ")");
    }
    llama_memory_clear(llama_get_memory(target.get()), true);
    test_glm5next_paged_state_restore(model, target.get(), tokens);

    llama_context_ptr draft = make_context(LLAMA_CONTEXT_TYPE_MTP, GGML_TYPE_F16);
    llama_context_ptr draft_control = make_context(LLAMA_CONTEXT_TYPE_MTP, GGML_TYPE_F32);
    const int32_t n_embd = llama_model_n_embd(model);
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    const auto decode_draft = [&](llama_context * ctx, llama_pos pos) {
        std::vector<float> hidden(n_embd);
        for (int32_t i = 0; i < n_embd; ++i) {
            hidden[i] = std::sin(float((pos + 1)*(i + 1))*0.001f);
        }
        common_batch batch(ctx);
        const llama_embd state { hidden.data(), 1, (size_t) n_embd };
        const int32_t idx = batch.add_embd(state, &pos, 0, true);
        batch.set_embd_state(idx, state);
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
            throw std::runtime_error("failed to decode GLM5Next native F16 draft state");
        }

        const float * logits = llama_get_logits_ith(ctx, -1);
        return std::vector<float>(logits, logits + n_vocab);
    };

    for (llama_pos pos = 0; pos < 16; ++pos) {
        decode_draft(draft.get(), pos);
        decode_draft(draft_control.get(), pos);
    }
    const size_t page_size = llama_state_seq_get_size_range_ext(
            draft.get(), 0, 0, 16, LLAMA_STATE_SEQ_FLAGS_NONE);
    std::vector<uint8_t> page(page_size);
    if (page_size == 0 || llama_state_seq_get_data_range_ext(
            draft.get(), page.data(), page.size(), 0, 0, 16, LLAMA_STATE_SEQ_FLAGS_NONE) != page_size) {
        throw std::runtime_error("failed to save GLM5Next native F16 draft state page");
    }
    const std::vector<float> draft_ref = decode_draft(draft.get(), 16);
    const std::vector<float> draft_f32 = decode_draft(draft_control.get(), 16);
    const double draft_f16_nmse = nmse(draft_f32, draft_ref);
    std::fprintf(stderr, "GLM5Next native F16 draft cache NMSE vs F32: %.9g\n", draft_f16_nmse);
    if (!std::isfinite(draft_f16_nmse) || draft_f16_nmse > 5e-4) {
        throw std::runtime_error("GLM5Next native F16 draft cache exceeded F32 control tolerance");
    }

    llama_memory_clear(llama_get_memory(draft.get()), true);
    if (llama_state_seq_set_data(draft.get(), page.data(), page.size(), 0) != page.size()) {
        throw std::runtime_error("failed to restore GLM5Next native F16 draft state page");
    }
    const std::vector<float> draft_restored = decode_draft(draft.get(), 16);
    if (draft_restored != draft_ref) {
        throw std::runtime_error("GLM5Next native F16 draft page restore changed logits");
    }
}

struct glm5next_mtp_driver_fixture {
    struct selection_probe {
        size_t calls = 0;
        bool single_token = true;
        std::vector<int32_t> last;

        static bool callback(ggml_tensor * tensor, bool ask, void * user_data) {
            static constexpr char name[] = "indexer_top_k_pools-2";
            const bool selected = std::strcmp(ggml_get_name(tensor), name) == 0;
            if (ask) {
                return selected;
            }
            auto & probe = *static_cast<selection_probe *>(user_data);
            if (!selected || tensor->type != GGML_TYPE_I32 || tensor->ne[0] != 2) {
                throw std::runtime_error("unexpected GLM5Next NextN pool-selection tensor");
            }
            ++probe.calls;
            probe.single_token &= tensor->ne[1] == 1;
            probe.last.resize(ggml_nelements(tensor));
            ggml_backend_tensor_get(tensor, probe.last.data(), 0, ggml_nbytes(tensor));
            return true;
        }
    } probe;

    llama_model * model;
    llama_context_ptr target;
    llama_context_ptr draft;
    common_speculative_ptr spec;

    size_t last_draft_forwards = 0;

    static llama_context_ptr make_context(llama_model * model, llama_context_type type, selection_probe * probe = nullptr) {
        llama_context_params params = llama_context_default_params();
        params.n_ctx = 256;
        params.n_batch = 64;
        params.n_ubatch = 64;
        params.n_threads = 4;
        params.n_threads_batch = 4;
        params.n_rs_seq = 5;
        params.type_k = GGML_TYPE_F16;
        params.type_v = GGML_TYPE_F16;
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        params.ctx_type = type;
        params.cb_eval = probe ? selection_probe::callback : nullptr;
        params.cb_eval_user_data = probe;
        llama_context_ptr result(llama_init_from_model(model, params));
        if (!result) {
            throw std::runtime_error("failed to create GLM5Next MTP driver context");
        }
        return result;
    }

    glm5next_mtp_driver_fixture(llama_model * model, float p_min = 0.0f, int n_min = 0)
        : model(model),
          target(make_context(model, LLAMA_CONTEXT_TYPE_DEFAULT)),
          draft(make_context(model, LLAMA_CONTEXT_TYPE_MTP, &probe)) {
        common_params_speculative params;
        params.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
        params.draft.ctx_tgt = target.get();
        params.draft.ctx_dft = draft.get();
        params.draft.n_max = 5;
        params.draft.n_min = n_min;
        params.draft.p_min = p_min;
        params.draft.backend_sampling = false;
        spec.reset(common_speculative_init(params, 1));
        if (!spec || !common_speculative_get_synth_probs(spec.get()).empty()) {
            throw std::runtime_error("failed to create ordinary GLM5Next MTP driver");
        }
    }

    static void decode(llama_context * ctx, const llama_tokens & tokens, llama_pos pos,
                       common_speculative * spec = nullptr) {
        common_batch batch(ctx);
        for (size_t i = 0; i < tokens.size(); ++i) {
            batch.add(tokens[i], pos + i, 0, true);
        }
        const bool ok = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0 && common_speculative_process(spec, batch);
        if (!ok) {
            throw std::runtime_error("GLM5Next MTP target/process decode failed");
        }
    }

    void process(const llama_tokens & tokens, llama_pos pos) {
        decode(target.get(), tokens, pos, spec.get());
    }

    void prefill(const llama_tokens & tokens) {
        // Leave draft memory and hidden carry dirty to exercise the position-zero reset.
        llama_memory_clear(llama_get_memory(target.get()), true);
        const size_t split = tokens.size()/2;
        process(llama_tokens(tokens.begin(), tokens.begin() + split), 0);
        process(llama_tokens(tokens.begin() + split, tokens.end()), split);
        common_speculative_begin(spec.get(), 0, tokens);
    }

    llama_tokens propose(const llama_tokens & prefix, llama_token seed) {
        llama_tokens result;
        auto & dp = common_speculative_get_draft_params(spec.get(), 0);
        dp.drafting = true;
        dp.pos0 = prefix.size();
        dp.id_last = seed;
        dp.prompt = &prefix;
        dp.result = &result;
        const size_t calls_before = probe.calls;
        probe.single_token = true;
        common_speculative_draft(spec.get());
        llama_synchronize(draft.get());
        last_draft_forwards = probe.calls - calls_before;
        if (!probe.single_token || last_draft_forwards == 0) {
            throw std::runtime_error("MTP draft did not execute single-token NextN indexing");
        }
        if (dp.drafting) {
            throw std::runtime_error("MTP driver retained an active draft request");
        }
        dp.prompt = nullptr;
        dp.result = nullptr;
        // The server removes speculative draft KV before target verification/catch-up.
        if (!llama_memory_seq_rm(llama_get_memory(draft.get()), 0, prefix.size(), -1)) {
            throw std::runtime_error("failed to discard speculative MTP KV");
        }
        return result;
    }

    void truncate(llama_pos pos) {
        if (!llama_memory_seq_rm(llama_get_memory(target.get()), 0, pos, -1) ||
            !llama_memory_seq_rm(llama_get_memory(draft.get()), 0, pos, -1)) {
            throw std::runtime_error("GLM5Next MTP driver rollback failed");
        }
    }

    static std::vector<float> logits(llama_context * ctx) {
        const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
        const float * values = llama_get_logits_ith(ctx, -1);
        if (!values) {
            throw std::runtime_error("GLM5Next MTP fixture requested missing logits");
        }
        return std::vector<float>(values, values + n_vocab);
    }

    static llama_token greedy(llama_context * ctx) {
        const auto values = logits(ctx);
        return std::max_element(values.begin(), values.end()) - values.begin();
    }
};

static void test_glm5next_mtp_driver(llama_model * model, const llama_tokens & tokens) {
    scoped_env_var turbo3_mla("LLAMA_GLM5NEXT_TURBO3_MLA", "0");
    scoped_env_var query_chunk("LLAMA_GLM5NEXT_SPARSE_QUERY_CHUNK", "0");
    scoped_env_var query_scatter("LLAMA_GLM5NEXT_SPARSE_QUERY_SCATTER", "0");
    using fixture = glm5next_mtp_driver_fixture;
    const llama_tokens prefix(tokens.begin(), tokens.begin() + 16);
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    auto oracle = fixture::make_context(model, LLAMA_CONTEXT_TYPE_DEFAULT);
    fixture::decode(oracle.get(), prefix, 0);
    llama_tokens continuation;
    for (int i = 0; i < 8; ++i) {
        continuation.push_back(fixture::greedy(oracle.get()));
        fixture::decode(oracle.get(), { continuation.back() }, prefix.size() + i);
    }

    const auto compare_logits = [](llama_context * actual, llama_context * expected, const char * label) {
        const double error = nmse(fixture::logits(expected), fixture::logits(actual));
        if (!std::isfinite(error) || error > 1e-8) {
            throw std::runtime_error(std::string("GLM5Next MTP ") + label + " NMSE " + std::to_string(error));
        }
    };

    fixture tested(model);
    fixture control(model);
    for (int n_accept = 0; n_accept <= 5; ++n_accept) {
        tested.prefill(prefix);
        control.prefill(prefix);
        const llama_tokens natural = tested.propose(prefix, continuation[0]);
        if (natural.size() != 5 || tested.last_draft_forwards != 5) {
            throw std::runtime_error("MTP driver did not execute its full five-step chain");
        }

        // Replace proposals only after running the real driver. The target oracle gives
        // every acceptance boundary without depending on random fixture MTP accuracy.
        llama_tokens proposals(continuation.begin() + 1, continuation.begin() + 6);
        if (n_accept < 5) {
            proposals[n_accept] = (proposals[n_accept] + 1)%n_vocab;
        }
        llama_tokens verify = { continuation[0] };
        verify.insert(verify.end(), proposals.begin(), proposals.end());
        tested.process(verify, prefix.size());

        common_params_sampling sampling;
        sampling.temp = 0.0f;
        common_sampler_ptr sampler(common_sampler_init(model, sampling));
        const llama_tokens accepted = common_sampler_sample_and_accept_n(
                sampler.get(), tested.target.get(), proposals);
        const llama_tokens expected(continuation.begin() + 1, continuation.begin() + n_accept + 2);
        if (accepted != expected) {
            throw std::runtime_error("target verifier changed greedy continuation at acceptance " + std::to_string(n_accept));
        }
        common_speculative_accept(tested.spec.get(), 0, n_accept);
        const llama_pos next_pos = prefix.size() + n_accept + 1;
        tested.truncate(next_pos);

        llama_tokens kept(verify.begin(), verify.begin() + n_accept + 1);
        control.process(kept, prefix.size());
        llama_tokens next_prefix = prefix;
        next_prefix.insert(next_prefix.end(), kept.begin(), kept.end());
        if (tested.propose(next_prefix, accepted.back()) != control.propose(next_prefix, accepted.back())) {
            throw std::runtime_error("accepted target hidden carry changed the next MTP chain");
        }
        compare_logits(tested.draft.get(), control.draft.get(), "accepted hidden carry");
        tested.process({ accepted.back() }, next_pos);
        control.process({ accepted.back() }, next_pos);
        compare_logits(tested.target.get(), control.target.get(), "rollback continuation");
        if (fixture::greedy(tested.target.get()) != continuation[n_accept + 2]) {
            throw std::runtime_error("rollback changed the next target greedy token");
        }
        std::fprintf(stderr, "GLM5Next MTP driver acceptance %d/5: verifier, carry and rollback pass\n", n_accept);
    }

    // A confidence stop still computes its first forward. n_min may discard a full
    // chain; both paths must permit a clean next request with the same driver.
    for (const bool confidence_stop : { true, false }) {
        fixture stopped(model, confidence_stop ? 1.0f : 0.0f, confidence_stop ? 0 : 6);
        stopped.prefill(prefix);
        if (!stopped.propose(prefix, continuation[0]).empty()) {
            throw std::runtime_error("MTP early-stop policy returned a draft");
        }
        if (stopped.last_draft_forwards != (confidence_stop ? 1u : 5u)) {
            throw std::runtime_error("MTP early-stop forward count was incorrect");
        }
        stopped.prefill(prefix);
        fixture fresh(model, confidence_stop ? 1.0f : 0.0f, confidence_stop ? 0 : 6);
        fresh.prefill(prefix);
        if (!stopped.propose(prefix, continuation[0]).empty() || !fresh.propose(prefix, continuation[0]).empty()) {
            throw std::runtime_error("MTP early-stop policy changed after reset");
        }
        compare_logits(stopped.draft.get(), fresh.draft.get(), "early-stop request reset");
    }
    std::fprintf(stderr, "GLM5Next MTP driver confidence stop, n_min discard and position-zero reset pass\n");

    tested.prefill(prefix);
    const auto save_page = [&](llama_context * ctx) {
        std::vector<uint8_t> data(llama_state_seq_get_size_range_ext(ctx, 0, 0, prefix.size(), 0));
        if (data.empty() || llama_state_seq_get_data_range_ext(
                ctx, data.data(), data.size(), 0, 0, prefix.size(), 0) != data.size()) {
            throw std::runtime_error("failed to save MTP driver prefix page");
        }
        return data;
    };
    const auto page_target = save_page(tested.target.get());
    const auto page_draft = save_page(tested.draft.get());
    const llama_tokens suffix(tokens.begin() + 16, tokens.begin() + 20);
    control.prefill(prefix);
    control.process(suffix, prefix.size());

    const llama_tokens displacement(tokens.begin() + 32, tokens.begin() + 48);
    tested.prefill(displacement);
    tested.propose(displacement, tokens[48]);
    const auto restore_page = [](llama_context * ctx, const std::vector<uint8_t> & data) {
        llama_memory_clear(llama_get_memory(ctx), true);
        if (llama_state_seq_set_data(ctx, data.data(), data.size(), 0) != data.size()) {
            throw std::runtime_error("failed to restore MTP driver prefix page");
        }
    };
    restore_page(tested.target.get(), page_target);
    restore_page(tested.draft.get(), page_draft);
    tested.process(suffix, prefix.size());
    compare_logits(tested.target.get(), control.target.get(), "page-restored target");
    llama_tokens restored_prefix = prefix;
    restored_prefix.insert(restored_prefix.end(), suffix.begin(), suffix.end());
    const llama_token restored_seed = fixture::greedy(control.target.get());
    const llama_tokens restored_draft = tested.propose(restored_prefix, restored_seed);
    llama_tokens restored_verify = { restored_seed };
    restored_verify.insert(restored_verify.end(), restored_draft.begin(), restored_draft.end());
    tested.process(restored_verify, restored_prefix.size());
    control.process(restored_verify, restored_prefix.size());
    compare_logits(tested.target.get(), control.target.get(), "page-restored verification");
    std::fprintf(stderr, "GLM5Next MTP driver displaced target+draft page restore and target continuation pass\n");
}

static void test_glm5next_turbo3_state(
        llama_model * model, llama_context * ctx_t3, const std::vector<llama_token> & tokens) {
    GGML_ASSERT(tokens.size() >= 128);

    const auto decode_one = [&](llama_context * ctx, llama_token token, llama_pos pos) {
        llama_batch batch = llama_batch_get_one(&token, 1);
        batch.pos = &pos;
        if (llama_decode(ctx, batch)) {
            throw std::runtime_error("failed to decode GLM5Next state continuation");
        }

        const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        const float * logits = llama_get_logits_ith(ctx, -1);
        return std::vector<float>(logits, logits + n_vocab);
    };

    const auto save_full_state = [&](llama_context * ctx, const char * label) {
        const size_t size = llama_state_get_size(ctx);
        std::vector<uint8_t> state(size);
        if (size == 0 || llama_state_get_data(ctx, state.data(), state.size()) != size) {
            throw std::runtime_error(std::string("failed to save GLM5Next ") + label + " full state");
        }
        return state;
    };

    const auto make_context = [&](bool turbo3_mla_enabled) {
        scoped_env_var turbo3_mla(
                "LLAMA_GLM5NEXT_TURBO3_MLA", turbo3_mla_enabled ? "1" : "0");

        llama_context_params params = llama_context_default_params();
        params.n_ctx = 0;
        params.n_threads = 4;
        params.n_threads_batch = 4;
        params.n_ubatch = 64;
        params.type_k = GGML_TYPE_F32;
        params.type_v = GGML_TYPE_F32;
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;

        llama_context_ptr ctx(llama_init_from_model(model, params));
        if (!ctx) {
            throw std::runtime_error(turbo3_mla_enabled ?
                    "failed to create GLM5Next Turbo3 state context" :
                    "failed to create GLM5Next F32 state context");
        }
        return ctx;
    };

    const std::vector<llama_token> prefix(tokens.begin(), tokens.begin() + 32);
    get_logits(model, ctx_t3, prefix);
    const std::vector<uint8_t> full_t3 = save_full_state(ctx_t3, "Turbo3");
    const std::vector<float> full_ref = decode_one(ctx_t3, tokens[32], 32);

    llama_memory_clear(llama_get_memory(ctx_t3), true);
    if (llama_state_set_data(ctx_t3, full_t3.data(), full_t3.size()) != full_t3.size()) {
        throw std::runtime_error("failed to restore GLM5Next Turbo3 full state");
    }
    const std::vector<float> full_restored = decode_one(ctx_t3, tokens[32], 32);
    if (full_restored != full_ref) {
        throw std::runtime_error("GLM5Next Turbo3 full-state restore changed logits");
    }
    llama_memory_clear(llama_get_memory(ctx_t3), true);

    // Exercise two ranged pages and append restore in the same Turbo3 mode.
    test_glm5next_paged_state_restore(model, ctx_t3, tokens);

    // Both blobs must contain cache rows or an empty state has no serialized
    // cache type with which to reject the opposite mode.
    get_logits(model, ctx_t3, prefix);
    const std::vector<uint8_t> mismatch_t3 = save_full_state(ctx_t3, "Turbo3 mismatch-source");

    llama_context_ptr ctx_f32 = make_context(false);
    get_logits(model, ctx_f32.get(), prefix);
    const std::vector<uint8_t> mismatch_f32 = save_full_state(ctx_f32.get(), "F32 mismatch-source");

    llama_memory_clear(llama_get_memory(ctx_f32.get()), true);
    get_logits(model, ctx_f32.get(), prefix);
    if (llama_state_set_data(ctx_f32.get(), mismatch_t3.data(), mismatch_t3.size()) != 0) {
        throw std::runtime_error("GLM5Next F32 context accepted nonempty Turbo3 state");
    }
    if (llama_memory_seq_pos_max(llama_get_memory(ctx_f32.get()), 0) != -1) {
        throw std::runtime_error("Turbo3-to-F32 state mismatch retained destination positions");
    }
    llama_context_ptr fresh_f32 = make_context(false);
    const std::vector<llama_token> probe = { tokens.back() };
    const std::vector<float> recovered_f32 = get_logits(model, ctx_f32.get(), probe);
    const std::vector<float> fresh_f32_logits = get_logits(model, fresh_f32.get(), probe);
    if (recovered_f32 != fresh_f32_logits) {
        throw std::runtime_error("Turbo3-to-F32 state mismatch left destination memory dirty");
    }

    llama_memory_clear(llama_get_memory(ctx_t3), true);
    get_logits(model, ctx_t3, prefix);
    if (llama_state_set_data(ctx_t3, mismatch_f32.data(), mismatch_f32.size()) != 0) {
        throw std::runtime_error("GLM5Next Turbo3 context accepted nonempty F32 state");
    }
    if (llama_memory_seq_pos_max(llama_get_memory(ctx_t3), 0) != -1) {
        throw std::runtime_error("F32-to-Turbo3 state mismatch retained destination positions");
    }
    llama_context_ptr fresh_t3 = make_context(true);
    const std::vector<float> recovered_t3 = get_logits(model, ctx_t3, probe);
    const std::vector<float> fresh_t3_logits = get_logits(model, fresh_t3.get(), probe);
    if (recovered_t3 != fresh_t3_logits) {
        throw std::runtime_error("F32-to-Turbo3 state mismatch left destination memory dirty");
    }
}

static bool moe_mandatory(const llm_arch arch) {
    switch (arch) {
        case LLM_ARCH_LLAMA4:
        case LLM_ARCH_COHERE2MOE:
        case LLM_ARCH_GROK:
        case LLM_ARCH_QWEN2MOE:
        case LLM_ARCH_QWEN3MOE:
        case LLM_ARCH_QWEN3NEXT:
        case LLM_ARCH_QWEN3VLMOE:
        case LLM_ARCH_QWEN35MOE:
        case LLM_ARCH_QWEN4EXP:
        case LLM_ARCH_PHIMOE:
        case LLM_ARCH_DBRX:
        case LLM_ARCH_OLMOE:
        case LLM_ARCH_ARCTIC:
        case LLM_ARCH_DEEPSEEK:
        case LLM_ARCH_DEEPSEEK2:
        case LLM_ARCH_DEEPSEEK32:
        case LLM_ARCH_DOTS3NOTE:
        case LLM_ARCH_DEEPSEEK4:
        case LLM_ARCH_GLM4_MOE:
        case LLM_ARCH_GLM_DSA:
        case LLM_ARCH_GLM5NEXT:
        case LLM_ARCH_ZAYA:
        case LLM_ARCH_EXAONE_MOE:
        case LLM_ARCH_BAILINGMOE:
        case LLM_ARCH_BAILINGMOE2:
        case LLM_ARCH_BAILINGMOE3:
        case LLM_ARCH_DOTS1:
        case LLM_ARCH_AFMOE:
        case LLM_ARCH_ERNIE4_5:
        case LLM_ARCH_ERNIE4_5_MOE:
        case LLM_ARCH_HUNYUAN_MOE:
        case LLM_ARCH_HY_V3:
        case LLM_ARCH_OPENAI_MOE:
        case LLM_ARCH_LFM2MOE:
        case LLM_ARCH_SMALLTHINKER:
        case LLM_ARCH_LLADA_MOE:
        case LLM_ARCH_GROVEMOE:
        case LLM_ARCH_MINIMAX_01:
        case LLM_ARCH_MINIMAX_M2:
        case LLM_ARCH_MINIMAX_M3:
        case LLM_ARCH_RND1:
        case LLM_ARCH_PADDLEOCR:
        case LLM_ARCH_MIMO2:
        case LLM_ARCH_KIMI_LINEAR:
        case LLM_ARCH_KIMI_K3:
        case LLM_ARCH_STEP35:
        case LLM_ARCH_MISTRAL4:
        case LLM_ARCH_MELLUM:
        case LLM_ARCH_LAGUNA:
            return true;
        default:
            return false;
    }
}

static bool moe_implemented(const llm_arch arch) {
    if (moe_mandatory(arch)) {
        return true;
    }
    switch (arch) {
        case LLM_ARCH_LLAMA:
        case LLM_ARCH_REFACT:
        case LLM_ARCH_MINICPM:
        case LLM_ARCH_GRANITE:
        case LLM_ARCH_GRANITE_MOE:
        case LLM_ARCH_MISTRAL3:
        case LLM_ARCH_LLAMA_EMBED:
            return true;
        default:
            return false;
    }
}

static bool arch_supported(const llm_arch arch) {
    if (arch == LLM_ARCH_CLIP || arch == LLM_ARCH_GPTJ || arch == LLM_ARCH_UNKNOWN) {
        return false; // These models don't have usable implementations.
    }
    if (arch == LLM_ARCH_CHAMELEON) {
        return false; // Only half-implemented and to be removed in the future.
    }
    if (arch == LLM_ARCH_WAVTOKENIZER_DEC) {
        return false; // FIXME CUDA backend crashes.
    }
    if (arch == LLM_ARCH_GEMMA4 || arch == LLM_ARCH_GEMMA4_ASSISTANT) {
        return false; // FIXME @ngxson
    }
    if (arch == LLM_ARCH_GRANITE_SWITCH) {
        return false; // FIXME adapter fixture
    }
    if (arch == LLM_ARCH_LLAMA_EMBED || arch == LLM_ARCH_GEMMA_EMBEDDING || arch == LLM_ARCH_T5ENCODER) {
        return false; // FIXME Embedding (?) models produce inconsistent results.
    }
    if (arch == LLM_ARCH_RWKV6 || arch == LLM_ARCH_RWKV6QWEN2 || arch == LLM_ARCH_RWKV7 || arch == LLM_ARCH_ARWKV7) {
        return false; // FIXME RWKV models hang indefinitely.
    }
    if (arch == LLM_ARCH_BERT || arch == LLM_ARCH_MODERN_BERT || arch == LLM_ARCH_NOMIC_BERT || arch == LLM_ARCH_NOMIC_BERT_MOE ||
            arch == LLM_ARCH_NEO_BERT || arch == LLM_ARCH_JINA_BERT_V2 || arch == LLM_ARCH_JINA_BERT_V3 || arch == LLM_ARCH_EUROBERT) {
        return false; // TODO vocab
    }
    if (arch == LLM_ARCH_PLM) {
        return false; // TODO tensor shapes
    }
    if (arch == LLM_ARCH_DEEPSEEK2OCR) {
        return false;
    }
    // FIXME: these hit scheduler/view-backed-output issues with WebGPU on CI.
#ifdef GGML_USE_WEBGPU
    if (arch == LLM_ARCH_DEEPSEEK32 || arch == LLM_ARCH_GLM_DSA || arch == LLM_ARCH_DOTS3NOTE || arch == LLM_ARCH_QWEN4EXP) {
        return false;
    }
#endif // GGML_USE_WEBGPU

    // FIXME: jamba produces incorrect output (~0.55 NMSE vs CPU) on the HIP
    // backend on RDNA3.5 (gfx1151); the SSM kernels need investigation.
#ifdef GGML_USE_HIP
    if (arch == LLM_ARCH_JAMBA) {
        return false;
    }
#endif // GGML_USE_HIP

    return true;
}

static int save_models(const llm_arch target_arch, const size_t seed, const ggml_log_level log_level, const std::string & dir) {
    struct user_data_t {
        struct {
            ggml_log_callback callback;
            void * user_data;
        } original_logger;
        ggml_log_level min_level; // prints below this log level go to debug log
    };
    user_data_t ud;
    llama_log_get(&ud.original_logger.callback, &ud.original_logger.user_data);
    ud.min_level = log_level;

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const user_data_t * ud = (const user_data_t *) user_data;
        const ggml_log_level level_eff = level >= ud->min_level ? level : GGML_LOG_LEVEL_DEBUG;
        ud->original_logger.callback(level_eff, text, ud->original_logger.user_data);
    }, &ud);

    for (const llm_arch & arch : llm_arch_all()) {
        if (arch == LLM_ARCH_UNKNOWN) {
            continue;
        }
        if (target_arch != LLM_ARCH_UNKNOWN && arch != target_arch) {
            continue;
        }
        if (arch == LLM_ARCH_GEMMA4 || arch == LLM_ARCH_GEMMA4_ASSISTANT) {
            continue; // FIXME: ISWA KV cache initialization needs more fixture params
        }
        if (arch == LLM_ARCH_EAGLE3 || arch == LLM_ARCH_DFLASH) {
            continue;
        }
        for (bool moe : {false, true}) {
            if (moe && !moe_implemented(arch)) {
                continue;
            }
            if (!moe && moe_mandatory(arch)) {
                continue;
            }
            if (!llama_model_saver_supports_arch(arch) || !arch_supported(arch)) {
                LOG_INF("%s: %s model (%s) is unsupported, skipping\n", __func__, llm_arch_name(arch), moe ? "MoE" : "dense");
                continue;
            }
            gguf_context_ptr gguf_ctx = get_gguf_ctx(arch, moe);
            auto model_and_ctx = get_model_and_ctx(
                    gguf_ctx.get(), nullptr, seed, {}, LLAMA_SPLIT_MODE_LAYER, false, arch == LLM_ARCH_GLM5NEXT);
            const std::string path = dir + "/" + llm_arch_name(arch) + (moe ? "-moe.gguf" : "-dense.gguf");
            LOG_INF("%s: Saving %s model (%s) to %s...\n", __func__, llm_arch_name(arch), moe ? "MoE" : "dense", path.c_str());
            llama_model_save_to_file(model_and_ctx.first.get(), path.c_str());
        }
    }
    llama_log_set(ud.original_logger.callback, ud.original_logger.user_data);
    return 0;
}

static int test_backends(const llm_arch target_arch, const size_t seed, const ggml_log_level log_level) {
    struct user_data_t {
        struct {
            ggml_log_callback callback;
            void * user_data;
        } original_logger;
        ggml_log_level min_level; // prints below this log level go to debug log
    };
    user_data_t ud;
    llama_log_get(&ud.original_logger.callback, &ud.original_logger.user_data);
    ud.min_level = log_level;

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const user_data_t * ud = (const user_data_t *) user_data;
        const ggml_log_level level_eff = level >= ud->min_level ? level : GGML_LOG_LEVEL_DEBUG;
        ud->original_logger.callback(level_eff, text, ud->original_logger.user_data);
    }, &ud);

    const std::vector<llama_token> tokens = get_tokens(128, 128, seed);

    struct device_config {
        std::vector<ggml_backend_dev_t> devs;
        std::string                     label;
        llama_split_mode                split_mode;

        device_config(std::vector<ggml_backend_dev_t> devs, std::string name, llama_split_mode split_mode)
            : devs(std::move(devs)), label(std::move(name)), split_mode(split_mode) {}
    };

    std::vector<device_config> dev_configs;
    size_t max_device_label_length = 4;
    {
        std::vector<ggml_backend_dev_t> devices_meta;
        {
            const size_t device_count = ggml_backend_dev_count();
            for (size_t i = 0; i < device_count; i++) {
                ggml_backend_dev_t dev = ggml_backend_dev_get(i);
                dev_configs.emplace_back(std::vector<ggml_backend_dev_t>{dev}, ggml_backend_dev_description(dev), LLAMA_SPLIT_MODE_LAYER);
                max_device_label_length = std::max(max_device_label_length, dev_configs.back().label.length());

                // cpu-based devices cannot be used in tensor split mode
                if (ggml_backend_dev_buffer_type(dev) != ggml_backend_cpu_buffer_type()) {
                    devices_meta.push_back(dev);
                }
            }
        }

        dev_configs.emplace_back(devices_meta, "Meta", LLAMA_SPLIT_MODE_TENSOR);
    }

    size_t max_arch_name_length = 0;
    for (const llm_arch & arch : llm_arch_all()) {
        max_arch_name_length = std::max(max_arch_name_length, strlen(llm_arch_name(arch)));
    }

    const std::string template_header  = std::string("|%" + std::to_string(max_arch_name_length) + "s|%") + std::to_string(max_device_label_length) + "s|%6s|%15s|%9s|\n";
    const std::string template_row_cfg = std::string("|%" + std::to_string(max_arch_name_length) + "s|%") + std::to_string(max_device_label_length) + "s|%6s|";
    const std::string template_row_res = "%15s %10s|%20s|\n";

    bool all_ok = true;
    common_log_flush(common_log_main());
    printf(template_header.c_str(), "Model arch.", "Device", "Config", "NMSE vs. CPU", "Roundtrip");
    printf("|");
    for (size_t i = 0; i < max_arch_name_length; i++) {
        printf("-");
    }
    printf("|");
    for (size_t i = 0; i < max_device_label_length; i++) {
        printf("-");
    }
    printf("|------|---------------|---------|\n");
    for (const llm_arch & arch : llm_arch_all()) {
        if (arch == LLM_ARCH_UNKNOWN) {
            continue;
        }
        if (target_arch != LLM_ARCH_UNKNOWN && arch != target_arch) {
            continue;
        }
        if (arch == LLM_ARCH_GEMMA4 || arch == LLM_ARCH_GEMMA4_ASSISTANT) {
            continue; // FIXME: ISWA KV cache initialization needs more fixture params
        }
        if (arch == LLM_ARCH_EAGLE3 || arch == LLM_ARCH_DFLASH) {
            continue;
        }

        const bool encode = arch == LLM_ARCH_T5 || arch == LLM_ARCH_DREAM || arch == LLM_ARCH_LLADA || arch == LLM_ARCH_LLADA_MOE || arch == LLM_ARCH_RND1;
        for (bool moe : {false, true}) {
            if (moe && !moe_implemented(arch)) {
                continue;
            }
            if (!moe && moe_mandatory(arch)) {
                continue;
            }
            const std::string config_name = moe ? "MoE" : "Dense";
            gguf_context_ptr gguf_ctx = get_gguf_ctx(arch, moe);
            if (arch == LLM_ARCH_BAILINGMOE3) {
                GGML_ASSERT(gguf_remove_key(gguf_ctx.get(), "bailingmoe3.kda.safe_gate") >= 0);
            }
            std::pair<llama_model_ptr, llama_context_ptr> model_and_ctx_cpu;
            std::vector<float> logits_cpu;
            for (device_config & dc : dev_configs) {
                // print test config first; should anything fail during model loading or inference, at least we know which test case caused it
                printf(template_row_cfg.c_str(),
                    llm_arch_name(arch), dc.label.c_str(), config_name.c_str());
                fflush(stdout);

                std::pair<llama_model_ptr, llama_context_ptr> model_and_ctx_dev;
                std::vector<float> logits_dev;
                std::string status_nmse      = "\033[1;33mSKIP\033[0m";
                std::string status_roundtrip = "\033[1;33mSKIP\033[0m";
                char nmse_str[12] = {0};
                bool skip = !arch_supported(arch) || (dc.split_mode == LLAMA_SPLIT_MODE_TENSOR && dc.devs.empty());
                if (!skip) {
                    if (logits_cpu.empty()) {
                        model_and_ctx_cpu = get_model_and_ctx(
                                gguf_ctx.get(), nullptr, seed, {}, LLAMA_SPLIT_MODE_LAYER, encode, arch == LLM_ARCH_GLM5NEXT);
                        if (arch == LLM_ARCH_GLM5NEXT) {
                            test_glm5next_fa_cache_types(model_and_ctx_cpu.first.get());
                            test_glm5next_native_f16_chunk_graph(model_and_ctx_cpu.first.get(), tokens);
                            test_glm5next_native_f16_state(model_and_ctx_cpu.first.get(), tokens);
                            test_glm5next_mtp_driver(model_and_ctx_cpu.first.get(), tokens);
                            test_glm5next_turbo3_cpu_guard(model_and_ctx_cpu.first.get());
                            test_glm5next_failed_state_restore_cleanup(model_and_ctx_cpu.first.get(), tokens);
                            test_glm5next_paged_state_restore(model_and_ctx_cpu.first.get(), model_and_ctx_cpu.second.get(), tokens);
                        }
                        logits_cpu = get_logits(model_and_ctx_cpu.first.get(), model_and_ctx_cpu.second.get(), tokens, encode);
                    }
                    if (dc.split_mode != LLAMA_SPLIT_MODE_TENSOR || llm_arch_supports_sm_tensor(arch)) {
                        model_and_ctx_dev = get_model_and_ctx(
                                gguf_ctx.get(), nullptr, seed, dc.devs, dc.split_mode, encode, arch == LLM_ARCH_GLM5NEXT);
                        if (arch == LLM_ARCH_GLM5NEXT) {
                            bool all_devices_are_gpu = !dc.devs.empty();
                            for (const ggml_backend_dev_t dev : dc.devs) {
                                all_devices_are_gpu = all_devices_are_gpu &&
                                    ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU;
                            }
                            if (all_devices_are_gpu && dc.split_mode == LLAMA_SPLIT_MODE_LAYER) {
                                test_glm5next_turbo3_graph(gguf_ctx.get(), seed, dc.devs, dc.split_mode, tokens);
                            }
                        }
                        logits_dev = get_logits(model_and_ctx_dev.first.get(), model_and_ctx_dev.second.get(), tokens, encode);
                        const double nmse_val = nmse(logits_cpu, logits_dev);
                        snprintf(nmse_str, sizeof(nmse_str), "(%.2e)", nmse_val);
                        status_nmse = "\033[1;32mOK\033[0m";
                        if (nmse_val > 1e-4) {
                            all_ok = false;
                            status_nmse = "\033[1;31mFAIL\033[0m";
                        }
                    }

                    FILE * file = tmpfile(); // Can be null on Windows without administrator privileges.
                    // FIXME: when adding a tensor to a gguf_context a copy is made, this changes the pointer which the meta backend
                    //     in turn uses to map the tensors to their simple equivalents - this is fundamentally incompatible
                    if (file != nullptr && llama_model_saver_supports_arch(arch) && dc.split_mode != LLAMA_SPLIT_MODE_TENSOR) {
                        GGML_ASSERT(model_and_ctx_dev.first && model_and_ctx_dev.second);
                        llama_model_saver ms = llama_model_saver(model_and_ctx_dev.first.get());
                        ms.add_kv_from_model();
                        ms.add_tensors_from_model();
                        ms.save(file);
                        rewind(file);

                        auto model_and_ctx_roundtrip = get_model_and_ctx(
                                nullptr, file, seed, dc.devs, dc.split_mode, encode, arch == LLM_ARCH_GLM5NEXT);
                        const std::vector<float> logits_roundtrip = get_logits(
                            model_and_ctx_roundtrip.first.get(), model_and_ctx_roundtrip.second.get(), tokens, encode);
                        status_roundtrip = "\033[1;32mOK\033[0m";
                        GGML_ASSERT(logits_roundtrip.size() == logits_dev.size());
                        for (size_t i = 0; i < logits_roundtrip.size(); i++) {
                            if (logits_roundtrip[i] != logits_dev[i]) {
                                all_ok = false;
                                status_roundtrip = "\033[1;31mFAIL\033[0m";
                                break;
                            }
                        }
                    }
                }

                // log the results for this test case
                printf(template_row_res.c_str(),
                    status_nmse.c_str(), nmse_str, status_roundtrip.c_str());
            }
        }
    }
    llama_log_set(ud.original_logger.callback, ud.original_logger.user_data);
    return all_ok ? 0 : 1;
}

int main(int argc, char ** argv) {
    // FIXME these tests are disabled in the CI for macOS-latest-cmake-arm64 because they are segfaulting
    common_init();
    std::random_device rd;

    llm_arch arch = LLM_ARCH_UNKNOWN;
    size_t seed = rd();
    ggml_log_level log_level = GGML_LOG_LEVEL_ERROR;
    std::string out;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv);
            return 0;
        }
        if (strcmp(argv[i], "-a") == 0 || strcmp(argv[i], "--arch") == 0) {
            if (i + 1 < argc) {
                const std::string arch_name = argv[++i];
                arch = llm_arch_from_string(arch_name);
                if (arch == LLM_ARCH_UNKNOWN) {
                    LOG_ERR("%s: unkown LLM architecture: %s\n", __func__, arch_name.c_str());
                    return 1;
                }
            } else {
                usage(argv);
                return 1;
            }
        }
        if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--seed") == 0) {
            if (i + 1 < argc) {
                seed = std::stoull(argv[++i]);
            } else {
                usage(argv);
                return 1;
            }
        }
        if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            log_level = GGML_LOG_LEVEL_INFO;
            continue;
        }
        if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--out") == 0) {
            if (i + 1 < argc) {
                out = argv[++i];
            } else {
                usage(argv);
                return 1;
            }
        }
    }
    printf("%s: using seed %zu\n", __func__, seed);

    try {
        if (!out.empty()) {
            return save_models(arch, seed, log_level, out);
        }
        return test_backends(arch, seed, log_level);
    } catch (const std::exception & err) {
        fprintf(stderr, "encountered runtime error: %s\n", err.what());
        return -1;
    }
}
