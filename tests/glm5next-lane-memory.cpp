// Metadata-only sizing: never loads weights or dispatches inference.
#include "llama.h"
#include "llama-cpp.h"
#include "llama-context.h"
#include "ggml-backend.h"
#include <iostream>
#include <stdexcept>
#include <vector>

static void dump(const char * role, int batch, const llama_context & ctx) {
    std::cout << "{\"role\":\"" << role << "\",\"batch\":" << batch << ",\"buffers\":[";
    bool first = true;
    for (const auto & entry : ctx.memory_breakdown()) {
        if (!first) std::cout << ",";
        first = false;
        const auto & m = entry.second;
        std::cout << "{\"name\":\"" << ggml_backend_buft_name(entry.first)
                  << "\",\"model\":" << m.model << ",\"context\":" << m.context
                  << ",\"compute\":" << m.compute << "}";
    }
    std::cout << "]}" << std::endl;
}
int main(int argc, char ** argv) {
    if (argc != 3) { std::cerr << "Usage: lane-memory GGUF PINNED_BACKEND_DIRECTORY\n"; return 2; }
    ggml_backend_load_all_from_path(argv[2]);
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.no_alloc = true;
    mp.load_mode = LLAMA_LOAD_MODE_NONE;
    mp.load_mtp = false;
    mp.n_gpu_layers = 47;
    std::vector<float> split(llama_max_devices(), 0);
    const float weights[] = {1.2f,.85f,.9f,1.05f,.8f,1.2f};
    std::copy(weights, weights+6, split.begin());
    mp.tensor_split = split.data();
    auto * device = ggml_backend_dev_by_name("ROCm4");
    if (!device) throw std::runtime_error("Pinned ROCm backend/device is unavailable");
    llama_model_tensor_buft_override overrides[] = {
        {"(blk\\.45\\..*|token_embd\\.weight)", ggml_backend_dev_buffer_type(device)},
        {nullptr, nullptr}
    };
    mp.tensor_buft_overrides = overrides;
    llama_model_ptr model(llama_model_load_from_file(argv[1], mp));
    if (!model) throw std::runtime_error("Metadata model creation failed");
    auto cp = llama_context_default_params();
    cp.n_ctx = 307200;
    cp.n_seq_max = 3;
    cp.kv_unified = true;
    cp.n_batch = 4096;
    cp.n_ubatch = 1024;
    cp.n_threads = cp.n_threads_batch = 48;
    cp.type_k = cp.type_v = GGML_TYPE_F16;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    llama_context_ptr owner(llama_init_from_model(model.get(), cp));
    if (!owner) throw std::runtime_error("Metadata context creation failed");
    dump("owner", 1024, *owner);
    for (const uint32_t batch : {64,128,256,512,1024}) {
        auto lane = owner->create_execution_lane(batch);
        dump("private_lane", batch, *lane);
    }
    owner.reset();
    model.reset();
    llama_backend_free();
}
