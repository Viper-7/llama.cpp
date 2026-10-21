#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <hip/hip_runtime_api.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

constexpr int skip_code = 77;

struct device_ref {
    ggml_backend_dev_t dev = nullptr;
    int physical = -1;
};

struct tensor_batch {
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buffer;
    std::vector<ggml_tensor *> tensors;
};

void hip_check(hipError_t status, const char * operation) {
    if (status != hipSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(status));
    }
}

int physical_device(const char * name) {
    int device = -1;
    int consumed = 0;
    if (name != nullptr && std::sscanf(name, "ROCm%d%n", &device, &consumed) == 1 &&
        consumed == (int) std::strlen(name)) {
        return device;
    }
    return -1;
}

std::vector<uint8_t> make_pattern(size_t size, size_t tensor_id) {
    std::vector<uint8_t> data(size);
    for (size_t i = 0; i < size; ++i) {
        data[i] = (uint8_t) ((tensor_id * 131 + i * 17 + (i >> 8) * 29) & 0xff);
    }
    return data;
}

tensor_batch make_tensors(ggml_backend_t backend, const std::vector<size_t> & sizes, const char * prefix) {
    const size_t metadata_size = sizes.size() * ggml_tensor_overhead() + 1024;
    tensor_batch result;
    result.ctx.reset(ggml_init({
        /* .mem_size   = */ metadata_size,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    }));
    if (!result.ctx) {
        throw std::runtime_error("failed to create tensor metadata context");
    }

    result.tensors.reserve(sizes.size());
    for (size_t i = 0; i < sizes.size(); ++i) {
        ggml_tensor * tensor = ggml_new_tensor_1d(result.ctx.get(), GGML_TYPE_I8, sizes[i]);
        ggml_format_name(tensor, "%s_%zu", prefix, i);
        result.tensors.push_back(tensor);
    }

    result.buffer.reset(ggml_backend_alloc_ctx_tensors(result.ctx.get(), backend));
    if (!result.buffer) {
        throw std::runtime_error("failed to allocate GPU tensor batch");
    }
    return result;
}

std::pair<device_ref, device_ref> find_no_peer_pair() {
    std::vector<device_ref> devices;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const int physical = physical_device(ggml_backend_dev_name(dev));
        if (physical >= 0) {
            devices.push_back({dev, physical});
        }
    }

    for (const device_ref & src : devices) {
        for (const device_ref & dst : devices) {
            if (src.physical == dst.physical) {
                continue;
            }
            int can_access_peer = 0;
            hip_check(hipDeviceCanAccessPeer(&can_access_peer, src.physical, dst.physical),
                    "hipDeviceCanAccessPeer");
            if (!can_access_peer) {
                return {src, dst};
            }
        }
    }
    return {};
}

void verify_tensors(
        const tensor_batch & dst, const std::vector<std::vector<uint8_t>> & patterns) {
    for (size_t i = 0; i < dst.tensors.size(); ++i) {
        std::vector<uint8_t> actual(patterns[i].size());
        ggml_backend_tensor_get(dst.tensors[i], actual.data(), 0, actual.size());
        if (actual != patterns[i]) {
            size_t mismatch = 0;
            while (mismatch < actual.size() && actual[mismatch] == patterns[i][mismatch]) {
                ++mismatch;
            }
            throw std::runtime_error("tensor " + std::to_string(i) + " mismatch at byte " +
                    std::to_string(mismatch));
        }
    }
}

} // namespace

int main() try {
    setenv("GGML_HIP_NO_PEER_STAGING", "1", 1);
    setenv("GGML_HIP_NO_PEER_STAGING_MAX_BYTES", "1048576", 1);
    ggml_backend_load_all();

    const auto [src_ref, dst_ref] = find_no_peer_pair();
    if (src_ref.dev == nullptr || dst_ref.dev == nullptr) {
        std::fprintf(stderr, "SKIP: two ROCm devices without peer access are required\n");
        return skip_code;
    }

    ggml_backend_ptr src_backend(ggml_backend_dev_init(src_ref.dev, nullptr));
    ggml_backend_ptr dst_backend(ggml_backend_dev_init(dst_ref.dev, nullptr));
    if (!src_backend || !dst_backend) {
        throw std::runtime_error("failed to initialize ROCm backends");
    }

    constexpr size_t sequential_count = 12;
    constexpr size_t concurrent_count = 16;
    const char * concurrent_env = std::getenv("GGML_HIP_STAGING_TEST_CONCURRENT");
    const bool run_concurrent = concurrent_env != nullptr && std::atoi(concurrent_env) != 0;
    const size_t total_count = sequential_count + (run_concurrent ? concurrent_count : 0);
    const size_t alternating_sizes[] = {32768 + 17, 196608 + 29, 65536 + 7};

    std::vector<size_t> sizes;
    std::vector<std::vector<uint8_t>> patterns;
    sizes.reserve(total_count);
    patterns.reserve(total_count);
    for (size_t i = 0; i < total_count; ++i) {
        sizes.push_back(alternating_sizes[i % 3]);
        patterns.push_back(make_pattern(sizes.back(), i));
    }

    tensor_batch src = make_tensors(src_backend.get(), sizes, "staging_src");
    tensor_batch dst = make_tensors(dst_backend.get(), sizes, "staging_dst");
    for (size_t i = 0; i < total_count; ++i) {
        ggml_backend_tensor_set(src.tensors[i], patterns[i].data(), 0, patterns[i].size());
        ggml_backend_tensor_memset(dst.tensors[i], 0xa5, 0, sizes[i]);
    }

    // Four staging slots exist. Submit twelve copies without a destination sync so
    // slot reuse and forced draining are both exercised.
    for (size_t i = 0; i < sequential_count; ++i) {
        ggml_backend_tensor_copy_async(src_backend.get(), dst_backend.get(), src.tensors[i], dst.tensors[i]);
    }

    if (run_concurrent) {
        // Optional stress mode: backend contexts are not generally promised to be
        // thread-safe, so this is intentionally not part of the default CTest gate.
        std::atomic<bool> start{false};
        std::exception_ptr thread_error;
        std::mutex error_mutex;
        auto submit = [&](size_t parity) {
            try {
                while (!start.load(std::memory_order_acquire)) {
                    std::this_thread::yield();
                }
                for (size_t i = sequential_count + parity; i < total_count; i += 2) {
                    ggml_backend_tensor_copy_async(
                            src_backend.get(), dst_backend.get(), src.tensors[i], dst.tensors[i]);
                }
            } catch (...) {
                std::lock_guard<std::mutex> lock(error_mutex);
                if (!thread_error) {
                    thread_error = std::current_exception();
                }
            }
        };

        std::thread first(submit, 0);
        std::thread second(submit, 1);
        start.store(true, std::memory_order_release);
        first.join();
        second.join();
        if (thread_error) {
            std::rethrow_exception(thread_error);
        }
    }

    ggml_backend_synchronize(dst_backend.get());
    verify_tensors(dst, patterns);

    std::printf("PASS: %zu exact async copies ROCm%d -> ROCm%d (%zu sequential, %zu concurrent)\n",
            total_count, src_ref.physical, dst_ref.physical, sequential_count,
            run_concurrent ? concurrent_count : 0);
    return 0;
} catch (const std::exception & error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
}
