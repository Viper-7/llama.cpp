#include "tools/server/server-page-ram.h"
#include <chrono>
#include <cstring>
#include <iostream>

int main() {
    const std::string root = "/dev/shm/llama-page-ram-bench-" + std::to_string(getpid());
    setenv("LLAMA_SERVER_PAGE_CACHE_RAM_DIR", root.c_str(), 1);
    setenv("LLAMA_SERVER_PAGE_CACHE_NAMESPACE", std::string(64, 'c').c_str(), 1);
    const size_t size = 256 * 1024 * 1024;
    const auto clock = [] { return std::chrono::steady_clock::now(); };
    server_page_ram_store::json metadata;
    {
        server_page_ram_store store(2 * size);
        server_page_bytes bytes;
        const auto start = clock(); store.allocate(bytes, size); const auto allocated = clock();
        memset(bytes.data(), 17, bytes.size()); const auto filled = clock();
        bytes.seal(); const auto sealed = clock(); metadata = bytes.metadata();
        std::cout << "synthetic_bytes=" << size << " reserve_ms=" << std::chrono::duration<double, std::milli>(allocated-start).count()
                  << " fill_ms=" << std::chrono::duration<double, std::milli>(filled-allocated).count()
                  << " checksum_ms=" << std::chrono::duration<double, std::milli>(sealed-filled).count() << '\n';
    }
    {
        server_page_ram_store store(2 * size); server_page_bytes bytes; size_t budget = 2 * size;
        const auto start = clock(); store.load(bytes, metadata, budget); const auto loaded = clock();
        std::cout << "reopen_mmap_and_checksum_ms=" << std::chrono::duration<double, std::milli>(loaded-start).count() << '\n';
    }
    std::filesystem::remove_all(root);
}
