#pragma once

#include "vendor/nlohmann/json.hpp"
#define XXH_INLINE_ALL
#include "vendor/hash/xxhash/xxhash.h"
#undef XXH_INLINE_ALL
#include <algorithm>
#include <cstdint>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(__linux__)
#include <fcntl.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/statfs.h>
#include <unistd.h>
#endif

static inline bool server_page_ram_page_name(const std::string & file) {
    return file.rfind("page-", 0) == 0 && file.size() > 5 && file.substr(5).find_first_not_of("0123456789-") == std::string::npos;
}

// Byte storage has one owner: heap when disabled, shared tmpfs pages when enabled.
class server_page_bytes {
public:
    server_page_bytes() = default;
    server_page_bytes(const server_page_bytes &) = delete;
    server_page_bytes & operator=(const server_page_bytes &) = delete;
    server_page_bytes(server_page_bytes && other) noexcept { swap(other); }
    server_page_bytes & operator=(server_page_bytes && other) noexcept { reset(); swap(other); return *this; }
    ~server_page_bytes() { reset(); }
    size_t size() const { return n; }
    size_t resident_size() const { return disk ? 0 : n; }
    bool on_disk() const { return disk; }
    bool backed() const { return ssd_backup; }
    void mark_backed() { if (n) ssd_backup = true; }
    void forget_backup() { if (!disk) ssd_backup = false; }
    void demote(const std::string & path) {
#if defined(__linux__)
        if (!ssd_backup || !n) throw std::runtime_error("SSD backup not ready");
        struct stat st{};
        if (lstat(path.c_str(), &st) || !S_ISREG(st.st_mode) || uint64_t(st.st_size) != n)
            throw std::runtime_error("SSD backup disappeared");
        if (mapped) munmap(mapped, n);
        mapped = nullptr; heap.clear(); disk = true; disk_path = path;

#else
        (void) path; throw std::runtime_error("SSD cache requires Linux");
#endif
    }
    void defer_disk(const std::string & path, size_t size, uint64_t checksum) {
#if defined(__linux__)
        if (n || !size) throw std::runtime_error("invalid cold descriptor");
        struct stat st{};
        if (lstat(path.c_str(), &st) || !S_ISREG(st.st_mode) || uint64_t(st.st_size) != size)
            throw std::runtime_error("missing SSD checkpoint");
        disk = true; ssd_backup = true; disk_path = path; n = size; digest = checksum;
        filename = std::filesystem::path(path).filename().string();

#else
        (void) path; (void) size; (void) checksum; throw std::runtime_error("SSD cache requires Linux");
#endif
    }
    void release_disk_mapping() const {
#if defined(__linux__)
        if (!disk || !mapped) return;
        madvise(mapped, n, MADV_DONTNEED); munmap(mapped, n); mapped = nullptr;
        const int fd = open(disk_path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
        if (fd >= 0) { posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED); close(fd); }

#endif
    }
    bool empty() const { return n == 0; }
    uint8_t * data() { prepare_disk(); return mapped ? mapped : heap.data(); }
    const uint8_t * data() const { prepare_disk(); return mapped ? mapped : heap.data(); }
    const uint8_t * begin() const { return data(); }
    const uint8_t * end() const { return data() + n; }
    bool operator==(const std::vector<uint8_t> & value) const {
        return n == value.size() && std::equal(value.begin(), value.end(), begin());
    }
    void resize(size_t size) { if (mapped) throw std::runtime_error("mapped bytes are immutable in size"); heap.resize(size); n = size; }
    void map(const std::string & path, size_t size, bool create) {
#if defined(__linux__)
        if (!size || mapped || n) throw std::runtime_error("invalid page allocation");
        const int fd = open(path.c_str(), (create ? O_RDWR | O_CREAT | O_EXCL : O_RDONLY) | O_CLOEXEC | O_NOFOLLOW, 0600);
        if (fd < 0) throw std::runtime_error("cannot open RAM page");
        struct stat st{};
        int error = 0;
        if (create) error = posix_fallocate(fd, 0, size);
        if (fstat(fd, &st) || !S_ISREG(st.st_mode) || st.st_size != (off_t) size) error = 1;
        void * ptr = error ? MAP_FAILED : mmap(nullptr, size, create ? PROT_READ | PROT_WRITE : PROT_READ, MAP_SHARED, fd, 0);
        close(fd);
        if (ptr == MAP_FAILED) { if (create) unlink(path.c_str()); throw std::runtime_error("cannot reserve RAM page"); }
        mapped = static_cast<uint8_t *>(ptr); n = size; filename = std::filesystem::path(path).filename().string();
#else
        (void) path; (void) size; (void) create;
        throw std::runtime_error("RAM pages require Linux tmpfs");
#endif
    }
    uint64_t checksum() const { return XXH3_64bits(data(), n); }
    void seal() { digest = checksum(); }
    bool validate_and_seal(uint64_t expected) { digest = checksum(); return digest == expected; }
    nlohmann::ordered_json metadata() const { return {{"file", filename}, {"bytes", n}, {"checksum", digest}, {"tier", disk ? "ssd" : "ram"}, {"ssd_backup", ssd_backup}}; }
private:
    void prepare_disk() const {
#if defined(__linux__)
        if (!disk || mapped) return;
        const int fd = open(disk_path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
        struct stat st{};
        if (fd < 0) throw std::runtime_error("SSD restore open");
        void * ptr = fstat(fd, &st) || !S_ISREG(st.st_mode) || uint64_t(st.st_size) != n
            ? MAP_FAILED : mmap(nullptr, n, PROT_READ, MAP_SHARED, fd, 0);
        close(fd);
        if (ptr == MAP_FAILED) throw std::runtime_error("SSD restore map");
        mapped = static_cast<uint8_t *>(ptr); madvise(mapped, n, MADV_SEQUENTIAL);
        auto hash = std::unique_ptr<XXH3_state_t, decltype(&XXH3_freeState)>(XXH3_createState(), XXH3_freeState);
        if (!hash) { release_disk_mapping(); throw std::bad_alloc(); }
        XXH3_64bits_reset(hash.get());
        for (size_t offset = 0; offset < n;) {
            const size_t bytes = std::min<size_t>(8 * 1024 * 1024, n - offset);
            XXH3_64bits_update(hash.get(), mapped + offset, bytes);
            // Drop verification RSS incrementally; subsequent tensor uploads
            // fault their own ranges through the ordinary OS page cache.
            madvise(mapped + offset, bytes, MADV_DONTNEED);
            offset += bytes;
        }
        if (XXH3_64bits_digest(hash.get()) != digest) {
            release_disk_mapping(); throw std::runtime_error("SSD restore checksum");
        }

#endif
    }
    void reset() {
#if defined(__linux__)
        if (mapped) munmap(mapped, n);
#endif
        mapped = nullptr; n = 0; heap.clear(); filename.clear(); digest = 0; disk = false; ssd_backup = false; disk_path.clear();
    }
    void swap(server_page_bytes & other) {
        std::swap(mapped, other.mapped); std::swap(n, other.n); heap.swap(other.heap);
        filename.swap(other.filename); std::swap(digest, other.digest);
        std::swap(disk, other.disk); std::swap(ssd_backup, other.ssd_backup); disk_path.swap(other.disk_path);
    }
    mutable uint8_t * mapped = nullptr;
    bool disk = false, ssd_backup = false;
    std::string disk_path;
    size_t n = 0;
    std::vector<uint8_t> heap;
    std::string filename;
    uint64_t digest = 0;
};

#if defined(__linux__)
class server_page_ram_lock {
public:
    explicit server_page_ram_lock(const std::string & path, bool nonblocking = false) {
        fd = open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600);
        if (fd >= 0 && flock(fd, LOCK_EX | (nonblocking ? LOCK_NB : 0))) { close(fd); fd = -1; }
    }
    ~server_page_ram_lock() { if (fd >= 0) close(fd); }
    bool owned() const { return fd >= 0; }
    server_page_ram_lock(const server_page_ram_lock &) = delete;
    server_page_ram_lock & operator=(const server_page_ram_lock &) = delete;
private:
    int fd = -1;
};
#endif

#include "server-page-ssd.h"

class server_page_ram_store {
public:
    using json = nlohmann::ordered_json;
    uint64_t recovered = 0, rejected = 0, commits = 0, failures = 0, orphan_files = 0, stale_namespaces = 0, stale_bytes = 0;
    static size_t effective_limit(size_t requested) {
#if defined(__linux__)
        const char * root = std::getenv("LLAMA_SERVER_PAGE_CACHE_RAM_DIR");
        if (!root || !*root) return requested;
        std::filesystem::path path(root);
        std::error_code error;
        while (!path.empty() && !std::filesystem::exists(path, error)) path = path.parent_path();
        struct statfs fs{};
        if (!path.empty() && statfs(path.c_str(), &fs) == 0 && fs.f_type == 0x01021994) {
            const uint64_t total = uint64_t(fs.f_blocks) * uint64_t(fs.f_bsize);
            const uint64_t margin = std::min<uint64_t>(128 * 1024 * 1024, total / 64);
            return size_t(std::min<uint64_t>(requested, total - margin));
        }
#endif
        return requested;
    }
    explicit server_page_ram_store(size_t limit) : limit(effective_limit(limit)), serial(uint64_t(std::chrono::steady_clock::now().time_since_epoch().count())) {
        const char * root = std::getenv("LLAMA_SERVER_PAGE_CACHE_RAM_DIR");
        const char * identity = std::getenv("LLAMA_SERVER_PAGE_CACHE_NAMESPACE");
        if (!root || !*root) return;
        try {
            if (!identity || std::string(identity).size() != 64 || std::string(identity).find_first_not_of("0123456789abcdef") != std::string::npos)
                throw std::runtime_error("RAM cache namespace must be a 64 digit lowercase SHA256 identity");
#if defined(__linux__)
            std::filesystem::create_directories(root);
            struct statfs fs{};
            if (statfs(root, &fs) || fs.f_type != 0x01021994) throw std::runtime_error("RAM cache root is not tmpfs");
            root_directory = root;
            server_page_ram_lock root_lock(root_directory + "/gc.lock");
            if (!root_lock.owned()) throw std::runtime_error("cannot lock RAM root");
            namespace_id = identity;
            directory = std::string(root) + "/" + identity;
            if (std::filesystem::is_symlink(std::filesystem::symlink_status(directory))) throw std::runtime_error("RAM namespace is a symlink");
            std::filesystem::create_directory(directory);
            chmod(directory.c_str(), 0700);
            lock_fd = open((directory + "/lock").c_str(), O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600);
            if (lock_fd < 0 || flock(lock_fd, LOCK_EX | LOCK_NB)) throw std::runtime_error("RAM cache namespace already owned");
            enabled = true;
            ssd = std::make_unique<server_page_ssd_store>(namespace_id);
            collect_stale_namespaces();
#else
            throw std::runtime_error("RAM cache requires Linux");
#endif
        } catch (...) { failures++; enabled = false; }
    }
    ~server_page_ram_store() {
        ssd.reset(); // Finish the bounded writer before releasing namespace ownership.
#if defined(__linux__)
        if (lock_fd >= 0) close(lock_fd);
#endif
    }
    bool active() const { return enabled; }
    server_page_ssd_store * disk_store() const { return ssd.get(); }
    const std::string & ram_directory() const { return directory; }
    void allocate(server_page_bytes & bytes, size_t size) {
        if (!enabled) { bytes.resize(size); return; }
#if defined(__linux__)
        if (size > limit) throw std::runtime_error("RAM page exceeds root budget");
        server_page_ram_lock root_lock(root_directory + "/gc.lock");
        if (!root_lock.owned()) throw std::runtime_error("cannot lock RAM root");
        size_t occupied = 0;
        for (const auto & space : std::filesystem::directory_iterator(root_directory)) {
            if (!managed_namespace(space)) continue;
            for (const auto & entry : std::filesystem::directory_iterator(space.path())) {
                if (entry.is_symlink() || !entry.is_regular_file() || !server_page_ram_page_name(entry.path().filename().string())) continue;
                const auto bytes = entry.file_size();
                if (bytes > limit - std::min(limit, occupied)) throw std::runtime_error("RAM root exceeds budget");
                occupied += bytes;
            }
        }
        if (size > limit - occupied) throw std::runtime_error("RAM root allocation exceeds budget");
        const std::string path = directory + "/page-" + std::to_string(getpid()) + "-" + std::to_string(++serial);
        bytes.map(path, size, true);
#endif
    }
    json read() {
        if (!enabled) return json::array();
        try {
            const auto path = directory + "/graph.json";
            if (!std::filesystem::exists(path)) {
                if (!ssd || !ssd->active()) return json::array();
                const auto graph = ssd->read_graph();
                if (!graph.is_object() || graph.value("version", 0) != 1 || graph.value("namespace", std::string()) != namespace_id ||
                        graph.value("checksum_algorithm", std::string()) != "xxh3-64" || graph.value("limit", size_t(0)) != limit)
                    return json::array();
                metadata_cur = graph.value("metadata", json::object());
                return graph.at("pages");
            }
            if (std::filesystem::file_size(path) > std::min<size_t>(limit, 64 * 1024 * 1024)) throw std::runtime_error("RAM graph too large");
            std::ifstream input(path);
            const auto graph = json::parse(input);
            if (graph.at("version") != 1 || graph.at("checksum_algorithm") != "xxh3-64" || graph.at("namespace") != namespace_id || graph.at("limit").get<size_t>() != limit) throw std::runtime_error("RAM graph contract changed");
            metadata_cur = graph.value("metadata", json::object());
            return graph.at("pages");
        } catch (...) { rejected++; return json::array(); }
    }
    void load(server_page_bytes & bytes, const json & entry, size_t & budget) {
        const size_t size = entry.at("bytes").get<size_t>();

        const std::string file = entry.at("file").get<std::string>();
        if (!size) { if (!file.empty()) throw std::runtime_error("invalid empty page"); return; }
        if (!server_page_ram_page_name(file))
            throw std::runtime_error("invalid RAM page name");
        const bool cold = entry.value("tier", std::string("ram")) == "ssd" ||
            !std::filesystem::exists(directory + "/" + file);
        if (cold) {
            if (!ssd || !ssd->active() || size > ssd->limit()) throw std::runtime_error("SSD tier unavailable");
            bytes.defer_disk(ssd->path(file), size, entry.at("checksum").get<uint64_t>());
            return;
        }
        if (size > budget) throw std::runtime_error("RAM graph exceeds limit");
        bytes.map(directory + "/" + file, size, false);
        if (!bytes.validate_and_seal(entry.at("checksum").get<uint64_t>())) throw std::runtime_error("corrupt RAM page");
        if (entry.value("ssd_backup", false) && ssd && ssd->active() && std::filesystem::exists(ssd->path(file))) bytes.mark_backed();
        budget -= size;
    }
    const json & metadata() const { return metadata_cur; }
    bool write(json pages, const json & metadata = json::object()) {
        if (!enabled) return true;
        try {
#if defined(__linux__)
            server_page_ram_lock root_lock(root_directory + "/gc.lock");
            if (!root_lock.owned()) throw std::runtime_error("cannot lock RAM root");
#endif
            json graph = {{"version", 1}, {"checksum_algorithm", "xxh3-64"}, {"namespace", namespace_id}, {"limit", limit}, {"pages", json::array()}, {"metadata", metadata}};
            graph["pages"] = std::move(pages);
            const std::string temporary = directory + "/graph.pending";
            { std::ofstream output(temporary, std::ios::binary | std::ios::trunc); output << graph.dump(); output.flush(); if (!output) throw std::runtime_error("RAM graph write failed"); }
            std::filesystem::rename(temporary, directory + "/graph.json");
            commits++;
            std::set<std::string> live, disk_live;
            for (const auto & page : graph.at("pages")) for (const char * part : {"target", "draft", "reference"}) {
                const auto & descriptor = page.at(part);
                const auto file = descriptor.at("file").get<std::string>();
                if (!file.empty()) {
                    disk_live.insert(file);
                    if (descriptor.value("tier", std::string("ram")) != "ssd") live.insert(file);
                }
            }
            if (ssd && ssd->active()) ssd->write_graph(std::move(graph));
            for (const auto & entry : std::filesystem::directory_iterator(directory)) {
                const auto file = entry.path().filename().string();
                if (server_page_ram_page_name(file) && !live.count(file)) { std::filesystem::remove(entry.path()); orphan_files++; }
            }
            if (ssd) ssd->clean(disk_live);
            return true;
        } catch (...) { failures++; return false; }
    }
private:
#if defined(__linux__)
    static bool managed_namespace(const std::filesystem::directory_entry & entry) {
        const auto name = entry.path().filename().string();
        return !entry.is_symlink() && entry.is_directory() && name.size() == 64 &&
            name.find_first_not_of("0123456789abcdef") == std::string::npos;
    }
    static bool dormant_lease_active(const std::filesystem::path & space) {
        bool has_protection = false;
        try {
            const auto path = space / "graph.json";
            if (!std::filesystem::exists(path) || std::filesystem::file_size(path) > 64 * 1024 * 1024) return false;
            std::ifstream input(path);
            auto graph = json::parse(input);
            if (graph.at("version") != 1 || graph.at("namespace") != space.filename().string()) return false;
            const auto leases = graph.value("metadata", json::object()).value("response_leases", json::array());
            if (!leases.is_array() || leases.size() > 4096) return false;
            const auto now = uint64_t(std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count());
            std::set<std::string> keys;
            for (const auto & page : graph.at("pages")) keys.insert(page.at("key").get<std::string>());
            auto protected_pages = [&](const json & claim) {
                const auto expires = claim.at("expires").get<uint64_t>();
                const auto & pages = claim.at("pages");
                if (expires <= now || expires > now + 7200000 || !pages.is_object() || pages.empty()) return false;
                for (auto page = pages.begin(); page != pages.end(); ++page)
                    if (!keys.count(page.key()) || page.value().get<size_t>() == 0) return false;
                return true;
            };
            std::set<std::string> keep;
            auto retain = [&](const json & claim) {
                if (!protected_pages(claim)) return;
                has_protection = true;
                for (auto page = claim.at("pages").begin(); page != claim.at("pages").end(); ++page)
                    keep.insert(page.key());
            };
            for (const auto & lease : leases) {
                if (!lease.at("head").get<std::string>().empty()) retain(lease);
                const auto & pending = lease.at("pending");
                if (!pending.at("revision").get<std::string>().empty() &&
                        pending.at("generation") == lease.at("generation")) retain(pending);
            }
            if (!has_protection) return false;
            // A dormant protected namespace must not shelter unrelated stale
            // entries from pressure GC. Retain grace pages and their ancestry.
            std::map<std::string, json> nodes;
            for (const auto & page : graph.at("pages")) {
                const auto key = page.at("key").get<std::string>();
                nodes.emplace(key, page);
                const auto grace = page.value("release_after_ms", uint64_t(0));
                if (grace > now && grace <= now + 30000) keep.insert(key);
            }
            std::vector<std::string> ancestry(keep.begin(), keep.end());
            for (size_t i = 0; i < ancestry.size(); ++i) {
                const auto found = nodes.find(ancestry[i]);
                if (found == nodes.end()) throw std::runtime_error("protected dormant ancestor missing");
                const auto parent = found->second.at("parent").get<std::string>();
                if (!parent.empty() && keep.insert(parent).second) ancestry.push_back(parent);
            }
            if (keep.size() == nodes.size()) return true;
            json retained = json::array();
            std::set<std::string> files;
            for (const auto & page : graph.at("pages")) if (keep.count(page.at("key").get<std::string>())) {
                retained.push_back(page);
                for (const char * part : {"target", "draft", "reference"}) {
                    const auto file = page.at(part).at("file").get<std::string>();
                    if (!file.empty()) files.insert(file);
                }
            }
            graph["pages"] = std::move(retained);
            const auto pending_path = space / "graph.pending";
            { std::ofstream output(pending_path); output << graph.dump(); output.flush();
              if (!output) throw std::runtime_error("dormant graph publication failed"); }
            std::filesystem::rename(pending_path, path);
            for (const auto & entry : std::filesystem::directory_iterator(space)) {
                const auto file = entry.path().filename().string();
                if (server_page_ram_page_name(file) && !files.count(file)) std::filesystem::remove(entry.path());
            }
            return true;
        } catch (...) { return has_protection; } // failed trimming cannot delete protected ownership
    }
    void collect_stale_namespaces() {
        for (const auto & space : std::filesystem::directory_iterator(root_directory)) {
            if (!managed_namespace(space) || space.path().filename() == namespace_id) continue;
            server_page_ram_lock owner(space.path().string() + "/lock", true);
            if (!owner.owned() || dormant_lease_active(space.path())) continue;
            std::vector<std::filesystem::path> files;
            uint64_t bytes = 0;
            bool recognized = true;
            for (const auto & entry : std::filesystem::directory_iterator(space.path())) {
                const auto file = entry.path().filename().string();
                if (entry.is_symlink() || !entry.is_regular_file() ||
                    (file != "lock" && file != "graph.json" && file != "graph.pending" && !server_page_ram_page_name(file))) {
                    recognized = false; break;
                }
                files.push_back(entry.path()); bytes += entry.file_size();
            }
            if (!recognized) continue;
            for (const auto & file : files) std::filesystem::remove(file);
            std::filesystem::remove(space.path()); stale_namespaces++; stale_bytes += bytes;
        }
    }
#endif
    size_t limit;
    std::string directory;
    std::string root_directory;
    std::string namespace_id;
    bool enabled = false;
    int lock_fd = -1;
    uint64_t serial = 0;
    json metadata_cur = json::object();
    std::unique_ptr<server_page_ssd_store> ssd;
};
