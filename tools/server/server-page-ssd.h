#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#if defined(__linux__)
#include <sys/statvfs.h>

// Included after server-page-ram's path/lock helpers. No model or GPU calls.
class server_page_ssd_store {
public:
    using json = nlohmann::ordered_json;
    struct completion { std::string key; bool success; uint64_t bytes; };
    explicit server_page_ssd_store(const std::string & identity) {
        const char * root = std::getenv("LLAMA_SERVER_PAGE_CACHE_SSD_DIR");
        if (!root || !*root) return;
        try {
            if (identity.size() != 64 || identity.find_first_not_of("0123456789abcdef") != std::string::npos)
                throw std::runtime_error("invalid SSD namespace");
            limit_bytes = env_bytes("LLAMA_SERVER_PAGE_CACHE_SSD_BYTES", uint64_t(384) << 30);
            reserve_bytes = env_bytes("LLAMA_SERVER_PAGE_CACHE_SSD_RESERVE_BYTES", uint64_t(64) << 30);
            queue_limit = env_bytes("LLAMA_SERVER_PAGE_CACHE_SSD_QUEUE_BYTES", uint64_t(1) << 30);
            if (!limit_bytes || !queue_limit) throw std::runtime_error("invalid SSD budget");
            root_directory = root;
            if (std::filesystem::is_symlink(root_directory)) throw std::runtime_error("SSD root symlink");
            std::filesystem::create_directories(root_directory);
            struct statfs fs{};
            if (statfs(root_directory.c_str(), &fs) || fs.f_type == 0x01021994)
                throw std::runtime_error("SSD root must be disk-backed");
            server_page_ram_lock root_lock(root_directory + "/gc.lock");
            if (!root_lock.owned()) throw std::runtime_error("SSD root lock");
            directory = root_directory + "/" + identity;
            if (std::filesystem::is_symlink(directory)) throw std::runtime_error("SSD namespace symlink");
            std::filesystem::create_directory(directory); chmod(directory.c_str(), 0700);
            owner = std::make_unique<server_page_ram_lock>(directory + "/lock", true);
            if (!owner->owned()) throw std::runtime_error("SSD namespace already owned");
            enabled = true;
            refresh_usage();
            writer = std::thread([this] { work(); });
        } catch (...) { failures++; enabled = false; }
    }
    ~server_page_ssd_store() {
        { std::lock_guard<std::mutex> guard(mutex); stopping = true; }
        ready.notify_all();
        if (writer.joinable()) writer.join();
    }
    bool active() const { return enabled; }
    uint64_t limit() const { return limit_bytes; }
    uint64_t reserve() const { return reserve_bytes; }
    std::string path(const std::string & file) const {
        if (!server_page_ram_page_name(file)) throw std::runtime_error("invalid SSD page name");
        return directory + "/" + file;
    }
    uint64_t used() const {
        if (!enabled) return 0;
        usage_scans++;
        uint64_t result = 0;
        for (const auto & space : std::filesystem::directory_iterator(root_directory)) {
            if (space.is_symlink() || !space.is_directory()) continue;
            const auto name = space.path().filename().string();
            if (name.size() != 64 || name.find_first_not_of("0123456789abcdef") != std::string::npos) continue;
            for (const auto & entry : std::filesystem::directory_iterator(space.path())) {
                if (!entry.is_symlink() && entry.is_regular_file() && (server_page_ram_page_name(entry.path().filename().string()) || entry.path().filename().string().rfind("pending-page-", 0) == 0))
                    { std::error_code error; const auto bytes = entry.file_size(error); if (!error) result += bytes; }
            }
        }
        return result;
    }
    uint64_t free_bytes() const {
        struct statvfs fs{};
        return enabled && !statvfs(root_directory.c_str(), &fs) ? uint64_t(fs.f_bavail) * fs.f_frsize : 0;
    }
    // Under disk pressure, inactive incompatible model namespaces yield first.
    // Both namespace owners and live canonical leases are fenced.
    bool reclaim_dormant(uint64_t wanted) {
        if (!enabled || can_fit(wanted)) return false;
        server_page_ram_lock root_lock(root_directory + "/gc.lock");
        if (!root_lock.owned()) return false;
        std::vector<std::pair<uint64_t, std::filesystem::path>> candidates;
        try {
            for (const auto & space : std::filesystem::directory_iterator(root_directory)) {
                const auto name = space.path().filename().string();
                if (space.is_symlink() || !space.is_directory() || space.path() == directory ||
                        name.size() != 64 || name.find_first_not_of("0123456789abcdef") != std::string::npos) continue;
                const auto manifest = space.path() / "graph.json";
                if (!std::filesystem::is_regular_file(manifest) || std::filesystem::is_symlink(manifest) ||
                        std::filesystem::file_size(manifest) > 64 * 1024 * 1024) continue;
                std::ifstream input(manifest); const auto graph = json::parse(input);
                uint64_t access = 0;
                for (const auto & page : graph.at("pages")) access = std::max(access, page.value("last_access_ms", uint64_t(0)));
                candidates.emplace_back(access, space.path());
            }
            std::sort(candidates.begin(), candidates.end());
            for (const auto & candidate : candidates) {
                if (can_fit(wanted)) break;
                server_page_ram_lock owner(candidate.second.string() + "/lock", true);
                if (!owner.owned()) continue;
                std::unique_ptr<server_page_ram_lock> ram_owner;
                if (const char * ram = std::getenv("LLAMA_SERVER_PAGE_CACHE_RAM_DIR")) {
                    const auto lock = std::filesystem::path(ram) / candidate.second.filename() / "lock";
                    if (std::filesystem::exists(lock)) {
                        ram_owner = std::make_unique<server_page_ram_lock>(lock.string(), true);
                        if (!ram_owner->owned()) continue;
                    }
                }
                std::ifstream input(candidate.second / "graph.json"); const auto graph = json::parse(input);
                if (graph.value("version", 0) != 1 || graph.value("namespace", std::string()) != candidate.second.filename().string()) continue;
                const auto now = uint64_t(std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count());
                bool protected_lease = false;
                const auto leases = graph.value("metadata", json::object()).value("response_leases", json::array());
                if (!leases.is_array() || leases.size() > 4096) continue;
                for (const auto & lease : leases) {
                    if (lease.value("expires", uint64_t(0)) > now ||
                            lease.value("pending", json::object()).value("expires", uint64_t(0)) > now) protected_lease = true;
                }
                for (const auto & page : graph.at("pages")) if (page.value("release_after_ms", uint64_t(0)) > now) protected_lease = true;
                if (protected_lease) continue;
                std::vector<std::filesystem::path> files; uint64_t reclaimed = 0; bool recognized = true;
                for (const auto & entry : std::filesystem::directory_iterator(candidate.second)) {
                    const auto file = entry.path().filename().string();
                    if (entry.is_symlink() || !entry.is_regular_file() || (file != "lock" && file != "graph.json" &&
                            file != "graph.pending" && !server_page_ram_page_name(file) && !server_page_ram_page_name(file.size() > 8 ? file.substr(8) : std::string()))) { recognized = false; break; }
                    files.push_back(entry.path()); reclaimed += entry.file_size();
                }
                if (!recognized) continue;
                // Preserve the ownership inode: a later opener cannot race a
                // lock replacement. The empty namespace can be reused safely.
                auto empty = graph; empty["pages"] = json::array(); empty["metadata"] = json::object();
                const auto pending = candidate.second / "graph.pending";
                { std::ofstream output(pending); output << empty.dump(); output.flush(); if (!output) continue; }
                sync_file(pending.string()); std::filesystem::rename(pending, candidate.second / "graph.json");
                for (const auto & file : files) if (server_page_ram_page_name(file.filename().string()) ||
                        server_page_ram_page_name(file.filename().string().size() > 8 ? file.filename().string().substr(8) : std::string())) std::filesystem::remove(file);
                namespace_evictions++; namespace_reclaimed_bytes += reclaimed;
            }
        } catch (...) { failures++; }
        return can_fit(wanted);
    }

    bool queue_space(uint64_t bytes) const {
        std::lock_guard<std::mutex> guard(mutex);
        return pending_keys.size() < 8 && (pending_keys.empty() || bytes <= queue_limit - std::min(queue_limit, pending_bytes));
    }
    bool degraded() const { return write_degraded.load(); }
    bool can_fit(uint64_t bytes) const {
        if (!enabled) return false;
        std::lock_guard<std::mutex> guard(mutex);
        const auto occupied = used(), free = free_bytes();
        return occupied <= limit_bytes && bytes <= limit_bytes - occupied &&
            pending_bytes <= limit_bytes - occupied - bytes &&
            free >= reserve_bytes && bytes <= free - reserve_bytes && pending_bytes <= free - reserve_bytes - bytes;
    }
    bool submit(const std::string & key, const std::string & ram_directory, const json & parts) {
        if (!enabled) return false;
        job request; request.key = key;
        uint64_t bytes = 0;
        try {
            for (const auto & part : parts) {
                if (!part.at("bytes").get<uint64_t>()) continue;
                const auto file = part.at("file").get<std::string>();
                if (!server_page_ram_page_name(file)) return false;
                const int fd = open((ram_directory + "/" + file).c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
                if (fd < 0) return false;
                request.parts.push_back({std::shared_ptr<int>(new int(fd), [](int * p) { close(*p); delete p; }), part});
                if (!std::filesystem::exists(path(file))) bytes += part.at("bytes").get<uint64_t>();
            }
            std::lock_guard<std::mutex> guard(mutex);
            if (stopping || pending_keys.size() >= 8 || pending_keys.count(key) || (bytes > queue_limit - std::min(queue_limit, pending_bytes) && !pending_keys.empty())) return false;
            const auto occupied = used(), free = free_bytes();
            if (occupied > limit_bytes || bytes > limit_bytes - occupied || pending_bytes > limit_bytes - occupied - bytes ||
                    free < reserve_bytes || bytes > free - reserve_bytes || pending_bytes > free - reserve_bytes - bytes) return false;
            request.reserved = bytes; pending_bytes += bytes; pending_keys.insert(key);
            for (const auto & part : request.parts) pending_parts[key].insert(part.descriptor.at("file").get<std::string>());
            queued.push_back(std::move(request)); ready.notify_one(); return true;
        } catch (...) { failures++; return false; }
    }
    std::vector<completion> poll() {
        std::lock_guard<std::mutex> guard(mutex);
        std::vector<completion> result;
        while (!completed.empty()) {
            pending_keys.erase(completed.front().key); pending_parts.erase(completed.front().key);
            result.push_back(completed.front()); completed.pop_front();
        }
        return result;
    }
    bool pending(const std::string & key) const {
        std::lock_guard<std::mutex> guard(mutex); return pending_keys.count(key);
    }
    json stats() const {
        std::lock_guard<std::mutex> guard(mutex);
        return {{"enabled", enabled}, {"limit_bytes", limit_bytes}, {"used_bytes", sampled_used.load()},
            {"filesystem_free_bytes", sampled_free.load()},
            {"usage_sample_ms", sampled_at_ms.load()}, {"usage_scans", usage_scans.load()},
            {"cleanup_scans", cleanup_scans.load()}, {"cleanup_skips", cleanup_skips.load()}, {"filesystem_reserve_bytes", reserve_bytes},
            {"namespace_evictions", namespace_evictions.load()}, {"namespace_reclaimed_bytes", namespace_reclaimed_bytes.load()},
            {"pending_bytes", pending_bytes}, {"pending_copies", pending_keys.size()},
            {"degraded", write_degraded.load()}, {"written_bytes", written_bytes.load()}, {"write_us", write_us.load()}, {"write_failures", failures.load()}};
    }
    bool write_graph(json graph) {
        if (!enabled) return true;
        std::lock_guard<std::mutex> guard(mutex);
        if (stopping) return false;
        next_graph = std::move(graph); ready.notify_one(); return true;
    }
    bool flush_graph(const json & graph) {
        if (!enabled) return true;
        try {
            const std::string temporary = directory + "/graph.pending";
            { std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
                output << graph.dump(); output.flush(); if (!output) throw std::runtime_error("SSD graph write"); }
            sync_file(temporary); std::filesystem::rename(temporary, directory + "/graph.json"); sync_directory();
            // Diagnostics contain counters only and never hold up generation.
            try {
                json telemetry = graph.value("metadata", json::object()).value("tiers", json::object());
                refresh_usage();
                const auto current = stats();
                for (auto field = current.begin(); field != current.end(); ++field) telemetry[field.key()] = field.value();
                telemetry["namespace"] = std::filesystem::path(directory).filename().string();
                telemetry["pid"] = getpid();
                telemetry["updated_at"] = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
                const char * configured = std::getenv("LLAMA_SERVER_PAGE_CACHE_TIER_STATS_PATH");
                const std::string destination = configured && *configured ? configured : root_directory + "/tier-stats.json";
                { std::ofstream output(destination + ".pending"); output << telemetry.dump(); output.flush();
                    if (!output) throw std::runtime_error("tier telemetry write"); }
                std::filesystem::rename(destination + ".pending", destination);
            } catch (...) { /* telemetry cannot invalidate a committed graph */ }
            return true;
        } catch (...) { failures++; return false; }
    }
    json read_graph() const {
        try {
            const auto file = directory + "/graph.json";
            if (!enabled || !std::filesystem::exists(file) || std::filesystem::file_size(file) > 64 * 1024 * 1024) return json();
            std::ifstream input(file); return json::parse(input);
        } catch (...) { return json(); }
    }
    void clean(const std::set<std::string> & live) {
        if (!enabled) return;
        std::lock_guard<std::mutex> guard(mutex);
        std::set<std::string> protected_files;
        for (const auto & part : pending_parts) protected_files.insert(part.second.begin(), part.second.end());
        if (clean_valid && live == clean_live && protected_files == clean_protected) { cleanup_skips++; return; }
        try {
            cleanup_scans++;
            for (const auto & entry : std::filesystem::directory_iterator(directory)) {
                const auto file = entry.path().filename().string();
                if (!entry.is_symlink() && server_page_ram_page_name(file) && !live.count(file) && !protected_files.count(file)) std::filesystem::remove(entry.path());
                if (!entry.is_symlink() && file.rfind("pending-page-", 0) == 0 && !protected_files.count(file.substr(8))) std::filesystem::remove(entry.path());
            }
            clean_live = live; clean_protected = protected_files; clean_valid = true;
        } catch (...) { clean_valid = false; failures++; }
    }
private:
    // Telemetry may lag; admission always uses fresh used()/free_bytes().
    void refresh_usage() {
        const auto occupied = used(), free = free_bytes();
        sampled_used = occupied; sampled_free = free;
        sampled_at_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    }
    struct part { std::shared_ptr<int> fd; json descriptor; };
    struct job { std::string key; std::vector<part> parts; uint64_t reserved = 0; };
    static uint64_t env_bytes(const char * name, uint64_t fallback) {
        const char * value = std::getenv(name);
        if (!value) return fallback;
        if (!*value || std::string(value).find_first_not_of("0123456789") != std::string::npos) throw std::runtime_error("invalid SSD byte configuration");
        return std::stoull(value);
    }
    static void sync_file(const std::string & name) {
        const int fd = open(name.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
        if (fd < 0) throw std::runtime_error("SSD sync open");
        const int result = fsync(fd); close(fd);
        if (result) throw std::runtime_error("SSD sync");
    }
    void sync_directory() const {
        const int fd = open(directory.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        if (fd < 0) throw std::runtime_error("SSD directory open");
        const int result = fsync(fd); close(fd);
        if (result) throw std::runtime_error("SSD directory sync");
    }
    uint64_t copy(const part & source) {
        const auto file = source.descriptor.at("file").get<std::string>();
        const auto size = source.descriptor.at("bytes").get<uint64_t>();
        const auto expected = source.descriptor.at("checksum").get<uint64_t>();
        struct stat st{};
        if (fstat(*source.fd, &st) || !S_ISREG(st.st_mode) || uint64_t(st.st_size) != size) throw std::runtime_error("SSD source changed");
        const int existing = open(path(file).c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
        if (existing >= 0) {
            struct stat saved{};
            bool valid = !fstat(existing, &saved) && S_ISREG(saved.st_mode) && uint64_t(saved.st_size) == size;
            auto hash = std::unique_ptr<XXH3_state_t, decltype(&XXH3_freeState)>(XXH3_createState(), XXH3_freeState);
            XXH3_64bits_reset(hash.get());
            std::vector<uint8_t> buffer(8 * 1024 * 1024);
            for (uint64_t offset = 0; valid && offset < size;) {
                const auto bytes = std::min<uint64_t>(buffer.size(), size - offset);
                valid = pread(existing, buffer.data(), bytes, offset) == ssize_t(bytes);
                if (valid) XXH3_64bits_update(hash.get(), buffer.data(), bytes);
                offset += bytes;
            }
            valid &= XXH3_64bits_digest(hash.get()) == expected;
            close(existing);
            if (valid) return 0;
            unlink(path(file).c_str()); // RAM source remains authoritative.
        }
        const std::string temporary = directory + "/pending-" + file;
        int fd = open(temporary.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
        if (fd < 0) throw std::runtime_error("SSD destination open");
        auto hash = std::unique_ptr<XXH3_state_t, decltype(&XXH3_freeState)>(XXH3_createState(), XXH3_freeState);
        XXH3_64bits_reset(hash.get());
        try {
            {
                server_page_ram_lock root_lock(root_directory + "/gc.lock");
                const auto occupied = used(), free = free_bytes();
                if (!root_lock.owned() || occupied > limit_bytes || size > limit_bytes - occupied ||
                        free < reserve_bytes || size > free - reserve_bytes || posix_fallocate(fd, 0, size))
                    throw std::runtime_error("SSD allocation/budget");
            }
            std::vector<uint8_t> buffer(8 * 1024 * 1024);
            uint64_t offset = 0;
            while (offset < size) {
                if (stopping) throw std::runtime_error("SSD writer stopping");
                const size_t bytes = std::min<uint64_t>(buffer.size(), size - offset);
                if (pread(*source.fd, buffer.data(), bytes, offset) != ssize_t(bytes)) throw std::runtime_error("SSD source read");
                XXH3_64bits_update(hash.get(), buffer.data(), bytes);
                size_t sent = 0;
                while (sent < bytes) {
                    const auto n = pwrite(fd, buffer.data() + sent, bytes - sent, offset + sent);
                    if (n <= 0) throw std::runtime_error("SSD write");
                    sent += size_t(n);
                }
                offset += bytes;
            }
            if (XXH3_64bits_digest(hash.get()) != expected || fsync(fd)) throw std::runtime_error("SSD checksum/sync");
            posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
            close(fd); fd = -1; std::filesystem::rename(temporary, path(file)); sync_directory(); return size;
        } catch (...) { if (fd >= 0) close(fd); unlink(temporary.c_str()); throw; }
    }
    void work() {
        while (true) {
            job request; json graph;
            { std::unique_lock<std::mutex> guard(mutex); ready.wait(guard, [&] { return stopping || !queued.empty() || !next_graph.is_null(); });
                if (stopping) break;
                if (!next_graph.is_null()) { graph = std::move(next_graph); next_graph = json(); }
                else { request = std::move(queued.front()); queued.pop_front(); }
            }
            if (!graph.is_null()) { flush_graph(graph); continue; }
            const auto start = std::chrono::steady_clock::now();
            bool success = true; uint64_t bytes = 0;
            try { for (const auto & source : request.parts) bytes += copy(source); }
            catch (...) { success = false; failures++; }
            write_degraded = !success;
            written_bytes += bytes;
            write_us += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count();
            { std::lock_guard<std::mutex> guard(mutex); pending_bytes -= request.reserved;
                completed.push_back({request.key, success, bytes}); }
        }
    }
    std::string directory, root_directory;
    std::unique_ptr<server_page_ram_lock> owner;
    uint64_t limit_bytes = 0, reserve_bytes = 0, queue_limit = 0, pending_bytes = 0;
    bool enabled = false;
    std::atomic<bool> stopping{false}, write_degraded{false};
    std::atomic<uint64_t> written_bytes{0}, write_us{0}, failures{0}, namespace_evictions{0}, namespace_reclaimed_bytes{0};
    std::atomic<uint64_t> sampled_used{0}, sampled_free{0}, sampled_at_ms{0}, cleanup_scans{0}, cleanup_skips{0};
    mutable std::atomic<uint64_t> usage_scans{0};
    bool clean_valid = false;
    std::set<std::string> clean_live, clean_protected;
    mutable std::mutex mutex;
    std::condition_variable ready;
    std::thread writer;
    std::deque<job> queued;
    std::deque<completion> completed;
    std::set<std::string> pending_keys;
    std::map<std::string, std::set<std::string>> pending_parts;
    json next_graph;
};

#else
class server_page_ssd_store {
public:
    using json = nlohmann::ordered_json;
    struct completion { std::string key; bool success; uint64_t bytes; };
    explicit server_page_ssd_store(const std::string &) {}
    bool active() const { return false; }
    uint64_t limit() const { return 0; }
    uint64_t reserve() const { return 0; }
    uint64_t used() const { return 0; }
    uint64_t free_bytes() const { return 0; }
    bool can_fit(uint64_t) const { return false; }
    bool queue_space(uint64_t) const { return false; }
    bool degraded() const { return false; }
    bool reclaim_dormant(uint64_t) { return false; }
    bool submit(const std::string &, const std::string &, const json &) { return false; }
    bool pending(const std::string &) const { return false; }
    std::vector<completion> poll() { return {}; }
    json stats() const { return {{"enabled", false}}; }
    bool write_graph(const json &) { return true; }
    json read_graph() const { return json(); }
    void clean(const std::set<std::string> &) {}
    std::string path(const std::string &) const { throw std::runtime_error("SSD cache requires Linux"); }
};
#endif
