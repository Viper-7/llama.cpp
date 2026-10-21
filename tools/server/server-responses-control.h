#pragma once

#include <chrono>
#include <functional>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

// HTTP preparation and the native queue share this CPU-only lifecycle ledger.
// A request is released only after preparation AND all queued/active tasks end.
class server_responses_registry {
public:
    struct entry {
        bool preparing = false;
        bool cancelled = false;
        std::set<int> tasks;
        std::chrono::steady_clock::time_point updated = std::chrono::steady_clock::now();
    };
    static bool valid_id(const std::string & id) {
        if (id.size() < 38 || id.size() > 128 || id.compare(0, 4, "job_") != 0 || id[36] != ':') return false;
        for (size_t i = 4; i < 36; ++i) if (!((id[i] >= '0' && id[i] <= '9') || (id[i] >= 'a' && id[i] <= 'f'))) return false;
        for (size_t i = 37; i < id.size(); ++i) if (id[i] < '0' || id[i] > '9') return false;
        return true;
    }
    void begin(const std::string & id) {
        if (!valid_id(id)) throw std::invalid_argument("invalid serverless request identity");
        std::lock_guard<std::mutex> lock(mutex);
        prune();
        if (requests.count(id)) throw std::invalid_argument("serverless request identity already used or cancelled");
        if (requests.size() >= 4096) throw std::runtime_error("serverless request registry admission limit");
        requests[id].preparing = true;
    }
    void prepared(const std::string & id) {
        std::lock_guard<std::mutex> lock(mutex);
        auto found = requests.find(id);
        if (found != requests.end()) { found->second.preparing = false; found->second.updated = std::chrono::steady_clock::now(); }
    }
    bool attach(const std::string & id, int task) {
        std::lock_guard<std::mutex> lock(mutex);
        auto found = requests.find(id);
        if (found == requests.end() || found->second.cancelled) return false;
        found->second.tasks.insert(task);
        task_owners[task] = id;
        return true;
    }
    bool post(const std::string & id, const std::vector<int> & tasks, const std::function<void()> & submit) {
        std::lock_guard<std::mutex> lock(mutex);
        auto found = requests.find(id);
        if (found == requests.end() || found->second.cancelled) return false;
        for (int task : tasks) { found->second.tasks.insert(task); task_owners[task] = id; }
        try {
            submit(); // cancellation cannot overtake registration before queue admission
        } catch (...) {
            for (int task : tasks) { found->second.tasks.erase(task); task_owners.erase(task); }
            throw;
        }
        found->second.preparing = false;
        return true;
    }
    std::vector<int> targets(const std::string & id) const {
        std::lock_guard<std::mutex> lock(mutex);
        auto found = requests.find(id);
        if (found == requests.end() || found->second.preparing || found->second.cancelled) return {};
        return {found->second.tasks.begin(), found->second.tasks.end()};
    }
    std::vector<int> cancel(const std::string & id) {
        if (!valid_id(id)) throw std::invalid_argument("invalid cancellation request identity");
        std::lock_guard<std::mutex> lock(mutex);
        prune();
        if (!requests.count(id) && requests.size() >= 4096) throw std::runtime_error("serverless request registry admission limit");
        auto & request = requests[id]; // tombstone prevents cancel-before-admission from starting later
        request.cancelled = true;
        request.updated = std::chrono::steady_clock::now();
        return {request.tasks.begin(), request.tasks.end()};
    }
    bool owns(int task) const {
        std::lock_guard<std::mutex> lock(mutex);
        return task_owners.count(task) != 0;
    }
    void finish(int task) {
        std::lock_guard<std::mutex> lock(mutex);
        auto owner = task_owners.find(task);
        if (owner == task_owners.end()) return;
        auto & request = requests.at(owner->second);
        request.tasks.erase(task);
        request.updated = std::chrono::steady_clock::now();
        task_owners.erase(owner);
    }
    bool released(const std::string & id) const {
        std::lock_guard<std::mutex> lock(mutex);
        auto found = requests.find(id);
        return found != requests.end() && !found->second.preparing && found->second.tasks.empty();
    }
private:
    void prune() { // caller holds mutex; live entries/tombstones inside the retry window survive
        const auto cutoff = std::chrono::steady_clock::now() - std::chrono::hours(2);
        for (auto it = requests.begin(); it != requests.end();) {
            if (!it->second.preparing && it->second.tasks.empty() && it->second.updated < cutoff) it = requests.erase(it);
            else ++it;
        }
    }
    mutable std::mutex mutex;
    std::map<std::string, entry> requests;
    std::map<int, std::string> task_owners;
};

struct server_responses_preparing_guard {
    server_responses_registry & registry;
    std::string id;
    server_responses_preparing_guard(server_responses_registry & registry, std::string id)
        : registry(registry), id(std::move(id)) { registry.begin(this->id); }
    ~server_responses_preparing_guard() { registry.prepared(id); }
};
