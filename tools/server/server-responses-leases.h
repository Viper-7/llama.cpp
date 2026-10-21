#pragma once

#include <cstdint>
#include <cstddef>
#include <utility>
#include <map>
#include <set>
#include <stdexcept>
#include <string>

// Called only by the inference queue owner. Graph ancestry/endpoint validation
// belongs to the page cache; this ledger controls revision and capacity fences.
class server_responses_leases {
public:
    using page_sizes = std::map<std::string, size_t>;
    struct candidate {
        std::string revision;
        uint64_t generation = 0;
        uint64_t expires = 0;
        page_sizes pages;
        bool endpoint_complete = false;
    };
    struct scope {
        std::string head;
        uint64_t generation = 0;
        uint64_t expires = 0;
        uint64_t touched = 0;
        page_sizes pages;
        candidate pending;
    };
    explicit server_responses_leases(size_t byte_limit, size_t scope_limit = 64)
        : byte_limit(byte_limit), scope_limit(scope_limit) {}

    bool prepare(const std::string & id, const std::string & revision,
                 const std::string & expected, uint64_t now, uint64_t ttl, uint64_t expected_generation = 0) {
        if (id.empty() || revision.empty() || !ttl || ttl > 7200000 || now > UINT64_MAX - ttl) return false;
        prune(now);
        auto found = scopes.find(id);
        size_t active = 0;
        for (const auto & entry : scopes)
            active += entry.second.expires > now || entry.second.pending.expires > now;
        if (found == scopes.end()) {
            if (!expected.empty() || scopes.size() >= 4096 || active >= scope_limit) return false;
            found = scopes.emplace(id, scope{}).first;
        }
        auto & state = found->second;
        if (state.head != expected || state.generation != expected_generation) return false;
        if (!state.pending.revision.empty() && state.pending.expires > now) {
            // Retry only the same immutable candidate, never overwrite a peer.
            return state.pending.revision == revision && state.pending.generation == state.generation;
        }
        if (state.expires <= now && state.pending.expires <= now && active >= scope_limit) return false;
        state.touched = now;
        state.pending = candidate{revision, state.generation, now + ttl, {}, false};
        return true;
    }

    bool observe(const std::string & id, const std::string & revision, const page_sizes & pages,
                 bool endpoint_complete, uint64_t now) {
        auto found = scopes.find(id);
        if (found == scopes.end()) return false;
        auto & state = found->second;
        if (!valid_candidate(state, revision, now) || pages.empty()) return false;
        for (const auto & page : pages) if (page.first.empty() || page.second == 0) return false;
        const auto previous = state.pending;
        state.pending.pages = pages;
        state.pending.endpoint_complete = endpoint_complete;
        try {
            if (protected_bytes(now) <= byte_limit) return true;
        } catch (...) {
            state.pending = previous;
            throw;
        }
        state.pending = previous; // capacity failure preserves both protections
        return false;
    }

    bool commit(const std::string & id, const std::string & revision,
                const std::string & expected, uint64_t now, uint64_t ttl) {
        auto found = scopes.find(id);
        if (found == scopes.end() || !ttl || ttl > 7200000 || now > UINT64_MAX - ttl) return false;
        auto & state = found->second;
        if (state.head != expected || !valid_candidate(state, revision, now) ||
                !state.pending.endpoint_complete || state.pending.pages.empty()) return false;
        state.head = revision;
        state.pages = std::move(state.pending.pages);
        state.expires = now + ttl;
        state.touched = now;
        state.pending = {};
        return true;
    }

    bool abandon(const std::string & id, const std::string & revision) {
        auto found = scopes.find(id);
        if (found == scopes.end() || found->second.pending.revision != revision) return false;
        found->second.pending = {};
        return true;
    }

    bool release(const std::string & id, const std::string & expected, uint64_t now = 0) {
        auto found = scopes.find(id);
        if (found == scopes.end()) return false;
        auto & state = found->second;
        const auto & latest = state.pending.revision.empty() ? state.head : state.pending.revision;
        if (expected != latest) return false; // stale release cannot remove newer ownership
        ++state.generation;
        state.touched = now;
        state.pages.clear();
        state.expires = 0;
        state.pending = {}; // late observe/commit cannot resurrect released protection
        return true;
    }

    bool protects(const std::string & page, uint64_t now) const {
        for (const auto & entry : scopes) {
            const auto & state = entry.second;
            if (state.expires > now && state.pages.count(page)) return true;
            if (state.pending.expires > now && state.pending.generation == state.generation &&
                    state.pending.pages.count(page)) return true;
        }
        return false;
    }

    size_t protected_bytes(uint64_t now) const {
        page_sizes unique;
        for (const auto & entry : scopes) {
            const auto & state = entry.second;
            if (state.expires > now) merge(unique, state.pages);
            if (state.pending.expires > now && state.pending.generation == state.generation) merge(unique, state.pending.pages);
        }
        size_t bytes = 0;
        for (const auto & page : unique) {
            if (page.second > SIZE_MAX - bytes) throw std::overflow_error("protected page size overflow");
            bytes += page.second;
        }
        return bytes;
    }

    void invalidate(const std::string & page, uint64_t now) {
        for (auto & entry : scopes) {
            auto & state = entry.second;
            if (state.pages.count(page) || state.pending.pages.count(page)) {
                ++state.generation;
                state.touched = now;
                state.expires = 0;
                state.pages.clear();
                state.pending = {};
            }
        }
    }

    bool restore(std::map<std::string, scope> recovered, const page_sizes & available, uint64_t now) {
        if (recovered.size() > 4096 || now > UINT64_MAX - 7200000) return false;
        bool intact = true;
        for (auto & entry : recovered) {
            auto & state = entry.second;
            if (entry.first.empty() || state.expires > now + 7200000 || state.pending.expires > now + 7200000)
                return false;
            auto valid = [&](const page_sizes & pages) {
                for (const auto & page : pages) {
                    const auto found = available.find(page.first);
                    if (found == available.end() || found->second != page.second || !page.second) return false;
                }
                return true;
            };
            if (!valid(state.pages) || !valid(state.pending.pages) ||
                    (state.pending.generation != state.generation && !state.pending.revision.empty())) {
                ++state.generation;
                state.expires = 0;
                state.pages.clear();
                state.pending = {};
                intact = false;
            }
        }
        size_t active = 0;
        for (const auto & entry : recovered)
            active += entry.second.expires > now || entry.second.pending.expires > now;
        if (active > scope_limit) return false;
        const auto previous = scopes;
        scopes = std::move(recovered);
        try {
            if (protected_bytes(now) > byte_limit) { scopes = previous; return false; }
        } catch (...) { scopes = previous; return false; }
        prune(now);
        return intact;
    }

    void prune(uint64_t now) {
        for (auto it = scopes.begin(); it != scopes.end();) {
            const auto & state = it->second;
            if (state.expires <= now && state.pending.expires <= now &&
                    now >= state.touched && now - state.touched >= 86400000) it = scopes.erase(it);
            else ++it;
        }
    }

    const std::map<std::string, scope> & snapshot() const { return scopes; }
    size_t limit() const { return byte_limit; }
private:
    static bool valid_candidate(const scope & state, const std::string & revision, uint64_t now) {
        return state.pending.revision == revision && state.pending.expires > now &&
               state.pending.generation == state.generation;
    }
    static void merge(page_sizes & target, const page_sizes & source) {
        for (const auto & page : source) {
            const auto found = target.find(page.first);
            if (found != target.end() && found->second != page.second)
                throw std::invalid_argument("shared protected page size changed");
            target.emplace(page);
        }
    }
    size_t byte_limit;
    size_t scope_limit;
    std::map<std::string, scope> scopes;
};
