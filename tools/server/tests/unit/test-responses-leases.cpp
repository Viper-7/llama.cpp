#include "tools/server/server-responses-leases.h"
#include <cassert>

int main() {
    server_responses_leases leases(100);
    assert(leases.prepare("primary", "first", "", 10, 1000));
    assert(!leases.prepare("primary", "peer", "", 10, 1000));
    assert(leases.observe("primary", "first", {{"root", 30}, {"old-tail", 20}}, false, 10));
    assert(!leases.commit("primary", "first", "", 10, 1000));
    assert(leases.observe("primary", "first", {{"root", 30}, {"old-tail", 20}}, true, 10));
    assert(leases.commit("primary", "first", "", 10, 1000));
    assert(leases.protects("root", 11) && leases.protected_bytes(11) == 50);
    assert(leases.prepare("primary", "replacement", "first", 20, 1000));
    assert(!leases.observe("primary", "replacement", {{"large", 80}}, true, 20));
    assert(leases.protects("old-tail", 20)); // failed candidate never unpins old head
    assert(leases.observe("primary", "replacement", {{"root", 30}, {"new-tail", 40}}, true, 20));
    assert(leases.protected_bytes(20) == 90); // shared ancestry charged once
    assert(!leases.commit("primary", "replacement", "wrong", 20, 1000));
    assert(!leases.release("primary", "first")); // accepted replacement fences stale release
    assert(leases.commit("primary", "replacement", "first", 20, 1000));
    assert(!leases.protects("old-tail", 21) && leases.protects("new-tail", 21));
    assert(leases.prepare("primary", "late", "replacement", 30, 1000));
    assert(leases.observe("primary", "late", {{"late-tail", 10}}, true, 30));
    assert(leases.release("primary", "late"));
    assert(!leases.commit("primary", "late", "replacement", 31, 1000));
    assert(!leases.protects("root", 31) && leases.protected_bytes(31) == 0);
    assert(!leases.prepare("primary", "late", "replacement", 40, 10)); // stale preparation ticket
    assert(leases.prepare("primary", "renew", "replacement", 40, 10, 1));
    assert(leases.observe("primary", "renew", {{"root", 30}}, true, 40));
    assert(!leases.commit("primary", "renew", "replacement", 50, 1000));
    assert(!leases.protects("root", 50)); // expiry includes candidates
    assert(leases.prepare("other", "first", "", 60, 1000));
    assert(leases.observe("other", "first", {{"root", 30}}, true, 60));
    assert(leases.commit("other", "first", "", 60, 1000));
    assert(!leases.release("primary", "first"));
    assert(leases.protects("root", 61)); // owner scope remains independent
    assert(!leases.prepare("bad", "x", "", UINT64_MAX, 1));
    server_responses_leases recovered(100);
    assert(recovered.restore(leases.snapshot(), {{"root", 30}}, 61));
    assert(recovered.protects("root", 61));
    recovered.invalidate("root", 62);
    assert(!recovered.protects("root", 62));
    server_responses_leases missing(100);
    assert(!missing.restore(leases.snapshot(), {}, 61));
    assert(!missing.protects("root", 61));
    assert(missing.snapshot().at("other").head == "first");
    server_responses_leases bounded(100, 1);
    assert(bounded.prepare("one", "first", "", 1, 100));
    assert(!bounded.prepare("two", "first", "", 1, 100));
    assert(bounded.release("one", "first", 2));
    assert(bounded.prepare("two", "first", "", 2, 100)); // inactive tombstone not an active scope
    bounded.prune(86400002);
    assert(bounded.snapshot().empty());
}
