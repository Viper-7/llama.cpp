// Deferred fake GPU verifies scheduler input ownership without allocating a GPU.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include <cstring>
#include <deque>
#include <functional>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <string>

struct stream {
    std::deque<std::function<void()>> work;
    size_t submitted = 0, completed = 0;
    int host_waits = 0, uploads = 0, waits = 0, copies = 0, unready_copies = 0;
    void push(std::function<void()> f) { work.push_back(std::move(f)); ++submitted; }
    void drain(size_t end) {
        while (completed < end) {
            GGML_ASSERT(!work.empty());
            auto f = std::move(work.front()); work.pop_front(); f(); ++completed;
        }
    }
};
struct marker { stream * source = nullptr; size_t end = 0; };
static stream & state(ggml_backend_t b) { return *static_cast<stream *>(b->context); }
struct fake {
    stream s;
    ggml_backend_device device{};
    ggml_backend_buffer_type buft{};
    ggml_backend backend{};
    explicit fake(bool events) {
        buft.device = &device;
        buft.iface.get_name = [](ggml_backend_buffer_type_t) { return "deferred"; };
        buft.iface.get_alignment = [](ggml_backend_buffer_type_t) { return size_t(64); };
        buft.iface.alloc_buffer = [](ggml_backend_buffer_type_t t, size_t size) {
            auto b = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
            b->buft = t; return b;
        };
        device.context = &buft;
        device.iface.get_name = [](ggml_backend_dev_t) { return "deferred"; };
        device.iface.get_type = [](ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_GPU; };
        device.iface.get_buffer_type = [](ggml_backend_dev_t d) { return static_cast<ggml_backend_buffer_type_t>(d->context); };
        device.iface.supports_buft = [](ggml_backend_dev_t d, ggml_backend_buffer_type_t t) { return d->context == t; };
        device.iface.supports_op = [](ggml_backend_dev_t, const ggml_tensor * t) { return t->op == GGML_OP_ADD || t->op == GGML_OP_NONE; };
        if (events) {
            device.iface.event_new = [](ggml_backend_dev_t d) { return new ggml_backend_event{d, new marker}; };
            device.iface.event_free = [](ggml_backend_dev_t, ggml_backend_event_t e) { delete static_cast<marker *>(e->context); delete e; };
            device.iface.event_synchronize = [](ggml_backend_dev_t, ggml_backend_event_t e) {
                auto m = *static_cast<marker *>(e->context);
                if (m.source) { ++m.source->host_waits; m.source->drain(m.end); }
            };
        }
        backend.device = &device; backend.context = &s;
        backend.iface.get_name = [](ggml_backend_t) { return "deferred"; };
        backend.iface.synchronize = [](ggml_backend_t b) { auto & s = state(b); ++s.host_waits; s.drain(s.submitted); };
        backend.iface.set_tensor_async = [](ggml_backend_t b, ggml_tensor * t, const void * data, size_t off, size_t size) {
            ++state(b).uploads;
            state(b).push([=] { std::memcpy(static_cast<char *>(t->data) + off, data, size); });
        };
        backend.iface.cpy_tensor_async = [](ggml_backend_t src, ggml_backend_t dst, const ggml_tensor * a, ggml_tensor * b) {
            if (src->iface.get_name != dst->iface.get_name) return false;
            auto & source = state(src);
            ++source.copies;
            if (source.completed < source.submitted) ++source.unready_copies;
            // Model HIP peer writes executing on the SOURCE stream.
            source.push([a, b] { std::memcpy(b->data, a->data, ggml_nbytes(a)); });
            const size_t end = source.submitted;
            state(dst).push([&source, end] { source.drain(end); });
            return true;
        };
        backend.iface.graph_compute = [](ggml_backend_t b, ggml_cgraph * g) {
            // Capture node addresses; the scheduler's split view is temporary.
            std::vector<ggml_tensor *> nodes(ggml_graph_nodes(g), ggml_graph_nodes(g) + ggml_graph_n_nodes(g));
            state(b).push([nodes] {
                for (auto * t : nodes) {
                    if (t->op == GGML_OP_NONE) continue;
                    GGML_ASSERT(t->op == GGML_OP_ADD);
                    const auto * a = static_cast<float *>(t->src[0]->data);
                    const auto * b = static_cast<float *>(t->src[1]->data);
                    auto * out = static_cast<float *>(t->data);
                    for (int64_t i = 0; i < ggml_nelements(t); ++i) out[i] = a[i] + b[i];
                }
            });
            return GGML_STATUS_SUCCESS;
        };
        if (events) {
            backend.iface.event_record = [](ggml_backend_t b, ggml_backend_event_t e) { *static_cast<marker *>(e->context) = {&state(b), state(b).submitted}; };
            backend.iface.event_wait = [](ggml_backend_t b, ggml_backend_event_t e) {
                auto m = *static_cast<marker *>(e->context);
                ++state(b).waits;
                state(b).push([m] { if (m.source) m.source->drain(m.end); });
            };
        }
    }
};

static void run(bool owned, bool events, bool waterfall = false, size_t ready_min = 0) {
    const auto threshold = std::to_string(ready_min);
    setenv("GGML_SCHED_READY_COPY_MIN_BYTES", threshold.c_str(), 1);
    fake gpu(events), peer(events);
    auto cpu = ggml_backend_cpu_init();
    ggml_backend_t backends[] = {&gpu.backend, &peer.backend, cpu};
    auto sched = ggml_backend_sched_new(backends, nullptr, 3, 64, false, false);
    ggml_backend_sched_enable_transport_profile(sched);
    if (owned) ggml_backend_sched_set_owned_inputs(sched, true);
    auto ctx = ggml_init({ggml_tensor_overhead()*8 + ggml_graph_overhead_custom(64, false), nullptr, true});
    auto x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    auto y = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_set_input(x); ggml_set_input(y);
    auto inputs = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    auto z = ggml_add(ctx, x, y);
    ggml_backend_sched_set_tensor_backend(sched, z, &gpu.backend);
    if (waterfall) {
        z = ggml_add(ctx, z, x);
        ggml_backend_sched_set_tensor_backend(sched, z, &peer.backend);
        z = ggml_add(ctx, z, y);
    }
    ggml_set_output(z);
    auto g = ggml_new_graph_custom(ctx, 64, false); ggml_build_forward_expand(g, z);
    ggml_backend_sched_set_tensor_backend(sched, z, &gpu.backend);
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched, g));
    GGML_ASSERT(ggml_backend_sched_get_n_copies(sched) == 1);
    for (int round = 0; round < 3; ++round) {
        const float a[] = {float(round), 2, 3, 4}, b[] = {5, 6, 7, 8};
        ggml_backend_tensor_set(x, a, 0, sizeof(a)); ggml_backend_tensor_set(y, b, 0, sizeof(b));
        gpu.s.host_waits = gpu.s.uploads = gpu.s.copies = gpu.s.unready_copies = 0;
        peer.s.host_waits = peer.s.copies = peer.s.unready_copies = 0;
        const auto before = ggml_backend_sched_get_transport_profile(sched);
        GGML_ASSERT(ggml_backend_sched_graph_compute_async(sched, g) == GGML_STATUS_SUCCESS);
        const bool gated = owned && waterfall && ready_min > 0 && ready_min <= sizeof(a);
        if (owned && events) {
            GGML_ASSERT(gpu.s.uploads >= 2);
            GGML_ASSERT(gated ? gpu.s.host_waits > 0 : gpu.s.host_waits == 0);
        }
        else { GGML_ASSERT(gpu.s.uploads == 0); GGML_ASSERT(gpu.s.host_waits >= 2); }
        if (waterfall) {
            GGML_ASSERT(gpu.s.copies > 0 && peer.s.copies > 0);
            if (gated) {
                GGML_ASSERT(gpu.s.unready_copies == 0 && peer.s.unready_copies == 0);
            } else if (owned && events) {
                GGML_ASSERT(gpu.s.unready_copies + peer.s.unready_copies > 0);
            }
        }
        // Owned inputs stay unchanged until completion; default inputs may change immediately.
        if (!owned) { float changed[] = {99, 99, 99, 99}; ggml_backend_tensor_set(x, changed, 0, sizeof(changed)); }
        ggml_backend_sched_synchronize(sched);
        const auto after = ggml_backend_sched_get_transport_profile(sched);
        const int calls = 8 + GGML_SCHED_TRANSPORT_PROFILE_BACKENDS;
        GGML_ASSERT(after.values[calls] - before.values[calls] == uint64_t(gated ? gpu.s.copies : 0));
        GGML_ASSERT(after.values[calls + 1] - before.values[calls + 1] == uint64_t(gated ? peer.s.copies : 0));
        GGML_ASSERT(after.values[calls + 2] == 0);
        GGML_ASSERT(after.values[8] + after.values[9] <= after.values[5]);
        ggml_backend_sched_synchronize(sched);
        const auto repeated = ggml_backend_sched_get_transport_profile(sched);
        for (int i = 0; i < GGML_SCHED_TRANSPORT_PROFILE_FIELDS; ++i) {
            GGML_ASSERT(repeated.values[i] == after.values[i]);
        }
        float result[4]; ggml_backend_tensor_get(z, result, 0, sizeof(result));
        for (int i = 0; i < 4; ++i) GGML_ASSERT(result[i] == (a[i] + b[i])*(waterfall ? 2 : 1));
    }
    ggml_backend_sched_free(sched); ggml_backend_buffer_free(inputs); ggml_free(ctx); ggml_backend_free(cpu);
}
int main() { run(false, true); run(true, true); run(true, false); run(false, true, true); run(true, true, true); run(true, false, true); run(true, true, true, 16); run(true, true, true, 17); run(false, true, true, 16); run(true, false, true, 16); unsetenv("GGML_SCHED_READY_COPY_MIN_BYTES"); puts("owned-input scheduling: PASS"); }
