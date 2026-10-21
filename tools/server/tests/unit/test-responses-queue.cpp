#include "tools/server/server-queue.h"
#include <atomic>
#include <cassert>
#include <chrono>
#include <thread>

int main() {
    server_queue queue;
    server_response results;
    server_response_reader before(queue, results, 1);
    server_task unavailable(SERVER_TASK_TYPE_RESPONSES_CONTEXT);
    unavailable.id = before.get_new_id();
    assert(!before.post_task_if_awake(std::move(unavailable), true));
    assert(!before.has_next());
    std::atomic<bool> sleeping{false};
    std::atomic<int> processed{0};
    queue.on_new_task([&](server_task && task, bool) {
        processed++;
        auto result = std::make_unique<server_task_result_responses_context>();
        result->id = task.id; result->proof = {{"success", true}};
        results.send(std::move(result));
        return true;
    });
    queue.on_update_slots([] {});
    queue.on_sleeping_state([&](bool state) { sleeping = state; });
    std::thread loop([&] { queue.start_loop(0); });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (!sleeping && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
    assert(sleeping);
    server_response_reader reader(queue, results, 1);
    server_task task(SERVER_TASK_TYPE_RESPONSES_CONTEXT);
    task.id = reader.get_new_id();
    assert(!reader.post_task_if_awake(std::move(task), true));
    assert(!reader.has_next());
    assert(queue.is_sleeping() && processed == 0); // no wake, no late queued mutation
    queue.terminate(); loop.join();
}
