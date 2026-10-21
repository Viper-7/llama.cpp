#include "tools/server/server-task.h"
#include "tools/server/server-chat.h"
#include <cassert>

int main() {
    server_task_result_cmpl_partial partial;
    server_task_result_cmpl_final final;
    server_task_result_responses_suspended suspended;
    suspended.request_id = "job_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:0";
    server_task_result_control unrelated;
    assert(server_task_result_is_completion_stream(&partial));
    assert(server_task_result_is_completion_stream(&final));
    // Progress output can precede the prefill suspension. The later reader
    // must accept the same marker type as the first reader without aborting.
    for (server_task_result * result : {static_cast<server_task_result *>(&partial),
                                       static_cast<server_task_result *>(&suspended)}) {
        assert(server_task_result_is_completion_stream(result));
    }
    assert(!server_task_result_is_completion_stream(&unrelated));
    assert(!server_task_result_is_completion_stream(nullptr));
    auto event = suspended.to_json();
    assert(event.is_array() && event.size() == 1);
    assert(event[0]["event"] == "serverless.suspended");
    assert(event[0]["data"]["request_id"] == suspended.request_id);
    assert(suspended.is_stop());

    const json request = {{"input", "unfinished user text"}, {"prefill_only", true}, {"max_output_tokens", 1}};
    const auto converted = server_chat_convert_responses_to_chatcmpl(request);
    assert(converted.at("prefill_only") == true);
    assert(converted.at("add_generation_prompt") == false);
    assert(converted.at("messages").back().at("content") == "unfinished user text");
    const auto normal = server_chat_convert_responses_to_chatcmpl({{"input", "unfinished user text"}});
    assert(!normal.contains("add_generation_prompt"));
    final.stop = STOP_TYPE_EOS;
    final.n_decoded = 0;
    final.n_prompt_tokens = 12;
    final.res_type = TASK_RESPONSE_TYPE_OAI_RESP;
    final.stream = true;
    task_result_state result_state;
    final.update(result_state);
    auto completed = final.to_json();
    assert(completed.is_array());
    const auto terminal = completed.back().at("data");
    assert(terminal.at("type") == "response.completed");
    assert(terminal.at("response").at("output").empty());
    assert(terminal.at("response").at("usage").at("output_tokens") == 0);
}
