import json
from scripts.analyze_agent_latency import summarize


def test_latency_does_not_double_count_nested_tool_calls_or_segment_boundary(tmp_path):
    path = tmp_path / "case.json"
    path.write_text(json.dumps({"termination": "success", "time_taken": 10, "state": {"all_steps": [
        {"action_type": "tool_call", "tool_name": "browse", "stage": "unified_react",
         "tokens": {"prompt": 20, "completion": 3}, "metadata": {"llm_duration_ms": 6000,
          "duration_ms": 3000, "tool_subcalls": [{"duration_ms": 2900}], "finish_reason": "stop"}},
        {"metadata": {"deterministic_segment_boundary": True, "llm_duration_ms": 6000}}]}}))
    report = summarize([path])
    assert report["wall_share_percent"] == {"llm_s": 60.0, "tool_s": 30.0, "other_or_unattributed_s": 10.0}
    assert report["case_distributions"]["completion_tokens"]["sum"] == 3
