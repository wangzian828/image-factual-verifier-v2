import json

from scripts.server.control_psd_posttrain_eval import (
    collected_smoke_successes,
    next_smoke_directory,
    smoke_anomaly_report,
)


def _attempt(root, name, successful_cases, *, tool_steps=None):
    directory = root / name
    directory.mkdir()
    (directory / "summary.json").write_text("{}")
    rows = []
    for case in successful_cases:
        trace = directory / f"{case}.json"
        trace.write_text(json.dumps({"state": {"all_steps": tool_steps or []}}))
        rows.append({"case_id": case, "status": "success", "termination": "success",
            "verdict": "real", "trace_path": trace.name,
            "fact_check_report": "A complete engineering smoke response.",
            "total_tool_calls": 0})
    (directory / "run_results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows))


def test_smoke_recovery_reuses_prior_successes_and_only_schedules_missing(tmp_path):
    _attempt(tmp_path, "smoke", [])
    _attempt(tmp_path, "smoke-budget-recovery-v2", ["a", "b", "c"])
    selected = collected_smoke_successes(tmp_path)
    assert set(selected) == {"a", "b", "c"}
    assert next_smoke_directory(tmp_path) == tmp_path / "smoke-infra-recovery-v3"

    _attempt(tmp_path, "smoke-infra-recovery-v3", ["d"])
    selected = collected_smoke_successes(tmp_path)
    report = smoke_anomaly_report(selected, ["a", "b", "c", "d"])
    assert report["passed"] is True
    assert report["terminal_successes"] == 4
    assert next_smoke_directory(tmp_path) is None


def test_smoke_loop_detection_uses_real_tool_args_not_missing_legacy_field(tmp_path):
    unique = [{"action_type": "tool_call", "tool_name": "text_search",
        "tool_args": {"queries": f"query {index}"}} for index in range(9)]
    _attempt(tmp_path, "smoke", ["a"], tool_steps=unique)
    assert smoke_anomaly_report(collected_smoke_successes(tmp_path), ["a"])["passed"] is True

    repeated = [{"action_type": "tool_call", "tool_name": "text_search",
        "tool_args": {"queries": "same query"}} for _ in range(8)]
    (tmp_path / "smoke" / "a.json").write_text(json.dumps({"state": {"all_steps": repeated}}))
    report = smoke_anomaly_report(collected_smoke_successes(tmp_path), ["a"])
    assert report["passed"] is False
    assert report["records"][0]["issues"] == ["repeated_identical_tool_loop"]
