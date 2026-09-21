import json

from scripts.server.control_psd_posttrain_eval import (
    collected_smoke_successes,
    next_smoke_directory,
    smoke_anomaly_report,
)


def _attempt(root, name, successful_cases):
    directory = root / name
    directory.mkdir()
    (directory / "summary.json").write_text("{}")
    rows = []
    for case in successful_cases:
        trace = directory / f"{case}.json"
        trace.write_text(json.dumps({"state": {"all_steps": []}}))
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
