import json
from pathlib import Path

import pytest

from scripts.server import retire_psd_contaminated_eval as retire
from scripts.server import run_psd_combined_cachefixed_agent_eval as flow


def test_retirement_requires_all_exact_paused_processes(monkeypatch):
    rows = {}
    parent = next(item[1] for item in retire.TARGETS if item[0] == "parent")
    for name, pid, ticks, *markers in retire.TARGETS:
        rows[pid] = {
            "pid": pid, "state": "S" if name == "sidecar" else "T",
            "ppid": 1 if name in {"sidecar", "parent"} else parent,
            "startticks": ticks, "cmdline": " ".join(markers),
        }
    monkeypatch.setattr(retire, "snapshot", lambda pid: rows[pid])
    assert len(retire.verified_targets()) == 5
    rows[2598192]["state"] = "S"
    with pytest.raises(RuntimeError, match="not paused"):
        retire.verified_targets()


def test_cachefixed_preflight_refuses_perception_cache(monkeypatch, tmp_path):
    source_output = tmp_path / "old-output"
    source_deploy = tmp_path / "old-deploy"
    source_output.mkdir()
    source_deploy.mkdir()
    probe = tmp_path / "probe.json"
    probe.write_text(json.dumps({
        "different_results": True,
        "records": [{"status": "success"}, {"status": "success"}],
    }))
    monkeypatch.setattr(flow, "SOURCE_OUTPUT", source_output)
    monkeypatch.setattr(flow, "SOURCE_DEPLOY", source_deploy)
    monkeypatch.setattr(flow, "PERCEPTION_PROBE", probe)
    monkeypatch.setattr(flow.previous, "training_result", lambda: {})
    monkeypatch.setattr(flow.previous, "frozen_cases", lambda: 1526)
    monkeypatch.setattr(flow.agent, "runtime_environment", lambda *a, **k: {
        "TOOL_CACHE_ENABLED": "0", "PERCEPTION_CACHE_ENABLED": "1",
    })
    with pytest.raises(RuntimeError, match="both independent tool caches"):
        flow.preflight()


def test_failed_full_agent_never_starts_an_ablation(monkeypatch, tmp_path):
    monkeypatch.setattr(flow, "DEPLOY", tmp_path / "deploy")
    monkeypatch.setattr(flow, "FULL_OUTPUT", tmp_path / "full")
    monkeypatch.setattr(flow, "preflight", lambda: {"perception_cache_enabled": False})
    monkeypatch.setattr(flow.agent, "evaluate_model", lambda **_: {
        "phase": "engineering_retry_budget_exhausted", "success": 1525,
    })
    monkeypatch.setattr(flow.previous, "evaluate_ablation", lambda **_: pytest.fail("ablation started"))
    with pytest.raises(RuntimeError, match="Full Agent did not close"):
        flow.execute()
    assert json.loads((flow.DEPLOY / "state.json").read_text())["phase"] == "held_requires_inspection"
    assert not (tmp_path / "full-no-web-search").exists()
