import json

import pytest

from scripts.server import continue_psd_combined_prioritized_ablations as flow


def test_priority_is_evidence_before_retrieval():
    assert flow.ORDER == ("no-web-search", "no-evidence-inspection", "no-image-retrieval")
    assert set(flow.ORDER) == set(flow.CONFIG)


def test_handoff_rejects_a_resumed_old_parent(monkeypatch):
    rows = {
        flow.OLD_PARENT[0]: {
            "pid": flow.OLD_PARENT[0], "state": "S", "ppid": 1,
            "startticks": flow.OLD_PARENT[1],
            "command": "run_psd_combined_cachefixed_agent_eval.py execute",
        },
        flow.OLD_CHILD[0]: None,
    }
    monkeypatch.setattr(flow, "process_identity", lambda pid: rows[pid])
    with pytest.raises(RuntimeError, match="verified stopped parent"):
        flow.attest_parent_child(child_required=False)


def test_finished_no_web_reuses_attempt_zero_and_only_retries_missing(monkeypatch, tmp_path):
    output = tmp_path / "no-web-search"
    (output / "attempt-0").mkdir(parents=True)
    (output / "smoke-tool-ablation-audit.json").write_text(json.dumps({
        "passed": True, "tool_families": flow.CONFIG["no-web-search"][1].to_manifest(),
    }))
    monkeypatch.setattr(flow, "DEPLOY", tmp_path / "deploy")
    flow.DEPLOY.mkdir()
    monkeypatch.setattr(flow.cachefixed, "output_for", lambda name: output)
    ids = [f"case-{index:04d}" for index in range(1526)]
    monkeypatch.setattr(flow.agent, "rows", lambda _: [{"case_id": x} for x in ids])
    monkeypatch.setattr(flow.agent, "runtime_environment", lambda *a, **k: {})
    monkeypatch.setattr(flow.previous, "attest_ablation_run_manifest", lambda *a: None)
    selections = iter(({x: ({}, output) for x in ids[:1500]},
                       {x: ({}, output) for x in ids[:1500]},
                       {x: ({}, output) for x in ids},
                       {x: ({}, output) for x in ids}))
    monkeypatch.setattr(flow.agent, "successful", lambda _: next(selections))
    calls = []
    monkeypatch.setattr(flow.agent, "run_attempt", lambda **kwargs: calls.append(kwargs))
    result = flow.finish_no_web()
    assert result["phase"] == "inference_complete"
    assert result["success"] == 1526
    assert len(calls) == 1
    assert calls[0]["name"] == "attempt-1"
    assert calls[0]["cases"] == ids[1500:]
    assert calls[0]["base_seed"] == 3903
    assert calls[0]["tool_ablation_flags"] == ("--disable-web-search",)
