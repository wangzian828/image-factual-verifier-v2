import json
from pathlib import Path

import pytest

from scripts.server import coordinate_psd_eval_agent_waves as gate
from scripts.server.coordinate_psd_eval_agent_waves import wave_decision


def add_success(group: Path, wave: str, case_id: str) -> None:
    output = group / wave
    output.mkdir(parents=True, exist_ok=True)
    row = {
        "case_id": case_id, "status": "success", "termination": "success",
        "verdict": "real", "trace_path": "trace.json",
    }
    with (output / "run_results.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def test_smoke_requires_four_successes_at_last_wave(tmp_path: Path) -> None:
    for index in range(3):
        add_success(tmp_path, "smoke-0", f"case-{index}")
    assert wave_decision(tmp_path, "smoke-0")["advance"] is True
    assert wave_decision(tmp_path, "smoke-1")["advance"] is False
    add_success(tmp_path, "smoke-1", "case-3")
    assert wave_decision(tmp_path, "smoke-1")["advance"] is True


def test_full_agent_failure_blocks_after_last_retry(tmp_path: Path) -> None:
    for index in range(5):
        add_success(tmp_path, "attempt-0", f"case-{index}")
    assert wave_decision(tmp_path, "attempt-0", expected=6)["advance"] is True
    assert wave_decision(tmp_path, "attempt-3", expected=6)["advance"] is False
    add_success(tmp_path, "attempt-3", "case-5")
    assert wave_decision(tmp_path, "attempt-3", expected=6)["advance"] is True


def test_judge_never_blocks_agent_group_transition(tmp_path: Path) -> None:
    for index in range(4):
        add_success(tmp_path, "attempt-0", f"case-{index}")
    decision = wave_decision(tmp_path, "attempt-0", expected=4)
    assert decision["advance"] is True
    assert decision["judge_not_a_stage_barrier"] is True


def test_unknown_wave_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unrecognized"):
        wave_decision(tmp_path, "attempt-4")


def test_unknown_success_case_is_rejected(tmp_path: Path) -> None:
    add_success(tmp_path, "attempt-0", "other")
    with pytest.raises(ValueError, match="frozen cohort"):
        wave_decision(tmp_path, "attempt-0", expected=1, expected_ids={"case-1"})


def test_smoke_requires_exact_frozen_ids(tmp_path: Path) -> None:
    for index in range(4):
        add_success(tmp_path, "smoke-0", f"case-{index}")
    expected = {f"case-{index}" for index in range(5)}
    decision = wave_decision(
        tmp_path, "smoke-1", expected=5, expected_ids=expected,
        smoke_ids={"case-0", "case-1", "case-2", "case-4"},
    )
    assert decision["advance"] is False


def test_new_ablation_judge_is_paused_without_touching_agent(tmp_path: Path, monkeypatch) -> None:
    full = tmp_path / "full"
    output = tmp_path / "full-no-web-search"
    deploy = tmp_path / "deploy"
    judge_path = deploy / "no-web-search" / f"judge-{gate.PROFILE_MODEL}.json"
    write = lambda path, value: (path.parent.mkdir(parents=True, exist_ok=True), path.write_text(json.dumps(value)))
    write(judge_path, {"pid": 123, "output": str(output / "judge-gemini37-stream-v1")})
    monkeypatch.setattr(gate, "DEPLOY_ROOT", deploy)
    current = ["S"]
    marker = "stream_agent_judges.py\x00--rollout-root\x00" + str(output)
    monkeypatch.setattr(gate, "process", lambda pid: (current[0], 100, 999, marker))
    signals = []

    def kill(pid, sig):
        signals.append((pid, sig))
        current[0] = "T"

    monkeypatch.setattr(gate.os, "kill", kill)
    gate.pause_new_ablation_judges(100, full, tmp_path / "gate")
    assert len(signals) == 1
    result = json.loads((tmp_path / "gate/judge-pauses/no-web-search.json").read_text())
    assert result["phase"] == "judge_paused"
    gate.pause_new_ablation_judges(100, full, tmp_path / "gate")
    assert len(signals) == 1


def test_new_ablation_judge_wrong_parent_fails_closed(tmp_path: Path, monkeypatch) -> None:
    full = tmp_path / "full"
    output = tmp_path / "full-no-web-search"
    deploy = tmp_path / "deploy"
    receipt = deploy / "no-web-search" / f"judge-{gate.PROFILE_MODEL}.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text(json.dumps({"pid": 123, "output": str(output / "judge-gemini37-stream-v1")}))
    monkeypatch.setattr(gate, "DEPLOY_ROOT", deploy)
    marker = "stream_agent_judges.py\x00--rollout-root\x00" + str(output)
    monkeypatch.setattr(gate, "process", lambda pid: ("S", 999, 777, marker))
    with pytest.raises(RuntimeError, match="identity mismatch"):
        gate.pause_new_ablation_judges(100, full, tmp_path / "gate")


def test_new_ablation_judge_transient_d_state_is_rechecked(tmp_path: Path, monkeypatch) -> None:
    full = tmp_path / "full"
    output = tmp_path / "full-no-web-search"
    deploy = tmp_path / "deploy"
    receipt = deploy / "no-web-search" / f"judge-{gate.PROFILE_MODEL}.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text(json.dumps({"pid": 123, "output": str(output / "judge-gemini37-stream-v1")}))
    monkeypatch.setattr(gate, "DEPLOY_ROOT", deploy)
    marker = "stream_agent_judges.py\x00--rollout-root\x00" + str(output)
    current = ["D"]
    monkeypatch.setattr(gate, "process", lambda pid: (current[0], 100, 999, marker))
    signals = []

    def kill(pid, sig):
        signals.append((pid, sig))
        current[0] = "T"

    monkeypatch.setattr(gate.os, "kill", kill)
    gate.pause_new_ablation_judges(100, full, tmp_path / "gate")
    assert not signals
    assert not (tmp_path / "gate/judge-pauses/no-web-search.json").exists()
    current[0] = "S"
    gate.pause_new_ablation_judges(100, full, tmp_path / "gate")
    assert signals == [(123, gate.STOP_SIGNAL)]
    assert json.loads((tmp_path / "gate/judge-pauses/no-web-search.json").read_text())["phase"] == "judge_paused"
