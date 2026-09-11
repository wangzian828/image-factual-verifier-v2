import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts import postprocess_psd_training as module


def write(path, value, lines=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in value) if lines else json.dumps(value), encoding="utf-8")


def setup(tmp_path, monkeypatch):
    run = tmp_path / "run"
    policy = tmp_path / "policy.json"
    write(policy, {"test": "policy"})
    write(run / "run_manifest.json", {"status": "completed", "source_access_policy": {"active": True, "path": str(policy)}})
    write(run / "run_results.jsonl", [{"case_id": "a", "episode_id": "a"}], True)
    write(run / "traces/a.json", {"state": {"runtime_case": {"case_id": "a"}}})
    write(tmp_path / "cases.jsonl", [{"case_id": "a", "split": "train"}], True)
    write(tmp_path / "gold.jsonl", [{"case_id": "a", "factual_status": "supported"}], True)
    monkeypatch.setattr(module.SourceAccessPolicy, "load", lambda path: object())
    monkeypatch.setattr(module, "score_process_trace", lambda *a: ({"result_correct": True}, {"components": {}}))
    monkeypatch.setattr(module, "audit_trace", lambda *a, **kw: SimpleNamespace(failures=lambda **kw: []))
    monkeypatch.setattr(module, "trajectory_policy_step_ids", lambda *a, **kw: ["a-step0"])
    return dict(run_dir=run, train_cases=tmp_path / "cases.jsonl", private_gold=tmp_path / "gold.jsonl", source_access_policy=policy)


def test_private_training_derivation_does_not_export_other_datasets(tmp_path, monkeypatch):
    args = setup(tmp_path, monkeypatch)
    result = module.postprocess(**args)
    assert result["correct"] == result["strict_pass"] == 1
    assert not (args["run_dir"] / "trajectory_sft.jsonl").exists()
    assert not (args["run_dir"] / "perception_trajectories.jsonl").exists()


@pytest.mark.parametrize("failure", ["heldout", "unknown_gold", "unfinished", "policy_changed", "duplicate"])
def test_fail_closed_training_inputs(tmp_path, monkeypatch, failure):
    args = setup(tmp_path, monkeypatch)
    if failure == "heldout":
        write(args["train_cases"], [{"case_id": "a", "split": "test"}], True)
    elif failure == "unknown_gold":
        write(args["private_gold"], [{"case_id": "a", "factual_status": "unknown"}], True)
    elif failure == "unfinished":
        write(args["run_dir"] / "run_manifest.json", {"status": "running"})
    elif failure == "policy_changed":
        changed = tmp_path / "changed-policy.json"
        write(changed, {})
        args["source_access_policy"] = changed
    else:
        write(args["run_dir"] / "run_results.jsonl", [{"case_id": "a"}] * 2, True)
    with pytest.raises(ValueError):
        module.postprocess(**args)
