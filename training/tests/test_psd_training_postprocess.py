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
    assert result["verified_full_task"] == 0
    assert result["source_review_pending"] == 1


@pytest.mark.parametrize("status,verified,pending", [("pass", 1, 0), ("fail", 0, 0), ("unresolved", 0, 1)])
def test_semantic_source_admission_propagates_into_rewards(tmp_path, monkeypatch, status, verified, pending):
    from test_psd_source_review import artifact, source_trace
    from scripts.review_psd_sources import review_path
    from ifv_training.psd_repair_storage import save_bound
    args = setup(tmp_path, monkeypatch)
    trace = source_trace()
    write(args["run_dir"] / "traces/a.json", trace)
    reviews = tmp_path / "reviews"
    save_bound(review_path(reviews, "a"), identity={"test": True}, payload=artifact(trace, status=status))
    result = module.postprocess(**args, source_reviews=reviews)
    assert result["verified_full_task"] == verified
    assert result["source_review_pending"] == pending
    reward = module.load_jsonl(args["run_dir"] / "post_rollout_rewards.jsonl")[0]
    assert reward["source_task_status"] == status
    assert reward["source_task_review"]["sha256"]


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
