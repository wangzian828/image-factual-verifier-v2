from types import SimpleNamespace

import pytest

from ifv_training.psd_collection import (build_task_repair_selection_from_rewards,
    build_task_repair_selection_parallel, select_task_repair_sources, verify_collection)
from ifv_training.psd_collection import require_token_capture_environment
from src.eval.rollout import rollout_specs
from src.eval.result_records import run_result_record


def test_native_capture_is_required_before_collection():
    with pytest.raises(ValueError, match="before dispatch"):
        require_token_capture_environment({})
    with pytest.raises(ValueError, match="top-20"):
        require_token_capture_environment({'IFV_CAPTURE_POLICY_TOKENS':'1','IFV_POLICY_TOPK':'1'})
    require_token_capture_environment({'IFV_CAPTURE_POLICY_TOKENS':'1','IFV_POLICY_TOPK':'20'})


def collection():
    manifest = {"git_commit": "frozen", "agent": {"model": "policy", "rollouts_per_case": 8,
        "base_sampling_seed": 0, "psd_sampling": {"temperature": .7, "collector_sha256": "a" * 64}}}
    rows = rollout_specs([{}, {}], [SimpleNamespace(case_id=c) for c in ["a", "b"]],
        rollouts_per_case=8, base_sampling_seed=0, policy_revision="frozen", model="policy")
    return rows, manifest


def test_complete_eight_sample_groups_are_required():
    rows, manifest = collection()
    assert verify_collection(rows, case_ids=["a", "b"], manifest=manifest, expected_rollouts=8)["episodes"] == 16
    with pytest.raises(ValueError, match="incomplete"):
        verify_collection(rows[::8], case_ids=["a", "b"], manifest=manifest, expected_rollouts=8)


def native_records():
    specs, manifest = collection()
    rows = [run_result_record({"case_id": row["case_id"]},
        {"termination": "error", "error": "retained model failure"},
        trace_path="trace.json", metadata=None,
        **{key: row[key] for key in ("episode_id", "prompt_group_id", "rollout_index", "sampling_seed")})
        for row in specs]
    return rows, manifest


def test_native_result_writer_without_redundant_group_size_is_bound():
    rows, manifest = native_records()
    assert all("group_size" not in row for row in rows)
    result = verify_collection(rows, case_ids=["a", "b"], manifest=manifest, expected_rollouts=8)
    assert result["rows_without_redundant_group_size"] == 16
    assert result["selection_by_success"] is False
    assert len(rows) == 16 and all(row["status"] == "error" for row in rows)


@pytest.mark.parametrize("kind", ["missing", "group", "explicit_size", "seed"])
def test_native_records_still_reject_missing_or_rebound_slots(kind):
    rows, manifest = native_records()
    if kind == "missing":
        rows.pop()
    elif kind == "group":
        rows[0]["prompt_group_id"] = "pg-wrong"
    elif kind == "explicit_size":
        rows[0]["group_size"] = 1
    else:
        rows[0]["sampling_seed"] += 1
    with pytest.raises(ValueError):
        verify_collection(rows, case_ids=["a", "b"], manifest=manifest, expected_rollouts=8)


@pytest.mark.parametrize("kind", ["duplicate", "seed", "index", "budget", "policy", "temperature"])
def test_collection_rejects_slot_and_policy_changes(kind):
    rows, manifest = collection()
    if kind == "duplicate":
        rows[-1] = rows[0]
    elif kind == "seed":
        rows[0]["sampling_seed"] += 1
    elif kind == "index":
        rows[0]["rollout_index"] = 8
    elif kind == "budget":
        manifest["agent"]["rollouts_per_case"] = 2
    elif kind == "temperature":
        manifest["agent"]["psd_sampling"]["temperature"] = 1.0
    else:
        manifest["git_commit"] = "other"
    with pytest.raises(ValueError):
        verify_collection(rows, case_ids=["a", "b"], manifest=manifest, expected_rollouts=8)


def test_longest_failed_episode_selected_per_task_without_discarding_sources():
    rows = [{"class": "repair_seed", "case_id": "a", "episode_id": e} for e in ["short", "long"]]
    selected, audit = select_task_repair_sources(rows,
        load_trace=lambda row: {"state": {"all_steps": [1] * (9 if row["episode_id"] == "long" else 2)}})
    assert selected == [rows[1]]
    assert len(rows) == 2
    assert audit["tasks"][0]["unselected_episode_ids"] == ["short"]


def test_parallel_longest_selection_streams_exact_selected_rows(tmp_path):
    import hashlib
    import json
    from ifv_training.io import load_json, load_jsonl, write_jsonl

    run_dir = tmp_path / "run"
    candidates = []
    for case_id, episode_id, count in [("b", "b-short", 2), ("a", "a-long", 9),
                                       ("a", "a-short", 3), ("b", "b-long", 8)]:
        trace = run_dir / "traces" / f"{episode_id}.json"
        trace.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps({"state": {"all_steps": [{}] * count}}).encode()
        trace.write_bytes(raw)
        candidates.append({"class": "repair_seed", "candidate_id": f"candidate:{episode_id}",
            "case_id": case_id, "episode_id": episode_id,
            "large": list(range(count * 100)), "source": {
                "source_trace_path": trace.relative_to(run_dir).as_posix(),
                "source_trace_sha256": hashlib.sha256(raw).hexdigest()}})
    candidate_path = tmp_path / "candidates.jsonl"
    write_jsonl(candidate_path, candidates)
    result = build_task_repair_selection_parallel(candidates_path=candidate_path,
        run_dir=run_dir, output_dir=tmp_path / "selected", workers=2)
    chosen = load_jsonl(tmp_path / "selected/selected_candidates.jsonl")
    assert [(row["case_id"], row["episode_id"]) for row in chosen] == [
        ("a", "a-long"), ("b", "b-long")]
    selection = load_json(tmp_path / "selected/selection.json")
    assert selection["original_candidates"] == 4
    assert selection["selected_tasks"] == 2
    assert result["selected_candidates_sha256"] == hashlib.sha256(
        (tmp_path / "selected/selected_candidates.jsonl").read_bytes()).hexdigest()


def test_reward_bound_selection_does_not_reopen_traces(tmp_path):
    import hashlib
    import json
    from ifv_training.io import load_json, load_jsonl, write_json, write_jsonl

    candidates = []
    rewards = []
    for case_id, episode_id, count in [
            ("b", "b-short", 2), ("a", "a-long", 9),
            ("a", "a-short", 3), ("b", "b-long", 8)]:
        candidates.append({"class": "repair_seed", "candidate_id": f"candidate:{episode_id}",
            "case_id": case_id, "episode_id": episode_id,
            "large": list(range(count * 100)), "source": {"source_trace_sha256": "a" * 64}})
        rewards.append({"case_id": case_id, "episode_id": episode_id,
            "step_ids": [f"{episode_id}:{index}" for index in range(count)]})
    candidate_path = tmp_path / "candidates.jsonl"
    rewards_path = tmp_path / "rewards.jsonl"
    gate_path = tmp_path / "rollout-gate.json"
    write_jsonl(candidate_path, candidates)
    write_jsonl(rewards_path, rewards)
    write_json(gate_path, {"passed": True, "run": {
        "post_rollout_rewards_sha256": hashlib.sha256(rewards_path.read_bytes()).hexdigest()}})

    result = build_task_repair_selection_from_rewards(
        candidates_path=candidate_path, rewards_path=rewards_path,
        rollout_gate_path=gate_path, output_dir=tmp_path / "selected")

    chosen = load_jsonl(tmp_path / "selected/selected_candidates.jsonl")
    assert [(row["case_id"], row["episode_id"]) for row in chosen] == [
        ("a", "a-long"), ("b", "b-long")]
    assert result["trace_files_reopened"] == 0
    assert load_json(tmp_path / "selected/selection.json")["selected_tasks"] == 2
