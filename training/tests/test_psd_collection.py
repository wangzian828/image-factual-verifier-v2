from types import SimpleNamespace

import pytest

from ifv_training.psd_collection import select_task_repair_sources, verify_collection
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
