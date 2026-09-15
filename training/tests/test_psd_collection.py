from types import SimpleNamespace

import pytest

from ifv_training.psd_collection import select_task_repair_sources, verify_collection
from src.eval.rollout import rollout_specs


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
