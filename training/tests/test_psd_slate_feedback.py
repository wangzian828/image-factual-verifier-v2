import copy
import json

import pytest

from ifv_training.psd_repair import _sha
from ifv_training.psd_slate_feedback import (
    checker_feedback, diagnostic_position, load_slate_state, slate_replay_anchor)


def trace():
    return {"state": {"all_steps": [
        {"stage": "unified_react", "action_type": "tool_call", "thought": "Inspect the image",
         "metadata": {"policy_input": {}, "policy_action": {}}},
        {"stage": "unified_react", "action_type": "tool_result", "tool_result": "No matching results"},
        {"stage": "unified_judgment", "action_type": "output", "output": "The institution does not exist",
         "metadata": {"policy_input": {}, "policy_action": {}}}]}}


def review():
    return {"decision": {"status": "fail", "explanation": "SECRET_REFERENCE_NOT_OBSERVED",
        "evidence": [{"trace": "source", "step_index": 2,
                      "quote": "The institution does not exist"}]}}


def test_checker_citations_not_private_explanation_reach_hint_writer():
    feedback = checker_feedback(review(), trace())
    assert feedback["cited_observations"][0]["quote"] == "The institution does not exist"
    assert "SECRET_REFERENCE" not in json.dumps(feedback)
    assert "correct verdict alone" in feedback["criteria"]
    assert diagnostic_position(feedback, trace()) == 24
    anchor = slate_replay_anchor(trace())
    assert anchor.source_step_index == 0
    assert "not_error_localization" in anchor.localization_kind


@pytest.mark.parametrize("mutation", ["quote", "step", "trace", "status"])
def test_feedback_rejects_unobserved_or_nonfailure_content(mutation):
    data = review()
    if mutation == "quote": data["decision"]["evidence"][0]["quote"] = "SECRET_REFERENCE_NOT_OBSERVED"
    if mutation == "step": data["decision"]["evidence"][0]["step_index"] = 0
    if mutation == "trace": data["decision"]["evidence"][0]["trace"] = "repaired"
    if mutation == "status": data["decision"]["status"] = "pass"
    with pytest.raises(ValueError): checker_feedback(data, trace())


def test_repair_feedback_can_quote_the_actual_previous_hint():
    data = review()
    data["decision"]["evidence"] = [{"trace": "repaired", "position": 0,
        "step_index": 0, "quote": "Check whether the conclusion is observed"}]
    with pytest.raises(ValueError): checker_feedback(data, trace(), repaired=True)
    feedback = checker_feedback(data, trace(), repaired=True,
        hints={0: "Check whether the conclusion is observed"})
    assert len(feedback["cited_observations"]) == 1


def test_only_exact_legacy_numeric_key_hash_is_accepted(tmp_path):
    from ifv_training.psd_repair_storage import save_bound
    identity = {"version": "slate-search-v3-observed-positions", "inputs": "bound"}
    payload = {"rounds": [{"hints": {4: "first", 14: "second"}}], "status": "repairing"}
    saved = {"identity": identity, "payload": payload, "payload_sha256": _sha(payload)}
    path = tmp_path / "slate-state.json"
    path.write_text(json.dumps(saved))
    actual = load_slate_state(path, identity=identity)
    assert actual["rounds"][0]["hints"] == {"4": "first", "14": "second"}
    save_bound(path, identity=identity, payload=actual)
    assert load_slate_state(path, identity=identity) == actual
    corrupt = copy.deepcopy(saved)
    corrupt["payload"]["rounds"][0]["hints"][4] = "tampered"
    path.write_text(json.dumps(corrupt))
    with pytest.raises(ValueError): load_slate_state(path, identity=identity)
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError): load_slate_state(path, identity={**identity, "inputs": "other"})
