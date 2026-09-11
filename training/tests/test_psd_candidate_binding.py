import pytest

from ifv_training.psd_candidates import _candidate_id, _policy_steps
from ifv_training.psd_candidate_binding import bind_localized_candidate
from dataclasses import dataclass


@dataclass
class Site:
    source_step_index: int
    stage: str
    step_id: str
    example_type: str = "react"


def inputs():
    trace = {"state": {"runtime_case": {"case_id": "case"}, "all_steps": [
        {"stage": "unified_react", "metadata": {"policy_input": {}, "policy_action": {},
          "policy_token_capture": {"status": "complete", "prompt_token_ids": [index + 1],
                                   "completion_token_ids": [10], "completion_logprobs": [-.1]}}}
        for index in range(2)]}}
    anchor = _policy_steps(trace, episode_id="episode")[1]
    candidate = {"class": "repair_seed", "case_id": "case", "episode_id": "episode",
                 "repair_site": anchor, "source": {"source_trace_sha256": "a" * 64}}
    candidate["candidate_id"] = _candidate_id(kind="repair", case_id="case", episode_id="episode",
        step_id=anchor["step_id"], source_trace_sha256="a" * 64)
    return candidate, trace


def test_semantic_relocation_rebinds_canonical_step_and_candidate_id():
    candidate, trace = inputs()
    localized, site = bind_localized_candidate(candidate, trace,
        Site(0, "unified_react", "different-exporter-id"), source_trace_sha256="a" * 64)
    assert localized["candidate_id"] != candidate["candidate_id"]
    assert localized["parent_candidate_id"] == candidate["candidate_id"]
    assert localized["repair_site"]["step_id"] == site.step_id
    assert localized["repair_site"]["rollout_token_capture"]["prompt_token_ids"] == [1]
    assert candidate["repair_site"]["source_step_index"] == 1


def test_driver_rejects_candidate_for_different_trace():
    candidate, trace = inputs()
    with pytest.raises(ValueError, match="trace hash mismatch"):
        bind_localized_candidate(candidate, trace, Site(0, "unified_react", "x"),
                                 source_trace_sha256="b" * 64)
