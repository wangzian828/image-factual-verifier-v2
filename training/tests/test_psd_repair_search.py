import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ifv_training.io import load_json, load_jsonl, sha256_file, write_json, write_jsonl
from ifv_training.psd_repair_search import (
    CHECKS, inspect_round, public_attempt_feedback, revision_context, run_search, search_lock,
)
from ifv_training.psd_repair import FailureSite, build_proposer_prompt


def make_round(directory, *, passed=False, pending=False, hint="Check the observed timing.",
               no_hint=False, bad_hint=False, local_pass=False):
    directory.mkdir(parents=True, exist_ok=True)
    episode = {"state": {"all_steps": [{"stage": "unified_react", "tool_result": "Observed event time differs."}]}}
    path = directory / "episodes/hint-00-teacher.json"
    write_json(path, episode)
    review = {**{k: passed for k in CHECKS}, "anchor_matches": True, "episode_complete": True,
              "earliest_error_step": 0, "explanation": "SECRET PRIVATE REFERENCE"}
    if local_pass:
        review.update(source_error_confirmed=True, repaired_at_selected_step=True, procedural_hint=True)
    record = {"candidate_id": "candidate", "attempt_id": directory.name,
              "hint_record": {"text": hint}, "repair_site": {"source_step_index": 0},
              "local_verification": {} if pending else {"review": review},
              "hinted_episode_pass": passed, "hinted_strict_trace_audit_pass": True,
              "accepted": passed, "verification": {"expected_verdict": "SECRET GOLD"},
              "continuation": {"hinted_teacher_episode_trace": str(path.relative_to(directory)),
                  "hinted_teacher_episode_trace_sha256": sha256_file(path)}}
    write_jsonl(directory / "repair_attempts.jsonl", [] if no_hint else [record])
    write_jsonl(directory / "repair_candidates.jsonl", [{"candidate_id": "candidate"}])
    write_json(directory / "manifest.json", {"status": "no_admissible_hints" if no_hint else "generated",
               "pending_hinted_episode_count": int(pending)})
    write_json(directory / "proposer-hint-audits.json", {"proposals":
               [{"passed": False, "reason": "structured_action_scaffold_leak"}] if bad_hint else []})
    return record, episode


def run(root, callback, **kwargs):
    return asyncio.run(run_search(root=root, identity={"case": "test"}, resume=kwargs.pop("resume", False),
                      execute_round=callback, **kwargs))


def test_failed_feedback_then_pass_stops_and_resume_makes_no_calls(tmp_path):
    contexts = []
    async def execute(directory, feedback, history):
        contexts.append(feedback)
        make_round(directory, passed=len(history) == 1, hint=f"Check observation {len(history)}.")
    result = run(tmp_path, execute)
    assert result["converged"] and result["candidate_count"] == 2
    assert len(contexts) == 2
    assert contexts[1]["previous_rounds"][0]["attempts"][0]["repaired_steps"][0]["tool_result"]
    assert "SECRET" not in json.dumps(contexts)
    assert result["accepted_count"] == 1
    assert run(tmp_path, execute, resume=True)["converged"]
    assert len(contexts) == 2
    assert len(load_jsonl(tmp_path / "repair_candidates.jsonl")) == 1


def test_attempt_budget_is_hard_and_all_failures_retained(tmp_path):
    async def execute(directory, feedback, history):
        make_round(directory, hint=str(len(history)))
    result = run(tmp_path, execute, max_attempts=3)
    assert result["status"] == "attempt_budget_exhausted"
    assert result["candidate_count"] == 3 and not result["converged"]


def test_invalid_proposals_do_not_spend_continuation_budget_but_are_bounded(tmp_path):
    async def execute(directory, feedback, history):
        make_round(directory, no_hint=True, bad_hint=True)
    result = run(tmp_path, execute, max_attempts=2, max_proposals=4)
    assert result["status"] == "proposal_budget_exhausted"
    assert result["candidate_count"] == 0 and result["proposal_rounds"] == 4


def test_proposer_can_stop_without_making_a_continuation(tmp_path):
    async def execute(directory, feedback, history):
        make_round(directory, no_hint=True)
    result = run(tmp_path, execute)
    assert result["status"] == "no_further_grounded_hint"
    assert result["candidate_count"] == 0 and result["proposal_rounds"] == 1


def test_pending_verifier_resumes_same_round_and_never_turns_into_rejection(tmp_path):
    calls = []
    async def execute(directory, feedback, history):
        calls.append(directory)
        make_round(directory, pending=len(calls) == 1, passed=len(calls) > 1)
    first = run(tmp_path, execute)
    assert first["status"] == "paused_pending_verifier" and first["candidate_count"] == 0
    second = run(tmp_path, execute, resume=True)
    assert second["converged"] and calls[0] == calls[1]


def test_transport_failure_pauses_without_new_hint_and_secret_not_logged(tmp_path):
    calls = []
    async def execute(directory, feedback, history):
        calls.append(directory)
        if len(calls) == 1:
            raise RuntimeError("SECRET PROVIDER KEY")
        make_round(directory, passed=True)
    first = run(tmp_path, execute)
    assert first["status"] == "paused_execution_error"
    assert "SECRET" not in json.dumps(first)
    assert run(tmp_path, execute, resume=True)["converged"]
    assert calls[0] == calls[1]


def test_changed_completed_artifact_refuses_resume(tmp_path):
    async def execute(directory, feedback, history):
        make_round(directory, passed=True)
    run(tmp_path, execute)
    write_json(tmp_path / "rounds/round-00/episodes/hint-00-teacher.json", {"changed": True})
    with pytest.raises(ValueError, match="changed"):
        run(tmp_path, execute, resume=True)


def test_changed_budget_refuses_resume(tmp_path):
    async def execute(directory, feedback, history):
        make_round(directory, passed=True)
    run(tmp_path, execute)
    with pytest.raises(ValueError, match="binding changed"):
        run(tmp_path, execute, resume=True, max_attempts=7)


def test_wall_budget_checked_between_rounds(tmp_path):
    now = [0.0]
    async def execute(directory, feedback, history):
        make_round(directory)
        now[0] += 3
    result = run(tmp_path, execute, max_seconds=2, clock=lambda: now[0])
    assert result["status"] == "time_budget_exhausted" and result["candidate_count"] == 1


def test_preserve_verified_hint_verbatim_and_private_judge_text_is_absent(tmp_path):
    record, episode = make_round(tmp_path / "round", local_pass=True)
    feedback = public_attempt_feedback(record, episode)
    context = revision_context([{"feedback": {"attempts": [feedback]}}])
    assert context["locked_hint"] == record["hint_record"]["text"]
    assert "SECRET" not in json.dumps(context)
    assert not context["needs_relocalization"]


def test_wrong_anchor_requests_relocalization_without_judge_explanation(tmp_path):
    record, episode = make_round(tmp_path / "round")
    record["local_verification"]["review"].update(anchor_matches=False, earliest_error_step=2)
    context = revision_context([{"feedback": {"attempts": [public_attempt_feedback(record, episode)]}}])
    assert context["needs_relocalization"] and context["anchor_feedback"]["earliest_error_step"] == 2
    assert "SECRET" not in json.dumps(context)


def test_private_gold_not_in_proposer_prompt():
    site = FailureSite(0, "site", "unified_react", "react",
                       {"input_payload": [{"role": "user", "content": "observed"}]}, {})
    prompt = build_proposer_prompt(failure_site=site, public_trace_context={}, hint_count=1,
                                  private_context={"expected_verdict": "SECRET GOLD", "url": "SECRET URL"})
    assert "SECRET" not in prompt and "privileged_reference" not in prompt


def test_search_lock_excludes_second_writer(tmp_path):
    with search_lock(tmp_path):
        with pytest.raises(OSError):
            with search_lock(tmp_path):
                pass


@pytest.mark.parametrize("budget", [0, -1, 33, True])
def test_invalid_budget_rejected_before_calls(tmp_path, budget):
    async def execute(*args):
        raise AssertionError("must not run")
    with pytest.raises(ValueError):
        run(tmp_path, execute, max_attempts=budget)
