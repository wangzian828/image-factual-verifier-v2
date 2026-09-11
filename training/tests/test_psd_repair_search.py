import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ifv_training.io import load_json, load_jsonl, sha256_file, write_json, write_jsonl
from ifv_training.psd_repair_search import (
    CHECKS, inspect_round, public_attempt_feedback, revision_context, revision_rejection, run_search, search_lock,
    validate_live_model,
)
from ifv_training.psd_repair import FailureSite, build_proposer_prompt, parse_proposer_response


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


def test_repeated_or_modified_verified_hint_is_rejected_before_policy_call():
    locked = "Recheck the observed time relation."
    feedback = {"excluded_hints": [locked], "locked_hint": locked}
    assert revision_rejection(locked, feedback) == "repeated_completed_hint"
    assert revision_rejection("Check something unrelated.", feedback) == "changed_verified_hint"
    assert revision_rejection(locked + "\nCheck the remaining conflicting observation.", feedback) == ""


@pytest.mark.parametrize("value", [{}, {"hints": None}, {"hints": [3]}, {"hints": [""]},
                                   {"hints": [], "extra": True}, {"hints": ["a", "b"]}])
def test_invalid_proposer_response_is_not_a_no_repair_vote(value):
    with pytest.raises(ValueError):
        parse_proposer_response(value, limit=1)


def test_empty_hint_list_is_an_explicit_stop_vote():
    assert parse_proposer_response({"hints": []}, limit=1) == []


def test_visual_tools_and_policy_share_the_attested_endpoint():
    from scripts.run_psd_repair_driver import _policy_runtime_kwargs
    args = SimpleNamespace(policy_provider="qwen_local", policy_model="frozen-model",
                           policy_wire_api="chat_completions")
    policy = object()
    values = _policy_runtime_kwargs(args, "http://127.0.0.1:8901/v1", policy)
    assert values["llm_base_url"] == values["vlm_base_url"] == "http://127.0.0.1:8901/v1"
    assert values["vlm_model"] == values["model_name"] == "frozen-model"
    assert values["source_access_policy"] is policy


def test_visual_probe_uses_real_image_binding_and_public_runtime_adapter(tmp_path):
    from training.scripts.probe.psd_visual_endpoint_smoke import bound_visual_tool
    from src.tools.focused_visual_inspection import FocusedVisualInspectionTool
    path = tmp_path / "synthetic-image"
    path.write_bytes(b"unit-test-not-a-real-image")
    episode = {"case_id": "test-case", "image_path": str(path),
               "state": {"runtime_case": {"image_sha256": sha256_file(path)}}}
    orchestrator = SimpleNamespace(all_tools={"focused_visual_inspection": FocusedVisualInspectionTool()})
    tool = bound_visual_tool(orchestrator, episode)
    bound = tool._provider_args({"question": "Which direction is the object facing?"})
    assert bound["image_input"] == str(path) and bound["visual_question_id"]
    assert bound["question"] == bound["expected_property"]
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="differs"):
        bound_visual_tool(orchestrator, episode)


def test_live_alias_cannot_silently_switch_checkpoint_or_context(tmp_path):
    profile = {"profile_id": "model", "model_path": str(tmp_path / "base"), "context_length": 131072}
    row = {"id": "model", "root": profile["model_path"], "max_model_len": 131072}
    assert validate_live_model(profile, {"data": [row]})["max_model_len"] == 131072
    with pytest.raises(ValueError, match="checkpoint"):
        validate_live_model(profile, {"data": [{**row, "root": str(tmp_path / "trained")} ]})
    with pytest.raises(ValueError, match="context"):
        validate_live_model(profile, {"data": [{**row, "max_model_len": 65536}]})
    with pytest.raises(ValueError, match="alias"):
        validate_live_model(profile, {"data": []})


def test_storage_budget_pauses_without_deleting_artifacts_or_generating(tmp_path, monkeypatch):
    import ifv_training.psd_repair_search as search
    monkeypatch.setattr(search, "MAX_SEARCH_ARTIFACT_BYTES", 1)
    async def execute(*args):
        raise AssertionError("must not consume provider calls")
    result = run(tmp_path, execute)
    assert result["status"] == "paused_storage_budget"
    assert (tmp_path / "search-state.json").exists()
