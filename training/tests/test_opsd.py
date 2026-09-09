from __future__ import annotations

import pytest

from ifv_training.opsd import (
    build_hint_proposal,
    build_opsd_attempt_record,
    build_proposer_prompt,
    build_student_messages,
    build_teacher_messages,
    locate_failure_site,
    parse_proposer_response,
    verify_repair,
)
from ifv_training.opsd_verifier import merge_suffix_into_trace, verify_continuation_pair
from ifv_training.opsd_runtime import QwenContinuationAdapter
from src.orchestrator.llm_backend import LLMResponse
from src.orchestrator.runtime_events import CaseRuntimeStore
from src.orchestrator.stage_runner import StageStep


def _trace() -> dict:
    return {
        "image_id": "episode-1",
        "state": {
            "all_steps": [
                {
                    "stage": "unified_react",
                    "action_type": "tool_call",
                    "metadata": {
                        "interaction_id": "call-1",
                        "policy_input": {
                            "system_instruction": "system",
                            "input_payload": [{"role": "user", "content": "inspect"}],
                            "tools": [],
                        },
                        "policy_action": {
                            "type": "tool_call",
                            "name": "text_search",
                            "arguments": {"query": "neutral terms"},
                        },
                    },
                },
                {
                    "stage": "unified_judgment",
                    "action_type": "output",
                    "metadata": {
                        "interaction_id": "judgment-1",
                        "policy_input": {
                            "system_instruction": "judge",
                            "input_payload": [{"role": "user", "content": "judge"}],
                            "tools": [],
                        },
                        "policy_action": {"verdict": "real"},
                    },
                },
            ]
        },
    }


def test_locates_exact_raw_step_and_separates_teacher_student_messages() -> None:
    site = locate_failure_site(
        _trace(),
        {"failures": [{"location": "state.all_steps[0].metadata.policy_action"}]},
    )
    assert site is not None
    assert site.source_step_index == 0
    hint = build_hint_proposal(
        text="Check the unresolved event relation before deciding.",
        level=1,
        provider="qwen_local",
        model="teacher",
        candidate_id="candidate-1",
        public_failure_context={"current_action": site.policy_action},
    )
    assert build_student_messages(site)[-1]["content"] == "inspect"
    assert build_teacher_messages(site, hint)[-1]["content"] == hint.text
    prompt = build_proposer_prompt(
        failure_site=site,
        public_trace_context={"observed_failure": "protocol"},
        hint_count=2,
    )
    assert "real/fake label" in prompt
    assert "expected_verdict" not in prompt


def test_hint_boundary_and_primary_repair_gate() -> None:
    with pytest.raises(ValueError, match="hint audit failed"):
        build_hint_proposal(
            text="The answer is fake.",
            level=1,
            provider="qwen_local",
            model="teacher",
            candidate_id="candidate-1",
            public_failure_context={},
        )
    site = locate_failure_site(
        _trace(),
        {"failures": [{"location": "state.all_steps[0]"}]},
    )
    assert site is not None
    verification = verify_repair(
        local_pass=True,
        recorded_verdict="fake",
        expected_verdict="fake",
        strict_trace_audit_pass=True,
        full_episode_pass=True,
    )
    attempt = build_opsd_attempt_record(
        candidate_id="candidate-1",
        failure_site=site,
        hint=build_hint_proposal(
            text="Inspect the unresolved relation.",
            level=1,
            provider="qwen_local",
            model="teacher",
            candidate_id="candidate-1",
            public_failure_context={},
        ),
        verification=verification,
    )
    assert attempt["accepted"] is True
    assert attempt["scaffold_only"] is False
    assert attempt["case_id"] == ""
    assert attempt["repair_tier"] == "causal_episode_pass"


def test_attempt_record_preserves_case_and_episode_identity() -> None:
    site = locate_failure_site(_trace(), {"failures": []})
    assert site is None
    site = locate_failure_site(
        _trace(), {"failures": [{"location": "state.all_steps[0]"}]}
    )
    assert site is not None
    hint = build_hint_proposal(
        text="Inspect the unresolved relation.",
        level=1,
        provider="qwen_local",
        model="teacher",
        candidate_id="candidate-1",
        public_failure_context={},
    )
    record = build_opsd_attempt_record(
        candidate_id="candidate-1",
        case_id="case-1",
        episode_id="episode-1",
        failure_site=site,
        hint=hint,
        verification=verify_repair(
            local_pass=True,
            recorded_verdict="fake",
            expected_verdict="fake",
            strict_trace_audit_pass=True,
            full_episode_pass=True,
        ),
    )
    assert record["case_id"] == "case-1"
    assert record["episode_id"] == "episode-1"


def test_terminal_mismatch_has_no_observed_site() -> None:
    assert locate_failure_site(_trace(), {"failures": []}) is None
    assert parse_proposer_response({"hints": ["one", "one", "two"]}) == ["one", "two"]


def test_primary_repair_requires_verdict_match_and_public_proposer_context() -> None:
    site = locate_failure_site(
        _trace(), {"failures": [{"location": "state.all_steps[0]"}]}
    )
    assert site is not None
    with pytest.raises(ValueError, match="private field"):
        build_proposer_prompt(
            failure_site=site,
            public_trace_context={"expected_verdict": "fake"},
            hint_count=1,
        )
    result = verify_repair(
        local_pass=True,
        recorded_verdict="real",
        expected_verdict="fake",
        strict_trace_audit_pass=True,
        full_episode_pass=True,
    )
    assert result.repair_tier == "local_pass_downstream"
    assert result.accepted_for_primary_psd is False


def test_opsd_suffix_merge_keeps_teacher_actions_out_of_student_trace() -> None:
    base = {
        "image_id": "episode-1",
        "state": {"all_steps": []},
    }
    teacher = [
        StageStep(
            stage_name="opsd_teacher_repair",
            action_type="tool_call",
            tool_name="text_search",
            tool_args={"queries": ["event relation"]},
        )
    ]
    student = [
        StageStep(
            stage_name="opsd_student_continuation",
            action_type="tool_call",
            tool_name="visit",
            tool_args={"url": ["https://example.org/source"]},
        )
    ]

    teacher_trace = merge_suffix_into_trace(base, teacher, role="teacher")
    student_trace = merge_suffix_into_trace(base, student, role="student")

    assert len(teacher_trace["state"]["all_steps"]) == 1
    assert len(student_trace["state"]["all_steps"]) == 1
    assert student_trace["state"]["all_steps"][0]["tool_name"] == "visit"
    assert "event relation" not in str(student_trace)


def test_opsd_pair_requires_complete_student_episode() -> None:
    result, artifacts = verify_continuation_pair(
        base_trace={"image_id": "episode-1", "state": {"all_steps": []}},
        teacher_steps=[StageStep(action_type="tool_call", tool_name="text_search")],
        student_steps=[StageStep(action_type="tool_call", tool_name="text_search")],
        student_episode_trace=None,
        gold={"factual_status": "fake", "case_id": "episode-1"},
        local_pass=True,
    )

    assert result.accepted_for_primary_psd is False
    assert result.repair_tier == "local_pass_downstream"
    assert "full_student_episode_trace_required" in result.reasons
    assert artifacts["student_episode_verified"] is False


def test_privileged_proposer_request_is_recorded_in_runtime_ledger(tmp_path) -> None:
    class ProposerBackend:
        provider = "qwen_local"
        wire_api = "chat_completions"
        model_name = "teacher"

        async def get_response(self, _messages, **_kwargs):
            return LLMResponse(
                text='{"hints":["Check the unresolved event relation."]}',
                prompt_tokens=12,
                completion_tokens=8,
                raw={"usage": {"input_tokens": 12, "output_tokens": 8}},
            )

    site = locate_failure_site(
        _trace(),
        {"failures": [{"location": "state.all_steps[0]"}]},
    )
    assert site is not None
    store = CaseRuntimeStore(tmp_path / "runtime", case_id="case-1", attempt_id="a1")
    adapter = QwenContinuationAdapter(
        llm=ProposerBackend(),
        tools=[],
        image_path="",
        runtime_store=store,
        require_runtime_archive=False,
    )

    import asyncio

    proposals = asyncio.run(
        adapter.propose_hints(
            failure_site=site,
            public_trace_context={"observed_failure": "protocol"},
            hint_count=1,
        )
    )

    assert len(proposals) == 1
    manifests = list((store.root / "context").glob("*.json"))
    assert len(manifests) == 1
    assert '"stage": "opsd_proposer"' in manifests[0].read_text(encoding="utf-8")
