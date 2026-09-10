from __future__ import annotations

from pathlib import Path

import pytest

from ifv_training.psd_repair import (
    PSDModelRoles,
    PSD_SEMANTIC_LOCALIZATION_SCHEMA_VERSION,
    build_hint_proposal,
    build_psd_attempt_record,
    build_proposer_prompt,
    build_student_messages,
    build_teacher_messages,
    locate_failure_site,
    parse_proposer_response,
    verify_repair,
)
import ifv_training.psd_repair_verifier as verifier_module
from ifv_training.psd_repair_verifier import (
    PSD_LOCAL_VERIFICATION_SCHEMA_VERSION,
    merge_suffix_into_trace,
    validate_local_verification,
    verify_continuation_pair,
)
from ifv_training.psd_repair_runtime import QwenContinuationAdapter
from src.orchestrator.llm_backend import LLMResponse
from src.orchestrator.runtime_events import CaseRuntimeStore
from src.orchestrator.stage_runner import StageStep


def _roles() -> PSDModelRoles:
    return PSDModelRoles(
        hint_constructor_provider="frontier",
        hint_constructor_model="strong-hint-constructor",
        frozen_self_teacher_provider="qwen_local",
        frozen_self_teacher_model="qwen-round-start",
        round_start_checkpoint="checkpoint-round-0",
        trainable_student_provider="qwen_local",
        trainable_student_model="qwen-round-start",
        trainable_student_initial_checkpoint="checkpoint-round-0",
    )


def _local_verification(step_id: str, *, passed: bool = True) -> dict:
    return {
        "schema_version": PSD_LOCAL_VERIFICATION_SCHEMA_VERSION,
        "repair_step_id": step_id,
        "passed": passed,
        "verifier": {
            "kind": "task",
            "name": "ifv-fact-check-local-verifier",
            "version": "v1",
        },
        "checks": [
            {"name": "missing_relation_resolved", "passed": passed},
        ],
        "evidence": [
            {"observation_id": "call-text_search", "status": "checked"},
        ],
    }


def _complete_episode(*, image_id: str, verdict: str) -> dict:
    judgment = {"fact_check_report": {"summary": "verified"}}
    return {
        "image_id": image_id,
        "verdict": verdict,
        "judgment": judgment,
        "state": {
            "all_steps": [
                {
                    "stage": "unified_judgment",
                    "action_type": "output",
                }
            ]
        },
    }


def _patch_episode_verifiers(monkeypatch: pytest.MonkeyPatch) -> None:
    class AuditReport:
        def failures(self, *, strict_scheduler: bool):
            assert strict_scheduler is True
            return []

    def fake_score(trace, _gold, *, score_metadata=None):
        del score_metadata
        correct = trace.get("image_id") == "hinted-pass"
        return {
            "result_correct": correct,
            "expected_verdict": "fake",
            "engineering_error": False,
        }, {}

    monkeypatch.setattr(verifier_module, "audit_trace", lambda _path: AuditReport())
    monkeypatch.setattr(verifier_module, "score_process_trace", fake_score)


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
        source_rollout_failed=True,
        hinted_local_pass=True,
        hinted_recorded_verdict="fake",
        expected_verdict="fake",
        hinted_strict_trace_audit_pass=True,
        hinted_episode_pass=True,
    )
    attempt = build_psd_attempt_record(
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
        model_roles=_roles(),
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
    record = build_psd_attempt_record(
        candidate_id="candidate-1",
        case_id="case-1",
        episode_id="episode-1",
        failure_site=site,
        hint=hint,
        model_roles=_roles(),
        verification=verify_repair(
            source_rollout_failed=True,
            hinted_local_pass=True,
            hinted_recorded_verdict="fake",
            expected_verdict="fake",
            hinted_strict_trace_audit_pass=True,
            hinted_episode_pass=True,
        ),
    )
    assert record["case_id"] == "case-1"
    assert record["episode_id"] == "episode-1"


def test_terminal_mismatch_has_no_observed_site() -> None:
    assert locate_failure_site(_trace(), {"failures": []}) is None
    assert parse_proposer_response({"hints": ["one", "one", "two"]}) == ["one", "two"]


def test_semantic_verifier_selects_a_student_reached_recoverable_step() -> None:
    report = {
        "schema_version": PSD_SEMANTIC_LOCALIZATION_SCHEMA_VERSION,
        "passed": True,
        "candidates": [
            {
                "source_step_index": 0,
                "category": "wrong_search_direction",
                "recoverable": True,
                "selected": True,
                "verifier": {
                    "kind": "task",
                    "name": "ifv-semantic-localizer",
                    "version": "v1",
                },
                "observed_basis": [
                    {"observation": "search omitted the depicted event relation"}
                ],
            }
        ],
    }

    site = locate_failure_site(_trace(), {"failures": []}, report)

    assert site is not None
    assert site.source_step_index == 0
    assert site.localization_kind == "semantic_task_verifier"
    assert site.semantic_category == "wrong_search_direction"
    assert site.localization_basis_sha256


def test_verdict_mismatch_alone_cannot_select_judgment() -> None:
    report = {
        "schema_version": PSD_SEMANTIC_LOCALIZATION_SCHEMA_VERSION,
        "passed": True,
        "candidates": [
            {
                "source_step_index": 1,
                "category": "verdict_mismatch",
                "recoverable": True,
                "selected": True,
                "verifier": {
                    "kind": "task",
                    "name": "ifv-semantic-localizer",
                    "version": "v1",
                },
                "observed_basis": ["final label differs"],
            }
        ],
    }

    assert locate_failure_site(_trace(), {"failures": []}, report) is None


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
        source_rollout_failed=True,
        hinted_local_pass=True,
        hinted_recorded_verdict="real",
        expected_verdict="fake",
        hinted_strict_trace_audit_pass=True,
        hinted_episode_pass=True,
    )
    assert result.repair_tier == "local_pass_downstream"
    assert result.accepted_for_primary_psd is False


def test_psd_suffix_merge_keeps_teacher_actions_out_of_student_trace() -> None:
    base = {
        "image_id": "episode-1",
        "state": {"all_steps": []},
    }
    teacher = [
        StageStep(
            stage_name="psd_teacher_repair",
            action_type="tool_call",
            tool_name="text_search",
            tool_args={"queries": ["event relation"]},
        )
    ]
    student = [
        StageStep(
            stage_name="psd_student_continuation",
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


def test_failed_source_and_passing_hinted_teacher_are_accepted_even_if_student_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_episode_verifiers(monkeypatch)
    repair_step_id = "episode-1:unified_react:call-1"
    result, artifacts = verify_continuation_pair(
        base_trace={"image_id": "source-fail", "state": {"all_steps": []}},
        teacher_steps=[StageStep(action_type="tool_call", tool_name="text_search")],
        student_steps=[StageStep(action_type="tool_call", tool_name="text_search")],
        hinted_teacher_episode_trace=_complete_episode(
            image_id="hinted-pass", verdict="fake"
        ),
        unhinted_student_episode_trace=_complete_episode(
            image_id="student-still-fails", verdict="real"
        ),
        gold={"factual_status": "fake", "case_id": "episode-1"},
        local_verification=_local_verification(repair_step_id),
        repair_step_id=repair_step_id,
    )

    assert result.source_rollout_failed is True
    assert result.hinted_local_pass is True
    assert result.hinted_episode_pass is True
    assert result.accepted_for_primary_psd is True
    assert artifacts["unhinted_student_diagnostic"]["passed"] is True
    assert artifacts["unhinted_student_affects_acceptance"] is False


def test_hinted_continuation_that_fails_task_verification_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_episode_verifiers(monkeypatch)
    repair_step_id = "episode-1:unified_react:call-1"

    result, _artifacts = verify_continuation_pair(
        base_trace={"image_id": "source-fail", "state": {"all_steps": []}},
        teacher_steps=[StageStep(action_type="tool_call", tool_name="text_search")],
        student_steps=[StageStep(action_type="tool_call", tool_name="text_search")],
        hinted_teacher_episode_trace=_complete_episode(
            image_id="hinted-still-fails", verdict="real"
        ),
        unhinted_student_episode_trace=None,
        gold={"factual_status": "fake", "case_id": "episode-1"},
        local_verification=_local_verification(repair_step_id, passed=False),
        repair_step_id=repair_step_id,
    )

    assert result.hinted_local_pass is False
    assert result.hinted_episode_pass is False
    assert result.accepted_for_primary_psd is False


def test_tool_calls_alone_cannot_become_local_pass() -> None:
    result = validate_local_verification(
        {"passed": True, "tool_call_count": 2},
        repair_step_id="episode-1:unified_react:call-1",
    )

    assert result["valid"] is False
    assert result["passed"] is False
    assert "local_verifier_schema_invalid" in result["errors"]


def test_privileged_proposer_request_is_recorded_in_runtime_ledger(tmp_path) -> None:
    class ProposerBackend:
        provider = "frontier"
        wire_api = "responses"
        model_name = "strong-hint-constructor"

        async def get_response(self, _messages, **_kwargs):
            return LLMResponse(
                text='{"hints":["Check the unresolved event relation."]}',
                prompt_tokens=12,
                completion_tokens=8,
                raw={"usage": {"input_tokens": 12, "output_tokens": 8}},
            )

    class PolicyBackend:
        provider = "qwen_local"
        wire_api = "chat_completions"
        model_name = "qwen-round-start"

    site = locate_failure_site(
        _trace(),
        {"failures": [{"location": "state.all_steps[0]"}]},
    )
    assert site is not None
    store = CaseRuntimeStore(tmp_path / "runtime", case_id="case-1", attempt_id="a1")
    adapter = QwenContinuationAdapter(
        policy_llm=PolicyBackend(),
        hint_constructor_llm=ProposerBackend(),
        model_roles=_roles(),
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
    assert proposals[0].provider == "frontier"
    assert _roles().record()["hint_constructor"]["supplies_training_distribution"] is False
    assert _roles().record()["frozen_self_teacher"]["supplies_training_distribution"] is True
    manifests = list((store.root / "context").glob("*.json"))
    assert len(manifests) == 1
    assert '"stage": "psd_proposer"' in manifests[0].read_text(encoding="utf-8")


def test_repository_has_no_legacy_psd_repair_name() -> None:
    repository = Path(__file__).resolve().parents[2]
    forbidden = ("OP" + "SD", "op" + "sd", "ifv-" + "op" + "sd")
    residual: list[str] = []
    for path in repository.rglob("*"):
        if not path.is_file() or any(
            part in {".git", "__pycache__"} for part in path.parts
        ):
            continue
        if path.suffix not in {".py", ".md", ".sh", ".toml", ".json", ".env"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(token in text or token in path.name for token in forbidden):
            residual.append(str(path.relative_to(repository)))
    assert residual == []
