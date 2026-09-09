"""Provider-neutral contracts for IFV on-policy privileged self-distillation.

This module owns the boundary between a privileged repair controller and the
student-visible raw-history runtime.  It does not run private gold through the
policy and never serializes a hint into a student prompt.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping, Sequence

from . import _repo_import  # noqa: F401
from .psd import audit_hint
from src.orchestrator.runtime_events import reconstruct_archived_request


OPSD_SCHEMA_VERSION = "ifv-opsd-v1"
TRAINABLE_HINT_LEVELS = frozenset({1, 2, 3})
_PRIVATE_CONTEXT_KEYS = frozenset(
    {
        "gold", "evaluation_gold", "factual_status", "expected_verdict",
        "ground_truth", "private_gold", "acceptable_evidence", "private_target",
    }
)


def _text(value: Any) -> str:
    return str(value or "").strip()


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _rows(value: Any) -> list[Mapping[str, Any]]:
    return [item for item in value if isinstance(item, Mapping)] if isinstance(value, list) else []


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _assert_public_context(value: Any, *, path: str = "") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = str(key)
            child_path = f"{path}.{name}" if path else name
            if name.casefold() in _PRIVATE_CONTEXT_KEYS:
                raise ValueError(f"private field in public proposer context: {child_path}")
            _assert_public_context(child, path=child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_public_context(child, path=f"{path}[{index}]")


@dataclass(frozen=True)
class FailureSite:
    """One exact raw-history policy decision selected for repair."""

    step_index: int
    step_id: str
    stage: str
    example_type: str
    policy_input: Mapping[str, Any]
    policy_action: Mapping[str, Any]
    source_step_index: int | None = None
    context_request_id: str = ""
    runtime_store_path: str = ""

    def public_record(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "source_step_index": self.source_step_index,
            "step_id": self.step_id,
            "stage": self.stage,
            "example_type": self.example_type,
            "policy_input_sha256": _sha(self.policy_input),
            "policy_action_sha256": _sha(self.policy_action),
            "context_request_id": self.context_request_id,
        }


@dataclass(frozen=True)
class HintProposal:
    text: str
    level: int
    provider: str
    model: str
    proposal_id: str
    audit: Mapping[str, Any]


@dataclass(frozen=True)
class VerificationResult:
    local_pass: bool
    full_episode_pass: bool
    strict_trace_audit_pass: bool
    recorded_verdict: str
    expected_verdict: str
    repair_tier: str
    reasons: tuple[str, ...] = ()

    @property
    def accepted_for_primary_psd(self) -> bool:
        return (
            self.repair_tier == "causal_episode_pass"
            and self.local_pass
            and self.full_episode_pass
            and self.strict_trace_audit_pass
            and bool(_text(self.recorded_verdict))
            and _text(self.recorded_verdict).casefold()
            == _text(self.expected_verdict).casefold()
        )


def _step_id(step: Mapping[str, Any], episode_id: str, index: int) -> str:
    metadata = _mapping(step.get("metadata"))
    interaction_id = _text(metadata.get("interaction_id"))
    kind = _text(step.get("stage") or step.get("stage_name")) or "step"
    kind = re.sub(r"[^a-zA-Z0-9_.-]+", "_", kind)
    return f"{episode_id}:{kind}:{interaction_id or index}"


def project_policy_steps(trace: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Project raw trace steps while retaining their original array index."""

    state = _mapping(trace.get("state"))
    episode_id = _text(trace.get("image_id") or state.get("image_id"))
    result: list[dict[str, Any]] = []
    for source_index, raw in enumerate(_rows(state.get("all_steps"))):
        policy = _mapping(raw.get("metadata"))
        metadata = policy
        policy_input = policy.get("policy_input")
        policy_action = policy.get("policy_action")
        if not isinstance(policy_input, Mapping) or not isinstance(policy_action, Mapping):
            continue
        stage = _text(raw.get("stage") or raw.get("stage_name"))
        example_type = {
            "unified_react": "react",
            "unified_judgment": "judgment",
        }.get(stage, stage)
        result.append(
            {
                "step_id": _step_id(raw, episode_id, source_index),
                "source_step_index": source_index,
                "stage": stage,
                "example_type": example_type,
                "policy_input": dict(policy_input),
                "policy_action": dict(policy_action),
                "action_type": _text(raw.get("action_type")),
                "tool_name": _text(raw.get("tool_name")),
                "context_request_id": _text(metadata.get("context_request_id")),
                "runtime_store_path": _text(
                    _mapping(_mapping(trace.get("state")).get("runtime_store")).get(
                        "runtime_path"
                    )
                ),
            }
        )
    return result


def locate_failure_site(
    trace: Mapping[str, Any],
    audit: Mapping[str, Any] | None = None,
) -> FailureSite | None:
    """Locate an observed failure; terminal mismatch alone stays unattributed."""

    steps = project_policy_steps(trace)
    if not steps:
        return None
    failures = _rows(_mapping(audit).get("failures"))
    for failure in failures:
        raw_location = _text(failure.get("location") or failure.get("path") or failure.get("message"))
        match = re.search(r"all_steps\[(\d+)\]", raw_location)
        if not match:
            continue
        source_index = int(match.group(1))
        for index, step in enumerate(steps):
            if step["source_step_index"] == source_index:
                return FailureSite(
                    step_index=index,
                    source_step_index=source_index,
                    step_id=step["step_id"],
                    stage=step["stage"],
                    example_type=step["example_type"],
                    policy_input=step["policy_input"],
                    policy_action=step["policy_action"],
                    context_request_id=step["context_request_id"],
                    runtime_store_path=step["runtime_store_path"],
                )
    for index, step in enumerate(steps):
        if step["action_type"] in {"format_error", "output_rejected", "policy_replan"}:
            return FailureSite(
                step_index=index,
                source_step_index=step["source_step_index"],
                step_id=step["step_id"],
                stage=step["stage"],
                example_type=step["example_type"],
                policy_input=step["policy_input"],
                    policy_action=step["policy_action"],
                    context_request_id=step["context_request_id"],
                    runtime_store_path=step["runtime_store_path"],
            )
    return None


def restore_failure_site_archive(
    failure_site: FailureSite,
    *,
    require: bool = False,
) -> FailureSite:
    """Replace snapshot placeholders with the immutable provider request."""

    if not failure_site.context_request_id or not failure_site.runtime_store_path:
        if require:
            raise ValueError("failure site lacks runtime archive binding")
        return failure_site
    reconstructed = reconstruct_archived_request(
        failure_site.runtime_store_path,
        failure_site.context_request_id,
    )
    policy_input = dict(failure_site.policy_input)
    for key in ("system_instruction", "input_payload", "tools", "response_format"):
        if key in reconstructed:
            policy_input[key] = reconstructed[key]
    return replace(failure_site, policy_input=policy_input)


def validate_hint(
    hint: str,
    *,
    level: int,
    public_failure_context: Mapping[str, Any],
    private_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit a proposed hint and reject answer-bearing or exact-action hints."""

    if level not in TRAINABLE_HINT_LEVELS:
        return {"passed": False, "errors": ["hint_level_not_trainable"]}
    row = {
        **dict(public_failure_context),
        **dict(private_context or {}),
    }
    result = audit_hint(row, hint=hint, hint_level=level)
    lowered = _text(hint).casefold()
    forbidden = ("exact query", "exact url", "private gold", "ground truth", "expected verdict")
    errors = list(result.get("errors", []))
    if any(term in lowered for term in forbidden):
        errors.append("procedural_hint_boundary_leak")
    if "{" in hint or "}" in hint:
        errors.append("structured_action_scaffold_leak")
    return {**result, "passed": not errors, "errors": sorted(set(errors))}


def build_hint_proposal(
    *,
    text: str,
    level: int,
    provider: str,
    model: str,
    candidate_id: str,
    public_failure_context: Mapping[str, Any],
    private_context: Mapping[str, Any] | None = None,
) -> HintProposal:
    audit = validate_hint(
        text,
        level=level,
        public_failure_context=public_failure_context,
        private_context=private_context,
    )
    if audit.get("passed") is not True:
        raise ValueError("hint audit failed: " + ",".join(audit.get("errors", [])))
    proposal_id = "opsd-hint:" + _sha({"candidate_id": candidate_id, "hint": text, "level": level})
    return HintProposal(
        text=_text(text),
        level=level,
        provider=_text(provider),
        model=_text(model),
        proposal_id=proposal_id,
        audit=audit,
    )


def build_teacher_messages(
    failure_site: FailureSite,
    hint: HintProposal,
) -> list[dict[str, Any]]:
    """Insert hint immediately before the failed assistant request."""

    payload = failure_site.policy_input.get("input_payload")
    if not isinstance(payload, list):
        raise ValueError("failure policy_input.input_payload must be a message list")
    messages = [dict(item) for item in payload if isinstance(item, Mapping)]
    if not messages:
        raise ValueError("failure policy input has no messages")
    return [*messages, {"role": "user", "content": hint.text}]


def build_student_messages(failure_site: FailureSite) -> list[dict[str, Any]]:
    payload = failure_site.policy_input.get("input_payload")
    if not isinstance(payload, list):
        raise ValueError("failure policy_input.input_payload must be a message list")
    return [dict(item) for item in payload if isinstance(item, Mapping)]


def verify_repair(
    *,
    local_pass: bool,
    recorded_verdict: str,
    expected_verdict: str,
    strict_trace_audit_pass: bool,
    full_episode_pass: bool,
    downstream_patch_count: int = 0,
    scaffold_only: bool = False,
    reasons: Iterable[str] = (),
) -> VerificationResult:
    if scaffold_only:
        tier = "scaffold_only"
    elif (
        full_episode_pass
        and strict_trace_audit_pass
        and downstream_patch_count == 0
        and bool(_text(recorded_verdict))
        and _text(recorded_verdict).casefold()
        == _text(expected_verdict).casefold()
    ):
        tier = "causal_episode_pass"
    elif local_pass:
        tier = "local_pass_downstream"
    else:
        tier = "unrepairable"
    return VerificationResult(
        local_pass=bool(local_pass),
        full_episode_pass=bool(full_episode_pass),
        strict_trace_audit_pass=bool(strict_trace_audit_pass),
        recorded_verdict=_text(recorded_verdict),
        expected_verdict=_text(expected_verdict),
        repair_tier=tier,
        reasons=tuple(_text(item) for item in reasons if _text(item)),
    )


def build_opsd_attempt_record(
    *,
    candidate_id: str,
    failure_site: FailureSite,
    hint: HintProposal,
    verification: VerificationResult,
    student_prompt_ids: Sequence[int] = (),
    teacher_prompt_ids: Sequence[int] = (),
    completion_ids: Sequence[int] = (),
    teacher_token_capture: Mapping[str, Any] | None = None,
    source_trace_sha256: str = "",
    l5_scaffold: bool = False,
    case_id: str = "",
    episode_id: str = "",
) -> dict[str, Any]:
    if verification.accepted_for_primary_psd and l5_scaffold:
        raise ValueError("L5 scaffold cannot be a primary PSD repair")
    return {
        "schema_version": OPSD_SCHEMA_VERSION,
        "candidate_id": _text(candidate_id),
        "attempt_id": hint.proposal_id,
        "case_id": _text(case_id),
        "episode_id": _text(episode_id),
        "repair_step_id": failure_site.step_id,
        "repair_site": failure_site.public_record(),
        "repair_tier": verification.repair_tier,
        "local_pass": verification.local_pass,
        "full_episode_pass": verification.full_episode_pass,
        "strict_trace_audit_pass": verification.strict_trace_audit_pass,
        "hint_record": {
            "text": hint.text,
            "level": hint.level,
            "provider": hint.provider,
            "model": hint.model,
            "proposal_id": hint.proposal_id,
            "audit": dict(hint.audit),
        },
        "verification": {
            "local_pass": verification.local_pass,
            "full_episode_pass": verification.full_episode_pass,
            "strict_trace_audit_pass": verification.strict_trace_audit_pass,
            "repair_tier": verification.repair_tier,
            "recorded_verdict": verification.recorded_verdict,
            "expected_verdict": verification.expected_verdict,
            "reasons": list(verification.reasons),
        },
        "accepted": verification.accepted_for_primary_psd,
        "scaffold_only": bool(l5_scaffold),
        "student_prompt_ids": list(student_prompt_ids),
        "teacher_prompt_ids": list(teacher_prompt_ids),
        "completion_ids": list(completion_ids),
        "repair_rollout_token_capture": (
            dict(teacher_token_capture) if teacher_token_capture else {}
        ),
        "source_trace_sha256": _text(source_trace_sha256),
    }


def build_proposer_prompt(
    *,
    failure_site: FailureSite,
    public_trace_context: Mapping[str, Any],
    hint_count: int,
    private_context: Mapping[str, Any] | None = None,
) -> str:
    """Create a public-only proposer prompt with no answer-bearing fields."""

    _assert_public_context(public_trace_context)

    visible_action = {
        "stage": failure_site.stage,
        "example_type": failure_site.example_type,
        "current_action": failure_site.policy_action,
        "current_messages": build_student_messages(failure_site),
        "trace_context": dict(public_trace_context),
    }
    privileged_reference = dict(private_context or {})
    if privileged_reference:
        visible_action["privileged_reference"] = privileged_reference
    return (
        "You are a privileged procedural repair proposer for an image fact-checking "
        "agent. Produce short procedural hints for the next decision. Use the "
        "privileged reference only to identify the missing relation or check; do "
        "not state "
        "the real/fake label, private target, exact query, URL, evidence ID, or "
        "exact tool arguments. A hint should identify what to check or what kind "
        "of missing relation/condition to resolve, without supplying the answer. "
        f"Return JSON exactly as {{\"hints\":[\"...\"]}} with at most {max(1, int(hint_count))} hints.\n\n"
        + json.dumps(visible_action, ensure_ascii=False, sort_keys=True)
    )


def parse_proposer_response(value: Any, *, limit: int = 4) -> list[str]:
    if isinstance(value, Mapping):
        raw = value.get("hints")
    else:
        raw = None
    if not isinstance(raw, list):
        return []
    result: list[str] = []
    for item in raw:
        hint = _text(item)
        if hint and hint not in result:
            result.append(hint)
    return result[: max(1, int(limit))]
