"""Pure canonical-trace to policy-example exporter."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Mapping, Protocol, Sequence

from src.trajectory.schema import PolicyExample
from src.orchestrator.evidence_semantics import (
    evidence_direction_is_coherent,
    evidence_is_qualified_for_stance,
    required_assessment_stances,
)
from src.orchestrator.tool_result import parse_tool_result


FORBIDDEN_PRIVATE_KEYS = frozenset(
    {
        "gold",
        "evaluation_gold",
        "factual_status",
        "acceptable_evidence",
        "expected_status",
        "ground_truth",
        "teacher_score",
        "process_metrics",
        "source_access_policy",
        "excluded_domains",
        "excluded_urls",
    }
)


class TokenizerAdapter(Protocol):
    tokenizer_id: str

    def encode(self, text: str) -> Sequence[int]:
        """Encode one canonical JSON string without adding implicit tokens."""


class Utf8ByteTokenizer:
    """Stable default adapter; model-specific tokenizers remain injectable."""

    tokenizer_id = "utf8-byte-v1"

    def encode(self, text: str) -> Sequence[int]:
        return list(text.encode("utf-8"))


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _rows(value: Any) -> List[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _assert_no_private_data(value: Any, path: str = "") -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            child_path = f"{path}.{key}" if path else key
            if key.casefold() in FORBIDDEN_PRIVATE_KEYS:
                raise ValueError(
                    f"private/evaluator field is forbidden in policy input: {child_path}"
                )
            _assert_no_private_data(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_private_data(child, f"{path}[{index}]")


def _observation_refs(policy_input: Mapping[str, Any]) -> List[str]:
    refs: List[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            for key in (
                "call_id",
                "function_call_id",
                "evidence_id",
                "finding_id",
                "task_id",
                "fact_id",
                "interaction_id",
            ):
                rendered = str(value.get(key, "") or "").strip()
                if rendered:
                    refs.append(rendered)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(policy_input.get("input_payload"))
    return list(dict.fromkeys(refs))[:64]


def _example_type(stage: str) -> str | None:
    if stage in {
        "image_only_investigation",
        "image_only_discrepancy_investigation",
    }:
        return "react"
    if stage == "image_only_reflection":
        return "reflection"
    if stage == "image_only_judgment":
        return "judgment"
    if stage == "image_only_evidence_decision":
        return "evidence_decision"
    if stage == "image_only_discrepancy_decision":
        return "discrepancy_decision"
    if stage == "image_only_query_concept_extraction":
        return "query_concept_extraction"
    if stage == "image_only_query_replan":
        return "query_replan"
    if stage in {
        "image_only_planning",
        "image_only_attribution_planning",
    }:
        return "planning"
    if stage == "image_account_planning":
        return "image_account_planning"
    if stage == "image_only_discrepancy_judgment":
        return "judgment"
    return None


def _v4_claim_has_directional_chain(
    *,
    claim_id: str,
    stance: str,
    selected_evidence_ids: set[str],
    claims: Mapping[str, Mapping[str, Any]],
    tasks: Mapping[str, Mapping[str, Any]],
    evidence: Mapping[str, Mapping[str, Any]],
    findings: Mapping[str, Mapping[str, Any]],
    successful_call_ids: set[str],
    selected_finding_ids: set[str] | None = None,
) -> bool:
    claim = claims.get(claim_id)
    if claim is None:
        return False
    claim_fact_id = str(claim.get("fact_id", "")).strip()
    for finding_id, finding in findings.items():
        if selected_finding_ids is not None and finding_id not in selected_finding_ids:
            continue
        if str(finding.get("stance", "")).strip() != stance:
            continue
        task_id = str(finding.get("task_id", "")).strip()
        task = tasks.get(task_id)
        if task is None or claim_id not in {
            str(item) for item in task.get("claim_ids", []) or []
        }:
            continue
        if claim_fact_id not in {
            str(item) for item in finding.get("fact_ids", []) or []
        }:
            continue
        for evidence_id in finding.get("evidence_ids", []) or []:
            evidence_id = str(evidence_id)
            row = evidence.get(evidence_id)
            if (
                evidence_id in selected_evidence_ids
                and row is not None
                and str(row.get("task_id", "")).strip() == task_id
                and evidence_is_qualified_for_stance(row, stance)
                and str(row.get("function_call_id", "")).strip()
                in successful_call_ids
            ):
                return True
    return False


def _v4_quality_gate(trace: Mapping[str, Any], state: Mapping[str, Any]) -> None:
    """Reject v4 episodes that cannot teach claim/discrepancy alignment."""

    if str(trace.get("termination", "")) != "success":
        raise ValueError("v4 policy export requires a successful trace")
    investigation = _mapping(state.get("investigation_state"))
    claims = {
        str(item.get("claim_id", "")): item
        for item in _rows(investigation.get("image_claims"))
        if str(item.get("claim_id", ""))
    }
    hypotheses = {
        str(item.get("hypothesis_id", "")): item
        for item in _rows(investigation.get("search_hypotheses"))
        if str(item.get("hypothesis_id", ""))
    }
    tasks = {
        str(item.get("task_id", "")): item
        for item in _rows(investigation.get("tasks"))
        if str(item.get("task_id", ""))
    }
    evidence = {
        str(item.get("evidence_id", "")): item
        for item in _rows(investigation.get("evidence"))
        if str(item.get("evidence_id", ""))
    }
    findings = {
        str(item.get("finding_id", "")): item
        for item in _rows(investigation.get("findings"))
        if str(item.get("finding_id", ""))
    }
    successful_call_ids: set[str] = set()
    for step in _rows(state.get("all_steps")):
        if str(step.get("action_type", "")) != "tool_call":
            continue
        metadata = _mapping(step.get("metadata"))
        call_id = str(metadata.get("function_call_id", "")).strip()
        try:
            _, succeeded = parse_tool_result(str(step.get("tool_result", "")))
        except Exception:
            succeeded = False
        if call_id and succeeded and metadata.get("tool_success") is not False:
            successful_call_ids.add(call_id)
    if any(
        not str(item.get("function_call_id", "")).strip()
        or str(item.get("function_call_id", "")).strip() not in successful_call_ids
        for item in evidence.values()
    ):
        raise ValueError("v4 policy export rejects Evidence without a successful call")
    if any(
        str(item.get("stance", "")) in {"support", "refute"}
        and not evidence_direction_is_coherent(
            item,
            str(item.get("stance", "")),
        )
        for item in evidence.values()
    ):
        raise ValueError("v4 policy export rejects incoherent Evidence stance")
    facts = {
        str(item.get("fact_id", "")): item
        for item in _rows(investigation.get("facts"))
        if str(item.get("fact_id", ""))
    }
    if not claims or not any(
        str(item.get("salience", "")) == "high" for item in claims.values()
    ):
        raise ValueError("v4 policy export requires high-salience ImageClaims")
    if investigation.get("core_verdict_fact_id"):
        raise ValueError("v4 policy export rejects active legacy core ownership")
    for assessment in _rows(investigation.get("claim_assessments")):
        claim_id = str(assessment.get("claim_id", ""))
        selected_ids = {
            str(item) for item in assessment.get("evidence_ids", []) or []
        }
        if claim_id not in claims or not selected_ids <= set(evidence):
            raise ValueError("v4 policy export rejects invalid ClaimAssessment Evidence")
        if any(
            claim_id
            not in {
                str(item)
                for item in tasks.get(
                    str(evidence[evidence_id].get("task_id", "")),
                    {},
                ).get("claim_ids", [])
                or []
            }
            for evidence_id in selected_ids
        ):
            raise ValueError("v4 policy export rejects unowned ClaimAssessment Evidence")
        for stance in required_assessment_stances(
            str(assessment.get("assessment", ""))
        ):
            if not _v4_claim_has_directional_chain(
                claim_id=claim_id,
                stance=stance,
                selected_evidence_ids=selected_ids,
                claims=claims,
                tasks=tasks,
                evidence=evidence,
                findings=findings,
                successful_call_ids=successful_call_ids,
            ):
                raise ValueError(
                    "v4 policy export rejects ClaimAssessment/Evidence "
                    "direction mismatch"
                )
    for hypothesis_id, hypothesis in hypotheses.items():
        claim_ids = {str(item) for item in hypothesis.get("claim_ids", []) or []}
        task = tasks.get(str(hypothesis.get("task_id", "")))
        if (
            not claim_ids
            or not claim_ids <= set(claims)
            or task is None
            or str(task.get("hypothesis_id", "")) != hypothesis_id
            or {str(item) for item in task.get("claim_ids", []) or []}
            != claim_ids
        ):
            raise ValueError("v4 policy export rejects invalid hypothesis ownership")
    for discrepancy in _rows(investigation.get("material_discrepancies")):
        claim_ids = {
            str(item) for item in discrepancy.get("affected_claim_ids", []) or []
        }
        anchor_ids = {
            str(item)
            for item in discrepancy.get("visual_anchor_fact_ids", []) or []
        }
        evidence_ids = {
            str(item) for item in discrepancy.get("evidence_ids", []) or []
        }
        if not claim_ids or not claim_ids <= set(claims):
            raise ValueError("v4 policy export rejects unowned discrepancy claims")
        if not evidence_ids or not evidence_ids <= set(evidence):
            raise ValueError("v4 policy export rejects invalid discrepancy Evidence")
        if any(
            not anchor_ids
            & {str(item) for item in claims[claim_id].get("anchor_fact_ids", []) or []}
            for claim_id in claim_ids
        ):
            raise ValueError("v4 policy export rejects unanchored discrepancy")
        if any(
            anchor_id not in facts
            or str(
                _mapping(facts[anchor_id].get("origin")).get("type", "")
            )
            not in {"input_image", "ocr"}
            for anchor_id in anchor_ids
        ):
            raise ValueError("v4 policy export rejects non-visual discrepancy anchors")
        if any(
            not any(
                claim_id
                in {
                    str(item)
                    for item in tasks.get(
                        str(evidence[evidence_id].get("task_id", "")),
                        {},
                    ).get("claim_ids", [])
                    or []
                }
                for evidence_id in evidence_ids
            )
            for claim_id in claim_ids
        ):
            raise ValueError("v4 policy export rejects discrepancy Evidence misalignment")
        if (
            str(discrepancy.get("materiality", "")) == "decisive"
            and str(discrepancy.get("status", "")) == "established"
            and any(
                not _v4_claim_has_directional_chain(
                    claim_id=claim_id,
                    stance="refute",
                    selected_evidence_ids=evidence_ids,
                    claims=claims,
                    tasks=tasks,
                    evidence=evidence,
                    findings=findings,
                    successful_call_ids=successful_call_ids,
                )
                for claim_id in claim_ids
            )
        ):
            raise ValueError(
                "v4 policy export rejects decisive discrepancy without "
                "qualified refute Evidence"
            )
    verdict = str(trace.get("verdict", ""))
    basis = _mapping(
        trace.get("verdict_basis")
        or investigation.get("discrepancy_verdict_basis")
    )
    judgment = _mapping(
        trace.get("judgment")
        or state.get("judgment")
        or investigation.get("discrepancy_judgment")
    )
    if str(judgment.get("verdict", "")) != verdict or any(
        {str(item) for item in judgment.get(judgment_field, []) or []}
        != {str(item) for item in basis.get(basis_field, []) or []}
        for judgment_field, basis_field in (
            ("selected_claim_ids", "claim_ids"),
            ("selected_discrepancy_ids", "discrepancy_ids"),
            ("selected_visual_anchor_fact_ids", "visual_anchor_fact_ids"),
            ("selected_finding_ids", "finding_ids"),
            ("selected_evidence_ids", "evidence_ids"),
        )
    ):
        raise ValueError("v4 policy export rejects Judgment/verdict-basis mismatch")
    if verdict in {"fake", "real"}:
        expected_stance = "refute" if verdict == "fake" else "support"
        basis_claim_ids = {
            str(item) for item in basis.get("claim_ids", []) or []
        }
        basis_evidence_ids = {
            str(item) for item in basis.get("evidence_ids", []) or []
        }
        basis_finding_ids = {
            str(item) for item in basis.get("finding_ids", []) or []
        }
        linked_evidence_ids = {
            str(evidence_id)
            for finding_id in basis_finding_ids & set(findings)
            for evidence_id in findings[finding_id].get("evidence_ids", []) or []
        }
        if (
            not basis_claim_ids
            or not basis_evidence_ids
            or not basis_finding_ids
            or not basis_evidence_ids <= set(evidence)
            or not basis_evidence_ids <= linked_evidence_ids
            or any(
                not _v4_claim_has_directional_chain(
                    claim_id=claim_id,
                    stance=expected_stance,
                    selected_evidence_ids=basis_evidence_ids,
                    selected_finding_ids=basis_finding_ids,
                    claims=claims,
                    tasks=tasks,
                    evidence=evidence,
                    findings=findings,
                    successful_call_ids=successful_call_ids,
                )
                for claim_id in basis_claim_ids
            )
        ):
            raise ValueError(
                "v4 policy export rejects incomplete verdict Finding/Evidence chain"
            )
    audits = _rows(investigation.get("discrepancy_coverage_audits"))
    investigation_stop = str(investigation.get("stop_reason", ""))
    terminal = [
        item
        for item in audits
        if (
            investigation_stop
            and str(item.get("stop_reason", "")) == investigation_stop
            or not investigation_stop
            and item.get("complete") is True
            and str(item.get("stop_reason", "")) == "verdict_determined"
        )
        and str(item.get("stop_reason", ""))
        in {"verdict_determined", "meaningful_routes_exhausted", "hard_budget_exhausted"}
    ]
    if not terminal:
        raise ValueError("v4 policy export requires terminal discrepancy Coverage")
    terminal_actions = int(terminal[-1].get("action_count", 0) or 0)
    action_steps = [
        item
        for item in _rows(state.get("all_steps"))
        if str(item.get("stage", ""))
        in {
            "image_only_discrepancy_investigation",
            "image_only_visual_reinspection",
        }
        and str(item.get("action_type", "")) == "tool_call"
    ]
    if len(action_steps) > terminal_actions:
        raise ValueError("v4 policy export rejects post-verdict actions")


def export_policy_examples(
    trace: Mapping[str, Any],
    *,
    tokenizer: TokenizerAdapter | None = None,
    source_metadata: Mapping[str, Any] | None = None,
) -> List[PolicyExample]:
    """Export actual model-visible requests/actions from one canonical trace."""

    state = _mapping(trace.get("state"))
    if str(trace.get("input_mode") or state.get("input_mode") or "") != (
        "image_only"
    ):
        raise ValueError("policy exporter accepts image-only traces only")
    policy_version = str(
        trace.get("decision_policy_version")
        or state.get("decision_policy_version")
        or ""
    )
    if policy_version not in {"reinspect-v2", "discrepancy-first-v4"}:
        raise ValueError("policy exporter received an unsupported decision policy")
    if policy_version == "discrepancy-first-v4":
        _v4_quality_gate(trace, state)

    tokenizer = tokenizer or Utf8ByteTokenizer()
    source_metadata = source_metadata or {}
    episode_id = str(
        trace.get("image_id") or state.get("image_id") or ""
    ).strip()
    if not episode_id:
        raise ValueError("canonical trace requires image_id")

    candidates: List[tuple[int, Mapping[str, Any], str]] = []
    for index, step in enumerate(_rows(state.get("all_steps"))):
        if str(step.get("action_type", "")) in {
            "planning_revision",
            "format_error",
            "output_rejected",
        }:
            continue
        stage = str(step.get("stage", "")).strip()
        example_type = _example_type(stage)
        metadata = _mapping(step.get("metadata"))
        if example_type is None:
            continue
        if not isinstance(metadata.get("policy_input"), Mapping):
            continue
        if not isinstance(metadata.get("policy_action"), Mapping):
            continue
        candidates.append((index, step, example_type))

    trace_failed = str(trace.get("termination", "")).strip() == "error"
    examples: List[PolicyExample] = []
    for position, (index, step, example_type) in enumerate(candidates):
        metadata = _mapping(step.get("metadata"))
        policy_input = dict(_mapping(metadata.get("policy_input")))
        policy_action = dict(_mapping(metadata.get("policy_action")))
        _assert_no_private_data(policy_input)
        _assert_no_private_data(policy_action)
        input_ids = list(tokenizer.encode(canonical_json(policy_input)))
        action_ids = list(tokenizer.encode(canonical_json(policy_action)))
        action_valid = str(step.get("action_type", "")) not in {
            "format_error",
            "output_rejected",
        }
        fatal_boundary = trace_failed and position == len(candidates) - 1
        trainable = action_valid and not fatal_boundary
        terminated = (
            example_type == "judgment"
            or position == len(candidates) - 1
            and str(trace.get("termination", "")) in {"success", "error"}
        )
        interaction_id = str(metadata.get("interaction_id", "")).strip()
        step_id = (
            f"{episode_id}:{example_type}:{interaction_id}"
            if interaction_id
            else f"{episode_id}:{example_type}:{index + 1}"
        )
        examples.append(
            PolicyExample(
                trajectory_version=(
                    "ifv-policy-v2"
                    if policy_version == "discrepancy-first-v4"
                    else "ifv-policy-v1"
                ),
                tokenizer_id=tokenizer.tokenizer_id,
                episode_id=episode_id,
                step_id=step_id,
                source_run_id=str(source_metadata.get("source_run_id", "")),
                runtime_commit=str(source_metadata.get("runtime_commit", "")),
                release_id=str(source_metadata.get("release_id", "")),
                runtime_contract_version=str(
                    source_metadata.get("runtime_contract_version", "")
                ),
                process_reference_protocol_version=str(
                    source_metadata.get(
                        "process_reference_protocol_version",
                        "",
                    )
                ),
                example_type=example_type,
                runtime_observation_refs=_observation_refs(policy_input),
                policy_input=policy_input,
                policy_action=policy_action,
                policy_input_token_ids=input_ids,
                policy_action_token_ids=action_ids,
                policy_action_loss_mask=[
                    1 if trainable else 0 for _ in action_ids
                ],
                action_valid=action_valid,
                terminated=terminated,
                fatal_boundary=fatal_boundary,
            )
        )
    return examples


def examples_as_jsonl(examples: Iterable[PolicyExample]) -> str:
    lines = [
        item.model_dump_json(exclude_none=False)
        for item in examples
    ]
    return "\n".join(lines) + ("\n" if lines else "")
