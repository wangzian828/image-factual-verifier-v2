"""Pure canonical-trace to policy-example exporter."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Mapping, Protocol, Sequence

from src.trajectory.schema import PolicyExample


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
    if stage == "image_only_investigation":
        return "react"
    if stage == "image_only_reflection":
        return "reflection"
    if stage == "image_only_judgment":
        return "judgment"
    if stage == "image_only_planning":
        return "planning"
    return None


def export_policy_examples(
    trace: Mapping[str, Any],
    *,
    tokenizer: TokenizerAdapter | None = None,
) -> List[PolicyExample]:
    """Export actual model-visible requests/actions from one canonical trace."""

    state = _mapping(trace.get("state"))
    if str(trace.get("input_mode") or state.get("input_mode") or "") != (
        "image_only"
    ):
        raise ValueError("policy exporter accepts image-only traces only")
    if str(
        trace.get("decision_policy_version")
        or state.get("decision_policy_version")
        or ""
    ) != "reinspect-v2":
        raise ValueError("policy exporter accepts reinspect-v2 traces only")

    tokenizer = tokenizer or Utf8ByteTokenizer()
    episode_id = str(
        trace.get("image_id") or state.get("image_id") or ""
    ).strip()
    if not episode_id:
        raise ValueError("canonical trace requires image_id")

    candidates: List[tuple[int, Mapping[str, Any], str]] = []
    for index, step in enumerate(_rows(state.get("all_steps"))):
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
                tokenizer_id=tokenizer.tokenizer_id,
                episode_id=episode_id,
                step_id=step_id,
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
