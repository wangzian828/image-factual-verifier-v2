"""Provider-neutral focused visual teacher export from canonical traces."""

from __future__ import annotations

from typing import Any, List, Mapping

from src.orchestrator.tool_result import parse_tool_result
from src.trajectory.exporter import _assert_no_private_data
from src.trajectory.schema import VisualReinspectionExample


VISUAL_REINSPECTION_INSTRUCTION = (
    "Inspect the complete original image and deterministic anchor-derived views "
    "to answer one focused visual question. Treat searched names and expected "
    "properties as hypotheses, report only visible pixels, preserve ambiguity, "
    "and do not decide the benchmark verdict."
)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _rows(value: Any) -> List[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def export_visual_reinspection_examples(
    trace: Mapping[str, Any],
    *,
    source_metadata: Mapping[str, Any] | None = None,
) -> List[VisualReinspectionExample]:
    """Export successful visual questions without mixing them into policy SFT."""

    state = _mapping(trace.get("state"))
    if str(trace.get("input_mode") or state.get("input_mode") or "") != "image_only":
        raise ValueError("visual reinspection exporter accepts image-only traces only")
    runtime_case = _mapping(state.get("runtime_case"))
    episode_id = str(
        trace.get("image_id")
        or state.get("image_id")
        or runtime_case.get("case_id")
        or ""
    ).strip()
    image_path = str(
        runtime_case.get("image_path")
        or trace.get("image_path")
        or state.get("image_path")
        or ""
    ).strip()
    image_sha256 = str(runtime_case.get("image_sha256") or "").strip()
    if not episode_id or not image_path or not image_sha256:
        raise ValueError(
            "canonical trace requires image identity for visual reinspection export"
        )

    investigation = _mapping(state.get("investigation_state"))
    records = {
        str(item.get("visual_question_id", "")).strip(): item
        for item in _rows(investigation.get("visual_reinspections"))
        if str(item.get("visual_question_id", "")).strip()
    }
    steps = {
        str(_mapping(item.get("metadata")).get("visual_question_id", "")).strip(): item
        for item in _rows(state.get("all_steps"))
        if str(item.get("stage", "")) == "image_only_visual_reinspection"
        and str(item.get("action_type", "")) == "tool_call"
        and str(_mapping(item.get("metadata")).get("visual_question_id", "")).strip()
    }
    source_metadata = source_metadata or {}
    examples: List[VisualReinspectionExample] = []
    for visual_question_id, record in records.items():
        if str(record.get("status", "")).strip() != "resolved":
            continue
        step = steps.get(visual_question_id)
        if step is None:
            raise ValueError(
                f"resolved visual question {visual_question_id!r} lacks its tool step"
            )
        payload, succeeded = parse_tool_result(str(step.get("tool_result", "")))
        if not succeeded:
            raise ValueError(
                f"resolved visual question {visual_question_id!r} has a failed tool result"
            )
        request = _mapping(record.get("request"))
        tool_args = _mapping(step.get("tool_args"))
        target = {
            "answer_status": payload.get("answer_status"),
            "summary": payload.get("summary"),
            "observations": payload.get("observations", []),
            "limitations": payload.get("limitations", []),
        }
        export_payload = {
            "active_fact": tool_args.get("active_fact"),
            "question": request.get("question"),
            "expected_property": request.get("expected_property"),
            "scope": request.get("scope"),
            "anchor_regions": record.get("anchor_regions", []),
            "grounding_evidence_ids": request.get(
                "grounding_evidence_ids",
                [],
            ),
            "evidence_context": tool_args.get("evidence_context", ""),
            "view_plan": payload.get("views", []),
            "target": target,
        }
        _assert_no_private_data(export_payload)
        examples.append(
            VisualReinspectionExample(
                episode_id=episode_id,
                visual_question_id=visual_question_id,
                source_run_id=str(source_metadata.get("source_run_id", "")),
                runtime_commit=str(source_metadata.get("runtime_commit", "")),
                release_id=str(source_metadata.get("release_id", "")),
                runtime_contract_version=str(
                    source_metadata.get("runtime_contract_version", "")
                ),
                image_path=image_path,
                image_sha256=image_sha256,
                instruction=VISUAL_REINSPECTION_INSTRUCTION,
                **export_payload,
            )
        )
    return examples
