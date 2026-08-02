#!/usr/bin/env python3
"""Replay one frozen v4 discrepancy-decision checkpoint from a saved snapshot.

This is a targeted mechanism validator, not a free-running evaluation entrypoint.
It restores a historical ImageOnlyInvestigationState that already contains source
Evidence, runs one Discrepancy Decision checkpoint, executes any accepted focused
visual reinspection, runs a second Decision checkpoint with the newly created
visual Evidence, then compiles deterministic coverage and verdict basis.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.release_adapter import (
    image_only_case_from_runtime_row,
    resolve_runtime_image_path,
)
from src.eval.scoring_release_adapter import (
    SCORING_PACKAGE_SCHEMA_VERSION,
    load_scoring_release,
    resolve_scoring_image_path,
)
from src.orchestrator.discrepancy_coverage import (
    audit_discrepancy_coverage,
    compile_discrepancy_verdict_basis,
)
from src.orchestrator.investigation_models import ImageOnlyInvestigationState
from src.orchestrator.pipeline import Orchestrator
from src.orchestrator.runtime_case import verify_case_image
from src.orchestrator.runtime_events import (
    CaseRuntimeStore,
    bind_case_runtime_store,
    reset_case_runtime_store,
)
from src.orchestrator.stage_runner import InteractionSession
from src.orchestrator.state import VerificationState
from src.orchestrator.task_store import (
    COMPOSITE_SOURCE_VISUAL_DISCREPANCY_FAMILY,
    pending_visual_reinspection,
)


def _ensure_loopback_no_proxy() -> None:
    required = ("127.0.0.1", "localhost", "::1")
    for name in ("NO_PROXY", "no_proxy"):
        configured = [
            item.strip()
            for item in os.environ.get(name, "").split(",")
            if item.strip()
        ]
        os.environ[name] = ",".join(dict.fromkeys([*configured, *required]))


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _load_runtime_row(benchmark: Path, case_id: str) -> dict[str, Any]:
    with benchmark.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, Mapping) and str(row.get("case_id")) == case_id:
                return dict(row)
    raise KeyError(f"case_id not found in benchmark: {case_id}")


def _runtime_case_from_benchmark(
    benchmark: Path,
    case_id: str,
):
    row = _load_runtime_row(benchmark, case_id)
    manifest_path = benchmark.parent.parent / "manifest.json"
    manifest = _load_json_object(manifest_path)
    if str(manifest.get("schema_version") or "") == SCORING_PACKAGE_SCHEMA_VERSION:
        release = load_scoring_release(benchmark)
        row = resolve_scoring_image_path(row, release)
    else:
        row = resolve_runtime_image_path(row, benchmark)
    return image_only_case_from_runtime_row(row)


def _snapshot_investigation(snapshot: Mapping[str, Any]) -> ImageOnlyInvestigationState:
    payload = snapshot.get("payload")
    if not isinstance(payload, Mapping):
        raise ValueError("snapshot payload must be an object")
    state = payload.get("investigation_state")
    if not isinstance(state, Mapping):
        raise ValueError("snapshot payload lacks investigation_state object")
    return ImageOnlyInvestigationState.model_validate(state)


def _existing_ids(rows: Iterable[Any], attr: str) -> set[str]:
    return {str(getattr(item, attr)) for item in rows}


def _validate_source_evidence_ids(
    investigation: ImageOnlyInvestigationState,
    evidence_ids: Sequence[str],
) -> list[str]:
    requested = [str(item).strip() for item in evidence_ids if str(item).strip()]
    if not requested:
        raise ValueError("at least one --source-evidence-id is required")
    existing = _existing_ids(investigation.evidence, "evidence_id")
    missing = [item for item in requested if item not in existing]
    if missing:
        raise KeyError("source Evidence ID(s) absent from snapshot: " + ", ".join(missing))
    return requested


def _collect_records(investigation: ImageOnlyInvestigationState) -> dict[str, Any]:
    return {
        "visual_reinspections": [
            item.model_dump(mode="json") for item in investigation.visual_reinspections
        ],
        "evidence": [item.model_dump(mode="json") for item in investigation.evidence],
        "findings": [item.model_dump(mode="json") for item in investigation.findings],
        "claim_assessments": [
            item.model_dump(mode="json") for item in investigation.claim_assessments
        ],
        "material_discrepancies": [
            item.model_dump(mode="json") for item in investigation.material_discrepancies
        ],
        "discrepancy_coverage_audits": [
            item.model_dump(mode="json")
            for item in investigation.discrepancy_coverage_audits
        ],
        "discrepancy_verdict_basis": (
            investigation.discrepancy_verdict_basis.model_dump(mode="json")
            if investigation.discrepancy_verdict_basis is not None
            else None
        ),
        "proposed_verdict": investigation.proposed_verdict,
        "stop_reason": investigation.stop_reason,
    }


def _composite_finding_ids(
    investigation: ImageOnlyInvestigationState,
) -> list[str]:
    return [
        item.finding_id
        for item in investigation.findings
        if COMPOSITE_SOURCE_VISUAL_DISCREPANCY_FAMILY in item.source_family_ids
    ]


async def replay_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    snapshot_path = args.snapshot.expanduser().resolve()
    benchmark_path = args.benchmark.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    snapshot = _load_json_object(snapshot_path)
    investigation = _snapshot_investigation(snapshot)

    case_id = args.case_id or investigation.brief.case_id
    if not case_id:
        raise ValueError("case id is required when snapshot brief lacks case_id")
    source_evidence_ids = _validate_source_evidence_ids(
        investigation,
        args.source_evidence_id,
    )

    runtime_case = _runtime_case_from_benchmark(benchmark_path, case_id)
    verify_case_image(runtime_case, runtime_case.image_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_store = CaseRuntimeStore(
        output_path.parent,
        case_id=f"{case_id}-snapshot-replay",
    )
    token = bind_case_runtime_store(runtime_store)
    state = VerificationState(
        image_path=runtime_case.image_path,
        image_id=case_id,
        runtime_case=runtime_case,
        input_mode="image_only",
        decision_policy_version="discrepancy-first-v4",
        investigation_state=investigation,
        investigation_brief=investigation.brief,
        visual_entities=list(investigation.entities),
        visual_facts=list(investigation.facts),
        research_tasks=list(investigation.tasks),
        findings=list(investigation.findings),
        retrieval_anchors=list(investigation.retrieval_anchors),
        runtime_store=runtime_store,
    )

    _ensure_loopback_no_proxy()
    orchestrator = Orchestrator(
        provider=args.provider,
        model_name=args.model,
        vlm_provider=args.vlm_provider,
        vlm_model=args.vlm_model,
        llm_wire_api=args.llm_wire_api,
        vlm_wire_api=args.vlm_wire_api,
        llm_base_url=args.llm_base_url,
        vlm_base_url=args.vlm_base_url,
        validate_startup=False,
        sampling_seed=args.sampling_seed,
    )

    try:
        decision_1 = await orchestrator._run_discrepancy_decision(
            state,
            investigation,
            reviewed_evidence_ids=source_evidence_ids,
            trigger="qualified_evidence",
            interaction_session=InteractionSession(),
        )
        visual_update: dict[str, Any] | None = None
        pending_record = pending_visual_reinspection(investigation)
        decision_2: dict[str, Any] | None = None
        reviewed_after_visual = list(source_evidence_ids)
        engineering_error = ""
        engineering_error_stage = ""
        if pending_record is not None:
            try:
                visual_update = await orchestrator._run_image_only_visual_reinspection(
                    state,
                    investigation,
                    image_path=runtime_case.image_path,
                    runtime_case=runtime_case,
                    visual_question_id=pending_record.visual_question_id,
                )
            except Exception as exc:
                engineering_error = f"{type(exc).__name__}: {exc}"
                engineering_error_stage = "image_only_visual_reinspection"
            if not engineering_error:
                reviewed_after_visual.extend(
                    evidence_id
                    for evidence_id in pending_record.evidence_ids
                    if evidence_id not in reviewed_after_visual
                )
                try:
                    decision_2 = await orchestrator._run_discrepancy_decision(
                        state,
                        investigation,
                        reviewed_evidence_ids=reviewed_after_visual,
                        trigger="qualified_evidence",
                        interaction_session=InteractionSession(),
                    )
                except Exception as exc:
                    engineering_error = f"{type(exc).__name__}: {exc}"
                    engineering_error_stage = "image_only_discrepancy_decision"
        visual_evidence_ids = (
            list(pending_record.evidence_ids)
            if pending_record is not None
            else []
        )
        audit = audit_discrepancy_coverage(investigation, decision_checkpoint=True)
        compile_error = ""
        compiled_verdict = ""
        basis_payload: dict[str, Any] | None = None
        try:
            compiled_verdict, basis = compile_discrepancy_verdict_basis(investigation)
            basis_payload = basis.model_dump(mode="json")
        except Exception as exc:
            compile_error = f"{type(exc).__name__}: {exc}"
        composite_finding_ids = _composite_finding_ids(investigation)
        composite_created_ids = (
            list(decision_2.get("created_composite_finding_ids", []))
            if decision_2 is not None
            else []
        )
        basis_evidence_ids = (
            list(basis_payload.get("evidence_ids", []))
            if basis_payload is not None
            else []
        )
        basis_finding_ids = (
            list(basis_payload.get("finding_ids", []))
            if basis_payload is not None
            else []
        )
        composite_success = bool(
            composite_created_ids
            and set(composite_created_ids) <= set(composite_finding_ids)
            and set(composite_created_ids) <= set(basis_finding_ids)
            and set(source_evidence_ids) <= set(basis_evidence_ids)
            and bool(set(visual_evidence_ids) & set(basis_evidence_ids))
        )
        result = {
            "snapshot_path": str(snapshot_path),
            "benchmark_path": str(benchmark_path),
            "case_id": case_id,
            "runtime_case": runtime_case.model_dump(mode="json"),
            "source_evidence_ids": source_evidence_ids,
            "decision_1_update": decision_1,
            "visual_inspection_update": visual_update,
            "decision_2_update": decision_2,
            "engineering_error": engineering_error,
            "engineering_error_stage": engineering_error_stage,
            "coverage_audit": audit.model_dump(mode="json"),
            "compiled_verdict": compiled_verdict,
            "compiled_basis": basis_payload,
            "compile_error": compile_error,
            "composite_finding_ids": composite_finding_ids,
            "composite_success": composite_success,
            "runtime_store": runtime_store.descriptor,
            "records": _collect_records(investigation),
            "stage_steps": state.to_dict().get("all_steps", []),
        }
    finally:
        reset_case_runtime_store(token)

    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--case-id", default="")
    parser.add_argument("--source-evidence-id", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provider", default="qwen_local")
    parser.add_argument("--model", default="ifv-qwen3.5-9b")
    parser.add_argument("--vlm-provider", default="qwen_local")
    parser.add_argument("--vlm-model", default="ifv-qwen3.5-9b")
    parser.add_argument("--llm-wire-api", default="chat_completions")
    parser.add_argument("--vlm-wire-api", default="chat_completions")
    parser.add_argument("--llm-base-url", default="http://127.0.0.1:8901/v1")
    parser.add_argument("--vlm-base-url", default="http://127.0.0.1:8901/v1")
    parser.add_argument("--sampling-seed", type=int)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    result = asyncio.run(replay_snapshot(args))
    summary = {
        "case_id": result["case_id"],
        "reinspection_requested": bool(
            result["records"]["visual_reinspections"]
        ),
        "visual_inspection_ran": result["visual_inspection_update"] is not None,
        "compiled_verdict": result["compiled_verdict"],
        "engineering_error": result["engineering_error"],
        "engineering_error_stage": result["engineering_error_stage"],
        "composite_finding_ids": result["composite_finding_ids"],
        "composite_success": result["composite_success"],
        "decision_mode": (
            result["compiled_basis"]["decision_mode"]
            if result["compiled_basis"] is not None
            else ""
        ),
        "evidence_ids": (
            result["compiled_basis"]["evidence_ids"]
            if result["compiled_basis"] is not None
            else []
        ),
        "finding_ids": (
            result["compiled_basis"]["finding_ids"]
            if result["compiled_basis"] is not None
            else []
        ),
        "compile_error": result["compile_error"],
        "output": str(args.output.expanduser().resolve()),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
