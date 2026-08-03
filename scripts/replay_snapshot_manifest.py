#!/usr/bin/env python3
"""Run frozen discrepancy replays from a reviewed snapshot manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _summary(result: Mapping[str, Any]) -> dict[str, Any]:
    records = result.get("records")
    records = records if isinstance(records, Mapping) else {}
    visual_records = records.get("visual_reinspections")
    visual_records = visual_records if isinstance(visual_records, list) else []
    evidence_records = records.get("evidence")
    evidence_records = (
        evidence_records if isinstance(evidence_records, list) else []
    )
    assessment_records = records.get("claim_assessments")
    assessment_records = (
        assessment_records if isinstance(assessment_records, list) else []
    )
    discrepancy_records = records.get("material_discrepancies")
    discrepancy_records = (
        discrepancy_records if isinstance(discrepancy_records, list) else []
    )
    decision_records = records.get("discrepancy_decisions")
    decision_records = (
        decision_records if isinstance(decision_records, list) else []
    )
    decision_2 = result.get("decision_2_update")
    decision_2 = decision_2 if isinstance(decision_2, Mapping) else {}
    accepted_assessment_ids = {
        str(item)
        for item in decision_2.get("accepted_assessment_ids", []) or []
    }
    accepted_discrepancy_id = str(
        decision_2.get("accepted_discrepancy_id") or ""
    )
    resolved_visual_evidence_ids = {
        str(evidence_id)
        for record in visual_records
        if isinstance(record, Mapping) and record.get("status") == "resolved"
        for evidence_id in record.get("evidence_ids", []) or []
        if any(
            isinstance(evidence, Mapping)
            and str(evidence.get("evidence_id")) == str(evidence_id)
            and evidence.get("evidence_kind") == "image_region"
            and evidence.get("tool_name") == "focused_visual_inspection"
            for evidence in evidence_records
        )
    }
    consumed_evidence_ids = {
        str(evidence_id)
        for assessment in assessment_records
        if isinstance(assessment, Mapping)
        and str(assessment.get("assessment_id")) in accepted_assessment_ids
        for evidence_id in assessment.get("evidence_ids", []) or []
    }
    consumed_evidence_ids.update(
        str(evidence_id)
        for discrepancy in discrepancy_records
        if isinstance(discrepancy, Mapping)
        and str(discrepancy.get("discrepancy_id"))
        == accepted_discrepancy_id
        for evidence_id in discrepancy.get("evidence_ids", []) or []
    )
    decision_record = next(
        (
            record
            for record in reversed(decision_records)
            if isinstance(record, Mapping)
            and str(record.get("decision_id"))
            == str(decision_2.get("decision_id") or "")
        ),
        {},
    )
    output = (
        decision_record.get("output")
        if isinstance(decision_record, Mapping)
        else {}
    )
    output = output if isinstance(output, Mapping) else {}
    disposition = output.get("visual_evidence_disposition")
    disposition = disposition if isinstance(disposition, Mapping) else {}
    disposition_value = str(disposition.get("disposition") or "")
    consumed_visual_ids = sorted(
        resolved_visual_evidence_ids & consumed_evidence_ids
    )
    if not decision_2:
        consumption_mode = "not_run"
    elif not resolved_visual_evidence_ids:
        consumption_mode = "not_required"
    elif consumed_visual_ids:
        consumption_mode = "consumed"
    elif (
        disposition_value
        == "irrelevant_to_current_claim_or_discrepancy"
    ):
        consumption_mode = "explicitly_irrelevant"
    else:
        consumption_mode = "missing"
    basis = result.get("compiled_basis")
    basis = basis if isinstance(basis, Mapping) else {}
    return {
        "case_id": result.get("case_id"),
        "engineering_error": result.get("engineering_error", ""),
        "engineering_failure_codes": result.get(
            "engineering_failure_codes",
            [],
        ),
        "reinspection_requested": bool(visual_records),
        "visual_inspection_ran": result.get("visual_inspection_update")
        is not None,
        "decision_2_ran": result.get("decision_2_update") is not None,
        "resolved_visual_evidence_ids": sorted(
            resolved_visual_evidence_ids
        ),
        "decision_2_consumed_visual_evidence_ids": consumed_visual_ids,
        "decision_2_visual_evidence_disposition": disposition_value,
        "decision_2_visual_consumption_mode": consumption_mode,
        "decision_2_visual_compliant": consumption_mode
        in {"consumed", "explicitly_irrelevant", "not_required"},
        "decision_2_deterministic_visual_consumption_fallback": bool(
            decision_2.get("deterministic_visual_consumption_fallback")
        ),
        "decision_2_deterministic_exhaustion_fallback": bool(
            decision_2.get("deterministic_decision_exhaustion_fallback")
        ),
        "composite_success": bool(result.get("composite_success")),
        "compiled_verdict": result.get("compiled_verdict", ""),
        "decision_mode": basis.get("decision_mode", ""),
        "basis_evidence_ids": basis.get("evidence_ids", []),
        "coverage_stop_reason": (
            result.get("coverage_audit", {}) or {}
        ).get("stop_reason", ""),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--provider", default="qwen_local")
    parser.add_argument("--model", default="ifv-qwen3.5-9b")
    parser.add_argument("--vlm-provider", default="qwen_local")
    parser.add_argument("--vlm-model", default="ifv-qwen3.5-9b")
    parser.add_argument("--llm-wire-api", default="chat_completions")
    parser.add_argument("--vlm-wire-api", default="chat_completions")
    parser.add_argument("--llm-base-url", default="")
    parser.add_argument("--vlm-base-url", default="")
    parser.add_argument("--sampling-seed", type=int, default=11)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    manifest = _load(args.manifest.expanduser().resolve())
    recommendations = manifest.get("recommendations")
    if not isinstance(recommendations, list):
        raise ValueError("manifest recommendations must be a list")
    if args.limit is not None:
        recommendations = recommendations[: args.limit]
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for index, row in enumerate(recommendations, start=1):
        if not isinstance(row, Mapping):
            continue
        case_id = str(row.get("case_id", "")).strip()
        snapshot = str(row.get("snapshot", "")).strip()
        evidence_id = str(row.get("evidence_id", "")).strip()
        case_dir = output_dir / case_id
        result_path = case_dir / "result.json"
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "replay_snapshot_discrepancy.py"),
            "--snapshot",
            snapshot,
            "--benchmark",
            str(args.benchmark.expanduser().resolve()),
            "--case-id",
            case_id,
            "--source-evidence-id",
            evidence_id,
            "--output",
            str(result_path),
            "--provider",
            args.provider,
            "--model",
            args.model,
            "--vlm-provider",
            args.vlm_provider,
            "--vlm-model",
            args.vlm_model,
            "--llm-wire-api",
            args.llm_wire_api,
            "--vlm-wire-api",
            args.vlm_wire_api,
            "--sampling-seed",
            str(args.sampling_seed),
        ]
        if args.llm_base_url:
            command.extend(["--llm-base-url", args.llm_base_url])
        if args.vlm_base_url:
            command.extend(["--vlm-base-url", args.vlm_base_url])
        print(
            f"[{index}/{len(recommendations)}] replay {case_id} "
            f"Evidence={evidence_id}",
            flush=True,
        )
        completed = subprocess.run(command, cwd=REPO_ROOT)
        if completed.returncode != 0 or not result_path.is_file():
            failures.append(
                {
                    "case_id": case_id,
                    "returncode": completed.returncode,
                    "output": str(result_path),
                }
            )
            continue
        result = _load(result_path)
        summary = _summary(result)
        summary["source_evidence_id"] = evidence_id
        summary["source_visible_property_hint"] = row.get(
            "source_visible_property_hint",
            "",
        )
        summary["output"] = str(result_path)
        summaries.append(summary)

    report = {
        "schema_version": "ifv-reviewed52-replay-batch-v1",
        "manifest": str(args.manifest.expanduser().resolve()),
        "benchmark": str(args.benchmark.expanduser().resolve()),
        "provider": args.provider,
        "model": args.model,
        "vlm_provider": args.vlm_provider,
        "vlm_model": args.vlm_model,
        "sampling_seed": args.sampling_seed,
        "requested_case_count": len(recommendations),
        "completed_case_count": len(summaries),
        "subprocess_failure_count": len(failures),
        "engineering_error_count": sum(
            bool(item["engineering_error"]) for item in summaries
        ),
        "reinspection_requested_count": sum(
            bool(item["reinspection_requested"]) for item in summaries
        ),
        "visual_inspection_ran_count": sum(
            bool(item["visual_inspection_ran"]) for item in summaries
        ),
        "resolved_visual_evidence_case_count": sum(
            bool(item["resolved_visual_evidence_ids"])
            for item in summaries
        ),
        "decision_2_visual_compliant_count": sum(
            bool(item["decision_2_visual_compliant"])
            for item in summaries
        ),
        "decision_2_visual_consumed_count": sum(
            item["decision_2_visual_consumption_mode"] == "consumed"
            for item in summaries
        ),
        "decision_2_visual_explicitly_irrelevant_count": sum(
            item["decision_2_visual_consumption_mode"]
            == "explicitly_irrelevant"
            for item in summaries
        ),
        "decision_2_visual_missing_count": sum(
            item["decision_2_visual_consumption_mode"] == "missing"
            for item in summaries
        ),
        "decision_2_deterministic_fallback_count": sum(
            bool(
                item[
                    "decision_2_deterministic_visual_consumption_fallback"
                ]
                or item[
                    "decision_2_deterministic_exhaustion_fallback"
                ]
            )
            for item in summaries
        ),
        "composite_success_count": sum(
            bool(item["composite_success"]) for item in summaries
        ),
        "terminal_verdict_count": sum(
            bool(item["compiled_verdict"]) for item in summaries
        ),
        "summaries": summaries,
        "failures": failures,
    }
    report_path = output_dir / "batch-summary.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
