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
