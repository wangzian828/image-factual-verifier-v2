#!/usr/bin/env python3
"""Build a mechanism-safe replay candidate manifest from reviewed snapshots.

This scanner never performs retrieval or model rollout.  It only inspects
historical investigation snapshots and reports which source Evidence records
can actually ground a focused pixel question under the current v4 binding
rules.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.orchestrator.investigation_models import ImageOnlyInvestigationState
from src.orchestrator.task_store import extract_source_visible_property


def _one_line(value: Any) -> str:
    return " ".join(str(value or "").split())


def _load_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _snapshot_number(path: Path) -> int:
    stem = path.stem
    try:
        return int(stem.rsplit("-", 1)[-1])
    except ValueError:
        return -1


def _reviewed_evidence_ids(state: ImageOnlyInvestigationState) -> set[str]:
    reviewed: set[str] = set()
    for record in state.discrepancy_decisions:
        reviewed.update(record.reviewed_evidence_ids)
    return reviewed


def _candidate_records(
    snapshot_path: Path,
    *,
    case_id: str,
) -> list[dict[str, Any]]:
    snapshot = _load_json(snapshot_path)
    payload = snapshot.get("payload")
    if not isinstance(payload, Mapping):
        return []
    state_payload = payload.get("investigation_state")
    if not isinstance(state_payload, Mapping):
        return []
    state = ImageOnlyInvestigationState.model_validate(state_payload)
    evidence_by_id = {item.evidence_id: item for item in state.evidence}
    task_by_id = {item.task_id: item for item in state.tasks}
    claim_by_id = {item.claim_id: item for item in state.image_claims}
    reviewed_ids = _reviewed_evidence_ids(state)
    rows: list[dict[str, Any]] = []

    for evidence in state.evidence:
        if evidence.evidence_kind == "image_region":
            continue
        task = task_by_id.get(evidence.task_id)
        if task is None or evidence.claim_binding != "source_assertion":
            continue
        claims = [
            claim_by_id[claim_id]
            for claim_id in task.claim_ids
            if claim_id in claim_by_id
            and claim_by_id[claim_id].fact_id in evidence.fact_ids
            and claim_by_id[claim_id].status
            in {"open", "unresolved", "conflicted"}
        ]
        if not claims:
            continue
        hint = extract_source_visible_property(evidence.exact_text)
        directional = (
            evidence.relation_scope == "same_relation"
            and evidence.relation_stance in {"supports", "contradicts"}
            and evidence.directness == "direct"
            and evidence.quality in {"strong", "moderate"}
        )
        rows.append(
            {
                "case_id": case_id,
                "snapshot": str(snapshot_path),
                "snapshot_number": _snapshot_number(snapshot_path),
                "evidence_id": evidence.evidence_id,
                "task_id": evidence.task_id,
                "claim_ids": [claim.claim_id for claim in claims],
                "claim_statements": [claim.statement for claim in claims],
                "relation_scope": evidence.relation_scope,
                "relation_stance": evidence.relation_stance,
                "directness": evidence.directness,
                "quality": evidence.quality,
                "reviewed_by_prior_decision": evidence.evidence_id in reviewed_ids,
                "source_text": _one_line(evidence.exact_text),
                "source_visible_property_hint": hint,
                "binding_status": (
                    "qualified"
                    if hint and directional and evidence.evidence_id not in reviewed_ids
                    else "hint_only"
                    if hint
                    else "no_concrete_visible_property"
                ),
                "visual_reinspection_count": len(state.visual_reinspections),
                "discrepancy_decision_count": len(state.discrepancy_decisions),
            }
        )
    return rows


def scan_reviewed_snapshots(
    run_root: Path,
    *,
    case_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    runtime_root = run_root / "traces" / "runtime"
    requested = {str(item).strip() for item in case_ids or [] if str(item).strip()}
    rows: list[dict[str, Any]] = []
    snapshot_count = 0
    case_count = 0
    for case_dir in sorted(runtime_root.iterdir() if runtime_root.exists() else []):
        if not case_dir.is_dir() or (requested and case_dir.name not in requested):
            continue
        case_count += 1
        for snapshot_path in sorted(
            case_dir.glob("*/snapshots/snapshot-*.json"),
            key=lambda path: (_snapshot_number(path), str(path)),
        ):
            snapshot_count += 1
            rows.extend(
                _candidate_records(snapshot_path, case_id=case_dir.name)
            )

    qualified = [
        row for row in rows if row["binding_status"] == "qualified"
    ]
    by_case: dict[str, list[dict[str, Any]]] = {}
    for row in qualified:
        by_case.setdefault(row["case_id"], []).append(row)
    recommendations: list[dict[str, Any]] = []
    for case_id, case_rows in sorted(by_case.items()):
        chosen = sorted(
            case_rows,
            key=lambda row: (
                row["visual_reinspection_count"] != 0,
                row["discrepancy_decision_count"] != 0,
                row["snapshot_number"],
                row["evidence_id"],
            ),
        )[0]
        recommendations.append(chosen)

    return {
        "schema_version": "ifv-reviewed52-replay-manifest-v1",
        "run_root": str(run_root),
        "case_count_scanned": case_count,
        "snapshot_count_scanned": snapshot_count,
        "candidate_evidence_count": len(rows),
        "qualified_evidence_count": len(qualified),
        "qualified_case_count": len(by_case),
        "recommended_case_count": len(recommendations),
        "recommendations": recommendations,
        "candidates": rows,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-id", action="append", default=[])
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    result = scan_reviewed_snapshots(
        args.run_root.expanduser().resolve(),
        case_ids=args.case_id,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "case_count_scanned",
                    "snapshot_count_scanned",
                    "candidate_evidence_count",
                    "qualified_evidence_count",
                    "qualified_case_count",
                    "recommended_case_count",
                    "output",
                )
                if key != "output"
            }
            | {"output": str(output)},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
