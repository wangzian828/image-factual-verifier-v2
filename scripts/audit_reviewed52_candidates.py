#!/usr/bin/env python3
"""Audit every frozen reviewed-52 source-Evidence candidate offline.

The input is the JSON produced by ``build_reviewed52_replay_manifest.py``.
No retrieval or model call is performed. Optional replay linkage consumes the
batch summary produced by ``replay_snapshot_manifest.py``.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "he",
    "her",
    "his",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "she",
    "that",
    "the",
    "their",
    "there",
    "they",
    "this",
    "to",
    "was",
    "were",
    "with",
}
PRIMARY_STATUS_ORDER = (
    "no_concrete_visible_property",
    "prior_decision_reviewed",
    "relation_unqualified",
    "strictly_qualified",
    "hint_only_unqualified",
)


def _load(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _one_line(value: Any) -> str:
    return " ".join(str(value or "").split())


def _tokens(value: Any) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", _one_line(value).casefold())
        if len(token) > 1 and token not in STOPWORDS
    }


def _claims(row: Mapping[str, Any]) -> list[str]:
    value = row.get("claim_statements")
    if not isinstance(value, list):
        return []
    return [_one_line(item) for item in value if _one_line(item)]


def _multi_property_hint(hint: str) -> bool:
    lowered = hint.casefold()
    separators = lowered.count(";") + lowered.count(" / ")
    conjunctions = len(re.findall(r"\b(?:and|while|but|whereas)\b", lowered))
    clauses = len(
        [
            item
            for item in re.split(r"[.;]", hint)
            if _one_line(item)
        ]
    )
    return separators > 0 or conjunctions >= 2 or clauses >= 3


def _relation_reasons(row: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if str(row.get("relation_scope") or "") != "same_relation":
        reasons.append("relation_scope_not_same_relation")
    if str(row.get("relation_stance") or "") not in {
        "supports",
        "contradicts",
    }:
        reasons.append("relation_stance_not_directional")
    if str(row.get("directness") or "") != "direct":
        reasons.append("directness_not_direct")
    if str(row.get("quality") or "") not in {"strong", "moderate"}:
        reasons.append("quality_not_strong_or_moderate")
    return reasons


def _replay_index(
    replay_batch: Mapping[str, Any] | None,
) -> dict[tuple[str, str], Mapping[str, Any]]:
    if replay_batch is None:
        return {}
    summaries = replay_batch.get("summaries")
    if not isinstance(summaries, list):
        raise ValueError("replay batch summaries must be a list")
    indexed: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in summaries:
        if not isinstance(row, Mapping):
            continue
        key = (
            str(row.get("case_id") or ""),
            str(row.get("source_evidence_id") or ""),
        )
        if not all(key):
            continue
        if key in indexed:
            raise ValueError(
                "duplicate replay summary for "
                f"case_id={key[0]} source_evidence_id={key[1]}"
            )
        indexed[key] = row
    return indexed


def _primary_status(
    *,
    hint: str,
    reviewed: bool,
    relation_reasons: list[str],
    binding_status: str,
) -> str:
    if not hint:
        return "no_concrete_visible_property"
    if reviewed:
        return "prior_decision_reviewed"
    if relation_reasons:
        return "relation_unqualified"
    if binding_status == "qualified":
        return "strictly_qualified"
    return "hint_only_unqualified"


def _audit_candidate(
    row: Mapping[str, Any],
    *,
    replay_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    case_id = str(row.get("case_id") or "")
    evidence_id = str(row.get("evidence_id") or "")
    snapshot = str(row.get("snapshot") or "")
    if not case_id or not evidence_id or not snapshot:
        raise ValueError(
            "candidate row requires case_id, evidence_id, and snapshot"
        )
    hint = _one_line(row.get("source_visible_property_hint"))
    source_text = _one_line(row.get("source_text"))
    claims = _claims(row)
    claim_text = " ".join(claims)
    hint_tokens = _tokens(hint)
    claim_tokens = _tokens(claim_text)
    source_tokens = _tokens(source_text)
    relation_reasons = _relation_reasons(row)
    reviewed = bool(row.get("reviewed_by_prior_decision"))
    binding_status = str(row.get("binding_status") or "")
    primary_status = _primary_status(
        hint=hint,
        reviewed=reviewed,
        relation_reasons=relation_reasons,
        binding_status=binding_status,
    )

    flags: list[str] = []
    if not hint:
        flags.append("no_concrete_visible_property")
    if len(hint) > 180:
        flags.append("hint_overlong")
    if hint and _multi_property_hint(hint):
        flags.append("hint_multi_property")
    if hint and claim_tokens and not (hint_tokens & claim_tokens):
        flags.append("weak_hint_claim_binding")
    if (
        hint
        and claim_tokens
        and not (source_tokens & claim_tokens)
        and str(row.get("relation_scope") or "") == "same_relation"
    ):
        flags.append("potential_claim_irrelevance")
    if reviewed:
        flags.append("reviewed_by_prior_decision")
    flags.extend(relation_reasons)
    if primary_status == "strictly_qualified":
        flags.append("strictly_qualified")

    replay = replay_by_key.get((case_id, evidence_id))
    replay_link: dict[str, Any] = {
        "available": replay is not None,
        "engineering_error": "",
        "engineering_failure_codes": [],
        "source_only_follow_up_blocked": False,
        "visual_consumption_mode": "",
        "deterministic_fallback": False,
    }
    if replay is None:
        flags.append("replay_unavailable")
    else:
        engineering_error = _one_line(replay.get("engineering_error"))
        failure_codes = [
            str(item)
            for item in replay.get("engineering_failure_codes", []) or []
        ]
        consumption_mode = str(
            replay.get("decision_2_visual_consumption_mode") or ""
        )
        deterministic_fallback = bool(
            replay.get(
                "decision_2_deterministic_visual_consumption_fallback"
            )
            or replay.get("decision_2_deterministic_exhaustion_fallback")
        )
        source_only_blocked = bool(
            replay.get("source_only_follow_up_blocked")
        )
        replay_link = {
            "available": True,
            "engineering_error": engineering_error,
            "engineering_failure_codes": failure_codes,
            "source_only_follow_up_blocked": source_only_blocked,
            "visual_consumption_mode": consumption_mode,
            "deterministic_fallback": deterministic_fallback,
        }
        if engineering_error:
            flags.append("replay_engineering_failure")
        if source_only_blocked:
            flags.append("source_only_follow_up_blocked")
        if consumption_mode == "missing":
            flags.append("resolved_pixel_not_consumed")
        if deterministic_fallback:
            flags.append("deterministic_fallback")

    risk_flags = {
        "hint_overlong",
        "hint_multi_property",
        "weak_hint_claim_binding",
        "potential_claim_irrelevance",
        "resolved_pixel_not_consumed",
        "replay_engineering_failure",
        "deterministic_fallback",
    }
    return {
        "case_id": case_id,
        "snapshot": snapshot,
        "snapshot_number": row.get("snapshot_number"),
        "evidence_id": evidence_id,
        "task_id": str(row.get("task_id") or ""),
        "claim_ids": list(row.get("claim_ids") or []),
        "claim_statements": claims,
        "source_text": source_text,
        "source_visible_property_hint": hint,
        "binding_status": binding_status,
        "primary_status": primary_status,
        "flags": list(dict.fromkeys(flags)),
        "risk_score": sum(flag in risk_flags for flag in flags),
        "relation": {
            "scope": str(row.get("relation_scope") or ""),
            "stance": str(row.get("relation_stance") or ""),
            "directness": str(row.get("directness") or ""),
            "quality": str(row.get("quality") or ""),
        },
        "lexical_audit": {
            "hint_claim_overlap": sorted(hint_tokens & claim_tokens),
            "source_claim_overlap": sorted(source_tokens & claim_tokens),
            "heuristic_only": True,
        },
        "reviewed_by_prior_decision": reviewed,
        "replay": replay_link,
    }


def audit_manifest(
    manifest: Mapping[str, Any],
    *,
    replay_batch: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    candidates = manifest.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("manifest candidates must be a list")
    declared_count = manifest.get("candidate_evidence_count")
    if declared_count is not None and int(declared_count) != len(candidates):
        raise ValueError(
            "manifest candidate_evidence_count does not match candidates"
        )
    replay_by_key = _replay_index(replay_batch)
    audited = [
        _audit_candidate(row, replay_by_key=replay_by_key)
        for row in candidates
        if isinstance(row, Mapping)
    ]
    if len(audited) != len(candidates):
        raise ValueError("every candidate must be a JSON object")
    primary_counts = Counter(row["primary_status"] for row in audited)
    unknown_primary = set(primary_counts) - set(PRIMARY_STATUS_ORDER)
    if unknown_primary:
        raise AssertionError(
            "unexpected primary statuses: " + ", ".join(sorted(unknown_primary))
        )
    flag_counts = Counter(
        flag for row in audited for flag in row["flags"]
    )
    case_counts = Counter(row["case_id"] for row in audited)
    replay_linked = sum(row["replay"]["available"] for row in audited)
    unique_evidence_keys = {
        (row["case_id"], row["evidence_id"]) for row in audited
    }
    replay_matched_keys = {
        (row["case_id"], row["evidence_id"])
        for row in audited
        if row["replay"]["available"]
    }
    qualified_keys = {
        (row["case_id"], row["evidence_id"])
        for row in audited
        if row["primary_status"] == "strictly_qualified"
    }
    qualified_replay_matched_keys = qualified_keys & replay_matched_keys
    primary_case_counts = {
        key: len(
            {
                row["case_id"]
                for row in audited
                if row["primary_status"] == key
            }
        )
        for key in PRIMARY_STATUS_ORDER
    }
    flag_unique_evidence_counts = {
        flag: len(
            {
                (row["case_id"], row["evidence_id"])
                for row in audited
                if flag in row["flags"]
            }
        )
        for flag in flag_counts
    }
    return {
        "schema_version": "ifv-reviewed52-offline-audit-v1",
        "source_manifest_schema_version": manifest.get("schema_version", ""),
        "run_root": manifest.get("run_root", ""),
        "case_count_scanned": manifest.get("case_count_scanned"),
        "snapshot_count_scanned": manifest.get("snapshot_count_scanned"),
        "candidate_evidence_count": len(audited),
        "unique_case_evidence_count": len(unique_evidence_keys),
        "classified_candidate_count": len(audited),
        "classification_complete": len(audited) == len(candidates),
        "primary_status_counts": {
            key: primary_counts.get(key, 0)
            for key in PRIMARY_STATUS_ORDER
        },
        "primary_status_case_counts": primary_case_counts,
        "flag_counts": dict(sorted(flag_counts.items())),
        "flag_unique_case_evidence_counts": dict(
            sorted(flag_unique_evidence_counts.items())
        ),
        "case_candidate_counts": dict(sorted(case_counts.items())),
        "replay_summary_count": len(replay_by_key),
        "replay_linked_candidate_count": replay_linked,
        "replay_matched_summary_count": len(replay_matched_keys),
        "replay_unmatched_summary_count": (
            len(replay_by_key) - len(replay_matched_keys)
        ),
        "qualified_unique_case_evidence_count": len(qualified_keys),
        "qualified_replay_matched_count": len(
            qualified_replay_matched_keys
        ),
        "qualified_replay_unavailable_count": (
            len(qualified_keys) - len(qualified_replay_matched_keys)
        ),
        "candidates": audited,
    }


def _markdown(report: Mapping[str, Any]) -> str:
    primary = report.get("primary_status_counts") or {}
    flags = report.get("flag_counts") or {}
    candidates = report.get("candidates") or []
    risky_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in candidates:
        if not isinstance(row, Mapping) or int(row.get("risk_score") or 0) <= 0:
            continue
        key = (
            str(row.get("case_id") or ""),
            str(row.get("evidence_id") or ""),
        )
        incumbent = risky_by_key.get(key)
        if incumbent is None or int(row.get("risk_score") or 0) > int(
            incumbent.get("risk_score") or 0
        ):
            risky_by_key[key] = row
    risky = sorted(
        risky_by_key.values(),
        key=lambda row: (
            -int(row.get("risk_score") or 0),
            str(row.get("case_id") or ""),
            int(row.get("snapshot_number") or -1),
        ),
    )
    lines = [
        "# Reviewed-52 offline hint and Decision audit",
        "",
        "This report is deterministic and offline. Lexical binding/relevance "
        "flags are triage signals, not semantic verdicts.",
        "",
        "## Coverage",
        "",
        f"- cases: {report.get('case_count_scanned')}",
        f"- snapshots: {report.get('snapshot_count_scanned')}",
        f"- candidate Evidence rows: {report.get('candidate_evidence_count')}",
        f"- unique case/Evidence pairs: {report.get('unique_case_evidence_count')}",
        f"- classified exactly once: {report.get('classification_complete')}",
        f"- replay-linked rows: {report.get('replay_linked_candidate_count')}",
        "- strictly qualified replay coverage: "
        f"{report.get('qualified_replay_matched_count')}/"
        f"{report.get('qualified_unique_case_evidence_count')}",
        "",
        "## Primary status",
        "",
        "| Status | Count |",
        "|---|---:|",
    ]
    for key in PRIMARY_STATUS_ORDER:
        lines.append(f"| `{key}` | {int(primary.get(key, 0) or 0)} |")
    lines.extend(
        [
            "",
            "## Flags",
            "",
            "| Flag | Count |",
            "|---|---:|",
        ]
    )
    for key, value in sorted(flags.items()):
        lines.append(f"| `{key}` | {int(value)} |")
    lines.extend(
        [
            "",
            "## Highest-risk examples",
            "",
            "| Case | Snapshot | Evidence | Risk | Flags | Hint |",
            "|---|---:|---|---:|---|---|",
        ]
    )
    for row in risky[:30]:
        hint = _one_line(row.get("source_visible_property_hint")).replace(
            "|", "\\|"
        )[:180]
        flags_text = ", ".join(row.get("flags") or []).replace("|", "\\|")
        lines.append(
            "| {case} | {snapshot} | `{evidence}` | {risk} | {flags} | {hint} |".format(
                case=str(row.get("case_id") or "").replace("|", "\\|"),
                snapshot=row.get("snapshot_number"),
                evidence=str(row.get("evidence_id") or ""),
                risk=int(row.get("risk_score") or 0),
                flags=flags_text,
                hint=hint,
            )
        )
    return "\n".join(lines) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--replay-summary", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    replay_path = (
        args.replay_summary.expanduser().resolve()
        if args.replay_summary is not None
        else None
    )
    report = audit_manifest(
        _load(manifest_path),
        replay_batch=_load(replay_path) if replay_path is not None else None,
    )
    output_json = args.output_json.expanduser().resolve()
    output_markdown = args.output_markdown.expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_markdown.write_text(_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "candidate_evidence_count": report[
                    "candidate_evidence_count"
                ],
                "classification_complete": report[
                    "classification_complete"
                ],
                "unique_case_evidence_count": report[
                    "unique_case_evidence_count"
                ],
                "primary_status_counts": report[
                    "primary_status_counts"
                ],
                "flag_counts": report["flag_counts"],
                "output_json": str(output_json),
                "output_markdown": str(output_markdown),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
