"""Compatibility boundary for benchmark releases produced by the data project.

This module intentionally duplicates no construction-pipeline models.  It accepts
only the public runtime handoff and resolves evaluator-private companion files;
the runtime's own ``VerificationCase`` remains the source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable

from src.orchestrator.state import VerificationCase


RUNTIME_REQUIRED_KEYS = frozenset(
    {
        "case_id",
        "image_path",
        "image_sha256",
        "claim_mode",
    }
)
RUNTIME_CASE_KEYS = frozenset(
    {
        "case_id",
        "image_path",
        "image_sha256",
        "claim_mode",
        "user_claim",
        "claim_surface",
        "claim_source_region",
        "claim_observed_at",
        "decision_policy_version",
    }
)


@dataclass(frozen=True)
class ReleaseCompanions:
    evaluator_private: Path
    evaluation_gold: Path
    source_access_policy: Path


def is_runtime_release_row(sample: Dict[str, Any]) -> bool:
    return RUNTIME_REQUIRED_KEYS.issubset(sample)


def require_uniform_runtime_release(rows: Iterable[Dict[str, Any]]) -> bool:
    """Return whether rows are release rows and reject mixed input formats."""

    classifications = [is_runtime_release_row(row) for row in rows]
    if not classifications:
        return False
    if any(classifications) and not all(classifications):
        raise ValueError(
            "benchmark input mixes runtime release rows with legacy evaluator rows"
        )
    return all(classifications)


def verification_case_from_runtime_row(
    sample: Dict[str, Any],
) -> VerificationCase | None:
    """Project one public runtime row into the runtime-owned case model."""

    if not is_runtime_release_row(sample):
        return None
    actual_keys = frozenset(sample)
    missing = sorted(RUNTIME_CASE_KEYS - actual_keys)
    unexpected = sorted(actual_keys - RUNTIME_CASE_KEYS)
    if missing or unexpected:
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unexpected:
            details.append("unexpected: " + ", ".join(unexpected))
        raise ValueError(
            "runtime release row must match the exact public field set ("
            + "; ".join(details)
            + ")"
        )
    claim_mode = str(sample.get("claim_mode", "")).strip()
    decision_policy = str(sample.get("decision_policy_version", "")).strip()
    if claim_mode == "image_only":
        raise ValueError(
            "image_only releases require the v0.3/reinspect-v2 runtime contract; "
            "v3 has not activated that contract yet"
        )
    if decision_policy != "reinspect-v1":
        raise ValueError(
            "the active runtime accepts decision_policy_version=reinspect-v1; "
            f"received {decision_policy or '<empty>'}"
        )
    return VerificationCase.model_validate(sample)


def release_companions(benchmark_path: Path) -> ReleaseCompanions | None:
    """Locate private companions for ``runtime_input/cases.jsonl``."""

    if benchmark_path.parent.name != "runtime_input":
        return None
    release_root = benchmark_path.parent.parent
    return ReleaseCompanions(
        evaluator_private=release_root / "evaluator_private" / "run_eval.jsonl",
        evaluation_gold=release_root / "evaluation_gold" / "gold.jsonl",
        source_access_policy=(
            release_root / "evaluator_private" / "source_access_policy.json"
        ),
    )


def resolve_runtime_image_path(
    sample: Dict[str, Any],
    benchmark_path: Path,
) -> Dict[str, Any]:
    """Resolve a release-relative image without changing the persisted row."""

    resolved = dict(sample)
    image_path = Path(str(resolved["image_path"]))
    if not image_path.is_absolute():
        resolved["image_path"] = str((benchmark_path.parent / image_path).resolve())
    return resolved
