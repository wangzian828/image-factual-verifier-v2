"""Strictly audit a derived ifv-policy-v1 dataset before any training."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.trajectory.exporter import FORBIDDEN_PRIVATE_KEYS
from src.trajectory.schema import DatasetExample


SPLITS = ("train", "validation", "test")


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} must be a JSON object")
        rows.append(value)
    return rows


def _private_paths(value: Any, path: str = "") -> Iterable[str]:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            child_path = f"{path}.{key}" if path else key
            if key.casefold() in FORBIDDEN_PRIVATE_KEYS:
                yield child_path
            yield from _private_paths(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _private_paths(child, f"{path}[{index}]")


def audit_dataset(dataset_dir: Path) -> Dict[str, Any]:
    root = dataset_dir.expanduser().resolve()
    metadata_rows = _load_jsonl(root / "episode_metadata.jsonl")
    excluded_metadata_rows = _load_jsonl(
        root / "excluded_episode_metadata.jsonl"
    )
    metadata_by_episode = {
        str(row.get("episode_id", "")): row for row in metadata_rows
    }
    errors: List[Dict[str, str]] = []
    examples: List[DatasetExample] = []
    step_splits: Dict[str, str] = {}
    episode_splits: Dict[str, set[str]] = defaultdict(set)
    family_splits: Dict[str, set[str]] = defaultdict(set)

    for split in SPLITS:
        for index, row in enumerate(_load_jsonl(root / f"{split}.jsonl")):
            location = f"{split}.jsonl[{index}]"
            try:
                example = DatasetExample.model_validate(row)
            except Exception as exc:
                errors.append(
                    {
                        "code": "SCHEMA_INVALID",
                        "location": location,
                        "message": str(exc),
                    }
                )
                continue
            examples.append(example)
            if example.split != split:
                errors.append(
                    {
                        "code": "SPLIT_FIELD_MISMATCH",
                        "location": location,
                        "message": (
                            f"record split={example.split}, file split={split}"
                        ),
                    }
                )
            if example.step_id in step_splits:
                errors.append(
                    {
                        "code": "DUPLICATE_STEP_ID",
                        "location": location,
                        "message": (
                            f"{example.step_id} also appears in "
                            f"{step_splits[example.step_id]}"
                        ),
                    }
                )
            step_splits[example.step_id] = split
            episode_splits[example.episode_id].add(split)
            for family in example.source_family_keys:
                family_splits[family].add(split)
            leaks = list(
                _private_paths(
                    {
                        "policy_input": example.policy_input,
                        "policy_action": example.policy_action,
                    }
                )
            )
            for leak in leaks:
                errors.append(
                    {
                        "code": "PRIVATE_DATA_LEAK",
                        "location": f"{location}.{leak}",
                        "message": "evaluator-private field is model-visible",
                    }
                )
            metadata = metadata_by_episode.get(example.episode_id)
            if metadata is None:
                errors.append(
                    {
                        "code": "EPISODE_METADATA_MISSING",
                        "location": location,
                        "message": "episode metadata is missing",
                    }
                )
                continue
            if not bool(metadata.get("training_eligible", False)):
                errors.append(
                    {
                        "code": "TRAINING_QUALITY_GATE_FAILED",
                        "location": location,
                        "message": ", ".join(
                            str(item)
                            for item in metadata.get(
                                "training_exclusion_reasons",
                                [],
                            )
                            or []
                        )
                        or "episode is not marked training eligible",
                    }
                )
                continue
            runtime_ids = {
                str(item) for item in metadata.get("runtime_ids", []) or []
            }
            unknown_refs = sorted(
                set(example.runtime_observation_refs) - runtime_ids
            )
            if unknown_refs:
                errors.append(
                    {
                        "code": "RUNTIME_REFERENCE_UNKNOWN",
                        "location": location,
                        "message": ", ".join(unknown_refs),
                    }
                )

    for episode_id, splits in sorted(episode_splits.items()):
        if len(splits) > 1:
            errors.append(
                {
                    "code": "EPISODE_CROSSES_SPLITS",
                    "location": episode_id,
                    "message": ", ".join(sorted(splits)),
                }
            )
    for family, splits in sorted(family_splits.items()):
        if family and len(splits) > 1:
            errors.append(
                {
                    "code": "SOURCE_FAMILY_CROSSES_SPLITS",
                    "location": family,
                    "message": ", ".join(sorted(splits)),
                }
            )

    scores = [float(item.teacher_score) for item in examples]
    excluded_ids = {
        str(item.get("episode_id", ""))
        for item in excluded_metadata_rows
        if str(item.get("episode_id", ""))
    }
    included_ids = set(episode_splits)
    for episode_id in sorted(excluded_ids & included_ids):
        errors.append(
            {
                "code": "EXCLUDED_EPISODE_INCLUDED",
                "location": episode_id,
                "message": "quality-gate-excluded episode appears in a split",
            }
        )
    example_types = Counter(item.example_type for item in examples)
    verdicts = Counter(
        str(item.policy_action.get("verdict", ""))
        for item in examples
        if item.example_type == "judgment"
    )
    report = {
        "schema_version": "ifv-policy-dataset-audit-v1",
        "dataset_dir": str(root),
        "passed": not errors,
        "error_count": len(errors),
        "errors": errors,
        "example_count": len(examples),
        "episode_count": len(episode_splits),
        "excluded_episode_count": len(excluded_ids),
        "split_counts": dict(
            Counter(item.split for item in examples)
        ),
        "example_type_counts": dict(example_types),
        "judgment_verdict_counts": dict(verdicts),
        "invalid_action_count": sum(
            not item.action_valid for item in examples
        ),
        "fatal_boundary_count": sum(
            item.fatal_boundary for item in examples
        ),
        "teacher_score_distribution": {
            "count": len(scores),
            "min": min(scores) if scores else 0.0,
            "max": max(scores) if scores else 0.0,
            "mean": statistics.mean(scores) if scores else 0.0,
        },
    }
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = audit_dataset(args.input)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.strict and not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
