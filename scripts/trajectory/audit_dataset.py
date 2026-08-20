"""Strictly audit a derived full-trajectory SFT dataset before training."""

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
from src.trajectory.schema import DatasetTrajectorySFTExample


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
    examples: List[DatasetTrajectorySFTExample] = []
    episode_row_splits: Dict[str, str] = {}
    episode_splits: Dict[str, set[str]] = defaultdict(set)
    family_splits: Dict[str, set[str]] = defaultdict(set)

    for split in SPLITS:
        for index, row in enumerate(_load_jsonl(root / f"{split}.jsonl")):
            location = f"{split}.jsonl[{index}]"
            try:
                example = DatasetTrajectorySFTExample.model_validate(row)
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
            if example.episode_id in episode_row_splits:
                errors.append(
                    {
                        "code": "DUPLICATE_EPISODE_ID",
                        "location": location,
                        "message": (
                            f"{example.episode_id} also appears in "
                            f"{episode_row_splits[example.episode_id]}"
                        ),
                    }
                )
            episode_row_splits[example.episode_id] = split
            episode_splits[example.episode_id].add(split)
            for family in example.source_family_keys:
                family_splits[family].add(split)
            leaks = list(
                _private_paths(
                    {
                        "messages": example.messages,
                        "tools": example.tools,
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
            # Full-trajectory rows carry the runtime context in the ordered
            # messages/tool responses themselves.  The old step-level schema
            # had a separate runtime_observation_refs field; do not require
            # that legacy field on one-episode-per-row examples.

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
    role_counts: Counter[str] = Counter(
        str(message.get("role", ""))
        for item in examples
        for message in item.messages
    )
    report = {
        "schema_version": "ifv-trajectory-sft-dataset-audit-v1",
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
        "message_role_counts": dict(sorted(role_counts.items())),
        "tool_call_count": sum(item.tool_call_count for item in examples),
        "longest_token_count_estimate": max(
            (item.token_count_estimate for item in examples),
            default=0,
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
