from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import assert_model_visible
from .io import (
    canonical_json,
    load_json,
    load_jsonl,
    require_new_or_empty,
    sha256_file,
    write_json,
    write_jsonl,
)


OUTPUT_VERSION = "ifv-ms-swift-perception-v1"
PERCEPTION_INSTRUCTION = """<image>
Report only literal, visible image content as one JSON object matching the
PerceptionReport contract. Include scene_description, image_type, entities with
normalized bounding boxes, positioned text regions, and uncertainty. Do not use
web knowledge, evaluator labels, or hidden benchmark context."""


def _scores(run_dir: Path) -> dict[str, Mapping[str, Any]]:
    return {
        str(row.get("case_id", "")): row
        for row in load_jsonl(run_dir / "trajectory_scores.jsonl")
        if str(row.get("case_id", ""))
    }


def perception_row(trace: Mapping[str, Any]) -> dict[str, Any]:
    state = trace.get("state")
    if not isinstance(state, Mapping):
        raise ValueError("trace.state must be an object")
    report = state.get("perception")
    if not isinstance(report, Mapping):
        raise ValueError("trace has no PerceptionReport")
    assert_model_visible(report, location="perception")
    image_path = Path(str(trace.get("image_path") or state.get("image_path") or ""))
    if not image_path.is_file():
        raise FileNotFoundError(f"trace image is unavailable: {image_path}")
    return {
        "messages": [
            {"role": "user", "content": PERCEPTION_INSTRUCTION},
            {
                "role": "assistant",
                "content": canonical_json(report),
                "loss": True,
            },
        ],
        "images": [str(image_path.resolve())],
        "chat_template_kwargs": {
            "enable_thinking": False,
            "max_pixels": 1048576,
        },
    }


def convert_perception_runs(
    run_dirs: Sequence[Path],
    output_dir: Path,
    *,
    split_map_path: Path,
) -> dict[str, Any]:
    if not run_dirs:
        raise ValueError("at least one run directory is required")
    require_new_or_empty(output_dir)
    split_map = {
        str(row.get("episode_id", "")): str(row.get("split", ""))
        for row in load_jsonl(split_map_path)
        if str(row.get("episode_id", ""))
    }
    if not split_map:
        raise ValueError(f"split map is empty: {split_map_path}")
    rows_by_split: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    index_rows: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()

    for raw_run_dir in run_dirs:
        run_dir = raw_run_dir.expanduser().resolve()
        run_manifest = load_json(run_dir / "run_manifest.json")
        scores = _scores(run_dir)
        for trace_path in sorted((run_dir / "traces").glob("*.json")):
            trace = load_json(trace_path)
            case_id = str(trace.get("image_id") or trace_path.stem)
            score = scores.get(case_id)
            if not score:
                exclusions["missing_training_quality_score"] += 1
                continue
            if not bool(score.get("training_eligible", False)):
                reasons = score.get("training_exclusion_reasons") or [
                    "training_quality_gate_failed"
                ]
                for reason in reasons:
                    exclusions[str(reason)] += 1
                continue
            split = split_map.get(case_id)
            if split not in rows_by_split:
                exclusions["missing_or_invalid_split"] += 1
                continue
            row = perception_row(trace)
            row["channel"] = "perception"
            rows_by_split[split].append(row)
            index_rows.append(
                {
                    "row_id": len(index_rows),
                    "case_id": case_id,
                    "split": split,
                    "source_run_id": run_manifest.get("run_id"),
                    "runtime_commit": run_manifest.get("git_commit"),
                    "trace_sha256": sha256_file(trace_path),
                    "image_sha256": sha256_file(Path(row["images"][0])),
                }
            )

    index_path = output_dir / "index.jsonl"
    write_jsonl(index_path, index_rows)
    artifacts: dict[str, dict[str, Any]] = {}
    for split, rows in rows_by_split.items():
        dataset_path = output_dir / f"{split}.jsonl"
        write_jsonl(dataset_path, rows)
        artifacts[split] = {
            "path": dataset_path.name,
            "rows": len(rows),
            "sha256": sha256_file(dataset_path),
        }
    artifacts["index"] = {
        "path": index_path.name,
        "rows": len(index_rows),
        "sha256": sha256_file(index_path),
    }
    manifest = {
        "schema_version": "ifv-ms-swift-dataset-manifest-v1",
        "dataset_version": OUTPUT_VERSION,
        "framework": {"name": "ms-swift", "version": "4.4.2"},
        "source_split_map_sha256": sha256_file(split_map_path),
        "example_count": sum(len(rows) for rows in rows_by_split.values()),
        "exclusion_counts": dict(sorted(exclusions.items())),
        "artifacts": artifacts,
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest
