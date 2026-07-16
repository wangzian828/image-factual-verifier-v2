"""Build a split-safe policy dataset from completed v3 evaluation runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.trajectory.schema import (
    DatasetExample,
    DatasetPerceptionExample,
    PerceptionExample,
    PolicyExample,
 )


SPLITS = ("train", "validation", "test")


def _load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
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


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    lines = [
        json.dumps(row, ensure_ascii=False, separators=(",", ":"))
        for row in rows
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(lines) + ("\n" if lines else ""),
        encoding="utf-8",
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _rows(value: Any) -> List[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _collect_runtime_ids(trace: Mapping[str, Any]) -> set[str]:
    result: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if str(key).endswith("_id") and isinstance(child, str):
                    if child.strip():
                        result.add(child.strip())
                elif str(key).endswith("_ids") and isinstance(child, list):
                    result.update(
                        str(item).strip()
                        for item in child
                        if str(item).strip()
                    )
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(trace)
    return result


def _source_families(trace: Mapping[str, Any]) -> List[str]:
    state = _mapping(trace.get("state"))
    investigation = _mapping(state.get("investigation_state"))
    families = {
        str(item.get("source_family", "")).strip()
        for item in _rows(investigation.get("evidence"))
        if str(item.get("source_family", "")).strip()
    }
    return sorted(families)


class _UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self.parent[max(root_left, root_right)] = min(root_left, root_right)


def _assign_split(
    group_id: str,
    *,
    train_ratio: float,
    validation_ratio: float,
    seed: str,
) -> str:
    digest = hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    if value < train_ratio:
        return "train"
    if value < train_ratio + validation_ratio:
        return "validation"
    return "test"


def export_dataset(
    run_dirs: Sequence[Path],
    output_dir: Path,
    *,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    seed: str = "ifv-policy-v1",
) -> Dict[str, Any]:
    if not run_dirs:
        raise ValueError("at least one run directory is required")
    if train_ratio <= 0 or validation_ratio < 0:
        raise ValueError("split ratios must be non-negative")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("train_ratio + validation_ratio must be below 1")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"dataset output directory must be new or empty: {output_dir}"
        )

    examples_by_episode: Dict[str, List[PolicyExample]] = defaultdict(list)
    perception_by_episode: Dict[str, PerceptionExample] = {}
    episode_metadata: Dict[str, Dict[str, Any]] = {}
    score_by_episode: Dict[str, Dict[str, Any]] = {}
    source_runs: List[Dict[str, Any]] = []
    for raw_run_dir in run_dirs:
        run_dir = raw_run_dir.expanduser().resolve()
        manifest = _load_json(run_dir / "run_manifest.json")
        source_runs.append(
            {
                "run_id": manifest.get("run_id"),
                "path": str(run_dir),
                "git_commit": manifest.get("git_commit"),
                "release_id": _mapping(manifest.get("benchmark")).get(
                    "release_id"
                ),
            }
        )
        for row in _load_jsonl(run_dir / "trajectory_scores.jsonl"):
            score_by_episode[str(row.get("case_id", ""))] = dict(row)
        for row in _load_jsonl(run_dir / "policy_trajectories.jsonl"):
            example = PolicyExample.model_validate(row)
            examples_by_episode[example.episode_id].append(example)
        for row in _load_jsonl(run_dir / "perception_trajectories.jsonl"):
            example = PerceptionExample.model_validate(row)
            if example.episode_id in perception_by_episode:
                raise ValueError(
                    "duplicate perception episode across source runs: "
                    f"{example.episode_id}"
                )
            perception_by_episode[example.episode_id] = example
        for trace_path in sorted((run_dir / "traces").glob("*.json")):
            trace = _load_json(trace_path)
            episode_id = str(
                trace.get("image_id") or trace_path.stem
            ).strip()
            score = score_by_episode.get(episode_id, {})
            training_eligible = bool(score.get("training_eligible", False))
            exclusion_reasons = [
                str(item)
                for item in score.get(
                    "training_exclusion_reasons",
                    ["missing_training_quality_score"],
                )
                or []
            ]
            if not score:
                exclusion_reasons = ["missing_training_quality_score"]
            episode_metadata[episode_id] = {
                "episode_id": episode_id,
                "source_run_id": manifest.get("run_id"),
                "source_trace": str(trace_path),
                "source_family_keys": _source_families(trace),
                "runtime_ids": sorted(_collect_runtime_ids(trace)),
                "teacher_score": float(score.get("total", 0.0) or 0.0),
                "training_eligible": training_eligible,
                "training_exclusion_reasons": exclusion_reasons,
            }

    candidate_ids = set(examples_by_episode) | set(perception_by_episode)
    unknown_metadata = sorted(candidate_ids - set(episode_metadata))
    if unknown_metadata:
        raise ValueError(
            "policy examples lack canonical trace metadata: "
            + ", ".join(unknown_metadata)
        )

    all_episodes = sorted(candidate_ids)
    episodes = [
        episode_id
        for episode_id in all_episodes
        if episode_metadata[episode_id]["training_eligible"]
    ]
    excluded_episode_rows = [
        {
            **episode_metadata[episode_id],
            "example_count": len(examples_by_episode[episode_id]),
            "perception_example_count": int(
                episode_id in perception_by_episode
            ),
        }
        for episode_id in all_episodes
        if episode_id not in episodes
    ]
    union = _UnionFind(episodes)
    family_episodes: Dict[str, List[str]] = defaultdict(list)
    for episode_id in episodes:
        for family in episode_metadata[episode_id]["source_family_keys"]:
            family_episodes[family].append(episode_id)
    for members in family_episodes.values():
        for member in members[1:]:
            union.union(members[0], member)

    components: Dict[str, List[str]] = defaultdict(list)
    for episode_id in episodes:
        components[union.find(episode_id)].append(episode_id)
    split_by_episode: Dict[str, str] = {}
    group_by_episode: Dict[str, str] = {}
    for members in components.values():
        group_payload = {
            "episodes": sorted(members),
            "families": sorted(
                {
                    family
                    for episode_id in members
                    for family in episode_metadata[episode_id][
                        "source_family_keys"
                    ]
                }
            ),
        }
        group_id = hashlib.sha256(
            json.dumps(
                group_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:20]
        split = _assign_split(
            group_id,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            seed=seed,
        )
        for episode_id in members:
            split_by_episode[episode_id] = split
            group_by_episode[episode_id] = group_id

    split_rows: Dict[str, List[Dict[str, Any]]] = {
        split: [] for split in SPLITS
    }
    perception_split_rows: Dict[str, List[Dict[str, Any]]] = {
        split: [] for split in SPLITS
    }
    metadata_rows: List[Dict[str, Any]] = []
    for episode_id in episodes:
        metadata = episode_metadata[episode_id]
        split = split_by_episode[episode_id]
        metadata_rows.append(
            {
                **metadata,
                "split": split,
                "split_group_id": group_by_episode[episode_id],
            }
        )
        for example in examples_by_episode[episode_id]:
            dataset_example = DatasetExample(
                **example.model_dump(mode="json"),
                split=split,
                split_group_id=group_by_episode[episode_id],
                source_family_keys=metadata["source_family_keys"],
                teacher_score=metadata["teacher_score"],
            )
            split_rows[split].append(
                dataset_example.model_dump(mode="json")
            )
        perception_example = perception_by_episode.get(episode_id)
        if perception_example is not None:
            dataset_perception = DatasetPerceptionExample(
                **perception_example.model_dump(mode="json"),
                split=split,
                split_group_id=group_by_episode[episode_id],
            )
            perception_split_rows[split].append(
                dataset_perception.model_dump(mode="json")
            )

    for split, rows in split_rows.items():
        rows.sort(key=lambda item: (item["episode_id"], item["step_id"]))
        _write_jsonl(output_dir / f"{split}.jsonl", rows)
        perception_rows = perception_split_rows[split]
        perception_rows.sort(key=lambda item: item["episode_id"])
        _write_jsonl(
            output_dir / f"perception.{split}.jsonl",
            perception_rows,
        )
    metadata_rows.sort(key=lambda item: item["episode_id"])
    _write_jsonl(output_dir / "episode_metadata.jsonl", metadata_rows)
    excluded_episode_rows.sort(key=lambda item: item["episode_id"])
    _write_jsonl(
        output_dir / "excluded_episode_metadata.jsonl",
        excluded_episode_rows,
    )
    manifest = {
        "schema_version": "ifv-policy-dataset-manifest-v2",
        "dataset_version": "ifv-policy-dataset-v2",
        "trajectory_version": "ifv-policy-v1",
        "perception_version": "ifv-perception-v1",
        "seed": seed,
        "split_ratios": {
            "train": train_ratio,
            "validation": validation_ratio,
            "test": 1 - train_ratio - validation_ratio,
        },
        "source_runs": source_runs,
        "episode_count": len(episodes),
        "candidate_episode_count": len(all_episodes),
        "excluded_episode_count": len(excluded_episode_rows),
        "group_count": len(components),
        "example_counts": {
            split: len(rows) for split, rows in split_rows.items()
        },
        "perception_example_counts": {
            split: len(rows)
            for split, rows in perception_split_rows.items()
        },
        "artifacts": {
            "train": "train.jsonl",
            "validation": "validation.jsonl",
            "test": "test.jsonl",
            "perception_train": "perception.train.jsonl",
            "perception_validation": "perception.validation.jsonl",
            "perception_test": "perception.test.jsonl",
            "episode_metadata": "episode_metadata.jsonl",
            "excluded_episode_metadata": (
                "excluded_episode_metadata.jsonl"
            ),
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", default="ifv-policy-v1")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = export_dataset(
        args.run_dir,
        args.output_dir,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
