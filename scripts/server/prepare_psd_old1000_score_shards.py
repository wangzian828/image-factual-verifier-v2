"""Create four balanced frozen-teacher input shards from canonical old1000 gzip cases."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
import os
from pathlib import Path
from typing import Any


def _stat(path: Path) -> dict[str, int]:
    s = path.stat()
    return {"device": s.st_dev, "inode": s.st_ino, "bytes": s.st_size,
            "mtime_ns": s.st_mtime_ns}


def _least_loaded(work: list[int], count: list[int]) -> int:
    if len(work) != 4 or len(count) != 4:
        raise ValueError("teacher scoring requires exactly four GPU shards")
    return min(range(4), key=lambda i: (work[i], count[i], i))


def prepare(*, materialized: Path, output_dir: Path, checkpoint_sha: str) -> dict[str, Any]:
    source_manifest = json.loads((materialized / "manifest.json").read_text())
    if (source_manifest.get("status") != "selected_unscored_requires_raw_teacher_top20"
            or source_manifest.get("checkpoint_manifest_sha256") != checkpoint_sha
            or source_manifest.get("counts") != {"repair": 972, "preserve": 972}):
        raise ValueError("old1000 selected source has not passed materialization gate")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    paths = sorted((materialized / "repair").glob("*.jsonl.gz")) + sorted(
        (materialized / "preserve").glob("*.jsonl.gz"))
    if len(paths) != 611:
        raise ValueError("old1000 canonical case output count changed")
    input_stats = {str(path): _stat(path) for path in paths}
    output_dir.mkdir(parents=True, exist_ok=False)
    shards = [output_dir / f"targets-{index}.jsonl" for index in range(4)]
    handles = [path.open("x", encoding="utf-8") for path in shards]
    work = [0] * 4
    count = [0] * 4
    kinds: Counter[str] = Counter()
    ids: set[str] = set()
    try:
        for path in paths:
            kind = path.parent.name
            done = materialized / "receipts" / (path.name.removesuffix(".jsonl.gz") + f".{kind}.json")
            record = json.loads(done.read_text())
            if record.get("output_identity") != input_stats[str(path)]:
                raise ValueError("canonical case target stat changed")
            case_count = 0
            with gzip.open(path, "rt", encoding="utf-8") as source:
                for line in source:
                    row = json.loads(line)
                    target_id = row.get("target_id")
                    if not isinstance(target_id, str) or not target_id or target_id in ids:
                        raise ValueError("old1000 targets missing/duplicate ID")
                    ids.add(target_id)
                    if (row.get("kind") != kind or row.get("target_status") != "pending_topk"
                            or row.get("teacher_topk_by_position") != []
                            or row.get("teacher_logprob_semantics") != ""):
                        raise ValueError("old1000 target carries non-raw or mislabeled teacher score")
                    teacher = row["model_roles"]["frozen_self_teacher"]
                    student = row["model_roles"]["trainable_student"]
                    if (teacher.get("checkpoint_manifest_sha256") != checkpoint_sha
                            or student.get("checkpoint_manifest_sha256") != checkpoint_sha
                            or teacher.get("model") != student.get("model")
                            or teacher.get("round_start_checkpoint") != student.get("initial_checkpoint")):
                        raise ValueError("mixed old1000 Qwen/checkpoint roles")
                    if 248056 in row["student_prompt_ids"]:
                        media = row.get("psd_media") or {}
                        images = media.get("image_paths") or []
                        if not images or any(not Path(image).is_file() for image in images):
                            raise ValueError("old1000 image-bearing target lacks media artifact")
                    index = _least_loaded(work, count)
                    handles[index].write(line)
                    count[index] += 1
                    work[index] += len(row["teacher_prompt_ids"]) + len(row["completion_ids"])
                    kinds[kind] += 1
                    case_count += 1
            if case_count != record["targets"]:
                raise ValueError("canonical case target count changed")
    finally:
        for handle in handles:
            handle.close()
    if kinds != {"repair": 972, "preserve": 972} or sum(count) != 1944:
        raise ValueError("old1000 teacher shard coverage incomplete")
    if any(_stat(path) != original for path, original in ((Path(raw), stat) for raw, stat in input_stats.items())):
        raise ValueError("canonical case targets changed during sharding")
    manifest = {"schema_version": "ifv-psd-old1000-raw-teacher-shards-v1",
                "status": "frozen_raw_teacher_scoring_pending",
                "source_manifest": str((materialized / "manifest.json").resolve()),
                "source_manifest_identity": _stat(materialized / "manifest.json"),
                "checkpoint_manifest_sha256": checkpoint_sha,
                "counts": dict(kinds), "shard_counts": count,
                "estimated_token_work": work,
                "work_imbalance_fraction": (max(work)-min(work)) / (sum(work)/4),
                "shards": {str(path): _stat(path) for path in shards},
                "no_image_hashing": True, "no_serving_logprobs": True,
                "formal_training_started": False}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--materialized", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--checkpoint-sha", required=True)
    args = p.parse_args()
    print(json.dumps(prepare(materialized=args.materialized, output_dir=args.output_dir,
                             checkpoint_sha=args.checkpoint_sha)))


if __name__ == "__main__":
    main()
