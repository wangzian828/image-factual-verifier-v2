"""Merge selected scored smallbank and exact raw-teacher-scored old1000 shards.

Each selected target is written once. Old1000 shard caches are checked against
target ID, exact prompt/completion IDs, image descriptor, teacher identity and
raw-unmasked semantics before being attached. No model/image file is hashed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any


def _stat(path: Path) -> dict[str, int]:
    s = path.stat()
    return {"device": s.st_dev, "inode": s.st_ino, "bytes": s.st_size,
            "mtime_ns": s.st_mtime_ns}


def merge(*, small_targets: Path, small_manifest: Path, shards: Path,
          score_root: Path, output_dir: Path, runtime_code: Path,
          model: str, checkpoint: str, checkpoint_sha: str) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    sys.path[:0] = [str(runtime_code.resolve()), str((runtime_code / "training").resolve())]
    from ifv_training.psd import (_teacher_identity, _token_ids_sha256,
                                  validate_topk_by_position)
    from ifv_training.psd_media import media_digest
    from ifv_training.psd_capture_semantics import RAW_POLICY_LOGPROBS

    small_record = json.loads(small_manifest.read_text())
    shard_record = json.loads((shards / "manifest.json").read_text())
    score_state = json.loads((score_root / "state.json").read_text())
    if (small_record.get("counts") != {"repair": 626, "preserve": 626}
            or small_record.get("checkpoint_manifest_sha256") != checkpoint_sha
            or shard_record.get("counts") != {"repair": 972, "preserve": 972}
            or shard_record.get("checkpoint_manifest_sha256") != checkpoint_sha
            or score_state.get("phase") != "raw_teacher_scoring_complete"
            or score_state.get("services_restored") is not True):
        raise ValueError("combined source or raw scoring completion gate not passed")
    input_stats = {str(path): _stat(path) for path in [small_targets, small_manifest,
        shards / "manifest.json", score_root / "state.json"]}
    for index in range(4):
        target_path = shards / f"targets-{index}.jsonl"
        cache_path = score_root / f"gpu-{index}/teacher_topk_cache.jsonl"
        score_manifest = json.loads((score_root / f"gpu-{index}/manifest.json").read_text())
        if score_manifest.get("status") != "ready_for_materialization":
            raise ValueError(f"GPU {index} teacher cache is not complete")
        if shard_record["shards"].get(str(target_path)) != _stat(target_path):
            raise ValueError(f"GPU {index} target shard changed")
        input_stats[str(target_path)] = _stat(target_path)
        input_stats[str(cache_path)] = _stat(cache_path)

    output_dir.mkdir(parents=True, exist_ok=False)
    target_ids: set[str] = set()
    counts: Counter[str] = Counter()
    gemini: Counter[str] = Counter()
    image_counts: Counter[str] = Counter()
    with (output_dir / "targets.jsonl").open("x", encoding="utf-8") as output:
        def append(row: dict[str, Any], batch: str) -> None:
            tid = row.get("target_id")
            if not isinstance(tid, str) or not tid or tid in target_ids:
                raise ValueError("combined PSD target ID missing/duplicated")
            target_ids.add(tid)
            identity = _teacher_identity(row)
            if (identity["provider"] != "qwen_local" or identity["model"] != model
                    or identity["checkpoint"] != checkpoint
                    or identity["checkpoint_manifest_sha256"] != checkpoint_sha):
                raise ValueError("combined teacher/student/checkpoint identity mismatch")
            if row.get("target_status") != "complete" or row.get("teacher_logprob_semantics") != RAW_POLICY_LOGPROBS:
                raise ValueError("combined target lacks raw unmasked teacher top20")
            validate_topk_by_position(row["completion_ids"], row["teacher_topk_by_position"], topk=20)
            kind = row.get("kind")
            if kind not in {"repair", "preserve"}:
                raise ValueError("combined target has unknown kind")
            source = row.setdefault("source", {})
            actual_gemini = (row["model_roles"]["hint_constructor"].get("model")
                             if kind == "repair" else source.get("gemini_source_review_model"))
            if not isinstance(actual_gemini, str) or not actual_gemini.startswith("gemini-"):
                raise ValueError("combined target lost genuine Gemini provenance")
            if kind == "repair" and source.get("gemini_hint_model", actual_gemini) != actual_gemini:
                raise ValueError("repair Gemini ledger and role source disagree")
            source["training_batch"] = batch
            gemini[actual_gemini] += 1
            if 248056 in row["student_prompt_ids"]:
                media = row.get("psd_media") or {}
                files = media.get("image_paths") or ([media["path"]] if media.get("path") else [])
                if not files or any(not Path(path).is_file() for path in files):
                    raise ValueError("image-bearing combined target lacks media")
                image_counts[batch] += 1
            if kind == "repair" and 248056 not in row["student_prompt_ids"]:
                raise ValueError("accepted image-conditioned repair lost image")
            counts[f"{batch}_{kind}"] += 1
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

        with small_targets.open(encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                teacher = row.get("teacher") or {}
                if (teacher.get("provider") != "qwen_local" or teacher.get("model") != model
                        or teacher.get("checkpoint") != checkpoint
                        or teacher.get("checkpoint_manifest_sha256") != checkpoint_sha
                        or teacher.get("teacher_prompt_sha256") != _token_ids_sha256(row["teacher_prompt_ids"])
                        or teacher.get("completion_sha256") != _token_ids_sha256(row["completion_ids"])):
                    raise ValueError("smallbank reused score differs from exact token/checkpoint binding")
                append(row, "smallbank")

        for index in range(4):
            cache_path = score_root / f"gpu-{index}/teacher_topk_cache.jsonl"
            target_path = shards / f"targets-{index}.jsonl"
            cache: dict[str, dict[str, Any]] = {}
            with cache_path.open(encoding="utf-8") as source:
                for line in source:
                    row = json.loads(line)
                    tid = row.get("target_id")
                    if not isinstance(tid, str) or tid in cache:
                        raise ValueError("raw teacher cache missing/duplicate target ID")
                    cache[tid] = row
            if len(cache) != shard_record["shard_counts"][index]:
                raise ValueError("raw teacher cache shard count incomplete")
            with target_path.open(encoding="utf-8") as source:
                for line in source:
                    row = json.loads(line)
                    tid = row["target_id"]
                    score = cache.pop(tid, None)
                    if score is None:
                        raise ValueError("missing raw teacher cache entry")
                    identity = _teacher_identity(row)
                    if (score.get("teacher_provider") != identity["provider"]
                            or score.get("teacher_model") != identity["model"]
                            or score.get("teacher_checkpoint") != identity["checkpoint"]
                            or score.get("teacher_checkpoint_manifest_sha256") != identity["checkpoint_manifest_sha256"]
                            or score.get("teacher_prompt_sha256") != _token_ids_sha256(row["teacher_prompt_ids"])
                            or score.get("completion_sha256") != _token_ids_sha256(row["completion_ids"])
                            or score.get("teacher_logprob_semantics") != RAW_POLICY_LOGPROBS
                            or (score.get("collection") or {}).get("engine") != "transformers"):
                        raise ValueError("old1000 score differs from exact raw teacher binding")
                    if row.get("psd_media") and score.get("media_sha256") != media_digest(row["psd_media"]):
                        raise ValueError("old1000 score differs from image descriptor")
                    row["target_status"] = "complete"
                    row["teacher_topk_by_position"] = score["teacher_topk_by_position"]
                    row["teacher_logprob_semantics"] = RAW_POLICY_LOGPROBS
                    row["teacher"] = {"provider": identity["provider"], "model": identity["model"],
                                      "checkpoint": identity["checkpoint"],
                                      "checkpoint_manifest_sha256": identity["checkpoint_manifest_sha256"],
                                      "teacher_prompt_sha256": score["teacher_prompt_sha256"],
                                      "completion_sha256": score["completion_sha256"]}
                    append(row, "old1000")
            if cache:
                raise ValueError("orphaned raw teacher cache entries")
    if counts != {"smallbank_repair": 626, "smallbank_preserve": 626,
                  "old1000_repair": 972, "old1000_preserve": 972}:
        raise ValueError(f"combined PSD counts differ: {dict(counts)}")
    for raw, identity in input_stats.items():
        if _stat(Path(raw)) != identity:
            raise ValueError("combined source changed during merge")
    manifest = {"schema_version": "ifv-psd-combined-sft3-scored-v1",
                "status": "combined_scored_pending_datum_and_dp4_gate",
                "targets": 3196, "repair_targets": 1598, "preservation_targets": 1598,
                "counts": dict(sorted(counts.items())),
                "image_bearing_by_source": dict(sorted(image_counts.items())),
                "gemini_sources": dict(sorted(gemini.items())),
                "qwen_model": model, "checkpoint": checkpoint,
                "checkpoint_manifest_sha256": checkpoint_sha,
                "teacher_logprob_semantics": RAW_POLICY_LOGPROBS,
                "source_stat_bindings": input_stats,
                "no_model_or_image_hashing": True, "formal_training_started": False}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("small-targets", "small-manifest", "shards", "score-root", "output-dir", "runtime-code"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("model", "checkpoint", "checkpoint-sha"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    print(json.dumps(merge(small_targets=a.small_targets, small_manifest=a.small_manifest,
        shards=a.shards, score_root=a.score_root, output_dir=a.output_dir,
        runtime_code=a.runtime_code, model=a.model, checkpoint=a.checkpoint,
        checkpoint_sha=a.checkpoint_sha)))


if __name__ == "__main__":
    main()
