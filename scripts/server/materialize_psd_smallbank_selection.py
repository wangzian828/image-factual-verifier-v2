"""Stream the previously scored smallbank into a whole-episode selected bank.

The 1.7 GB scored source is read exactly once. No model, pixel, or source
trajectory is hashed or copied; only selected target rows are written.
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


def _jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def _review_model(episode: dict[str, Any]) -> str:
    review = episode.get("source_task_review") or {}
    path = Path(review.get("path", ""))
    if not path.is_file():
        raise ValueError("selected preservation review is unavailable")
    value = json.loads(path.read_text(encoding="utf-8"))
    payload = value.get("payload") or value
    verifier = payload.get("verifier") or {}
    model = verifier.get("model")
    if not isinstance(model, str) or not model.startswith("gemini-"):
        raise ValueError("selected preservation has no real Gemini reviewer")
    return model


def materialize(*, scored: Path, selection: Path, episode_index: Path,
                assembled: Path, output_dir: Path, model: str,
                checkpoint: str, checkpoint_sha: str, runtime_code: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    sys.path[:0] = [str(runtime_code.resolve()), str((runtime_code / "training").resolve())]
    from ifv_training.psd import _teacher_identity, validate_topk_by_position
    from ifv_training.psd_capture_semantics import RAW_POLICY_LOGPROBS

    selected = {r["case_id"]: r for r in _jsonl(selection) if r["source"] == "smallbank"}
    indexed = {r["case_id"]: r for r in _jsonl(episode_index)}
    if len(selected) != 44 or not set(selected) <= set(indexed):
        raise ValueError("smallbank selected episodes differ from frozen index")
    expected: dict[str, list[str]] = {}
    reviewer: dict[str, str] = {}
    assembled_before = _stat(assembled)
    with assembled.open("rb") as source:
        for case_id, choice in selected.items():
            meta = indexed[case_id]
            if choice["episode_id"] != meta["episode_id"] or choice["step_count"] != meta["step_count"]:
                raise ValueError("selected smallbank episode identity changed")
            source.seek(meta["source_offset"])
            episode = json.loads(source.read(meta["source_length"]))
            if episode.get("case_id") != case_id or episode.get("episode_id") != choice["episode_id"]:
                raise ValueError("selected assembled preservation differs from index")
            if episode.get("verified_full_task") is not True or episode.get("strict_trace_audit_pass") is not True:
                raise ValueError("selected preservation was not strictly verified")
            steps = episode["preservation_steps"]
            if len(steps) != choice["step_count"] or 248056 not in steps[-1]["student_prompt_ids"]:
                raise ValueError("selected preservation is incomplete or judgment lacks image")
            expected[case_id] = [step["step_id"] for step in steps]
            reviewer[case_id] = _review_model(episode)
    if _stat(assembled) != assembled_before:
        raise ValueError("assembled source changed during selected episode audit")

    scored_before = _stat(scored)
    output_dir.mkdir(parents=True, exist_ok=False)
    observed: dict[str, list[str]] = {case: [] for case in selected}
    kinds: Counter[str] = Counter()
    provider_models: Counter[str] = Counter()
    ids: set[str] = set()
    with scored.open(encoding="utf-8") as source, (output_dir / "targets.jsonl").open("x", encoding="utf-8") as output:
        for line in source:
            target = json.loads(line)
            kind, case_id = target.get("kind"), target.get("case_id")
            if kind == "preserve" and case_id not in selected:
                continue
            if kind not in {"repair", "preserve"}:
                raise ValueError("unknown scored smallbank target kind")
            identity = _teacher_identity(target)
            if (identity["provider"] != "qwen_local" or identity["model"] != model
                    or identity["checkpoint"] != checkpoint
                    or identity["checkpoint_manifest_sha256"] != checkpoint_sha):
                raise ValueError("smallbank Qwen teacher/student/checkpoint mismatch")
            if target.get("target_status") != "complete" or target.get("teacher_logprob_semantics") != RAW_POLICY_LOGPROBS:
                raise ValueError("smallbank target does not have raw complete teacher scores")
            completion = target.get("completion_ids")
            validate_topk_by_position(completion, target.get("teacher_topk_by_position"), topk=20)
            tid = target.get("target_id")
            if not isinstance(tid, str) or tid in ids:
                raise ValueError("duplicate/missing smallbank target ID")
            ids.add(tid)
            media = target.get("psd_media") or {}
            if 248056 in target.get("student_prompt_ids", []):
                files = media.get("image_paths") or ([media["path"]] if media.get("path") else [])
                if not files or any(not Path(path).is_file() for path in files):
                    raise ValueError("image-bearing smallbank target lacks media artifact")
            if kind == "repair":
                constructor = target["model_roles"]["hint_constructor"]
                if constructor.get("provider") != "gemini" or not constructor.get("model"):
                    raise ValueError("smallbank repair has no real Gemini hint source")
                provider_models[constructor["model"]] += 1
            else:
                observed[case_id].append(target["repair_site"]["step_id"])
                target.setdefault("source", {})["gemini_source_review_model"] = reviewer[case_id]
                provider_models[reviewer[case_id]] += 1
                if target["repair_site"]["step_id"] == expected[case_id][-1] and 248056 not in target["student_prompt_ids"]:
                    raise ValueError("smallbank final judgment target lost its image")
            kinds[kind] += 1
            output.write(json.dumps(target, ensure_ascii=False, separators=(",", ":")) + "\n")
    if _stat(scored) != scored_before:
        raise ValueError("scored smallbank source changed during streaming")
    if any(observed[case] != steps for case, steps in expected.items()):
        raise ValueError("smallbank whole preservation episode incomplete/out of order")
    if kinds != {"repair": 626, "preserve": 626}:
        raise ValueError(f"smallbank selected target counts differ: {dict(kinds)}")
    manifest = {"schema_version": "ifv-psd-selected-scored-smallbank-v1",
                "status": "selected_scored_source_pending_combined_gate",
                "source_scored": str(scored.resolve()), "source_scored_identity": scored_before,
                "selected_preservation_episodes": len(expected),
                "counts": dict(kinds), "gemini_sources": dict(sorted(provider_models.items())),
                "qwen_model": model, "checkpoint_manifest_sha256": checkpoint_sha,
                "top20_semantics": RAW_POLICY_LOGPROBS, "formal_training_started": False,
                "large_payload_hashing": False, "source_trajectory_rescan": False}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("scored", "selection", "episode-index", "assembled", "output-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--runtime-code", required=True, type=Path)
    for name in ("model", "checkpoint", "checkpoint-sha"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(scored=args.scored, selection=args.selection,
        episode_index=args.episode_index, assembled=args.assembled,
        output_dir=args.output_dir, model=args.model, checkpoint=args.checkpoint,
        checkpoint_sha=args.checkpoint_sha, runtime_code=args.runtime_code)))


if __name__ == "__main__":
    main()
