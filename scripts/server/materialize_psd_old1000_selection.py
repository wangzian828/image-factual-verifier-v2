"""Incrementally materialize frozen old1000 PSD targets without resuming repair.

Read accepted gzip ledgers by frozen row reference and seek only the selected
verified preservation candidate offsets. Each case gets one canonical gzip
target file. Servings' masked logprobs are explicitly discarded: all these
targets require frozen Transformers raw top-20 scoring before training.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, Counter
import gzip
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any


def _stat(path: Path) -> dict[str, int]:
    s = path.stat()
    return {"device": s.st_dev, "inode": s.st_ino, "bytes": s.st_size,
            "mtime_ns": s.st_mtime_ns}


def _rows(path: Path):
    with path.open(encoding="utf-8") as inp:
        for line in inp:
            yield json.loads(line)


def _write_case(path: Path, targets: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(path)
    with gzip.open(temporary, "wt", encoding="utf-8") as out:
        for target in targets:
            out.write(json.dumps(target, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def _pending(target: dict[str, Any]) -> dict[str, Any]:
    # A serving-captured distribution, even if present, is not the unmasked
    # SFT3 self-teacher distribution required by this training package.
    target["target_status"] = "pending_topk"
    target["teacher_topk_by_position"] = []
    target["teacher_logprob_semantics"] = ""
    return target


def materialize(*, accepted_index: Path, selected_episodes: Path,
                fixed_index: Path, fixed_candidates: Path, output_dir: Path,
                runtime_code: Path, audit_script: Path, checkpoint: str, checkpoint_sha: str,
                model: str) -> dict[str, Any]:
    sys.path[:0] = [str(runtime_code.resolve()), str((runtime_code / "training").resolve())]
    from ifv_training.psd import build_repair_target, build_preservation_targets
    from ifv_training.psd_repairs import _preservation_row
    spec = importlib.util.spec_from_file_location("frozen_psd_audit", audit_script)
    if spec is None or spec.loader is None:
        raise ValueError("frozen accepted audit code unavailable")
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)

    accepted_before, fixed_before = _stat(accepted_index), _stat(fixed_candidates)
    accepted = list(_rows(accepted_index))
    selected = [row for row in _rows(selected_episodes) if row["source"] == "old1000"]
    fixed = {row["case_id"]: row for row in _rows(fixed_index)}
    if len(accepted) != 990 or len(selected) != 67 or len(fixed) != 428:
        raise ValueError("frozen old1000 inputs have unexpected counts")
    if len({row["case_id"] for row in selected}) != len(selected):
        raise ValueError("duplicate selected preservation case")
    for row in selected:
        meta = fixed.get(row["case_id"])
        if not meta or row["episode_id"] != meta["episode_id"] or row["step_count"] != meta["step_count"]:
            raise ValueError("selected old1000 episode differs from frozen index")
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError("old1000 materialization output is not a directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = output_dir / "source-binding.json"
    binding = {"schema_version": "ifv-psd-old1000-selected-source-binding-v1",
               "accepted_index": str(accepted_index.resolve()), "accepted_identity": accepted_before,
               "selected_episodes": str(selected_episodes.resolve()),
               "selected_identity": _stat(selected_episodes),
               "fixed_index": str(fixed_index.resolve()), "fixed_identity": _stat(fixed_index),
               "fixed_candidates": str(fixed_candidates.resolve()), "fixed_identity": fixed_before,
               "checkpoint": checkpoint, "checkpoint_sha": checkpoint_sha,
               "model": model, "repair_is_frozen": True}
    if receipt_path.exists():
        if json.loads(receipt_path.read_text()) != binding:
            raise ValueError("old1000 selection source binding changed")
    else:
        receipt_path.write_text(json.dumps(binding, indent=2) + "\n")

    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in accepted:
        if row["image_count"]:
            by_case[row["case_id"]].append(row)
    if sum(map(len, by_case.values())) != 972:
        raise ValueError("old1000 image-bearing accepted repair count changed")
    generated: Counter[str] = Counter()
    gemini: Counter[str] = Counter()
    target_ids: set[str] = set()
    core_roles: dict[str, Any] | None = None

    for case_id, refs in sorted(by_case.items()):
        destination = output_dir / "repair" / (case_id + ".jsonl.gz")
        done = output_dir / "receipts" / (case_id + ".repair.json")
        if done.exists():
            record = json.loads(done.read_text())
            if record.get("output_identity") != _stat(destination) or record.get("targets") != len(refs):
                raise ValueError("completed repair case output changed")
            generated["repair"] += len(refs)
            gemini.update(record["gemini_sources"])
            continue
        # Usually one canonical ledger per case. Read it once even when the
        # verified case contains several accepted repair steps.
        wanted: dict[Path, dict[int, dict[str, Any]]] = defaultdict(dict)
        for ref in refs:
            wanted[Path(ref["source_path"])][ref["source_row"]] = ref
        targets: list[dict[str, Any]] = []
        local_sources: Counter[str] = Counter()
        for path, positions in wanted.items():
            with gzip.open(path, "rt", encoding="utf-8") as ledger:
                for ordinal, line in enumerate(ledger):
                    if ordinal not in positions:
                        continue
                    ref = positions.pop(ordinal)
                    attempt = json.loads(line)
                    if attempt.get("attempt_id") != ref["attempt_id"] or attempt.get("case_id") != case_id:
                        raise ValueError("frozen accepted repair reference changed")
                    audit._strict(attempt)
                    source_model = audit._roles(attempt, model, checkpoint_sha, checkpoint)
                    if source_model != ref["gemini_hint_model"]:
                        raise ValueError("repair Gemini source binding changed")
                    if core_roles is None:
                        core_roles = attempt["model_roles"]
                    target = _pending(build_repair_target(attempt))
                    target.setdefault("source", {})["gemini_hint_model"] = source_model
                    if target["target_id"] in target_ids:
                        raise ValueError("duplicate old1000 target ID")
                    target_ids.add(target["target_id"])
                    targets.append(target)
                    local_sources[source_model] += 1
            if positions:
                raise ValueError("accepted ledger ended before frozen repair row")
        if len(targets) != len(refs):
            raise ValueError("frozen case repair target count changed")
        _write_case(destination, targets)
        done.parent.mkdir(parents=True, exist_ok=True)
        done.write_text(json.dumps({"case_id": case_id, "kind": "repair",
            "targets": len(targets), "output_identity": _stat(destination),
            "gemini_sources": dict(local_sources)}, indent=2) + "\n")
        generated["repair"] += len(targets)
        gemini.update(local_sources)

    if core_roles is None:
        # Resuming a completed repair stage still needs the Qwen role template
        # for the selected fixed episodes. Read one frozen accepted row only.
        ref = next(row for row in accepted if row["image_count"])
        with gzip.open(ref["source_path"], "rt", encoding="utf-8") as ledger:
            for ordinal, line in enumerate(ledger):
                if ordinal == ref["source_row"]:
                    core_roles = json.loads(line)["model_roles"]
                    break
    if core_roles is None:
        raise ValueError("missing frozen Qwen model roles")

    with fixed_candidates.open("rb") as source:
        for choice in sorted(selected, key=lambda row: row["case_id"]):
            case_id = choice["case_id"]
            destination = output_dir / "preserve" / (case_id + ".jsonl.gz")
            done = output_dir / "receipts" / (case_id + ".preserve.json")
            if done.exists():
                record = json.loads(done.read_text())
                if record.get("output_identity") != _stat(destination) or record.get("targets") != choice["step_count"]:
                    raise ValueError("completed preservation case output changed")
                generated["preserve"] += choice["step_count"]
                gemini.update(record["gemini_sources"])
                continue
            meta = fixed[case_id]
            source.seek(meta["source_offset"])
            candidate = json.loads(source.read(meta["source_length"]))
            if (candidate.get("case_id") != case_id or candidate.get("episode_id") != choice["episode_id"]
                    or candidate.get("verified_full_task") is not True
                    or candidate.get("strict_trace_audit_pass") is not True
                    or len(candidate.get("preservation_steps", [])) != choice["step_count"]):
                raise ValueError("selected preservation candidate changed")
            if [step.get("step_id") for step in candidate["preservation_steps"]] != meta["step_ids"]:
                raise ValueError("selected preservation step sequence changed")
            last_ids = candidate["preservation_steps"][-1]["rollout_token_capture"]["prompt_token_ids"]
            if 248056 not in last_ids:
                raise ValueError("selected fixed final judgment has no image")
            review = json.loads(Path(meta["source_review_path"]).read_text())
            payload = review.get("payload") or {}
            model_name = (payload.get("verifier") or {}).get("model")
            if (payload.get("decision") or {}).get("status") != "pass" or not isinstance(model_name, str) or not model_name.startswith("gemini-"):
                raise ValueError("fixed source lacks genuine passed Gemini review")
            roles = json.loads(json.dumps(core_roles))
            roles["hint_constructor"] = {"provider": "gemini", "model": model_name,
                                         "supplies_training_distribution": False}
            preservation = _preservation_row(candidate, model_roles=roles,
                media_dir=output_dir / "media")
            targets = [_pending(target) for target in build_preservation_targets(preservation)]
            if len(targets) != choice["step_count"]:
                raise ValueError("complete fixed episode target count changed")
            for target in targets:
                target.setdefault("source", {})["gemini_source_review_model"] = model_name
                if target["target_id"] in target_ids:
                    raise ValueError("duplicate old1000 target ID")
                target_ids.add(target["target_id"])
            _write_case(destination, targets)
            done.parent.mkdir(parents=True, exist_ok=True)
            done.write_text(json.dumps({"case_id": case_id, "kind": "preserve",
                "targets": len(targets), "output_identity": _stat(destination),
                "gemini_sources": {model_name: len(targets)}}, indent=2) + "\n")
            generated["preserve"] += len(targets)
            gemini[model_name] += len(targets)

    if _stat(accepted_index) != accepted_before or _stat(fixed_candidates) != fixed_before:
        raise ValueError("frozen source changed during old1000 materialization")
    if generated != {"repair": 972, "preserve": 972}:
        raise ValueError(f"old1000 materialized counts differ: {dict(generated)}")
    manifest = {"schema_version": "ifv-psd-selected-old1000-unscored-v1",
                "status": "selected_unscored_requires_raw_teacher_top20",
                "counts": dict(generated), "gemini_sources": dict(sorted(gemini.items())),
                "checkpoint_manifest_sha256": checkpoint_sha,
                "selected_preservation_episodes": len(selected),
                "canonical_per_case_gzip": True, "raw_teacher_scoring_complete": False,
                "formal_training_started": False, "source_trajectory_rescan": False}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("accepted-index", "selected-episodes", "fixed-index", "fixed-candidates",
                "output-dir", "runtime-code", "audit-script"):
        parser.add_argument("--" + key, required=True, type=Path)
    for key in ("checkpoint", "checkpoint-sha", "model"):
        parser.add_argument("--" + key, required=True)
    a = parser.parse_args()
    print(json.dumps(materialize(accepted_index=a.accepted_index,
        selected_episodes=a.selected_episodes, fixed_index=a.fixed_index,
        fixed_candidates=a.fixed_candidates, output_dir=a.output_dir,
        runtime_code=a.runtime_code, audit_script=a.audit_script, checkpoint=a.checkpoint,
        checkpoint_sha=a.checkpoint_sha, model=a.model)))


if __name__ == "__main__":
    main()
