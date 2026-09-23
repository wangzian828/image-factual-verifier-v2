"""Audit frozen accepted PSD repairs without copying trajectories or hashing media.

The input is the small case index produced at the user-requested repair stop.
Only accepted attempt ledgers referenced by that index are read. The output is
metadata, not a training target package or a teacher-scoring request.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
from typing import Any


def _stat(path: Path) -> dict[str, int]:
    value = path.stat()
    return {"device": value.st_dev, "inode": value.st_ino, "bytes": value.st_size,
            "mtime_ns": value.st_mtime_ns}


def _roles(row: dict[str, Any], model: str, checkpoint_sha: str, checkpoint: str) -> str:
    roles = row.get("model_roles")
    if not isinstance(roles, dict):
        raise ValueError("accepted target has no model roles")
    teacher, student = roles.get("frozen_self_teacher"), roles.get("trainable_student")
    if not isinstance(teacher, dict) or not isinstance(student, dict):
        raise ValueError("accepted target lacks Qwen teacher/student binding")
    for name, role, path_key in (("teacher", teacher, "round_start_checkpoint"),
                                 ("student", student, "initial_checkpoint")):
        if (role.get("provider") != "qwen_local" or role.get("model") != model
                or role.get("checkpoint_manifest_sha256") != checkpoint_sha
                or role.get(path_key) != checkpoint):
            raise ValueError(f"{name} model/checkpoint differs from frozen SFT3")
    hint = roles.get("hint_constructor")
    if not isinstance(hint, dict) or hint.get("provider") != "gemini" or not hint.get("model"):
        raise ValueError("accepted repair lacks real Gemini source")
    record = row.get("hint_record")
    if isinstance(record, dict) and record.get("model") and record.get("model") != hint["model"]:
        raise ValueError("hint record and model roles disagree on Gemini source")
    return hint["model"]


def _strict(row: dict[str, Any]) -> None:
    if row.get("accepted") is not True or row.get("scaffold_only") is not False:
        raise ValueError("non-accepted/scaffold attempt in frozen success")
    if not all(row.get(key) is True for key in
               ("hinted_episode_pass", "hinted_local_pass", "hinted_strict_trace_audit_pass")):
        raise ValueError("accepted target lacks strict episode/local/trace pass")
    verdict = row.get("verification")
    if not isinstance(verdict, dict) or verdict.get("reasons") != []:
        raise ValueError("accepted target has rejection reasons")
    if verdict.get("expected_verdict") != verdict.get("hinted_recorded_verdict"):
        raise ValueError("accepted target differs from private verdict")
    local = row.get("local_verification")
    if not isinstance(local, dict) or local.get("passed") is not True:
        raise ValueError("accepted target lacks local verification")
    checks = local.get("checks")
    if not isinstance(checks, list) or not checks or any(check.get("passed") is not True for check in checks):
        raise ValueError("accepted target lacks complete local checks")
    for key in ("teacher_prompt_ids", "student_prompt_ids", "completion_ids"):
        ids = row.get(key)
        if not isinstance(ids, list) or not ids or any(type(value) is not int for value in ids):
            raise ValueError(f"accepted target has invalid {key}")
    media = row.get("psd_media")
    has_image_token = 248056 in row["teacher_prompt_ids"] or 248056 in row["student_prompt_ids"]
    if has_image_token:
        if not isinstance(media, dict) or not isinstance(media.get("image_paths"), list) or not media["image_paths"]:
            raise ValueError("image-bearing accepted target lacks exact media binding")
    elif media is not None and (not isinstance(media, dict) or not isinstance(media.get("image_paths"), list)):
        raise ValueError("text-only accepted target has malformed media binding")


def audit_frozen_repairs(*, frozen_index: Path, output_dir: Path, run_root: Path,
                         model: str, checkpoint_sha: str, checkpoint: str) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    run_root = run_root.resolve()
    cases: set[str] = set()
    attempts: set[str] = set()
    result: list[dict[str, Any]] = []
    gemini = Counter()
    with frozen_index.open(encoding="utf-8") as source:
        for line in source:
            case = json.loads(line)
            case_id = case.get("case_id")
            if not isinstance(case_id, str) or not case_id or case_id in cases:
                raise ValueError("duplicate or missing frozen case ID")
            cases.add(case_id)
            path = Path(case["attempts_path"]).resolve()
            if not path.is_relative_to(run_root):
                raise ValueError("attempt ledger outside frozen run")
            if _stat(path) != case["attempts_identity"]:
                raise ValueError("attempt ledger stat changed after freeze")
            count = 0
            with gzip.open(path, "rt", encoding="utf-8") as ledger:
                for ordinal, attempt_line in enumerate(ledger):
                    row = json.loads(attempt_line)
                    if row.get("accepted") is not True:
                        continue
                    if row.get("case_id") != case_id:
                        raise ValueError("accepted attempt/case mismatch")
                    _strict(row)
                    source_model = _roles(row, model, checkpoint_sha, checkpoint)
                    attempt_id = row.get("attempt_id")
                    if not isinstance(attempt_id, str) or not attempt_id or attempt_id in attempts:
                        raise ValueError("duplicate or missing accepted attempt ID")
                    attempts.add(attempt_id)
                    gemini[source_model] += 1
                    count += 1
                    result.append({"case_id": case_id, "attempt_id": attempt_id,
                                   "source_path": str(path), "source_row": ordinal,
                                   "gemini_hint_model": source_model,
                                   "completion_tokens": len(row["completion_ids"]),
                                   "prompt_tokens": len(row["student_prompt_ids"]),
                                   "image_count": len((row.get("psd_media") or {}).get("image_paths", [])),
                                   "verification": "strict_complete_pass",
                                   "checkpoint_manifest_sha256": checkpoint_sha})
            if _stat(path) != case["attempts_identity"]:
                raise ValueError("attempt ledger changed during audit")
            if count != case["accepted_count"]:
                raise ValueError("frozen receipt and accepted ledger disagree")
    if not result:
        raise ValueError("no accepted repair targets")
    result.sort(key=lambda row: (row["case_id"], row["source_row"]))
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "accepted-target-index.jsonl").open("x", encoding="utf-8") as target:
        for row in result:
            target.write(json.dumps(row, sort_keys=True) + "\n")
    manifest = {"schema_version": "ifv-frozen-repair-audit-index-v1",
                "status": "strictly_audited_metadata_only_not_train_ready",
                "frozen_index": str(frozen_index.resolve()),
                "frozen_index_identity": _stat(frozen_index),
                "checkpoint_manifest_sha256": checkpoint_sha, "qwen_model": model,
                "qwen_checkpoint": checkpoint,
                "accepted_cases": len(cases), "accepted_targets": len(result),
                "gemini_hint_models": dict(sorted(gemini.items())),
                "target_index": "accepted-target-index.jsonl",
                "no_payload_copy": True, "no_media_hashing": True,
                "formal_training_started": False}
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint-sha", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    print(json.dumps(audit_frozen_repairs(frozen_index=args.frozen_index,
        output_dir=args.output_dir, run_root=args.run_root, model=args.model,
        checkpoint_sha=args.checkpoint_sha, checkpoint=args.checkpoint), ensure_ascii=False))


if __name__ == "__main__":
    main()
