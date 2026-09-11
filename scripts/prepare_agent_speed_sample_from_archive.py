#!/usr/bin/env python3
"""Materialize a small, training-prohibited Agent speed sample from one archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.prepare_agent_test_release import (
    _case_id,
    _select_balanced,
    prepare_agent_test_release,
)
from src.orchestrator.source_access import benchmark_source_access_policy


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl_from_member(archive: tarfile.TarFile, name: str) -> list[dict]:
    member = archive.getmember(name)
    if not member.isfile():
        raise ValueError(f"archive member is not a regular file: {name}")
    handle = archive.extractfile(member)
    if handle is None:
        raise FileNotFoundError(name)
    return [json.loads(line) for line in handle.read().decode("utf-8").split("\n") if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--staging-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--selection-seed", default="ifv-agent-speed-v1")
    args = parser.parse_args()

    archive_path = Path(args.archive).resolve()
    staging = Path(args.staging_root).resolve()
    output = Path(args.output_dir).resolve()
    if _sha256(archive_path) != args.archive_sha256:
        raise ValueError("archive SHA-256 mismatch")
    if staging.exists() or output.exists():
        raise FileExistsError("staging-root and output-dir must both be new")

    with tarfile.open(archive_path, "r:gz") as archive:
        rows = _jsonl_from_member(archive, "./test-manifest.jsonl")
        gold = _jsonl_from_member(
            archive, "./evaluator_private/private-gold-v1/test-private-gold.jsonl"
        )
        selected, strata = _select_balanced(
            rows,
            limit=args.limit,
            fields=("construction_subroute",),
            seed=args.selection_seed,
        )
        selected_ids = {_case_id(row) for row in selected}
        selected_gold = [row for row in gold if str(row.get("case_id") or "").strip() in selected_ids]
        if len(selected_gold) != len(selected):
            raise ValueError("selected manifest/private-gold mismatch")

        for row in selected:
            relative = Path(str(row.get("unified_image_path") or row.get("local_image_path") or ""))
            if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                raise ValueError(f"unsafe image path for {_case_id(row)}")
            member = archive.getmember("./" + relative.as_posix())
            if not member.isfile() or member.issym() or member.islnk():
                raise ValueError(f"unsafe image member: {member.name}")
            source = archive.extractfile(member)
            if source is None:
                raise FileNotFoundError(member.name)
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("wb") as target:
                shutil.copyfileobj(source, target)

    manifest = staging / "test-manifest.jsonl"
    gold_path = staging / "evaluator_private/private-gold-v1/test-private-gold.jsonl"
    _write_jsonl(manifest, selected)
    _write_jsonl(gold_path, selected_gold)
    policy = benchmark_source_access_policy(
        (str(row.get("source_url") or "") for row in rows),
        policy_id="factcheck-test-1682-speed-sample",
    )
    policy_path = staging / "source-access-policy.json"
    policy_path.write_text(json.dumps(policy.to_dict(), indent=2) + "\n", encoding="utf-8")

    result = prepare_agent_test_release(
        dataset_root=staging,
        test_manifest=manifest,
        private_gold_sidecar=gold_path,
        output_dir=output,
        limit=len(selected),
        balanced_by=("construction_subroute",),
        selection_seed=args.selection_seed,
        source_access_policy=policy_path,
    )
    print(json.dumps({
        "archive_sha256": args.archive_sha256,
        "archive_rows": len(rows),
        "selected_rows": len(selected),
        "strata": strata,
        "runtime_cases_sha256": result["runtime_cases_sha256"],
        "benchmark": result["benchmark"],
        "training_prohibited": True,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
