#!/usr/bin/env python3
"""Build a direct-QA manifest from generated test-set images."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)[:150]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata_paths(root: Path) -> list[Path]:
    direct = root / "metadata.jsonl"
    if direct.is_file():
        return [direct]
    paths = sorted(root.glob("test-set-part-*/metadata.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no metadata.jsonl found below {root}")
    return paths


def _result_paths(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("results-*.jsonl") if path.is_file())


def _image_index(generated_root: Path) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    for result_path in _result_paths(generated_root):
        for row in _read_jsonl(result_path):
            case_id = str(row.get("case_id") or "").strip()
            image_name = str(row.get("image") or "").strip()
            if not case_id or row.get("status") != "ok" or not image_name:
                continue
            image_path = result_path.parent / image_name
            if not image_path.is_file():
                continue
            previous = indexed.get(case_id)
            if previous is not None and previous.resolve() != image_path.resolve():
                raise ValueError(f"multiple generated images for case_id: {case_id}")
            indexed[case_id] = image_path.resolve()
    if indexed:
        return indexed
    for image_path in generated_root.rglob("*"):
        if not image_path.is_file() or image_path.suffix.lower() not in {
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
        }:
            continue
        indexed.setdefault(image_path.stem, image_path.resolve())
    return indexed


def _case_id(row: Mapping[str, Any]) -> str:
    value = str(
        row.get("case_id")
        or row.get("unified_case_id")
        or row.get("candidate_id")
        or ""
    ).strip()
    if not value:
        raise ValueError("metadata row lacks case_id")
    return value


def _copy_metadata_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "archive_source_version_id",
        "candidate_id",
        "factual_status",
        "gold_verdict",
        "production_class",
        "construction_subroute",
        "target_route",
        "target_subtype",
        "target_capability_cell",
        "target_claim",
        "primary_claim",
        "claim_atom",
        "decisive_visual_atom",
        "automatic_qa",
        "evidence",
        "source_url",
        "generation_prompt_id",
        "assignment_id",
    )
    return {field: row[field] for field in fields if field in row}


def prepare_inputs(
    *,
    package_root: Path,
    generated_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    package_root = package_root.expanduser().resolve()
    generated_root = generated_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not package_root.is_dir():
        raise FileNotFoundError(f"test package does not exist: {package_root}")
    if not generated_root.is_dir():
        raise FileNotFoundError(f"generated image root does not exist: {generated_root}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_rows: list[dict[str, Any]] = []
    for path in _metadata_paths(package_root):
        metadata_rows.extend(_read_jsonl(path))
    by_case: dict[str, dict[str, Any]] = {}
    for row in metadata_rows:
        case_id = _case_id(row)
        if case_id in by_case:
            raise ValueError(f"duplicate metadata case_id: {case_id}")
        by_case[case_id] = row

    images = _image_index(generated_root)
    manifest_rows: list[dict[str, Any]] = []
    gold_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for case_id in sorted(by_case):
        image_path = images.get(case_id) or images.get(safe_name(case_id))
        if image_path is None:
            missing.append(case_id)
            continue
        destination = output_dir / "images" / f"{safe_name(case_id)}{image_path.suffix.lower()}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and _sha256(destination) != _sha256(image_path):
            raise ValueError(f"image destination collision: {destination.name}")
        if not destination.exists():
            destination.write_bytes(image_path.read_bytes())
        row = by_case[case_id]
        fields = _copy_metadata_fields(row)
        manifest_rows.append(
            {
                "unified_case_id": case_id,
                "split": "test",
                "unified_image_path": destination.relative_to(output_dir).as_posix(),
                "image_sha256": _sha256(destination),
                **fields,
            }
        )
        gold_rows.append({"case_id": case_id, **fields})

    manifest_path = output_dir / "manifest.jsonl"
    gold_path = output_dir / "private-gold.jsonl"
    _write_jsonl(manifest_path, manifest_rows)
    _write_jsonl(gold_path, gold_rows)
    summary = {
        "schema_version": "ifv-direct-qa-inputs-v1",
        "package_root": str(package_root),
        "generated_root": str(generated_root),
        "output_dir": str(output_dir),
        "metadata_count": len(metadata_rows),
        "image_count": len(images),
        "manifest_count": len(manifest_rows),
        "missing_image_count": len(missing),
        "missing_case_ids": missing,
        "manifest": str(manifest_path),
        "private_gold": str(gold_path),
        "training_prohibited": True,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", required=True, type=Path)
    parser.add_argument("--generated-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = prepare_inputs(
        package_root=args.package_root,
        generated_root=args.generated_root,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
