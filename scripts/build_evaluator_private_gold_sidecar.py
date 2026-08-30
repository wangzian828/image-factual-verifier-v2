"""Materialize complete evaluator-private gold for unified train/test splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.evaluator_private_gold import (
    EVALUATOR_PRIVATE_GOLD_SCHEMA_VERSION,
    build_evaluator_private_gold_records,
    case_alias_rows,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--archive-root",
        type=Path,
        required=True,
        help="Immutable archive containing human-review-candidates.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <dataset-root>/evaluator_private/private-gold-v1.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    archive_root = args.archive_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else dataset_root / "evaluator_private" / "private-gold-v1"
    )
    manifests = {
        split: dataset_root / f"{split}-manifest.jsonl"
        for split in ("test", "train")
    }
    archive_path = archive_root / "human-review-candidates.jsonl"
    missing = [
        str(path)
        for path in (*manifests.values(), archive_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("missing required input(s): " + ", ".join(missing))
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"output exists and is non-empty: {output_dir}; pass --overwrite"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    records, summary = build_evaluator_private_gold_records(
        split_rows={split: _read_jsonl(path) for split, path in manifests.items()},
        archive_rows=_read_jsonl(archive_path),
    )
    aliases = case_alias_rows(records)
    output_rows = output_dir / "private-gold.jsonl"
    output_test_rows = output_dir / "test-private-gold.jsonl"
    output_train_rows = output_dir / "train-private-gold.jsonl"
    output_aliases = output_dir / "case-alias-index.jsonl"
    _write_jsonl(output_rows, records)
    _write_jsonl(
        output_test_rows,
        [row for row in records if row.get("split") == "test"],
    )
    _write_jsonl(
        output_train_rows,
        [row for row in records if row.get("split") == "train"],
    )
    _write_jsonl(output_aliases, aliases)
    summary.update(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset_root": str(dataset_root),
            "archive_root": str(archive_root),
            "inputs": {
                f"{split}_manifest": {
                    "path": str(path),
                    "sha256": _sha256(path),
                }
                for split, path in manifests.items()
            }
            | {
                "archive_candidates": {
                    "path": str(archive_path),
                    "sha256": _sha256(archive_path),
                }
            },
            "artifacts": {
                "private_gold": str(output_rows),
                "test_private_gold": str(output_test_rows),
                "train_private_gold": str(output_train_rows),
                "case_alias_index": str(output_aliases),
            },
        }
    )
    _write_json(output_dir / "summary.json", summary)
    (output_dir / "README.md").write_text(
        "# Evaluator-private gold sidecar\n\n"
        "This directory is evaluator-only. Do not pass it to rollout, SFT export, "
        "RL training, or public benchmark consumers.\n\n"
        "`private-gold.jsonl` contains one complete private target per stable "
        "runtime case ID across both train and test. The split-specific "
        "`test-private-gold.jsonl` and `train-private-gold.jsonl` files are the "
        "default inputs for their respective evaluators. `case-alias-index.jsonl` "
        "contains only aliases that resolve unambiguously. `summary.json` records "
        "source hashes and completeness validation.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
