"""Summarize Agent and direct-QA audits with one private-gold metric contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.private_gold_metrics import (
    annotate_private_gold_category,
    private_gold_audit_summary,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            rows.append(row)
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _parse_input(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError(
            "inputs must use NAME=PATH, for example agent=/tmp/audit-results.jsonl"
        )
    return name.strip(), Path(path.strip()).expanduser().resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        dest="inputs",
        action="append",
        required=True,
        type=_parse_input,
        metavar="NAME=JSONL",
        help="One or more completed private-gold audit JSONL files.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    combined: dict[str, Any] = {
        "schema_version": "ifv-private-gold-comparison-v1",
        "category_definition": {
            "correct_point_with_strong_evidence": (
                "verdict_matches_gold=true and quality_bucket=strong"
            ),
            "correct_verdict_insufficient_evidence": (
                "verdict_matches_gold=true and quality_bucket!=strong"
            ),
            "wrong_verdict": "verdict_matches_gold=false",
        },
        "audits": {},
    }
    for name, path in args.inputs:
        if name in combined["audits"]:
            raise SystemExit(f"duplicate audit name: {name}")
        rows = [annotate_private_gold_category(row) for row in _read_jsonl(path)]
        annotated_path = output_dir / f"{name}.annotated.jsonl"
        _write_jsonl(annotated_path, rows)
        summary = private_gold_audit_summary(rows)
        summary.update(
            {
                "name": name,
                "source": str(path),
                "annotated_results": str(annotated_path),
            }
        )
        _write_json(output_dir / f"{name}.summary.json", summary)
        combined["audits"][name] = summary

    combined["output_dir"] = str(output_dir)
    _write_json(output_dir / "comparison-summary.json", combined)
    print(json.dumps(combined, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
