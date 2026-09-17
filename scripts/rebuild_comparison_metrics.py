#!/usr/bin/env python3
"""Recompute the main table from recovered integer counts; never call a judge.

Rows are true real/fake, columns are predicted real/fake/missing_or_invalid.
Only the two task classes are macro-averaged. Abstentions contribute a false
negative to their true class, not a fabricated prediction of the other class.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COUNTS = ROOT / "docs/evaluation-metric-counts-20260917.json"


def binary_metrics(matrix, support=(377, 1150)):
    if len(matrix) != 2 or any(len(row) != 3 for row in matrix):
        raise ValueError("Expected two gold rows and three prediction columns")
    if any(type(n) is not int or n < 0 for row in matrix for n in row):
        raise ValueError("Counts must be nonnegative integers")
    if len(support) != 2 or any(type(n) is not int or n <= 0 for n in support):
        raise ValueError("Both frozen gold classes must be nonempty")
    if tuple(map(sum, matrix)) != tuple(support):
        raise ValueError("Counts do not cover the full frozen class supports")
    precision, recall, f1 = [], [], []
    for i in range(2):
        tp = matrix[i][i]
        fp = matrix[1-i][i]
        fn = support[i] - tp  # Includes invalid/missing, without dropping cases.
        precision.append(tp / (tp + fp) if tp + fp else 0.0)
        recall.append(tp / support[i])
        f1.append(2 * tp / (2 * tp + fp + fn))
    return {
        "bacc": sum(recall) / 2,
        "macro_precision": sum(precision) / 2,
        "macro_f1": sum(f1) / 2,
        "supported_recall": recall[0],
        "refuted_recall": recall[1],
        "precision_by_class": dict(zip(("real", "fake"), precision)),
        "f1_by_class": dict(zip(("real", "fake"), f1)),
        "missing_or_invalid": sum(row[2] for row in matrix),
    }


def recover(document):
    if document["prediction_columns"] != ["real", "fake", "missing_or_invalid"]:
        raise ValueError("Unexpected prediction-column order")
    if document["gold_rows"] != ["real", "fake"]:
        raise ValueError("Unexpected gold-row order")
    support = document["support"]
    seen = set()
    rows = []
    for record in document["models"]:
        if record["id"] in seen:
            raise ValueError("Duplicate experiment ID")
        seen.add(record["id"])
        if record["source"] not in document["sources"]:
            raise ValueError("Missing provenance")
        metrics = binary_metrics(record["matrix"], support)
        for key, previous in record["previous_percent"].items():
            if f"{100 * metrics[key]:.2f}" != previous:
                raise ValueError(f"Existing metric changed: {record['id']} / {key}")
        rows.append({**record, "metrics": metrics})
    return rows


def markdown(rows, group):
    lines = [
        "| 排名 | 模型 | BAcc ↑ | Macro-Precision ↑ | Macro-F1 ↑ | Supported Recall ↑ | Refuted Recall ↑ | SESR ↑ |",
        "|---:|:---|---:|---:|---:|---:|---:|---:|",
    ]
    keys = ("bacc", "macro_precision", "macro_f1", "supported_recall", "refuted_recall")
    selected = sorted((r for r in rows if r["group"] == group),
                      key=lambda r: r["metrics"]["bacc"], reverse=True)
    for rank, row in enumerate(selected, 1):
        values = [f"{100 * row['metrics'][k]:.2f}" for k in keys]
        lines.append("| " + " | ".join([str(rank), row["model"], *values,
                                          row["sesr_existing_percent"]]) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", type=Path, default=DEFAULT_COUNTS)
    args = parser.parse_args()
    rows = recover(json.loads(args.counts.read_text(encoding="utf-8")))
    for group in ("agent", "direct_qa"):
        print(group)
        print(markdown(rows, group))
        print()


if __name__ == "__main__":
    main()
