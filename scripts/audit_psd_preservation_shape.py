"""Stream a preservation JSONL and report target-shape inflation diagnostics."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    rows = 0
    lengths: list[int] = []
    stages: Counter[str] = Counter()
    actions: Counter[str] = Counter()
    examples: Counter[str] = Counter()
    sequences: Counter[str] = Counter()
    rejected = 0
    empty_completion = 0
    samples = []
    case_rows: Counter[str] = Counter()
    with args.path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            case_rows[str(row.get("case_id") or "")] += 1
            steps = row.get("preservation_steps") or []
            rows += 1
            lengths.append(len(steps))
            sequence = []
            for step in steps:
                stage = str(step.get("stage") or "")
                action = str(step.get("action_type") or "")
                example = str(step.get("example_type") or "")
                stages[stage] += 1
                actions[action] += 1
                examples[example] += 1
                sequence.append(f"{stage}:{action}")
                rejected += bool(step.get("protocol_rejected"))
                empty_completion += not bool(step.get("completion_ids"))
            sequences[" -> ".join(sequence)] += 1
            if len(samples) < 3:
                samples.append([{key: step.get(key) for key in (
                    "step_id", "source_step_index", "stage", "action_type",
                    "example_type", "protocol_rejected")}
                    for step in steps])
    result = {
        "rows": rows,
        "steps": sum(lengths),
        "unique_cases": len(case_rows),
        "rows_per_case": dict(Counter(case_rows.values())),
        "steps_per_row": {
            "min": min(lengths, default=0),
            "median": statistics.median(lengths) if lengths else 0,
            "max": max(lengths, default=0),
        },
        "stages": dict(stages),
        "actions": dict(actions),
        "examples": dict(examples),
        "protocol_rejected": rejected,
        "empty_completion": empty_completion,
        "common_sequences": sequences.most_common(20),
        "samples": samples,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
