"""Retired H20 v2 relocation entry point plus reporting helpers.

The former executable copied legacy ms-swift rows unchanged and emitted a
processor-v2 report. That path cannot prove causal observation IDs or loss
masks, so executing it now fails closed. Historical reporting scripts still
import the small read-only helpers below.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]


def sha(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rows(path: str | Path) -> list[Any]:
    # JSONL boundaries are LF, not every Unicode separator accepted by
    # ``splitlines()``.
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write(path: str | Path, value: Any) -> None:
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def dist(values: list[int | float]) -> dict[str, int | float]:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0}
    return {
        "count": len(ordered),
        "sum": sum(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
        **{
            f"p{quantile}": ordered[
                max(0, math.ceil(len(ordered) * quantile / 100) - 1)
            ]
            for quantile in (25, 50, 75, 90, 95, 99)
        },
    }


def main() -> None:
    raise SystemExit(
        "prepare_full_data.py is retired because it preserves the broken v2 "
        "causal contract. Use `python -m ifv_training repair-policy-contract`, "
        "then verify_ms_swift_agent_dataset.py and verify_sft_data_contract.py."
    )


if __name__ == "__main__":
    main()
