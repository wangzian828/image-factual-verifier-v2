"""Bounded owner for resumable Gemini-only PSD slate precomputation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, separators=(",", ":"))
        stream.write("\n")
    os.replace(temporary, path)


def alive(pid: int) -> bool:
    path = Path("/proc") / str(pid) / "stat"
    if not path.exists():
        return False
    try:
        return path.read_text(encoding="utf-8").split(") ", 1)[1][0] != "Z"
    except (FileNotFoundError, IndexError):
        return False


def retry_delay_seconds(
    attempt: int,
    *,
    base_seconds: float,
    max_seconds: float,
) -> float:
    if attempt < 1:
        raise ValueError("retry attempt must be positive")
    if base_seconds < 0 or max_seconds <= 0:
        raise ValueError("invalid retry delay bounds")
    return min(max_seconds, base_seconds * (2 ** (attempt - 1)))


def run(args: argparse.Namespace) -> dict[str, Any]:
    state_path = args.output / "state.json"
    owner_state = args.owner_state
    if args.adopt_pid:
        _save(owner_state, {"phase": "waiting_adopted_worker", "pid": args.adopt_pid})
        while alive(args.adopt_pid):
            time.sleep(5)
    previous_completed = -1
    no_progress = 0
    for attempt in range(1, args.max_passes + 1):
        current = _load(state_path) if state_path.exists() else {}
        if current.get("phase") == "complete":
            result = {"phase": "complete", "attempt": attempt - 1, **current}
            _save(owner_state, result)
            return result
        before = int(current.get("completed_cases", 0))
        _save(owner_state, {
            "phase": "running_pass", "attempt": attempt,
            "completed_before": before, "max_passes": args.max_passes,
        })
        with args.log.open("ab") as log:
            result = subprocess.run(args.command, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT)
        current = _load(state_path)
        after = int(current.get("completed_cases", 0))
        if result.returncode != 0:
            raise RuntimeError("PSD slate precompute worker exited nonzero")
        if current.get("phase") == "complete":
            final = {"phase": "complete", "attempt": attempt, **current}
            _save(owner_state, final)
            return final
        no_progress = no_progress + 1 if after <= max(before, previous_completed) else 0
        previous_completed = after
        _save(owner_state, {
            "phase": "pass_incomplete", "attempt": attempt,
            "completed_cases": after,
            "failed_cases": current.get("failed_cases", 0),
            "no_progress_passes": no_progress,
        })
        if no_progress >= args.max_no_progress_passes:
            break
        time.sleep(
            retry_delay_seconds(
                attempt,
                base_seconds=args.retry_base_seconds,
                max_seconds=args.retry_max_seconds,
            )
        )
    final = _load(state_path)
    result = {**final, "phase": "bounded_incomplete"}
    _save(owner_state, result)
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--owner-state", type=Path, required=True)
    result.add_argument("--log", type=Path, required=True)
    result.add_argument("--adopt-pid", type=int)
    result.add_argument("--max-passes", type=int, default=3)
    result.add_argument("--max-no-progress-passes", type=int, default=2)
    result.add_argument("--retry-base-seconds", type=float, default=5.0)
    result.add_argument("--retry-max-seconds", type=float, default=20.0)
    result.add_argument("command", nargs=argparse.REMAINDER)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if not args.command:
        raise ValueError("worker command is required after --")
    if not 1 <= args.max_passes <= 10:
        raise ValueError("max passes must be between 1 and 10")
    if not 1 <= args.max_no_progress_passes <= args.max_passes:
        raise ValueError("invalid no-progress pass budget")
    if args.retry_base_seconds < 0 or args.retry_max_seconds <= 0:
        raise ValueError("invalid retry delay bounds")
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
