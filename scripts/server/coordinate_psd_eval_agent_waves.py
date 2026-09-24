"""Gate each frozen Agent wave without waiting for its streaming judge.

The existing immutable evaluation owner is paused only while its direct
``run_cases`` child runs.  This sidecar resumes it between waves after checking
durable Agent ledgers, and refuses to advance a variant after its final retry
when any frozen case is still missing.  It neither dispatches Agent or judge
requests nor edits their existing receipts.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import time

from scripts.server import control_corrected_agent_eval as agent


EXPECTED_RUNNABLE = 1526
SMOKE_COUNT = 4
OWNER_MARKER = "run_psd_combined_cached_full_eval.py\x00execute"


def process(pid: int) -> tuple[str, int, int, str] | None:
    root = Path("/proc") / str(pid)
    try:
        stat = (root / "stat").read_text().split()
        command = (root / "cmdline").read_bytes().decode(errors="replace")
    except (FileNotFoundError, ProcessLookupError):
        return None
    return stat[2], int(stat[3]), int(stat[21]), command


def verified_owner(pid: int, startticks: int) -> str:
    details = process(pid)
    if details is None:
        return "exited"
    state, _, actual_startticks, command = details
    if actual_startticks != startticks:
        raise RuntimeError("evaluation owner identity changed")
    if state == "Z":
        return state
    if OWNER_MARKER not in command:
        raise RuntimeError("evaluation owner command changed")
    return state


def direct_agent_child(owner_pid: int) -> tuple[int, int, Path] | None:
    found: list[tuple[int, int, Path]] = []
    for item in Path("/proc").iterdir():
        if not item.name.isdigit():
            continue
        pid = int(item.name)
        details = process(pid)
        if details is None:
            continue
        state, parent, startticks, command = details
        if parent != owner_pid or state == "Z" or "-m\x00src.eval.run_cases\x00" not in command:
            continue
        arguments = command.split("\x00")
        if "--output-dir" not in arguments:
            raise RuntimeError("Agent child lacks output-dir binding")
        output = Path(arguments[arguments.index("--output-dir") + 1]).resolve()
        found.append((pid, startticks, output))
    if len(found) > 1:
        raise RuntimeError("more than one direct Agent child")
    return found[0] if found else None


def wave_decision(
    group: Path,
    wave: str,
    expected: int = EXPECTED_RUNNABLE,
    *,
    expected_ids: set[str] | None = None,
    smoke_ids: set[str] | None = None,
) -> dict[str, object]:
    selected = agent.successful(group)
    if expected_ids is not None and (
        len(expected_ids) != expected or not set(selected) <= expected_ids
    ):
        raise ValueError("Agent success ledger differs from frozen cohort")
    count = len(selected)
    if wave in {"smoke-0", "smoke-1"}:
        target = SMOKE_COUNT
        last = wave == "smoke-1"
        complete = smoke_ids <= set(selected) if smoke_ids is not None else count >= target
    elif wave in {f"attempt-{index}" for index in range(4)}:
        target = expected
        last = wave == "attempt-3"
        complete = count == target
    else:
        raise ValueError(f"unrecognized Agent wave: {wave}")
    if count > expected:
        raise ValueError("more successful Agent cases than frozen cohort")
    return {
        "group": str(group),
        "wave": wave,
        "agent_success": count,
        "target": target,
        "remaining": max(0, target - count),
        "advance": complete or not last,
        "reason": "complete" if complete else "retry_wave" if not last else "agent_incomplete",
        "judge_not_a_stage_barrier": True,
    }


def atomic_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def coordinate(owner_pid: int, startticks: int, full_output: Path, state_path: Path) -> None:
    if verified_owner(owner_pid, startticks) not in {"T", "t"}:
        raise RuntimeError("evaluation owner must be paused before handoff")
    state_path.parent.mkdir(parents=True, exist_ok=False)
    benchmark_ids = {str(row["case_id"]) for row in agent.rows(agent.BENCHMARK)}
    if len(benchmark_ids) != EXPECTED_RUNNABLE or not set(agent.SMOKE_CASES) <= benchmark_ids:
        raise ValueError("frozen benchmark or smoke cohort changed")
    atomic_json(state_path, {"phase": "watching", "owner_pid": owner_pid, "owner_startticks": startticks})
    seen: set[tuple[int, int]] = set()
    pending: tuple[int, int, Path] | None = None
    while True:
        owner_state = verified_owner(owner_pid, startticks)
        if owner_state in {"exited", "Z"}:
            atomic_json(state_path, {"phase": "owner_exited", "owner_pid": owner_pid})
            return
        child = direct_agent_child(owner_pid)
        if child is not None and (child[0], child[1]) not in seen:
            if owner_state not in {"T", "t"}:
                os.kill(owner_pid, signal.SIGSTOP)
                if verified_owner(owner_pid, startticks) not in {"T", "t"}:
                    raise RuntimeError("owner could not be paused for Agent wave")
            pending = child
            seen.add((child[0], child[1]))
            atomic_json(state_path, {
                "phase": "agent_wave_running", "owner_pid": owner_pid,
                "child_pid": child[0], "child_startticks": child[1],
                "output": str(child[2]),
            })
        if pending is not None:
            child_state = process(pending[0])
            if child_state is None or child_state[0] == "Z":
                if child_state is not None and child_state[2] != pending[1]:
                    raise RuntimeError("Agent child PID was reused")
                output = pending[2]
                if output.parent != full_output and not (
                    output.parent.parent == full_output.parent
                    and output.parent.name in {
                        full_output.name + "-" + variant
                        for variant in ("no-web-search", "no-image-retrieval", "no-evidence-inspection")
                    }
                ):
                    raise RuntimeError("Agent child wrote outside the frozen evaluation outputs")
                if not (output / "summary.json").is_file() or not (output / "run_results.jsonl").is_file():
                    raise RuntimeError("Agent child exited without durable wave receipts")
                decision = wave_decision(
                    output.parent, output.name,
                    expected_ids=benchmark_ids, smoke_ids=set(agent.SMOKE_CASES),
                )
                atomic_json(state_path, {"phase": "wave_checked", "owner_pid": owner_pid, **decision})
                if not decision["advance"]:
                    return
                if verified_owner(owner_pid, startticks) not in {"T", "t"}:
                    raise RuntimeError("owner escaped the Agent wave gate")
                os.kill(owner_pid, signal.SIGCONT)
                pending = None
        time.sleep(0.1 if pending is None else 2.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner-pid", type=int, required=True)
    parser.add_argument("--owner-startticks", type=int, required=True)
    parser.add_argument("--full-output", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    args = parser.parse_args()
    coordinate(args.owner_pid, args.owner_startticks, args.full_output.resolve(), args.state_dir / "state.json")


if __name__ == "__main__":
    main()
