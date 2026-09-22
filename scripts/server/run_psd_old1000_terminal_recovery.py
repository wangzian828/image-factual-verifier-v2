"""Append-only recovery wave for old-1000 PSD cases with terminal failures.

The normal repair owner deliberately treats exhausted cases as terminal under
the original budget. This controller does not reopen or rewrite those
receipts. After the normal owner exits, it runs one separately accounted
recovery wave using the frozen Gemini slate cache and writes all new attempts
under ``recoveries/terminal-case-recovery-v1``.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - server is Linux
    fcntl = None


ROOT = Path("/volume/ybo/wza")
SOURCE_OUTPUT = ROOT / "runs/psd-production1000x4-20260918-v1/processing-lightweight-v1/repair-search-v3"
os.environ.setdefault("IFV_OLD1000_REPAIR_OUTPUT", str(SOURCE_OUTPUT))

from scripts.server import run_psd_old1000_repair as source  # noqa: E402
from scripts.run_psd_repair_driver import _parser as repair_parser, _run as repair_run  # noqa: E402
from ifv_training.psd_infrastructure_retry import InfrastructureRetriesExhausted  # noqa: E402


RECOVERY = SOURCE_OUTPUT / "recoveries/terminal-case-recovery-v1"
TARGET_STATUSES = {"attempt_budget_exhausted", "infrastructure_budget_exhausted"}


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def receipt_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recovery_candidates() -> list[dict[str, Any]]:
    """Return only terminal, non-accepted cases from the frozen source run."""
    entries = {row["case_id"]: row for row in source.selected_index()}
    selected = []
    for path in sorted((SOURCE_OUTPUT / "case-receipts").glob("*.json")):
        row = load(path)
        if row.get("status") not in TARGET_STATUSES or row.get("accepted_count", 0):
            continue
        case_id = row.get("case_id")
        if case_id not in entries:
            raise ValueError(f"terminal receipt is not in frozen selection: {case_id}")
        selected.append({
            "case_id": case_id,
            "status": row["status"],
            "original_receipt": str(path),
            "original_receipt_sha256": receipt_digest(path),
        })
    if not selected:
        raise ValueError("no terminal old-1000 cases eligible for recovery")
    return selected


def bind_or_validate() -> list[dict[str, Any]]:
    rows = recovery_candidates()
    path = RECOVERY / "binding.json"
    expected = {
        "schema_version": "ifv-old1000-terminal-recovery-v1",
        "source_output": str(SOURCE_OUTPUT),
        "source_commit": "9eefb34",
        "original_budget_unchanged": True,
        "additional_recovery_budget": "one fresh six-round search per selected case",
        "cases": rows,
    }
    if path.exists():
        if load(path) != expected:
            raise ValueError("terminal recovery binding changed")
    else:
        save(path, expected)
    return rows


def owner_active() -> bool:
    result = subprocess.run(["ps", "-eo", "pid,args"], text=True,
                            capture_output=True, check=True)
    marker = "run_psd_old1000_repair.py"
    return any(marker in line and "terminal_recovery" not in line
               for line in result.stdout.splitlines())


def recovery_args(entry: dict[str, Any], benchmark: dict[str, dict[str, Any]],
                  gold: dict[str, dict[str, Any]]) -> list[str]:
    args = source.case_cli(entry, benchmark, gold)
    output_index = args.index("--output-dir") + 1
    original = Path(args[output_index])
    target = RECOVERY / "repairs" / original.name
    args[output_index] = str(target)
    while "--resume" in args:
        args.remove("--resume")
    if (target / "run-inputs.json").exists():
        args.append("--resume")
    return args


def write_state(value: dict[str, Any]) -> None:
    save(RECOVERY / "state.json", value)


async def run_wave(rows: list[dict[str, Any]], concurrency: int) -> dict[str, Any]:
    if not 1 <= concurrency <= 16:
        raise ValueError("recovery concurrency must be 1..16")
    if fcntl is None:
        raise RuntimeError("terminal recovery requires Linux process locks")
    RECOVERY.mkdir(parents=True, exist_ok=True)
    with (RECOVERY / "owner.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source.serving_ready()
        source.no_competing_eval()
        benchmark = source.by_case(source.RUN / "selection/runtime-release/runtime_input/cases.jsonl")
        gold = source.by_case(source.GOLD)
        entries = {row["case_id"]: row for row in source.selected_index()}
        queue = asyncio.Queue()
        for row in rows:
            queue.put_nowait(row)
        counts = Counter()
        write_state({"phase": "recovering", "selected": len(rows),
                     "concurrency": concurrency, "original_budget_unchanged": True,
                     "completed": 0, "errors": 0})

        async def one() -> None:
            while True:
                try:
                    row = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                case_id = row["case_id"]
                try:
                    args = repair_parser().parse_args(recovery_args(
                        entries[case_id], benchmark, gold))
                    result = await repair_run(args)
                    status = result.get("status")
                    if status not in source.TERMINAL:
                        raise RuntimeError("recovery case did not reach a terminal manifest")
                    save(RECOVERY / "case-receipts" / f"{case_id}.json", {
                        "case_id": case_id, "status": status,
                        "complete_reruns": result.get("complete_reruns"),
                        "accepted_count": result.get("accepted_count"),
                        "source_receipt_sha256": row["original_receipt_sha256"],
                        "manifest": str(args.output_dir / "manifest.json"),
                        "completed_unix": time.time(),
                    })
                    counts[status] += 1
                except InfrastructureRetriesExhausted as error:
                    save(RECOVERY / "case-receipts" / f"{case_id}.json", {
                        "case_id": case_id, "status": "infrastructure_budget_exhausted",
                        "reason": str(error), "source_receipt_sha256": row["original_receipt_sha256"],
                        "manifest": None, "completed_unix": time.time(),
                    })
                    counts["infrastructure_budget_exhausted"] += 1
                except Exception as error:
                    save(RECOVERY / "case-errors" / f"{case_id}.json", {
                        "case_id": case_id, "error_type": type(error).__name__,
                        "message": str(error)[:500], "source_receipt_sha256": row["original_receipt_sha256"],
                        "at_unix": time.time(),
                    })
                    counts["case_error"] += 1
                finally:
                    queue.task_done()
                    write_state({"phase": "recovering", "selected": len(rows),
                                 "concurrency": concurrency, "original_budget_unchanged": True,
                                 "completed": sum(counts[s] for s in source.TERMINAL),
                                 "errors": counts["case_error"],
                                 "statuses": dict(counts)})

        await asyncio.gather(*(one() for _ in range(concurrency)))
        result = {"phase": "complete", "selected": len(rows),
                  "original_budget_unchanged": True, "statuses": dict(counts)}
        write_state(result)
        return result


async def main(args: argparse.Namespace) -> None:
    while owner_active():
        write_state({"phase": "waiting_for_original_owner", "original_budget_unchanged": True,
                     "updated_unix": time.time()})
        await asyncio.sleep(args.poll_seconds)
    rows = bind_or_validate()
    await run_wave(rows, args.concurrency)


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--poll-seconds", type=int, default=30)
    asyncio.run(main(parser.parse_args()))


if __name__ == "__main__":
    cli()
