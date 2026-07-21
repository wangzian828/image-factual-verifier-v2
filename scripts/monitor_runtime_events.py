"""Follow the durable per-case runtime event stream during a canary."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterable, Mapping


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print concise stage, token, tool, and error progress for one case."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    return parser.parse_args()


def _latest_events_path(run_dir: Path, case_id: str) -> Path | None:
    root = run_dir / "traces" / "runtime" / case_id
    candidates = list(root.glob("*/events.jsonl"))
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def _complete_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            events.append(value)
    return events


def summarize_event(event: Mapping[str, Any]) -> str | None:
    event_type = str(event.get("event_type", ""))
    payload = event.get("payload")
    payload = payload if isinstance(payload, Mapping) else {}
    sequence = int(event.get("sequence", 0) or 0)
    prefix = f"[{sequence:04d}]"
    if event_type == "case_attempt_started":
        return f"{prefix} case started attempt={payload.get('attempt_id', '')}"
    if event_type == "context_request_started":
        return (
            f"{prefix} request started id={payload.get('request_id', '')} "
            f"stage={payload.get('stage', '')} "
            f"input_est={payload.get('explicit_input_tokens_estimate', 0)} "
            f"max_output={payload.get('max_output_tokens', '')}"
        )
    if event_type == "context_request_completed":
        error = str(payload.get("error", "")).strip()
        return (
            f"{prefix} request {payload.get('status', '')} "
            f"id={payload.get('request_id', '')} "
            f"input={payload.get('provider_input_tokens')} "
            f"output={payload.get('provider_output_tokens')}"
            + (f" error={error}" if error else "")
        )
    if event_type == "tool_result_archived":
        metadata = payload.get("metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        return (
            f"{prefix} tool action={payload.get('action_index', '')} "
            f"name={payload.get('tool_name', '')} "
            f"success={bool(metadata.get('tool_success', False))}"
        )
    if event_type == "snapshot_committed":
        snapshot_type = str(payload.get("snapshot_type", ""))
        if snapshot_type in {"engineering_error", "final_state"}:
            return f"{prefix} snapshot={snapshot_type}"
    return None


def _new_summaries(events: Iterable[Mapping[str, Any]], after: int) -> tuple[int, list[str]]:
    lines: list[str] = []
    latest = after
    for event in events:
        sequence = int(event.get("sequence", 0) or 0)
        if sequence <= after:
            continue
        latest = max(latest, sequence)
        summary = summarize_event(event)
        if summary:
            lines.append(summary)
    return latest, lines


def main() -> int:
    args = _parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    seen = 0
    while True:
        path = _latest_events_path(run_dir, args.case_id)
        if path is not None:
            seen, lines = _new_summaries(_complete_events(path), seen)
            for line in lines:
                print(line, flush=True)
        if not args.follow:
            return 0
        time.sleep(max(0.2, float(args.poll_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())
