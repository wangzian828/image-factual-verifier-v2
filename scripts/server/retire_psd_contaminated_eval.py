"""Retire only the verified, paused v3 evaluation tree without service rollback."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import time

ROOT = Path("/volume/ybo/wza")
OUT = ROOT / "training-artifacts/psd-combined-contamination-handoff-20260924-v1"
TARGETS = (
    ("sidecar", 2601223, 719105469, "coordinate_psd_eval_agent_waves.py", "psd-combined-eval-wave-gate-20260924-v4"),
    ("agent_child", 2598192, 718965535, "src.eval.run_cases", "-no-web-search/attempt-0"),
    ("full_judge", 2573275, 717459879, "stream_agent_judges.py", "full1526-selfextract-20260924-v3/judge-gemini37-stream-v1"),
    ("no_web_judge", 2598025, 718942282, "stream_agent_judges.py", "-no-web-search/judge-gemini37-stream-v1"),
    ("parent", 2567752, 717428752, "run_psd_combined_cached_full_eval.py", "execute"),
)


def snapshot(pid: int) -> dict:
    proc = Path("/proc") / str(pid)
    stat = (proc / "stat").read_text().rsplit(") ", 1)[1].split()
    return {
        "pid": pid,
        "state": stat[0],
        "ppid": int(stat[1]),
        "startticks": int(stat[19]),
        "cmdline": (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace"),
    }


def verified_targets() -> dict[str, dict]:
    found = {}
    for name, pid, ticks, *markers in TARGETS:
        row = snapshot(pid)
        if row["startticks"] != ticks or not all(marker in row["cmdline"] for marker in markers):
            raise RuntimeError(f"{name} PID/cmdline/startticks mismatch")
        if name != "sidecar" and row["state"] not in {"T", "t"}:
            raise RuntimeError(f"{name} is not paused: {row['state']}")
        found[name] = row
    if any(found[name]["ppid"] != found["parent"]["pid"] for name in ("agent_child", "full_judge", "no_web_judge")):
        raise RuntimeError("parent-child ownership mismatch")
    return found


def gone_or_zombie(pid: int, ticks: int) -> bool:
    try:
        row = snapshot(pid)
    except (FileNotFoundError, ProcessLookupError):
        return True
    return row["startticks"] != ticks or row["state"] == "Z"


def wait_exit(pid: int, ticks: int, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if gone_or_zombie(pid, ticks):
            return True
        time.sleep(0.1)
    return gone_or_zombie(pid, ticks)


def save(name: str, value: dict) -> None:
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    before = verified_targets()
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    save("before.json", {
        "reason": "cross-image perception cache reuse in v3 trajectories",
        "targets": before,
        "old_traces_retained": True,
        "old_judge_receipts_retained": True,
        "service_rollback_prohibited": True,
    })
    sidecar = before["sidecar"]
    os.kill(sidecar["pid"], signal.SIGTERM)
    if not wait_exit(sidecar["pid"], sidecar["startticks"], 5):
        raise RuntimeError("sidecar did not exit; do not touch evaluation owner")
    save("sidecar-stopped.json", {"pid": sidecar["pid"], "time": time.time()})
    # SIGKILL targets are stopped, individually verified processes.  In
    # particular, never let the old parent run its SFT3 rollback finally.
    for name in ("agent_child", "full_judge", "no_web_judge", "parent"):
        row = before[name]
        current = snapshot(row["pid"])
        if current["startticks"] != row["startticks"] or current["state"] not in {"T", "t"}:
            raise RuntimeError(f"{name} changed before retirement")
        os.kill(row["pid"], signal.SIGKILL)
        if not wait_exit(row["pid"], row["startticks"], 5):
            raise RuntimeError(f"{name} did not exit")
        save(name + "-stopped.json", {"pid": row["pid"], "time": time.time()})
    save("result.json", {
        "passed": True, "old_owner_exited": True,
        "old_outputs_preserved": True,
        "service_not_modified_by_retirement": True,
        "next": "verify new merged service and launch cachefixed v4 Agent owner",
    })


if __name__ == "__main__":
    main()
