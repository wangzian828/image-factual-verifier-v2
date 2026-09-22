"""Charge and reconcile Qwen attempts interrupted by the owned cache-off handoff.

Never resets the original three-attempt limit. A completed result is left
untouched; every changed bound ledger has its exact original saved beside a
small receipt. This operation is independent of checker/gold outcomes.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

from ifv_training.psd_repair_storage import load_bound, save_bound


ROOT = Path("/volume/ybo/wza")
OUTPUT = (ROOT / "runs/psd-production1000x4-20260918-v1"
          / "processing-lightweight-v1/repair-search-v1")
SWITCH = ROOT / "training-artifacts/psd-old1000-cache-off-recovery-run-20260922-v161/state.json"
RECOVERY = OUTPUT / "recoveries/cache-off-interrupted-v1"
EXPECTED_INTERRUPTED = 8


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def alive(pid: int) -> bool:
    try:
        return (Path("/proc") / str(pid) / "stat").read_text().split(") ", 1)[1][0] != "Z"
    except (FileNotFoundError, ProcessLookupError):
        return False


def reconcile_marker(marker: Path) -> dict:
    original = marker.read_bytes()
    raw = json.loads(original)
    identity = raw["identity"]
    state = load_bound(marker, identity=identity)
    attempts = state.get("attempts")
    if (not isinstance(attempts, list) or not attempts
            or attempts[-1].get("status") != "interrupted"
            or attempts[-1].get("index") != len(attempts)
            or len(attempts) > identity.get("max_attempts", 0)):
        raise ValueError("unexpected interrupted Qwen retry ledger")
    if (marker.parent / "result.json").exists():
        raise RuntimeError("cannot reconcile a completed Qwen retry result")
    latest = attempts[-1]
    attempt_dir = marker.parent / f"attempt-{len(attempts):03d}"
    if Path(latest["directory"]).resolve() != attempt_dir.resolve():
        raise ValueError("interrupted Qwen attempt directory changed")
    if any(attempt_dir.rglob("result.json")):
        raise RuntimeError("interrupted Qwen attempt contains a completed result")
    relative = marker.relative_to(OUTPUT)
    backup = RECOVERY / relative
    before_sha = hashlib.sha256(original).hexdigest()
    if backup.exists():
        if backup.read_bytes() != original:
            raise ValueError("interrupted Qwen ledger backup differs")
    else:
        backup.parent.mkdir(parents=True, exist_ok=True)
        backup.write_bytes(original)
    latest.update(status="infrastructure_failed",
                  reason="controlled_cache_off_handoff_sigint",
                  legacy_status="interrupted",
                  migration_version="ifv-old1000-cache-off-interruption-v1")
    save_bound(marker, identity=identity, payload=state)
    return {"marker": str(relative), "backup": str(backup.relative_to(OUTPUT)),
            "attempts_charged": len(attempts), "before_sha256": before_sha,
            "after_sha256": hashlib.sha256(marker.read_bytes()).hexdigest()}


def main() -> None:
    if load(SWITCH).get("phase") != "complete_cache_off":
        raise RuntimeError("cache-off service transition is not complete")
    latest = load(OUTPUT / "latest-process.json")
    if alive(int(latest["pid"])):
        raise RuntimeError("old-1000 repair owner is still active")
    if (RECOVERY / "receipt.json").exists():
        raise RuntimeError("interruption reconciliation already recorded")
    pattern = "repairs/*/slate-rounds/*/infrastructure-attempts/retry-state.json"
    pending = []
    for marker in OUTPUT.glob(pattern):
        raw = load(marker)
        if raw["payload"]["attempts"][-1]["status"] == "interrupted":
            pending.append(marker)
    if len(pending) != EXPECTED_INTERRUPTED:
        raise ValueError(f"expected {EXPECTED_INTERRUPTED} interrupted attempts, found {len(pending)}")
    records = [reconcile_marker(marker) for marker in sorted(pending)]
    receipt = {"schema_version": "ifv-old1000-cache-off-interruption-v1",
               "count": len(records), "budget_reset": False,
               "source_owner_pid": int(latest["pid"]),
               "cache_off_state": str(SWITCH), "records": records, "time": time.time()}
    save(RECOVERY / "receipt.json", receipt)
    print(json.dumps({"reconciled": len(records), "budget_reset": False}))


if __name__ == "__main__":
    main()
