"""Reconcile Qwen repair attempts interrupted by a controlled owner handoff.

This is deliberately narrow: it only charges the final ``running`` attempt
as an infrastructure failure after every repair owner has exited, preserves
the original bound ledger beside a small receipt, and never touches a
completed result or a checker outcome.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from ifv_training.psd_repair_storage import load_bound, save_bound


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


def active_repair_owners(*, self_pid: int) -> list[dict]:
    rows = []
    output = subprocess.run(["ps", "-eo", "pid=,args="], text=True,
                            capture_output=True, check=True).stdout
    for line in output.splitlines():
        line = line.strip()
        if not line or "run_psd_old1000_repair.py" not in line:
            continue
        pid_text, _, command = line.partition(" ")
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid != self_pid and alive(pid):
            rows.append({"pid": pid, "command": command})
    return rows


def reconcile_marker(marker: Path, *, output: Path, recovery: Path,
                     owner_pids: list[int]) -> dict:
    original = marker.read_bytes()
    raw = json.loads(original)
    identity = raw["identity"]
    state = load_bound(marker, identity=identity)
    attempts = state.get("attempts")
    if (not isinstance(attempts, list) or not attempts
            or attempts[-1].get("status") != "running"
            or attempts[-1].get("index") != len(attempts)
            or len(attempts) > identity.get("max_attempts", 0)):
        raise ValueError("unexpected running Qwen retry ledger")
    if (marker.parent / "result.json").exists():
        raise RuntimeError("cannot reconcile a Qwen retry result that already completed")
    latest = attempts[-1]
    attempt_dir = marker.parent / f"attempt-{len(attempts):03d}"
    if Path(latest["directory"]).resolve() != attempt_dir.resolve():
        raise ValueError("running Qwen attempt directory changed")
    if not attempt_dir.is_dir() or any(attempt_dir.rglob("result.json")):
        raise RuntimeError("running Qwen attempt contains a completed result")
    relative = marker.relative_to(output)
    backup = recovery / relative
    before_sha = hashlib.sha256(original).hexdigest()
    if backup.exists():
        if backup.read_bytes() != original:
            raise ValueError("handoff ledger backup differs")
    else:
        backup.parent.mkdir(parents=True, exist_ok=True)
        backup.write_bytes(original)
    latest.update(status="infrastructure_failed",
                  reason="controlled_repair_owner_handoff",
                  legacy_status="running",
                  migration_version="ifv-old1000-repair-owner-handoff-v1",
                  source_owner_pids=list(owner_pids))
    save_bound(marker, identity=identity, payload=state)
    return {"marker": str(relative), "backup": str(backup.relative_to(output)),
            "attempts_charged": len(attempts), "before_sha256": before_sha,
            "after_sha256": hashlib.sha256(marker.read_bytes()).hexdigest()}


def migrate_known_pre_fix_target_error(marker: Path, *, output: Path,
                                       recovery: Path, evidence: Path) -> dict:
    """Reopen one proven pre-fix target-binding error without resetting budget."""
    original = marker.read_bytes()
    raw = json.loads(original)
    identity = raw["identity"]
    state = load_bound(marker, identity=identity)
    attempts = state.get("attempts")
    if (not isinstance(attempts, list) or not attempts
            or attempts[-1].get("status") != "nonretryable_error"):
        raise ValueError("known pre-fix marker is not an unresolved nonretryable ledger")
    error = load(evidence)
    if (error.get("error_type") != "ValueError"
            or error.get("message") != "PSD slate actual teacher prefix differs from corrected history"):
        raise ValueError("evidence is not the frozen pre-fix PSD target-binding error")
    if (marker.parent / "result.json").exists():
        raise RuntimeError("cannot migrate a completed Qwen retry result")
    latest = attempts[-1]
    attempt_dir = marker.parent / f"attempt-{len(attempts):03d}"
    if Path(latest["directory"]).resolve() != attempt_dir.resolve():
        raise ValueError("known pre-fix attempt directory changed")
    if not attempt_dir.is_dir() or any(attempt_dir.rglob("result.json")):
        raise RuntimeError("known pre-fix attempt contains a completed result")
    relative = marker.relative_to(output)
    backup = recovery / relative / "ledger-original.json"
    if backup.exists():
        if backup.read_bytes() != original:
            raise ValueError("known pre-fix ledger backup differs")
    else:
        backup.parent.mkdir(parents=True, exist_ok=True)
        backup.write_bytes(original)
    latest.update(status="infrastructure_failed",
                  reason="pre_fix_psd_target_binding_error",
                  legacy_status="nonretryable_error",
                  migration_version="ifv-old1000-pre-fix-target-binding-v1",
                  evidence_path=str(evidence),
                  evidence_sha256=hashlib.sha256(evidence.read_bytes()).hexdigest())
    save_bound(marker, identity=identity, payload=state)
    return {"marker": str(relative), "backup": str(backup.relative_to(output)),
            "attempts_charged": len(attempts), "evidence": str(evidence),
            "before_sha256": hashlib.sha256(original).hexdigest(),
            "after_sha256": hashlib.sha256(marker.read_bytes()).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-pid", type=int, action="append", required=True)
    parser.add_argument("--known-prefix-marker", type=Path, action="append")
    parser.add_argument("--known-prefix-evidence", type=Path, action="append")
    args = parser.parse_args()
    output = args.output.resolve()
    recovery = output / "recoveries/owner-handoff-v1"
    if not output.is_dir():
        raise FileNotFoundError(output)
    owners = active_repair_owners(self_pid=os.getpid())
    if owners:
        raise RuntimeError("repair owner is still active: " + repr(owners))
    known_markers = args.known_prefix_marker or []
    known_evidence = args.known_prefix_evidence or []
    if len(known_markers) != len(known_evidence):
        raise ValueError("known pre-fix markers and evidence must be paired")
    if known_markers:
        known_recovery = output / "recoveries/pre-fix-target-binding-v1"
        receipt_path = known_recovery / "receipt.json"
        if receipt_path.exists():
            raise RuntimeError("pre-fix target-binding migration already recorded")
        records = [migrate_known_pre_fix_target_error(marker.resolve(), output=output,
                                                       recovery=known_recovery,
                                                       evidence=evidence.resolve())
                   for marker, evidence in zip(known_markers, known_evidence)]
        save(receipt_path, {"schema_version": "ifv-old1000-pre-fix-target-binding-v1",
                            "count": len(records), "budget_reset": False,
                            "records": records, "time": time.time()})
        print(json.dumps({"migrated": len(records), "budget_reset": False}))
        return
    if (recovery / "receipt.json").exists():
        raise RuntimeError("repair handoff reconciliation already recorded")
    markers = []
    for marker in output.glob("repairs/*/slate-rounds/*/infrastructure-attempts/retry-state.json"):
        raw = load(marker)
        payload = raw.get("payload", {})
        attempts = payload.get("attempts", [])
        if attempts and attempts[-1].get("status") == "running":
            markers.append(marker)
    records = [reconcile_marker(marker, output=output, recovery=recovery,
                                 owner_pids=args.source_pid)
               for marker in sorted(markers)]
    receipt = {"schema_version": "ifv-old1000-repair-owner-handoff-v1",
               "count": len(records), "budget_reset": False,
               "source_owner_pids": list(args.source_pid), "records": records,
               "time": time.time()}
    save(recovery / "receipt.json", receipt)
    print(json.dumps({"reconciled": len(records), "budget_reset": False}))


if __name__ == "__main__":
    main()
