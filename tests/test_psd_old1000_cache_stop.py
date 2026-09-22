import json

import pytest

from scripts.server import reconcile_old1000_cache_stop as recovery
from ifv_training.psd_repair_storage import load_bound, save_bound


def test_interruption_reconciliation_charges_existing_attempt(tmp_path, monkeypatch):
    output = tmp_path / "repair-search-v1"
    marker = output / "repairs/key/slate-rounds/00/infrastructure-attempts/retry-state.json"
    attempt = marker.parent / "attempt-002"
    attempt.mkdir(parents=True)
    identity = {"version": "ifv-psd-infrastructure-retry-v2", "inputs": {}, "max_attempts": 3}
    state = {"attempts": [
        {"index": 1, "directory": str(marker.parent / "attempt-001"),
         "status": "infrastructure_failed"},
        {"index": 2, "directory": str(attempt), "status": "interrupted"}]}
    save_bound(marker, identity=identity, payload=state)
    monkeypatch.setattr(recovery, "OUTPUT", output)
    monkeypatch.setattr(recovery, "RECOVERY", output / "recoveries/cache-off-interrupted-v1")
    result = recovery.reconcile_marker(marker)
    assert result["attempts_charged"] == 2
    assert load_bound(marker, identity=identity)["attempts"][-1]["status"] == "infrastructure_failed"
    backup = output / result["backup"]
    assert json.loads(backup.read_text())["payload"]["attempts"][-1]["status"] == "interrupted"


def test_interruption_with_completed_result_is_never_reclassified(tmp_path, monkeypatch):
    output = tmp_path / "repair-search-v1"
    marker = output / "repairs/key/slate-rounds/00/infrastructure-attempts/retry-state.json"
    attempt = marker.parent / "attempt-001"
    attempt.mkdir(parents=True)
    identity = {"version": "ifv-psd-infrastructure-retry-v2", "inputs": {}, "max_attempts": 3}
    save_bound(marker, identity=identity, payload={"attempts": [
        {"index": 1, "directory": str(attempt), "status": "interrupted"}]})
    (marker.parent / "result.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(recovery, "OUTPUT", output)
    monkeypatch.setattr(recovery, "RECOVERY", output / "recoveries/cache-off-interrupted-v1")
    with pytest.raises(RuntimeError, match="completed"):
        recovery.reconcile_marker(marker)
    assert load_bound(marker, identity=identity)["attempts"][-1]["status"] == "interrupted"
