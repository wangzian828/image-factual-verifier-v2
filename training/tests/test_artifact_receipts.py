from pathlib import Path

import pytest

from ifv_training.artifact_receipts import freeze_artifact, verify_artifact


def test_payload_is_never_read_to_freeze_or_verify(tmp_path: Path, monkeypatch):
    artifact = tmp_path / "large.jsonl.gz"
    artifact.write_bytes(b"payload")
    receipt = tmp_path / "receipt.json"
    frozen = freeze_artifact(artifact, receipt)

    def forbidden(_path):
        raise AssertionError("payload was hashed more than once")

    monkeypatch.setattr(Path, "read_bytes", forbidden)
    assert verify_artifact(receipt) == frozen
    assert freeze_artifact(artifact, receipt) == frozen


def test_changed_artifact_is_rejected_without_payload_read(tmp_path: Path, monkeypatch):
    artifact = tmp_path / "large.jsonl.gz"
    artifact.write_bytes(b"payload")
    receipt = tmp_path / "receipt.json"
    freeze_artifact(artifact, receipt)
    artifact.write_bytes(b"changed")
    monkeypatch.setattr(Path, "read_bytes", lambda _path: (_ for _ in ()).throw(
        AssertionError("unexpected payload read")))
    with pytest.raises(ValueError, match="identity changed"):
        verify_artifact(receipt)
