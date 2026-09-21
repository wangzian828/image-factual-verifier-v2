from pathlib import Path

import pytest

from scripts.server.run_prefix_cache_hybrid_gate import (
    BLOCK_SIZE,
    attest_block_size,
    canary_gateway_command,
    stop_gateway,
    validate_original_replicas,
    validate_source_snapshot,
)


def receipt(root: str) -> dict:
    return {
        "command": [
            "python",
            "vllm",
            "serve",
            root,
            "--no-enable-prefix-caching",
            "--mamba-cache-mode",
            "none",
        ]
    }


def test_original_replicas_must_be_uniform_and_cache_off():
    from scripts.server.run_prefix_cache_hybrid_gate import EXPECTED_SFT_ROOT

    root = validate_original_replicas(
        [receipt(str(EXPECTED_SFT_ROOT)) for _ in range(4)]
    )
    assert root == EXPECTED_SFT_ROOT.resolve()
    invalid = receipt(str(EXPECTED_SFT_ROOT))
    invalid["command"].remove("--no-enable-prefix-caching")
    with pytest.raises(ValueError, match="cache-off"):
        validate_original_replicas(
            [invalid, *[receipt(str(EXPECTED_SFT_ROOT)) for _ in range(3)]]
        )


def test_canary_gateway_uses_immutable_overlay():
    command = ["python", "-m", "uvicorn", "app", "--app-dir", "/old"]
    result = canary_gateway_command(command, Path("/new"))
    assert result[result.index("--app-dir") + 1] == str(Path("/new"))
    assert command[command.index("--app-dir") + 1] == "/old"


def test_block_size_attestation_requires_every_replica(tmp_path):
    logs = []
    for index in range(4):
        path = tmp_path / f"{index}.log"
        path.write_text(f"Setting attention block size to {BLOCK_SIZE} tokens")
        logs.append(path)
    attest_block_size(logs)
    logs[-1].write_text("wrong")
    with pytest.raises(ValueError, match="attestation missing"):
        attest_block_size(logs)


def test_source_snapshot_rejects_partial_overlay(tmp_path):
    with pytest.raises(ValueError, match="incomplete"):
        validate_source_snapshot(tmp_path)


def test_stop_gateway_treats_zombie_as_stopped(monkeypatch):
    import scripts.server.run_prefix_cache_hybrid_gate as gate

    class FakeStat:
        def read_text(self):
            return "1 (gateway) Z 0"

    class FakeProc:
        def exists(self):
            return True

        def __truediv__(self, name):
            assert name == "stat"
            return FakeStat()

    monkeypatch.setattr(gate.os, "killpg", lambda *_: None, raising=False)
    monkeypatch.setattr(gate.os, "waitpid", lambda *_: (7, 0), raising=False)
    monkeypatch.setattr(gate.os, "WNOHANG", 1, raising=False)
    monkeypatch.setattr(gate, "Path", lambda *_: FakeProc())
    stop_gateway(7)
