import json
from pathlib import Path

import pytest

import scripts.server.control_corrected_agent_eval as corrected_eval
from scripts.server.control_corrected_agent_eval import (
    ServiceSession,
    extraction_protocol_report,
    normalized_command,
)


def test_normalized_command_uses_same_parser_for_every_weight(tmp_path):
    command = [
        "python",
        "vllm",
        "serve",
        "/old",
        "--served-model-name",
        "old-alias",
        "--tool-call-parser",
        "legacy",
        "--tool-parser-plugin",
        "/legacy.py",
    ]
    result = normalized_command(command, tmp_path / "model")
    assert result[3] == str(tmp_path / "model")
    assert result[result.index("--tool-call-parser") + 1] == "qwen3_coder"
    assert "--tool-parser-plugin" not in result


def test_extract_protocol_requires_qwen_local_and_exact_model(tmp_path):
    run = tmp_path / "smoke"
    run.mkdir()
    trace = run / "trace.json"
    trace.write_text(
        json.dumps(
            {
                "state": {
                    "tool_observations": [
                        {
                            "subcalls": [
                                {
                                    "kind": "page_extract",
                                    "provider": "qwen_local",
                                    "model": "policy-under-test",
                                    "status": "success",
                                    "thinking_enabled": True,
                                    "thinking_token_budget": 2048,
                                }
                            ]
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    selected = {"a": ({"trace_path": trace.name}, run)}
    report = extraction_protocol_report(selected, ["a"], "policy-under-test")
    assert report["passed"] is True
    assert report["page_extracts"] == 1

    mismatch = extraction_protocol_report(selected, ["a"], "different-policy")
    assert mismatch["passed"] is False
    assert mismatch["mismatches"][0]["model"] == "policy-under-test"


class _ReceiptOwner:
    def __init__(self, replacement, alive):
        self.replacement = replacement
        self.alive = set(alive)
        self.stopped = []

    def checked(self, receipt):
        if receipt["pid"] not in self.alive:
            raise FileNotFoundError(receipt["pid"])
        return {}

    def load(self, path):
        assert Path(path).name == "replica-0.json"
        return self.replacement

    def stop(self, receipt):
        self.checked(receipt)
        self.stopped.append(receipt["pid"])


def test_service_session_stops_identical_external_replacement(tmp_path, monkeypatch):
    command = ["python", "vllm", "serve", "/model", "--port", "19002"]
    stale = {"pid": 10, "command": command}
    replacement = {"pid": 20, "command": command.copy()}
    monkeypatch.setattr(corrected_eval, "SERVICE", tmp_path)
    session = object.__new__(ServiceSession)
    session.owner = _ReceiptOwner(replacement, alive={20})
    session.active = [stale]

    session._stop_active()

    assert session.owner.stopped == [20]
    assert session.active == []


def test_service_session_rejects_live_replacement_command_drift(tmp_path, monkeypatch):
    stale = {"pid": 10, "command": ["python", "vllm", "--port", "19002"]}
    replacement = {
        "pid": 20,
        "command": ["python", "vllm", "--port", "19002", "--different"],
    }
    monkeypatch.setattr(corrected_eval, "SERVICE", tmp_path)
    session = object.__new__(ServiceSession)
    session.owner = _ReceiptOwner(replacement, alive={20})
    session.active = [stale]

    with pytest.raises(RuntimeError, match="replacement command"):
        session._stop_active()

    assert session.owner.stopped == []
