from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts import run_psd_round
from ifv_training.io import write_json


def test_training_entrypoint_carries_adapter_and_same_round_resume(monkeypatch, tmp_path):
    ready = {"round_index": 2, "serving_profile": str(tmp_path / "serving.json"),
        "checkpoint_manifest": str(tmp_path / "checkpoint.json"),
        "datums": str(tmp_path / "datums.jsonl"), "adapter": str(tmp_path / "previous-adapter")}
    monkeypatch.setattr(run_psd_round, "load_ready", lambda _: ready)
    calls = []
    monkeypatch.setattr(run_psd_round.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)))
    args = SimpleNamespace(ready=tmp_path / "ready.json", model_profile=tmp_path / "model.env",
        psd_profile=tmp_path / "psd.env", experiment_id="test-round-2",
        resume_checkpoint=tmp_path / "same-round/checkpoint-1")
    assert run_psd_round.train(args)["round_index"] == 2
    command, kwargs = calls[0]
    assert command[-1] == str(args.resume_checkpoint.resolve())
    assert kwargs["env"]["IFV_PSD_INITIAL_ADAPTER"] == ready["adapter"]
    assert kwargs["env"]["IFV_PSD_ROUND_READY"] == str(args.ready.resolve())
    assert kwargs["check"] is True


def test_finalization_rejects_an_unrelated_optimizer_initialization(monkeypatch, tmp_path):
    monkeypatch.setattr(run_psd_round, "load_ready", lambda _: {"initialization": {"checkpoint": "round-1"}})
    gate = tmp_path / "gate.json"
    write_json(gate, {"checkpoint": "other-round"})
    with pytest.raises(ValueError, match="initialization differs"):
        run_psd_round.finalize(SimpleNamespace(ready=tmp_path / "ready.json", initialization_gate=gate))
