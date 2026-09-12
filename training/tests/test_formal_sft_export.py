from __future__ import annotations

import json
from pathlib import Path

import ifv_training.checkpoints as checkpoints


def test_full_parameter_audit_can_accept_explicit_model_only_checkpoint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model = tmp_path / "model"
    base = tmp_path / "base"
    state = tmp_path / "state"
    for directory in (model, base, state):
        directory.mkdir()
    (model / "args.json").write_text(
        json.dumps(
            {
                "tuner_type": "full",
                "freeze_llm": False,
                "freeze_vit": False,
                "freeze_aligner": False,
                "save_only_model": True,
            }
        ),
        encoding="utf-8",
    )
    (model / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    monkeypatch.setattr(
        checkpoints,
        "_component_weight_deltas",
        lambda *_: {
            component: {"updated": True}
            for component in ("language", "vision", "aligner")
        },
    )

    result = checkpoints.audit_full_parameter_checkpoint(
        checkpoint_dir=model,
        base_model_dir=base,
        state_checkpoint_dir=state,
        require_training_state=False,
    )

    assert result["passed"] is True
    assert result["training_state_required"] is False
    assert result["checks"]["optimizer_state_present"] is False
    assert "optimizer_state_present" not in result["required_checks"]


def test_full_parameter_audit_still_requires_state_by_default(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model = tmp_path / "model"
    base = tmp_path / "base"
    state = tmp_path / "state"
    for directory in (model, base, state):
        directory.mkdir()
    (model / "args.json").write_text(
        json.dumps(
            {
                "tuner_type": "full",
                "freeze_llm": False,
                "freeze_vit": False,
                "freeze_aligner": False,
            }
        ),
        encoding="utf-8",
    )
    (model / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    monkeypatch.setattr(
        checkpoints,
        "_component_weight_deltas",
        lambda *_: {
            component: {"updated": True}
            for component in ("language", "vision", "aligner")
        },
    )

    result = checkpoints.audit_full_parameter_checkpoint(
        checkpoint_dir=model,
        base_model_dir=base,
        state_checkpoint_dir=state,
    )

    assert result["passed"] is False
    assert result["training_state_required"] is True
    assert "optimizer_state_present" in result["required_checks"]
