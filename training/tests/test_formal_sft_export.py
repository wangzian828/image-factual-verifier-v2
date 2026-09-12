from __future__ import annotations

import json
from pathlib import Path

import ifv_training.checkpoints as checkpoints
from scripts.h20.export_formal_sft import (
    _bool_arg,
    _full_state_weight_files,
    _link_or_copy,
)


def test_full_state_weight_files_follow_index(tmp_path: Path) -> None:
    (tmp_path / "weights-a.safetensors").write_bytes(b"a")
    (tmp_path / "weights-b.safetensors").write_bytes(b"b")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.a": "weights-a.safetensors",
                    "model.b": "weights-b.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )

    assert [path.name for path in _full_state_weight_files(tmp_path)] == [
        "weights-a.safetensors",
        "weights-b.safetensors",
    ]


def test_full_state_weight_files_reject_missing_index_shard(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.a": "missing.safetensors"}}),
        encoding="utf-8",
    )

    try:
        _full_state_weight_files(tmp_path)
    except ValueError as exc:
        assert "missing FULL_STATE_DICT shard" in str(exc)
    else:
        raise AssertionError("missing shard was accepted")


def test_link_or_copy_preserves_source_and_payload(tmp_path: Path) -> None:
    source = tmp_path / "source.safetensors"
    destination = tmp_path / "output" / source.name
    destination.parent.mkdir()
    source.write_bytes(b"immutable-weights")

    mode = _link_or_copy(source, destination)

    assert mode in {"hardlink", "copy"}
    assert source.read_bytes() == destination.read_bytes() == b"immutable-weights"


def test_bool_arg_accepts_serialized_true_only() -> None:
    assert _bool_arg(True)
    assert _bool_arg("YES")
    assert not _bool_arg(False)
    assert not _bool_arg("false")


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
