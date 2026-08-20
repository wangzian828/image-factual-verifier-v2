from __future__ import annotations

import json
from pathlib import Path

from ifv_training.checkpoint_selection import validate_checkpoints


def _checkpoint(root: Path, step: int) -> None:
    path = root / f"checkpoint-{step}"
    path.mkdir(parents=True)
    (path / "trainer_state.json").write_text(
        json.dumps({"global_step": step}),
        encoding="utf-8",
    )
    (path / "model.safetensors").write_bytes(b"model")


def _train_log(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "Executing: swift sft --per_device_train_batch_size 1 "
                "--gradient_accumulation_steps 2",
                "[INFO:swift] rank: 0, local_rank: 0, world_size: 4",
                "{'train_dataset': 'size=16'}",
                "{'eval_loss': '0.80', 'global_step/max_steps': '2/4'}",
                "{'eval_loss': '0.60', 'global_step/max_steps': '4/4'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_validate_checkpoints_selects_lowest_eval_loss_and_computes_epoch(
    tmp_path: Path,
) -> None:
    root = tmp_path / "checkpoints"
    root.mkdir()
    _checkpoint(root, 2)
    _checkpoint(root, 4)
    log = tmp_path / "train.log"
    _train_log(log)

    output = tmp_path / "validation.json"
    result = validate_checkpoints(
        train_log=log,
        checkpoint_root=root,
        output=output,
    )

    assert result["passed"] is True
    assert result["selection_mode"] == "eval_loss"
    assert result["optimizer_steps_per_epoch"] == 2
    assert result["selected"]["global_step"] == 4
    assert result["candidates"][0]["epoch"] == 2.0
    assert json.loads(output.read_text(encoding="utf-8"))["selected"][
        "global_step"
    ] == 4


def test_explicit_behavior_score_can_choose_earlier_checkpoint(
    tmp_path: Path,
) -> None:
    root = tmp_path / "checkpoints"
    root.mkdir()
    _checkpoint(root, 2)
    _checkpoint(root, 4)
    log = tmp_path / "train.log"
    _train_log(log)
    behavior = tmp_path / "behavior.jsonl"
    behavior.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "global_step": 2,
                        "metrics": {
                            "behavior_score": 0.95,
                            "verdict_accuracy": 0.95,
                            "evidence_chain_complete_rate": 0.90,
                        },
                    }
                ),
                json.dumps(
                    {
                        "global_step": 4,
                        "metrics": {
                            "behavior_score": 0.70,
                            "verdict_accuracy": 0.80,
                            "evidence_chain_complete_rate": 0.75,
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_checkpoints(
        train_log=log,
        checkpoint_root=root,
        behavior_metrics=behavior,
        output=tmp_path / "validation.json",
    )

    assert result["selection_mode"] == "behavior"
    assert result["selected"]["global_step"] == 2
