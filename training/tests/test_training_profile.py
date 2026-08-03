from __future__ import annotations

import json
from pathlib import Path

from ifv_training.profile import summarize_training_log, write_training_profile


def test_training_profile_summarizes_ms_swift_metric_lines(tmp_path: Path) -> None:
    train_log = tmp_path / "train.log"
    train_log.write_text(
        "\n".join(
            [
                "{'loss': '1.50', 'global_step/max_steps': '1/10', "
                "'memory(GiB)': '21.5', 'train_speed(s/it)': '31.2'}",
                "{'loss': '1.20', 'global_step/max_steps': '2/10', "
                "'memory(GiB)': '22.0', 'train_speed(s/it)': '28.8'}",
                "{'eval_loss': '0.52', 'global_step/max_steps': '2/10'}",
                "{'model_type': 'qwen3_5', 'hidden_size': 4096}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = summarize_training_log(
        train_log,
        steady_window=2,
        experiment_id="exp-a",
        profile_id="profile-a",
    )

    assert result["schema_version"] == "ifv-training-profile-v1"
    assert result["experiment_id"] == "exp-a"
    assert result["profile_id"] == "profile-a"
    assert result["passed_basic_log_gate"] is True
    assert result["metric_rows"] == 3
    assert result["steps"]["last"] == 2
    assert result["speed_seconds_per_step"]["steady_mean"] == 30.0
    assert result["memory_gib"]["max"] == 22.0
    assert result["loss"]["last"] == 1.2
    assert result["eval_loss"]["last"] == 0.52


def test_training_profile_detects_error_signals_and_writes_json(tmp_path: Path) -> None:
    train_log = tmp_path / "train.log"
    output = tmp_path / "profile.json"
    train_log.write_text(
        "torch.cuda.OutOfMemoryError: CUDA out of memory\n"
        "{'global_step/max_steps': '1/1', 'train_speed(s/it)': '66.17'}\n",
        encoding="utf-8",
    )

    result = write_training_profile(train_log=train_log, output=output)
    persisted = json.loads(output.read_text(encoding="utf-8"))

    assert result["passed_basic_log_gate"] is False
    assert "cuda_oom" in result["detected_errors"]
    assert persisted["speed_seconds_per_step"]["last"] == 66.17
