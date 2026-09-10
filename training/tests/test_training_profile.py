from __future__ import annotations

import json
from pathlib import Path

from ifv_training.profile import summarize_training_log, write_training_profile


def test_training_profile_summarizes_ms_swift_metric_lines(tmp_path: Path) -> None:
    train_log = tmp_path / "train.log"
    train_log.write_text(
        "\n".join(
            [
                "Executing: swift sft --per_device_train_batch_size 1 "
                "--gradient_accumulation_steps 2 --sequence_parallel_size 4",
                "[INFO:swift] rank: 0, local_rank: 0, world_size: 4",
                "{'loss': '1.50', 'global_step/max_steps': '1/10', "
                "'elapsed_time': '31s', 'memory(GiB)': '21.5', "
                "'train_speed(s/it)': '31.2'}",
                "{'loss': '1.20', 'global_step/max_steps': '2/10', "
                "'elapsed_time': '58s', 'memory(GiB)': '22.0', "
                "'train_speed(s/it)': '28.8'}",
                "{'eval_loss': '0.52', 'global_step/max_steps': '2/10'}",
                "{'train_runtime': '88.0', 'global_step/max_steps': '2/10', "
                "'train_speed(s/it)': '44.0'}",
                '{"train_dataset": "size=445"}',
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
    assert result["metric_rows"] == 4
    assert result["train_step_metric_rows"] == 2
    assert result["eval_metric_rows"] == 1
    assert result["summary_metric_rows"] == 1
    assert result["steps"]["last"] == 2
    assert result["speed_seconds_per_step"]["steady_mean"] == 30.0
    assert result["speed_seconds_per_step"]["last"] == 28.8
    assert result["step_wall_seconds"]["values"] == [31.2, 26.4]
    assert result["step_wall_seconds"]["steady_mean"] == 28.8
    assert result["step_wall_seconds"]["startup_count"] == 0
    assert result["step_wall_seconds"]["steady_count"] == 2
    assert result["step_wall_seconds"]["steady_median"] == 28.8
    assert result["step_wall_seconds"]["steady_p90"] == 30.72
    assert result["step_wall_seconds"]["steady_max"] == 31.2
    assert (
        result["step_wall_seconds"]["steady_coefficient_of_variation"]
        == 0.083333
    )
    assert result["step_wall_seconds"]["steady_stall_count"] == 0
    assert result["parallelism"]["world_size"] == 4
    assert result["parallelism"]["sequence_parallel_size"] == 4
    assert result["parallelism"]["data_parallel_size"] == 1
    assert result["parallelism"]["unique_samples_per_optimizer_step"] == 2
    assert result["throughput"]["train_dataset_size"] == 445
    assert result["throughput"]["observed_unique_samples"] == 4
    assert result["throughput"]["unique_samples_per_second"] == 0.069444
    assert result["throughput"]["steady_unique_samples_per_second"] == 0.069444
    assert result["runtime_seconds"]["train_runtime"] == 88.0
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


def test_training_profile_accepts_num_train_epochs_schedule(tmp_path: Path) -> None:
    train_log = tmp_path / "train.log"
    train_log.write_text(
        "\n".join(
            [
                "Executing: swift sft --num_train_epochs 1 "
                "--per_device_train_batch_size 1 "
                "--gradient_accumulation_steps 2",
                "[INFO:swift] rank: 0, local_rank: 0, world_size: 4",
                "{'loss': '1.20', 'epoch': '0.5', "
                "'global_step/max_steps': '1/2', "
                "'train_speed(s/it)': '20.0'}",
                "{'loss': '1.00', 'epoch': '1.0', "
                "'global_step/max_steps': '2/2', "
                "'train_speed(s/it)': '20.0'}",
                "{'eval_loss': '0.70', 'epoch': '1.0', "
                "'global_step/max_steps': '2/2'}",
                "{'train_runtime': '45.0', 'global_step/max_steps': '2/2'}",
                "[INFO:swift] End time of running main: 2026-08-17 12:00:00",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = summarize_training_log(
        train_log,
        train_exit_code=0,
    )

    assert result["steps"]["configured_max_steps"] is None
    assert result["steps"]["configured_num_train_epochs"] == 1.0
    assert result["steps"]["last_observed_epoch"] == 1.0
    assert result["steps"]["complete"] is True


def test_training_profile_proves_cache_validation_save_resume_and_resources(
    tmp_path: Path,
) -> None:
    resume = tmp_path / "checkpoint-10"
    resume.mkdir()
    (resume / "optimizer.pt").write_bytes(b"optimizer")
    (resume / "scheduler.pt").write_bytes(b"scheduler")
    (resume / "rng_state_0.pth").write_bytes(b"rng")
    (resume / "trainer_state.json").write_text(
        json.dumps({"global_step": 10}),
        encoding="utf-8",
    )
    saved = tmp_path / "checkpoint-11"
    saved.mkdir()
    (saved / "optimizer.pt").write_bytes(b"optimizer")
    (saved / "scheduler.pt").write_bytes(b"scheduler")
    (saved / "rng_state_0.pth").write_bytes(b"rng")
    (saved / "trainer_state.json").write_text(
        json.dumps({"global_step": 11}),
        encoding="utf-8",
    )
    train_log = tmp_path / "train.log"
    train_log.write_text(
        "\n".join(
            [
                "Executing: swift sft --cached_dataset /cache/train "
                "--cached_val_dataset /cache/val --max_steps 11 "
                "--per_device_train_batch_size 1 "
                "--gradient_accumulation_steps 2 "
                "--eval_strategy steps --eval_steps 11 "
                "--save_strategy steps --save_steps 11 "
                f"--resume_from_checkpoint {resume.as_posix()}",
                "[INFO:swift] rank: 0, local_rank: 0, world_size: 4",
                "{'loss': '1.20', 'global_step/max_steps': '11/11', "
                "'memory(GiB)': '12.7', 'train_speed(s/it)': '37.5'}",
                "{'eval_loss': '0.52', 'eval_runtime': '20.0', "
                "'global_step/max_steps': '11/11'}",
                f"[INFO:swift] Saving model checkpoint to {saved.as_posix()}",
                "{'train_runtime': '93.0', 'global_step/max_steps': '11/11'}",
                f"[INFO:swift] last_model_checkpoint: {saved.as_posix()}",
                "[INFO:swift] End time of running main: 2026-08-03 12:00:00",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    resource_summary = tmp_path / "resource-summary.json"
    resource_summary.write_text(
        json.dumps(
            {
                "schema_version": "ifv-training-resource-summary-v1",
                "exit_code": 0,
                "process_tree_peak_rss_mib": 12345.0,
            }
        ),
        encoding="utf-8",
    )
    cache_verification = tmp_path / "cache-verification.json"
    cache_verification.write_text(
        json.dumps(
            {
                "schema_version": "ifv-cached-dataset-gate-v1",
                "passed": True,
            }
        ),
        encoding="utf-8",
    )
    checkpoint_preflight = tmp_path / "checkpoint-preflight.json"
    checkpoint_preflight.write_text(
        json.dumps(
            {
                "schema_version": "ifv-checkpoint-storage-preflight-v1",
                "passed": True,
            }
        ),
        encoding="utf-8",
    )
    checkpoint_io = tmp_path / "checkpoint-io.json"
    checkpoint_io.write_text(
        json.dumps(
            {
                "schema_version": "ifv-checkpoint-io-profile-v1",
                "passed": True,
            }
        ),
        encoding="utf-8",
    )

    result = summarize_training_log(
        train_log,
        resource_summary=resource_summary,
        cache_verification=cache_verification,
        checkpoint_preflight=checkpoint_preflight,
        checkpoint_io_profile=checkpoint_io,
        train_exit_code=0,
    )

    assert result["passed_production_gate"] is True
    assert result["steps"]["complete"] is True
    assert result["validation"]["observed"] is True
    assert result["checkpoint_save"]["states"][0]["global_step"] == 11
    assert result["checkpoint_save"]["states"][0]["optimizer_state_available"] is True
    assert result["resume"]["source"]["global_step"] == 10
    assert result["resume"]["advanced"] is True
    assert result["resources"]["summary"]["process_tree_peak_rss_mib"] == 12345.0
    assert result["cached_dataset_gate"]["passed"] is True
    assert result["checkpoint_storage_preflight"]["passed"] is True
    assert result["checkpoint_io"]["passed"] is True


def test_scheduler_warning_requires_wrapper_false_positive_audit(
    tmp_path: Path,
) -> None:
    train_log = tmp_path / "train.log"
    train_log.write_text(
        "Detected call of `lr_scheduler.step()` before `optimizer.step()`.\n",
        encoding="utf-8",
    )

    unresolved = summarize_training_log(train_log)

    assert unresolved["scheduler_order"]["warning_count"] == 1
    assert unresolved["scheduler_order"]["audit_required"] is True
    assert unresolved["scheduler_order"]["passed"] is False

    audit = tmp_path / "scheduler-audit.json"
    audit.write_text(
        json.dumps(
            {
                "schema_version": "ifv-deepspeed-scheduler-audit-v1",
                "passed": True,
                "classification": "wrapper_false_positive",
            }
        ),
        encoding="utf-8",
    )

    resolved = summarize_training_log(
        train_log,
        scheduler_audit=audit,
    )

    assert resolved["scheduler_order"]["passed"] is True


def test_noeval_training_profile_is_explicitly_smoke_only(tmp_path: Path) -> None:
    train_log = tmp_path / "train.log"
    train_log.write_text(
        "\n".join(
            [
                "Executing: swift sft --eval_strategy no --save_strategy steps "
                "--save_steps 1 --max_steps 1",
                "{'loss': '0.10', 'global_step/max_steps': '1/1', "
                "'train_speed(s/it)': '1.0'}",
                "[INFO:swift] Saving model checkpoint to "
                f"{tmp_path / 'checkpoint-1'}",
                "[INFO:swift] End time of running main: 2026-09-06 12:00:00",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = summarize_training_log(train_log, train_exit_code=0)

    assert result["smoke_only"] is True
    assert result["run_mode"] == "smoke_only"
    assert result["validation"]["required"] is False
    assert result["passed_production_gate"] is False


def test_training_profile_rejects_failed_resource_acceptance(
    tmp_path: Path,
) -> None:
    train_log = tmp_path / "train.log"
    train_log.write_text(
        "Executing: swift sft --max_steps 1 --eval_strategy no "
        "--save_strategy no\n"
        "{'loss': '0.8', 'global_step/max_steps': '1/1'}\n"
        "[INFO:swift] End time of running main: 2026-09-10 12:00:00\n",
        encoding="utf-8",
    )
    resource_summary = tmp_path / "resource-summary.json"
    resource_summary.write_text(
        json.dumps(
            {
                "schema_version": "ifv-training-resource-summary-v2",
                "exit_code": 0,
                "acceptance": {"required": True, "passed": False},
            }
        ),
        encoding="utf-8",
    )

    result = summarize_training_log(
        train_log,
        resource_summary=resource_summary,
        train_exit_code=0,
    )

    assert result["resources"]["acceptance_passed"] is False
    assert result["resources"]["passed"] is False
    assert result["passed_production_gate"] is False


def test_training_profile_requires_raw_dataset_gate(tmp_path: Path) -> None:
    train_log = tmp_path / "train.log"
    train_log.write_text(
        "Executing: swift sft --dataset /data/train.jsonl --max_steps 1\n",
        encoding="utf-8",
    )

    missing = summarize_training_log(train_log)

    assert missing["raw_dataset_gate"]["required"] is True
    assert missing["raw_dataset_gate"]["passed"] is False

    verification = tmp_path / "raw-dataset-gate.json"
    verification.write_text(json.dumps({"passed": True}), encoding="utf-8")
    accepted = summarize_training_log(
        train_log,
        dataset_verification=verification,
    )

    assert accepted["raw_dataset_gate"]["passed"] is True
