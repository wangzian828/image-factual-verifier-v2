from __future__ import annotations

import json
import sys
from pathlib import Path

from ifv_training import cli


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, values: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value) + "\n" for value in values),
        encoding="utf-8",
    )


def test_cli_keeps_runtime_only_psd_repair_imports_lazy() -> None:
    source = Path(cli.__file__).read_text(encoding="utf-8")
    module_imports = source.split("def _parser", 1)[0]

    assert "from .psd_repair import" not in module_imports
    assert "from .psd_repair_verifier import" not in module_imports


def test_build_run_rewards_cli_executes_real_branch(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    _write_jsonl(
        tmp_path / "deterministic.jsonl",
        [
            {
                "episode_id": "episode-0",
                "case_id": "case-reward-1",
                "classification_correct": True,
                "strict_trace_audit_pass": True,
                "fatal_engineering_error": False,
                "step_ids": ["episode-0:judgment:1"],
                "process_components": {
                    "evidence_chain_reward": 1.0,
                    "discrepancy_alignment_reward": 1.0,
                    "stop_quality_reward": 1.0,
                },
            }
        ],
    )
    _write_jsonl(
        tmp_path / "members.jsonl",
        [
            {
                "schema_version": "ifv-rollout-group-member-v1",
                "prompt_group_id": "pg-1",
                "case_id": "case-reward-1",
                "episode_id": "episode-0",
                "rollout_index": 0,
                "group_size": 1,
                "sampling_seed": 1,
                "training_prohibited": False,
                "training_eligible": True,
            }
        ],
    )
    root = Path(__file__).resolve().parents[1]
    argv = [
        "ifv-training",
        "build-run-rewards",
        "--deterministic",
        str(tmp_path / "deterministic.jsonl"),
        "--rollout-members",
        str(tmp_path / "members.jsonl"),
        "--profile",
        str(root / "configs" / "rl" / "deterministic-process-v2.json"),
        "--ledger-output",
        str(tmp_path / "ledgers.jsonl"),
        "--group-output",
        str(tmp_path / "groups.jsonl"),
    ]
    monkeypatch.setattr(sys, "argv", argv)  # type: ignore[attr-defined]

    cli.main()

    assert (tmp_path / "ledgers.jsonl").is_file()
    groups = [
        json.loads(line)
        for line in (tmp_path / "groups.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert groups[0]["skip_reason"] == "insufficient_valid_members"


def test_training_profile_cli_executes_real_branch(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    train_log = tmp_path / "train.log"
    train_log.write_text(
        "{'global_step/max_steps': '1/1', 'train_speed(s/it)': '12.5'}\n",
        encoding="utf-8",
    )
    output = tmp_path / "profile.json"
    argv = [
        "ifv-training",
        "training-profile",
        "--train-log",
        str(train_log),
        "--output",
        str(output),
        "--experiment-id",
        "exp-cli",
        "--profile-id",
        "profile-cli",
    ]
    monkeypatch.setattr(sys, "argv", argv)  # type: ignore[attr-defined]

    cli.main()

    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["experiment_id"] == "exp-cli"
    assert result["profile_id"] == "profile-cli"
    assert result["speed_seconds_per_step"]["last"] == 12.5
    assert result["train_step_metric_rows"] == 0
