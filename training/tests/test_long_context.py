from __future__ import annotations

import json
from pathlib import Path

from ifv_training.io import sha256_file
from ifv_training.long_context import (
    LONG_CONTEXT_BOUNDARIES,
    build_long_context_plan,
    parse_env_profile,
    validate_128k_stage,
    validate_sp8_128k_profile,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "training" / "configs" / "sft"
PROFILES = {
    "memory-probe": CONFIG_ROOT
    / "qwen3.5-full-1step-8gpu-fsdp2-sp8-flash-128k-memory-probe.env",
    "canary": CONFIG_ROOT
    / "qwen3.5-full-10step-8gpu-fsdp2-sp8-flash-128k-canary.env",
    "resume": CONFIG_ROOT
    / "qwen3.5-full-11step-8gpu-fsdp2-sp8-flash-128k-resume.env",
}


def _training_profile(stage: str) -> dict:
    step = {"memory-probe": 1, "canary": 10, "resume": 11}[stage]
    profile = {
        "passed_basic_log_gate": True,
        "passed_production_gate": stage != "memory-probe",
        "run_mode": "smoke_only" if stage == "memory-probe" else "production_candidate",
        "steps": {"complete": True, "last": step},
        "process": {"clean_exit": True},
        "resources": {"required": True, "passed": True},
        "raw_dataset_gate": {
            "required": True,
            "passed": True,
            "verification": {
                "model": "/model",
                "template_contract": {"max_length": 131072},
                "datasets": {
                    "train": {"sha256": "a" * 64},
                    "validation": {"sha256": "b" * 64},
                },
                "processor_report": {"sha256": "c" * 64},
            },
        },
        "environment_preflight": {"required": True, "passed": True},
        "encoded_processor_cache": {"required": True, "passed": True},
        "checkpoint_save": {"passed": stage != "memory-probe"},
    }
    if stage == "resume":
        profile["resume"] = {
            "requested": True,
            "advanced": True,
            "source": {"global_step": 10},
        }
    return profile


def test_repository_128k_profiles_are_a_valid_ordered_chain() -> None:
    for stage, path in PROFILES.items():
        result = validate_sp8_128k_profile(parse_env_profile(path), stage=stage)
        assert result["passed"] is True


def test_long_context_plan_requires_real_120k_coverage(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    train.write_text("{}\n", encoding="utf-8")
    boundaries = [
        {
            "boundary_tokens": value,
            "rows_at_or_above": 1 if value <= 120_000 else 0,
            "rows_at_or_below": 1,
            "closest_at_or_above": None,
            "closest_at_or_below": None,
        }
        for value in LONG_CONTEXT_BOUNDARIES
    ]
    report = tmp_path / "processor.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": "ifv-ms-swift-agent-processor-verification-v2",
                "passed": True,
                "dataset_files": [
                    {"path": str(train.resolve()), "sha256": sha256_file(train)}
                ],
                "input_tokens_by_dataset": {
                    str(train.resolve()): {"count": 8490, "max": 125000}
                },
                "input_token_boundaries_by_dataset": {
                    str(train.resolve()): boundaries
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "plan.json"

    result = build_long_context_plan(
        processor_report_path=report,
        train_jsonl=train,
        sft_profile_paths=list(PROFILES.values()),
        output=output,
    )

    assert result["passed_static_gate"] is True
    assert result["parallelism"]["data_parallel_size"] == 1
    assert result["schedule"]["optimizer_steps_total"] == 8490
    assert result["memory_budget"]["fit_proven"] is False
    assert result["throughput_forecast"]["forecast_available"] is False


def test_128k_stage_gate_enforces_prior_stage_and_same_data(tmp_path: Path) -> None:
    memory_profile = tmp_path / "memory-profile.json"
    memory_profile.write_text(json.dumps(_training_profile("memory-probe")), encoding="utf-8")
    memory_gate = tmp_path / "memory-gate.json"
    first = validate_128k_stage(
        stage="memory-probe",
        training_profile_path=memory_profile,
        sft_profile_path=PROFILES["memory-probe"],
        output=memory_gate,
    )
    assert first["passed"] is True

    canary_profile = tmp_path / "canary-profile.json"
    canary_profile.write_text(json.dumps(_training_profile("canary")), encoding="utf-8")
    canary_gate = tmp_path / "canary-gate.json"
    second = validate_128k_stage(
        stage="canary",
        training_profile_path=canary_profile,
        sft_profile_path=PROFILES["canary"],
        prior_gate_path=memory_gate,
        output=canary_gate,
    )
    assert second["passed"] is True

    drifted = _training_profile("canary")
    drifted["raw_dataset_gate"]["verification"]["datasets"]["train"]["sha256"] = "d" * 64
    canary_profile.write_text(json.dumps(drifted), encoding="utf-8")
    rejected = validate_128k_stage(
        stage="canary",
        training_profile_path=canary_profile,
        sft_profile_path=PROFILES["canary"],
        prior_gate_path=memory_gate,
        output=canary_gate,
    )
    assert rejected["passed"] is False
    assert rejected["checks"]["same_data_lineage"]["passed"] is False
