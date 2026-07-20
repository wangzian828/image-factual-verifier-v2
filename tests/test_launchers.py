from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _source(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_sft_launchers_delegate_training_to_ms_swift() -> None:
    single = _source("scripts/train/run_sft.sh")
    curriculum = _source("scripts/train/run_curriculum_sft.sh")

    for source in (single, curriculum):
        assert "swift sft" in source
        assert "Trainer" not in source
        assert "torchrun" not in source
        assert "NPROC_PER_NODE=2" not in source
        assert "--resume_from_checkpoint" in source
        assert "--deepspeed" in source
    assert "--interleave_prob 0.20 0.15 0.35 0.20 0.05 0.05" in curriculum
    assert "train.group-decision.jsonl" in curriculum


def test_grpo_launcher_uses_framework_gym_reward() -> None:
    source = _source("scripts/rl/run_mock_grpo.sh")

    assert "swift rlhf" in source
    assert "--rlhf_type" in source
    assert "--multi_turn_scheduler gym_scheduler" in source
    assert "--use_gym_env true" in source
    assert "--reward_funcs" not in source


def test_serving_launcher_uses_qwen35_vllm_protocol() -> None:
    source = _source("scripts/serve/start_vllm.sh")

    assert "vllm serve" in source
    assert "--tensor-parallel-size" in source
    assert "--max-model-len" in source
    assert "--tool-call-parser qwen3_coder" in source
    assert "--reasoning-parser qwen3" in source
    assert "require_idle_gpus" in source


def test_launchers_enforce_physical_gpu_allowlist() -> None:
    common = _source("scripts/lib/common.sh")
    selector = _source("scripts/server/select_idle_gpus.sh")

    assert 'IFV_ALLOWED_GPU_IDS:-4,5,6,7' in common
    assert "outside the allowed physical GPU set" in common
    assert "$1 + 0 >= 4 && $1 + 0 <= 7" in selector


def test_qwen35_quick_and_primary_profiles_exist() -> None:
    quick = _source("configs/models/qwen3.5-4b.env")
    primary = _source("configs/models/qwen3.5-9b.env")

    assert "Qwen3.5-4B" in quick
    assert "quick_bootstrap" in quick
    assert "Qwen3.5-9B" in primary
    assert "primary_student" in primary
