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
    assert "--interleave_prob 0.25 0.10 0.45 0.15 0.05" in curriculum


def test_grpo_launcher_uses_framework_gym_reward() -> None:
    source = _source("scripts/rl/run_mock_grpo.sh")

    assert "swift rlhf" in source
    assert "--rlhf_type" in source
    assert "--multi_turn_scheduler gym_scheduler" in source
    assert "--use_gym_env true" in source
    assert "--reward_funcs" not in source


def test_serving_launcher_uses_vllm_names_from_ms_swift_4_4() -> None:
    source = _source("scripts/serve/start_vllm.sh")

    assert "swift deploy" in source
    assert "--vllm_tensor_parallel_size" in source
    assert "--vllm_max_model_len" in source
    assert "\n  --tensor_parallel_size" not in source
