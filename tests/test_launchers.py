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


def test_rl_stack_is_pinned_and_isolated_from_serving() -> None:
    profile = _source("configs/rl/qwen3-vl-grpo-mock.env")
    requirements = _source("requirements/rl.txt")

    assert "IFV_RL_FRAMEWORK=rllm_verl" in profile
    assert "IFV_RLLM_REVISION=cd9ea0" in profile
    assert "IFV_VERL_REVISION=6a6242" in profile
    assert "rllm[verl] @ git+https://" in requirements


def test_serving_launcher_uses_qwen3_vl_lmdeploy_protocol() -> None:
    source = _source("scripts/serve/start_lmdeploy.sh")

    assert "lmdeploy serve api_server" in source
    assert "--backend pytorch" in source
    assert 'IFV_LMDEPLOY_TOOL_CALL_PARSER:-qwen3' in source
    assert 'SERVING_ROLE="${6:-agent}"' in source
    assert "--logprobs-mode raw_logprobs" in source
    assert "--distributed-executor-backend" in source
    assert "--session-len" in source
    assert "require_idle_gpus" in source


def test_locked_vllm_launcher_uses_native_qwen3_vl_protocol() -> None:
    source = _source("scripts/serve/start_vllm_qwen3vl.sh")

    assert 'max-model-len "$CONTEXT_LENGTH"' in source
    assert "--reasoning-parser qwen3" in source
    assert '"reasoning_parser":"qwen3"' in source
    assert '"backend":"xgrammar"' in source
    assert "--tool-call-parser qwen3_xml" in source
    assert '"image":1,"video":0' in source
    assert "--enable-tokenizer-info-endpoint" in source
    assert "require_idle_gpus" in source
    assert "131072" in source


def test_vllm_lifecycle_only_stops_its_verified_process_group() -> None:
    source = _source("scripts/serve/manage_vllm_qwen3vl.sh")

    assert "owned_process" in source
    assert 'kill -TERM -- "-$pid"' in source
    assert "SIGKILL was sent" in source
    assert "setsid bash" in source
    assert "120" in source


def test_launchers_enforce_physical_gpu_allowlist() -> None:
    common = _source("scripts/lib/common.sh")
    selector = _source("scripts/server/select_idle_gpus.sh")

    assert 'IFV_ALLOWED_GPU_IDS:-4,5,6,7' in common
    assert "outside the allowed physical GPU set" in common
    assert "$1 + 0 >= 4 && $1 + 0 <= 7" in selector


def test_gpu13_bootstrap_isolates_serving_from_sft_installation() -> None:
    source = _source("scripts/server/bootstrap_gpu13.sh")

    assert 'MODE="${1:-all}"' in source
    assert '"serve" || "$MODE" == "all"' in source
    assert '"sft" || "$MODE" == "all"' in source
    assert "ifv-qwen3vl-serve" in source
    assert "ifv-qwen3vl-sft" in source
    assert '"$env_prefix/bin/python"' in source
    assert 'conda run -n "$name" python' not in source
    assert "pip uninstall -y lmdeploy vllm sglang" in source
    assert '"cuda-nvcc=$cuda_nvcc_version"' in source
    assert 'CUDA_HOME="$sft_prefix"' in source
    assert '"$env_prefix/bin/python" -m pip check' in source
    assert '"$sft_prefix/bin/swift" sft --help' in source


def test_vllm_bootstrap_is_fresh_pinned_and_refuses_existing_prefix() -> None:
    source = _source("scripts/server/bootstrap_vllm_qwen3vl_gpu13.sh")
    requirements = _source("requirements/serve-vllm-qwen3vl.txt")
    constraints = _source("requirements/constraints-vllm-qwen3vl.txt")

    assert "conda create -y -p" in source
    assert "--clone" not in source
    assert 'if [[ -e "$ENV_PREFIX" ]]' in source
    assert "pip check" in source
    assert 'pip install --constraint "$CONSTRAINTS" --requirement "$REQUIREMENTS"' in source
    assert "verify_vllm_environment.py" in source
    assert "pip freeze --all" in source
    assert "input-locks.sha256" in source
    assert "vllm==0.11.2" in requirements
    assert "transformers==4.57.6" in requirements
    assert "torch==2.9.0" in constraints
    assert "xgrammar==0.1.25" in constraints


def test_qwen3_vl_thinking_is_the_only_primary_profile() -> None:
    primary = _source("configs/models/qwen3-vl-8b-thinking.env")

    assert "Qwen3-VL-8B-Thinking" in primary
    assert "primary_student" in primary
    assert not list((ROOT / "configs/models").glob("qwen3.5-*.env"))


def test_qwen3_vl_full_parameter_step_profiles_exist() -> None:
    for steps in (1, 3, 20):
        source = _source(f"configs/sft/qwen3-vl-full-{steps}step.env")
        assert f"IFV_MAX_STEPS={steps}" in source
        assert "IFV_TUNER_TYPE=full" in source
        assert "IFV_FREEZE_LLM=false" in source
        assert "IFV_FREEZE_VIT=false" in source
        assert "IFV_FREEZE_ALIGNER=false" in source
