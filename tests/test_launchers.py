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
    assert '--enable_thinking "$IFV_ENABLE_THINKING"' in source


def test_rl_stack_is_pinned_and_isolated_from_serving() -> None:
    profile = _source("configs/rl/qwen3.5-grpo-mock.env")
    requirements = _source("requirements/rl-qwen35.txt")

    assert "IFV_RL_FRAMEWORK=ms_swift_grpo" in profile
    assert "IFV_MS_SWIFT_VERSION=4.4.2" in profile
    assert "IFV_ENABLE_THINKING=false" in profile
    assert "vllm==0.22.1" in requirements


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
    assert '"disable_any_whitespace":true' in source
    assert "--tool-call-parser qwen3_xml" in source
    assert '"image":1,"video":0' in source
    assert "--enable-tokenizer-info-endpoint" in source
    assert "require_idle_gpus" in source
    assert "131072" in source
    assert "NCCL_CUMEM_HOST_ENABLE=0" in source
    assert "--disable-custom-all-reduce" in source
    assert "prepare_qwen3vl_chat_template.py" in source
    assert '--chat-template "$CHAT_TEMPLATE"' in source


def test_qwen35_launcher_uses_native_multimodal_hybrid_thinking_protocol() -> None:
    source = _source("scripts/serve/start_vllm_qwen35.sh")

    assert 'max-model-len "$CONTEXT_LENGTH"' in source
    assert "--reasoning-parser qwen3" in source
    assert '"reasoning_parser":"qwen3"' in source
    assert '"backend":"xgrammar"' in source
    assert "--tool-call-parser qwen3_coder" in source
    assert "--default-chat-template-kwargs" in source
    assert '"enable_thinking":false' in source
    assert '"image":1,"video":0' in source
    assert "--enable-tokenizer-info-endpoint" in source
    assert "require_idle_gpus" in source
    assert "131072" in source
    assert "NCCL_CUMEM_HOST_ENABLE=0" in source
    assert "--disable-custom-all-reduce" in source
    assert "prepare_qwen3vl_chat_template.py" not in source


def test_qwen35_lifecycle_only_stops_its_verified_process_group() -> None:
    source = _source("scripts/serve/manage_vllm_qwen35.sh")

    assert "owned_process" in source
    assert 'kill -TERM -- "-$pid"' in source
    assert "no SIGKILL was sent" in source
    assert "setsid bash" in source
    assert "120" in source


def test_qwen35_freeze_gate_records_environment_nccl_and_model_hashes() -> None:
    source = _source("scripts/server/freeze_vllm_qwen35_gpu13.sh")

    assert "verify_vllm_qwen35_environment.py" in source
    assert "verify_nccl_tensor_parallel.py" in source
    assert "pip freeze --all" in source
    assert "model-files.sha256" in source
    assert 'data.get("weight_map", {}).values()' in source
    assert '"$MODEL/$shard"' in source
    assert ".ifv-vllm-qwen35-ready" in source


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
    resolved = _source("requirements/constraints-vllm-qwen3vl-resolved.txt")

    assert "conda create -y -p" in source
    assert "--clone" not in source
    assert 'if [[ -e "$ENV_PREFIX" ]]' in source
    assert "pip check" in source
    assert '--constraint "$CONSTRAINTS"' in source
    assert '--constraint "$RESOLVED_CONSTRAINTS"' in source
    assert '--requirement "$REQUIREMENTS"' in source
    assert "verify_vllm_environment.py" in source
    assert "verify_nccl_tensor_parallel.py" in source
    assert 'NCCL_CUMEM_HOST_ENABLE=0' in source
    assert '"$ENV_PREFIX/bin/torchrun"' in source
    assert "pip freeze --all" in source
    assert "input-locks.sha256" in source
    assert "vllm==0.11.2" in requirements
    assert "transformers==4.57.6" in requirements
    assert "torch==2.9.0" in constraints
    assert "xgrammar==0.1.25" in constraints
    assert "nvidia-nccl-cu12==2.27.5" in resolved
    assert "ray==2.56.1" in resolved
    assert "fastapi==0.139.2" in resolved
    assert "@ file:" not in resolved


def test_qwen35_is_the_primary_training_profile() -> None:
    primary = _source("configs/models/qwen3.5-9b.env")

    assert "Qwen3.5-9B" in primary
    assert "primary_student" in primary
    assert "IFV_ENABLE_THINKING=false" in primary
    assert "IFV_ADD_NON_THINKING_PREFIX=true" in primary


def test_qwen35_full_parameter_step_profiles_exist() -> None:
    for steps in (1, 3, 20):
        source = _source(f"configs/sft/qwen3.5-full-{steps}step.env")
        assert f"IFV_MAX_STEPS={steps}" in source
        assert "IFV_TUNER_TYPE=full" in source
        assert "IFV_FREEZE_LLM=false" in source
        assert "IFV_FREEZE_VIT=false" in source
        assert "IFV_FREEZE_ALIGNER=false" in source


def test_qwen35_training_bootstrap_is_fresh_pinned_and_isolated() -> None:
    source = _source("scripts/server/bootstrap_qwen35_training_gpu13.sh")
    sft = _source("requirements/train-qwen35.txt")
    rl = _source("requirements/rl-qwen35.txt")

    assert 'MODE="${1:-all}"' in source
    assert "python=3.12" in source
    assert "pip=25.2" in source
    assert 'prepare_base "$SFT_PREFIX" 12.8.93' in source
    assert 'prepare_base "$RL_PREFIX" 13.0.88' in source
    assert '"cuda-nvcc=$cuda_nvcc_version"' in source
    assert 'CUDA_HOME="$SFT_PREFIX"' in source
    assert "refusing to modify an existing environment" in source
    assert "ifv-qwen35-sft-ms-swift442" in source
    assert "ifv-qwen35-rl-ms-swift442-vllm0221" in source
    assert "pip freeze --all" in source
    assert "get_model_processor" in source
    assert 'enable_thinking=False' in source
    assert "ms-swift==4.4.2" in sft
    assert "transformers==5.12.1" in sft
    assert "flash-linear-attention==0.5.1" in sft
    assert "causal-conv1d==1.6.2.post1" in sft
    assert "vllm==0.22.1" in rl
