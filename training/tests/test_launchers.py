from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _source(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_training_launchers_delegate_to_ms_swift() -> None:
    single = _source("scripts/train/run_sft.sh")
    curriculum = _source("scripts/train/run_curriculum_sft.sh")

    for source in (single, curriculum):
        assert "swift sft" in source
        assert "Trainer" not in source
        assert "torchrun" not in source
        assert "--resume_from_checkpoint" in source
        assert "configure_distributed_backend" in source
        assert 'training_backend_args+=(--deepspeed "$IFV_DEEPSPEED")' in source
        assert 'training_backend_args+=(--fsdp "$IFV_FSDP")' in source
        assert "verify_cached_dataset_gate" in source
        assert "run_with_resource_monitor.py" in source
        assert "watch_sft.py" in source
        assert "audit_deepspeed_scheduler.py" in source
        assert "checkpoint-storage-preflight" in source
        assert "checkpoint-io-profile" in source

    assert "candidate_weights=(0.20 0.15 0.35 0.20 0.05 0.05)" in curriculum
    assert "Skipping absent curriculum channel" in curriculum
    assert '--interleave_prob "${interleave_prob[@]}"' in curriculum


def test_current_qwen35_profiles_are_explicit_and_bounded() -> None:
    model = _source("configs/models/qwen3.5-9b.env")
    production = _source(
        "configs/sft/qwen3.5-full-10step-4gpu-zero3-offload-accum1-"
        "bf16params-sdpa-checkpointed-16k-logits-512px.env"
    )
    portable = _source(
        "configs/sft/qwen3.5-full-10step-4gpu-zero3-offload-accum1-"
        "bf16params-sdpa-checkpointed-8k.env"
    )
    smoke = _source(
        "configs/sft/qwen3.5-full-10step-4gpu-zero3-offload-accum1-"
        "bf16params-sdpa-checkpointed-8k-truncated-smoke-noeval.env"
    )

    assert "IFV_QWEN35_MODEL" in model
    assert "IFV_MODEL_ROLE=primary_student" in model
    assert "IFV_ENABLE_THINKING=false" in model
    assert "IFV_ADD_NON_THINKING_PREFIX=false" in model

    for profile in (production, portable, smoke):
        assert "IFV_TUNER_TYPE=full" in profile
        assert "IFV_MAX_STEPS=10" in profile
        assert "IFV_DEEPSPEED=training/configs/deepspeed/zero3-optimizer-offload.json" in profile
        assert "IFV_GRADIENT_CHECKPOINTING=true" in profile
        assert "IFV_PADDING_FREE=false" in profile
        assert "IFV_USE_LOGITS_TO_KEEP=true" in profile

    assert "IFV_MAX_LENGTH=16384" in production
    assert "IFV_MAX_PIXELS=262144" in production
    assert "IFV_MAX_LENGTH=8192" in portable
    assert "IFV_EVAL_STRATEGY=no" in smoke
    assert "IFV_TRUNCATION_STRATEGY=left" in smoke


def test_deepspeed_optimizer_offload_keeps_model_parameters_on_gpu() -> None:
    config = _source("configs/deepspeed/zero3-optimizer-offload.json")
    common = _source("scripts/lib/common.sh")

    assert '"offload_optimizer"' in config
    assert '"device": "cpu"' in config
    assert '"offload_param"' in config
    assert '"device": "none"' in config
    assert 'elif [[ -f "$config" ]]' in common
    assert "do not set both IFV_DEEPSPEED and IFV_FSDP" in common


def test_psd_and_grpo_use_framework_entrypoints() -> None:
    psd = _source("scripts/train/run_psd_topk.sh")
    grpo = _source("scripts/rl/run_mock_grpo.sh")
    rl_profile = _source("configs/rl/qwen3.5-grpo-mock.env")
    requirements = _source("requirements/rl-qwen35.txt")

    assert "swift sft" in psd
    assert "--external_plugins" in psd
    assert "ifv_psd_topk_plugin.py" in psd
    assert "--loss_type ifv_psd_topk" in psd
    assert "Trainer" not in psd

    assert "swift rlhf" in grpo
    assert "--multi_turn_scheduler gym_scheduler" in grpo
    assert "--use_gym_env true" in grpo
    assert "--reward_funcs" not in grpo
    assert "IFV_RL_FRAMEWORK=ms_swift_grpo" in rl_profile
    assert "IFV_MS_SWIFT_VERSION=4.4.2" in rl_profile
    assert "vllm==0.22.1" in requirements


def test_qwen35_serving_and_lifecycle_are_fail_closed() -> None:
    launcher = _source("scripts/serve/start_vllm_qwen35.sh")
    manager = _source("scripts/serve/manage_vllm_qwen35.sh")
    freeze = _source("scripts/server/freeze_vllm_qwen35_gpu13.sh")

    assert '--reasoning-parser qwen3' in launcher
    assert '--tool-call-parser qwen3_coder' in launcher
    assert '--default-chat-template-kwargs' in launcher
    assert '"enable_thinking":false' in launcher
    assert '--limit-mm-per-prompt' in launcher
    assert "require_idle_runtime_gpus" in launcher
    assert "NCCL_CUMEM_HOST_ENABLE=0" in launcher

    assert "owned_process" in manager
    assert 'kill -TERM -- "-$pid"' in manager
    assert "no SIGKILL was sent" in manager
    assert "setsid bash" in manager

    assert "verify_vllm_qwen35_environment.py" in freeze
    assert "verify_nccl_tensor_parallel.py" in freeze
    assert "pip freeze --all" in freeze
    assert "model-files.sha256" in freeze
    assert ".ifv-vllm-qwen35-ready" in freeze


def test_gpu_runtime_policy_is_shared_by_serving_and_training() -> None:
    common = _source("scripts/lib/common.sh")
    selector = _source("scripts/server/select_idle_gpus.sh")
    diagnose = _source("scripts/server/diagnose_gpu_io.sh")
    measure = _source("scripts/server/measure_gpu_io.py")

    assert 'ALLOWED_GPU_IDS="${IFV_ALLOWED_GPU_IDS:-}"' in common
    assert "outside the allowed physical GPU set" in common
    assert "server policy requires IFV_OMP_NUM_THREADS=1" in common
    assert "export OMP_NUM_THREADS=1" in common
    assert "NCCL_CUMEM_HOST_ENABLE" in common
    assert "between one and eight GPUs" in common
    assert "$1 + 0 >= 4 && $1 + 0 <= 7" in selector
    assert "measure_gpu_io.py" in diagnose
    assert "physical_gpu" in measure
    assert "pin_memory=True" in measure


def test_cache_and_checkpoint_tools_remain_audited() -> None:
    encode_cache = _source("ifv_training/encode_cache.py")
    bootstrap = _source("ifv_training_bootstrap/sitecustomize.py")
    prewarm = _source("scripts/train/prewarm_encode_cache.py")
    cache_export = _source("scripts/train/cache_curriculum_sft.sh")
    registrar = _source("scripts/train/register_cached_dataset.sh")
    fsdp_export = _source("scripts/export/register_fsdp2_checkpoint.sh")
    full_export = _source("scripts/export/register_full_checkpoint.sh")

    assert 'SCHEMA_VERSION = "ifv-ms-swift-encode-cache-v1"' in encode_cache
    assert "contract_digest" in encode_cache
    assert "install_ms_swift_encode_cache" in bootstrap
    assert "os._exit(70)" in bootstrap
    assert "torch.equal" in prewarm

    assert "swift export" in cache_export
    assert "source-dataset-fingerprints.tsv" in cache_export
    assert "cached-dataset-manifest" in cache_export
    assert "refusing to replace existing cached dataset manifest" in registrar
    assert '"model_profile"' in registrar
    assert '"sft_profile"' in registrar
    assert '"sha256"' in registrar

    assert "merge_fsdp_weights" in fsdp_export
    assert "audit-full-checkpoint" in fsdp_export
    assert "checkpoint-manifest" in fsdp_export
    assert "load_qwen35_checkpoint.py" in fsdp_export
    assert "audit-full-checkpoint" in full_export
    assert "checkpoint-manifest" in full_export
    assert "load_qwen35_checkpoint.py" in full_export


def test_scheduler_warning_has_a_persisted_audit_gate() -> None:
    source = _source("scripts/probe/audit_deepspeed_scheduler.py")
    profile = _source("ifv_training/profile.py")

    assert "DeepSpeedEngineWrapper.backward" in source
    assert "DeepSpeedOptimizerWrapper.step" in source
    assert "scheduler_last_epoch_matches_global_step" in source
    assert "deepspeed_skipped_steps_zero" in source
    assert "scheduler_audit_passed" in profile


def test_qwen35_training_bootstrap_is_fresh_pinned_and_isolated() -> None:
    source = _source("scripts/server/bootstrap_qwen35_training_gpu13.sh")
    sft = _source("requirements/train-qwen35.txt")
    rl = _source("requirements/rl-qwen35.txt")

    assert 'MODE="${1:-all}"' in source
    assert "python=3.12" in source
    assert "refusing to modify an existing environment" in source
    assert "ifv-qwen35-sft-ms-swift442" in source
    assert "ifv-qwen35-rl-ms-swift442-vllm0221" in source
    assert "pip freeze --all" in source
    assert "Qwen3_5ForConditionalGeneration" in source
    assert "ms-swift==4.4.2" in sft
    assert "transformers==5.12.1" in sft
    assert "vllm==0.22.1" in rl
