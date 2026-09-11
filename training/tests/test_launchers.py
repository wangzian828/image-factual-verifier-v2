from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _source(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_training_launchers_delegate_to_ms_swift() -> None:
    single = _source("scripts/train/run_sft.sh")

    assert "swift sft" in single
    assert "Trainer" not in single
    assert "torchrun" not in single
    assert "--resume_from_checkpoint" in single
    assert "configure_distributed_backend" in single
    assert 'training_backend_args+=(--deepspeed "$IFV_DEEPSPEED")' in single
    assert 'training_backend_args+=(--fsdp "$IFV_FSDP")' in single
    assert "verify_cached_dataset_gate" in single
    assert "verify_sft_data_contract.py" in single
    assert "IFV_PROCESSOR_VERIFICATION" in single
    assert "raw-dataset-gate.json" in single
    assert "run_with_resource_monitor.py" in single
    assert "watch_sft.py" in single
    assert "audit_deepspeed_scheduler.py" in single
    assert "checkpoint-storage-preflight" in single
    assert "checkpoint-io-profile" in single
    assert "IFV_GPU_MEMORY_TARGET_MIN_MIB" in single
    assert "IFV_GPU_MEMORY_TARGET_MAX_MIB" in single
    assert "IFV_GPU_MEMORY_MAX_IMBALANCE_MIB" in single
    assert "IFV_GPU_UTILIZATION_TARGET_MIN_PERCENT" in single
    assert "--utilization-target-min-percent" in single
    assert "verify_qwen35_sft_environment.py" in single
    assert "environment-preflight.json" in single
    assert "IFV_REQUIRE_TRAINING_ENV_PREFLIGHT" in single
    assert 'SAVE_STRATEGY="${IFV_SAVE_STRATEGY:-steps}"' in single
    assert 'if [[ "$SAVE_STRATEGY" != "no" ]]' in single
    assert 'args+=(--enable_thinking "$IFV_ENABLE_THINKING")' in single
    assert (
        'args+=(--add_non_thinking_prefix '
        '"${IFV_ADD_NON_THINKING_PREFIX:-false}")' in single
    )


def test_teacher_pipeline_binds_processor_probe_to_training_profile() -> None:
    pipeline = (ROOT.parent / "scripts/server/run_teacher_sft_pipeline.sh").read_text(
        encoding="utf-8"
    )

    assert 'source "${training_profile}"' in pipeline
    for option in (
        "--max-context",
        "--max-pixels",
        "--truncation-strategy",
        "--padding-free",
        "--sequence-parallel-size",
        "--loss-scale",
        "--enable-thinking",
        "--add-non-thinking-prefix",
        "--image-max-token-num",
    ):
        assert option in pipeline


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
    memory_probe = _source(
        "configs/sft/qwen3.5-full-1step-8gpu-fsdp2-sp8-flash-"
        "128k-memory-probe.env"
    )
    canary_128k = _source(
        "configs/sft/qwen3.5-full-10step-8gpu-fsdp2-sp8-flash-"
        "128k-canary.env"
    )
    resume_128k = _source(
        "configs/sft/qwen3.5-full-11step-8gpu-fsdp2-sp8-flash-"
        "128k-resume.env"
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

    for profile in (memory_probe, canary_128k, resume_128k):
        assert "IFV_TUNER_TYPE=full" in profile
        assert "IFV_FSDP=fsdp2" in profile
        assert "IFV_FSDP_VERSION=2" in profile
        assert "IFV_ATTN_IMPL=flash_attn" in profile
        assert "IFV_MAX_LENGTH=131072" in profile
        assert "IFV_TRUNCATION_STRATEGY=raise" in profile
        assert "IFV_MIN_PROCESSOR_TRAIN_INPUT_TOKENS=120000" in profile
        assert "IFV_MIN_PROCESSOR_TRAIN_ROWS_AT_OR_ABOVE=1" in profile
        assert "IFV_PADDING_FREE=true" in profile
        assert "IFV_SEQUENCE_PARALLEL_SIZE=8" in profile
        assert "IFV_USE_LOGITS_TO_KEEP=false" in profile
        assert "CELOSS_PARALLEL_SIZE=2048" in profile
        assert "IFV_GRADIENT_CHECKPOINTING=true" in profile
        assert "IFV_REQUIRE_TRAINING_ENV_PREFLIGHT=true" in profile
        assert "IFV_EXPECTED_GPU_COUNT=8" in profile
        assert "IFV_GPU_MEMORY_TARGET_MIN_MIB=36000" in profile
        assert "IFV_GPU_MEMORY_TARGET_MAX_MIB=38912" in profile
        assert "IFV_GPU_MEMORY_MAX_IMBALANCE_MIB=1024" in profile

    assert "IFV_MAX_STEPS=1" in memory_probe
    assert "IFV_EVAL_STRATEGY=no" in memory_probe
    assert "IFV_SAVE_STRATEGY=no" in memory_probe
    assert "IFV_MAX_STEPS=10" in canary_128k
    assert "IFV_SAVE_STEPS=10" in canary_128k
    assert "IFV_EVAL_STEPS=10" in canary_128k
    assert "IFV_MAX_STEPS=11" in resume_128k
    assert "IFV_SAVE_STEPS=11" in resume_128k
    assert "IFV_EVAL_STEPS=11" in resume_128k

    acceptance = _source("scripts/train/run_128k_acceptance.sh")
    assert "long-context-plan" in acceptance
    assert "validate-128k-stage" in acceptance
    assert "checkpoint-10" in acceptance
    assert "Existing incomplete" in acceptance


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
    assert "verify-psd-datums" in psd
    assert "psd-input-gate.json" in psd
    assert "psd_ms_swift_plugin_smoke.py" in psd
    assert "environment-preflight.json" in psd
    assert "checkpoint-storage-preflight" in psd
    assert "--resume_from_checkpoint" in psd
    assert "run_with_resource_monitor.py" in psd
    assert "Trainer" not in psd

    assert "swift rlhf" in grpo
    assert "--multi_turn_scheduler gym_scheduler" in grpo
    assert "--use_gym_env true" in grpo
    assert "--reward_funcs" not in grpo
    assert "IFV_RL_FRAMEWORK=ms_swift_grpo" in rl_profile
    assert "IFV_MS_SWIFT_VERSION=4.4.2" in rl_profile
    assert "vllm==0.22.1" in requirements


def test_psd_repair_binds_the_served_round_start_checkpoint() -> None:
    driver = (ROOT.parent / "scripts/run_psd_repair_driver.py").read_text(
        encoding="utf-8"
    )
    serving = _source("scripts/serve/start_vllm_qwen35.sh")

    assert "--policy-serving-profile" in driver
    assert "--round-start-checkpoint-manifest" in driver
    assert "checkpoint_manifest_sha256" in driver
    assert "round-start checkpoint does not match served model path" in driver
    assert "IFV_CHECKPOINT_MANIFEST" in serving
    assert '--checkpoint-manifest "$CHECKPOINT_MANIFEST"' in serving


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
    assert "IFV_DATA_PARALLEL_SIZE" in common
    assert "must be divisible by IFV_SEQUENCE_PARALLEL_SIZE" in common
    assert "sequence parallel training requires IFV_PADDING_FREE=true" in common
    assert "sequence parallel training requires a positive CELOSS_PARALLEL_SIZE" in common
    assert "$1 + 0 >= 4 && $1 + 0 <= 7" in selector
    assert "measure_gpu_io.py" in diagnose
    assert "physical_gpu" in measure
    assert "pin_memory=True" in measure


def test_cache_and_checkpoint_tools_remain_audited() -> None:
    encode_cache = _source("ifv_training/encode_cache.py")
    bootstrap = _source("ifv_training_bootstrap/sitecustomize.py")
    prewarm = _source("scripts/train/prewarm_encode_cache.py")
    registrar = _source("scripts/train/register_cached_dataset.sh")
    fsdp_export = _source("scripts/export/register_fsdp2_checkpoint.sh")
    full_export = _source("scripts/export/register_full_checkpoint.sh")

    assert 'SCHEMA_VERSION = "ifv-ms-swift-encode-cache-v1"' in encode_cache
    assert "contract_digest" in encode_cache
    assert "install_ms_swift_encode_cache" in bootstrap
    assert "os._exit(70)" in bootstrap
    assert "torch.equal" in prewarm

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
    assert "sft-long" in source
    assert "IFV_QWEN35_SFT_LONG_ENV_PREFIX" in source
    assert "python=3.12" in source
    assert "refusing to modify an existing environment" in source
    assert "ifv-qwen35-sft-ms-swift442" in source
    assert "ifv-qwen35-rl-ms-swift442-vllm0221" in source
    assert "pip freeze --all" in source
    assert "Qwen3_5ForConditionalGeneration" in source
    assert "verify_qwen35_sft_environment.py" in source
    assert "FLASH_ATTENTION_FORCE_BUILD=TRUE" in source
    assert "CAUSAL_CONV1D_FORCE_BUILD=TRUE" in source
    assert "install_cuda_extensions()" in source
    assert 'install_cuda_extensions "$LONG_SFT_PREFIX"' in source
    assert 'install_cuda_extensions "$RL_PREFIX"' in source
    assert 'CPATH="$cuda_target/include${CPATH:+:$CPATH}"' in source
    assert 'LIBRARY_PATH="$cuda_target/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"' in source
    assert "PIP_NO_CACHE_DIR=1" in source
    assert "--no-cache-dir" in source
    assert "--force-reinstall" in source
    assert "--no-binary=:all:" in source
    assert 'TORCH_CUDA_ARCH_LIST="${IFV_TORCH_CUDA_ARCH_LIST:-8.0}"' in source
    assert 'FLASH_ATTN_CUDA_ARCHS="${IFV_FLASH_ATTN_CUDA_ARCHS:-80}"' in source
    assert 'MAX_JOBS="${IFV_EXTENSION_MAX_JOBS:-16}"' in source
    assert 'NVCC_THREADS="${IFV_EXTENSION_NVCC_THREADS:-2}"' in source
    assert "ifv-cuda-extension-wheel-cache-v1" in source
    assert "abi_fingerprint" in source
    assert "sha256sum -c SHA256SUMS" in source
    assert "refusing to replace an incomplete or corrupt CUDA wheel cache" in source
    assert "python\" -m pip wheel" in source
    assert '--find-links "$wheelhouse"' in source
    assert "import causal_conv1d_cuda" in source
    assert "import flash_attn_2_cuda" in source
    assert "IFV_RESUME_INCOMPLETE_ENV" in source
    assert "refusing to resume an unmarked or invalid environment" in source
    assert "ms-swift==4.4.2" in sft
    assert "transformers==5.12.1" in sft
    assert "liger-kernel==0.8.0" in sft
    assert "vllm==0.22.1" in rl
    assert "openai==2.32.0" in rl
    assert "jiter==0.14.0" in rl
    assert "verify_qwen35_rl_environment.py" in source
    assert 'rm -f "$RL_PREFIX/.ifv-qwen35-rl-ready"' in source
    assert 'sha256sum "$rl_preflight"' in source
