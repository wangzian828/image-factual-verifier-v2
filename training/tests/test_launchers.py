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
    assert "candidate_weights=(0.20 0.15 0.35 0.20 0.05 0.05)" in curriculum
    assert 'Skipping absent curriculum channel' in curriculum
    assert '--interleave_prob "${interleave_prob[@]}"' in curriculum
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
    assert "require_idle_runtime_gpus" in source


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
    assert "require_idle_runtime_gpus" in source
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
    assert "require_idle_runtime_gpus" in source
    assert "131072" in source
    assert "NCCL_CUMEM_HOST_ENABLE=0" in source
    assert "--disable-custom-all-reduce" in source
    assert "VLLM_USE_FLASHINFER_SAMPLER=0" in source
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
    diagnose = _source("scripts/server/diagnose_gpu_io.sh")
    measure = _source("scripts/server/measure_gpu_io.py")

    assert 'IFV_ALLOWED_GPU_IDS:-4,5,6,7' in common
    assert 'configure_training_runtime' in common
    assert 'configure_cuda_toolkit' in common
    assert 'configure_conda_compilers' in common
    assert 'prepare_deepspeed_cpu_adam' in common
    assert 'libcurand.so' in common
    assert 'IFV_CUDA_HOME' in common
    assert 'x86_64-conda-linux-gnu-g++' in common
    assert 'TORCH_EXTENSIONS_DIR' in common
    assert 'HF_DATASETS_CACHE' in common
    assert 'OMP_NUM_THREADS="${IFV_OMP_NUM_THREADS:-1}"' in common
    assert 'NCCL_CUMEM_HOST_ENABLE' in common
    assert 'MPLBACKEND=Agg' in common
    assert "between one and eight GPUs" in common
    assert "outside the allowed physical GPU set" in common
    assert "$1 + 0 >= 4 && $1 + 0 <= 7" in selector
    assert "measure_gpu_io.py" in diagnose
    assert 'GPU_LIST="${2:-visible}"' in diagnose
    assert "nvidia-smi topo -m" in diagnose
    assert "cpu-affinity.txt" in diagnose
    assert "ifv-gpu-io-diagnostic-v1" in measure
    assert "pin_memory=True" in measure


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
        assert "IFV_MAX_LENGTH=32768" in source
        assert "IFV_IMAGE_MAX_TOKEN_NUM=1024" in source
        if steps in (3, 20):
            assert (
                "IFV_DEEPSPEED=training/configs/deepspeed/zero3-optimizer-offload.json"
                in source
            )


def test_qwen35_pilot30_profile_matches_frozen_optimizer_budget() -> None:
    source = _source("configs/sft/qwen3.5-full-pilot30.env")

    assert "IFV_MAX_STEPS=99" in source
    assert "IFV_TRAIN_BATCH_SIZE=1" in source
    assert "IFV_GRADIENT_ACCUMULATION_STEPS=2" in source
    assert "IFV_SAVE_STEPS=25" in source
    assert "IFV_GROUP_BY_LENGTH=true" in source
    assert "IFV_MAX_LENGTH=32768" in source
    assert "IFV_IMAGE_MAX_TOKEN_NUM=1024" in source
    assert "IFV_TUNER_TYPE=full" in source
    assert (
        "IFV_DEEPSPEED=training/configs/deepspeed/zero3-optimizer-offload.json"
        in source
    )


def test_qwen35_pilot30_two_gpu_profile_preserves_global_batch() -> None:
    source = _source("configs/sft/qwen3.5-full-pilot30-2gpu.env")

    assert "IFV_MAX_STEPS=99" in source
    assert "IFV_TRAIN_BATCH_SIZE=1" in source
    assert "IFV_GRADIENT_ACCUMULATION_STEPS=4" in source
    assert "IFV_SAVE_STEPS=25" in source
    assert "IFV_GROUP_BY_LENGTH=true" in source
    assert "IFV_MAX_LENGTH=32768" in source
    assert "IFV_IMAGE_MAX_TOKEN_NUM=1024" in source
    assert (
        "IFV_DEEPSPEED=training/configs/deepspeed/zero3-optimizer-offload.json"
        in source
    )


def test_sft_launchers_support_cached_datasets_and_tunable_dataloaders() -> None:
    single = _source("scripts/train/run_sft.sh")
    curriculum = _source("scripts/train/run_curriculum_sft.sh")

    for source in (single, curriculum):
        assert "configure_training_runtime" in source
        assert "IFV_CACHED_DATASET" in source
        assert "--cached_dataset" in source
        assert "IFV_CACHED_VAL_DATASET" in source
        assert "--cached_val_dataset" in source
        assert 'IFV_DATASET_NUM_PROC:-2' in source
        assert 'IFV_DATALOADER_NUM_WORKERS:-2' in source
        assert "IFV_DATALOADER_PERSISTENT_WORKERS" in source
        assert "IFV_DATALOADER_PREFETCH_FACTOR" in source
        assert 'args+=(--group_by_length "$IFV_GROUP_BY_LENGTH")' in source
        assert 'args+=(--bf16 "$IFV_BF16")' in source
        assert 'args+=(--optim "$IFV_OPTIM")' in source
        assert 'args+=(--use_liger_kernel "$IFV_USE_LIGER_KERNEL")' in source
        assert 'args+=(--use_logits_to_keep "$IFV_USE_LOGITS_TO_KEEP")' in source
        assert 'args+=(--torch_empty_cache_steps "$IFV_TORCH_EMPTY_CACHE_STEPS")' in source
        assert "python -m ifv_training training-profile" in source
        assert '--output "$LOG_DIR/profile.json"' in source
        assert 'train_status="${PIPESTATUS[0]}"' in source


def test_sft_launchers_support_fsdp2_gradient_checkpointing_and_sequence_parallel() -> None:
    single = _source("scripts/train/run_sft.sh")
    curriculum = _source("scripts/train/run_curriculum_sft.sh")
    common = _source("scripts/lib/common.sh")

    for source in (single, curriculum):
        assert "training_backend_args" in source
        assert "configure_distributed_backend" in source
        assert 'training_backend_args+=(--deepspeed "$IFV_DEEPSPEED")' in source
        assert 'training_backend_args+=(--fsdp "$IFV_FSDP")' in source
        assert '--gradient_checkpointing "${IFV_GRADIENT_CHECKPOINTING:-true}"' in source
        assert '--gradient_checkpointing_kwargs' in source
        assert 'args+=(--sequence_parallel_size "$IFV_SEQUENCE_PARALLEL_SIZE")' in source

    assert "requires exactly one backend" in common
    assert "do not set both IFV_DEEPSPEED and IFV_FSDP" in common
    assert "IFV_FSDP=fsdp2" in common
    assert "ACCELERATE_USE_FSDP=true" in common
    assert 'FSDP_VERSION="${IFV_FSDP_VERSION:-2}"' in common


def test_curriculum_cache_exporter_uses_ms_swift_cached_dataset_contract() -> None:
    source = _source("scripts/train/cache_curriculum_sft.sh")

    assert "swift export" in source
    assert "--to_cached_dataset true" in source
    assert "--cached_dataset" not in source
    assert "--interleave_prob" in source
    assert "--stopping_strategy all_exhausted" in source
    assert "cache.env" in source
    assert "source-dataset-fingerprints.tsv" in source
    assert "sha256sum" in source
    assert "IFV_CACHED_DATASET" in source
    assert "IFV_CACHED_VAL_DATASET" in source
    assert "cached-dataset-manifest" in source


def test_fsdp2_exporter_merges_audits_and_reload_smokes() -> None:
    source = _source("scripts/export/register_fsdp2_checkpoint.sh")
    probe = _source("scripts/probe/load_qwen35_checkpoint.py")

    assert "merge_fsdp_weights" in source
    assert "configure_training_runtime" in source
    assert "--state-checkpoint-dir" in source
    assert "cached-dataset-manifest" in source
    assert "load_qwen35_checkpoint.py" in source
    assert "AutoModelForImageTextToText" in probe
    assert "parameter_count > 9_000_000_000" in probe


def test_full_checkpoint_exporter_initializes_runtime_and_reload_smokes() -> None:
    source = _source("scripts/export/register_full_checkpoint.sh")

    assert "configure_training_runtime" in source
    assert "audit-full-checkpoint" in source
    assert "checkpoint-manifest" in source
    assert "load_qwen35_checkpoint.py" in source
    assert "--model-dir \"$CHECKPOINT_DIR\"" in source
    assert "load-smoke.json" in source


def test_qwen35_flash_cached_and_no_offload_profiles_exist() -> None:
    cached = _source("configs/sft/qwen3.5-full-pilot30-2gpu-flash-offload-cached.env")
    four_gpu = _source("configs/sft/qwen3.5-full-pilot30-4gpu-flash-no-offload.env")
    four_gpu_gate = _source("configs/sft/qwen3.5-full-1step-4gpu-flash-no-offload.env")
    five_gpu_gate = _source("configs/sft/qwen3.5-full-1step-5gpu-flash-no-offload.env")
    oom_gate = _source("configs/sft/qwen3.5-full-1step-2gpu-flash-no-offload.env")
    no_offload = _source("configs/deepspeed/zero3-no-offload.json")

    assert "IFV_ATTN_IMPL=flash_attn" in cached
    assert "IFV_LOAD_FROM_CACHE_FILE=true" in cached
    assert "IFV_DATALOADER_PERSISTENT_WORKERS=true" in cached
    assert "IFV_DEEPSPEED=training/configs/deepspeed/zero3-no-offload.json" in four_gpu
    assert "IFV_GRADIENT_ACCUMULATION_STEPS=2" in four_gpu
    assert "IFV_MAX_STEPS=1" in four_gpu_gate
    assert "IFV_GRADIENT_ACCUMULATION_STEPS=2" in four_gpu_gate
    assert "IFV_MAX_STEPS=1" in five_gpu_gate
    assert "IFV_GRADIENT_ACCUMULATION_STEPS=1" in five_gpu_gate
    assert "IFV_MAX_STEPS=1" in oom_gate
    assert "IFV_GRADIENT_ACCUMULATION_STEPS=4" in oom_gate
    assert '"stage": 3' in no_offload
    assert "offload_optimizer" not in no_offload
    assert "offload_param" not in no_offload


def test_qwen35_fsdp2_candidate_profiles_are_backend_exclusive() -> None:
    one_step = _source("configs/sft/qwen3.5-full-1step-4gpu-fsdp2-no-offload.env")
    ten_step = _source("configs/sft/qwen3.5-full-10step-4gpu-fsdp2-no-offload.env")
    padding_free = _source("configs/sft/qwen3.5-full-10step-4gpu-fsdp2-padding-free.env")
    sp4_negative = _source(
        "configs/sft/qwen3.5-full-10step-4gpu-fsdp2-sp4-padding-free.env"
    )
    sp4_one_step = _source(
        "configs/sft/qwen3.5-full-1step-4gpu-fsdp2-sp4-padding-free-accum1.env"
    )
    sp4_ten_step = _source(
        "configs/sft/qwen3.5-full-10step-4gpu-fsdp2-sp4-padding-free-accum1.env"
    )
    sp4_bf16_gate = _source(
        "configs/sft/qwen3.5-full-2step-4gpu-fsdp2-sp4-padding-free-accum1-bf16params.env"
    )
    sp4_adafactor_gate = _source(
        "configs/sft/qwen3.5-full-2step-4gpu-fsdp2-sp4-padding-free-accum1-adafactor.env"
    )
    sp4_logits_probe = _source(
        "configs/sft/qwen3.5-full-2step-4gpu-fsdp2-sp4-padding-free-accum1-logits-to-keep-probe.env"
    )
    sp4_resume = _source(
        "configs/sft/qwen3.5-full-11step-resume-4gpu-fsdp2-sp4-padding-free-accum1-bf16params.env"
    )
    sp1_accum1 = _source(
        "configs/sft/qwen3.5-full-2step-4gpu-fsdp2-padding-free-accum1-bf16params.env"
    )
    sp1_accum2 = _source(
        "configs/sft/qwen3.5-full-2step-4gpu-fsdp2-padding-free-accum2-bf16params.env"
    )
    sp2_accum1 = _source(
        "configs/sft/qwen3.5-full-2step-4gpu-fsdp2-sp2-padding-free-accum1-bf16params.env"
    )
    sp2_accum4 = _source(
        "configs/sft/qwen3.5-full-2step-4gpu-fsdp2-sp2-padding-free-accum4-bf16params.env"
    )
    sp4_accum8 = _source(
        "configs/sft/qwen3.5-full-2step-4gpu-fsdp2-sp4-padding-free-accum8-bf16params.env"
    )
    zero3_cached = _source("configs/sft/qwen3.5-full-10step-4gpu-zero3-offload-cached.env")
    zero3_no_checkpoint_gate = _source(
        "configs/sft/qwen3.5-full-2step-4gpu-zero3-offload-cached-no-llm-checkpoint.env"
    )
    zero3_no_checkpoint_ten = _source(
        "configs/sft/qwen3.5-full-10step-4gpu-zero3-offload-cached-no-llm-checkpoint.env"
    )
    zero3_no_checkpoint_resume = _source(
        "configs/sft/qwen3.5-full-11step-resume-4gpu-zero3-offload-cached-no-llm-checkpoint.env"
    )

    for source in (
        one_step,
        ten_step,
        padding_free,
        sp4_negative,
        sp4_one_step,
        sp4_ten_step,
        sp4_bf16_gate,
        sp4_adafactor_gate,
        sp4_logits_probe,
        sp4_resume,
        sp1_accum1,
        sp1_accum2,
        sp2_accum1,
        sp2_accum4,
        sp4_accum8,
    ):
        assert "IFV_FSDP=fsdp2" in source
        assert "IFV_DEEPSPEED" not in source
        assert "IFV_GRADIENT_CHECKPOINTING=false" in source
        assert "IFV_ATTN_IMPL=flash_attn" in source
        assert "IFV_MAX_LENGTH=32768" in source
        assert "IFV_TUNER_TYPE=full" in source

    assert "IFV_MAX_STEPS=1" in one_step
    assert "IFV_MAX_STEPS=10" in ten_step
    assert "IFV_PADDING_FREE=true" in padding_free
    for source in (sp4_negative, sp4_one_step, sp4_ten_step):
        assert "IFV_PADDING_FREE=true" in source
        assert "IFV_SEQUENCE_PARALLEL_SIZE=4" in source
    assert "IFV_GRADIENT_ACCUMULATION_STEPS=2" in sp4_negative
    assert "bounded negative control" in sp4_negative
    assert "IFV_MAX_STEPS=1" in sp4_one_step
    assert "IFV_GRADIENT_ACCUMULATION_STEPS=1" in sp4_one_step
    assert "IFV_MAX_STEPS=10" in sp4_ten_step
    assert "IFV_GRADIENT_ACCUMULATION_STEPS=1" in sp4_ten_step
    assert "IFV_MAX_STEPS=2" in sp4_bf16_gate
    assert "IFV_BF16=false" in sp4_bf16_gate
    assert "IFV_FP16=false" in sp4_bf16_gate
    assert "IFV_MAX_STEPS=2" in sp4_adafactor_gate
    assert "IFV_OPTIM=adafactor" in sp4_adafactor_gate
    assert "IFV_MAX_STEPS=2" in sp4_logits_probe
    assert "IFV_USE_LOGITS_TO_KEEP=true" in sp4_logits_probe
    assert "IFV_MAX_STEPS=11" in sp4_resume
    assert "IFV_SAVE_STEPS=11" in sp4_resume
    assert "IFV_EVAL_STEPS=11" in sp4_resume
    assert "IFV_BF16=false" in sp4_resume
    for source in (sp1_accum1, sp1_accum2):
        assert "IFV_SEQUENCE_PARALLEL_SIZE" not in source
        assert "IFV_BF16=false" in source
        assert "IFV_FP16=false" in source
        assert "IFV_GROUP_BY_LENGTH=true" in source
    assert "IFV_GRADIENT_ACCUMULATION_STEPS=1" in sp1_accum1
    assert "IFV_GRADIENT_ACCUMULATION_STEPS=2" in sp1_accum2
    for source in (sp2_accum1, sp2_accum4):
        assert "IFV_SEQUENCE_PARALLEL_SIZE=2" in source
        assert "IFV_BF16=false" in source
    assert "IFV_GRADIENT_ACCUMULATION_STEPS=1" in sp2_accum1
    assert "IFV_GRADIENT_ACCUMULATION_STEPS=4" in sp2_accum4
    assert "IFV_SEQUENCE_PARALLEL_SIZE=4" in sp4_accum8
    assert "IFV_GRADIENT_ACCUMULATION_STEPS=8" in sp4_accum8
    assert "IFV_DEEPSPEED=training/configs/deepspeed/zero3-optimizer-offload.json" in zero3_cached
    assert "IFV_FSDP" not in zero3_cached
    assert "IFV_GRADIENT_CHECKPOINTING=true" in zero3_cached
    for source in (
        zero3_no_checkpoint_gate,
        zero3_no_checkpoint_ten,
        zero3_no_checkpoint_resume,
    ):
        assert "IFV_DEEPSPEED=training/configs/deepspeed/zero3-optimizer-offload.json" in source
        assert "IFV_FSDP" not in source
        assert "IFV_GRADIENT_CHECKPOINTING=false" in source
        assert "IFV_VIT_GRADIENT_CHECKPOINTING=true" in source
        assert "IFV_GRADIENT_ACCUMULATION_STEPS=2" in source
        assert "IFV_MAX_LENGTH=32768" in source
    assert "IFV_MAX_STEPS=2" in zero3_no_checkpoint_gate
    assert "IFV_MAX_STEPS=10" in zero3_no_checkpoint_ten
    assert "IFV_SAVE_STEPS=10" in zero3_no_checkpoint_ten
    assert "IFV_EVAL_STEPS=10" in zero3_no_checkpoint_ten
    assert "IFV_MAX_STEPS=11" in zero3_no_checkpoint_resume
    assert "IFV_SAVE_STEPS=11" in zero3_no_checkpoint_resume
    assert "IFV_EVAL_STEPS=11" in zero3_no_checkpoint_resume


def test_qwen35_zero3_cpu_adam_thread_profiles_are_bounded() -> None:
    profiles = {
        threads: _source(
            f"configs/sft/qwen3.5-full-2step-4gpu-zero3-offload-cached-omp{threads}.env"
        )
        for threads in (1, 4, 8, 16)
    }

    for threads, source in profiles.items():
        assert "IFV_DEEPSPEED=training/configs/deepspeed/zero3-optimizer-offload.json" in source
        assert "IFV_FSDP" not in source
        assert "IFV_MAX_STEPS=2" in source
        assert "IFV_GRADIENT_ACCUMULATION_STEPS=2" in source
        assert "IFV_GRADIENT_CHECKPOINTING=true" in source
        assert "IFV_DATALOADER_NUM_WORKERS=4" in source
        assert f"IFV_OMP_NUM_THREADS={threads}" in source


def test_qwen35_zero3_dataloader_worker_profiles_are_bounded() -> None:
    profiles = {
        workers: _source(
            "configs/sft/"
            f"qwen3.5-full-2step-4gpu-zero3-offload-cached-workers{workers}-omp8.env"
        )
        for workers in (0, 2, 4, 8)
    }

    for workers, source in profiles.items():
        assert "IFV_DEEPSPEED=training/configs/deepspeed/zero3-optimizer-offload.json" in source
        assert "IFV_FSDP" not in source
        assert "IFV_MAX_STEPS=2" in source
        assert "IFV_GRADIENT_ACCUMULATION_STEPS=2" in source
        assert "IFV_GRADIENT_CHECKPOINTING=true" in source
        assert "IFV_OMP_NUM_THREADS=8" in source
        assert f"IFV_DATALOADER_NUM_WORKERS={workers}" in source

    assert "IFV_DATALOADER_PERSISTENT_WORKERS" not in profiles[0]
    assert "IFV_DATALOADER_PREFETCH_FACTOR" not in profiles[0]
    for workers in (2, 4, 8):
        assert "IFV_DATALOADER_PERSISTENT_WORKERS=true" in profiles[workers]
        assert "IFV_DATALOADER_PREFETCH_FACTOR=2" in profiles[workers]


def test_qwen35_zero3_dataloader_followup_profiles_are_bounded() -> None:
    no_persist = _source(
        "configs/sft/"
        "qwen3.5-full-2step-4gpu-zero3-offload-cached-workers4-nopersist-omp8.env"
    )
    prefetch4 = _source(
        "configs/sft/"
        "qwen3.5-full-2step-4gpu-zero3-offload-cached-workers4-prefetch4-omp8.env"
    )

    for source in (no_persist, prefetch4):
        assert "IFV_DEEPSPEED=training/configs/deepspeed/zero3-optimizer-offload.json" in source
        assert "IFV_MAX_STEPS=2" in source
        assert "IFV_DATALOADER_NUM_WORKERS=4" in source
        assert "IFV_OMP_NUM_THREADS=8" in source

    assert "IFV_DATALOADER_PERSISTENT_WORKERS=false" in no_persist
    assert "IFV_DATALOADER_PREFETCH_FACTOR=2" in no_persist
    assert "IFV_DATALOADER_PERSISTENT_WORKERS=true" in prefetch4
    assert "IFV_DATALOADER_PREFETCH_FACTOR=4" in prefetch4


def test_qwen35_zero3_workers0_production_profiles_are_bounded() -> None:
    ten_step = _source(
        "configs/sft/"
        "qwen3.5-full-10step-4gpu-zero3-offload-cached-workers0-omp8.env"
    )
    resume = _source(
        "configs/sft/"
        "qwen3.5-full-11step-resume-4gpu-zero3-offload-cached-workers0-omp8.env"
    )

    for source in (ten_step, resume):
        assert "IFV_DEEPSPEED=training/configs/deepspeed/zero3-optimizer-offload.json" in source
        assert "IFV_FSDP" not in source
        assert "IFV_GRADIENT_ACCUMULATION_STEPS=2" in source
        assert "IFV_GRADIENT_CHECKPOINTING=true" in source
        assert "IFV_DATALOADER_NUM_WORKERS=0" in source
        assert "IFV_DATALOADER_PERSISTENT_WORKERS" not in source
        assert "IFV_DATALOADER_PREFETCH_FACTOR" not in source
        assert "IFV_OMP_NUM_THREADS=8" in source

    assert "IFV_MAX_STEPS=10" in ten_step
    assert "IFV_SAVE_STEPS=10" in ten_step
    assert "IFV_EVAL_STEPS=10" in ten_step
    assert "IFV_MAX_STEPS=11" in resume
    assert "IFV_SAVE_STEPS=11" in resume
    assert "IFV_EVAL_STEPS=11" in resume


def test_qwen35_zero3_speed_probe_is_full_parameter_and_32k() -> None:
    source = _source("configs/sft/qwen3.5-full-1step-zero3.env")

    assert "IFV_DEEPSPEED=zero3" in source
    assert "IFV_TUNER_TYPE=full" in source
    assert "IFV_MAX_LENGTH=32768" in source
    assert "IFV_FREEZE_LLM=false" in source


def test_qwen35_optimizer_offload_probe_keeps_parameters_on_gpu() -> None:
    profile = _source("configs/sft/qwen3.5-full-1step-optimizer-offload.env")
    config = _source("configs/deepspeed/zero3-optimizer-offload.json")
    common = _source("scripts/lib/common.sh")

    assert "IFV_DEEPSPEED=training/configs/deepspeed/zero3-optimizer-offload.json" in profile
    assert '"offload_optimizer"' in config
    assert '"device": "cpu"' in config
    assert '"offload_param"' in config
    assert '"device": "none"' in config
    assert 'elif [[ -f "$config" ]]' in common


def test_qwen35_batch2_capacity_probe_is_optimizer_only_and_bounded() -> None:
    source = _source(
        "configs/sft/qwen3.5-full-1step-batch2-optimizer-offload.env"
    )

    assert "IFV_MAX_STEPS=1" in source
    assert "IFV_TRAIN_BATCH_SIZE=2" in source
    assert "IFV_EVAL_BATCH_SIZE=1" in source
    assert "IFV_GRADIENT_ACCUMULATION_STEPS=1" in source
    assert (
        "IFV_DEEPSPEED=training/configs/deepspeed/zero3-optimizer-offload.json"
        in source
    )
    assert "IFV_MAX_LENGTH=32768" in source


def test_qwen35_batch2_length_grouped_probe_uses_native_sampler() -> None:
    source = _source(
        "configs/sft/qwen3.5-full-10step-batch2-length-grouped.env"
    )
    launcher = _source("scripts/train/run_curriculum_sft.sh")

    assert "IFV_MAX_STEPS=10" in source
    assert "IFV_TRAIN_BATCH_SIZE=2" in source
    assert "IFV_GROUP_BY_LENGTH=true" in source
    assert 'args+=(--group_by_length "$IFV_GROUP_BY_LENGTH")' in launcher


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
    assert 'link_torch_cuda_runtime "$SFT_PREFIX"' in source
    assert 'prebuild_cpu_adam "$SFT_PREFIX"' in source
    assert "refusing to modify an existing environment" in source
    assert "ifv-qwen35-sft-ms-swift442" in source
    assert "ifv-qwen35-rl-ms-swift442-vllm0221" in source
    assert "pip freeze --all" in source
    assert "get_model_processor" in source
    assert "Qwen3_5ForConditionalGeneration" in source
    assert "is_causal_conv1d_available" in source
    assert "is_flash_attn_2_available" in source
    assert 'enable_thinking=False' in source
    assert "ms-swift==4.4.2" in sft
    assert "transformers==5.12.1" in sft
    assert "flash-linear-attention==0.5.1" in sft
    assert "causal-conv1d==1.6.2.post1" not in sft
    assert "flash-attn==2.8.3" not in sft
    assert "vllm==0.22.1" in rl
    assert "llguidance==1.7.5" in rl
