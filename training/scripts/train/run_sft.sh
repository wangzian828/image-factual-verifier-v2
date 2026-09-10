#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"

if [[ "$#" -lt 5 || "$#" -gt 6 ]]; then
  echo "usage: $0 MODEL_PROFILE SFT_PROFILE TRAIN_JSONL VAL_JSONL EXPERIMENT_ID [RESUME_CHECKPOINT]" >&2
  exit 2
fi

MODEL_PROFILE="$1"
SFT_PROFILE="$2"
TRAIN_DATASET="$3"
VAL_DATASET="$4"
EXPERIMENT_ID="$5"
RESUME_CHECKPOINT="${6:-}"

load_profile "$MODEL_PROFILE"
load_profile "$SFT_PROFILE"
export PYTHONPATH="$REPO_ROOT/training${PYTHONPATH:+:$PYTHONPATH}"
configure_training_runtime
require_idle_gpus
require_full_parameter_profile
configure_distributed_backend
require_model_path
prepare_deepspeed_cpu_adam
require_value EXPERIMENT_ID

case "${IFV_PADDING_FREE:-false}:${IFV_ATTN_IMPL,,}" in
  true:flash_attn|true:flash_attention_2|true:flash_attention_3|true:flash_attention_4|false:*) ;;
  true:*)
    echo "IFV_PADDING_FREE=true requires a flash attention implementation; use IFV_PADDING_FREE=false with IFV_ATTN_IMPL=${IFV_ATTN_IMPL:-sdpa}." >&2
    exit 2
    ;;
esac

training_backend_args=()
if [[ -n "${IFV_DEEPSPEED:-}" ]]; then
  training_backend_args+=(--deepspeed "$IFV_DEEPSPEED")
fi
if [[ -n "${IFV_FSDP:-}" ]]; then
  training_backend_args+=(--fsdp "$IFV_FSDP")
fi

dataset_args=()
cached_train_datasets=()
cached_val_datasets=()
raw_dataset_mode=false
raw_dataset_dir=""
if [[ -n "${IFV_CACHED_DATASET:-}" || -n "${IFV_CACHED_VAL_DATASET:-}" ]]; then
  if [[ -z "${IFV_CACHED_DATASET:-}" || -z "${IFV_CACHED_VAL_DATASET:-}" ]]; then
    echo "cached training requires both IFV_CACHED_DATASET and IFV_CACHED_VAL_DATASET" >&2
    exit 2
  fi
  read -r -a cached_train_datasets <<<"$IFV_CACHED_DATASET"
  for cached_dataset in "${cached_train_datasets[@]}"; do
    if [[ ! -d "$cached_dataset" ]]; then
      echo "cached training dataset does not exist: $cached_dataset" >&2
      exit 2
    fi
  done
  read -r -a cached_val_datasets <<<"$IFV_CACHED_VAL_DATASET"
  for cached_dataset in "${cached_val_datasets[@]}"; do
    if [[ ! -d "$cached_dataset" ]]; then
      echo "cached validation dataset does not exist: $cached_dataset" >&2
      exit 2
    fi
  done
  if [[ "${#cached_train_datasets[@]}" -ne "${#cached_val_datasets[@]}" ]]; then
    echo "cached train/validation dataset counts must match" >&2
    exit 2
  fi
  dataset_args+=(--cached_dataset "${cached_train_datasets[@]}")
  dataset_args+=(--cached_val_dataset "${cached_val_datasets[@]}")
else
  raw_dataset_mode=true
  require_dataset "$TRAIN_DATASET"
  require_dataset "$VAL_DATASET"
  require_value IFV_PROCESSOR_VERIFICATION
  if [[ ! -s "$IFV_PROCESSOR_VERIFICATION" ]]; then
    echo "processor verification report does not exist or is empty: $IFV_PROCESSOR_VERIFICATION" >&2
    exit 2
  fi
  raw_dataset_dir="${IFV_RAW_DATASET_DIR:-$(dirname "$TRAIN_DATASET")}"
  if [[ ! -s "$raw_dataset_dir/manifest.json" ]]; then
    echo "raw SFT dataset manifest does not exist or is empty: $raw_dataset_dir/manifest.json" >&2
    exit 2
  fi
  if [[ "${IFV_ALLOW_UNDERSIZED_DISTRIBUTED_DATASET:-false}" != "true" ]]; then
    IFS=',' read -r -a visible_gpu_ids <<< "${CUDA_VISIBLE_DEVICES:-}"
    distributed_gpu_count="${#visible_gpu_ids[@]}"
    train_row_count="$(wc -l < "$TRAIN_DATASET")"
    val_row_count="$(wc -l < "$VAL_DATASET")"
    if (( distributed_gpu_count > 1 && (train_row_count < distributed_gpu_count || val_row_count < distributed_gpu_count) )); then
      echo "distributed SFT requires at least one train and validation row per visible GPU (train=${train_row_count}, validation=${val_row_count}, gpus=${distributed_gpu_count}); use a larger dataset, one GPU, or set IFV_ALLOW_UNDERSIZED_DISTRIBUTED_DATASET=true only for a controlled experiment." >&2
      exit 2
    fi
  fi
  dataset_args+=(--dataset "$TRAIN_DATASET")
  dataset_args+=(--val_dataset "$VAL_DATASET")
fi

OUTPUT_DIR="$DATA_ROOT/checkpoints/$EXPERIMENT_ID"
EXPERIMENT_DIR="$DATA_ROOT/logs/$EXPERIMENT_ID"
LOG_DIR="$EXPERIMENT_DIR"
new_output_dir "$OUTPUT_DIR"
new_output_dir "$EXPERIMENT_DIR"
record_environment "$EXPERIMENT_DIR"
ENVIRONMENT_PREFLIGHT=""
if [[ "${IFV_REQUIRE_TRAINING_ENV_PREFLIGHT:-false}" == "true" ]]; then
  for name in \
    IFV_EXPECTED_GPU_COUNT \
    IFV_EXPECTED_GPU_NAME \
    IFV_EXPECTED_GPU_MEMORY_MIB \
    IFV_EXPECTED_PYTHON \
    IFV_EXPECTED_TORCH_CUDA \
    IFV_EXPECTED_TORCH_VERSION \
    IFV_EXPECTED_TRANSFORMERS_VERSION \
    IFV_EXPECTED_MS_SWIFT_VERSION \
    IFV_EXPECTED_DEEPSPEED_VERSION \
    IFV_EXPECTED_FLASH_ATTN_VERSION \
    IFV_EXPECTED_FLA_VERSION \
    IFV_EXPECTED_CAUSAL_CONV1D_VERSION \
    IFV_EXPECTED_LIGER_VERSION
  do
    require_value "$name"
  done
  ENVIRONMENT_PREFLIGHT="$EXPERIMENT_DIR/environment-preflight.json"
  python "$REPO_ROOT/training/scripts/probe/verify_qwen35_sft_environment.py" \
    --model "$IFV_MODEL_ID" \
    --max-context "$IFV_MAX_LENGTH" \
    --expected-package-version "torch=$IFV_EXPECTED_TORCH_VERSION" \
    --expected-package-version "transformers=$IFV_EXPECTED_TRANSFORMERS_VERSION" \
    --expected-package-version "ms-swift=$IFV_EXPECTED_MS_SWIFT_VERSION" \
    --expected-package-version "deepspeed=$IFV_EXPECTED_DEEPSPEED_VERSION" \
    --expected-package-version "flash-attn=$IFV_EXPECTED_FLASH_ATTN_VERSION" \
    --expected-package-version "flash-linear-attention=$IFV_EXPECTED_FLA_VERSION" \
    --expected-package-version "causal-conv1d=$IFV_EXPECTED_CAUSAL_CONV1D_VERSION" \
    --expected-package-version "liger-kernel=$IFV_EXPECTED_LIGER_VERSION" \
    --required-package torch \
    --required-package transformers \
    --required-package ms-swift \
    --required-package deepspeed \
    --required-package flash-attn \
    --required-package flash-linear-attention \
    --required-package causal-conv1d \
    --required-package liger-kernel \
    --expected-python "$IFV_EXPECTED_PYTHON" \
    --expected-torch-cuda "$IFV_EXPECTED_TORCH_CUDA" \
    --expected-gpu-count "$IFV_EXPECTED_GPU_COUNT" \
    --expected-gpu-name "$IFV_EXPECTED_GPU_NAME" \
    --expected-gpu-memory-mib "$IFV_EXPECTED_GPU_MEMORY_MIB" \
    --gpu-memory-tolerance-mib "${IFV_GPU_MEMORY_TOLERANCE_MIB:-128}" \
    --output "$ENVIRONMENT_PREFLIGHT"
fi
SAVE_STRATEGY="${IFV_SAVE_STRATEGY:-steps}"
CHECKPOINT_PREFLIGHT=""
if [[ "$SAVE_STRATEGY" != "no" ]]; then
  CHECKPOINT_PREFLIGHT="$EXPERIMENT_DIR/checkpoint-storage-preflight.json"
  python -m ifv_training checkpoint-storage-preflight \
    --output-dir "$OUTPUT_DIR" \
    --estimated-checkpoint-bytes "${IFV_CHECKPOINT_ESTIMATED_BYTES:-150000000000}" \
    --reserve-multiplier "${IFV_CHECKPOINT_RESERVE_MULTIPLIER:-1.25}" \
    --output "$CHECKPOINT_PREFLIGHT"
fi
CACHE_VERIFICATION=""
if [[ "${#cached_train_datasets[@]}" -gt 0 ]]; then
  CACHE_VERIFICATION="$EXPERIMENT_DIR/cached-dataset-gate.json"
  verify_cached_dataset_gate \
    "$CACHE_VERIFICATION" \
    cached_train_datasets \
    cached_val_datasets
fi
RAW_DATASET_VERIFICATION=""
if [[ "$raw_dataset_mode" == "true" ]]; then
  RAW_DATASET_VERIFICATION="$EXPERIMENT_DIR/raw-dataset-gate.json"
  raw_gate_args=(
    python "$REPO_ROOT/training/scripts/probe/verify_sft_data_contract.py"
    --train-jsonl "$TRAIN_DATASET"
    --validation-jsonl "$VAL_DATASET"
    --dataset-dir "$raw_dataset_dir"
    --processor-report "$IFV_PROCESSOR_VERIFICATION"
    --model "$IFV_MODEL_ID"
    --output "$RAW_DATASET_VERIFICATION"
    --max-context "$IFV_MAX_LENGTH"
    --truncation-strategy "${IFV_TRUNCATION_STRATEGY:-raise}"
    --padding-free "${IFV_PADDING_FREE:-false}"
    --sequence-parallel-size "${IFV_SEQUENCE_PARALLEL_SIZE:-1}"
    --loss-scale "$IFV_LOSS_SCALE"
    --enable-thinking "${IFV_ENABLE_THINKING:-false}"
    --add-non-thinking-prefix "${IFV_ADD_NON_THINKING_PREFIX:-false}"
    --image-max-token-num "$IFV_IMAGE_MAX_TOKEN_NUM"
  )
  if [[ -n "${IFV_MAX_PIXELS:-}" ]]; then
    raw_gate_args+=(--max-pixels "$IFV_MAX_PIXELS")
  fi
  "${raw_gate_args[@]}"
fi
ENCODE_CACHE_REPORT=""
if [[ "${IFV_ENCODE_CACHE_ENABLED:-false}" == "true" ]]; then
  configure_encode_cache "$EXPERIMENT_DIR"
  ENCODE_CACHE_REPORT="$EXPERIMENT_DIR/encode-cache-report.json"
fi

args=(
  swift sft
  --model "$IFV_MODEL_ID"
  "${dataset_args[@]}"
  --split_dataset_ratio 0
  --strict true
  --load_from_cache_file "$IFV_LOAD_FROM_CACHE_FILE"
  --tuner_type "$IFV_TUNER_TYPE"
  --torch_dtype "$IFV_TORCH_DTYPE"
  --per_device_train_batch_size "$IFV_TRAIN_BATCH_SIZE"
  --per_device_eval_batch_size "$IFV_EVAL_BATCH_SIZE"
  --gradient_accumulation_steps "$IFV_GRADIENT_ACCUMULATION_STEPS"
  --learning_rate "$IFV_LEARNING_RATE"
  --freeze_llm "$IFV_FREEZE_LLM"
  --freeze_vit "$IFV_FREEZE_VIT"
  --freeze_aligner "$IFV_FREEZE_ALIGNER"
  --gradient_checkpointing "${IFV_GRADIENT_CHECKPOINTING:-true}"
  --vit_gradient_checkpointing "$IFV_VIT_GRADIENT_CHECKPOINTING"
  --eval_strategy "${IFV_EVAL_STRATEGY:-steps}"
  --save_strategy "$SAVE_STRATEGY"
  --logging_steps "$IFV_LOGGING_STEPS"
  --max_length "$IFV_MAX_LENGTH"
  --attn_impl "$IFV_ATTN_IMPL"
  --loss_scale "$IFV_LOSS_SCALE"
  --output_dir "$OUTPUT_DIR"
  --warmup_ratio 0.05
  "${training_backend_args[@]}"
  --dataset_num_proc "${IFV_DATASET_NUM_PROC:-2}"
  --dataloader_num_workers "${IFV_DATALOADER_NUM_WORKERS:-2}"
  --report_to tensorboard
)

if [[ "$SAVE_STRATEGY" != "no" ]]; then
  args+=(--save_steps "$IFV_SAVE_STEPS")
  args+=(--save_total_limit "$IFV_SAVE_TOTAL_LIMIT")
fi

if [[ "${IFV_EVAL_STRATEGY:-steps}" != "no" ]]; then
  args+=(--eval_steps "$IFV_EVAL_STEPS")
fi

if [[ -n "${IFV_NUM_TRAIN_EPOCHS:-}" ]]; then
  args+=(--num_train_epochs "$IFV_NUM_TRAIN_EPOCHS")
elif [[ -n "${IFV_MAX_STEPS:-}" ]]; then
  args+=(--max_steps "$IFV_MAX_STEPS")
else
  echo "training profile must set IFV_NUM_TRAIN_EPOCHS or IFV_MAX_STEPS" >&2
  exit 2
fi

if [[ "${IFV_GRADIENT_CHECKPOINTING:-true}" == "true" ]]; then
  args+=(--gradient_checkpointing_kwargs '{"use_reentrant": false}')
fi
if [[ -n "${IFV_DATALOADER_PREFETCH_FACTOR:-}" ]]; then
  args+=(--dataloader_prefetch_factor "$IFV_DATALOADER_PREFETCH_FACTOR")
fi
if [[ -n "${IFV_MAX_PIXELS:-}" ]]; then
  args+=(--max_pixels "$IFV_MAX_PIXELS")
fi
if [[ -n "${IFV_TRUNCATION_STRATEGY:-}" ]]; then
  args+=(--truncation_strategy "$IFV_TRUNCATION_STRATEGY")
fi
if [[ -n "${IFV_DATALOADER_PERSISTENT_WORKERS:-}" ]]; then
  args+=(--dataloader_persistent_workers "$IFV_DATALOADER_PERSISTENT_WORKERS")
fi
if [[ -n "${IFV_PADDING_FREE:-}" ]]; then
  args+=(--padding_free "$IFV_PADDING_FREE")
fi
if [[ -n "${IFV_SEQUENCE_PARALLEL_SIZE:-}" ]]; then
  args+=(--sequence_parallel_size "$IFV_SEQUENCE_PARALLEL_SIZE")
fi
if [[ -n "${IFV_ENABLE_THINKING:-}" ]]; then
  args+=(--enable_thinking "$IFV_ENABLE_THINKING")
fi
args+=(--add_non_thinking_prefix "${IFV_ADD_NON_THINKING_PREFIX:-false}")
if [[ -n "${IFV_GROUP_BY_LENGTH:-}" ]]; then
  args+=(--group_by_length "$IFV_GROUP_BY_LENGTH")
fi
if [[ -n "${IFV_BF16:-}" ]]; then
  args+=(--bf16 "$IFV_BF16")
fi
if [[ -n "${IFV_FP16:-}" ]]; then
  args+=(--fp16 "$IFV_FP16")
fi
if [[ -n "${IFV_OPTIM:-}" ]]; then
  args+=(--optim "$IFV_OPTIM")
fi
if [[ -n "${IFV_OPTIM_ARGS:-}" ]]; then
  args+=(--optim_args "$IFV_OPTIM_ARGS")
fi
if [[ -n "${IFV_USE_LIGER_KERNEL:-}" ]]; then
  args+=(--use_liger_kernel "$IFV_USE_LIGER_KERNEL")
fi
if [[ -n "${IFV_USE_LOGITS_TO_KEEP:-}" ]]; then
  args+=(--use_logits_to_keep "$IFV_USE_LOGITS_TO_KEEP")
fi
if [[ -n "${IFV_TORCH_EMPTY_CACHE_STEPS:-}" ]]; then
  args+=(--torch_empty_cache_steps "$IFV_TORCH_EMPTY_CACHE_STEPS")
fi
if [[ -n "${IFV_EXPERTS_IMPL:-}" ]]; then
  args+=(--experts_impl "$IFV_EXPERTS_IMPL")
fi
if [[ -n "$RESUME_CHECKPOINT" ]]; then
  if [[ ! -d "$RESUME_CHECKPOINT" ]]; then
    echo "resume checkpoint does not exist: $RESUME_CHECKPOINT" >&2
    exit 2
  fi
  args+=(--resume_from_checkpoint "$RESUME_CHECKPOINT")
fi

export IMAGE_MAX_TOKEN_NUM="$IFV_IMAGE_MAX_TOKEN_NUM"
print_command "${args[@]}"
RESOURCE_SUMMARY="$LOG_DIR/resource-summary.json"
RESOURCE_SAMPLES="$LOG_DIR/resource-samples.jsonl"
WATCHDOG_OUTPUT="$LOG_DIR/monitor-latest.json"
WATCHDOG_LOG="$LOG_DIR/monitor.log"
WATCHDOG_PID=""
touch "$LOG_DIR/train.log"
watchdog_args=(
  python "$SCRIPT_DIR/watch_sft.py"
  --train-log "$LOG_DIR/train.log"
  --checkpoint-root "$OUTPUT_DIR"
  --resource-samples "$RESOURCE_SAMPLES"
  --output "$WATCHDOG_OUTPUT"
  --interval-seconds "${IFV_SFT_WATCHDOG_INTERVAL_SECONDS:-60}"
  --stale-seconds "${IFV_SFT_WATCHDOG_STALE_SECONDS:-900}"
)
if [[ -n "${IFV_GPU_MEMORY_TARGET_MIN_MIB:-}" ]]; then
  watchdog_args+=(--gpu-memory-target-min-mib "$IFV_GPU_MEMORY_TARGET_MIN_MIB")
fi
if [[ -n "${IFV_GPU_MEMORY_TARGET_MAX_MIB:-}" ]]; then
  watchdog_args+=(--gpu-memory-target-max-mib "$IFV_GPU_MEMORY_TARGET_MAX_MIB")
fi
if [[ -n "${IFV_GPU_MEMORY_MAX_IMBALANCE_MIB:-}" ]]; then
  watchdog_args+=(--gpu-memory-max-imbalance-mib "$IFV_GPU_MEMORY_MAX_IMBALANCE_MIB")
fi
if [[ -n "${IFV_GPU_UTILIZATION_TARGET_MIN_PERCENT:-}" ]]; then
  watchdog_args+=(
    --gpu-utilization-target-min-percent
    "$IFV_GPU_UTILIZATION_TARGET_MIN_PERCENT"
  )
fi
if [[ -n "${IFV_SFT_BEHAVIOR_METRICS:-}" ]]; then
  watchdog_args+=(--behavior-metrics "$IFV_SFT_BEHAVIOR_METRICS")
fi
stop_watchdog() {
  if [[ -n "$WATCHDOG_PID" ]] && kill -0 "$WATCHDOG_PID" 2>/dev/null; then
    kill "$WATCHDOG_PID" 2>/dev/null || true
    wait "$WATCHDOG_PID" 2>/dev/null || true
  fi
  WATCHDOG_PID=""
}
if [[ "${IFV_SFT_WATCHDOG_ENABLED:-true}" == "true" ]]; then
  "${watchdog_args[@]}" >"$WATCHDOG_LOG" 2>&1 &
  WATCHDOG_PID="$!"
  trap stop_watchdog EXIT
fi
resource_monitor_args=(
  python "$SCRIPT_DIR/run_with_resource_monitor.py"
  --summary-output "$RESOURCE_SUMMARY"
  --samples-output "$RESOURCE_SAMPLES"
  --gpu-ids "$CUDA_VISIBLE_DEVICES"
  --sample-interval "${IFV_RESOURCE_SAMPLE_INTERVAL:-2}"
)
if [[ -n "${IFV_GPU_MEMORY_TARGET_MIN_MIB:-}" ]]; then
  resource_monitor_args+=(--memory-target-min-mib "$IFV_GPU_MEMORY_TARGET_MIN_MIB")
fi
if [[ -n "${IFV_GPU_MEMORY_TARGET_MAX_MIB:-}" ]]; then
  resource_monitor_args+=(--memory-target-max-mib "$IFV_GPU_MEMORY_TARGET_MAX_MIB")
fi
if [[ -n "${IFV_GPU_MEMORY_MAX_IMBALANCE_MIB:-}" ]]; then
  resource_monitor_args+=(
    --memory-max-imbalance-mib
    "$IFV_GPU_MEMORY_MAX_IMBALANCE_MIB"
  )
fi
set +e
"${resource_monitor_args[@]}" -- "${args[@]}" 2>&1 | tee "$LOG_DIR/train.log"
train_status="${PIPESTATUS[0]}"
set -e
stop_watchdog
trap - EXIT
final_watchdog_args=("${watchdog_args[@]}" --once)
"${final_watchdog_args[@]}" >>"$WATCHDOG_LOG" 2>&1 || true
encode_cache_status=0
if [[ -n "$ENCODE_CACHE_REPORT" ]]; then
  IFV_ENCODE_CACHE_ENABLED=false python -m ifv_training encode-cache-report \
    --metrics-dir "$IFV_ENCODE_CACHE_METRICS_DIR" \
    --output "$ENCODE_CACHE_REPORT" || encode_cache_status="$?"
fi
scheduler_audit_status=0
SCHEDULER_AUDIT=""
checkpoint_io_status=0
CHECKPOINT_IO_PROFILE=""
if [[ "$train_status" -eq 0 ]]; then
  latest_checkpoint="$(
    find "$OUTPUT_DIR" -type d -name 'checkpoint-*' -print |
      sort -V |
      tail -n 1
  )"
  if [[ -n "$latest_checkpoint" && -s "$latest_checkpoint/scheduler.pt" ]]; then
    CHECKPOINT_IO_PROFILE="$EXPERIMENT_DIR/checkpoint-io-profile.json"
    IFV_ENCODE_CACHE_ENABLED=false python -m ifv_training \
      checkpoint-io-profile \
      --checkpoint "$latest_checkpoint" \
      --output "$CHECKPOINT_IO_PROFILE" || checkpoint_io_status="$?"
    SCHEDULER_AUDIT="$EXPERIMENT_DIR/scheduler-order-audit.json"
    IFV_ENCODE_CACHE_ENABLED=false python \
      "$REPO_ROOT/training/scripts/probe/audit_deepspeed_scheduler.py" \
      --train-log "$LOG_DIR/train.log" \
      --checkpoint "$latest_checkpoint" \
      --output "$SCHEDULER_AUDIT" || scheduler_audit_status="$?"
  fi
fi
profile_status=0
profile_args=(
  python -m ifv_training training-profile
  --train-log "$LOG_DIR/train.log"
  --output "$LOG_DIR/profile.json"
  --steady-window "${IFV_TRAINING_STEADY_WINDOW:-5}"
  --experiment-id "$EXPERIMENT_ID"
  --profile-id "$(basename "$SFT_PROFILE")"
  --resource-summary "$RESOURCE_SUMMARY"
  --train-exit-code "$train_status"
)
if [[ -n "$CACHE_VERIFICATION" ]]; then
  profile_args+=(--cache-verification "$CACHE_VERIFICATION")
fi
if [[ -n "$RAW_DATASET_VERIFICATION" ]]; then
  profile_args+=(--dataset-verification "$RAW_DATASET_VERIFICATION")
fi
if [[ -n "$ENVIRONMENT_PREFLIGHT" ]]; then
  profile_args+=(--environment-preflight "$ENVIRONMENT_PREFLIGHT")
fi
if [[ -n "$ENCODE_CACHE_REPORT" ]]; then
  profile_args+=(--encode-cache-report "$ENCODE_CACHE_REPORT")
fi
if [[ -n "$SCHEDULER_AUDIT" ]]; then
  profile_args+=(--scheduler-audit "$SCHEDULER_AUDIT")
fi
if [[ -n "$CHECKPOINT_PREFLIGHT" ]]; then
  profile_args+=(--checkpoint-preflight "$CHECKPOINT_PREFLIGHT")
fi
if [[ -n "$CHECKPOINT_IO_PROFILE" ]]; then
  profile_args+=(--checkpoint-io-profile "$CHECKPOINT_IO_PROFILE")
fi
"${profile_args[@]}" || profile_status="$?"
if [[ "$train_status" -eq 0 && "$encode_cache_status" -ne 0 ]]; then
  exit "$encode_cache_status"
fi
if [[ "$train_status" -eq 0 && "$scheduler_audit_status" -ne 0 ]]; then
  exit "$scheduler_audit_status"
fi
if [[ "$train_status" -eq 0 && "$checkpoint_io_status" -ne 0 ]]; then
  exit "$checkpoint_io_status"
fi
if [[ "$train_status" -eq 0 && "$profile_status" -ne 0 ]]; then
  exit "$profile_status"
fi
exit "$train_status"
