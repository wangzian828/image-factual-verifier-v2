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
configure_training_runtime
require_idle_gpus
require_full_parameter_profile
configure_distributed_backend
require_model_path
prepare_deepspeed_cpu_adam
require_value EXPERIMENT_ID

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
  require_dataset "$TRAIN_DATASET"
  require_dataset "$VAL_DATASET"
  dataset_args+=(--dataset "$TRAIN_DATASET")
  dataset_args+=(--val_dataset "$VAL_DATASET")
fi

OUTPUT_DIR="$DATA_ROOT/checkpoints/$EXPERIMENT_ID"
EXPERIMENT_DIR="$DATA_ROOT/logs/$EXPERIMENT_ID"
LOG_DIR="$EXPERIMENT_DIR"
new_output_dir "$OUTPUT_DIR"
new_output_dir "$EXPERIMENT_DIR"
record_environment "$EXPERIMENT_DIR"
CACHE_VERIFICATION=""
if [[ "${#cached_train_datasets[@]}" -gt 0 ]]; then
  CACHE_VERIFICATION="$EXPERIMENT_DIR/cached-dataset-gate.json"
  verify_cached_dataset_gate \
    "$CACHE_VERIFICATION" \
    cached_train_datasets \
    cached_val_datasets
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
  --max_steps "$IFV_MAX_STEPS"
  --per_device_train_batch_size "$IFV_TRAIN_BATCH_SIZE"
  --per_device_eval_batch_size "$IFV_EVAL_BATCH_SIZE"
  --gradient_accumulation_steps "$IFV_GRADIENT_ACCUMULATION_STEPS"
  --learning_rate "$IFV_LEARNING_RATE"
  --freeze_llm "$IFV_FREEZE_LLM"
  --freeze_vit "$IFV_FREEZE_VIT"
  --freeze_aligner "$IFV_FREEZE_ALIGNER"
  --gradient_checkpointing "${IFV_GRADIENT_CHECKPOINTING:-true}"
  --vit_gradient_checkpointing "$IFV_VIT_GRADIENT_CHECKPOINTING"
  --eval_strategy steps
  --eval_steps "$IFV_EVAL_STEPS"
  --save_strategy steps
  --save_steps "$IFV_SAVE_STEPS"
  --save_total_limit "$IFV_SAVE_TOTAL_LIMIT"
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

if [[ "${IFV_GRADIENT_CHECKPOINTING:-true}" == "true" ]]; then
  args+=(--gradient_checkpointing_kwargs '{"use_reentrant": false}')
fi
if [[ -n "${IFV_DATALOADER_PREFETCH_FACTOR:-}" ]]; then
  args+=(--dataloader_prefetch_factor "$IFV_DATALOADER_PREFETCH_FACTOR")
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
if [[ "${IFV_ADD_NON_THINKING_PREFIX:-false}" == "true" ]]; then
  args+=(--add_non_thinking_prefix true)
fi
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
set +e
python "$SCRIPT_DIR/run_with_resource_monitor.py" \
  --summary-output "$RESOURCE_SUMMARY" \
  --samples-output "$RESOURCE_SAMPLES" \
  --gpu-ids "$CUDA_VISIBLE_DEVICES" \
  --sample-interval "${IFV_RESOURCE_SAMPLE_INTERVAL:-2}" \
  -- "${args[@]}" 2>&1 | tee "$LOG_DIR/train.log"
train_status="${PIPESTATUS[0]}"
set -e
profile_status=0
profile_args=(
  python -m ifv_training training-profile
  --train-log "$LOG_DIR/train.log"
  --output "$LOG_DIR/profile.json"
  --experiment-id "$EXPERIMENT_ID"
  --profile-id "$(basename "$SFT_PROFILE")"
  --resource-summary "$RESOURCE_SUMMARY"
  --train-exit-code "$train_status"
)
if [[ -n "$CACHE_VERIFICATION" ]]; then
  profile_args+=(--cache-verification "$CACHE_VERIFICATION")
fi
"${profile_args[@]}" || profile_status="$?"
if [[ "$train_status" -eq 0 && "$profile_status" -ne 0 ]]; then
  exit "$profile_status"
fi
exit "$train_status"
