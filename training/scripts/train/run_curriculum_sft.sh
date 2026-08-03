#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"

if [[ "$#" -lt 5 || "$#" -gt 6 ]]; then
  echo "usage: $0 MODEL_PROFILE SFT_PROFILE PERCEPTION_DIR POLICY_DIR EXPERIMENT_ID [RESUME_CHECKPOINT]" >&2
  exit 2
fi

MODEL_PROFILE="$1"
SFT_PROFILE="$2"
PERCEPTION_DIR="$3"
POLICY_DIR="$4"
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

candidate_train_datasets=(
  "$PERCEPTION_DIR/train.jsonl"
  "$POLICY_DIR/train.group-planning.jsonl"
  "$POLICY_DIR/train.group-react.jsonl"
  "$POLICY_DIR/train.group-decision.jsonl"
  "$POLICY_DIR/train.group-reflection.jsonl"
  "$POLICY_DIR/train.group-judgment.jsonl"
)
candidate_validation_datasets=(
  "$PERCEPTION_DIR/validation.jsonl"
  "$POLICY_DIR/validation.group-planning.jsonl"
  "$POLICY_DIR/validation.group-react.jsonl"
  "$POLICY_DIR/validation.group-decision.jsonl"
  "$POLICY_DIR/validation.group-reflection.jsonl"
  "$POLICY_DIR/validation.group-judgment.jsonl"
)
candidate_channels=(perception planning react decision reflection judgment)
candidate_weights=(0.20 0.15 0.35 0.20 0.05 0.05)
train_datasets=()
validation_datasets=()
active_channels=()
active_weights=()
for index in "${!candidate_channels[@]}"; do
  train_dataset="${candidate_train_datasets[$index]}"
  validation_dataset="${candidate_validation_datasets[$index]}"
  if [[ -s "$train_dataset" && -s "$validation_dataset" ]]; then
    train_datasets+=("$train_dataset")
    validation_datasets+=("$validation_dataset")
    active_channels+=("${candidate_channels[$index]}")
    active_weights+=("${candidate_weights[$index]}")
  elif [[ -s "$train_dataset" || -s "$validation_dataset" ]]; then
    echo "curriculum channel has only one non-empty split: ${candidate_channels[$index]}" >&2
    exit 2
  else
    echo "Skipping absent curriculum channel: ${candidate_channels[$index]}" >&2
  fi
done
if [[ "${#active_channels[@]}" -lt 1 ]]; then
  echo "no non-empty curriculum channels are available" >&2
  exit 2
fi
weight_sum="$({ printf '%s\n' "${active_weights[@]}"; } | awk '{sum += $1} END {printf "%.12g", sum}')"
interleave_prob=()
for weight in "${active_weights[@]}"; do
  interleave_prob+=("$(awk -v numerator="$weight" -v denominator="$weight_sum" 'BEGIN {printf "%.12g", numerator / denominator}')")
done

dataset_args=()
if [[ -n "${IFV_CACHED_DATASET:-}" ]]; then
  read -r -a cached_train_datasets <<<"$IFV_CACHED_DATASET"
  for cached_dataset in "${cached_train_datasets[@]}"; do
    if [[ ! -d "$cached_dataset" ]]; then
      echo "cached training dataset does not exist: $cached_dataset" >&2
      exit 2
    fi
  done
  dataset_args+=(--cached_dataset "${cached_train_datasets[@]}")
  if [[ -n "${IFV_CACHED_VAL_DATASET:-}" ]]; then
    read -r -a cached_val_datasets <<<"$IFV_CACHED_VAL_DATASET"
    for cached_dataset in "${cached_val_datasets[@]}"; do
      if [[ ! -d "$cached_dataset" ]]; then
        echo "cached validation dataset does not exist: $cached_dataset" >&2
        exit 2
      fi
    done
    dataset_args+=(--cached_val_dataset "${cached_val_datasets[@]}")
  fi
else
  dataset_args+=(
    --dataset "${train_datasets[@]}"
    --val_dataset "${validation_datasets[@]}"
    --interleave_prob "${interleave_prob[@]}"
    --stopping_strategy all_exhausted
  )
fi

OUTPUT_DIR="$DATA_ROOT/checkpoints/$EXPERIMENT_ID"
EXPERIMENT_DIR="$DATA_ROOT/logs/$EXPERIMENT_ID"
LOG_DIR="$EXPERIMENT_DIR"
new_output_dir "$OUTPUT_DIR"
new_output_dir "$EXPERIMENT_DIR"
record_environment "$EXPERIMENT_DIR"
{
  printf 'channel\tweight\ttrain_dataset\tvalidation_dataset\n'
  for index in "${!active_channels[@]}"; do
    printf '%s\t%s\t%s\t%s\n' \
      "${active_channels[$index]}" \
      "${interleave_prob[$index]}" \
      "${train_datasets[$index]}" \
      "${validation_datasets[$index]}"
  done
} >"$EXPERIMENT_DIR/curriculum-selection.tsv"

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
set +e
"${args[@]}" 2>&1 | tee "$LOG_DIR/train.log"
train_status="${PIPESTATUS[0]}"
set -e
profile_status=0
python -m ifv_training training-profile \
  --train-log "$LOG_DIR/train.log" \
  --output "$LOG_DIR/profile.json" \
  --experiment-id "$EXPERIMENT_ID" \
  --profile-id "$(basename "$SFT_PROFILE")" || profile_status="$?"
if [[ "$train_status" -eq 0 && "$profile_status" -ne 0 ]]; then
  exit "$profile_status"
fi
exit "$train_status"
