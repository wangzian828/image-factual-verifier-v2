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
require_idle_gpus
require_full_parameter_profile
require_model_path
require_dataset "$TRAIN_DATASET"
require_dataset "$VAL_DATASET"
require_value EXPERIMENT_ID

OUTPUT_DIR="$DATA_ROOT/checkpoints/$EXPERIMENT_ID"
EXPERIMENT_DIR="$DATA_ROOT/logs/$EXPERIMENT_ID"
LOG_DIR="$EXPERIMENT_DIR"
new_output_dir "$OUTPUT_DIR"
new_output_dir "$EXPERIMENT_DIR"
record_environment "$EXPERIMENT_DIR"

args=(
  swift sft
  --model "$IFV_MODEL_ID"
  --dataset "$TRAIN_DATASET"
  --val_dataset "$VAL_DATASET"
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
  --gradient_checkpointing true
  --vit_gradient_checkpointing "$IFV_VIT_GRADIENT_CHECKPOINTING"
  --gradient_checkpointing_kwargs '{"use_reentrant": false}'
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
  --deepspeed "$IFV_DEEPSPEED"
  --dataset_num_proc 2
  --dataloader_num_workers 2
  --report_to tensorboard
)

if [[ "${IFV_ADD_NON_THINKING_PREFIX:-false}" == "true" ]]; then
  args+=(--add_non_thinking_prefix true)
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
"${args[@]}" 2>&1 | tee "$LOG_DIR/train.log"
