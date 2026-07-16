#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"

if [[ "$#" -ne 3 ]]; then
  echo "usage: $0 MODEL_PROFILE RL_PROFILE EXPERIMENT_ID" >&2
  exit 2
fi

MODEL_PROFILE="$1"
RL_PROFILE="$2"
EXPERIMENT_ID="$3"
load_profile "$MODEL_PROFILE"
load_profile "$RL_PROFILE"
require_two_gpus
require_model_path
require_value EXPERIMENT_ID

DATASET="$REPO_ROOT/fixtures/rl/mock_search.jsonl"
require_dataset "$DATASET"
OUTPUT_DIR="$DATA_ROOT/checkpoints/$EXPERIMENT_ID"
EXPERIMENT_DIR="$DATA_ROOT/logs/$EXPERIMENT_ID"
LOG_DIR="$EXPERIMENT_DIR"
new_output_dir "$OUTPUT_DIR"
new_output_dir "$EXPERIMENT_DIR"
record_environment "$EXPERIMENT_DIR"

export IMAGE_MAX_TOKEN_NUM="$IFV_IMAGE_MAX_TOKEN_NUM"
args=(
  swift rlhf
  --rlhf_type "$IFV_RLHF_TYPE"
  --model "$IFV_MODEL_ID"
  --tuner_type "$IFV_TUNER_TYPE"
  --dataset "$DATASET"
  --split_dataset_ratio 0
  --strict true
  --load_from_cache_file false
  --external_plugins "$REPO_ROOT/ifv_training/rl_mock_plugin.py"
  --gym_env ifv_mock_search
  --use_gym_env true
  --multi_turn_scheduler gym_scheduler
  --max_turns "$IFV_MAX_TURNS"
  --use_vllm true
  --vllm_mode colocate
  --vllm_gpu_memory_utilization "$IFV_VLLM_GPU_MEMORY_UTILIZATION"
  --vllm_tensor_parallel_size 1
  --vllm_server_pass_dataset true
  --enable_thinking false
  --torch_dtype "$IFV_TORCH_DTYPE"
  --max_steps "$IFV_MAX_STEPS"
  --max_length "$IFV_MAX_LENGTH"
  --max_completion_length "$IFV_MAX_COMPLETION_LENGTH"
  --per_device_train_batch_size "$IFV_TRAIN_BATCH_SIZE"
  --gradient_accumulation_steps "$IFV_GRADIENT_ACCUMULATION_STEPS"
  --learning_rate "$IFV_LEARNING_RATE"
  --num_generations "$IFV_NUM_GENERATIONS"
  --steps_per_generation "$IFV_STEPS_PER_GENERATION"
  --save_steps "$IFV_SAVE_STEPS"
  --save_total_limit 2
  --logging_steps "$IFV_LOGGING_STEPS"
  --gradient_checkpointing true
  --deepspeed "$IFV_DEEPSPEED"
  --temperature 1.0
  --log_completions true
  --report_to tensorboard
  --output_dir "$OUTPUT_DIR"
)
print_command "${args[@]}"
"${args[@]}" 2>&1 | tee "$LOG_DIR/grpo.log"
