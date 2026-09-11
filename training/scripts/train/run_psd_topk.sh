#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"

if [[ "$#" -lt 4 || "$#" -gt 5 ]]; then
  echo "usage: $0 MODEL_PROFILE PSD_PROFILE PSD_DATUMS_JSONL EXPERIMENT_ID [RESUME_CHECKPOINT]" >&2
  exit 2
fi

MODEL_PROFILE="$1"
PSD_PROFILE="$2"
PSD_DATUMS="$3"
EXPERIMENT_ID="$4"
RESUME_CHECKPOINT="${5:-}"

load_profile "$MODEL_PROFILE"
load_profile "$PSD_PROFILE"
configure_training_runtime
require_idle_gpus
configure_distributed_backend
require_model_path
require_dataset "$PSD_DATUMS"
require_value EXPERIMENT_ID

if [[ "${IFV_PADDING_FREE:-false}" != "false" ]]; then
  echo "PSD top-k training requires IFV_PADDING_FREE=false" >&2
  exit 2
fi
if [[ "${IFV_SEQUENCE_PARALLEL_SIZE:-1}" != "1" ]]; then
  echo "PSD top-k training requires IFV_SEQUENCE_PARALLEL_SIZE=1" >&2
  exit 2
fi

training_backend_args=()
if [[ -n "${IFV_DEEPSPEED:-}" ]]; then
  training_backend_args+=(--deepspeed "$IFV_DEEPSPEED")
fi
if [[ -n "${IFV_FSDP:-}" ]]; then
  training_backend_args+=(--fsdp "$IFV_FSDP")
fi

OUTPUT_DIR="$DATA_ROOT/checkpoints/$EXPERIMENT_ID"
LOG_DIR="$DATA_ROOT/logs/$EXPERIMENT_ID"
new_output_dir "$OUTPUT_DIR"
new_output_dir "$LOG_DIR"
record_environment "$LOG_DIR"

export PYTHONPATH="$REPO_ROOT/training${PYTHONPATH:+:$PYTHONPATH}"
PSD_INPUT_GATE="$LOG_DIR/psd-input-gate.json"
PSD_DATUM_MANIFEST="${IFV_PSD_DATUM_MANIFEST:-$(dirname "$PSD_DATUMS")/manifest.json}"
python -m ifv_training verify-psd-datums \
  --datums "$PSD_DATUMS" \
  --manifest "$PSD_DATUM_MANIFEST" \
  --expected-topk "${IFV_PSD_TOPK:-20}" \
  --max-context "$IFV_MAX_LENGTH" \
  --output "$PSD_INPUT_GATE"

PSD_PLUGIN_PREFLIGHT="$LOG_DIR/psd-plugin-preflight.json"
python "$REPO_ROOT/training/scripts/probe/psd_ms_swift_plugin_smoke.py" \
  --plugin "$REPO_ROOT/training/plugins/ifv_psd_topk_plugin.py" \
  --output "$PSD_PLUGIN_PREFLIGHT"

ENVIRONMENT_PREFLIGHT=""
if [[ "${IFV_REQUIRE_TRAINING_ENV_PREFLIGHT:-false}" == "true" ]]; then
  for name in \
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
  ENVIRONMENT_PREFLIGHT="$LOG_DIR/environment-preflight.json"
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
    --expected-gpu-count "$NPROC_PER_NODE" \
    --expected-gpu-name "${IFV_EXPECTED_GPU_NAME:-NVIDIA A100-SXM4-40GB}" \
    --expected-gpu-memory-mib "${IFV_EXPECTED_GPU_MEMORY_MIB:-40960}" \
    --gpu-memory-tolerance-mib "${IFV_GPU_MEMORY_TOLERANCE_MIB:-128}" \
    --output "$ENVIRONMENT_PREFLIGHT"
fi

CHECKPOINT_PREFLIGHT="$LOG_DIR/checkpoint-storage-preflight.json"
python -m ifv_training checkpoint-storage-preflight \
  --output-dir "$OUTPUT_DIR" \
  --estimated-checkpoint-bytes "${IFV_PSD_CHECKPOINT_ESTIMATED_BYTES:-25000000000}" \
  --reserve-multiplier "${IFV_CHECKPOINT_RESERVE_MULTIPLIER:-1.25}" \
  --output "$CHECKPOINT_PREFLIGHT"

args=(
  swift sft
  --model "$IFV_MODEL_ID"
  --dataset "$PSD_DATUMS"
  --split_dataset_ratio 0
  --strict true
  --template ifv_psd_topk
  --loss_type ifv_psd_topk
  --external_plugins "$REPO_ROOT/training/plugins/ifv_psd_topk_plugin.py"
  --remove_unused_columns false
  --padding_free false
  --sequence_parallel_size 1
  --max_length "$IFV_MAX_LENGTH"
  --truncation_strategy raise
  --tuner_type "$IFV_TUNER_TYPE"
  --torch_dtype "$IFV_TORCH_DTYPE"
  --per_device_train_batch_size "$IFV_TRAIN_BATCH_SIZE"
  --gradient_accumulation_steps "$IFV_GRADIENT_ACCUMULATION_STEPS"
  --learning_rate "$IFV_LEARNING_RATE"
  --gradient_checkpointing "${IFV_GRADIENT_CHECKPOINTING:-true}"
  --logging_steps "$IFV_LOGGING_STEPS"
  --save_strategy steps
  --save_steps "$IFV_SAVE_STEPS"
  --save_total_limit "$IFV_SAVE_TOTAL_LIMIT"
  --output_dir "$OUTPUT_DIR"
  --dataset_num_proc "${IFV_DATASET_NUM_PROC:-1}"
  --dataloader_num_workers "${IFV_DATALOADER_NUM_WORKERS:-0}"
  --report_to tensorboard
  "${training_backend_args[@]}"
)

if [[ -n "$RESUME_CHECKPOINT" ]]; then
  if [[ ! -d "$RESUME_CHECKPOINT" ]]; then
    echo "resume checkpoint does not exist: $RESUME_CHECKPOINT" >&2
    exit 2
  fi
  args+=(--resume_from_checkpoint "$RESUME_CHECKPOINT")
fi

if [[ -n "${IFV_NUM_TRAIN_EPOCHS:-}" ]]; then
  args+=(--num_train_epochs "$IFV_NUM_TRAIN_EPOCHS")
elif [[ -n "${IFV_MAX_STEPS:-}" ]]; then
  args+=(--max_steps "$IFV_MAX_STEPS")
else
  echo "PSD profile must set IFV_NUM_TRAIN_EPOCHS or IFV_MAX_STEPS" >&2
  exit 2
fi
if [[ -n "${IFV_BF16:-}" ]]; then
  args+=(--bf16 "$IFV_BF16")
fi
if [[ -n "${IFV_FP16:-}" ]]; then
  args+=(--fp16 "$IFV_FP16")
fi
if [[ "${IFV_GRADIENT_CHECKPOINTING:-true}" == "true" ]]; then
  args+=(--gradient_checkpointing_kwargs '{"use_reentrant": false}')
fi

print_command "${args[@]}"
set +e
python "$SCRIPT_DIR/run_with_resource_monitor.py" \
  --summary-output "$LOG_DIR/resource-summary.json" \
  --samples-output "$LOG_DIR/resource-samples.jsonl" \
  --gpu-ids "$CUDA_VISIBLE_DEVICES" \
  --sample-interval "${IFV_RESOURCE_SAMPLE_INTERVAL:-2}" \
  -- "${args[@]}" 2>&1 | tee "$LOG_DIR/train.log"
train_status="${PIPESTATUS[0]}"
set -e

python -m ifv_training training-profile \
  --train-log "$LOG_DIR/train.log" \
  --output "$LOG_DIR/profile.json" \
  --steady-window "${IFV_TRAINING_STEADY_WINDOW:-5}" \
  --experiment-id "$EXPERIMENT_ID" \
  --profile-id "$(basename "$PSD_PROFILE")" \
  --resource-summary "$LOG_DIR/resource-summary.json" \
  --dataset-verification "$PSD_INPUT_GATE" \
  --environment-preflight "$ENVIRONMENT_PREFLIGHT" \
  --checkpoint-preflight "$CHECKPOINT_PREFLIGHT" \
  --train-exit-code "$train_status" || true

exit "$train_status"
