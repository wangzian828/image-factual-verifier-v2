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
if [[ "$IFV_PSD_PROFILE_MODE" == "production" ]]; then
  require_value IFV_PSD_ROUND_READY
fi
for name in \
  IFV_PSD_PROFILE_MODE \
  IFV_PSD_TOPK \
  IFV_PSD_LOSS_CHUNK_TOKENS \
  IFV_PADDING_FREE \
  IFV_SEQUENCE_PARALLEL_SIZE \
  IFV_LORA_RANK \
  IFV_LORA_ALPHA \
  IFV_LORA_DROPOUT \
  IFV_LORA_TARGET_MODULES
do
  require_value "$name"
done

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
export IFV_PSD_DATASET_PATH="$PSD_DATUMS"
PSD_INPUT_GATE="$LOG_DIR/psd-input-gate.json"
PSD_DATUM_MANIFEST="${IFV_PSD_DATUM_MANIFEST:-$(dirname "$PSD_DATUMS")/manifest.json}"
python -m ifv_training verify-psd-datums \
  --datums "$PSD_DATUMS" \
  --manifest "$PSD_DATUM_MANIFEST" \
  --expected-topk "${IFV_PSD_TOPK:-20}" \
  --max-context "$IFV_MAX_LENGTH" \
  --output "$PSD_INPUT_GATE"

# Datums deliberately omit private teacher context, but the launch must still
# attest the model used to generate their distribution. A later round uses
# --model BASE --adapters PREVIOUS_ADAPTER, not a new random LoRA from BASE.
require_value IFV_PSD_SERVING_PROFILE
require_value IFV_PSD_ROUND_START_MANIFEST
initialization_args=(
  python "$REPO_ROOT/training/scripts/probe/verify_psd_initialization.py"
  --datum-manifest "$PSD_DATUM_MANIFEST"
  --serving-profile "$IFV_PSD_SERVING_PROFILE"
  --checkpoint-manifest "$IFV_PSD_ROUND_START_MANIFEST"
  --model "$IFV_MODEL_ID"
  --output "$LOG_DIR/psd-initialization-gate.json"
)
if [[ -n "${IFV_PSD_INITIAL_ADAPTER:-}" ]]; then
  initialization_args+=(--adapter "$IFV_PSD_INITIAL_ADAPTER")
fi
"${initialization_args[@]}"
if [[ "$IFV_PSD_PROFILE_MODE" == "production" ]]; then
  PYTHONPATH="$REPO_ROOT:$PYTHONPATH" python - "$IFV_PSD_ROUND_READY" "$PSD_DATUMS" "$LOG_DIR/psd-initialization-gate.json" <<'PY'
import json
import sys
from pathlib import Path
from scripts.run_psd_round import load_ready
ready = load_ready(Path(sys.argv[1]))
if Path(ready['datums']).resolve() != Path(sys.argv[2]).resolve():
    raise SystemExit('PSD launch datums differ from prepared round')
if ready['initialization'] != json.loads(Path(sys.argv[3]).read_text()):
    raise SystemExit('PSD launch initialization differs from prepared round')
PY
fi

PSD_PROFILE_GATE="$LOG_DIR/psd-training-profile-gate.json"
profile_gate_args=(
  python "$REPO_ROOT/training/scripts/probe/verify_psd_training_profile.py"
  --mode "$IFV_PSD_PROFILE_MODE"
  --world-size "$NPROC_PER_NODE"
  --sequence-parallel-size "$IFV_SEQUENCE_PARALLEL_SIZE"
  --padding-free "$IFV_PADDING_FREE"
  --max-context "$IFV_MAX_LENGTH"
  --topk "$IFV_PSD_TOPK"
  --loss-chunk-tokens "$IFV_PSD_LOSS_CHUNK_TOKENS"
  --tuner-type "$IFV_TUNER_TYPE"
  --lora-rank "$IFV_LORA_RANK"
  --lora-alpha "$IFV_LORA_ALPHA"
  --lora-dropout "$IFV_LORA_DROPOUT"
  --target-modules "$IFV_LORA_TARGET_MODULES"
  --train-batch-size "$IFV_TRAIN_BATCH_SIZE"
  --gradient-accumulation-steps "$IFV_GRADIENT_ACCUMULATION_STEPS"
  --learning-rate "$IFV_LEARNING_RATE"
  --lr-scheduler-type constant
  --adam-beta1 0.9
  --adam-beta2 0.95
  --adam-epsilon 1e-12
  --weight-decay 0
  --loss-reduction sum
  --output "$PSD_PROFILE_GATE"
)
if [[ -n "${IFV_NUM_TRAIN_EPOCHS:-}" ]]; then
  profile_gate_args+=(--num-train-epochs "$IFV_NUM_TRAIN_EPOCHS")
fi
if [[ -n "${IFV_MAX_STEPS:-}" ]]; then
  profile_gate_args+=(--max-steps "$IFV_MAX_STEPS")
fi
"${profile_gate_args[@]}"

resume_gate_args=(
  python "$REPO_ROOT/training/scripts/probe/verify_psd_resume.py"
  --output-root "$OUTPUT_DIR"
  --datum-manifest-path "$PSD_DATUM_MANIFEST"
  --initialization-gate-path "$LOG_DIR/psd-initialization-gate.json"
  --profile-gate-path "$PSD_PROFILE_GATE"
  --seed "${IFV_SEED:-0}"
  --max-grad-norm "${IFV_MAX_GRAD_NORM:-1.0}"
)
if [[ -n "$RESUME_CHECKPOINT" ]]; then
  resume_gate_args+=(--checkpoint "$RESUME_CHECKPOINT")
fi
"${resume_gate_args[@]}"

PSD_PLUGIN_PREFLIGHT="$LOG_DIR/psd-plugin-preflight.json"
python "$REPO_ROOT/training/scripts/probe/psd_ms_swift_plugin_smoke.py" \
  --plugin "$REPO_ROOT/training/plugins/ifv_psd_topk_plugin.py" \
  --output "$PSD_PLUGIN_PREFLIGHT"
CUDA_VISIBLE_DEVICES="" python "$REPO_ROOT/training/scripts/probe/psd_trainer_accumulation_smoke.py" \
  --output "$LOG_DIR/psd-trainer-accumulation-preflight.json"

PSD_SP_PREFLIGHT="$LOG_DIR/psd-sequence-parallel-preflight.json"
CUDA_VISIBLE_DEVICES="" \
IFV_PSD_SP_SMOKE_OUTPUT="$PSD_SP_PREFLIGHT" \
python -m torch.distributed.run \
  --standalone \
  --nproc_per_node "$IFV_SEQUENCE_PARALLEL_SIZE" \
  "$REPO_ROOT/training/scripts/probe/psd_sequence_parallel_smoke.py"

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
  --padding_free "$IFV_PADDING_FREE"
  --sequence_parallel_size "$IFV_SEQUENCE_PARALLEL_SIZE"
  --max_length "$IFV_MAX_LENGTH"
  --truncation_strategy delete
  --tuner_type "$IFV_TUNER_TYPE"
  --freeze_llm false
  --freeze_vit false
  --freeze_aligner false
  --lora_rank "$IFV_LORA_RANK"
  --lora_alpha "$IFV_LORA_ALPHA"
  --lora_dropout "$IFV_LORA_DROPOUT"
  --target_modules "$IFV_LORA_TARGET_MODULES"
  --torch_dtype "$IFV_TORCH_DTYPE"
  --per_device_train_batch_size "$IFV_TRAIN_BATCH_SIZE"
  --gradient_accumulation_steps "$IFV_GRADIENT_ACCUMULATION_STEPS"
  --learning_rate "$IFV_LEARNING_RATE"
  --lr_scheduler_type constant
  --warmup_steps 0
  --warmup_ratio 0
  --optim adamw_torch
  --adam_beta1 0.9
  --adam_beta2 0.95
  --adam_epsilon 1e-12
  --weight_decay 0
  --save_only_model false
  --max_grad_norm "${IFV_MAX_GRAD_NORM:-1.0}"
  --seed "${IFV_SEED:-0}"
  --gradient_checkpointing "${IFV_GRADIENT_CHECKPOINTING:-true}"
  --logging_steps "$IFV_LOGGING_STEPS"
  --output_dir "$OUTPUT_DIR"
  --dataset_num_proc "${IFV_DATASET_NUM_PROC:-1}"
  --dataloader_num_workers "${IFV_DATALOADER_NUM_WORKERS:-0}"
  --report_to tensorboard
  "${training_backend_args[@]}"
)

SAVE_STRATEGY="${IFV_SAVE_STRATEGY:-steps}"
args+=(--save_strategy "$SAVE_STRATEGY")
if [[ "$SAVE_STRATEGY" != "no" ]]; then
  args+=(--save_steps "$IFV_SAVE_STEPS")
  args+=(--save_total_limit "$IFV_SAVE_TOTAL_LIMIT")
fi
if [[ -n "${IFV_ATTN_IMPL:-}" ]]; then
  args+=(--attn_impl "$IFV_ATTN_IMPL")
fi
if [[ -n "${IFV_GROUP_BY_LENGTH:-}" ]]; then
  args+=(--group_by_length "$IFV_GROUP_BY_LENGTH")
fi
if [[ -n "${IFV_USE_LOGITS_TO_KEEP:-}" ]]; then
  args+=(--use_logits_to_keep "$IFV_USE_LOGITS_TO_KEEP")
fi
if [[ -n "${IFV_DATALOADER_PREFETCH_FACTOR:-}" ]]; then
  args+=(--dataloader_prefetch_factor "$IFV_DATALOADER_PREFETCH_FACTOR")
fi
if [[ -n "${IFV_DATALOADER_PERSISTENT_WORKERS:-}" ]]; then
  args+=(--dataloader_persistent_workers "$IFV_DATALOADER_PERSISTENT_WORKERS")
fi

if [[ -n "$RESUME_CHECKPOINT" ]]; then
  if [[ ! -d "$RESUME_CHECKPOINT" ]]; then
    echo "resume checkpoint does not exist: $RESUME_CHECKPOINT" >&2
    exit 2
  fi
  args+=(--resume_from_checkpoint "$RESUME_CHECKPOINT")
elif [[ -n "${IFV_PSD_INITIAL_ADAPTER:-}" ]]; then
  args+=(--adapters "$IFV_PSD_INITIAL_ADAPTER" --load_args false)
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
  resource_monitor_args+=(--memory-max-imbalance-mib "$IFV_GPU_MEMORY_MAX_IMBALANCE_MIB")
fi
if [[ -n "${IFV_GPU_UTILIZATION_TARGET_MIN_PERCENT:-}" ]]; then
  resource_monitor_args+=(
    --utilization-target-min-percent
    "$IFV_GPU_UTILIZATION_TARGET_MIN_PERCENT"
  )
fi
set +e
"${resource_monitor_args[@]}" -- "${args[@]}" 2>&1 | tee "$LOG_DIR/train.log"
train_status="${PIPESTATUS[0]}"
set -e
stop_watchdog
trap - EXIT
"${watchdog_args[@]}" --once >>"$WATCHDOG_LOG" 2>&1 || true

checkpoint_io_status=0
CHECKPOINT_IO_PROFILE=""
if [[ "$train_status" -eq 0 ]]; then
  latest_checkpoint="$(
    find "$OUTPUT_DIR" -type d -name 'checkpoint-*' -print |
      sort -V |
      tail -n 1
  )"
  if [[ -n "$latest_checkpoint" ]]; then
    CHECKPOINT_IO_PROFILE="$LOG_DIR/checkpoint-io-profile.json"
    python -m ifv_training checkpoint-io-profile \
      --checkpoint "$latest_checkpoint" \
      --output "$CHECKPOINT_IO_PROFILE" || checkpoint_io_status="$?"
  fi
fi

profile_args=(
  python -m ifv_training training-profile
  --train-log "$LOG_DIR/train.log" \
  --output "$LOG_DIR/profile.json" \
  --steady-window "${IFV_TRAINING_STEADY_WINDOW:-5}" \
  --experiment-id "$EXPERIMENT_ID" \
  --profile-id "$(basename "$PSD_PROFILE")" \
  --resource-summary "$RESOURCE_SUMMARY" \
  --dataset-verification "$PSD_INPUT_GATE" \
  --environment-preflight "$ENVIRONMENT_PREFLIGHT" \
  --checkpoint-preflight "$CHECKPOINT_PREFLIGHT" \
  --train-exit-code "$train_status"
)
if [[ -n "$CHECKPOINT_IO_PROFILE" ]]; then
  profile_args+=(--checkpoint-io-profile "$CHECKPOINT_IO_PROFILE")
fi
profile_status=0
"${profile_args[@]}" || profile_status="$?"

if [[ "$train_status" -eq 0 && "$checkpoint_io_status" -ne 0 ]]; then
  exit "$checkpoint_io_status"
fi
if [[ "$train_status" -eq 0 && "$profile_status" -ne 0 ]]; then
  exit "$profile_status"
fi
if [[ "$train_status" -eq 0 && "$IFV_PSD_PROFILE_MODE" == "production" ]]; then
  python - "$LOG_DIR/profile.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    profile = json.load(handle)
if profile.get("passed_production_gate") is not True:
    raise SystemExit("PSD optimizer run did not pass its production gate")
PY
  python "$REPO_ROOT/scripts/run_psd_round.py" finalize \
    --ready "$IFV_PSD_ROUND_READY" --checkpoint "$latest_checkpoint" \
    --training-profile "$LOG_DIR/profile.json" \
    --initialization-gate "$LOG_DIR/psd-initialization-gate.json" \
    --output "$LOG_DIR/round-output"
fi

exit "$train_status"
