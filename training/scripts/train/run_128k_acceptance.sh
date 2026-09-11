#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"

if [[ "$#" -ne 6 ]]; then
  echo "usage: $0 MODEL_PROFILE TRAIN_JSONL VAL_JSONL PROCESSOR_REPORT RUN_PREFIX PLAN_OUTPUT" >&2
  exit 2
fi

MODEL_PROFILE="$1"
TRAIN_JSONL="$2"
VAL_JSONL="$3"
PROCESSOR_REPORT="$4"
RUN_PREFIX="$5"
PLAN_OUTPUT="$6"

case "$RUN_PREFIX" in
  *[!A-Za-z0-9._-]*|'')
    echo "RUN_PREFIX must be a non-empty filesystem-safe name" >&2
    exit 2
    ;;
esac

MEMORY_PROFILE="$REPO_ROOT/training/configs/sft/qwen3.5-full-1step-8gpu-fsdp2-sp8-flash-128k-memory-probe.env"
CANARY_PROFILE="$REPO_ROOT/training/configs/sft/qwen3.5-full-10step-8gpu-fsdp2-sp8-flash-128k-canary.env"
RESUME_PROFILE="$REPO_ROOT/training/configs/sft/qwen3.5-full-11step-8gpu-fsdp2-sp8-flash-128k-resume.env"
ACCEPTANCE_DIR="$DATA_ROOT/logs/${RUN_PREFIX}-128k-acceptance"
mkdir -p "$ACCEPTANCE_DIR"

export IFV_PROCESSOR_VERIFICATION="$PROCESSOR_REPORT"
python -m ifv_training long-context-plan \
  --processor-report "$PROCESSOR_REPORT" \
  --train-jsonl "$TRAIN_JSONL" \
  --sft-profile "$MEMORY_PROFILE" \
  --sft-profile "$CANARY_PROFILE" \
  --sft-profile "$RESUME_PROFILE" \
  --output "$PLAN_OUTPUT"

gate_passed() {
  local path="$1"
  [[ -s "$path" ]] && python - "$path" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    value = json.load(handle)
raise SystemExit(0 if value.get("passed") is True else 1)
PY
}

run_stage() {
  local stage="$1"
  local profile="$2"
  local experiment_id="$3"
  local gate="$4"
  local prior_gate="${5:-}"
  local resume_checkpoint="${6:-}"
  local training_profile="$DATA_ROOT/logs/$experiment_id/profile.json"

  if gate_passed "$gate"; then
    echo "Reusing passed $stage gate: $gate"
    return 0
  fi
  if [[ -e "$DATA_ROOT/logs/$experiment_id" || -e "$DATA_ROOT/checkpoints/$experiment_id" ]]; then
    echo "Existing incomplete $stage artifacts require same-directory diagnosis/recovery: $experiment_id" >&2
    exit 2
  fi

  if [[ -n "$resume_checkpoint" ]]; then
    "$SCRIPT_DIR/run_sft.sh" \
      "$MODEL_PROFILE" "$profile" "$TRAIN_JSONL" "$VAL_JSONL" \
      "$experiment_id" "$resume_checkpoint"
  else
    "$SCRIPT_DIR/run_sft.sh" \
      "$MODEL_PROFILE" "$profile" "$TRAIN_JSONL" "$VAL_JSONL" \
      "$experiment_id"
  fi

  gate_args=(
    python -m ifv_training validate-128k-stage
    --stage "$stage"
    --training-profile "$training_profile"
    --sft-profile "$profile"
    --output "$gate"
  )
  if [[ -n "$prior_gate" ]]; then
    gate_args+=(--prior-gate "$prior_gate")
  fi
  "${gate_args[@]}"
}

MEMORY_EXPERIMENT="${RUN_PREFIX}-128k-memory-probe"
CANARY_EXPERIMENT="${RUN_PREFIX}-128k-canary"
RESUME_EXPERIMENT="${RUN_PREFIX}-128k-resume"
MEMORY_GATE="$ACCEPTANCE_DIR/memory-probe-gate.json"
CANARY_GATE="$ACCEPTANCE_DIR/canary-gate.json"
RESUME_GATE="$ACCEPTANCE_DIR/resume-gate.json"

run_stage memory-probe "$MEMORY_PROFILE" "$MEMORY_EXPERIMENT" "$MEMORY_GATE"
run_stage canary "$CANARY_PROFILE" "$CANARY_EXPERIMENT" "$CANARY_GATE" "$MEMORY_GATE"

CANARY_CHECKPOINT="$DATA_ROOT/checkpoints/$CANARY_EXPERIMENT/checkpoint-10"
if [[ ! -d "$CANARY_CHECKPOINT" ]]; then
  echo "passed canary did not leave checkpoint-10: $CANARY_CHECKPOINT" >&2
  exit 2
fi
run_stage resume "$RESUME_PROFILE" "$RESUME_EXPERIMENT" "$RESUME_GATE" "$CANARY_GATE" "$CANARY_CHECKPOINT"

echo "128K acceptance chain passed: $RESUME_GATE"
