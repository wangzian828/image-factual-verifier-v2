#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"

if [[ "$#" -lt 4 || "$#" -gt 7 ]]; then
  echo "usage: $0 MODEL_DIR SERVED_NAME PORT TENSOR_PARALLEL_SIZE [CONTEXT_LENGTH] [THINKING_ENABLED] [CHECKPOINT_MANIFEST]" >&2
  exit 2
fi

MODEL="$1"
SERVED_NAME="$2"
PORT="$3"
TP_SIZE="$4"
CONTEXT_LENGTH="${5:-32768}"
THINKING_ENABLED="${6:-false}"
CHECKPOINT_MANIFEST="${7:-}"
require_idle_gpus
export OMP_NUM_THREADS=1

if [[ ! -e "$MODEL" && "$MODEL" == /* ]]; then
  echo "model path does not exist: $MODEL" >&2
  exit 2
fi
if [[ "$TP_SIZE" -lt 1 ]]; then
  echo "tensor parallel size must be positive" >&2
  exit 2
fi
IFS=',' read -r -a visible_devices <<<"$CUDA_VISIBLE_DEVICES"
if [[ "$TP_SIZE" -gt "${#visible_devices[@]}" ]]; then
  echo "tensor parallel size $TP_SIZE exceeds visible GPU count ${#visible_devices[@]}" >&2
  exit 2
fi
if [[ "$THINKING_ENABLED" != "true" && "$THINKING_ENABLED" != "false" ]]; then
  echo "THINKING_ENABLED must be true or false" >&2
  exit 2
fi

PROFILE_DIR="$DATA_ROOT/exports/$SERVED_NAME"
mkdir -p "$PROFILE_DIR"
profile_args=(
  python -m ifv_training serving-profile
  --output "$PROFILE_DIR/serving-profile.json"
  --profile-id "$SERVED_NAME"
  --model-path "$MODEL"
  --engine vllm
  --port "$PORT"
  --tensor-parallel-size "$TP_SIZE"
  --dtype bfloat16
  --context-length "$CONTEXT_LENGTH"
  --tool-call-parser qwen3_coder
  --reasoning-parser qwen3
  --thinking-enabled "$THINKING_ENABLED"
)
if [[ -n "$CHECKPOINT_MANIFEST" ]]; then
  profile_args+=(--checkpoint-manifest "$CHECKPOINT_MANIFEST")
fi
"${profile_args[@]}"

args=(
  vllm serve "$MODEL"
  --host 127.0.0.1
  --port "$PORT"
  --served-model-name "$SERVED_NAME"
  --tensor-parallel-size "$TP_SIZE"
  --max-model-len "$CONTEXT_LENGTH"
  --dtype bfloat16
  --reasoning-parser qwen3
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
  --default-chat-template-kwargs "{\"enable_thinking\": $THINKING_ENABLED}"
)
print_command "${args[@]}"
"${args[@]}"
