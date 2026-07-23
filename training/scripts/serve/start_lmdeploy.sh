#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"

if [[ "$#" -lt 4 || "$#" -gt 7 ]]; then
  echo "usage: $0 MODEL_DIR SERVED_NAME PORT TENSOR_PARALLEL_SIZE [CONTEXT_LENGTH] [SERVING_ROLE] [CHECKPOINT_MANIFEST]" >&2
  exit 2
fi

MODEL="$1"
SERVED_NAME="$2"
PORT="$3"
TP_SIZE="$4"
CONTEXT_LENGTH="${5:-32768}"
SERVING_ROLE="${6:-agent}"
CHECKPOINT_MANIFEST="${7:-}"
require_idle_runtime_gpus
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
if [[ "$SERVING_ROLE" != "agent" && "$SERVING_ROLE" != "rollout" ]]; then
  echo "SERVING_ROLE must be agent or rollout" >&2
  exit 2
fi

TOOL_CALL_PARSER="${IFV_LMDEPLOY_TOOL_CALL_PARSER:-qwen3}"
REASONING_PARSER="${IFV_LMDEPLOY_REASONING_PARSER:-default}"
PROFILE_DIR="$DATA_ROOT/exports/$SERVED_NAME"
mkdir -p "$PROFILE_DIR"
profile_args=(
  python -m ifv_training serving-profile
  --output "$PROFILE_DIR/serving-profile.json"
  --profile-id "$SERVED_NAME"
  --model-path "$MODEL"
  --engine lmdeploy
  --port "$PORT"
  --tensor-parallel-size "$TP_SIZE"
  --dtype bfloat16
  --context-length "$CONTEXT_LENGTH"
  --tool-call-parser "$TOOL_CALL_PARSER"
  --reasoning-parser "$REASONING_PARSER"
  --thinking-enabled true
)
if [[ -n "$CHECKPOINT_MANIFEST" ]]; then
  profile_args+=(--checkpoint-manifest "$CHECKPOINT_MANIFEST")
fi
"${profile_args[@]}"

# The validated gpu-13 installation falls back to LMDeploy's PyTorch engine.
# Use uni/mp rather than Ray so an unrelated node-wide RSS spike cannot kill TP=1.
EXECUTOR_BACKEND=mp
if [[ "$TP_SIZE" -eq 1 ]]; then
  EXECUTOR_BACKEND=uni
fi
args=(
  lmdeploy serve api_server "$MODEL"
  --server-name 127.0.0.1
  --server-port "$PORT"
  --model-name "$SERVED_NAME"
  --backend pytorch
  --dtype bfloat16
  --tp "$TP_SIZE"
  --session-len "$CONTEXT_LENGTH"
  --tool-call-parser "$TOOL_CALL_PARSER"
  --reasoning-parser "$REASONING_PARSER"
  --distributed-executor-backend "$EXECUTOR_BACKEND"
  --max-concurrent-requests 4
  --enable-abort-handling
)
if [[ "$SERVING_ROLE" == "rollout" ]]; then
  args+=(--logprobs-mode raw_logprobs)
fi
print_command "${args[@]}"
"${args[@]}"
