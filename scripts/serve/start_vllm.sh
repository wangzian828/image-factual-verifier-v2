#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"

if [[ "$#" -lt 4 || "$#" -gt 5 ]]; then
  echo "usage: $0 MERGED_MODEL_DIR SERVED_NAME PORT TENSOR_PARALLEL_SIZE [CHECKPOINT_MANIFEST]" >&2
  exit 2
fi

MODEL="$1"
SERVED_NAME="$2"
PORT="$3"
TP_SIZE="$4"
CHECKPOINT_MANIFEST="${5:-}"
require_value CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=1

if [[ ! -e "$MODEL" && "$MODEL" == /* ]]; then
  echo "model path does not exist: $MODEL" >&2
  exit 2
fi
if [[ "$TP_SIZE" -lt 1 ]]; then
  echo "tensor parallel size must be positive" >&2
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
  --context-length 16384
)
if [[ -n "$CHECKPOINT_MANIFEST" ]]; then
  profile_args+=(--checkpoint-manifest "$CHECKPOINT_MANIFEST")
fi
"${profile_args[@]}"

args=(
  swift deploy
  --model "$MODEL"
  --infer_backend vllm
  --host 127.0.0.1
  --port "$PORT"
  --served_model_name "$SERVED_NAME"
  --vllm_tensor_parallel_size "$TP_SIZE"
  --vllm_max_model_len 16384
  --torch_dtype bfloat16
  --enable_thinking false
)
print_command "${args[@]}"
"${args[@]}"
