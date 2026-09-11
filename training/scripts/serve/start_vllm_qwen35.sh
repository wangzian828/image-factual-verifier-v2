#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"

MODEL="${1:-${IFV_QWEN35_MODEL:-${IFV_MODEL_ID:-}}}"
SERVED_NAME="${2:-ifv-qwen3.5-9b}"
PORT="${3:-8901}"
TP_SIZE="${4:-2}"
CONTEXT_LENGTH="${5:-131072}"
CHECKPOINT_MANIFEST="${6:-${IFV_CHECKPOINT_MANIFEST:-}}"
ENV_PREFIX="${IFV_VLLM_ENV_PREFIX:-${CONDA_PREFIX:-}}"
VLLM="$ENV_PREFIX/bin/vllm"

if [[ -z "$MODEL" ]]; then
  echo "set IFV_QWEN35_MODEL/IFV_MODEL_ID or pass MODEL as argument 1" >&2
  exit 2
fi
if [[ -z "$ENV_PREFIX" ]]; then
  echo "set IFV_VLLM_ENV_PREFIX or activate the vLLM environment" >&2
  exit 2
fi
if [[ ! -x "$VLLM" || ! -f "$ENV_PREFIX/.ifv-vllm-qwen35-ready" ]]; then
  echo "frozen Qwen3.5 vLLM environment is absent or incomplete: $ENV_PREFIX" >&2
  exit 2
fi
if [[ ! -d "$MODEL" ]]; then
  echo "model directory does not exist: $MODEL" >&2
  exit 2
fi
if [[ "$CONTEXT_LENGTH" -ne 131072 ]]; then
  echo "this validated profile requires a 131072-token context" >&2
  exit 2
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "set CUDA_VISIBLE_DEVICES explicitly for this server" >&2
  exit 2
fi
require_idle_runtime_gpus
IFS=',' read -r -a visible_devices <<<"$CUDA_VISIBLE_DEVICES"
if [[ "$TP_SIZE" -ne "${#visible_devices[@]}" ]]; then
  echo "tensor parallel size $TP_SIZE must equal visible GPU count ${#visible_devices[@]}" >&2
  exit 2
fi
if ss -ltnH "sport = :$PORT" | grep -q .; then
  echo "port is already listening: $PORT" >&2
  exit 2
fi

export OMP_NUM_THREADS=1
export NCCL_CUMEM_HOST_ENABLE=0
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="$NO_PROXY"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# The frozen serving image intentionally has no CUDA toolkit.  vLLM's optional
# FlashInfer sampler JIT-compiles an extension on first use and therefore needs
# nvcc; use the built-in sampler for this runtime-only environment.
export VLLM_USE_FLASHINFER_SAMPLER=0
PROFILE_DIR="$DATA_ROOT/exports/$SERVED_NAME"
mkdir -p "$PROFILE_DIR"
profile_args=(
  "$ENV_PREFIX/bin/python" -m ifv_training serving-profile
  --output "$PROFILE_DIR/serving-profile.json" \
  --profile-id "$SERVED_NAME" \
  --model-path "$MODEL" \
  --engine vllm \
  --port "$PORT" \
  --tensor-parallel-size "$TP_SIZE" \
  --dtype bfloat16 \
  --context-length "$CONTEXT_LENGTH" \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --thinking-enabled true
)
if [[ -n "$CHECKPOINT_MANIFEST" ]]; then
  if [[ ! -s "$CHECKPOINT_MANIFEST" ]]; then
    echo "checkpoint manifest does not exist or is empty: $CHECKPOINT_MANIFEST" >&2
    exit 2
  fi
  profile_args+=(--checkpoint-manifest "$CHECKPOINT_MANIFEST")
fi
PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" "${profile_args[@]}"

args=(
  "$VLLM" serve "$MODEL"
  --host 127.0.0.1
  --port "$PORT"
  --served-model-name "$SERVED_NAME"
  --dtype bfloat16
  --tensor-parallel-size "$TP_SIZE"
  --disable-custom-all-reduce
  --max-model-len "$CONTEXT_LENGTH"
  --gpu-memory-utilization 0.92
  --max-num-seqs 4
  --max-num-batched-tokens 8192
  --reasoning-parser qwen3
  --structured-outputs-config '{"backend":"xgrammar","reasoning_parser":"qwen3","disable_any_whitespace":true}'
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
  --default-chat-template-kwargs '{"enable_thinking":false}'
  --limit-mm-per-prompt '{"image":1,"video":0}'
  --enable-tokenizer-info-endpoint
  --max-log-len 4000
  --disable-uvicorn-access-log
)
print_command "${args[@]}"
exec "${args[@]}"
