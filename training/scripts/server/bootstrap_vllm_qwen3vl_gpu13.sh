#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_PREFIX="${IFV_VLLM_ENV_PREFIX:-}"
MODEL="${IFV_QWEN3VL_MODEL:-${IFV_MODEL_ID:-}}"
ARTIFACT_ROOT="${IFV_TRAINING_DATA_ROOT:-${IFV_DATA_ROOT:+${IFV_DATA_ROOT}/training}}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${XDG_DATA_HOME:-${HOME}/.local/share}/image-factual-verifier/training}"
ARTIFACT_DIR="$ARTIFACT_ROOT/logs/environments/ifv-qwen3vl-vllm0112-locked"
REQUIREMENTS="$REPO_ROOT/requirements/serve-vllm-qwen3vl.txt"
CONSTRAINTS="$REPO_ROOT/requirements/constraints-vllm-qwen3vl.txt"
RESOLVED_CONSTRAINTS="$REPO_ROOT/requirements/constraints-vllm-qwen3vl-resolved.txt"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required" >&2
  exit 2
fi
if [[ -z "$ENV_PREFIX" || -z "$MODEL" ]]; then
  echo "set IFV_VLLM_ENV_PREFIX and IFV_QWEN3VL_MODEL/IFV_MODEL_ID" >&2
  exit 2
fi
if [[ -e "$ENV_PREFIX" ]]; then
  echo "refusing to modify an existing environment: $ENV_PREFIX" >&2
  echo "choose a new IFV_VLLM_ENV_PREFIX or explicitly remove this failed build" >&2
  exit 2
fi
if [[ ! -d "$MODEL" ]]; then
  echo "model directory does not exist: $MODEL" >&2
  exit 2
fi

# gpu-13's validated egress proxy.  Never inherit the retired 47894 endpoint.
proxy="${IFV_HTTP_PROXY:-${IFV_SERVER_PROXY:-${http_proxy:-}}}"
if [[ -n "$proxy" ]]; then
  export http_proxy="$proxy"
  export https_proxy="${IFV_HTTPS_PROXY:-$proxy}"
  export HTTP_PROXY="$http_proxy"
  export HTTPS_PROXY="$https_proxy"
fi
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="$NO_PROXY"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export OMP_NUM_THREADS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p "$(dirname "$ENV_PREFIX")" "$ARTIFACT_DIR"
printf 'creating a fresh environment at %s\n' "$ENV_PREFIX"
conda create -y -p "$ENV_PREFIX" python=3.11 pip=25.2

PYTHON="$ENV_PREFIX/bin/python"
"$PYTHON" -m pip install \
  --constraint "$CONSTRAINTS" \
  --constraint "$RESOLVED_CONSTRAINTS" \
  --requirement "$REQUIREMENTS"
"$PYTHON" -m pip install --no-deps --editable "$REPO_ROOT"
"$PYTHON" -m pip check

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES}" \
  "$PYTHON" "$REPO_ROOT/scripts/probe/verify_vllm_environment.py" \
  --model "$MODEL" \
  --output "$ARTIFACT_DIR/verification.json"
# gpu-13's R580 driver crashes inside NCCL 2.27's default cuMem host path.
# Verify the exact tensor-parallel communication path, not only single-GPU CUDA.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES}" \
  NCCL_CUMEM_HOST_ENABLE=0 \
  "$ENV_PREFIX/bin/torchrun" --standalone --nproc-per-node=2 \
  "$REPO_ROOT/scripts/probe/verify_nccl_tensor_parallel.py" \
  --expected-world-size 2 \
  --output "$ARTIFACT_DIR/nccl-tensor-parallel.json"
"$PYTHON" -m pip freeze --all >"$ARTIFACT_DIR/pip-freeze.txt"
sha256sum "$ARTIFACT_DIR/pip-freeze.txt" >"$ARTIFACT_DIR/pip-freeze.sha256"
sha256sum "$REQUIREMENTS" "$CONSTRAINTS" "$RESOLVED_CONSTRAINTS" \
  >"$ARTIFACT_DIR/input-locks.sha256"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES}" \
  "$PYTHON" -m ifv_training environment-manifest \
  --repo-root "$REPO_ROOT" \
  --output "$ARTIFACT_DIR/environment.json"

touch "$ENV_PREFIX/.ifv-vllm-qwen3vl-ready"
echo "prepared immutable serving environment: $ENV_PREFIX"
echo "environment evidence: $ARTIFACT_DIR"
