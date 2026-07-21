#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_PREFIX="${IFV_VLLM_ENV_PREFIX:-/gsdata/home/wza/conda/envs/ifv-qwen35-vllm-nightly}"
MODEL="${IFV_QWEN35_MODEL:-/gsdata/home/wza/models/Qwen3.5-9B}"
ARTIFACT_ROOT="${IFV_TRAINING_DATA_ROOT:-/gsdata/home/wza/image-factual-verifier-v2-data/training}"
ARTIFACT_DIR="$ARTIFACT_ROOT/logs/environments/ifv-qwen35-vllm-nightly"
PYTHON="$ENV_PREFIX/bin/python"

if [[ ! -x "$PYTHON" || ! -x "$ENV_PREFIX/bin/vllm" ]]; then
  echo "Qwen3.5 serving environment is incomplete: $ENV_PREFIX" >&2
  exit 2
fi
if [[ ! -d "$MODEL" ]]; then
  echo "model directory does not exist: $MODEL" >&2
  exit 2
fi
if find "$MODEL" -maxdepth 2 -type f -path '*/._____temp/*' -print -quit | grep -q .; then
  echo "model download is incomplete: temporary shards remain" >&2
  exit 2
fi
if [[ "$(find "$MODEL" -maxdepth 1 -name 'model-*.safetensors' -type f | wc -l)" -ne 4 ]]; then
  echo "model directory does not contain four final safetensors shards" >&2
  exit 2
fi

mkdir -p "$ARTIFACT_DIR"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}" \
  "$PYTHON" "$REPO_ROOT/scripts/probe/verify_vllm_qwen35_environment.py" \
  --model "$MODEL" \
  --output "$ARTIFACT_DIR/verification.json"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}" \
  NCCL_CUMEM_HOST_ENABLE=0 \
  "$ENV_PREFIX/bin/torchrun" --standalone --nproc-per-node=2 \
  "$REPO_ROOT/scripts/probe/verify_nccl_tensor_parallel.py" \
  --expected-world-size 2 \
  --output "$ARTIFACT_DIR/nccl-tensor-parallel.json"
"$PYTHON" -m pip check
"$PYTHON" -m pip freeze --all >"$ARTIFACT_DIR/pip-freeze.txt"
sha256sum "$ARTIFACT_DIR/pip-freeze.txt" >"$ARTIFACT_DIR/pip-freeze.sha256"
sha256sum "$MODEL"/config.json "$MODEL"/tokenizer_config.json \
  "$MODEL"/preprocessor_config.json "$MODEL"/model.safetensors.index.json \
  "$MODEL"/model-*.safetensors >"$ARTIFACT_DIR/model-files.sha256"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}" \
  "$PYTHON" -m ifv_training environment-manifest \
  --repo-root "$REPO_ROOT" \
  --output "$ARTIFACT_DIR/environment.json"

touch "$ENV_PREFIX/.ifv-vllm-qwen35-ready"
echo "frozen Qwen3.5 serving environment: $ENV_PREFIX"
echo "environment evidence: $ARTIFACT_DIR"
