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
mapfile -t MODEL_SHARDS < <("$PYTHON" - "$MODEL/model.safetensors.index.json" <<'PY'
import json
import sys
from pathlib import Path

index = Path(sys.argv[1])
data = json.loads(index.read_text(encoding="utf-8"))
shards = sorted(set(data.get("weight_map", {}).values()))
if len(shards) != 4:
    raise SystemExit(f"expected four shards in {index}, got {len(shards)}")
for shard in shards:
    if not isinstance(shard, str) or Path(shard).name != shard or not shard.endswith(".safetensors"):
        raise SystemExit(f"unsafe shard entry in {index}: {shard!r}")
    print(shard)
PY
)
if [[ "${#MODEL_SHARDS[@]}" -ne 4 ]]; then
  echo "model index does not resolve to four final safetensors shards" >&2
  exit 2
fi
for shard in "${MODEL_SHARDS[@]}"; do
  if [[ ! -f "$MODEL/$shard" ]]; then
    echo "model shard listed by index is missing: $MODEL/$shard" >&2
    exit 2
  fi
done

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
MODEL_HASH_INPUTS=(
  "$MODEL/config.json"
  "$MODEL/tokenizer_config.json"
  "$MODEL/preprocessor_config.json"
  "$MODEL/model.safetensors.index.json"
)
for shard in "${MODEL_SHARDS[@]}"; do
  MODEL_HASH_INPUTS+=("$MODEL/$shard")
done
sha256sum "${MODEL_HASH_INPUTS[@]}" >"$ARTIFACT_DIR/model-files.sha256"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}" \
  "$PYTHON" -m ifv_training environment-manifest \
  --repo-root "$REPO_ROOT" \
  --output "$ARTIFACT_DIR/environment.json"

touch "$ENV_PREFIX/.ifv-vllm-qwen35-ready"
echo "frozen Qwen3.5 serving environment: $ENV_PREFIX"
echo "environment evidence: $ARTIFACT_DIR"
