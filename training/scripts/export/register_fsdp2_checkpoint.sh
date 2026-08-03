#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"

if [[ "$#" -ne 4 ]]; then
  echo "usage: $0 CHECKPOINT_DIR BASE_MODEL_DIR EXPORT_ID CACHE_DIR" >&2
  exit 2
fi

CHECKPOINT_DIR="$1"
BASE_MODEL_DIR="$2"
EXPORT_ID="$3"
CACHE_DIR="$4"
configure_training_runtime
MODEL_SHARDS="$CHECKPOINT_DIR/pytorch_model_fsdp_0"
if [[ ! -s "$MODEL_SHARDS/.metadata" ]]; then
  echo "FSDP2 model shards are missing: $MODEL_SHARDS" >&2
  exit 2
fi
if [[ ! -s "$CHECKPOINT_DIR/optimizer_0/.metadata" ]]; then
  echo "FSDP2 optimizer shards are missing: $CHECKPOINT_DIR/optimizer_0" >&2
  exit 2
fi
if [[ ! -d "$BASE_MODEL_DIR" ]]; then
  echo "base model does not exist: $BASE_MODEL_DIR" >&2
  exit 2
fi
if [[ ! -d "$CACHE_DIR/train" || ! -d "$CACHE_DIR/val" ]]; then
  echo "cached dataset is incomplete: $CACHE_DIR" >&2
  exit 2
fi

OUTPUT_DIR="$DATA_ROOT/exports/$EXPORT_ID"
MODEL_DIR="$OUTPUT_DIR/model"
new_output_dir "$OUTPUT_DIR"
mkdir -p "$MODEL_DIR"
record_environment "$OUTPUT_DIR"

python -m ifv_training cached-dataset-manifest \
  --cache-dir "$CACHE_DIR" \
  --output "$OUTPUT_DIR/dataset-manifest.json"

python - "$MODEL_SHARDS" "$MODEL_DIR" <<'PY'
import sys
from accelerate.utils.fsdp_utils import merge_fsdp_weights

merge_fsdp_weights(sys.argv[1], sys.argv[2], safe_serialization=True)
PY

for name in \
  chat_template.jinja \
  config.json \
  configuration.json \
  generation_config.json \
  merges.txt \
  preprocessor_config.json \
  processor_config.json \
  tokenizer.json \
  tokenizer_config.json \
  video_preprocessor_config.json \
  vocab.json
do
  if [[ -f "$BASE_MODEL_DIR/$name" ]]; then
    cp -p "$BASE_MODEL_DIR/$name" "$MODEL_DIR/$name"
  fi
done
if [[ -f "$CHECKPOINT_DIR/../args.json" ]]; then
  cp -p "$CHECKPOINT_DIR/../args.json" "$MODEL_DIR/args.json"
fi

python -m ifv_training audit-full-checkpoint \
  --checkpoint-dir "$MODEL_DIR" \
  --state-checkpoint-dir "$CHECKPOINT_DIR" \
  --base-model-dir "$BASE_MODEL_DIR" \
  --output "$OUTPUT_DIR/full-parameter-audit.json"

python -m ifv_training checkpoint-manifest \
  --checkpoint-dir "$MODEL_DIR" \
  --state-checkpoint-dir "$CHECKPOINT_DIR" \
  --dataset-manifest "$OUTPUT_DIR/dataset-manifest.json" \
  --output "$OUTPUT_DIR/checkpoint-manifest.json" \
  --base-model-id "$BASE_MODEL_DIR" \
  --method full

python training/scripts/probe/load_qwen35_checkpoint.py \
  --model-dir "$MODEL_DIR" \
  --output "$OUTPUT_DIR/load-smoke.json"

printf '%s\n' "$MODEL_DIR" >"$OUTPUT_DIR/model-path.txt"
