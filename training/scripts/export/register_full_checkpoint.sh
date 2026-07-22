#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"

if [[ "$#" -ne 4 ]]; then
  echo "usage: $0 CHECKPOINT_DIR BASE_MODEL_DIR EXPORT_ID DATASET_MANIFEST" >&2
  exit 2
fi

CHECKPOINT_DIR="$1"
BASE_MODEL_DIR="$2"
EXPORT_ID="$3"
DATASET_MANIFEST="$4"
if [[ ! -d "$CHECKPOINT_DIR" ]]; then
  echo "checkpoint does not exist: $CHECKPOINT_DIR" >&2
  exit 2
fi
if [[ ! -d "$BASE_MODEL_DIR" ]]; then
  echo "base model does not exist: $BASE_MODEL_DIR" >&2
  exit 2
fi
if [[ ! -f "$DATASET_MANIFEST" ]]; then
  echo "dataset manifest does not exist: $DATASET_MANIFEST" >&2
  exit 2
fi

OUTPUT_DIR="$DATA_ROOT/exports/$EXPORT_ID"
new_output_dir "$OUTPUT_DIR"
record_environment "$OUTPUT_DIR"

python -m ifv_training audit-full-checkpoint \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --base-model-dir "$BASE_MODEL_DIR" \
  --output "$OUTPUT_DIR/full-parameter-audit.json"

python -m ifv_training checkpoint-manifest \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --dataset-manifest "$DATASET_MANIFEST" \
  --output "$OUTPUT_DIR/checkpoint-manifest.json" \
  --base-model-id "$BASE_MODEL_DIR" \
  --method full

printf '%s\n' "$CHECKPOINT_DIR" >"$OUTPUT_DIR/model-path.txt"
