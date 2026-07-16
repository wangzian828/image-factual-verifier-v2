#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"

if [[ "$#" -ne 3 ]]; then
  echo "usage: $0 CHECKPOINT_DIR EXPORT_ID DATASET_MANIFEST" >&2
  exit 2
fi

CHECKPOINT_DIR="$1"
EXPORT_ID="$2"
DATASET_MANIFEST="$3"
if [[ ! -d "$CHECKPOINT_DIR" ]]; then
  echo "checkpoint does not exist: $CHECKPOINT_DIR" >&2
  exit 2
fi
if [[ ! -f "$DATASET_MANIFEST" ]]; then
  echo "dataset manifest does not exist: $DATASET_MANIFEST" >&2
  exit 2
fi

OUTPUT_DIR="$DATA_ROOT/exports/$EXPORT_ID"
new_output_dir "$OUTPUT_DIR"
record_environment "$OUTPUT_DIR"

args=(
  swift export
  --adapters "$CHECKPOINT_DIR"
  --merge_lora true
  --output_dir "$OUTPUT_DIR/model"
)
print_command "${args[@]}"
"${args[@]}" 2>&1 | tee "$OUTPUT_DIR/export.log"

python -m ifv_training checkpoint-manifest \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --dataset-manifest "$DATASET_MANIFEST" \
  --output "$OUTPUT_DIR/checkpoint-manifest.json" \
  --base-model-id "$(python -c "import json,sys; print(json.load(open(sys.argv[1], encoding='utf-8'))['model'])" "$CHECKPOINT_DIR/args.json")" \
  --method lora
