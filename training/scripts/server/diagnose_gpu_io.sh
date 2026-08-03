#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"

if [[ "$#" -lt 1 || "$#" -gt 2 ]]; then
  echo "usage: $0 OUTPUT_DIR [GPU_LIST]" >&2
  exit 2
fi

OUTPUT_DIR="$1"
GPU_LIST="${2:-visible}"

require_training_gpus
mkdir -p "$OUTPUT_DIR"

{
  printf 'schema_version\tifv-gpu-io-wrapper-v1\n'
  printf 'cuda_visible_devices\t%s\n' "${CUDA_VISIBLE_DEVICES:-}"
  printf 'nproc_per_node\t%s\n' "${NPROC_PER_NODE:-}"
  printf 'allowed_gpu_ids\t%s\n' "$ALLOWED_GPU_IDS"
} >"$OUTPUT_DIR/gpu-io-context.tsv"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi topo -m >"$OUTPUT_DIR/nvidia-smi-topo.txt" || true
  nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,memory.used,utilization.gpu \
    --format=csv >"$OUTPUT_DIR/nvidia-smi-gpus.csv" || true
fi

taskset -pc $$ >"$OUTPUT_DIR/cpu-affinity.txt" 2>&1 || true

python "$SCRIPT_DIR/measure_gpu_io.py" \
  --gpus "$GPU_LIST" \
  >"$OUTPUT_DIR/gpu-io.json"

printf 'GPU I/O diagnostic written to %s\n' "$OUTPUT_DIR"
