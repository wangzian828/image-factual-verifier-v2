#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="${IFV_TRAINING_DATA_ROOT:-/gsdata/home/wza/image-factual-verifier-v2-data/training}"

load_profile() {
  local profile="$1"
  if [[ ! -f "$profile" ]]; then
    echo "profile not found: $profile" >&2
    exit 2
  fi
  set -a
  # shellcheck disable=SC1090
  source "$profile"
  set +a
}

require_value() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "required value is empty: $name" >&2
    exit 2
  fi
}

require_two_gpus() {
  require_value CUDA_VISIBLE_DEVICES
  IFS=',' read -r -a devices <<<"$CUDA_VISIBLE_DEVICES"
  if [[ "${#devices[@]}" -ne 2 ]]; then
    echo "CUDA_VISIBLE_DEVICES must name exactly two GPUs, got: $CUDA_VISIBLE_DEVICES" >&2
    exit 2
  fi
  if [[ "${devices[0]}" == "${devices[1]}" ]]; then
    echo "CUDA_VISIBLE_DEVICES contains the same GPU twice" >&2
    exit 2
  fi
  export NPROC_PER_NODE=2
  export OMP_NUM_THREADS=1
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
}

require_model_path() {
  require_value IFV_MODEL_ID
  if [[ "$IFV_MODEL_ID" == REQUIRED_* ]]; then
    echo "model profile still contains a placeholder: $IFV_MODEL_ID" >&2
    exit 2
  fi
  if [[ "$IFV_MODEL_ID" == /* && ! -e "$IFV_MODEL_ID" ]]; then
    echo "local model path does not exist: $IFV_MODEL_ID" >&2
    exit 2
  fi
}

require_dataset() {
  local path="$1"
  if [[ ! -s "$path" ]]; then
    echo "dataset does not exist or is empty: $path" >&2
    exit 2
  fi
}

new_output_dir() {
  local path="$1"
  if [[ -e "$path" ]]; then
    if [[ ! -d "$path" || -n "$(find "$path" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
      echo "output directory must be new or empty: $path" >&2
      exit 2
    fi
  else
    mkdir -p "$path"
  fi
}

record_environment() {
  local output_dir="$1"
  python -m ifv_training environment-manifest \
    --repo-root "$REPO_ROOT" \
    --output "$output_dir/environment.json"
  python -m pip freeze >"$output_dir/pip-freeze.txt"
  sha256sum "$output_dir/pip-freeze.txt" >"$output_dir/pip-freeze.sha256"
  nvidia-smi -q >"$output_dir/nvidia-smi.txt"
}

print_command() {
  printf 'Executing:'
  printf ' %q' "$@"
  printf '\n'
}
