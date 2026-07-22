#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="${IFV_TRAINING_DATA_ROOT:-/gsdata/home/wza/image-factual-verifier-v2-data/training}"
ALLOWED_GPU_IDS="${IFV_ALLOWED_GPU_IDS:-4,5,6,7}"

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

configure_cuda_toolkit() {
  # gpu-13 keeps nvcc inside the dedicated conda environment rather than under
  # /usr/local/cuda.  Resolve that environment explicitly so DeepSpeed never
  # depends on an interactive shell having exported CUDA_HOME beforehand.
  local candidate="${IFV_CUDA_HOME:-${CUDA_HOME:-}}"
  if [[ -z "$candidate" ]]; then
    candidate="$(python -c 'import sys; print(sys.prefix)')"
  fi
  if [[ ! -x "$candidate/bin/nvcc" ]]; then
    echo "CUDA toolkit is unavailable: expected nvcc at $candidate/bin/nvcc. Set IFV_CUDA_HOME or run with the intended training Python environment first." >&2
    exit 2
  fi
  export CUDA_HOME="$candidate"
  export PATH="$CUDA_HOME/bin:$PATH"
}

require_training_gpus() {
  configure_cuda_toolkit
  require_value CUDA_VISIBLE_DEVICES
  IFS=',' read -r -a devices <<<"$CUDA_VISIBLE_DEVICES"
  if [[ "${#devices[@]}" -lt 1 || "${#devices[@]}" -gt 4 ]]; then
    echo "CUDA_VISIBLE_DEVICES must name between one and four GPUs, got: $CUDA_VISIBLE_DEVICES" >&2
    exit 2
  fi
  local seen=","
  local allowed=",${ALLOWED_GPU_IDS//[[:space:]]/},"
  local device
  for device in "${devices[@]}"; do
    device="${device//[[:space:]]/}"
    if [[ ! "$device" =~ ^[0-9]+$ ]]; then
      echo "CUDA_VISIBLE_DEVICES contains an invalid GPU index: $device" >&2
      exit 2
    fi
    if [[ "$seen" == *",$device,"* ]]; then
      echo "CUDA_VISIBLE_DEVICES contains duplicate GPU index: $device" >&2
      exit 2
    fi
    if [[ "$allowed" != *",$device,"* ]]; then
      echo "GPU $device is outside the allowed physical GPU set: $ALLOWED_GPU_IDS" >&2
      exit 2
    fi
    seen+="$device,"
  done
  export NPROC_PER_NODE="${#devices[@]}"
  export OMP_NUM_THREADS=1
  # Required on gpu-13: the R580/NCCL 2.27 default cuMem-host allocation path
  # has previously crashed during multi-rank startup.
  export NCCL_CUMEM_HOST_ENABLE="${NCCL_CUMEM_HOST_ENABLE:-0}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
}

require_idle_gpus() {
  require_training_gpus
  local device used util
  IFS=',' read -r -a devices <<<"$CUDA_VISIBLE_DEVICES"
  for device in "${devices[@]}"; do
    device="${device//[[:space:]]/}"
    IFS=',' read -r used util < <(
      nvidia-smi --id="$device" \
        --query-gpu=memory.used,utilization.gpu \
        --format=csv,noheader,nounits |
        awk -F',' '{gsub(/^[ \t]+|[ \t]+$/, "", $1); gsub(/^[ \t]+|[ \t]+$/, "", $2); print $1 "," $2}'
    )
    if [[ "$used" -gt 1024 ]]; then
      echo "GPU $device is not idle: ${used} MiB allocated, ${util}% utilization" >&2
      exit 2
    fi
  done
}

require_full_parameter_profile() {
  require_value IFV_TUNER_TYPE
  require_value IFV_FREEZE_LLM
  require_value IFV_FREEZE_VIT
  require_value IFV_FREEZE_ALIGNER
  require_value IFV_DEEPSPEED
  if [[ "$IFV_TUNER_TYPE" != "full" ]]; then
    echo "full-parameter training requires IFV_TUNER_TYPE=full" >&2
    exit 2
  fi
  local name
  for name in IFV_FREEZE_LLM IFV_FREEZE_VIT IFV_FREEZE_ALIGNER; do
    if [[ "${!name,,}" != "false" ]]; then
      echo "full-parameter multimodal training requires $name=false" >&2
      exit 2
    fi
  done
  if [[ "$IFV_DEEPSPEED" != "zero3" && "$IFV_DEEPSPEED" != "zero3_offload" ]]; then
    echo "full-parameter training requires IFV_DEEPSPEED=zero3 or zero3_offload" >&2
    exit 2
  fi
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
