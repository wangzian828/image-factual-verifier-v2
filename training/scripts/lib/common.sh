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
  local python_bin
  python_bin="$CUDA_HOME/bin/python"
  if [[ ! -x "$python_bin" ]]; then
    python_bin="python"
  fi
  local python_minor
  python_minor="$("$python_bin" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  local torch_lib
  torch_lib="$CUDA_HOME/lib/python$python_minor/site-packages/torch/lib"
  local target_lib="$CUDA_HOME/targets/x86_64-linux/lib"
  local curand_dir
  curand_dir="$CUDA_HOME/lib/python$python_minor/site-packages/nvidia/curand/lib"
  if [[ -d "$target_lib" ]]; then
    export LD_LIBRARY_PATH="$target_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
  if [[ -d "$torch_lib" ]]; then
    export LD_LIBRARY_PATH="$torch_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
  if [[ -d "$curand_dir" ]]; then
    export LD_LIBRARY_PATH="$CUDA_HOME/lib:$curand_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
}

configure_conda_compilers() {
  if [[ -x "$CUDA_HOME/bin/x86_64-conda-linux-gnu-gcc" ]]; then
    export CC="${CC:-$CUDA_HOME/bin/x86_64-conda-linux-gnu-gcc}"
  fi
  if [[ -x "$CUDA_HOME/bin/x86_64-conda-linux-gnu-g++" ]]; then
    export CXX="${CXX:-$CUDA_HOME/bin/x86_64-conda-linux-gnu-g++}"
    export CUDAHOSTCXX="${CUDAHOSTCXX:-$CUDA_HOME/bin/x86_64-conda-linux-gnu-g++}"
  fi
}

configure_training_caches() {
  local cache_root="${IFV_TRAINING_CACHE_ROOT:-$DATA_ROOT/cache}"
  mkdir -p \
    "$cache_root/huggingface" \
    "$cache_root/huggingface/datasets" \
    "$cache_root/torch" \
    "$cache_root/torch-extensions" \
    "$cache_root/triton" \
    "$DATA_ROOT/tmp"
  export HF_HOME="${HF_HOME:-$cache_root/huggingface}"
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$cache_root/huggingface/datasets}"
  export TORCH_HOME="${TORCH_HOME:-$cache_root/torch}"
  export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$cache_root/triton}"
  export TMPDIR="${TMPDIR:-$DATA_ROOT/tmp}"
  if [[ -n "${CUDA_HOME:-}" ]]; then
    export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$cache_root/torch-extensions/$(basename "$CUDA_HOME")}"
    mkdir -p "$TORCH_EXTENSIONS_DIR"
  fi
}

configure_training_runtime() {
  configure_cuda_toolkit
  configure_conda_compilers
  configure_training_caches
}

prepare_deepspeed_cpu_adam() {
  local config="${IFV_DEEPSPEED:-}"
  local needs_cpu_adam=false
  if [[ "$config" == "zero3_offload" ]]; then
    needs_cpu_adam=true
  elif [[ -f "$config" ]] && python - "$config" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    config = json.load(handle)
device = config.get("zero_optimization", {}).get("offload_optimizer", {}).get("device")
raise SystemExit(0 if device == "cpu" else 1)
PY
  then
    needs_cpu_adam=true
  fi
  if [[ "$needs_cpu_adam" != "true" ]]; then
    return 0
  fi
  configure_training_runtime
  if [[ ! -e "$CUDA_HOME/lib/libcurand.so" ]]; then
    echo "DeepSpeed CPUAdam requires $CUDA_HOME/lib/libcurand.so; rebuild the frozen SFT environment" >&2
    exit 2
  fi
  MAX_JOBS="${MAX_JOBS:-4}" python - <<'PY'
from deepspeed.ops.op_builder.cpu_adam import CPUAdamBuilder

module = CPUAdamBuilder().load(verbose=True)
if module.__name__ != "cpu_adam":
    raise RuntimeError(f"unexpected DeepSpeed CPUAdam module: {module.__name__}")
PY
}

require_visible_gpus() {
  require_value CUDA_VISIBLE_DEVICES
  IFS=',' read -r -a devices <<<"$CUDA_VISIBLE_DEVICES"
  if [[ "${#devices[@]}" -lt 1 || "${#devices[@]}" -gt 8 ]]; then
    echo "CUDA_VISIBLE_DEVICES must name between one and eight GPUs, got: $CUDA_VISIBLE_DEVICES" >&2
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
  local omp_threads="${IFV_OMP_NUM_THREADS:-1}"
  if [[ "$omp_threads" != "1" ]]; then
    echo "server policy requires IFV_OMP_NUM_THREADS=1; got: $omp_threads" >&2
    exit 2
  fi
  export OMP_NUM_THREADS=1
# Training runs are non-interactive.  A Jupyter kernel may export the inline
# matplotlib backend, which makes ms-swift's final loss-plot hook fail after a
# successful optimizer step.  Pin a headless backend for every launcher.
export MPLBACKEND=Agg
  # Required on gpu-13: the R580/NCCL 2.27 default cuMem-host allocation path
  # has previously crashed during multi-rank startup.
  export NCCL_CUMEM_HOST_ENABLE="${NCCL_CUMEM_HOST_ENABLE:-0}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-$PYTORCH_CUDA_ALLOC_CONF}"
}

require_training_gpus() {
  configure_training_runtime
  require_visible_gpus
}

require_idle_gpu_memory() {
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

require_idle_gpus() {
  require_training_gpus
  require_idle_gpu_memory
}

require_idle_runtime_gpus() {
  require_visible_gpus
  require_idle_gpu_memory
}

require_full_parameter_profile() {
  require_value IFV_TUNER_TYPE
  require_value IFV_FREEZE_LLM
  require_value IFV_FREEZE_VIT
  require_value IFV_FREEZE_ALIGNER
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
  local deepspeed_config="${IFV_DEEPSPEED:-}"
  local fsdp_config="${IFV_FSDP:-}"
  if [[ -z "$deepspeed_config" && -z "$fsdp_config" ]]; then
    echo "full-parameter training requires exactly one backend: IFV_DEEPSPEED or IFV_FSDP" >&2
    exit 2
  fi
  if [[ -n "$deepspeed_config" && -n "$fsdp_config" ]]; then
    echo "full-parameter training requires exactly one backend; do not set both IFV_DEEPSPEED and IFV_FSDP" >&2
    exit 2
  fi
  if [[ -n "$deepspeed_config" && "$deepspeed_config" != "zero3" && "$deepspeed_config" != "zero3_offload" && ! -s "$deepspeed_config" ]]; then
    echo "full-parameter DeepSpeed training requires IFV_DEEPSPEED=zero3, zero3_offload, or a non-empty DeepSpeed config" >&2
    exit 2
  fi
  if [[ -n "$fsdp_config" && "$fsdp_config" != "fsdp2" && ! -s "$fsdp_config" ]]; then
    echo "full-parameter FSDP training requires IFV_FSDP=fsdp2 or a non-empty FSDP config" >&2
    exit 2
  fi
}

configure_distributed_backend() {
  if [[ -n "${IFV_FSDP:-}" ]]; then
    # ms-swift resolves the default model device map before its FSDP argument
    # initialization runs.  Set Accelerate's FSDP marker early so the model is
    # loaded on CPU/meta rather than materialized in full on every GPU first.
    export ACCELERATE_USE_FSDP=true
    export FSDP_VERSION="${IFV_FSDP_VERSION:-2}"
  else
    unset ACCELERATE_USE_FSDP
    unset FSDP_VERSION
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

verify_cached_dataset_gate() {
  if [[ "$#" -ne 3 ]]; then
    echo "usage: verify_cached_dataset_gate OUTPUT TRAIN_ARRAY_NAME VAL_ARRAY_NAME" >&2
    return 2
  fi
  local output="$1"
  local train_array_name="$2"
  local validation_array_name="$3"
  local -n cached_train_ref="$train_array_name"
  local -n cached_validation_ref="$validation_array_name"
  if [[ "${#cached_train_ref[@]}" -ne "${#cached_validation_ref[@]}" ]]; then
    echo "cached train/validation dataset counts must match" >&2
    return 2
  fi
  if [[ "${#cached_train_ref[@]}" -lt 1 ]]; then
    echo "cached dataset gate requires at least one train/validation pair" >&2
    return 2
  fi

  local manifests=()
  if [[ -n "${IFV_CACHED_DATASET_MANIFEST:-}" ]]; then
    read -r -a manifests <<<"$IFV_CACHED_DATASET_MANIFEST"
    if [[ "${#manifests[@]}" -ne "${#cached_train_ref[@]}" ]]; then
      echo "IFV_CACHED_DATASET_MANIFEST count must match cached dataset pairs" >&2
      return 2
    fi
  else
    local index train_parent validation_parent
    for index in "${!cached_train_ref[@]}"; do
      train_parent="$(cd "${cached_train_ref[$index]}/.." && pwd)"
      validation_parent="$(cd "${cached_validation_ref[$index]}/.." && pwd)"
      if [[ "$train_parent" != "$validation_parent" ]]; then
        echo "cached train/validation directories must share one cache root" >&2
        return 2
      fi
      manifests+=("$train_parent/dataset-manifest.json")
    done
  fi

  local args=(python -m ifv_training verify-cached-dataset --output "$output")
  local path
  for path in "${cached_train_ref[@]}"; do
    args+=(--train-dir "$path")
  done
  for path in "${cached_validation_ref[@]}"; do
    args+=(--validation-dir "$path")
  done
  for path in "${manifests[@]}"; do
    if [[ ! -s "$path" ]]; then
      echo "cached dataset manifest does not exist or is empty: $path" >&2
      return 2
    fi
    args+=(--manifest "$path")
  done
  "${args[@]}"
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
