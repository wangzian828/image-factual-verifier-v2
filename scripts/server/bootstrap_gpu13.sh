#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required" >&2
  exit 2
fi

MODE="${1:-all}"
if [[ "$MODE" != "serve" && "$MODE" != "sft" && "$MODE" != "all" ]]; then
  echo "usage: $0 [serve|sft|all]" >&2
  exit 2
fi

create_env() {
  local name="$1"
  local requirements="$2"
  local seed_env="$3"
  if ! conda env list | awk '{print $1}' | grep -Fxq "$name"; then
    if ! conda env list | awk '{print $1}' | grep -Fxq "$seed_env"; then
      echo "seed environment does not exist: $seed_env" >&2
      exit 2
    fi
    conda create -y -n "$name" --clone "$seed_env"
  fi
  local env_prefix
  env_prefix="$(
    conda env list | awk -v target="$name" '$1 == target {print $NF; exit}'
  )"
  if [[ -z "$env_prefix" || ! -x "$env_prefix/bin/python" ]]; then
    echo "cannot resolve Python for Conda environment: $name" >&2
    exit 2
  fi
  # The gpu-13 ifv-agent kernel deliberately keeps its own bin directory first
  # in PATH. Use absolute target-environment executables so that this bootstrap
  # cannot accidentally validate or modify ifv-agent through `conda run`.
  "$env_prefix/bin/python" -c \
    "import sys; assert sys.version_info[:2] == (3, 10), sys.version"
  if ! "$env_prefix/bin/python" -c "import torch; assert torch.cuda.is_available()" >/dev/null 2>&1; then
    if [[ -z "${IFV_TORCH_INDEX_URL:-}" || -z "${IFV_TORCH_PACKAGES:-}" ]]; then
      echo "$name has no CUDA-enabled PyTorch." >&2
      echo "Set IFV_TORCH_INDEX_URL and IFV_TORCH_PACKAGES after checking gpu-13's driver." >&2
      echo "Example form only: IFV_TORCH_PACKAGES='torch==... torchvision==...'" >&2
      exit 3
    fi
    # Intentional word splitting: IFV_TORCH_PACKAGES is a package-spec list.
    # shellcheck disable=SC2086
    "$env_prefix/bin/python" -m pip install \
      --index-url "$IFV_TORCH_INDEX_URL" $IFV_TORCH_PACKAGES
    "$env_prefix/bin/python" -c \
      "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda)"
  fi
  if [[ "$name" == "ifv-qwen3vl-sft" ]]; then
    # qwen3vl is only a CUDA/PyTorch seed on gpu-13.  It also contains serving
    # engines from earlier experiments; keeping those packages in the SFT
    # environment leaves an internally inconsistent vLLM/SGLang dependency
    # graph after ms-swift installs its training stack.
    "$env_prefix/bin/python" -m pip uninstall -y lmdeploy vllm sglang >/dev/null 2>&1 || true
  fi
  pip_args=()
  if [[ -n "${IFV_WHEELHOUSE:-}" ]]; then
    if [[ ! -d "$IFV_WHEELHOUSE" ]]; then
      echo "IFV_WHEELHOUSE does not exist: $IFV_WHEELHOUSE" >&2
      exit 2
    fi
    pip_args+=(--no-index --find-links "$IFV_WHEELHOUSE")
  fi
  "$env_prefix/bin/python" -m pip install "${pip_args[@]}" -r "$requirements"
  "$env_prefix/bin/python" -m pip install --no-deps -e "$REPO_ROOT"
  "$env_prefix/bin/python" -m pip check
}

# Never modifies ifv-agent.
SEED_ENV="${IFV_QWEN3VL_SEED_ENV:-qwen3vl}"
if [[ "$MODE" == "serve" || "$MODE" == "all" ]]; then
  create_env ifv-qwen3vl-serve "$REPO_ROOT/requirements/serve.txt" "$SEED_ENV"
  "$(conda env list | awk '$1 == "ifv-qwen3vl-serve" {print $NF; exit}')/bin/lmdeploy" --help >/dev/null
  echo "Prepared isolated environment: ifv-qwen3vl-serve"
fi
if [[ "$MODE" == "sft" || "$MODE" == "all" ]]; then
  create_env ifv-qwen3vl-sft "$REPO_ROOT/requirements/train.txt" "$SEED_ENV"
  sft_prefix="$(conda env list | awk '$1 == "ifv-qwen3vl-sft" {print $NF; exit}')"
  "$sft_prefix/bin/swift" sft --help >/dev/null
  "$sft_prefix/bin/deepspeed" --help >/dev/null
  echo "Prepared isolated environment: ifv-qwen3vl-sft"
fi
