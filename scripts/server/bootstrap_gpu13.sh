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
  conda run -n "$name" python -c \
    "import sys; assert sys.version_info[:2] == (3, 10), sys.version"
  if ! conda run -n "$name" python -c "import torch; assert torch.cuda.is_available()" >/dev/null 2>&1; then
    if [[ -z "${IFV_TORCH_INDEX_URL:-}" || -z "${IFV_TORCH_PACKAGES:-}" ]]; then
      echo "$name has no CUDA-enabled PyTorch." >&2
      echo "Set IFV_TORCH_INDEX_URL and IFV_TORCH_PACKAGES after checking gpu-13's driver." >&2
      echo "Example form only: IFV_TORCH_PACKAGES='torch==... torchvision==...'" >&2
      exit 3
    fi
    # Intentional word splitting: IFV_TORCH_PACKAGES is a package-spec list.
    # shellcheck disable=SC2086
    conda run -n "$name" python -m pip install \
      --index-url "$IFV_TORCH_INDEX_URL" $IFV_TORCH_PACKAGES
    conda run -n "$name" python -c \
      "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda)"
  fi
  pip_args=()
  if [[ -n "${IFV_WHEELHOUSE:-}" ]]; then
    if [[ ! -d "$IFV_WHEELHOUSE" ]]; then
      echo "IFV_WHEELHOUSE does not exist: $IFV_WHEELHOUSE" >&2
      exit 2
    fi
    pip_args+=(--no-index --find-links "$IFV_WHEELHOUSE")
  fi
  conda run -n "$name" python -m pip install "${pip_args[@]}" -r "$requirements"
  conda run -n "$name" python -m pip install --no-deps -e "$REPO_ROOT"
}

# Never modifies ifv-agent.
SEED_ENV="${IFV_QWEN3VL_SEED_ENV:-qwen3vl}"
if [[ "$MODE" == "serve" || "$MODE" == "all" ]]; then
  create_env ifv-qwen3vl-serve "$REPO_ROOT/requirements/serve.txt" "$SEED_ENV"
  conda run -n ifv-qwen3vl-serve lmdeploy --help >/dev/null
  echo "Prepared isolated environment: ifv-qwen3vl-serve"
fi
if [[ "$MODE" == "sft" || "$MODE" == "all" ]]; then
  create_env ifv-qwen3vl-sft "$REPO_ROOT/requirements/train.txt" "$SEED_ENV"
  conda run -n ifv-qwen3vl-sft swift --help >/dev/null
  conda run -n ifv-qwen3vl-sft deepspeed --help >/dev/null
  echo "Prepared isolated environment: ifv-qwen3vl-sft"
fi
