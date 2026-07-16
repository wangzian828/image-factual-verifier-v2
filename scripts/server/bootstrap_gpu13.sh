#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required" >&2
  exit 2
fi

create_env() {
  local name="$1"
  local requirements="$2"
  if ! conda env list | awk '{print $1}' | grep -Fxq "$name"; then
    conda create -y -n "$name" python=3.11 pip
  fi
  conda run -n "$name" python -c \
    "import sys; assert sys.version_info[:2] == (3, 11), sys.version"
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
  conda run -n "$name" python -m pip install -r "$requirements"
  conda run -n "$name" python -m pip install -e "$REPO_ROOT"
}

# Never modifies ifv-agent.
create_env ifv-qwen-train "$REPO_ROOT/requirements/train.txt"
create_env ifv-qwen-serve "$REPO_ROOT/requirements/serve.txt"

conda run -n ifv-qwen-train swift --help >/dev/null
conda run -n ifv-qwen-serve swift deploy --help >/dev/null

echo "Created isolated environments: ifv-qwen-train, ifv-qwen-serve"
