#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:-all}"
CONDA="${IFV_CONDA_BIN:-/gs/home/wza/anaconda3/bin/conda}"
SFT_PREFIX="${IFV_QWEN35_SFT_ENV_PREFIX:-/gsdata/home/wza/conda/envs/ifv-qwen35-sft-ms-swift442}"
RL_PREFIX="${IFV_QWEN35_RL_ENV_PREFIX:-/gsdata/home/wza/conda/envs/ifv-qwen35-rl-ms-swift442-vllm0171}"
MODEL="${IFV_QWEN35_MODEL:-/gsdata/home/wza/models/Qwen3.5-9B}"
ARTIFACT_ROOT="${IFV_TRAINING_DATA_ROOT:-/gsdata/home/wza/image-factual-verifier-v2-data/training}"

if [[ "$MODE" != "sft" && "$MODE" != "rl" && "$MODE" != "all" ]]; then
  echo "usage: $0 [sft|rl|all]" >&2
  exit 2
fi
if [[ ! -x "$CONDA" ]]; then
  echo "conda is required: $CONDA" >&2
  exit 2
fi
if [[ ! -d "$MODEL" ]]; then
  echo "Qwen3.5 model directory does not exist: $MODEL" >&2
  exit 2
fi

export http_proxy="${IFV_HTTP_PROXY:-http://100.10.1.210:47899}"
export https_proxy="${IFV_HTTPS_PROXY:-$http_proxy}"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="$NO_PROXY"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export OMP_NUM_THREADS=1

prepare_base() {
  local prefix="$1"
  if [[ -e "$prefix" ]]; then
    echo "refusing to modify an existing environment: $prefix" >&2
    exit 2
  fi
  "$CONDA" create -y -p "$prefix" python=3.12 pip=26.2
  "$CONDA" install -y -p "$prefix" -c nvidia cuda-nvcc=12.8.93
}

freeze_env() {
  local prefix="$1"
  local role="$2"
  local out="$ARTIFACT_ROOT/logs/environments/$role"
  mkdir -p "$out"
  "$prefix/bin/python" -m pip check
  "$prefix/bin/python" -m pip freeze --all >"$out/pip-freeze.txt"
  sha256sum "$out/pip-freeze.txt" >"$out/pip-freeze.sha256"
  CUDA_VISIBLE_DEVICES=4 "$prefix/bin/python" -m ifv_training environment-manifest \
    --repo-root "$REPO_ROOT" --output "$out/environment.json"
  CUDA_VISIBLE_DEVICES=4 "$prefix/bin/python" - <<'PY' >"$out/qwen35-import-gate.json"
import json
import torch
from swift import get_model_processor, get_template

model = "/gsdata/home/wza/models/Qwen3.5-9B"
loaded, processor = get_model_processor(model, load_model=False)
template = get_template(processor, enable_thinking=False)
print(json.dumps({
    "passed": True,
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda": torch.cuda.is_available(),
    "processor": type(processor).__name__,
    "template": type(template).__name__,
    "model_loaded": loaded is not None,
}, indent=2))
PY
}

install_sft() {
  prepare_base "$SFT_PREFIX"
  "$SFT_PREFIX/bin/python" -m pip install \
    torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0
  CUDA_HOME="$SFT_PREFIX" "$SFT_PREFIX/bin/python" -m pip install \
    --no-build-isolation --requirement "$REPO_ROOT/requirements/train-qwen35.txt"
  "$SFT_PREFIX/bin/python" -m pip install --no-deps --editable "$REPO_ROOT"
  freeze_env "$SFT_PREFIX" ifv-qwen35-sft-ms-swift442
  "$SFT_PREFIX/bin/swift" sft --help >/dev/null
  "$SFT_PREFIX/bin/deepspeed" --help >/dev/null
  touch "$SFT_PREFIX/.ifv-qwen35-sft-ready"
}

install_rl() {
  prepare_base "$RL_PREFIX"
  "$RL_PREFIX/bin/python" -m pip install \
    torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0
  CUDA_HOME="$RL_PREFIX" "$RL_PREFIX/bin/python" -m pip install \
    --no-build-isolation --requirement "$REPO_ROOT/requirements/rl-qwen35.txt"
  "$RL_PREFIX/bin/python" -m pip install --no-deps --editable "$REPO_ROOT"
  freeze_env "$RL_PREFIX" ifv-qwen35-rl-ms-swift442-vllm0171
  "$RL_PREFIX/bin/swift" rlhf --help >/dev/null
  "$RL_PREFIX/bin/vllm" serve --help >/dev/null
  touch "$RL_PREFIX/.ifv-qwen35-rl-ready"
}

if [[ "$MODE" == "sft" || "$MODE" == "all" ]]; then
  install_sft
fi
if [[ "$MODE" == "rl" || "$MODE" == "all" ]]; then
  install_rl
fi
