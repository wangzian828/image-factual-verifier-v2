#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:-all}"
CONDA="${IFV_CONDA_BIN:-$(command -v conda || true)}"
SFT_PREFIX="${IFV_QWEN35_SFT_ENV_PREFIX:-}"
LONG_SFT_PREFIX="${IFV_QWEN35_SFT_LONG_ENV_PREFIX:-}"
RL_PREFIX="${IFV_QWEN35_RL_ENV_PREFIX:-}"
MODEL="${IFV_QWEN35_MODEL:-${IFV_MODEL_ID:-}}"
ARTIFACT_ROOT="${IFV_TRAINING_DATA_ROOT:-${IFV_DATA_ROOT:+${IFV_DATA_ROOT}/training}}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${XDG_DATA_HOME:-${HOME}/.local/share}/image-factual-verifier/training}"

if [[ "$MODE" != "sft" && "$MODE" != "sft-long" && "$MODE" != "rl" && "$MODE" != "all" ]]; then
  echo "usage: $0 [sft|sft-long|rl|all]" >&2
  exit 2
fi
if [[ ! -x "$CONDA" ]]; then
  echo "conda is required: $CONDA" >&2
  exit 2
fi
if [[ ( "$MODE" == "sft" || "$MODE" == "all" ) && -z "$SFT_PREFIX" ]]; then
  echo "set IFV_QWEN35_SFT_ENV_PREFIX" >&2
  exit 2
fi
if [[ "$MODE" == "sft-long" && -z "$LONG_SFT_PREFIX" ]]; then
  echo "set IFV_QWEN35_SFT_LONG_ENV_PREFIX" >&2
  exit 2
fi
if [[ ( "$MODE" == "rl" || "$MODE" == "all" ) && -z "$RL_PREFIX" ]]; then
  echo "set IFV_QWEN35_RL_ENV_PREFIX" >&2
  exit 2
fi
if [[ ! -d "$MODEL" ]]; then
  echo "Qwen3.5 model directory does not exist: $MODEL" >&2
  exit 2
fi

proxy="${IFV_HTTP_PROXY:-${IFV_SERVER_PROXY:-${http_proxy:-}}}"
if [[ -n "$proxy" ]]; then
  export http_proxy="$proxy"
  export https_proxy="${IFV_HTTPS_PROXY:-$proxy}"
  export HTTP_PROXY="$http_proxy"
  export HTTPS_PROXY="$https_proxy"
fi
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="$NO_PROXY"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_PROGRESS_BAR=off
export OMP_NUM_THREADS=1

prepare_base() {
  local prefix="$1"
  local cuda_nvcc_version="$2"
  local incomplete_marker="$3"
  if [[ -e "$prefix" ]]; then
    if [[ "${IFV_RESUME_INCOMPLETE_ENV:-0}" != "1" ]]; then
      echo "refusing to modify an existing environment: $prefix" >&2
      exit 2
    fi
    if [[ ! -f "$prefix/$incomplete_marker" || ! -x "$prefix/bin/python" ]]; then
      echo "refusing to resume an unmarked or invalid environment: $prefix" >&2
      exit 2
    fi
    echo "resuming marked incomplete environment: $prefix" >&2
  else
    "$CONDA" create -y -p "$prefix" python=3.12 pip=25.2
    touch "$prefix/$incomplete_marker"
  fi
  "$CONDA" install -y -p "$prefix" -c nvidia "cuda-nvcc=$cuda_nvcc_version"
}

freeze_env() {
  local prefix="$1"
  local role="$2"
  local out="$ARTIFACT_ROOT/logs/environments/$role"
  mkdir -p "$out"
  local runtime_ld="$prefix/targets/x86_64-linux/lib:$prefix/lib:$prefix/lib/python3.12/site-packages/nvidia/curand/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  LD_LIBRARY_PATH="$runtime_ld" "$prefix/bin/python" -m pip check
  LD_LIBRARY_PATH="$runtime_ld" "$prefix/bin/python" -m pip freeze --all >"$out/pip-freeze.txt"
  sha256sum "$out/pip-freeze.txt" >"$out/pip-freeze.sha256"
  CUDA_HOME="$prefix" PATH="$prefix/bin:$PATH" \
    LD_LIBRARY_PATH="$runtime_ld" \
    CUDA_VISIBLE_DEVICES="${IFV_BOOTSTRAP_GPU_ID:-0}" \
    "$prefix/bin/python" -m ifv_training environment-manifest \
    --repo-root "$REPO_ROOT" --output "$out/environment.json"
  LD_LIBRARY_PATH="$runtime_ld" IFV_BOOTSTRAP_MODEL="$MODEL" \
    CUDA_VISIBLE_DEVICES="${IFV_BOOTSTRAP_GPU_ID:-0}" \
    "$prefix/bin/python" - <<'PY' >"$out/qwen35-import-gate.json"
import json
import os
import torch
from transformers import Qwen3_5ForConditionalGeneration
from transformers.utils.import_utils import (
    is_causal_conv1d_available,
    is_flash_attn_2_available,
    is_flash_linear_attention_available,
)
from swift import get_model_processor, get_template

model = os.environ["IFV_BOOTSTRAP_MODEL"]
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
    "model_class": Qwen3_5ForConditionalGeneration.__name__,
    "causal_conv1d_available": is_causal_conv1d_available(),
    "flash_attention_2_available": is_flash_attn_2_available(),
    "flash_linear_attention_available": is_flash_linear_attention_available(),
}, indent=2))
PY
}

link_torch_cuda_runtime() {
  local prefix="$1"
  local curand_dir="$prefix/lib/python3.12/site-packages/nvidia/curand/lib"
  local curand_so
  curand_so="$(find "$curand_dir" -maxdepth 1 -type f -name 'libcurand.so.*' -print -quit)"
  if [[ -z "$curand_so" ]]; then
    echo "nvidia-curand runtime library is missing below $curand_dir" >&2
    exit 2
  fi
  ln -sfn "$curand_so" "$prefix/lib/libcurand.so"
}

prebuild_cpu_adam() {
  local prefix="$1"
  CUDA_HOME="$prefix" PATH="$prefix/bin:$PATH" \
    LD_LIBRARY_PATH="$prefix/targets/x86_64-linux/lib:$prefix/lib:$prefix/lib/python3.12/site-packages/nvidia/curand/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    MAX_JOBS=4 "$prefix/bin/python" - <<'PY'
from deepspeed.ops.op_builder.cpu_adam import CPUAdamBuilder

module = CPUAdamBuilder().load(verbose=True)
if module.__name__ != "cpu_adam":
    raise RuntimeError(f"unexpected DeepSpeed CPUAdam module: {module.__name__}")
PY
}

install_sft() {
  local incomplete_marker=".ifv-qwen35-sft-bootstrap-incomplete"
  prepare_base "$SFT_PREFIX" 12.8.93 "$incomplete_marker"
  "$SFT_PREFIX/bin/python" -m pip install \
    torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0
  CUDA_HOME="$SFT_PREFIX" PATH="$SFT_PREFIX/bin:$PATH" \
    "$SFT_PREFIX/bin/python" -m pip install \
    --no-build-isolation --requirement "$REPO_ROOT/requirements/train-qwen35.txt"
  "$SFT_PREFIX/bin/python" -m pip install --no-deps --editable "$REPO_ROOT"
  link_torch_cuda_runtime "$SFT_PREFIX"
  prebuild_cpu_adam "$SFT_PREFIX"
  freeze_env "$SFT_PREFIX" ifv-qwen35-sft-ms-swift442
  "$SFT_PREFIX/bin/swift" sft --help >/dev/null
  CUDA_HOME="$SFT_PREFIX" PATH="$SFT_PREFIX/bin:$PATH" \
    "$SFT_PREFIX/bin/deepspeed" --help >/dev/null
  rm -f "$SFT_PREFIX/$incomplete_marker"
  touch "$SFT_PREFIX/.ifv-qwen35-sft-ready"
}

install_cuda_extensions() {
  local prefix="$1"
  local cuda_target="$prefix/targets/x86_64-linux"
  local runtime_ld="$cuda_target/lib:$prefix/lib:$prefix/lib/python3.12/site-packages/nvidia/curand/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  if [[ ! -f "$cuda_target/include/cuda_runtime.h" ]]; then
    echo "CUDA runtime headers are missing below $cuda_target/include" >&2
    exit 2
  fi
  local requirements="$REPO_ROOT/requirements/train-qwen35-long-context.txt"
  local requirements_sha
  requirements_sha="$(sha256sum "$requirements" | awk '{print $1}')"
  local abi_fingerprint
  abi_fingerprint="$(
    IFV_REQUIREMENTS_SHA="$requirements_sha" \
      IFV_BUILD_ARCH="${IFV_TORCH_CUDA_ARCH_LIST:-8.0}" \
      "$prefix/bin/python" - <<'PY'
import os
import platform
import re
import sys

import torch

raw = "-".join(
    (
        f"cp{sys.version_info.major}{sys.version_info.minor}",
        f"torch-{torch.__version__}",
        f"cuda-{torch.version.cuda}",
        f"glibc-{platform.libc_ver()[1]}",
        f"arch-{os.environ['IFV_BUILD_ARCH']}",
        f"req-{os.environ['IFV_REQUIREMENTS_SHA'][:16]}",
    )
)
print(re.sub(r"[^A-Za-z0-9._-]+", "_", raw))
PY
  )"
  local wheel_root="${IFV_CUDA_EXTENSION_WHEEL_ROOT:-$ARTIFACT_ROOT/wheels/qwen35-cuda-extensions}"
  local wheelhouse="$wheel_root/$abi_fingerprint"
  local wheel_manifest="$wheelhouse/SHA256SUMS"
  local wheel_provenance="$wheelhouse/build-context.json"
  mkdir -p "$wheel_root" "$ARTIFACT_ROOT/tmp"
  exec 9>"$wheel_root/$abi_fingerprint.lock"
  flock 9

  wheel_cache_valid() {
    [[ -s "$wheel_manifest" && -s "$wheel_provenance" ]] || return 1
    [[ "$(find "$wheelhouse" -maxdepth 1 -type f -name '*.whl' | wc -l)" -eq 2 ]] || return 1
    grep -q '  flash_attn-' "$wheel_manifest" || return 1
    grep -q '  causal_conv1d-' "$wheel_manifest" || return 1
    (cd "$wheelhouse" && sha256sum -c SHA256SUMS >/dev/null)
  }

  if ! wheel_cache_valid; then
    if [[ -e "$wheelhouse" ]]; then
      echo "refusing to replace an incomplete or corrupt CUDA wheel cache: $wheelhouse" >&2
      exit 2
    fi
    local build_dir
    build_dir="$(mktemp -d "$ARTIFACT_ROOT/tmp/qwen35-cuda-wheels.XXXXXX")"
    CUDA_HOME="$prefix" PATH="$prefix/bin:$PATH" \
      LD_LIBRARY_PATH="$runtime_ld" \
      CPATH="$cuda_target/include${CPATH:+:$CPATH}" \
      LIBRARY_PATH="$cuda_target/lib${LIBRARY_PATH:+:$LIBRARY_PATH}" \
      CC="$prefix/bin/x86_64-conda-linux-gnu-cc" \
      CXX="$prefix/bin/x86_64-conda-linux-gnu-c++" \
      TORCH_CUDA_ARCH_LIST="${IFV_TORCH_CUDA_ARCH_LIST:-8.0}" \
      FLASH_ATTN_CUDA_ARCHS="${IFV_FLASH_ATTN_CUDA_ARCHS:-80}" \
      MAX_JOBS="${IFV_EXTENSION_MAX_JOBS:-16}" \
      NVCC_THREADS="${IFV_EXTENSION_NVCC_THREADS:-2}" \
      FLASH_ATTENTION_FORCE_BUILD=TRUE \
      CAUSAL_CONV1D_FORCE_BUILD=TRUE \
      PIP_NO_CACHE_DIR=1 \
      "$prefix/bin/python" -m pip wheel \
      --wheel-dir "$build_dir" \
      --no-build-isolation \
      --no-cache-dir \
      --no-deps \
      --no-binary=:all: \
      --requirement "$requirements"
    if [[ "$(find "$build_dir" -maxdepth 1 -type f -name '*.whl' | wc -l)" -ne 2 ]] || \
       ! find "$build_dir" -maxdepth 1 -type f -name 'flash_attn-*.whl' | grep -q . || \
       ! find "$build_dir" -maxdepth 1 -type f -name 'causal_conv1d-*.whl' | grep -q .; then
      echo "CUDA extension build did not produce the two required wheels: $build_dir" >&2
      exit 2
    fi
    (
      cd "$build_dir"
      sha256sum ./*.whl | sed 's# \./# #' >SHA256SUMS
    )
    IFV_WHEEL_CONTEXT_OUTPUT="$build_dir/build-context.json" \
      IFV_WHEEL_ABI_FINGERPRINT="$abi_fingerprint" \
      IFV_REQUIREMENTS_SHA="$requirements_sha" \
      IFV_BUILD_ARCH="${IFV_TORCH_CUDA_ARCH_LIST:-8.0}" \
      IFV_BUILD_COMPILER="$prefix/bin/x86_64-conda-linux-gnu-cc" \
      "$prefix/bin/python" - <<'PY'
import json
import os
import platform
import subprocess

import torch

payload = {
    "schema_version": "ifv-cuda-extension-wheel-cache-v1",
    "abi_fingerprint": os.environ["IFV_WHEEL_ABI_FINGERPRINT"],
    "requirements_sha256": os.environ["IFV_REQUIREMENTS_SHA"],
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "glibc": platform.libc_ver()[1],
    "torch_cuda_arch_list": os.environ["IFV_BUILD_ARCH"],
    "compiler": subprocess.run(
        [os.environ["IFV_BUILD_COMPILER"], "--version"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.splitlines()[0],
}
with open(os.environ["IFV_WHEEL_CONTEXT_OUTPUT"], "w", encoding="utf-8") as handle:
    json.dump(payload, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
PY
    mv "$build_dir" "$wheelhouse"
  fi

  "$prefix/bin/python" -m pip uninstall -y flash-attn causal-conv1d || true
  CUDA_HOME="$prefix" PATH="$prefix/bin:$PATH" \
    LD_LIBRARY_PATH="$runtime_ld" \
    "$prefix/bin/python" -m pip install \
    --no-index \
    --find-links "$wheelhouse" \
    --no-build-isolation \
    --no-deps \
    --force-reinstall \
    --requirement "$requirements"
  CUDA_HOME="$prefix" PATH="$prefix/bin:$PATH" \
    LD_LIBRARY_PATH="$runtime_ld" \
    "$prefix/bin/python" - <<'PY'
import causal_conv1d_cuda
import flash_attn_2_cuda

assert causal_conv1d_cuda.__file__
assert flash_attn_2_cuda.__file__
PY
}

install_long_sft() {
  local incomplete_marker=".ifv-qwen35-sft-long-bootstrap-incomplete"
  prepare_base "$LONG_SFT_PREFIX" 12.8.93 "$incomplete_marker"
  "$LONG_SFT_PREFIX/bin/python" -m pip install \
    torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0
  CUDA_HOME="$LONG_SFT_PREFIX" PATH="$LONG_SFT_PREFIX/bin:$PATH" \
    "$LONG_SFT_PREFIX/bin/python" -m pip install \
    --no-build-isolation --requirement "$REPO_ROOT/requirements/train-qwen35.txt"
  install_cuda_extensions "$LONG_SFT_PREFIX"
  "$LONG_SFT_PREFIX/bin/python" -m pip install \
    --no-deps --editable "$REPO_ROOT"
  link_torch_cuda_runtime "$LONG_SFT_PREFIX"
  freeze_env "$LONG_SFT_PREFIX" ifv-qwen35-sft-long-ms-swift442
  local runtime_ld="$LONG_SFT_PREFIX/targets/x86_64-linux/lib:$LONG_SFT_PREFIX/lib:$LONG_SFT_PREFIX/lib/python3.12/site-packages/nvidia/curand/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  CUDA_HOME="$LONG_SFT_PREFIX" PATH="$LONG_SFT_PREFIX/bin:$PATH" \
    LD_LIBRARY_PATH="$runtime_ld" \
    CUDA_VISIBLE_DEVICES="${IFV_BOOTSTRAP_GPU_ID:-0}" \
    "$LONG_SFT_PREFIX/bin/python" \
    "$REPO_ROOT/scripts/probe/verify_qwen35_sft_environment.py" \
    --model "$MODEL" \
    --max-context 131072 \
    --expected-package-version torch=2.10.0 \
    --expected-package-version transformers=5.12.1 \
    --expected-package-version ms-swift=4.4.2 \
    --expected-package-version deepspeed=0.19.2 \
    --expected-package-version flash-attn=2.8.3 \
    --expected-package-version flash-linear-attention=0.5.1 \
    --expected-package-version causal-conv1d=1.6.2.post1 \
    --expected-package-version liger-kernel=0.8.0 \
    --required-package torch \
    --required-package transformers \
    --required-package ms-swift \
    --required-package deepspeed \
    --required-package flash-attn \
    --required-package flash-linear-attention \
    --required-package causal-conv1d \
    --required-package liger-kernel \
    --expected-python 3.12 \
    --expected-torch-cuda 12.8 \
    --expected-gpu-count 1 \
    --expected-gpu-name "${IFV_EXPECTED_GPU_NAME:-NVIDIA A100-SXM4-40GB}" \
    --expected-gpu-memory-mib "${IFV_EXPECTED_GPU_MEMORY_MIB:-40960}" \
    --output "$ARTIFACT_ROOT/logs/environments/ifv-qwen35-sft-long-ms-swift442/environment-preflight.json"
  "$LONG_SFT_PREFIX/bin/swift" sft --help >/dev/null
  rm -f "$LONG_SFT_PREFIX/$incomplete_marker"
  touch "$LONG_SFT_PREFIX/.ifv-qwen35-sft-long-ready"
}

install_rl() {
  local incomplete_marker=".ifv-qwen35-rl-bootstrap-incomplete"
  rm -f "$RL_PREFIX/.ifv-qwen35-rl-ready"
  prepare_base "$RL_PREFIX" 13.0.88 "$incomplete_marker"
  "$RL_PREFIX/bin/python" -m pip install \
    torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0
  CUDA_HOME="$RL_PREFIX" "$RL_PREFIX/bin/python" -m pip install \
    --no-build-isolation --requirement "$REPO_ROOT/requirements/rl-qwen35.txt"
  install_cuda_extensions "$RL_PREFIX"
  "$RL_PREFIX/bin/python" -m pip install --no-deps --editable "$REPO_ROOT"
  freeze_env "$RL_PREFIX" ifv-qwen35-rl-ms-swift442-vllm0221
  local rl_preflight="$ARTIFACT_ROOT/logs/environments/ifv-qwen35-rl-ms-swift442-vllm0221/environment-preflight.json"
  local runtime_ld="$RL_PREFIX/targets/x86_64-linux/lib:$RL_PREFIX/lib:$RL_PREFIX/lib/python3.12/site-packages/nvidia/curand/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  CUDA_HOME="$RL_PREFIX" PATH="$RL_PREFIX/bin:$PATH" \
    LD_LIBRARY_PATH="$runtime_ld" \
    CUDA_VISIBLE_DEVICES="${IFV_BOOTSTRAP_GPU_ID:-0}" \
    "$RL_PREFIX/bin/python" \
    "$REPO_ROOT/scripts/probe/verify_qwen35_rl_environment.py" \
    --model "$MODEL" \
    --max-context 131072 \
    --expected-package-version torch=2.11.0 \
    --expected-package-version transformers=5.12.1 \
    --expected-package-version ms-swift=4.4.2 \
    --expected-package-version deepspeed=0.19.2 \
    --expected-package-version flash-attn=2.8.3 \
    --expected-package-version flash-linear-attention=0.5.1 \
    --expected-package-version causal-conv1d=1.6.2.post1 \
    --expected-package-version liger-kernel=0.8.0 \
    --expected-package-version vllm=0.22.1 \
    --expected-package-version openai=2.32.0 \
    --expected-package-version jiter=0.14.0 \
    --expected-package-version llguidance=1.7.5 \
    --expected-package-version huggingface-hub=1.24.0 \
    --expected-python 3.12 \
    --expected-torch-cuda 13.0 \
    --expected-gpu-count 1 \
    --expected-gpu-name "${IFV_EXPECTED_GPU_NAME:-NVIDIA A100-SXM4-40GB}" \
    --expected-gpu-memory-mib "${IFV_EXPECTED_GPU_MEMORY_MIB:-40960}" \
    --output "$rl_preflight"
  rm -f "$RL_PREFIX/$incomplete_marker"
  sha256sum "$rl_preflight" >"$RL_PREFIX/.ifv-qwen35-rl-ready"
}

if [[ "$MODE" == "sft" || "$MODE" == "all" ]]; then
  install_sft
fi
if [[ "$MODE" == "sft-long" ]]; then
  install_long_sft
fi
if [[ "$MODE" == "rl" || "$MODE" == "all" ]]; then
  install_rl
fi
