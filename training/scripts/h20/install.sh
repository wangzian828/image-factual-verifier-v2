#!/usr/bin/env bash
set -euo pipefail
: "${IFV_H20_ROOT:?Set the allowed absolute server root}"
PREFIX="$IFV_H20_ROOT/envs/h20-qwen35-128k"
export CONDA_PKGS_DIRS="$IFV_H20_ROOT/cache/conda/pkgs" PIP_CACHE_DIR="$IFV_H20_ROOT/cache/pip" TMPDIR="$IFV_H20_ROOT/tmp"
export XDG_CACHE_HOME="$IFV_H20_ROOT/cache" XDG_CONFIG_HOME="$IFV_H20_ROOT/cache/config" PIP_PROGRESS_BAR=off OMP_NUM_THREADS=1
mkdir -p "$TMPDIR" "$IFV_H20_ROOT/logs"
cd "$IFV_H20_ROOT"
if [[ ! -d "$PREFIX" ]]; then
 "$IFV_H20_ROOT/miniforge3/bin/conda" create -y -p "$PREFIX" python=3.12 pip ninja packaging
fi
"$IFV_H20_ROOT/miniforge3/bin/conda" install -y -p "$PREFIX" -c nvidia cuda-nvcc=12.8.93 cuda-cudart-dev=12.8 cuda-cccl=12.8
export PATH="$PREFIX/bin:$PATH" CUDA_HOME="$PREFIX"
export CPATH="$PREFIX/targets/x86_64-linux/include${CPATH:+:$CPATH}" LIBRARY_PATH="$PREFIX/targets/x86_64-linux/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
python -m pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0
python -m pip install ms-swift==4.4.2 transformers==5.12.1 deepspeed==0.19.2 qwen-vl-utils==0.0.14 flash-linear-attention==0.5.1 liger-kernel==0.8.0 psutil
python -m pip install tilelang==0.1.14
export MAX_JOBS=16 NVCC_THREADS=2 TORCH_CUDA_ARCH_LIST=9.0 FLASH_ATTN_CUDA_ARCHS=90
python -m pip install --no-build-isolation --no-deps --no-clean flash-attn==2.8.3 causal-conv1d==1.6.2.post1
python -m pip check
python -m pip freeze > "$IFV_H20_ROOT/logs/h20-128k-pip-freeze.txt"
