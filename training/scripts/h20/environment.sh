#!/usr/bin/env bash
set -euo pipefail
: "${IFV_H20_ROOT:?Set the allowed absolute server root}"
export IFV_H20_RUN_ROOT="${IFV_H20_RUN_ROOT:-$IFV_H20_ROOT/training-artifacts/h20-benchmark}"
export CUDA_HOME="$IFV_H20_ROOT/envs/h20-qwen35-128k"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:$CUDA_HOME/lib:$CUDA_HOME/lib/python3.12/site-packages/nvidia/curand/lib:$CUDA_HOME/lib/python3.12/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CPATH="$CUDA_HOME/targets/x86_64-linux/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export HF_HOME="$IFV_H20_ROOT/cache/h20-huggingface" MODELSCOPE_CACHE="$IFV_H20_ROOT/cache/h20-modelscope"
export PIP_CACHE_DIR="$IFV_H20_ROOT/cache/pip" TMPDIR="$IFV_H20_ROOT/tmp" XDG_CACHE_HOME="$IFV_H20_ROOT/cache"
export TORCH_HOME="$IFV_H20_ROOT/cache/torch" TORCH_EXTENSIONS_DIR="$IFV_H20_ROOT/cache/torch-extensions" TRITON_CACHE_DIR="$IFV_H20_ROOT/cache/triton"
export TILELANG_CACHE_DIR="$IFV_H20_ROOT/cache/tilelang" TVM_FFI_CACHE_DIR="$IFV_H20_ROOT/cache/tvm-ffi"
export HF_DATASETS_CACHE="$HF_HOME/datasets" XDG_CONFIG_HOME="$IFV_H20_ROOT/cache/config" MPLCONFIGDIR="$IFV_H20_ROOT/cache/matplotlib"
export WANDB_MODE=disabled IMAGE_MAX_TOKEN_NUM=1024 MAX_PIXELS=262144 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p "$IFV_H20_RUN_ROOT" "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$XDG_CONFIG_HOME" "$MPLCONFIGDIR" "$TMPDIR"
cd "$IFV_H20_ROOT"
exec "$@"
