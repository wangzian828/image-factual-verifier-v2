#!/usr/bin/env bash

# Source this file before every project command on gpu-13.
# IFV_SERVER_PROXY exists only to support an explicitly announced proxy change.
_ifv_proxy="${IFV_SERVER_PROXY:-http://100.10.1.210:47899}"
_ifv_data_root="${IFV_DATA_ROOT:-/gsdata/home/wza/image-factual-verifier-v2-data}"

export OMP_NUM_THREADS=1
export http_proxy="${_ifv_proxy}"
export https_proxy="${_ifv_proxy}"
export HTTP_PROXY="${_ifv_proxy}"
export HTTPS_PROXY="${_ifv_proxy}"
export IFV_DATA_ROOT="${_ifv_data_root}"
export HF_HOME="${IFV_DATA_ROOT}/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TORCH_HOME="${IFV_DATA_ROOT}/cache/torch"
export EASYOCR_MODULE_PATH="${IFV_DATA_ROOT}/cache/easyocr"
export TOOL_CACHE_DIR="${IFV_DATA_ROOT}/cache/tools"

unset _ifv_proxy _ifv_data_root
