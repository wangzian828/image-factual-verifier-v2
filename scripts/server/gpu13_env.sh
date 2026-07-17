#!/usr/bin/env bash

# Source this file before every project command on gpu-13.
# Use a deliberately named override so stale login/Jupyter environment variables
# cannot silently replace the committed working proxy.
_ifv_proxy="${IFV_SERVER_PROXY_OVERRIDE:-http://100.10.1.210:47899}"
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
export IFV_LOG_RETENTION_DAYS="${IFV_LOG_RETENTION_DAYS:-30}"

_ifv_runtime_env="${HOME}/.config/image-factual-verifier/runtime.env"
if [[ -z "${IFV_ENV_FILE:-}" && -f "${_ifv_runtime_env}" ]]; then
    export IFV_ENV_FILE="${_ifv_runtime_env}"
fi

unset _ifv_proxy _ifv_data_root _ifv_runtime_env
