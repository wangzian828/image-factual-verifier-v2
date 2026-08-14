#!/usr/bin/env bash

# Source this file before every project command on gpu-13.
# Use a deliberately named override so stale login/Jupyter environment variables
# cannot silently replace the committed working proxy.
_ifv_proxy="${IFV_SERVER_PROXY_OVERRIDE:-http://100.10.1.210:47899}"
_ifv_data_root="${IFV_DATA_ROOT:-/gsdata/home/wza/image-factual-verifier-v2-data}"
_ifv_no_proxy="${NO_PROXY:-${no_proxy:-}}"
for _ifv_loopback in 127.0.0.1 localhost ::1; do
    case ",${_ifv_no_proxy}," in
        *",${_ifv_loopback},"*) ;;
        *) _ifv_no_proxy="${_ifv_loopback}${_ifv_no_proxy:+,${_ifv_no_proxy}}" ;;
    esac
done

export OMP_NUM_THREADS=1
export http_proxy="${_ifv_proxy}"
export https_proxy="${_ifv_proxy}"
export HTTP_PROXY="${_ifv_proxy}"
export HTTPS_PROXY="${_ifv_proxy}"
export no_proxy="${_ifv_no_proxy}"
export NO_PROXY="${_ifv_no_proxy}"
export IFV_DATA_ROOT="${_ifv_data_root}"
export HF_HOME="${IFV_DATA_ROOT}/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TORCH_HOME="${IFV_DATA_ROOT}/cache/torch"
export PADDLE_HOME="${IFV_DATA_ROOT}/cache/paddle"
export PADDLE_PDX_CACHE_HOME="${IFV_DATA_ROOT}/cache/paddle"
export TOOL_CACHE_DIR="${IFV_DATA_ROOT}/cache/tools"
export IFV_LOG_RETENTION_DAYS="${IFV_LOG_RETENTION_DAYS:-30}"

_ifv_runtime_env="${HOME}/.config/image-factual-verifier/runtime.env"
if [[ -z "${IFV_ENV_FILE:-}" && -f "${_ifv_runtime_env}" ]]; then
    export IFV_ENV_FILE="${_ifv_runtime_env}"
fi

unset _ifv_proxy _ifv_data_root _ifv_no_proxy _ifv_loopback _ifv_runtime_env
