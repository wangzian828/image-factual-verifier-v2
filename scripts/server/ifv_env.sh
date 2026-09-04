#!/usr/bin/env bash

# Host-independent runtime environment for IFV.
#
# Machine-specific values belong in IFV_ENV_FILE or the calling process.
# This file deliberately contains no server name, account, proxy, model, or
# absolute project-data path.

_ifv_runtime_env="${IFV_ENV_FILE:-${HOME}/.config/image-factual-verifier/runtime.env}"
if [[ -f "${_ifv_runtime_env}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${_ifv_runtime_env}"
    set +a
    export IFV_ENV_FILE="${_ifv_runtime_env}"
fi

_ifv_default_data_root="${XDG_DATA_HOME:-${HOME}/.local/share}/image-factual-verifier"
_ifv_data_root="${IFV_DATA_ROOT:-${_ifv_default_data_root}}"
_ifv_proxy="${IFV_SERVER_PROXY:-}"
_ifv_no_proxy="${NO_PROXY:-${no_proxy:-}}"

for _ifv_loopback in 127.0.0.1 localhost ::1; do
    case ",${_ifv_no_proxy}," in
        *",${_ifv_loopback},"*) ;;
        *) _ifv_no_proxy="${_ifv_loopback}${_ifv_no_proxy:+,${_ifv_no_proxy}}" ;;
    esac
done

export OMP_NUM_THREADS="${IFV_OMP_NUM_THREADS:-1}"
export IFV_DATA_ROOT="${_ifv_data_root}"
export HF_HOME="${HF_HOME:-${IFV_DATA_ROOT}/cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-${IFV_DATA_ROOT}/cache/torch}"
export TOOL_CACHE_DIR="${TOOL_CACHE_DIR:-${IFV_DATA_ROOT}/cache/tools}"
export IFV_LOG_RETENTION_DAYS="${IFV_LOG_RETENTION_DAYS:-30}"
export no_proxy="${_ifv_no_proxy}"
export NO_PROXY="${_ifv_no_proxy}"

if [[ -n "${_ifv_proxy}" ]]; then
    export http_proxy="${_ifv_proxy}"
    export https_proxy="${_ifv_proxy}"
    export HTTP_PROXY="${_ifv_proxy}"
    export HTTPS_PROXY="${_ifv_proxy}"
fi

unset _ifv_default_data_root _ifv_data_root _ifv_proxy
unset _ifv_no_proxy _ifv_loopback _ifv_runtime_env
