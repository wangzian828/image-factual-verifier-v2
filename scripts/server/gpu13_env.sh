#!/usr/bin/env bash

# Source this file before every project command on gpu-13.
# IFV_SERVER_PROXY exists only to support an explicitly announced proxy change.
_ifv_proxy="${IFV_SERVER_PROXY:-http://100.10.1.210:47899}"

export OMP_NUM_THREADS=1
export http_proxy="${_ifv_proxy}"
export https_proxy="${_ifv_proxy}"
export HTTP_PROXY="${_ifv_proxy}"
export HTTPS_PROXY="${_ifv_proxy}"

unset _ifv_proxy
