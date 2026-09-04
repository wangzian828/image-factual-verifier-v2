#!/usr/bin/env bash

# Private gpu-13 profile. Machine values are supplied by runtime.env or the
# calling process; the committed file contains no account or mount paths.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${IFV_SERVER_PROXY_OVERRIDE:-}" ]]; then
    export IFV_SERVER_PROXY="${IFV_SERVER_PROXY_OVERRIDE}"
fi
export IFV_OMP_NUM_THREADS=1
# shellcheck source=ifv_env.sh
source "${SCRIPT_DIR}/ifv_env.sh"
