#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=gpu13_env.sh
source "${SCRIPT_DIR}/gpu13_env.sh"

if [[ "$(hostname)" != "gpu-13" ]]; then
    echo "run_gpu13.sh must run on gpu-13" >&2
    exit 2
fi
exec "${SCRIPT_DIR}/run_ifv.sh" "$@"
