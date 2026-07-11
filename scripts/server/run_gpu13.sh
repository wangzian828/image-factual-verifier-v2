#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=gpu13_env.sh
source "${SCRIPT_DIR}/gpu13_env.sh"

if [[ "${OMP_NUM_THREADS}" != "1" ]]; then
    echo "OMP_NUM_THREADS must be 1 on gpu-13" >&2
    exit 2
fi

if [[ "$#" -eq 0 ]]; then
    echo "usage: $0 COMMAND [ARG ...]" >&2
    exit 2
fi

exec "$@"
