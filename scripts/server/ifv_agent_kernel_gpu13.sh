#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
    echo "usage: $0 CONNECTION_FILE" >&2
    exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=gpu13_env.sh
source "${SCRIPT_DIR}/gpu13_env.sh"

if [[ "$(hostname)" != "gpu-13" ]]; then
    echo "ifv_agent_kernel_gpu13.sh must run on gpu-13" >&2
    exit 2
fi

if [[ "${OMP_NUM_THREADS}" != "1" ]]; then
    echo "OMP_NUM_THREADS must be 1 on gpu-13" >&2
    exit 2
fi

ENV_NAME="${IFV_CONDA_ENV:-ifv-agent}"
CONDA_BASE="$(conda info --base)"
PYTHON_BIN="${CONDA_BASE}/envs/${ENV_NAME}/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Missing Python executable for Conda environment: ${ENV_NAME}" >&2
    exit 2
fi

export PATH="${CONDA_BASE}/envs/${ENV_NAME}/bin:${PATH}"

exec "${PYTHON_BIN}" -m ipykernel_launcher -f "$1"
