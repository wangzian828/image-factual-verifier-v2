#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=gpu13_env.sh
source "${SCRIPT_DIR}/gpu13_env.sh"

ENV_NAME="${IFV_CONDA_ENV:-ifv-agent}"
PYTHON_VERSION="${IFV_PYTHON_VERSION:-3.11}"

if [[ "$(hostname)" != "gpu-13" ]]; then
    echo "bootstrap_gpu13.sh must run on gpu-13; current host: $(hostname)" >&2
    exit 2
fi

if [[ "${OMP_NUM_THREADS}" != "1" ]]; then
    echo "OMP_NUM_THREADS must be 1 on gpu-13" >&2
    exit 2
fi

mkdir -p \
    "${IFV_DATA_ROOT}/datasets" \
    "${IFV_DATA_ROOT}/artifacts" \
    "${IFV_DATA_ROOT}/benchmarks" \
    "${IFV_DATA_ROOT}/cache/huggingface" \
    "${IFV_DATA_ROOT}/cache/torch" \
    "${IFV_DATA_ROOT}/cache/paddle" \
    "${IFV_DATA_ROOT}/cache/tools" \
    "${IFV_DATA_ROOT}/runs/_logs" \
    "${IFV_DATA_ROOT}/runs/traces" \
    "${IFV_DATA_ROOT}/runs/eval" \
    "${IFV_DATA_ROOT}/generated"

if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain)" ]]; then
    echo "Refusing to install from a dirty server worktree." >&2
    echo "Commit locally, push to GitHub, then update this checkout." >&2
    exit 2
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
    conda create --yes --name "${ENV_NAME}" "python=${PYTHON_VERSION}" pip
fi

# Persist the server-specific threading guard inside the isolated environment.
conda env config vars set --name "${ENV_NAME}" OMP_NUM_THREADS=1

conda run --no-capture-output --name "${ENV_NAME}" \
    python -m pip install --upgrade pip setuptools wheel
conda run --no-capture-output --name "${ENV_NAME}" \
    python -m pip install --editable "${REPO_ROOT}[dev]"

actual_omp="$(conda run --name "${ENV_NAME}" python -c \
    'import os; print(os.environ.get("OMP_NUM_THREADS", ""))')"
if [[ "${actual_omp}" != "1" ]]; then
    echo "Conda environment did not retain OMP_NUM_THREADS=1" >&2
    exit 2
fi

conda run --no-capture-output --name "${ENV_NAME}" python - <<'PY'
import sys

import httpx
import paddle
import paddleocr
import pydantic
import requests

print(f"python={sys.version.split()[0]}")
print(f"paddlepaddle={paddle.__version__}")
print(f"paddleocr={paddleocr.__version__}")
print(f"httpx={httpx.__version__}")
print(f"pydantic={pydantic.__version__}")
print(f"requests={requests.__version__}")
PY

if [[ "${IFV_INSTALL_JUPYTER_KERNEL:-1}" == "1" ]]; then
    bash "${SCRIPT_DIR}/install_jupyter_kernel_gpu13.sh"
fi

echo "gpu-13 environment ready: ${ENV_NAME}"
