#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=ifv_env.sh
source "${SCRIPT_DIR}/ifv_env.sh"

python_bin="${IFV_BOOTSTRAP_PYTHON:-python3}"
venv_dir="${IFV_RUNTIME_VENV:-${HOME}/.local/share/image-factual-verifier/venv}"

if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain)" ]]; then
    echo "Refusing to install from a dirty worktree." >&2
    exit 2
fi

mkdir -p \
    "${IFV_DATA_ROOT}/datasets" \
    "${IFV_DATA_ROOT}/artifacts" \
    "${IFV_DATA_ROOT}/cache/huggingface" \
    "${IFV_DATA_ROOT}/cache/torch" \
    "${IFV_DATA_ROOT}/cache/tools" \
    "${IFV_DATA_ROOT}/runs/_jobs" \
    "${IFV_DATA_ROOT}/runs/_locks" \
    "${IFV_DATA_ROOT}/runs/_logs" \
    "${IFV_DATA_ROOT}/runs/eval" \
    "${IFV_DATA_ROOT}/generated"

if [[ ! -x "${venv_dir}/bin/python" ]]; then
    "${python_bin}" -m venv "${venv_dir}"
fi
"${venv_dir}/bin/python" -m pip install --upgrade pip setuptools wheel
"${venv_dir}/bin/python" -m pip install --editable "${REPO_ROOT}[dev,ops]"

export IFV_RUNTIME_PYTHON="${venv_dir}/bin/python"
"${IFV_RUNTIME_PYTHON}" "${SCRIPT_DIR}/doctor.py" --require-provider none
printf 'runtime_python=%s\n' "${IFV_RUNTIME_PYTHON}"
