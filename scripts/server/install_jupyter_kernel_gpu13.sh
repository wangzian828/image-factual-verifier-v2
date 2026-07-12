#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=gpu13_env.sh
source "${SCRIPT_DIR}/gpu13_env.sh"

ENV_NAME="${IFV_CONDA_ENV:-ifv-agent}"
KERNEL_NAME="${IFV_JUPYTER_KERNEL_NAME:-ifv-agent}"
KERNEL_DISPLAY_NAME="${IFV_JUPYTER_KERNEL_DISPLAY_NAME:-IFV Agent (Python 3.11)}"

if [[ "$(hostname)" != "gpu-13" ]]; then
    echo "install_jupyter_kernel_gpu13.sh must run on gpu-13" >&2
    exit 2
fi

if [[ "${OMP_NUM_THREADS}" != "1" ]]; then
    echo "OMP_NUM_THREADS must be 1 on gpu-13" >&2
    exit 2
fi

if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain)" ]]; then
    echo "Refusing to install a kernel from a dirty server worktree." >&2
    exit 2
fi

CONDA_BASE="$(conda info --base)"
PYTHON_BIN="${CONDA_BASE}/envs/${ENV_NAME}/bin/python"
JUPYTER_BIN="${CONDA_BASE}/bin/jupyter"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Missing Python executable for Conda environment: ${ENV_NAME}" >&2
    exit 2
fi
if [[ ! -x "${JUPYTER_BIN}" ]]; then
    echo "Missing Jupyter executable: ${JUPYTER_BIN}" >&2
    exit 2
fi

if ! "${PYTHON_BIN}" -c 'import ipykernel' >/dev/null 2>&1; then
    "${PYTHON_BIN}" -m pip install "ipykernel>=6.29"
fi

KERNEL_DIR="$(mktemp -d)"
cleanup() {
    rm -rf -- "${KERNEL_DIR}"
}
trap cleanup EXIT

export IFV_KERNEL_WRAPPER="${SCRIPT_DIR}/ifv_agent_kernel_gpu13.sh"
export IFV_KERNEL_DISPLAY_NAME="${KERNEL_DISPLAY_NAME}"
"${PYTHON_BIN}" - "${KERNEL_DIR}" <<'PY'
import json
import os
import pathlib
import sys

kernel_dir = pathlib.Path(sys.argv[1])
kernel_dir.mkdir(parents=True, exist_ok=True)
payload = {
    "argv": [
        "/bin/bash",
        os.environ["IFV_KERNEL_WRAPPER"],
        "{connection_file}",
    ],
    "display_name": os.environ["IFV_KERNEL_DISPLAY_NAME"],
    "language": "python",
    "metadata": {"debugger": False},
}
(kernel_dir / "kernel.json").write_text(
    json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
    encoding="utf-8",
)
PY
unset IFV_KERNEL_WRAPPER IFV_KERNEL_DISPLAY_NAME

"${JUPYTER_BIN}" kernelspec install --user --replace \
    --name "${KERNEL_NAME}" "${KERNEL_DIR}" >/dev/null
KERNEL_LIST_JSON="${KERNEL_DIR}/kernelspecs.json"
"${JUPYTER_BIN}" kernelspec list --json >"${KERNEL_LIST_JSON}"
"${PYTHON_BIN}" - "${KERNEL_NAME}" "${KERNEL_LIST_JSON}" <<'PY'
import json
from pathlib import Path
import sys

kernel_name = sys.argv[1]
payload = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
spec = payload.get("kernelspecs", {}).get(kernel_name, {}).get("spec", {})
argv = spec.get("argv", [])
if len(argv) < 3 or argv[0] != "/bin/bash" or argv[2] != "{connection_file}":
    raise SystemExit(f"Invalid installed kernel specification for {kernel_name}")
if spec.get("env", {}).get("OMP_NUM_THREADS") not in (None, "1"):
    raise SystemExit(f"Invalid OMP_NUM_THREADS in kernel specification for {kernel_name}")
print(f"installed Jupyter kernel: {kernel_name}")
PY
