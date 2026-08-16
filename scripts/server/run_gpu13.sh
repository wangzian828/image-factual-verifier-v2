#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=gpu13_env.sh
source "${SCRIPT_DIR}/gpu13_env.sh"

EXPECTED_BRANCH="codex/gpu13-canary-20260804-plan-relaxation-01"
EXPECTED_ROOT="/gs/home/wza/projects/image-factual-verifier-v2-worktrees/gpu13-canary-20260804-plan-relaxation-01"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
actual_branch="$(git -C "${REPO_ROOT}" branch --show-current)"
if [[ "${REPO_ROOT}" != "${EXPECTED_ROOT}" ]]; then
    echo "Refusing to run from a non-canonical gpu-13 checkout." >&2
    echo "expected=${EXPECTED_ROOT}" >&2
    echo "actual=${REPO_ROOT}" >&2
    exit 2
fi
if [[ "${actual_branch}" != "${EXPECTED_BRANCH}" ]]; then
    echo "Refusing to run from a non-canonical gpu-13 branch." >&2
    echo "expected=${EXPECTED_BRANCH}" >&2
    echo "actual=${actual_branch}" >&2
    exit 2
fi

if [[ "${OMP_NUM_THREADS}" != "1" ]]; then
    echo "OMP_NUM_THREADS must be 1 on gpu-13" >&2
    exit 2
fi

if [[ "$#" -eq 0 ]]; then
    echo "usage: $0 COMMAND [ARG ...]" >&2
    exit 2
fi

exec "$@"
