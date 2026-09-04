#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=ifv_env.sh
source "${SCRIPT_DIR}/ifv_env.sh"

REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

if [[ -n "${IFV_EXPECTED_REPO_ROOT:-}" ]]; then
    expected_root="$(realpath -m -- "${IFV_EXPECTED_REPO_ROOT}")"
    if [[ "${REPO_ROOT}" != "${expected_root}" ]]; then
        echo "Repository root mismatch." >&2
        echo "expected=${expected_root}" >&2
        echo "actual=${REPO_ROOT}" >&2
        exit 2
    fi
fi

actual_branch="$(git -C "${REPO_ROOT}" branch --show-current)"
if [[ -n "${IFV_EXPECTED_BRANCH:-}" && "${actual_branch}" != "${IFV_EXPECTED_BRANCH}" ]]; then
    echo "Repository branch mismatch." >&2
    echo "expected=${IFV_EXPECTED_BRANCH}" >&2
    echo "actual=${actual_branch}" >&2
    exit 2
fi

if [[ "${IFV_REQUIRE_CLEAN_CHECKOUT:-1}" == "1" ]] \
    && [[ -n "$(git -C "${REPO_ROOT}" status --short)" ]]; then
    echo "Repository worktree must be clean before a server run." >&2
    git -C "${REPO_ROOT}" status --short >&2
    exit 2
fi

if [[ "${OMP_NUM_THREADS}" != "1" ]]; then
    echo "IFV server runs require OMP_NUM_THREADS=1; got ${OMP_NUM_THREADS}." >&2
    exit 2
fi

if [[ "$#" -eq 0 ]]; then
    echo "usage: $0 COMMAND [ARG ...]" >&2
    exit 2
fi

cd -- "${REPO_ROOT}"
exec "$@"
