#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=gpu13_env.sh
source "${SCRIPT_DIR}/gpu13_env.sh"

BRANCH="${1:-codex/gemini-interactions-agent}"

if [[ "$(hostname)" != "gpu-13" ]]; then
    echo "update_gpu13_checkout.sh must run on gpu-13" >&2
    exit 2
fi

if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain)" ]]; then
    echo "Refusing to update a dirty server worktree." >&2
    echo "Server-side source edits are prohibited; inspect and discard them manually." >&2
    exit 2
fi

git -C "${REPO_ROOT}" fetch --prune origin
git -C "${REPO_ROOT}" checkout "${BRANCH}"
git -C "${REPO_ROOT}" merge --ff-only "origin/${BRANCH}"
git -C "${REPO_ROOT}" status --short --branch
