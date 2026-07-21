#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=gpu13_env.sh
source "${SCRIPT_DIR}/gpu13_env.sh"

BRANCH="${1:-codex/image-factual-verifier-v3}"

if [[ "$(hostname)" != "gpu-13" ]]; then
    echo "update_gpu13_checkout.sh must run on gpu-13" >&2
    exit 2
fi

if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain)" ]]; then
    echo "Refusing to update a dirty server worktree." >&2
    echo "Server-side source edits are prohibited; inspect and discard them manually." >&2
    exit 2
fi

REMOTE_REF="refs/heads/${BRANCH}"
TRACKING_REF="refs/remotes/origin/${BRANCH}"

# Some single-branch clones do not have a wildcard fetch refspec. An unqualified
# `git fetch origin` then succeeds while leaving this branch's tracking ref stale.
# Fetch the requested branch into its exact tracking ref before fast-forwarding.
git -C "${REPO_ROOT}" fetch --prune origin \
    "+${REMOTE_REF}:${TRACKING_REF}"
git -C "${REPO_ROOT}" checkout "${BRANCH}"
git -C "${REPO_ROOT}" merge --ff-only "origin/${BRANCH}"

remote_head="$(git -C "${REPO_ROOT}" rev-parse "${TRACKING_REF}")"
local_head="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
if [[ "${local_head}" != "${remote_head}" ]]; then
    echo "Server checkout did not reach ${TRACKING_REF}." >&2
    echo "local=${local_head} remote=${remote_head}" >&2
    exit 2
fi
git -C "${REPO_ROOT}" status --short --branch
