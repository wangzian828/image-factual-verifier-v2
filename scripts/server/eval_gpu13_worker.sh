#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Backward-compatible private wrapper for existing gpu-13 job records.
exec "${SCRIPT_DIR}/eval_worker.sh" "$@"
