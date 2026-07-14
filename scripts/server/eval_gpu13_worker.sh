#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$#" -lt 2 ]]; then
    echo "eval_gpu13_worker.sh is an internal helper for start_eval_gpu13.sh" >&2
    exit 2
fi

job_name="$1"
shift
if [[ "${job_name}" != eval-* ]] || [[ "${job_name}" == *[!A-Za-z0-9._-]* ]]; then
    echo "Invalid evaluation job name" >&2
    exit 2
fi

pid_file="/tmp/image-factual-verifier-v3/${job_name}.pid"
child_pid=""

cleanup() {
    rm -f -- "${pid_file}"
}
terminate() {
    if [[ -n "${child_pid}" ]] && kill -0 "${child_pid}" 2>/dev/null; then
        kill -TERM "${child_pid}" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap terminate INT TERM

umask 077
printf '%s\n' "$$" >"${pid_file}"
printf 'started_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'pid=%s\n' "$$"

"${SCRIPT_DIR}/run_gpu13.sh" \
    conda run --no-capture-output --name ifv-agent \
    python -m src.eval.run_eval "$@" &
child_pid=$!

set +e
wait "${child_pid}"
status=$?
set -e

printf 'finished_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'exit_status=%s\n' "${status}"
exit "${status}"
