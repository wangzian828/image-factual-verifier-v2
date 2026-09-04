#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$#" -lt 2 ]]; then
    echo "eval_worker.sh is an internal helper for start_eval.sh" >&2
    exit 2
fi

job_name="$1"
shift
if [[ "${job_name}" != eval-* ]] || [[ "${job_name}" == *[!A-Za-z0-9._-]* ]]; then
    echo "Invalid evaluation job name" >&2
    exit 2
fi

pid_root="${IFV_PID_ROOT:-${TMPDIR:-/tmp}/image-factual-verifier}"
pid_file="${pid_root}/${job_name}.pid"
child_pid=""
gemini_lock_dir="${IFV_GEMINI_LOCK_DIR:-}"

cleanup() {
    rm -f -- "${pid_file}"
    if [[ -n "${gemini_lock_dir}" && -d "${gemini_lock_dir}" ]]; then
        owner_pid=""
        if [[ -f "${gemini_lock_dir}/owner.env" ]]; then
            owner_pid="$(
                sed -n 's/^owner_pid=//p' "${gemini_lock_dir}/owner.env" \
                    | head -n 1
            )"
        fi
        if [[ "${owner_pid}" == "$$" ]]; then
            rm -f -- "${gemini_lock_dir}/owner.env"
            rmdir -- "${gemini_lock_dir}" 2>/dev/null || true
        fi
    fi
}
terminate() {
    if [[ -n "${child_pid}" ]] && kill -0 "${child_pid}" 2>/dev/null; then
        kill -TERM "${child_pid}" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap terminate INT TERM

mkdir -p -- "${pid_root}"
umask 077
printf '%s\n' "$$" >"${pid_file}"
printf 'started_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'pid=%s\n' "$$"

runtime_python="${IFV_RUNTIME_PYTHON:-python}"
"${SCRIPT_DIR}/run_ifv.sh" \
    "${runtime_python}" -m src.eval.run_eval "$@" &
child_pid=$!

set +e
wait "${child_pid}"
status=$?
set -e

printf 'finished_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'exit_status=%s\n' "${status}"
exit "${status}"
