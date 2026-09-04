#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=ifv_env.sh
source "${SCRIPT_DIR}/ifv_env.sh"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

if [[ "$#" -lt 1 || "$#" -gt 3 ]]; then
    echo "usage: $0 RUN_ID [--tail LINES]" >&2
    exit 2
fi
run_id="$1"
shift
tail_lines=40
if [[ "$#" -gt 0 ]]; then
    if [[ "$1" != "--tail" || "$#" -ne 2 || ! "${2}" =~ ^[0-9]+$ ]]; then
        echo "usage: $0 RUN_ID [--tail LINES]" >&2
        exit 2
    fi
    tail_lines="$2"
fi
if [[ -z "${run_id}" || "${run_id}" == */* || "${run_id}" == *[!A-Za-z0-9._-]* ]]; then
    echo "RUN_ID must be one safe directory name." >&2
    exit 2
fi

run_dir="${IFV_DATA_ROOT}/runs/eval/${run_id}"
[[ -d "${run_dir}" ]] || {
    echo "Evaluation run does not exist: ${run_dir}" >&2
    exit 1
}

printf 'run_id=%s\n' "${run_id}"
printf 'run_dir=%s\n' "${run_dir}"
printf 'checkout_branch=%s\n' "$(git -C "${REPO_ROOT}" branch --show-current)"
printf 'checkout_head=%s\n' "$(git -C "${REPO_ROOT}" log -1 --oneline --decorate)"
printf 'trace_count=%s\n' "$(
    find "${run_dir}/traces" -maxdepth 1 -type f -name '*.json' \
        2>/dev/null | wc -l | tr -d ' '
)"
if [[ -f "${run_dir}/run_results.jsonl" ]]; then
    printf 'result_count=%s\n' "$(wc -l <"${run_dir}/run_results.jsonl" | tr -d ' ')"
else
    printf 'result_count=0\n'
fi
if [[ -f "${run_dir}/summary.json" ]]; then
    printf '%s\n' 'summary:'
    python3 - "${run_dir}/summary.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
for key in (
    "status",
    "completed",
    "episode_count",
    "case_count",
    "correct",
    "accuracy",
    "engineering_errors",
):
    if key in payload:
        print(f"{key}={payload[key]}")
PY
else
    printf 'summary=not_yet_written\n'
fi

job_record="$(
    grep -l -F -- "run_id=${run_id}" "${IFV_DATA_ROOT}/runs/_jobs"/*.env \
        2>/dev/null | head -n 1 || true
)"
if [[ -n "${job_record}" ]]; then
    printf 'job_record=%s\n' "${job_record}"
    awk -F= '/^(log_file|pid_file|pid)=/ { print }' "${job_record}"
    log_file="$(
        awk -F= '$1 == "log_file" {
            print substr($0, index($0, "=")+1)
        }' "${job_record}"
    )"
    if [[ -n "${log_file}" && -f "${log_file}" ]]; then
        printf '%s\n' 'log_tail:'
        tail -n "${tail_lines}" "${log_file}"
    fi
fi
