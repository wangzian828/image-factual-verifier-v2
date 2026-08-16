#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=gpu13_env.sh
source "${SCRIPT_DIR}/gpu13_env.sh"

EXPECTED_BRANCH="codex/gpu13-canary-20260804-plan-relaxation-01"
EXPECTED_ROOT="/gs/home/wza/projects/image-factual-verifier-v2-worktrees/gpu13-canary-20260804-plan-relaxation-01"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

if [[ "${REPO_ROOT}" != "${EXPECTED_ROOT}" ]]; then
    echo "Refusing to poll from a non-canonical gpu-13 checkout." >&2
    echo "expected=${EXPECTED_ROOT}" >&2
    echo "actual=${REPO_ROOT}" >&2
    exit 2
fi
if [[ "$(git -C "${REPO_ROOT}" branch --show-current)" != "${EXPECTED_BRANCH}" ]]; then
    echo "Refusing to poll from a non-canonical gpu-13 branch." >&2
    echo "expected=${EXPECTED_BRANCH}" >&2
    echo "actual=$(git -C "${REPO_ROOT}" branch --show-current)" >&2
    exit 2
fi

usage() {
    cat >&2 <<'EOF'
usage: poll_eval_gpu13.sh RUN_ID [--tail LINES]

RUN_ID is the final directory name under:
  $IFV_DATA_ROOT/runs/eval/<run-id>

Do not pass a checkout path or a full output path. This command resolves the
run directory from the canonical data root and refuses unsafe run identifiers.
EOF
}

if [[ "$#" -lt 1 || "$#" -gt 3 ]]; then
    usage
    exit 2
fi

run_id="$1"
shift
tail_lines=40
if [[ "$#" -gt 0 ]]; then
    if [[ "$1" != "--tail" || "$#" -ne 2 || ! "${2}" =~ ^[0-9]+$ ]]; then
        usage
        exit 2
    fi
    tail_lines="$2"
fi
if [[ -z "${run_id}" || "${run_id}" == */* || "${run_id}" == *[!A-Za-z0-9._-]* ]]; then
    echo "RUN_ID must be one safe directory name, not a path." >&2
    exit 2
fi

run_dir="${IFV_DATA_ROOT}/runs/eval/${run_id}"
if [[ ! -d "${run_dir}" ]]; then
    echo "Evaluation run does not exist: ${run_id}" >&2
    echo "Expected directory: ${IFV_DATA_ROOT}/runs/eval/${run_id}" >&2
    exit 1
fi

printf 'run_id=%s\n' "${run_id}"
printf 'run_dir=%s\n' "${run_dir}"
printf 'checkout_branch=%s\n' "$(git -C "${REPO_ROOT}" branch --show-current)"
printf 'checkout_head=%s\n' "$(git -C "${REPO_ROOT}" log -1 --oneline --decorate)"

trace_count="$(find "${run_dir}/traces" -maxdepth 1 -type f -name '*.json' 2>/dev/null | wc -l | tr -d ' ')"
result_count=0
if [[ -f "${run_dir}/run_results.jsonl" ]]; then
    result_count="$(wc -l <"${run_dir}/run_results.jsonl" | tr -d ' ')"
fi
printf 'trace_count=%s\n' "${trace_count}"
printf 'result_count=%s\n' "${result_count}"

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

processes="$(ps -eo pid=,etime=,args= | grep -F -- "--output-dir ${run_dir}" | grep -v grep || true)"
if [[ -n "${processes}" ]]; then
    printf '%s\n' 'status=running'
    printf '%s\n' 'processes:'
    printf '%s\n' "${processes}"
else
    printf '%s\n' 'status=not_running_or_finished'
fi

job_record_dir="${IFV_DATA_ROOT}/runs/_jobs"
job_record="$(grep -l -F -- "run_id=${run_id}" "${job_record_dir}"/*.env 2>/dev/null | head -1 || true)"
if [[ -n "${job_record}" ]]; then
    printf 'job_record=%s\n' "${job_record}"
    awk -F= '/^(log_file|pid_file|pid)=/ { print }' "${job_record}"
    log_file="$(awk -F= '$1 == "log_file" { print substr($0, index($0, "=")+1) }' "${job_record}")"
    if [[ -n "${log_file}" && -f "${log_file}" ]]; then
        printf '%s\n' 'log_tail:'
        tail -n "${tail_lines}" "${log_file}"
    fi
else
    printf 'job_record=not_registered_by_current_launcher\n'
fi
