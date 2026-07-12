#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=gpu13_env.sh
source "${SCRIPT_DIR}/gpu13_env.sh"

usage() {
    cat >&2 <<'EOF'
usage: start_eval_gpu13.sh --output-dir DIR [run_eval ARG ...]

Starts python -m src.eval.run_eval in the gpu-13 ifv-agent Conda environment.
All run_eval arguments, including --output-dir, are passed through unchanged.
EOF
}

args=("$@")
output_dir_seen=false
output_dir=""
for ((i = 0; i < ${#args[@]}; i++)); do
    case "${args[i]}" in
        --output-dir)
            if ((i + 1 >= ${#args[@]})) || [[ -z "${args[i + 1]}" ]]; then
                echo "--output-dir requires a non-empty value" >&2
                usage
                exit 2
            fi
            output_dir_seen=true
            output_dir="${args[i + 1]}"
            ((i += 1))
            ;;
        --output-dir=*)
            if [[ -z "${args[i]#--output-dir=}" ]]; then
                echo "--output-dir requires a non-empty value" >&2
                usage
                exit 2
            fi
            output_dir_seen=true
            output_dir="${args[i]#--output-dir=}"
            ;;
    esac
done

if [[ "${output_dir_seen}" != "true" ]]; then
    echo "A formal evaluation requires an explicit --output-dir." >&2
    usage
    exit 2
fi

if [[ -d "${output_dir}" ]] && find "${output_dir}" -mindepth 1 -print -quit | grep -q .; then
    echo "Evaluation output directory must be new or empty: ${output_dir}" >&2
    exit 2
fi

LOG_DIR="${IFV_DATA_ROOT}/runs/_logs"
PID_DIR="/tmp/image-factual-verifier-v2"

mkdir -p -- "${LOG_DIR}"
if [[ ! "${IFV_LOG_RETENTION_DAYS}" =~ ^[0-9]+$ ]]; then
    echo "IFV_LOG_RETENTION_DAYS must be a non-negative integer" >&2
    exit 2
fi
find "${LOG_DIR}" -maxdepth 1 -type f -name 'eval-*.log' \
    -mtime "+${IFV_LOG_RETENTION_DAYS}" -delete
if [[ -L "${PID_DIR}" ]]; then
    echo "Refusing to use symlinked PID directory: ${PID_DIR}" >&2
    exit 2
fi
mkdir -p -- "${PID_DIR}"
if [[ ! -O "${PID_DIR}" ]]; then
    echo "PID directory is not owned by the current user: ${PID_DIR}" >&2
    exit 2
fi
chmod 700 -- "${PID_DIR}"

umask 077
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
log_file="$(mktemp "${LOG_DIR}/eval-${timestamp}-XXXXXX.log")"
job_name="$(basename -- "${log_file}" .log)"
pid_file="${PID_DIR}/${job_name}.pid"

nohup bash "${SCRIPT_DIR}/eval_gpu13_worker.sh" "${job_name}" "$@" \
    </dev/null >>"${log_file}" 2>&1 &
pid=$!

printf 'pid=%s\n' "${pid}"
printf 'pid_file=%s\n' "${pid_file}"
printf 'log_file=%s\n' "${log_file}"
