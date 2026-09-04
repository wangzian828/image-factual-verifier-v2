#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=ifv_env.sh
source "${SCRIPT_DIR}/ifv_env.sh"

usage() {
    cat >&2 <<'EOF'
usage: start_eval.sh --output-dir DIR [run_eval ARG ...]

Starts src.eval.run_eval as a managed background job. DIR must be one new
directory under $IFV_DATA_ROOT/runs/eval.
EOF
}

args=("$@")
output_dir_seen=false
output_dir=""
requested_concurrency=1
explicit_profile=""
explicit_provider=""
for ((i = 0; i < ${#args[@]}; i++)); do
    case "${args[i]}" in
        --output-dir)
            ((i + 1 < ${#args[@]})) || { usage; exit 2; }
            output_dir_seen=true
            output_dir="${args[i + 1]}"
            ((i += 1))
            ;;
        --output-dir=*)
            output_dir_seen=true
            output_dir="${args[i]#--output-dir=}"
            ;;
        --concurrency)
            ((i + 1 < ${#args[@]})) || { usage; exit 2; }
            requested_concurrency="${args[i + 1]}"
            ((i += 1))
            ;;
        --concurrency=*)
            requested_concurrency="${args[i]#--concurrency=}"
            ;;
        --profile)
            ((i + 1 < ${#args[@]})) || { usage; exit 2; }
            explicit_profile="${args[i + 1]}"
            ((i += 1))
            ;;
        --profile=*)
            explicit_profile="${args[i]#--profile=}"
            ;;
        --provider)
            ((i + 1 < ${#args[@]})) || { usage; exit 2; }
            explicit_provider="${args[i + 1]}"
            ((i += 1))
            ;;
        --provider=*)
            explicit_provider="${args[i]#--provider=}"
            ;;
    esac
done

if [[ "${output_dir_seen}" != "true" || -z "${output_dir}" ]]; then
    usage
    exit 2
fi
if [[ ! "${requested_concurrency}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--concurrency must be a positive integer" >&2
    exit 2
fi

eval_root="$(realpath -m -- "${IFV_DATA_ROOT}/runs/eval")"
output_dir="$(realpath -m -- "${output_dir}")"
case "${output_dir}" in
    "${eval_root}"/*) ;;
    *)
        echo "Evaluation output must be under ${eval_root}/<run-id>." >&2
        exit 2
        ;;
esac
run_id="${output_dir#"${eval_root}"/}"
if [[ -z "${run_id}" || "${run_id}" == */* || "${run_id}" == *[!A-Za-z0-9._-]* ]]; then
    echo "Evaluation output must contain one safe run-id component." >&2
    exit 2
fi
if [[ -d "${output_dir}" ]] \
    && find "${output_dir}" -mindepth 1 -print -quit | grep -q .; then
    echo "Evaluation output directory must be new or empty: ${output_dir}" >&2
    exit 2
fi
for ((i = 0; i < ${#args[@]}; i++)); do
    case "${args[i]}" in
        --output-dir) args[i + 1]="${output_dir}" ;;
        --output-dir=*) args[i]="--output-dir=${output_dir}" ;;
    esac
done

gemini_mode=true
if [[ -n "${explicit_profile}" ]]; then
    [[ "${explicit_profile}" == "teacher-gemini" ]] || gemini_mode=false
elif [[ -n "${explicit_provider}" ]]; then
    [[ "${explicit_provider,,}" == "gemini" ]] || gemini_mode=false
fi

gemini_lock_dir=""
gemini_lock_handoff=false
release_gemini_lock() {
    if [[ -z "${gemini_lock_dir}" || ! -d "${gemini_lock_dir}" ]]; then
        return 0
    fi
    rm -f -- "${gemini_lock_dir}/owner.env"
    rmdir -- "${gemini_lock_dir}" 2>/dev/null || true
}
cleanup_startup_lock() {
    if [[ "${gemini_lock_handoff}" != "true" ]]; then
        release_gemini_lock
    fi
}
trap cleanup_startup_lock EXIT

effective_gemini_request_limit=""
if [[ "${gemini_mode}" == "true" ]]; then
    gemini_eval_cap="${GEMINI_EVAL_MAX_CONCURRENCY:-${requested_concurrency}}"
    configured_request_limit="${GEMINI_MAX_INFLIGHT_REQUESTS:-${requested_concurrency}}"
    for value in "${gemini_eval_cap}" "${configured_request_limit}"; do
        [[ "${value}" =~ ^[1-9][0-9]*$ ]] || {
            echo "Gemini concurrency limits must be positive integers." >&2
            exit 2
        }
    done
    if ((requested_concurrency > gemini_eval_cap)); then
        echo "Requested concurrency exceeds GEMINI_EVAL_MAX_CONCURRENCY." >&2
        exit 2
    fi
    effective_gemini_request_limit="${configured_request_limit}"
    if ((effective_gemini_request_limit > gemini_eval_cap)); then
        effective_gemini_request_limit="${gemini_eval_cap}"
    fi
    export GEMINI_MAX_INFLIGHT_REQUESTS="${effective_gemini_request_limit}"

    lock_parent="${IFV_DATA_ROOT}/runs/_locks"
    gemini_lock_dir="${lock_parent}/gemini-agent-eval.lock"
    mkdir -p -- "${lock_parent}"
    for _attempt in 1 2; do
        if mkdir -- "${gemini_lock_dir}" 2>/dev/null; then
            break
        fi
        owner_pid="$(
            sed -n 's/^owner_pid=//p' "${gemini_lock_dir}/owner.env" \
                2>/dev/null | head -n 1
        )"
        if [[ "${owner_pid}" =~ ^[1-9][0-9]*$ ]] \
            && kill -0 "${owner_pid}" 2>/dev/null; then
            echo "A Gemini evaluation is already active." >&2
            exit 3
        fi
        rm -f -- "${gemini_lock_dir}/owner.env"
        rmdir -- "${gemini_lock_dir}" 2>/dev/null || true
    done
    [[ -d "${gemini_lock_dir}" ]] || {
        echo "Could not acquire Gemini evaluation lock." >&2
        exit 3
    }
fi

log_dir="${IFV_DATA_ROOT}/runs/_logs"
job_dir="${IFV_DATA_ROOT}/runs/_jobs"
pid_root="${IFV_PID_ROOT:-${TMPDIR:-/tmp}/image-factual-verifier}"
mkdir -p -- "${log_dir}" "${job_dir}" "${pid_root}"
chmod 700 -- "${pid_root}"
find "${log_dir}" -maxdepth 1 -type f -name 'eval-*.log' \
    -mtime "+${IFV_LOG_RETENTION_DAYS}" -delete

umask 077
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
log_file="$(mktemp "${log_dir}/eval-${timestamp}-XXXXXX.log")"
job_name="$(basename -- "${log_file}" .log)"
pid_file="${pid_root}/${job_name}.pid"

if [[ "${gemini_mode}" == "true" ]]; then
    export IFV_GEMINI_LOCK_DIR="${gemini_lock_dir}"
fi
nohup bash "${SCRIPT_DIR}/eval_worker.sh" "${job_name}" "${args[@]}" \
    </dev/null >>"${log_file}" 2>&1 &
pid=$!

if [[ "${gemini_mode}" == "true" ]]; then
    {
        printf 'owner_pid=%s\n' "${pid}"
        printf 'run_id=%s\n' "${run_id}"
        printf 'requested_concurrency=%s\n' "${requested_concurrency}"
        printf 'request_limit=%s\n' "${effective_gemini_request_limit}"
        printf 'output_dir=%s\n' "${output_dir}"
        printf 'log_file=%s\n' "${log_file}"
    } >"${gemini_lock_dir}/owner.env"
    gemini_lock_handoff=true
fi

job_record="${job_dir}/${job_name}.env"
{
    printf 'run_id=%s\n' "${run_id}"
    printf 'output_dir=%s\n' "${output_dir}"
    printf 'log_file=%s\n' "${log_file}"
    printf 'pid_file=%s\n' "${pid_file}"
    printf 'pid=%s\n' "${pid}"
} >"${job_record}"

printf 'pid=%s\n' "${pid}"
printf 'log_file=%s\n' "${log_file}"
printf 'run_id=%s\n' "${run_id}"
printf 'status_command=scripts/server/poll_eval.sh %s\n' "${run_id}"
