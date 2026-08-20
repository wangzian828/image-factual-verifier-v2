#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=gpu13_env.sh
source "${SCRIPT_DIR}/gpu13_env.sh"

EXPECTED_BRANCH="codex/gpu13-canary-20260804-plan-relaxation-01"
EXPECTED_ROOT="/gs/home/wza/projects/image-factual-verifier-v2-worktrees/gpu13-canary-20260804-plan-relaxation-01"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
actual_branch="$(git -C "${REPO_ROOT}" branch --show-current)"
if [[ "${REPO_ROOT}" != "${EXPECTED_ROOT}" ]]; then
    echo "Refusing to start evaluation from a non-canonical gpu-13 checkout." >&2
    echo "expected=${EXPECTED_ROOT}" >&2
    echo "actual=${REPO_ROOT}" >&2
    exit 2
fi
if [[ "${actual_branch}" != "${EXPECTED_BRANCH}" ]]; then
    echo "Refusing to start evaluation from a non-canonical gpu-13 branch." >&2
    echo "expected=${EXPECTED_BRANCH}" >&2
    echo "actual=${actual_branch}" >&2
    exit 2
fi

usage() {
    cat >&2 <<'EOF'
usage: start_eval_gpu13.sh --output-dir DIR [run_eval ARG ...]

Starts python -m src.eval.run_eval in the gpu-13 ifv-agent Conda environment.
All run_eval arguments are forwarded; --output-dir is normalized to the
canonical absolute run path before the child starts.
The output directory must be a single run directory under
$IFV_DATA_ROOT/runs/eval/<run-id>. The launcher prints the canonical run-id
and the status command to use for polling.
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
        --concurrency)
            if ((i + 1 >= ${#args[@]})) || [[ -z "${args[i + 1]}" ]]; then
                echo "--concurrency requires a positive integer" >&2
                usage
                exit 2
            fi
            requested_concurrency="${args[i + 1]}"
            ((i += 1))
            ;;
        --concurrency=*)
            requested_concurrency="${args[i]#--concurrency=}"
            ;;
        --profile)
            if ((i + 1 >= ${#args[@]})) || [[ -z "${args[i + 1]}" ]]; then
                echo "--profile requires a non-empty value" >&2
                usage
                exit 2
            fi
            explicit_profile="${args[i + 1]}"
            ((i += 1))
            ;;
        --profile=*)
            explicit_profile="${args[i]#--profile=}"
            ;;
        --provider)
            if ((i + 1 >= ${#args[@]})) || [[ -z "${args[i + 1]}" ]]; then
                echo "--provider requires a non-empty value" >&2
                usage
                exit 2
            fi
            explicit_provider="${args[i + 1]}"
            ((i += 1))
            ;;
        --provider=*)
            explicit_provider="${args[i]#--provider=}"
            ;;
    esac
done

if [[ ! "${requested_concurrency}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--concurrency must be a positive integer" >&2
    exit 2
fi

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

if [[ "${output_dir_seen}" != "true" ]]; then
    echo "A formal evaluation requires an explicit --output-dir." >&2
    usage
    exit 2
fi

eval_root="$(realpath -m -- "${IFV_DATA_ROOT}/runs/eval")"
output_dir="$(realpath -m -- "${output_dir}")"
case "${output_dir}" in
    "${eval_root}"/*) ;;
    *)
        echo "Evaluation output directory must be under ${eval_root}/<run-id>." >&2
        echo "actual=${output_dir}" >&2
        exit 2
        ;;
esac
run_id="${output_dir#"${eval_root}"/}"
if [[ -z "${run_id}" || "${run_id}" == */* || "${run_id}" == *[!A-Za-z0-9._-]* ]]; then
    echo "Evaluation output directory must contain one safe <run-id> component." >&2
    echo "actual=${output_dir}" >&2
    exit 2
fi

# Pass the same canonical absolute path to run_eval that was validated above.
# Otherwise a relative --output-dir could be validated here but resolved again
# against a different working directory by the child process.
for ((i = 0; i < ${#args[@]}; i++)); do
    case "${args[i]}" in
        --output-dir)
            args[i + 1]="${output_dir}"
            ;;
        --output-dir=*)
            args[i]="--output-dir=${output_dir}"
            ;;
    esac
done

if [[ -d "${output_dir}" ]] && find "${output_dir}" -mindepth 1 -print -quit | grep -q .; then
    echo "Evaluation output directory must be new or empty: ${output_dir}" >&2
    exit 2
fi

effective_gemini_request_limit=""
if [[ "${gemini_mode}" == "true" ]]; then
    gemini_eval_cap="${GEMINI_EVAL_MAX_CONCURRENCY:-4}"
    if [[ ! "${gemini_eval_cap}" =~ ^[1-9][0-9]*$ ]]; then
        echo "GEMINI_EVAL_MAX_CONCURRENCY must be a positive integer" >&2
        exit 2
    fi
    if ((requested_concurrency > gemini_eval_cap)); then
        echo "Gemini rollout concurrency ${requested_concurrency} exceeds the "
        echo "configured cap ${gemini_eval_cap}. Set --concurrency <= "
        echo "${gemini_eval_cap}, or raise GEMINI_EVAL_MAX_CONCURRENCY deliberately." >&2
        exit 2
    fi

    configured_request_limit="${GEMINI_MAX_INFLIGHT_REQUESTS:-${requested_concurrency}}"
    if [[ ! "${configured_request_limit}" =~ ^[1-9][0-9]*$ ]]; then
        echo "GEMINI_MAX_INFLIGHT_REQUESTS must be a positive integer" >&2
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

    lock_acquired=false
    for _lock_attempt in 1 2; do
        if mkdir -- "${gemini_lock_dir}" 2>/dev/null; then
            lock_acquired=true
            break
        fi

        owner_pid=""
        if [[ -f "${gemini_lock_dir}/owner.env" ]]; then
            owner_pid="$(
                sed -n 's/^owner_pid=//p' "${gemini_lock_dir}/owner.env" \
                    | head -n 1
            )"
        fi
        if [[ "${owner_pid}" =~ ^[1-9][0-9]*$ ]] \
            && kill -0 "${owner_pid}" 2>/dev/null; then
            echo "A Gemini evaluation is already active; refusing a second "
            echo "launch to avoid provider overload." >&2
            echo "lock=${gemini_lock_dir}" >&2
            if [[ -f "${gemini_lock_dir}/owner.env" ]]; then
                sed 's/^/  /' "${gemini_lock_dir}/owner.env" >&2
            fi
            exit 3
        fi

        # The owner died without running its EXIT trap (for example SIGKILL).
        # The lock contains only owner.env, so remove that file and retry the
        # atomic mkdir rather than recursively deleting a computed path.
        rm -f -- "${gemini_lock_dir}/owner.env"
        rmdir -- "${gemini_lock_dir}" 2>/dev/null || true
    done
    if [[ "${lock_acquired}" != "true" ]]; then
        echo "Could not acquire the Gemini evaluation lock: ${gemini_lock_dir}" >&2
        exit 3
    fi
    {
        printf 'owner_pid=%s\n' "$$"
        printf 'run_id=%s\n' "${run_id}"
        printf 'requested_concurrency=%s\n' "${requested_concurrency}"
        printf 'request_limit=%s\n' "${effective_gemini_request_limit}"
        printf 'created_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    } >"${gemini_lock_dir}/owner.env"
fi

LOG_DIR="${IFV_DATA_ROOT}/runs/_logs"
PID_DIR="/tmp/image-factual-verifier-v3"
JOB_DIR="${IFV_DATA_ROOT}/runs/_jobs"

mkdir -p -- "${LOG_DIR}"
mkdir -p -- "${JOB_DIR}"
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

if [[ "${gemini_mode}" == "true" ]]; then
    export IFV_GEMINI_LOCK_DIR="${gemini_lock_dir}"
fi
nohup bash "${SCRIPT_DIR}/eval_gpu13_worker.sh" "${job_name}" "$@" \
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
        printf 'created_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    } >"${gemini_lock_dir}/owner.env"
    gemini_lock_handoff=true
fi

printf 'pid=%s\n' "${pid}"
printf 'pid_file=%s\n' "${pid_file}"
printf 'log_file=%s\n' "${log_file}"
printf 'run_id=%s\n' "${run_id}"
if [[ "${gemini_mode}" == "true" ]]; then
    printf 'gemini_rollout_concurrency=%s\n' "${requested_concurrency}"
    printf 'gemini_request_limit=%s\n' "${effective_gemini_request_limit}"
    printf 'gemini_lock=%s\n' "${gemini_lock_dir}"
fi
printf 'status_command=scripts/server/poll_eval_gpu13.sh %s\n' "${run_id}"

job_record="${JOB_DIR}/${job_name}.env"
{
    printf 'run_id=%s\n' "${run_id}"
    printf 'output_dir=%s\n' "${output_dir}"
    printf 'log_file=%s\n' "${log_file}"
    printf 'pid_file=%s\n' "${pid_file}"
    printf 'pid=%s\n' "${pid}"
    if [[ "${gemini_mode}" == "true" ]]; then
        printf 'gemini_rollout_concurrency=%s\n' "${requested_concurrency}"
        printf 'gemini_request_limit=%s\n' "${effective_gemini_request_limit}"
        printf 'gemini_lock=%s\n' "${gemini_lock_dir}"
    fi
} >"${job_record}"
