#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/gpu13_env.sh"

if [[ "${OMP_NUM_THREADS:-}" != "1" ]]; then
    echo "Teacher rollout requires OMP_NUM_THREADS=1; got ${OMP_NUM_THREADS:-unset}." >&2
    exit 2
fi

args=("$@")
background="${IFV_TEACHER_ROLLOUT_BACKGROUND:-0}"
filtered=()
has_dataset_root=false
has_output_dir=false
reroll_mode=false
for ((index = 0; index < ${#args[@]}; index++)); do
    argument="${args[index]}"
    case "${argument}" in
        --background)
            background="1"
            ;;
        --dataset-root)
            has_dataset_root=true
            filtered+=("${argument}")
            ((index += 1))
            ((index < ${#args[@]})) || { echo "--dataset-root requires a value." >&2; exit 2; }
            filtered+=("${args[index]}")
            ;;
        --dataset-root=*)
            has_dataset_root=true
            filtered+=("${argument}")
            ;;
        --output-dir)
            has_output_dir=true
            filtered+=("${argument}")
            ((index += 1))
            ((index < ${#args[@]})) || { echo "--output-dir requires a value." >&2; exit 2; }
            filtered+=("${args[index]}")
            ;;
        --output-dir=*)
            has_output_dir=true
            filtered+=("${argument}")
            ;;
        --reroll-from|--bootstrap-initial-run|--bootstrap-initial-eligibility)
            reroll_mode=true
            filtered+=("${argument}")
            ((index += 1))
            ((index < ${#args[@]})) || { echo "${argument} requires a value." >&2; exit 2; }
            filtered+=("${args[index]}")
            ;;
        --reroll-from=*|--bootstrap-initial-run=*|--bootstrap-initial-eligibility=*)
            reroll_mode=true
            filtered+=("${argument}")
            ;;
        *)
            filtered+=("${argument}")
            ;;
    esac
done

if [[ "${reroll_mode}" != "true" ]]; then
    if [[ "${has_dataset_root}" != "true" ]]; then
        dataset_root="${IFV_TEACHER_DATASET_ROOT:-${IFV_DATA_ROOT}/datasets/route-aware-hrc-stage2-10563-final-organized-20260824-r2/unified-dataset}"
        filtered+=("--dataset-root" "${dataset_root}")
    fi
    if [[ "${has_output_dir}" != "true" ]]; then
        run_id="${IFV_TEACHER_ROLLOUT_RUN_ID:-teacher-rollout-$(date -u +%Y%m%dT%H%M%SZ)}"
        filtered+=("--output-dir" "${IFV_DATA_ROOT}/generated/teacher-rollouts/${run_id}")
    fi
fi

command=(
    "${SCRIPT_DIR}/run_gpu13.sh"
    python
    "${REPO_ROOT}/scripts/trajectory/run_teacher_rollout_autopilot.py"
    "${filtered[@]}"
)

if [[ "${background}" != "1" ]]; then
    exec "${command[@]}"
fi

log_root="${IFV_DATA_ROOT}/runs/_logs"
pid_root="${IFV_DATA_ROOT}/runs/_pids"
mkdir -p -- "${log_root}" "${pid_root}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
log_file="${log_root}/teacher-rollout-${timestamp}.log"
pid_file="${pid_root}/teacher-rollout-${timestamp}.pid"
nohup "${command[@]}" </dev/null >"${log_file}" 2>&1 &
pid=$!
printf '%s\n' "${pid}" >"${pid_file}"
printf 'pid=%s\nlog_file=%s\npid_file=%s\n' "${pid}" "${log_file}" "${pid_file}"
