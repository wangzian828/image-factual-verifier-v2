#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/ifv_env.sh"

if [[ "${OMP_NUM_THREADS:-}" != "1" ]]; then
    echo "IFV server runs require OMP_NUM_THREADS=1; got ${OMP_NUM_THREADS:-unset}." >&2
    exit 2
fi

output_dir=""
run_training="0"
smoke_then_full="0"
smoke_limit="${IFV_SMOKE_CASE_COUNT:-10}"
autopilot_has_limit="0"
training_model_profile="${IFV_TRAINING_MODEL_PROFILE:-}"
training_sft_profile="${IFV_TRAINING_SFT_PROFILE:-}"
training_experiment_id="${IFV_TRAINING_EXPERIMENT_ID:-}"
training_resume_checkpoint="${IFV_TRAINING_RESUME_CHECKPOINT:-}"
args=("$@")
autopilot_args=()
index=0
while ((index < ${#args[@]})); do
    argument="${args[index]}"
    case "${argument}" in
        --smoke-then-full)
            smoke_then_full="1"
            index=$((index + 1))
            ;;
        --smoke-limit)
            if ((index + 1 >= ${#args[@]})); then
                echo "--smoke-limit requires a value." >&2
                exit 2
            fi
            smoke_limit="${args[index + 1]}"
            index=$((index + 2))
            ;;
        --smoke-limit=*)
            smoke_limit="${argument#*=}"
            index=$((index + 1))
            ;;
        --run-training)
            run_training="1"
            index=$((index + 1))
            ;;
        --no-run-training)
            run_training="0"
            index=$((index + 1))
            ;;
        --training-model-profile|--training-sft-profile|--training-experiment-id|--training-resume-checkpoint)
            if ((index + 1 >= ${#args[@]})); then
                echo "${argument} requires a value." >&2
                exit 2
            fi
            value="${args[index + 1]}"
            case "${argument}" in
                --training-model-profile) training_model_profile="${value}" ;;
                --training-sft-profile) training_sft_profile="${value}" ;;
                --training-experiment-id) training_experiment_id="${value}" ;;
                --training-resume-checkpoint) training_resume_checkpoint="${value}" ;;
            esac
            index=$((index + 2))
            ;;
        --training-model-profile=*|--training-sft-profile=*|--training-experiment-id=*|--training-resume-checkpoint=*)
            value="${argument#*=}"
            case "${argument%%=*}" in
                --training-model-profile) training_model_profile="${value}" ;;
                --training-sft-profile) training_sft_profile="${value}" ;;
                --training-experiment-id) training_experiment_id="${value}" ;;
                --training-resume-checkpoint) training_resume_checkpoint="${value}" ;;
            esac
            index=$((index + 1))
            ;;
        --output-dir)
            if ((index + 1 >= ${#args[@]})); then
                echo "--output-dir requires a value." >&2
                exit 2
            fi
            output_dir="${args[index + 1]}"
            autopilot_args+=("--output-dir" "${output_dir}")
            index=$((index + 2))
            ;;
        --output-dir=*)
            output_dir="${argument#--output-dir=}"
            autopilot_args+=("--output-dir=${output_dir}")
            index=$((index + 1))
            ;;
        --limit)
            if ((index + 1 >= ${#args[@]})); then
                echo "--limit requires a value." >&2
                exit 2
            fi
            autopilot_has_limit="1"
            autopilot_args+=("${argument}" "${args[index + 1]}")
            index=$((index + 2))
            ;;
        --limit=*)
            autopilot_has_limit="1"
            autopilot_args+=("${argument}")
            index=$((index + 1))
            ;;
        *)
            autopilot_args+=("${argument}")
            index=$((index + 1))
            ;;
    esac
done

if [[ ! "${smoke_limit}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--smoke-limit must be a positive integer." >&2
    exit 2
fi
if [[ "${smoke_then_full}" == "1" && "${autopilot_has_limit}" == "1" ]]; then
    echo "--smoke-then-full cannot be combined with --limit." >&2
    exit 2
fi
if [[ "${smoke_then_full}" == "1" && "${run_training}" == "1" ]]; then
    echo "--smoke-then-full does not run GPU training; train only after final delivery." >&2
    exit 2
fi
if [[ -z "${output_dir}" ]]; then
    echo "--output-dir is required so the pipeline can be audited." >&2
    exit 2
fi
output_dir="$(realpath -m -- "${output_dir}")"
for ((index = 0; index < ${#autopilot_args[@]}; index++)); do
    case "${autopilot_args[index]}" in
        --output-dir)
            autopilot_args[index + 1]="${output_dir}"
            ;;
        --output-dir=*)
            autopilot_args[index]="--output-dir=${output_dir}"
            ;;
    esac
done

if [[ "${smoke_then_full}" == "1" ]]; then
    smoke_output_dir="${output_dir}-smoke${smoke_limit}"
    smoke_args=()
    index=0
    while ((index < ${#autopilot_args[@]})); do
        case "${autopilot_args[index]}" in
            --output-dir)
                smoke_args+=("--output-dir" "${smoke_output_dir}")
                index=$((index + 2))
                ;;
            --output-dir=*)
                smoke_args+=("--output-dir=${smoke_output_dir}")
                index=$((index + 1))
                ;;
            *)
                smoke_args+=("${autopilot_args[index]}")
                index=$((index + 1))
                ;;
        esac
    done
    smoke_args+=("--limit" "${smoke_limit}")
    smoke_args+=("--validation-count" "${IFV_SMOKE_VALIDATION_COUNT:-1}")
    printf 'phase=smoke output_dir=%s\n' "${smoke_output_dir}"
    IFV_REQUIRE_FULL_TEACHER_DELIVERY=0 \
        "${SCRIPT_DIR}/run_teacher_sft_pipeline.sh" "${smoke_args[@]}"
    printf 'phase=full output_dir=%s\n' "${output_dir}"
    IFV_REQUIRE_FULL_TEACHER_DELIVERY=1 \
        IFV_EXPECTED_TEACHER_CASE_COUNT="${IFV_EXPECTED_TEACHER_CASE_COUNT:-8490}" \
        "${SCRIPT_DIR}/run_teacher_sft_pipeline.sh" "${autopilot_args[@]}"
    exit 0
fi

export IFV_SERVER_RUNNER="${IFV_SERVER_RUNNER:-${SCRIPT_DIR}/run_ifv.sh}"
export PYTHONPATH="${REPO_ROOT}/training${PYTHONPATH:+:${PYTHONPATH}}"
policy="${IFV_SOURCE_ACCESS_POLICY:-${REPO_ROOT}/configs/source-access-policy-web-refuted-v3-expanded-20260821.json}"

"${IFV_SERVER_RUNNER}" python "${REPO_ROOT}/scripts/trajectory/run_teacher_rollout_autopilot.py" "${autopilot_args[@]}"

audit_root="${output_dir}/audits/final-traces"
mkdir -p "${audit_root}"
while IFS= read -r trace_dir; do
    name="$(basename "$(dirname "$(dirname "${trace_dir}")")")"
    "${IFV_SERVER_RUNNER}" python "${REPO_ROOT}/scripts/audit_real_trace.py" \
        "${trace_dir}" --json --strict-scheduler --source-access-policy "${policy}" \
        > "${audit_root}/${name}.json"
done < <(find "${output_dir}/rollouts" -type d -path '*/merged/traces' | sort)

package_dir="${output_dir}/sft-training-package"
if [[ ! -d "${package_dir}/ms-swift-policy" || ! -d "${package_dir}/ms-swift-perception" ]]; then
    echo "autopilot did not produce both SFT datasets: ${package_dir}" >&2
    exit 1
fi

training_python="${IFV_TRAINING_PYTHON:-$(command -v python)}"
if [[ ! -x "${training_python}" ]]; then
    echo "training Python does not exist: ${training_python}" >&2
    exit 2
fi
if [[ "${run_training}" == "1" ]]; then
    for training_profile in "${training_model_profile}" "${training_sft_profile}"; do
        if [[ -z "${training_profile}" || ! -s "${training_profile}" ]]; then
            echo "training profile does not exist or is empty: ${training_profile:-unset}" >&2
            exit 2
        fi
        set -a
        # shellcheck disable=SC1090
        source "${training_profile}"
        set +a
    done
fi
"${training_python}" -m ifv_training audit --strict --input "${package_dir}/ms-swift-policy"
"${training_python}" -m ifv_training audit --strict --input "${package_dir}/ms-swift-perception"

processor_status="not_run"
processor_report="${package_dir}/audits/processor-verification.json"
if [[ -n "${IFV_MODEL_ID:-}" && "${IFV_SKIP_PROCESSOR:-0}" != "1" ]] \
    && "${training_python}" -c 'import swift' >/dev/null 2>&1; then
    processor_args=(
        "${training_python}"
        "${REPO_ROOT}/training/scripts/probe/verify_ms_swift_agent_dataset.py"
        --model "${IFV_MODEL_ID}"
        --policy-dir "${package_dir}/ms-swift-policy"
        --perception-dir "${package_dir}/ms-swift-perception"
        --output "${processor_report}"
        --max-context "${IFV_MAX_LENGTH:-${IFV_PROCESSOR_MAX_CONTEXT:-131072}}"
        --truncation-strategy "${IFV_TRUNCATION_STRATEGY:-raise}"
        --padding-free "${IFV_PADDING_FREE:-false}"
        --sequence-parallel-size "${IFV_SEQUENCE_PARALLEL_SIZE:-1}"
        --loss-scale "${IFV_LOSS_SCALE:-ignore_empty_think}"
        --enable-thinking "${IFV_ENABLE_THINKING:-false}"
        --add-non-thinking-prefix "${IFV_ADD_NON_THINKING_PREFIX:-false}"
        --image-max-token-num "${IFV_IMAGE_MAX_TOKEN_NUM:-1024}"
    )
    if [[ -n "${IFV_MAX_PIXELS:-}" ]]; then
        processor_args+=(--max-pixels "${IFV_MAX_PIXELS}")
    fi
    "${processor_args[@]}"
    processor_status="passed"
elif [[ "${IFV_REQUIRE_PROCESSOR:-0}" == "1" ]]; then
    echo "processor verification requested but IFV_MODEL_ID or swift is unavailable." >&2
    exit 2
else
    processor_status="skipped_missing_model_or_swift"
fi

training_status="not_requested"
training_returncode=""
training_command_json="null"
if [[ "${run_training}" == "1" ]]; then
    training_script="${IFV_TRAINING_SCRIPT:-${REPO_ROOT}/training/scripts/train/run_sft.sh}"
    if [[ -z "${training_model_profile}" || -z "${training_sft_profile}" || -z "${training_experiment_id}" ]]; then
        echo "--run-training requires training model profile, SFT profile, and experiment id." >&2
        exit 2
    fi
    if [[ ! -x "${training_script}" ]]; then
        echo "training script does not exist or is not executable: ${training_script}" >&2
        exit 2
    fi
    if [[ ! -s "${package_dir}/ms-swift-policy/train.jsonl" || ! -s "${package_dir}/ms-swift-policy/validation.jsonl" ]]; then
        echo "training requires non-empty policy train and validation datasets." >&2
        exit 2
    fi
    if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        echo "--run-training requires CUDA_VISIBLE_DEVICES." >&2
        exit 2
    fi
    training_args=(
        "${training_model_profile}"
        "${training_sft_profile}"
        "${package_dir}/ms-swift-policy/train.jsonl"
        "${package_dir}/ms-swift-policy/validation.jsonl"
        "${training_experiment_id}"
    )
    if [[ -n "${training_resume_checkpoint}" ]]; then
        training_args+=("${training_resume_checkpoint}")
    fi
    training_status="running"
    training_command_json="$(${training_python} - "${training_script}" "${training_args[@]}" <<'PY'
import json
import sys
print(json.dumps([sys.argv[1], *sys.argv[2:]], ensure_ascii=False))
PY
)"
    set +e
    PATH="$(dirname "${training_python}"):${PATH}" \
        OMP_NUM_THREADS=1 IFV_MODEL_ID="${IFV_MODEL_ID}" \
        IFV_PROCESSOR_VERIFICATION="${processor_report}" \
        bash "${training_script}" "${training_args[@]}"
    training_returncode="$?"
    set -e
    if [[ "${training_returncode}" == "0" ]]; then
        training_status="completed"
    else
        training_status="failed"
        exit "${training_returncode}"
    fi
fi

"${IFV_SERVER_RUNNER}" python - "${output_dir}" "${processor_status}" "${training_status}" "${training_returncode}" "${training_command_json}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
processor_status = sys.argv[2]
training_status = sys.argv[3]
training_returncode = sys.argv[4]
training_command = json.loads(sys.argv[5])
rows = []
for path in sorted((root / "audits" / "final-traces").glob("*.json")):
    value = json.loads(path.read_text(encoding="utf-8"))
    rows.append(
        {
            "file": str(path),
            "trace_count": value.get("trace_count", 0),
            "passed_count": value.get("passed_count", 0),
            "failed_count": value.get("failed_count", 0),
            "warning_count": value.get("warning_count", 0),
        }
    )
summary = {
    "schema_version": "ifv-teacher-sft-pipeline-audit-v1",
    "pipeline_dir": str(root),
    "trace_audits": rows,
    "processor_status": processor_status,
    "training_status": training_status,
    "training_returncode": int(training_returncode) if training_returncode else None,
    "training_command": training_command,
    "all_trace_audits_passed": all(row["failed_count"] == 0 for row in rows),
}
(root / "audits" / "pipeline-summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
if not summary["all_trace_audits_passed"]:
    raise SystemExit(1)
PY

if [[ "${IFV_REQUIRE_FULL_TEACHER_DELIVERY:-0}" == "1" ]]; then
    "${IFV_SERVER_RUNNER}" python \
        "${REPO_ROOT}/scripts/trajectory/verify_teacher_sft_delivery.py" \
        --pipeline-dir "${output_dir}" \
        --expected-case-count "${IFV_EXPECTED_TEACHER_CASE_COUNT:-8490}" \
        --output "${output_dir}/audits/final-delivery.json"
fi
