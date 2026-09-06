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
        *)
            autopilot_args+=("${argument}")
            index=$((index + 1))
            ;;
    esac
done

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
"${training_python}" -m ifv_training audit --strict --input "${package_dir}/ms-swift-policy"
"${training_python}" -m ifv_training audit --strict --input "${package_dir}/ms-swift-perception"

processor_status="not_run"
if [[ -n "${IFV_MODEL_ID:-}" && "${IFV_SKIP_PROCESSOR:-0}" != "1" ]] \
    && "${training_python}" -c 'import swift' >/dev/null 2>&1; then
    "${training_python}" "${REPO_ROOT}/training/scripts/probe/verify_ms_swift_agent_dataset.py" \
        --model "${IFV_MODEL_ID}" \
        --policy-dir "${package_dir}/ms-swift-policy" \
        --perception-dir "${package_dir}/ms-swift-perception" \
        --output "${package_dir}/audits/processor-verification.json" \
        --max-context "${IFV_PROCESSOR_MAX_CONTEXT:-131072}"
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
