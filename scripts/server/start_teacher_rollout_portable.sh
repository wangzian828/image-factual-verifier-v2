#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=ifv_env.sh
source "${SCRIPT_DIR}/ifv_env.sh"

usage() {
    cat >&2 <<'EOF'
usage: start_teacher_rollout_portable.sh [--full | --smoke-only | --limit N] [OPTIONS]

The default runs a 10-case smoke and automatically starts all 8,490 training
cases only after the smoke pipeline succeeds. --full skips the smoke and starts
the full set directly. The command prepares the ModelScope archive, preflights
the teacher/judge endpoints, and runs rollout, strict trace audit, SFT
eligibility, quality rerolls, accepted release, and SFT package export.

Options:
  --dataset-root DIR  Override the extracted training dataset.
  --output-dir DIR    Override the final full pipeline output directory.
  --foreground        Run in the foreground.
  --background        Run in the background (default).
  --full              Skip smoke and run all training cases directly.
  --smoke-only        Run only the default 10-case smoke.
  --limit N           Run only a deterministic bounded smoke subset.
EOF
}

dataset_root="${IFV_TEACHER_DATASET_ROOT:-${IFV_DATA_ROOT}/datasets/factcheck_train-8490-20260907}"
output_dir=""
run_mode="smoke_then_full"
smoke_limit="${IFV_SMOKE_CASE_COUNT:-10}"
quality_reroll_rounds="3"
limit=""
background="1"
forward=()
while (($#)); do
    case "$1" in
        --dataset-root)
            (($# >= 2)) || { usage; exit 2; }
            dataset_root="$2"
            shift 2
            ;;
        --dataset-root=*)
            dataset_root="${1#*=}"
            shift
            ;;
        --output-dir)
            (($# >= 2)) || { usage; exit 2; }
            output_dir="$2"
            shift 2
            ;;
        --output-dir=*)
            output_dir="${1#*=}"
            shift
            ;;
        --full)
            run_mode="full"
            limit=""
            shift
            ;;
        --smoke-only)
            run_mode="smoke"
            limit="${smoke_limit}"
            shift
            ;;
        --limit)
            (($# >= 2)) || { usage; exit 2; }
            run_mode="smoke"
            limit="$2"
            shift 2
            ;;
        --limit=*)
            run_mode="smoke"
            limit="${1#*=}"
            shift
            ;;
        --foreground)
            background="0"
            shift
            ;;
        --background)
            background="1"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            forward+=("$1")
            shift
            ;;
    esac
done

if [[ ! "${smoke_limit}" =~ ^[1-9][0-9]*$ ]]; then
    echo "IFV_SMOKE_CASE_COUNT must be a positive integer." >&2
    exit 2
fi
if [[ "${run_mode}" == "smoke" && ! "${limit}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--limit must be a positive integer." >&2
    exit 2
fi
dataset_root="$(realpath -m -- "${dataset_root}")"

rollout_profile="${IFV_TEACHER_ROLLOUT_PROFILE:-teacher-qwen-server}"
rollout_model="${IFV_TEACHER_ROLLOUT_MODEL:-${QWEN_TEACHER_MODEL:-}}"
judge_provider="${IFV_SFT_ELIGIBILITY_PROVIDER:-qwen_local}"
judge_model="${IFV_SFT_ELIGIBILITY_MODEL:-${rollout_model}}"
judge_base_url="${IFV_SFT_ELIGIBILITY_BASE_URL:-${QWEN_TEACHER_BASE_URL:-}}"
judge_wire_api="${IFV_SFT_ELIGIBILITY_WIRE_API:-chat_completions}"
judge_key_env="${IFV_SFT_ELIGIBILITY_API_KEY_ENV:-}"

if [[ "${rollout_profile}" == "teacher-qwen-server" ]]; then
    : "${QWEN_TEACHER_BASE_URL:?QWEN_TEACHER_BASE_URL is required}"
    if [[ -z "${rollout_model}" ]]; then
        echo "QWEN_TEACHER_MODEL or IFV_TEACHER_ROLLOUT_MODEL is required." >&2
        exit 2
    fi
    configured_vision_model="${QWEN_TEACHER_VISION_MODEL:-${rollout_model}}"
    if [[ "${configured_vision_model}" != "${rollout_model}" ]]; then
        echo "QWEN_TEACHER_VISION_MODEL must match the main teacher model." >&2
        exit 2
    fi
    export QWEN_TEACHER_MODEL="${rollout_model}"
    export QWEN_TEACHER_VISION_MODEL="${rollout_model}"
    export QWEN_LOCAL_API_KEY="${QWEN_TEACHER_API_KEY:-none}"
    export BROWSE_EXTRACT_PROVIDER="qwen_local"
    export BROWSE_EXTRACT_MODEL="${rollout_model}"
    export BROWSE_EXTRACT_BASE_URL="${QWEN_TEACHER_BASE_URL}"
    export BROWSE_EXTRACT_API_KEY="${QWEN_TEACHER_API_KEY:-none}"
    export BROWSE_EXTRACT_WIRE_API="chat_completions"
fi
if [[ -z "${rollout_model}" ]]; then
    echo "IFV_TEACHER_ROLLOUT_MODEL or the selected profile model is required." >&2
    exit 2
fi
if [[ -z "${judge_model}" ]]; then
    echo "IFV_SFT_ELIGIBILITY_MODEL is required." >&2
    exit 2
fi
if [[ -z "${judge_key_env}" && -n "${IFV_SFT_ELIGIBILITY_API_KEY:-}" ]]; then
    judge_key_env="IFV_SFT_ELIGIBILITY_API_KEY"
fi

require_agent_env() {
    local name="$1"
    local value
    value="$(printenv "${name}" 2>/dev/null || true)"
    if [[ -z "${value}" ]]; then
        echo "${name} is required for the Agent toolchain." >&2
        exit 2
    fi
}

validate_agent_tool_configuration() {
    if [[ "${OCR_BACKEND:-baidu}" != "baidu" ]]; then
        echo "OCR_BACKEND=baidu is required; local OCR and OCR fallback are prohibited." >&2
        exit 2
    fi
    if [[ -z "${BAIDU_OCR_ACCESS_TOKEN:-}" ]]; then
        require_agent_env BAIDU_OCR_API_KEY
        require_agent_env BAIDU_OCR_SECRET_KEY
    fi

    if [[ -z "${SERPER_API_KEY:-${SERPER_KEY_ID:-}}" ]]; then
        echo "SERPER_API_KEY is required for text, image, and Lens search." >&2
        exit 2
    fi
    if [[ "${BROWSE_FETCH_PROVIDER:-jina}" != "jina" ]]; then
        echo "BROWSE_FETCH_PROVIDER=jina is required for this handoff." >&2
        exit 2
    fi
    if [[ -z "${JINA_API_KEY:-${JINA_API_KEYS:-}}" ]]; then
        echo "JINA_API_KEY is required for page fetch and search reranking." >&2
        exit 2
    fi
    if [[ "${VISUAL_SEARCH_PROVIDER:-serper_lens}" != "serper_lens" ]]; then
        echo "VISUAL_SEARCH_PROVIDER=serper_lens is required for this handoff." >&2
        exit 2
    fi

    case "${IMAGE_UPLOAD_PROVIDER:-}" in
        oss)
            require_agent_env OSS_ACCESS_KEY_ID
            require_agent_env OSS_ACCESS_KEY_SECRET
            require_agent_env OSS_ENDPOINT
            require_agent_env OSS_BUCKET_NAME
            ;;
        custom)
            require_agent_env IMAGE_UPLOAD_API_URL
            ;;
        temp)
            ;;
        *)
            echo "IMAGE_UPLOAD_PROVIDER must be explicitly set to oss, custom, or temp." >&2
            exit 2
            ;;
    esac
}

validate_agent_tool_configuration

preflight_endpoint() {
    local label="$1"
    local provider="$2"
    local base_url="$3"
    local model="$4"
    local api_key="${5:-}"
    if [[ "${provider}" != "qwen_local" && "${provider}" != "lmdeploy" ]]; then
        return
    fi
    if [[ -z "${base_url}" ]]; then
        echo "${label} base URL is required for provider=${provider}." >&2
        exit 2
    fi
    python - "${label}" "${base_url}" "${model}" "${api_key}" <<'PY'
import json
import sys
import urllib.request

label, base_url, model, api_key = sys.argv[1:]
request = urllib.request.Request(base_url.rstrip("/") + "/models")
if api_key and api_key != "none":
    request.add_header("Authorization", f"Bearer {api_key}")
try:
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = json.load(response)
except Exception as exc:
    raise SystemExit(f"{label} endpoint preflight failed: {type(exc).__name__}: {exc}")
models = {
    str(item.get("id") or "")
    for item in payload.get("data", [])
    if isinstance(item, dict)
}
if models and model not in models:
    raise SystemExit(
        f"{label} model {model!r} is absent from endpoint /models: "
        + ", ".join(sorted(models)[:10])
    )
print(f"{label}_endpoint=ok model={model}")
PY
}

if [[ "${IFV_SKIP_MODEL_ENDPOINT_PREFLIGHT:-0}" != "1" ]]; then
    preflight_endpoint \
        teacher \
        "$([[ "${rollout_profile}" == "teacher-qwen-server" ]] && printf qwen_local || printf '%s' "${rollout_profile}")" \
        "${QWEN_TEACHER_BASE_URL:-}" \
        "${rollout_model}" \
        "${QWEN_LOCAL_API_KEY:-}"
    judge_key=""
    if [[ -n "${judge_key_env}" ]]; then
        judge_key="${!judge_key_env-}"
    elif [[ "${judge_provider}" == "qwen_local" ]]; then
        judge_key="${QWEN_LOCAL_API_KEY:-}"
    fi
    preflight_endpoint \
        judge \
        "${judge_provider}" \
        "${judge_base_url}" \
        "${judge_model}" \
        "${judge_key}"
fi

if [[ ! -f "${dataset_root}/train-manifest.jsonl" ]] \
    || [[ ! -d "${dataset_root}/images" ]] \
    || [[ ! -f "${dataset_root}/evaluator_private/private-gold-v1/train-private-gold.jsonl" ]]; then
    "${SCRIPT_DIR}/prepare_factcheck_dataset.sh" \
        --split train \
        --output-dir "${dataset_root}"
fi

case "${run_mode}" in
    smoke_then_full)
        mode="full"
        export IFV_REQUIRE_FULL_TEACHER_DELIVERY=0
        ;;
    full)
        mode="full"
        export IFV_REQUIRE_FULL_TEACHER_DELIVERY=1
        export IFV_EXPECTED_TEACHER_CASE_COUNT="${IFV_EXPECTED_TEACHER_CASE_COUNT:-8490}"
        ;;
    smoke)
        mode="smoke${limit}"
        export IFV_REQUIRE_FULL_TEACHER_DELIVERY=0
        ;;
esac
if [[ -z "${output_dir}" ]]; then
    run_id="${IFV_TEACHER_ROLLOUT_RUN_ID:-teacher-${rollout_profile}-${mode}-$(date -u +%Y%m%dT%H%M%SZ)}"
    output_dir="${IFV_DATA_ROOT}/generated/teacher-rollouts/${run_id}"
fi
output_dir="$(realpath -m -- "${output_dir}")"

pipeline_args=(
    --dataset-root "${dataset_root}"
    --output-dir "${output_dir}"
    --rollout-profile "${rollout_profile}"
    --rollout-model "${rollout_model}"
    --sft-judge-provider "${judge_provider}"
    --sft-model "${judge_model}"
    --sft-judge-wire-api "${judge_wire_api}"
    --quality-reroll-rounds "${quality_reroll_rounds}"
)
if [[ -n "${judge_base_url}" ]]; then
    pipeline_args+=(--sft-judge-base-url "${judge_base_url}")
fi
if [[ -n "${judge_key_env}" ]]; then
    pipeline_args+=(--sft-judge-api-key-env "${judge_key_env}")
fi
if [[ "${IFV_SFT_ELIGIBILITY_ENABLE_THINKING:-true}" == "true" ]]; then
    pipeline_args+=(--sft-judge-enable-thinking)
else
    pipeline_args+=(--no-sft-judge-enable-thinking)
fi
if [[ "${run_mode}" == "smoke" ]]; then
    pipeline_args+=(--limit "${limit}")
    pipeline_args+=(--validation-count "${IFV_SMOKE_VALIDATION_COUNT:-1}")
elif [[ "${run_mode}" == "smoke_then_full" ]]; then
    pipeline_args+=(--smoke-then-full)
    pipeline_args+=(--smoke-limit "${smoke_limit}")
fi
pipeline_args+=("${forward[@]}")

command=("${SCRIPT_DIR}/run_teacher_sft_pipeline.sh" "${pipeline_args[@]}")
delivery_scope="full_train_set"
smoke_output_dir=""
if [[ "${run_mode}" == "smoke" ]]; then
    delivery_scope="smoke_not_final"
elif [[ "${run_mode}" == "smoke_then_full" ]]; then
    delivery_scope="smoke_then_full"
    smoke_output_dir="${output_dir}-smoke${smoke_limit}"
fi
if [[ "${background}" == "0" ]]; then
    printf 'delivery_scope=%s\n' "${delivery_scope}"
    if [[ -n "${smoke_output_dir}" ]]; then
        printf 'smoke_output_dir=%s\n' "${smoke_output_dir}"
    fi
    exec "${command[@]}"
fi

log_root="${IFV_DATA_ROOT}/runs/_logs"
pid_root="${IFV_DATA_ROOT}/runs/_pids"
mkdir -p -- "${log_root}" "${pid_root}" "${output_dir}/logs"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
log_file="${log_root}/teacher-portable-${timestamp}.log"
pid_file="${pid_root}/teacher-portable-${timestamp}.pid"
nohup "${command[@]}" </dev/null >"${log_file}" 2>&1 &
pid=$!
printf '%s\n' "${pid}" >"${pid_file}"
printf 'pid=%s\noutput_dir=%s\nlog_file=%s\npid_file=%s\ndelivery_scope=%s\n' \
    "${pid}" "${output_dir}" "${log_file}" "${pid_file}" \
    "${delivery_scope}"
if [[ -n "${smoke_output_dir}" ]]; then
    printf 'smoke_output_dir=%s\n' "${smoke_output_dir}"
fi
