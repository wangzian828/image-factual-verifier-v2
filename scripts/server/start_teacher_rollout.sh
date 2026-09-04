#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

for arg in "$@"; do
    case "${arg}" in
        --profile|--profile=*|--provider|--provider=*|--model|--model=*|\
        --vlm-provider|--vlm-provider=*|--vlm-model|--vlm-model=*|\
        --llm-wire-api|--llm-wire-api=*|--vlm-wire-api|--vlm-wire-api=*)
            echo "start_teacher_rollout.sh fixes profile=teacher-gemini." >&2
            exit 2
            ;;
    esac
done

export IFV_EVAL_MODULE=src.eval.run_cases
export IFV_VISION_TOOL_IMAGE_MODE="${IFV_VISION_TOOL_IMAGE_MODE:-compressed}"
export GEMINI_VISION_TIMEOUT_SECONDS="${GEMINI_VISION_TIMEOUT_SECONDS:-150}"
export VLM_TOOL_REQUEST_TIMEOUT_SECONDS="${VLM_TOOL_REQUEST_TIMEOUT_SECONDS:-150}"
export VLM_TOOL_REQUEST_MAX_RETRIES="${VLM_TOOL_REQUEST_MAX_RETRIES:-0}"
export AGENT_TOOL_ACTION_TIMEOUT_SECONDS="${AGENT_TOOL_ACTION_TIMEOUT_SECONDS:-210}"
export BAIDU_OCR_MAX_RETRIES="${BAIDU_OCR_MAX_RETRIES:-1}"
export BAIDU_OCR_RETRY_BACKOFF_SECONDS="${BAIDU_OCR_RETRY_BACKOFF_SECONDS:-1}"
export AGENT_LLM_REQUEST_TIMEOUT_SECONDS="${AGENT_LLM_REQUEST_TIMEOUT_SECONDS:-90}"
export AGENT_LLM_REQUEST_MAX_RETRIES="${AGENT_LLM_REQUEST_MAX_RETRIES:-1}"
export AGENT_STAGE_REQUEST_TIMEOUT_SECONDS="${AGENT_STAGE_REQUEST_TIMEOUT_SECONDS:-900}"
export GEMINI_RETRY_BASE_DELAY_SECONDS="${GEMINI_RETRY_BASE_DELAY_SECONDS:-2}"
export GEMINI_RETRY_JITTER_SECONDS="${GEMINI_RETRY_JITTER_SECONDS:-1}"
export GEMINI_RETRY_MAX_DELAY_SECONDS="${GEMINI_RETRY_MAX_DELAY_SECONDS:-10}"

exec "${SCRIPT_DIR}/start_eval.sh" --profile teacher-gemini "$@"
