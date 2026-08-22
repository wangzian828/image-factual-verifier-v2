#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

for arg in "$@"; do
    case "${arg}" in
        --profile|--profile=*|--provider|--provider=*|--model|--model=*|\
        --vlm-provider|--vlm-provider=*|--vlm-model|--vlm-model=*|\
        --llm-wire-api|--llm-wire-api=*|--vlm-wire-api|--vlm-wire-api=*)
            echo "start_gemini_eval_gpu13.sh fixes profile=teacher-gemini; " \
                "provider/model/wire overrides are not accepted." >&2
            exit 2
            ;;
    esac
done

# Gemini supports concurrent teacher requests. Keep the default rollout cap and
# request gate high enough for explicit multi-worker runs; transport recovery
# must not silently serialize the provider. Every value remains explicitly
# overridable for a controlled canary.
export GEMINI_EVAL_MAX_CONCURRENCY="${GEMINI_EVAL_MAX_CONCURRENCY:-16}"
export GEMINI_MAX_INFLIGHT_REQUESTS="${GEMINI_MAX_INFLIGHT_REQUESTS:-16}"
export IFV_VISION_TOOL_IMAGE_MODE="${IFV_VISION_TOOL_IMAGE_MODE:-original}"
export GEMINI_VISION_TIMEOUT_SECONDS="${GEMINI_VISION_TIMEOUT_SECONDS:-150}"
export VLM_TOOL_REQUEST_TIMEOUT_SECONDS="${VLM_TOOL_REQUEST_TIMEOUT_SECONDS:-150}"
export VLM_TOOL_REQUEST_MAX_RETRIES="${VLM_TOOL_REQUEST_MAX_RETRIES:-0}"
export AGENT_TOOL_ACTION_TIMEOUT_SECONDS="${AGENT_TOOL_ACTION_TIMEOUT_SECONDS:-210}"
export AGENT_LLM_REQUEST_TIMEOUT_SECONDS="${AGENT_LLM_REQUEST_TIMEOUT_SECONDS:-90}"
export AGENT_LLM_REQUEST_MAX_RETRIES="${AGENT_LLM_REQUEST_MAX_RETRIES:-1}"
export AGENT_STAGE_REQUEST_TIMEOUT_SECONDS="${AGENT_STAGE_REQUEST_TIMEOUT_SECONDS:-900}"
export GEMINI_RETRY_BASE_DELAY_SECONDS="${GEMINI_RETRY_BASE_DELAY_SECONDS:-2}"
export GEMINI_RETRY_JITTER_SECONDS="${GEMINI_RETRY_JITTER_SECONDS:-1}"
export GEMINI_RETRY_MAX_DELAY_SECONDS="${GEMINI_RETRY_MAX_DELAY_SECONDS:-10}"

exec "${SCRIPT_DIR}/start_eval_gpu13.sh" \
    --profile teacher-gemini \
    "$@"
