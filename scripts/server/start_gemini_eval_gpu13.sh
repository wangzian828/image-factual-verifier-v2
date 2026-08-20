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

# One Gemini run at a time, and at most four complete rollout episodes by
# default.  start_eval_gpu13.sh also installs the process-wide request gate and
# refuses a second active Gemini launch.
export GEMINI_EVAL_MAX_CONCURRENCY="${GEMINI_EVAL_MAX_CONCURRENCY:-4}"

exec "${SCRIPT_DIR}/start_eval_gpu13.sh" \
    --profile teacher-gemini \
    "$@"
