#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="${IFV_QWEN_REPO:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
ENV_PREFIX="${IFV_VLLM_ENV_PREFIX:-${CONDA_PREFIX:-}}"
PYTHON="${IFV_QWEN_PYTHON:-${ENV_PREFIX:+${ENV_PREFIX}/bin/python}}"
VLLM="${IFV_QWEN_VLLM:-${ENV_PREFIX:+${ENV_PREFIX}/bin/vllm}}"
MODEL="${IFV_QWEN_MODEL:-${IFV_MODEL_ID:-}}"
RUN_ROOT="${IFV_QWEN_RUN_ROOT:-${IFV_DATA_ROOT:+${IFV_DATA_ROOT}/runs}}"
LOG_ROOT="$RUN_ROOT/_logs"
PID_ROOT="$RUN_ROOT/queues"

for name in REPO PYTHON VLLM MODEL RUN_ROOT; do
  if [[ -z "${!name:-}" ]]; then
    echo "required value is empty: ${name}" >&2
    exit 2
  fi
done

BASE_ARGS=(
  serve "$MODEL"
  --served-model-name ifv-qwen3.5-9b
  --dtype bfloat16
  --tensor-parallel-size 2
  --disable-custom-all-reduce
  --enforce-eager
  --max-model-len 131072
  --gpu-memory-utilization 0.92
  --max-num-seqs 2
  --max-num-batched-tokens 8192
  --reasoning-parser qwen3
  --structured-outputs-config '{"backend":"xgrammar","reasoning_parser":"qwen3","disable_any_whitespace":true}'
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
  --default-chat-template-kwargs '{"enable_thinking":false}'
  --limit-mm-per-prompt '{"image":1,"video":0}'
  --enable-tokenizer-info-endpoint
  --max-log-len 4000
  --disable-uvicorn-access-log
)

gateway_pid="$PID_ROOT/qwen35-replica-gateway.pid"
replica_a_pid="$PID_ROOT/qwen35-replica-a.pid"
replica_b_pid="$PID_ROOT/qwen35-replica-b.pid"

pid_from_file() {
  local path="$1"
  [[ -s "$path" ]] || return 1
  local pid
  pid="$(tr -d '[:space:]' < "$path")"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  printf '%s\n' "$pid"
}

stop_pid_file() {
  local path="$1"
  local pattern="$2"
  local pid pgid command
  pid="$(pid_from_file "$path" || true)"
  [[ -n "$pid" ]] || return 0
  command="$(ps -o args= -p "$pid" || true)"
  [[ "$command" == *"$pattern"* ]] || {
    echo "refusing to stop unrecognized process $pid: $command" >&2
    return 1
  }
  pgid="$(ps -o pgid= -p "$pid" | tr -d ' ')"
  kill -TERM -- "-$pgid" 2>/dev/null || true
  sleep 2
  if kill -0 "$pid" 2>/dev/null; then
    kill -KILL -- "-$pgid" 2>/dev/null || true
  fi
  rm -f "$path"
}

status() {
  for path in "$replica_a_pid" "$replica_b_pid" "$gateway_pid"; do
    if pid="$(pid_from_file "$path" || true)" && [[ -n "$pid" ]]; then
      ps -p "$pid" -o pid=,ppid=,pgid=,etime=,args=
    else
      echo "stopped: $path"
    fi
  done
  curl -fsS --max-time 5 http://127.0.0.1:8901/health || true
  echo
}

start() {
  mkdir -p "$LOG_ROOT" "$PID_ROOT"
  [[ -x "$PYTHON" ]] || { echo "missing Python: $PYTHON" >&2; return 2; }
  [[ -x "$VLLM" ]] || { echo "missing vLLM: $VLLM" >&2; return 2; }
  [[ -d "$REPO" ]] || { echo "missing repo: $REPO" >&2; return 2; }
  if pid_from_file "$replica_a_pid" >/dev/null ||
     pid_from_file "$replica_b_pid" >/dev/null ||
     pid_from_file "$gateway_pid" >/dev/null; then
    echo "Qwen replica process already exists; use status or stop first" >&2
    return 2
  fi

  local log_a="$LOG_ROOT/qwen35-replica-a-$(date -u +%Y%m%d).log"
  local log_b="$LOG_ROOT/qwen35-replica-b-$(date -u +%Y%m%d).log"
  local log_gateway="$LOG_ROOT/qwen35-replica-gateway-$(date -u +%Y%m%d).log"

  setsid env NCCL_CUMEM_ENABLE=0 NCCL_CUMEM_HOST_ENABLE=0 \
    VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_VISIBLE_DEVICES=4,5 \
    "$VLLM" "${BASE_ARGS[@]}" --host 127.0.0.1 --port 8902 \
    </dev/null >"$log_a" 2>&1 &
  echo "$!" >"$replica_a_pid"

  setsid env NCCL_CUMEM_ENABLE=0 NCCL_CUMEM_HOST_ENABLE=0 \
    VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_VISIBLE_DEVICES=6,7 \
    "$VLLM" "${BASE_ARGS[@]}" --host 127.0.0.1 --port 8903 \
    </dev/null >"$log_b" 2>&1 &
  echo "$!" >"$replica_b_pid"

  setsid env \
    QWEN_REPLICA_BACKENDS=http://127.0.0.1:8902,http://127.0.0.1:8903 \
    QWEN_REPLICA_MODEL_ID=ifv-qwen3.5-9b \
    QWEN_REPLICA_GATEWAY_TIMEOUT_SECONDS=900 \
    "$PYTHON" -m uvicorn scripts.server.qwen_replica_gateway:app \
    --app-dir "$REPO" --host 127.0.0.1 --port 8901 --log-level warning \
    </dev/null >"$log_gateway" 2>&1 &
  echo "$!" >"$gateway_pid"
  echo "started Qwen replicas on 8902/8903 with gateway on 8901"
}

stop() {
  stop_pid_file "$gateway_pid" "qwen_replica_gateway:app"
  stop_pid_file "$replica_a_pid" "vllm serve"
  stop_pid_file "$replica_b_pid" "vllm serve"
  echo "stopped Qwen replicas and gateway"
}

case "${1:-status}" in
  start) start ;;
  stop) stop ;;
  status) status ;;
  *)
    echo "usage: $0 {start|stop|status}" >&2
    exit 2
    ;;
esac
