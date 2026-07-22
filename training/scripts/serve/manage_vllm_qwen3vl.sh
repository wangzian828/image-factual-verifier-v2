#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ACTION="${1:-status}"
MODEL="${IFV_QWEN3VL_MODEL:-/gsdata/home/wza/models/Qwen3-VL-8B-Thinking}"
SERVED_NAME="${IFV_VLLM_SERVED_NAME:-ifv-qwen3-vl-8b-thinking-vllm}"
PORT="${IFV_VLLM_PORT:-8901}"
TP_SIZE="${IFV_VLLM_TP_SIZE:-2}"
CONTEXT_LENGTH="${IFV_VLLM_CONTEXT_LENGTH:-131072}"
ENV_PREFIX="${IFV_VLLM_ENV_PREFIX:-/gsdata/home/wza/conda/envs/ifv-qwen3vl-vllm0112-locked}"
STATE_ROOT="${IFV_TRAINING_DATA_ROOT:-/gsdata/home/wza/image-factual-verifier-v2-data/training}/logs/serving/$SERVED_NAME"
PID_FILE="$STATE_ROOT/server.pid"
LOG_FILE="$STATE_ROOT/server.log"

mkdir -p "$STATE_ROOT"

read_pid() {
  [[ -s "$PID_FILE" ]] || return 1
  local pid
  pid="$(tr -d '[:space:]' <"$PID_FILE")"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  printf '%s\n' "$pid"
}

owned_process() {
  local pid="$1"
  [[ -r "/proc/$pid/cmdline" ]] || return 1
  local command_line
  command_line="$(tr '\0' ' ' <"/proc/$pid/cmdline")"
  [[ "$command_line" == *"$ENV_PREFIX"* ]] || return 1
  [[ "$command_line" == *"vllm"* ]] || return 1
  [[ "$command_line" == *"$MODEL"* ]] || return 1
  [[ "$command_line" == *"--port $PORT"* ]] || return 1
}

status() {
  local pid
  if pid="$(read_pid)" && kill -0 "$pid" 2>/dev/null && owned_process "$pid"; then
    echo "running pid=$pid url=http://127.0.0.1:$PORT/v1 log=$LOG_FILE"
    return 0
  fi
  echo "stopped"
  return 1
}

case "$ACTION" in
  start)
    if status >/dev/null 2>&1; then
      status
      exit 0
    fi
    rm -f "$PID_FILE"
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"
    : >"$LOG_FILE"
    setsid bash "$SCRIPT_DIR/start_vllm_qwen3vl.sh" \
      "$MODEL" "$SERVED_NAME" "$PORT" "$TP_SIZE" "$CONTEXT_LENGTH" \
      >>"$LOG_FILE" 2>&1 &
    pid=$!
    printf '%s\n' "$pid" >"$PID_FILE"
    for _ in $(seq 1 120); do
      if ! kill -0 "$pid" 2>/dev/null; then
        echo "vLLM exited during startup; inspect $LOG_FILE" >&2
        tail -n 80 "$LOG_FILE" >&2 || true
        rm -f "$PID_FILE"
        exit 1
      fi
      if curl --noproxy '*' -fsS "http://127.0.0.1:$PORT/health" >/dev/null; then
        status
        exit 0
      fi
      sleep 5
    done
    echo "vLLM did not become healthy within 10 minutes; inspect $LOG_FILE" >&2
    exit 1
    ;;
  status)
    status
    ;;
  stop)
    if ! pid="$(read_pid)"; then
      echo "no managed PID file: $PID_FILE" >&2
      exit 1
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      rm -f "$PID_FILE"
      echo "removed stale PID file"
      exit 0
    fi
    if ! owned_process "$pid"; then
      echo "refusing to stop PID $pid because it is not the managed vLLM command" >&2
      exit 2
    fi
    sid="$(ps -o sid= -p "$pid" | tr -d '[:space:]')"
    if [[ "$sid" != "$pid" ]]; then
      echo "refusing to signal unexpected process group: pid=$pid sid=$sid" >&2
      exit 2
    fi
    kill -TERM -- "-$pid"
    for _ in $(seq 1 60); do
      if ! kill -0 "$pid" 2>/dev/null; then
        rm -f "$PID_FILE"
        echo "stopped cleanly"
        exit 0
      fi
      sleep 1
    done
    echo "managed vLLM did not exit after SIGTERM; no SIGKILL was sent" >&2
    exit 1
    ;;
  logs)
    tail -n "${IFV_LOG_LINES:-120}" "$LOG_FILE"
    ;;
  *)
    echo "usage: $0 {start|status|stop|logs}" >&2
    exit 2
    ;;
esac
