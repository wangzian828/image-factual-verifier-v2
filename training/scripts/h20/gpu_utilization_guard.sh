#!/usr/bin/env bash
set -euo pipefail

ROOT="${IFV_H20_ROOT:-/volume/ybo/wza}"
PYTHON="${IFV_GPU_GUARD_PYTHON:-${ROOT}/envs/h20-qwen35-vllm-0181/bin/python}"
SCRIPT="${IFV_GPU_GUARD_SCRIPT:-${ROOT}/image-factual-verifier-v2/training/scripts/h20/gpu_utilization_guard.py}"
RUN_ROOT="${IFV_GPU_GUARD_RUN_ROOT:-${ROOT}/gpu-utilization-guard}"
STATE_FILE="${RUN_ROOT}/state.json"
LOG_FILE="${RUN_ROOT}/samples.jsonl"
PID_FILE="${RUN_ROOT}/guard.pid"

live_pid() {
  local pid
  [[ -s "$PID_FILE" ]] || return 1
  pid="$(tr -d '[:space:]' <"$PID_FILE")"
  [[ "$pid" =~ ^[1-9][0-9]*$ ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  printf '%s\n' "$pid"
}

start() {
  [[ -x "$PYTHON" ]] || { echo "missing Python: $PYTHON" >&2; exit 2; }
  [[ -f "$SCRIPT" ]] || { echo "missing guard: $SCRIPT" >&2; exit 2; }
  mkdir -p "$RUN_ROOT"
  if live_pid >/dev/null; then
    echo "GPU utilization guard is already running" >&2
    exit 2
  fi
  setsid "$PYTHON" "$SCRIPT" \
    --state-file "$STATE_FILE" \
    --log-file "$LOG_FILE" \
    --poll-seconds "${IFV_GPU_GUARD_POLL_SECONDS:-60}" \
    --utilization-threshold-percent "${IFV_GPU_GUARD_THRESHOLD_PERCENT:-10}" \
    --low-window-seconds "${IFV_GPU_GUARD_LOW_WINDOW_SECONDS:-5400}" \
    --free-memory-limit-mib "${IFV_GPU_GUARD_FREE_MEMORY_LIMIT_MIB:-1024}" \
    --pulse-tokens "${IFV_GPU_GUARD_PULSE_TOKENS:-1024}" \
    </dev/null >>"${RUN_ROOT}/process.log" 2>&1 &
  echo "$!" >"$PID_FILE"
  echo "started GPU utilization guard pid=$!"
}

status() {
  local pid
  pid="$(live_pid || true)"
  if [[ -z "$pid" ]]; then
    echo "GPU utilization guard is stopped"
    exit 1
  fi
  ps -p "$pid" -o pid=,etime=,args=
  [[ -f "$STATE_FILE" ]] && tail -n 30 "$STATE_FILE"
}

stop() {
  local pid command pgid
  pid="$(live_pid || true)"
  if [[ -n "$pid" ]]; then
    command="$(ps -o args= -p "$pid" || true)"
    [[ "$command" == *"gpu_utilization_guard.py"* ]] || {
      echo "refusing to stop unrecognized process $pid: $command" >&2
      exit 1
    }
    pgid="$(ps -o pgid= -p "$pid" | tr -d ' ')"
    kill -TERM -- "-$pgid" 2>/dev/null || true
  fi
  rm -f -- "$PID_FILE"
  echo "stopped GPU utilization guard"
}

case "${1:-status}" in
  start) start ;;
  status) status ;;
  stop) stop ;;
  *) echo "usage: $0 {start|status|stop}" >&2; exit 2 ;;
esac
