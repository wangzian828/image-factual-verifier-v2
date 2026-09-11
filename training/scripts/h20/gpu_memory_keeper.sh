#!/usr/bin/env bash
set -euo pipefail

ROOT="${IFV_H20_ROOT:-/volume/ybo/wza}"
PYTHON="${IFV_BASE_PYTHON:-${ROOT}/envs/h20-qwen35-128k/bin/python}"
SCRIPT="${IFV_GPU_KEEPER_SCRIPT:-${ROOT}/image-factual-verifier-v2/training/scripts/h20/gpu_memory_keeper.py}"
RUN_ROOT="${IFV_GPU_KEEPER_RUN_ROOT:-${ROOT}/gpu-memory-keeper}"
HOLD_GIB="${IFV_GPU_KEEPER_GIB:-12}"
HOLD_DUTY="${IFV_GPU_KEEPER_DUTY_CYCLE:-0.25}"
GPU_IDS=(0 1 2 3)

export LD_LIBRARY_PATH="${ROOT}/envs/h20-qwen35-128k/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

live_pid() {
  local path="$1" pid
  [[ -s "$path" ]] || return 1
  pid="$(tr -d '[:space:]' <"$path")"
  [[ "$pid" =~ ^[1-9][0-9]*$ ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  printf '%s\n' "$pid"
}

start() {
  [[ -x "$PYTHON" ]] || { echo "missing Python: $PYTHON" >&2; exit 2; }
  [[ -f "$SCRIPT" ]] || { echo "missing keeper: $SCRIPT" >&2; exit 2; }
  mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/pids"
  for gpu in "${GPU_IDS[@]}"; do
    path="${RUN_ROOT}/pids/gpu-${gpu}.pid"
    if live_pid "$path" >/dev/null; then
      echo "keeper already running for GPU $gpu" >&2
      exit 2
    fi
  done
  for gpu in "${GPU_IDS[@]}"; do
    setsid "$PYTHON" "$SCRIPT" --device "$gpu" --gib "$HOLD_GIB" --duty-cycle "$HOLD_DUTY" \
      </dev/null >"${RUN_ROOT}/logs/gpu-${gpu}.log" 2>&1 &
    echo "$!" >"${RUN_ROOT}/pids/gpu-${gpu}.pid"
  done
  echo "started ${HOLD_GIB}-GiB, ${HOLD_DUTY}-duty idle keepers on GPUs 0-3"
}

status() {
  for gpu in "${GPU_IDS[@]}"; do
    path="${RUN_ROOT}/pids/gpu-${gpu}.pid"
    if pid="$(live_pid "$path" || true)" && [[ -n "$pid" ]]; then
      ps -p "$pid" -o pid=,etime=,args=
    else
      echo "stopped: GPU $gpu"
    fi
  done
}

stop() {
  for gpu in "${GPU_IDS[@]}"; do
    path="${RUN_ROOT}/pids/gpu-${gpu}.pid"
    pid="$(live_pid "$path" || true)"
    if [[ -n "$pid" ]]; then
      command="$(ps -o args= -p "$pid" || true)"
      [[ "$command" == *"gpu_memory_keeper.py"* ]] || {
        echo "refusing to stop unrecognized process $pid: $command" >&2
        exit 1
      }
      pgid="$(ps -o pgid= -p "$pid" | tr -d ' ')"
      kill -TERM -- "-$pgid" 2>/dev/null || true
    fi
    rm -f -- "$path"
  done
  echo "stopped idle-memory keepers"
}

case "${1:-status}" in
  start) start ;;
  status) status ;;
  stop) stop ;;
  *) echo "usage: $0 {start|status|stop}" >&2; exit 2 ;;
esac
