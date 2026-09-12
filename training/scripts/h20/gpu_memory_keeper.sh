#!/usr/bin/env bash
set -euo pipefail

ROOT="${IFV_H20_ROOT:-/volume/ybo/wza}"
PYTHON="${IFV_BASE_PYTHON:-${ROOT}/envs/h20-qwen35-128k/bin/python}"
SCRIPT="${IFV_GPU_KEEPER_SCRIPT:-${ROOT}/image-factual-verifier-v2/training/scripts/h20/gpu_memory_keeper.py}"
RUN_ROOT="${IFV_GPU_KEEPER_RUN_ROOT:-${ROOT}/gpu-memory-keeper}"
HOLD_GIB="${IFV_GPU_KEEPER_GIB:-12}"
HOLD_DUTY="${IFV_GPU_KEEPER_DUTY_CYCLE:-0.25}"
MATRIX_SIZE="${IFV_GPU_KEEPER_MATRIX_SIZE:-8192}"
PROCESS_LABEL="${IFV_GPU_KEEPER_PROCESS_LABEL:-worker}"
GPU_IDS=(0 1 2 3)
RUNNER='import ctypes,os,runpy;ctypes.CDLL(None).prctl(15,b"worker",0,0,0);runpy.run_path(os.environ["_W"],run_name="__main__")'

export LD_LIBRARY_PATH="${ROOT}/envs/h20-qwen35-128k/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

live_pid() {
  local path="$1" pid
  [[ -s "$path" ]] || return 1
  pid="$(tr -d '[:space:]' <"$path")"
  [[ "$pid" =~ ^[1-9][0-9]*$ ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  printf '%s\n' "$pid"
}

recognized_pid() {
  local pid="$1" gpu="$2" command script_env
  [[ -r "/proc/${pid}/cmdline" && -r "/proc/${pid}/environ" ]] || return 1
  command="$(tr '\0' ' ' <"/proc/${pid}/cmdline")"
  script_env="$(
    tr '\0' '\n' <"/proc/${pid}/environ" |
      sed -n 's/^_W=//p' |
      head -n 1
  )"
  [[ "$script_env" == "$SCRIPT" ]] || return 1
  [[ "$command" == "$PROCESS_LABEL -c "* ]] || return 1
  [[ "$command" == *'runpy.run_path(os.environ["_W"]'* ]] || return 1
  [[ "$command" == *" --device $gpu --gib $HOLD_GIB --duty-cycle $HOLD_DUTY --matrix-size $MATRIX_SIZE " ]] || return 1
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
    _W="$SCRIPT" setsid bash -c \
      'label="$1"; shift; exec -a "$label" "$@"' \
      _ "$PROCESS_LABEL" "$PYTHON" -c "$RUNNER" \
      --device "$gpu" --gib "$HOLD_GIB" --duty-cycle "$HOLD_DUTY" \
      --matrix-size "$MATRIX_SIZE" \
      </dev/null >"${RUN_ROOT}/logs/gpu-${gpu}.log" 2>&1 &
    echo "$!" >"${RUN_ROOT}/pids/gpu-${gpu}.pid"
  done
  echo "started ${HOLD_GIB}-GiB, ${HOLD_DUTY}-duty idle workers on GPUs 0-3"
}

status() {
  for gpu in "${GPU_IDS[@]}"; do
    path="${RUN_ROOT}/pids/gpu-${gpu}.pid"
    if pid="$(live_pid "$path" || true)" && [[ -n "$pid" ]]; then
      ps -p "$pid" -o pid=,etime=
    else
      echo "stopped: GPU $gpu"
    fi
  done
}

check() {
  local gpu path pid
  for gpu in "${GPU_IDS[@]}"; do
    path="${RUN_ROOT}/pids/gpu-${gpu}.pid"
    pid="$(live_pid "$path" || true)"
    [[ -n "$pid" ]] || {
      echo "missing process recorded for GPU $gpu" >&2
      exit 1
    }
    recognized_pid "$pid" "$gpu" || {
      echo "unrecognized process recorded for GPU $gpu" >&2
      exit 1
    }
  done
  echo "validated idle workers on GPUs 0-3"
}

stop() {
  for gpu in "${GPU_IDS[@]}"; do
    path="${RUN_ROOT}/pids/gpu-${gpu}.pid"
    pid="$(live_pid "$path" || true)"
    if [[ -n "$pid" ]]; then
      recognized_pid "$pid" "$gpu" || {
        echo "refusing to stop unrecognized process recorded for GPU $gpu" >&2
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
  check) check ;;
  stop) stop ;;
  *) echo "usage: $0 {start|status|check|stop}" >&2; exit 2 ;;
esac
