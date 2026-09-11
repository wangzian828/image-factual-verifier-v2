#!/usr/bin/env bash
set -euo pipefail

ROOT="${IFV_H20_ROOT:-/volume/ybo/wza}"
REPO="${IFV_QWEN_REPO:-${ROOT}/image-factual-verifier-v2}"
ENV_PREFIX="${IFV_VLLM_ENV_PREFIX:-${ROOT}/envs/h20-qwen35-vllm-0181}"
BASE_PYTHON_ENV="${IFV_BASE_PYTHON_ENV:-${ROOT}/envs/h20-qwen35-128k}"
MODEL="${IFV_QWEN_MODEL:-${ROOT}/models/Qwen3.5-9B-local}"
RUN_ROOT="${IFV_QWEN_RUN_ROOT:-${ROOT}/inference/qwen35-base-vllm0181}"
VLLM="${ENV_PREFIX}/bin/vllm"
PYTHON="${ENV_PREFIX}/bin/python"
LOG_ROOT="${RUN_ROOT}/logs"
PID_ROOT="${RUN_ROOT}/pids"
PORTS=(8902 8903 8904 8905)
GPU_IDS=(0 1 2 3)

export LD_LIBRARY_PATH="${BASE_PYTHON_ENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

for path in "$REPO" "$ENV_PREFIX" "$BASE_PYTHON_ENV" "$MODEL"; do
  [[ -d "$path" ]] || { echo "missing directory: $path" >&2; exit 2; }
done
[[ -x "$VLLM" && -x "$PYTHON" ]] || { echo "missing vLLM environment" >&2; exit 2; }

pid_path() { printf '%s/replica-%s.pid\n' "$PID_ROOT" "$1"; }

live_pid() {
  local path="$1" pid
  [[ -s "$path" ]] || return 1
  pid="$(tr -d '[:space:]' <"$path")"
  [[ "$pid" =~ ^[1-9][0-9]*$ ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  printf '%s\n' "$pid"
}

start() {
  mkdir -p "$LOG_ROOT" "$PID_ROOT" "${RUN_ROOT}/cache" "${RUN_ROOT}/tmp"
  for gpu in "${GPU_IDS[@]}"; do
    if live_pid "$(pid_path "$gpu")" >/dev/null; then
      echo "replica already running for GPU $gpu" >&2
      exit 2
    fi
  done
  if live_pid "${PID_ROOT}/gateway.pid" >/dev/null; then
    echo "gateway already running" >&2
    exit 2
  fi

  for index in "${!GPU_IDS[@]}"; do
    gpu="${GPU_IDS[$index]}"
    port="${PORTS[$index]}"
    setsid env \
      CUDA_VISIBLE_DEVICES="$gpu" \
      HF_HOME="${ROOT}/cache/h20-huggingface" \
      XDG_CACHE_HOME="${RUN_ROOT}/cache" \
      FLASHINFER_WORKSPACE_DIR="${RUN_ROOT}/cache/flashinfer" \
      TMPDIR="${RUN_ROOT}/tmp" \
      VLLM_USE_FLASHINFER_SAMPLER=0 \
      "$VLLM" serve "$MODEL" \
        --host 127.0.0.1 --port "$port" \
        --served-model-name ifv-qwen3.5-9b \
        --dtype bfloat16 \
        --tensor-parallel-size 1 \
        --max-model-len 131072 \
        --gpu-memory-utilization 0.94 \
        --max-num-seqs 4 \
        --max-num-batched-tokens 32768 \
        --performance-mode throughput \
        --gdn-prefill-backend triton \
        --reasoning-parser qwen3 \
        --enable-auto-tool-choice \
        --tool-call-parser qwen3_coder \
        --structured-outputs-config '{"backend":"xgrammar","reasoning_parser":"qwen3","disable_any_whitespace":true}' \
        --limit-mm-per-prompt '{"image":32,"video":0}' \
        --enable-tokenizer-info-endpoint \
        --max-log-len 4000 \
        --disable-uvicorn-access-log \
      </dev/null >"${LOG_ROOT}/replica-${gpu}.log" 2>&1 &
    echo "$!" >"$(pid_path "$gpu")"
  done

  backends="http://127.0.0.1:${PORTS[0]},http://127.0.0.1:${PORTS[1]},http://127.0.0.1:${PORTS[2]},http://127.0.0.1:${PORTS[3]}"
  setsid env \
    QWEN_REPLICA_BACKENDS="$backends" \
    QWEN_REPLICA_MODEL_ID=ifv-qwen3.5-9b \
    QWEN_REPLICA_GATEWAY_TIMEOUT_SECONDS=900 \
    "$PYTHON" -m uvicorn scripts.server.qwen_replica_gateway:app \
      --app-dir "$REPO" --host 127.0.0.1 --port 8901 --log-level warning \
    </dev/null >"${LOG_ROOT}/gateway.log" 2>&1 &
  echo "$!" >"${PID_ROOT}/gateway.pid"
  echo "started four single-H20 base-model replicas and gateway"
}

status() {
  for gpu in "${GPU_IDS[@]}"; do
    path="$(pid_path "$gpu")"
    if pid="$(live_pid "$path" || true)" && [[ -n "$pid" ]]; then
      ps -p "$pid" -o pid=,etime=,args=
    else
      echo "stopped: GPU $gpu"
    fi
  done
  if pid="$(live_pid "${PID_ROOT}/gateway.pid" || true)" && [[ -n "$pid" ]]; then
    ps -p "$pid" -o pid=,etime=,args=
  else
    echo "stopped: gateway"
  fi
  curl -fsS --max-time 5 http://127.0.0.1:8901/health || true
  echo
}

stop_one() {
  local path="$1" pattern="$2" pid command pgid
  pid="$(live_pid "$path" || true)"
  [[ -n "$pid" ]] || return 0
  command="$(ps -o args= -p "$pid" || true)"
  [[ "$command" == *"$pattern"* ]] || {
    echo "refusing to stop unrecognized process $pid: $command" >&2
    return 1
  }
  pgid="$(ps -o pgid= -p "$pid" | tr -d ' ')"
  kill -TERM -- "-$pgid" 2>/dev/null || true
  for _ in {1..15}; do
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
  done
  if kill -0 "$pid" 2>/dev/null; then
    kill -KILL -- "-$pgid" 2>/dev/null || true
  fi
  rm -f -- "$path"
}

stop() {
  stop_one "${PID_ROOT}/gateway.pid" "qwen_replica_gateway:app"
  for gpu in "${GPU_IDS[@]}"; do
    stop_one "$(pid_path "$gpu")" "vllm serve"
  done
  echo "stopped H20 base-model replicas and gateway"
}

case "${1:-status}" in
  start) start ;;
  status) status ;;
  stop) stop ;;
  *) echo "usage: $0 {start|status|stop}" >&2; exit 2 ;;
esac
