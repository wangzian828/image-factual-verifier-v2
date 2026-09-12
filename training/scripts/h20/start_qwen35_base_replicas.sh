#!/usr/bin/env bash
set -euo pipefail

ROOT="${IFV_H20_ROOT:-/volume/ybo/wza}"
REPO="${IFV_QWEN_REPO:-${ROOT}/image-factual-verifier-v2}"
ENV_PREFIX="${IFV_VLLM_ENV_PREFIX:-${ROOT}/envs/h20-qwen35-vllm-0181}"
BASE_PYTHON_ENV="${IFV_BASE_PYTHON_ENV:-${ROOT}/envs/h20-qwen35-128k}"
MODEL="${IFV_QWEN_MODEL:-${ROOT}/models/Qwen3.5-9B-local}"
LORA_ADAPTER="${IFV_QWEN_LORA_ADAPTER:-}"
CHECKPOINT_MANIFEST="${IFV_QWEN_CHECKPOINT_MANIFEST:-}"
RUN_ROOT="${IFV_QWEN_RUN_ROOT:-${ROOT}/inference/qwen35-base-vllm0181}"
VLLM="${ENV_PREFIX}/bin/vllm"
PYTHON="${ENV_PREFIX}/bin/python"
LOG_ROOT="${RUN_ROOT}/logs"
PID_ROOT="${RUN_ROOT}/pids"
read -r -a PORTS <<< "${IFV_QWEN_PORTS:-8902 8903 8904 8905}"
read -r -a GPU_IDS <<< "${IFV_QWEN_GPU_IDS:-0 1 2 3}"
GATEWAY_PORT="${IFV_QWEN_GATEWAY_PORT:-8901}"
BASE_GATEWAY_PORT="${IFV_QWEN_BASE_GATEWAY_PORT:-}"
MODEL_ALIAS="${IFV_QWEN_MODEL_ALIAS:-ifv-qwen3.5-9b}"
[[ "${#PORTS[@]}" == "${#GPU_IDS[@]}" && "${#GPU_IDS[@]}" -gt 0 ]] || {
  echo "GPU and port lists must have equal nonzero lengths" >&2; exit 2;
}

export LD_LIBRARY_PATH="${BASE_PYTHON_ENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

for path in "$REPO" "$ENV_PREFIX" "$BASE_PYTHON_ENV" "$MODEL"; do
  [[ -d "$path" ]] || { echo "missing directory: $path" >&2; exit 2; }
done
if [[ -n "$LORA_ADAPTER" && ! -d "$LORA_ADAPTER" ]]; then
  echo "missing LoRA adapter: $LORA_ADAPTER" >&2
  exit 2
fi
if [[ -n "$CHECKPOINT_MANIFEST" && ! -s "$CHECKPOINT_MANIFEST" ]]; then
  echo "missing checkpoint manifest: $CHECKPOINT_MANIFEST" >&2
  exit 2
fi
if [[ -n "$BASE_GATEWAY_PORT" ]]; then
  [[ -n "$LORA_ADAPTER" ]] || {
    echo "a separate base gateway requires a LoRA adapter" >&2
    exit 2
  }
  [[ "$BASE_GATEWAY_PORT" =~ ^[1-9][0-9]*$ && "$BASE_GATEWAY_PORT" != "$GATEWAY_PORT" ]] || {
    echo "IFV_QWEN_BASE_GATEWAY_PORT must be a distinct positive port" >&2
    exit 2
  }
fi
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

launch_replicas() {
  mkdir -p "$LOG_ROOT" "$PID_ROOT" "${RUN_ROOT}/cache" "${RUN_ROOT}/tmp"
  for gpu in "${GPU_IDS[@]}"; do
    if live_pid "$(pid_path "$gpu")" >/dev/null; then
      echo "replica already running for GPU $gpu" >&2
      exit 2
    fi
  done
  for index in "${!GPU_IDS[@]}"; do
    gpu="${GPU_IDS[$index]}"
    port="${PORTS[$index]}"
    served_name="$MODEL_ALIAS"
    lora_args=()
    execution_args=()
    if [[ -n "$LORA_ADAPTER" ]]; then
      served_name="${MODEL_ALIAS}-base"
      lora_args=(
        --enable-lora
        --max-lora-rank 32
        --enable-tower-connector-lora
        --lora-modules "$MODEL_ALIAS=$LORA_ADAPTER"
      )
    fi
    if [[ "${IFV_QWEN_ENFORCE_EAGER:-false}" == "true" ]]; then
      # vLLM 0.18.1 cannot CUDA-graph-warm all packed multimodal LoRA
      # projections. Eager mode bypasses graph capture without changing the
      # loaded adapter, KV cache, chunked prefill, or async scheduler.
      execution_args+=(--enforce-eager)
    fi
    setsid env \
      CUDA_VISIBLE_DEVICES="$gpu" \
      HF_HOME="${ROOT}/cache/h20-huggingface" \
      XDG_CACHE_HOME="${RUN_ROOT}/cache" \
      FLASHINFER_WORKSPACE_DIR="${RUN_ROOT}/cache/flashinfer" \
      TMPDIR="${RUN_ROOT}/tmp" \
      VLLM_USE_FLASHINFER_SAMPLER=0 \
      "$VLLM" serve "$MODEL" \
        --host 127.0.0.1 --port "$port" \
        --served-model-name "$served_name" \
        --dtype bfloat16 \
        --tensor-parallel-size 1 \
        --max-model-len 131072 \
        --gpu-memory-utilization 0.94 \
        --max-num-seqs "${IFV_QWEN_MAX_NUM_SEQS:-4}" \
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
        "${execution_args[@]}" \
        "${lora_args[@]}" \
      </dev/null >"${LOG_ROOT}/replica-${gpu}.log" 2>&1 &
    echo "$!" >"$(pid_path "$gpu")"
  done
}

write_serving_profile() {
  local profile="$RUN_ROOT/serving-profile.json"
  local args=(
    "$BASE_PYTHON_ENV/bin/python" -m ifv_training serving-profile
    --output "$profile"
    --profile-id "$MODEL_ALIAS"
    --model-path "${LORA_ADAPTER:-$MODEL}"
    --engine-model-path "$MODEL"
    --engine vllm
    --port "$GATEWAY_PORT"
    --tensor-parallel-size 1
    --dtype bfloat16
    --context-length 131072
    --tool-call-parser qwen3_coder
    --reasoning-parser qwen3
    --thinking-enabled false
  )
  if [[ -n "$LORA_ADAPTER" ]]; then
    args+=(--adapter-path "$LORA_ADAPTER")
  fi
  if [[ -n "$CHECKPOINT_MANIFEST" ]]; then
    args+=(--checkpoint-manifest "$CHECKPOINT_MANIFEST")
  fi
  PYTHONPATH="$REPO/training:$REPO${PYTHONPATH:+:$PYTHONPATH}" "${args[@]}"
}

launch_gateway() {
  local gateway_port="$1" model_id="$2" pid_file="$3" log_file="$4"
  backends=""
  for port in "${PORTS[@]}"; do
    backends+="${backends:+,}http://127.0.0.1:${port}"
  done
  setsid env \
    QWEN_REPLICA_BACKENDS="$backends" \
    QWEN_REPLICA_MODEL_ID="$model_id" \
    QWEN_REPLICA_GATEWAY_TIMEOUT_SECONDS=900 \
    "$PYTHON" -m uvicorn scripts.server.qwen_replica_gateway:app \
      --app-dir "$REPO" --host 127.0.0.1 --port "$gateway_port" --log-level warning \
    </dev/null >"$log_file" 2>&1 &
  echo "$!" >"$pid_file"
}

start() {
  if live_pid "${PID_ROOT}/gateway.pid" >/dev/null; then
    echo "gateway already running" >&2
    exit 2
  fi
  if [[ -n "$BASE_GATEWAY_PORT" ]] \
      && live_pid "${PID_ROOT}/base-gateway.pid" >/dev/null; then
    echo "base gateway already running" >&2
    exit 2
  fi
  write_serving_profile
  launch_replicas
  launch_gateway "$GATEWAY_PORT" "$MODEL_ALIAS" \
    "${PID_ROOT}/gateway.pid" "${LOG_ROOT}/gateway.log"
  if [[ -n "$BASE_GATEWAY_PORT" ]]; then
    launch_gateway "$BASE_GATEWAY_PORT" "${MODEL_ALIAS}-base" \
      "${PID_ROOT}/base-gateway.pid" "${LOG_ROOT}/base-gateway.log"
  fi
  echo "started ${#GPU_IDS[@]} single-H20 replicas for $MODEL_ALIAS; gateway=$GATEWAY_PORT"
}

add_replicas() {
  launch_replicas
  echo "added ${#GPU_IDS[@]} single-H20 replicas for $MODEL_ALIAS; gateway unchanged"
}

port_is_bindable() {
  local port="${1:-$GATEWAY_PORT}"
  "$PYTHON" -c \
    'import socket,sys; s=socket.socket(); s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1); s.bind(("127.0.0.1",int(sys.argv[1]))); s.close()' \
    "$port" >/dev/null 2>&1
}

replace_gateway() {
  local old_pid old_command drain_path
  old_pid="$(live_pid "${PID_ROOT}/gateway.pid" || true)"
  [[ -n "$old_pid" ]] || { echo "gateway is not running" >&2; exit 2; }
  old_command="$(ps -o args= -p "$old_pid" || true)"
  [[ "$old_command" == *"qwen_replica_gateway:app"* ]] || {
    echo "refusing to replace unrecognized process $old_pid: $old_command" >&2
    exit 2
  }
  for port in "${PORTS[@]}"; do
    curl -fsS --max-time 5 "http://127.0.0.1:${port}/health" >/dev/null || {
      echo "replica on port $port is not healthy" >&2
      exit 2
    }
  done

  # SIGTERM makes uvicorn close its listening socket while allowing requests
  # already accepted by the old gateway to drain.  Once the port is free, a
  # replacement gateway can accept new requests without killing those tasks.
  kill -TERM "$old_pid"
  drain_path="${PID_ROOT}/gateway-draining-${old_pid}.pid"
  printf '%s\n' "$old_pid" >"$drain_path"
  rm -f -- "${PID_ROOT}/gateway.pid"
  for _ in {1..300}; do
    port_is_bindable && break
    sleep 0.1
  done
  port_is_bindable || {
    echo "gateway port $GATEWAY_PORT did not become bindable" >&2
    exit 2
  }
  launch_gateway "$GATEWAY_PORT" "$MODEL_ALIAS" \
    "${PID_ROOT}/gateway.pid" "${LOG_ROOT}/gateway.log"
  for _ in {1..180}; do
    if curl -fsS --max-time 2 "http://127.0.0.1:${GATEWAY_PORT}/health" >/dev/null 2>&1; then
      echo "replaced gateway $old_pid; draining requests continue; backends=${PORTS[*]}"
      return 0
    fi
    sleep 1
  done
  echo "replacement gateway failed health check" >&2
  exit 2
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
  curl -fsS --max-time 5 "http://127.0.0.1:$GATEWAY_PORT/health" || true
  echo
  if [[ -n "$BASE_GATEWAY_PORT" ]]; then
    if pid="$(live_pid "${PID_ROOT}/base-gateway.pid" || true)" && [[ -n "$pid" ]]; then
      ps -p "$pid" -o pid=,etime=,args=
    else
      echo "stopped: base gateway"
    fi
    curl -fsS --max-time 5 "http://127.0.0.1:$BASE_GATEWAY_PORT/health" || true
    echo
  fi
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
  stop_one "${PID_ROOT}/base-gateway.pid" "qwen_replica_gateway:app"
  stop_one "${PID_ROOT}/gateway.pid" "qwen_replica_gateway:app"
  for gpu in "${GPU_IDS[@]}"; do
    stop_one "$(pid_path "$gpu")" "vllm serve"
  done
  echo "stopped H20 base-model replicas and gateway"
}

case "${1:-status}" in
  start) start ;;
  add-replicas) add_replicas ;;
  replace-gateway) replace_gateway ;;
  status) status ;;
  stop) stop ;;
  *) echo "usage: $0 {start|add-replicas|replace-gateway|status|stop}" >&2; exit 2 ;;
esac
