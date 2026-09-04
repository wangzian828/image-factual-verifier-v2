#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 2 || "$#" -gt 4 ]]; then
  echo "usage: $0 CLOSEOUT_SUMMARY {stability|pilot} [RUN_TAG] [STABILITY_SUMMARY]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CLOSEOUT_SUMMARY="$(realpath "$1")"
PHASE="$2"
RUN_TAG="${3:-$(date -u +%Y%m%dT%H%M%SZ)}"
STABILITY_SUMMARY="${4:-}"
DATA_ROOT="${IFV_TRAINING_DATA_ROOT:-${IFV_DATA_ROOT:+${IFV_DATA_ROOT}/training}}"
DATA_ROOT="${DATA_ROOT:-${XDG_DATA_HOME:-${HOME}/.local/share}/image-factual-verifier/training}"
CONDA="${IFV_CONDA_BIN:-$(command -v conda || true)}"
SFT_ENV="${IFV_QWEN35_SFT_ENV_PREFIX:-}"
CACHE_ROOT="${IFV_QWEN35_CACHED_DATASET_ROOT:-$DATA_ROOT/cache/ms-swift-datasets/qwen35-pilot30-v3-3571ef8-cached-img512-sdpa-20260803-031028}"
CACHE_ENV="${IFV_QWEN35_CACHED_DATASET_ENV:-$CACHE_ROOT/cache.env}"
MODEL_PROFILE="$REPO_ROOT/training/configs/models/qwen3.5-9b.env"
TRAIN_JSONL="$CACHE_ROOT/source-snapshot/train.jsonl"
VAL_JSONL="$CACHE_ROOT/source-snapshot/validation.jsonl"
SERVICE_MANAGER="$REPO_ROOT/training/scripts/serve/manage_vllm_qwen35.sh"
RUN_ROOT="$DATA_ROOT/logs/omp1-$PHASE-$RUN_TAG"
LOCK_DIR="$DATA_ROOT/locks/qwen35-omp1-followup.lock"
SUMMARY="$RUN_ROOT/$PHASE-summary.json"

if [[ "$PHASE" != "stability" && "$PHASE" != "pilot" ]]; then
  echo "phase must be stability or pilot: $PHASE" >&2
  exit 2
fi
if [[ ! "$RUN_TAG" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "RUN_TAG must be filesystem-safe: $RUN_TAG" >&2
  exit 2
fi
if [[ "${CUDA_VISIBLE_DEVICES:-}" != "4,5,6,7" ]]; then
  echo "OMP1 follow-up requires CUDA_VISIBLE_DEVICES=4,5,6,7" >&2
  exit 2
fi
if [[ "${OMP_NUM_THREADS:-1}" != "1" ]]; then
  echo "OMP1 follow-up requires OMP_NUM_THREADS=1" >&2
  exit 2
fi
for path in \
  "$CLOSEOUT_SUMMARY" \
  "$CONDA" \
  "$SFT_ENV/bin/python" \
  "$CACHE_ENV" \
  "$TRAIN_JSONL" \
  "$VAL_JSONL" \
  "$MODEL_PROFILE" \
  "$SERVICE_MANAGER"; do
  if [[ ! -e "$path" ]]; then
    echo "required follow-up input is missing: $path" >&2
    exit 2
  fi
done
if [[ "$PHASE" == "pilot" ]]; then
  if [[ -z "$STABILITY_SUMMARY" ]]; then
    echo "pilot requires the matching stability summary" >&2
    exit 2
  fi
  STABILITY_SUMMARY="$(realpath "$STABILITY_SUMMARY")"
  if [[ ! -s "$STABILITY_SUMMARY" ]]; then
    echo "stability summary is missing or empty: $STABILITY_SUMMARY" >&2
    exit 2
  fi
fi

read -r winner expected_steps < <(
  "$SFT_ENV/bin/python" - \
    "$CLOSEOUT_SUMMARY" \
    "$PHASE" \
    "$STABILITY_SUMMARY" <<'PY'
import json
import sys
from pathlib import Path

closeout_path, phase, stability_path = sys.argv[1:]
closeout = json.loads(Path(closeout_path).read_text(encoding="utf-8"))
if closeout.get("schema_version") != "ifv-qwen35-omp1-closeout-v1":
    raise SystemExit("unsupported closeout summary schema")
if closeout.get("passed") is not True:
    raise SystemExit("H1 closeout summary did not pass")
winner = str(closeout.get("winner") or "")
if winner not in {"workers0", "workers4"}:
    raise SystemExit(f"unknown H1 winner: {winner!r}")
if phase == "pilot":
    stability = json.loads(Path(stability_path).read_text(encoding="utf-8"))
    if stability.get("schema_version") != "ifv-qwen35-omp1-followup-v1":
        raise SystemExit("unsupported stability summary schema")
    if stability.get("phase") != "stability" or stability.get("passed") is not True:
        raise SystemExit("stability summary did not pass")
    if stability.get("winner") != winner:
        raise SystemExit("stability winner does not match H1 closeout winner")
print(winner, 20 if phase == "stability" else 99)
PY
)

if [[ "$winner" == "workers4" ]]; then
  if [[ "$PHASE" == "stability" ]]; then
    PROFILE="$REPO_ROOT/training/configs/sft/qwen3.5-full-20step-4gpu-zero3-offload-cached-workers4-omp1.env"
  else
    PROFILE="$REPO_ROOT/training/configs/sft/qwen3.5-full-pilot30-4gpu-zero3-offload-cached-workers4-omp1.env"
  fi
else
  if [[ "$PHASE" == "stability" ]]; then
    PROFILE="$REPO_ROOT/training/configs/sft/qwen3.5-full-20step-4gpu-zero3-offload-cached-workers0-omp1.env"
  else
    PROFILE="$REPO_ROOT/training/configs/sft/qwen3.5-full-pilot30-4gpu-zero3-offload-cached-workers0-omp1.env"
  fi
fi

# GPU 4/5 may hold only the managed service. Do not stop it while another
# user's work still occupies either of the independently required GPUs 6/7.
while IFS=',' read -r index used; do
  index="${index//[[:space:]]/}"
  used="${used//[[:space:]]/}"
  if [[ "$used" -gt 1024 ]]; then
    echo "GPU $index is not idle (${used} MiB); follow-up was not started" >&2
    exit 2
  fi
done < <(
  nvidia-smi --id=6,7 \
    --query-gpu=index,memory.used \
    --format=csv,noheader,nounits
)

mkdir -p "$DATA_ROOT/locks" "$RUN_ROOT"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "another Qwen3.5 OMP1 follow-up owns $LOCK_DIR" >&2
  exit 2
fi
service_was_running=false
cleanup() {
  local status="$?"
  trap - EXIT
  if [[ "$service_was_running" == "true" ]]; then
    if ! CUDA_VISIBLE_DEVICES=4,5 OMP_NUM_THREADS=1 \
      bash "$SERVICE_MANAGER" start >>"$RUN_ROOT/service-restore.log" 2>&1; then
      echo "failed to restore the managed Qwen3.5 service" >&2
      status=1
    fi
  fi
  rmdir "$LOCK_DIR" 2>/dev/null || true
  exit "$status"
}
trap cleanup EXIT

export CUDA_VISIBLE_DEVICES=4,5,6,7
export OMP_NUM_THREADS=1
export IFV_OMP_NUM_THREADS=1
export IFV_ALLOWED_GPU_IDS=4,5,6,7
export IFV_TRAINING_DATA_ROOT="$DATA_ROOT"
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost,::1}"
export no_proxy="${no_proxy:-$NO_PROXY}"
set -a
# shellcheck disable=SC1090
source "$CACHE_ENV"
set +a
export IFV_CACHED_DATASET_MANIFEST="$CACHE_ROOT/dataset-manifest.json"

if bash "$SERVICE_MANAGER" status >"$RUN_ROOT/service-before.txt" 2>&1; then
  service_was_running=true
  bash "$SERVICE_MANAGER" stop | tee "$RUN_ROOT/service-stop.log"
else
  echo "managed Qwen3.5 service was already stopped" >"$RUN_ROOT/service-stop.log"
fi

experiment_id="qwen35-omp1-$winner-$PHASE-$RUN_TAG"
"$CONDA" run --no-capture-output -p "$SFT_ENV" \
  bash "$REPO_ROOT/training/scripts/train/run_sft.sh" \
  "$MODEL_PROFILE" \
  "$PROFILE" \
  "$TRAIN_JSONL" \
  "$VAL_JSONL" \
  "$experiment_id"

PROFILE_RESULT="$DATA_ROOT/logs/$experiment_id/profile.json"
"$SFT_ENV/bin/python" - \
  "$SUMMARY" \
  "$CLOSEOUT_SUMMARY" \
  "$STABILITY_SUMMARY" \
  "$PHASE" \
  "$winner" \
  "$expected_steps" \
  "$PROFILE_RESULT" <<'PY'
import json
import sys
from pathlib import Path

(
    output,
    closeout_path,
    stability_path,
    phase,
    winner,
    expected_steps,
    profile_path,
) = sys.argv[1:]
profile = json.loads(Path(profile_path).read_text(encoding="utf-8"))
if profile.get("passed_production_gate") is not True:
    raise SystemExit("follow-up profile did not pass its production gate")
steps = profile.get("steps", {})
if int(steps.get("last") or 0) != int(expected_steps):
    raise SystemExit(
        f"follow-up ended at step {steps.get('last')}, expected {expected_steps}"
    )
cache = profile.get("encoded_processor_cache", {}).get("report", {})
totals = cache.get("totals", {})
requests = int(totals.get("requests", 0) or 0)
hits = int(totals.get("hits", 0) or 0)
misses = int(totals.get("misses", 0) or 0)
if requests < 1 or hits != requests or misses != 0:
    raise SystemExit(
        f"encoded cache is not a complete hit: "
        f"requests={requests} hits={hits} misses={misses}"
    )
step_wall = profile.get("step_wall_seconds", {})
if phase == "stability":
    if int(step_wall.get("steady_count") or 0) != 15:
        raise SystemExit("20-step stability profile requires 15 steady steps")
    if step_wall.get("steady_p90") is None:
        raise SystemExit("20-step stability profile lacks steady p90")
summary = {
    "schema_version": "ifv-qwen35-omp1-followup-v1",
    "phase": phase,
    "winner": winner,
    "h1_closeout_summary": closeout_path,
    "stability_summary": stability_path,
    "profile": profile_path,
    "steps": profile["steps"],
    "step_wall_seconds": step_wall,
    "throughput": profile["throughput"],
    "resources": profile["resources"],
    "encoded_processor_cache": profile["encoded_processor_cache"],
    "scheduler_order": profile["scheduler_order"],
    "validation": profile["validation"],
    "checkpoint_save": profile["checkpoint_save"],
    "checkpoint_io": profile.get("checkpoint_io", {}),
    "runtime_seconds": profile.get("runtime_seconds", {}),
    "passed": True,
}
path = Path(output)
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
