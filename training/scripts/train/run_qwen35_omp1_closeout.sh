#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
RUN_TAG="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"
DATA_ROOT="${IFV_TRAINING_DATA_ROOT:-/gsdata/home/wza/image-factual-verifier-v2-data/training}"
CONDA="${IFV_CONDA_BIN:-/gs/home/wza/anaconda3/bin/conda}"
SFT_ENV="${IFV_QWEN35_SFT_ENV_PREFIX:-/gsdata/home/wza/conda/envs/ifv-qwen35-sft-ms-swift442}"
CACHE_ROOT="${IFV_QWEN35_CACHED_DATASET_ROOT:-$DATA_ROOT/cache/ms-swift-datasets/qwen35-pilot30-v3-3571ef8-cached-img512-sdpa-20260803-031028}"
CACHE_ENV="${IFV_QWEN35_CACHED_DATASET_ENV:-$CACHE_ROOT/cache.env}"
MODEL_PROFILE="$REPO_ROOT/training/configs/models/qwen3.5-9b.env"
TRAIN_JSONL="$CACHE_ROOT/source-snapshot/train.jsonl"
VAL_JSONL="$CACHE_ROOT/source-snapshot/validation.jsonl"
SERVICE_MANAGER="$REPO_ROOT/training/scripts/serve/manage_vllm_qwen35.sh"
RUN_ROOT="$DATA_ROOT/logs/omp1-closeout-$RUN_TAG"
LOCK_DIR="$DATA_ROOT/locks/qwen35-omp1-closeout.lock"
SUMMARY="$RUN_ROOT/closeout-summary.json"

if [[ ! "$RUN_TAG" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "RUN_TAG must be filesystem-safe: $RUN_TAG" >&2
  exit 2
fi
for path in \
  "$CONDA" \
  "$SFT_ENV/bin/python" \
  "$CACHE_ENV" \
  "$TRAIN_JSONL" \
  "$VAL_JSONL" \
  "$MODEL_PROFILE" \
  "$SERVICE_MANAGER"; do
  if [[ ! -e "$path" ]]; then
    echo "required closeout input is missing: $path" >&2
    exit 2
  fi
done
if [[ "${CUDA_VISIBLE_DEVICES:-}" != "4,5,6,7" ]]; then
  echo "OMP1 closeout requires CUDA_VISIBLE_DEVICES=4,5,6,7" >&2
  exit 2
fi
if [[ "${OMP_NUM_THREADS:-1}" != "1" ]]; then
  echo "OMP1 closeout requires OMP_NUM_THREADS=1" >&2
  exit 2
fi
mkdir -p "$DATA_ROOT/locks" "$RUN_ROOT"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "another Qwen3.5 OMP1 closeout owns $LOCK_DIR" >&2
  exit 2
fi

service_was_running=false
cleanup() {
  local status="$?"
  trap - EXIT
  rmdir "$LOCK_DIR" 2>/dev/null || true
  if [[ "$service_was_running" == "true" ]]; then
    if ! CUDA_VISIBLE_DEVICES=4,5 OMP_NUM_THREADS=1 \
      bash "$SERVICE_MANAGER" start >>"$RUN_ROOT/service-restore.log" 2>&1; then
      echo "failed to restore the managed Qwen3.5 service" >&2
      status=1
    fi
  fi
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

for _ in $(seq 1 120); do
  idle=true
  while IFS=',' read -r used util; do
    used="${used//[[:space:]]/}"
    util="${util//[[:space:]]/}"
    if [[ "$used" -gt 1024 ]]; then
      idle=false
      break
    fi
  done < <(
    nvidia-smi --id=4,5,6,7 \
      --query-gpu=memory.used,utilization.gpu \
      --format=csv,noheader,nounits
  )
  if [[ "$idle" == "true" ]]; then
    break
  fi
  sleep 1
done
if [[ "$idle" != "true" ]]; then
  echo "GPUs 4,5,6,7 did not become idle after managed service shutdown" >&2
  exit 2
fi

run_profile() {
  local profile="$1"
  local experiment_id="$2"
  local resume_checkpoint="${3:-}"
  local args=(
    "$CONDA" run --no-capture-output -p "$SFT_ENV"
    bash "$REPO_ROOT/training/scripts/train/run_sft.sh"
    "$MODEL_PROFILE"
    "$profile"
    "$TRAIN_JSONL"
    "$VAL_JSONL"
    "$experiment_id"
  )
  if [[ -n "$resume_checkpoint" ]]; then
    args+=("$resume_checkpoint")
  fi
  "${args[@]}"
}

worker0_id="qwen35-omp1-workers0-2step-$RUN_TAG"
worker4_id="qwen35-omp1-workers4-2step-$RUN_TAG"
worker0_profile="$REPO_ROOT/training/configs/sft/qwen3.5-full-2step-4gpu-zero3-offload-cached-workers0-omp1.env"
worker4_profile="$REPO_ROOT/training/configs/sft/qwen3.5-full-2step-4gpu-zero3-offload-cached-workers4-omp1.env"
run_profile "$worker0_profile" "$worker0_id"
run_profile "$worker4_profile" "$worker4_id"

worker0_result="$DATA_ROOT/logs/$worker0_id/profile.json"
worker4_result="$DATA_ROOT/logs/$worker4_id/profile.json"
read -r winner worker0_speed worker4_speed < <(
  "$SFT_ENV/bin/python" - "$worker0_result" "$worker4_result" <<'PY'
import json
import sys
from pathlib import Path

profiles = [json.loads(Path(path).read_text(encoding="utf-8")) for path in sys.argv[1:]]
for path, profile in zip(sys.argv[1:], profiles):
    if profile.get("passed_production_gate") is not True:
        raise SystemExit(f"worker gate did not pass: {path}")
speeds = [
    float(profile["throughput"]["unique_samples_per_second"])
    for profile in profiles
]
# Treat a <=5% difference as run variance and retain the simpler workers0 path.
winner = "workers4" if speeds[1] > speeds[0] * 1.05 else "workers0"
print(winner, *speeds)
PY
)

if [[ "$winner" == "workers4" ]]; then
  production_profile="$REPO_ROOT/training/configs/sft/qwen3.5-full-10step-4gpu-zero3-offload-cached-workers4-omp1.env"
  resume_profile="$REPO_ROOT/training/configs/sft/qwen3.5-full-11step-resume-4gpu-zero3-offload-cached-workers4-omp1.env"
else
  production_profile="$REPO_ROOT/training/configs/sft/qwen3.5-full-10step-4gpu-zero3-offload-cached-workers0-omp1.env"
  resume_profile="$REPO_ROOT/training/configs/sft/qwen3.5-full-11step-resume-4gpu-zero3-offload-cached-workers0-omp1.env"
fi

production_id="qwen35-omp1-$winner-10step-$RUN_TAG"
resume_id="qwen35-omp1-$winner-resume11-$RUN_TAG"
run_profile "$production_profile" "$production_id"
checkpoint="$(
  find "$DATA_ROOT/checkpoints/$production_id" \
    -type d -name checkpoint-10 -print |
    sort -V |
    tail -n 1
)"
if [[ -z "$checkpoint" || ! -d "$checkpoint" ]]; then
  echo "checkpoint-10 was not produced for $production_id" >&2
  exit 2
fi
run_profile "$resume_profile" "$resume_id" "$checkpoint"

"$SFT_ENV/bin/python" - \
  "$SUMMARY" \
  "$RUN_TAG" \
  "$winner" \
  "$worker0_result" \
  "$worker4_result" \
  "$DATA_ROOT/logs/$production_id/profile.json" \
  "$DATA_ROOT/logs/$resume_id/profile.json" \
  "$checkpoint" <<'PY'
import json
import sys
from pathlib import Path

(
    output,
    run_tag,
    winner,
    worker0_path,
    worker4_path,
    production_path,
    resume_path,
    checkpoint,
) = sys.argv[1:]

def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))

worker0 = load(worker0_path)
worker4 = load(worker4_path)
production = load(production_path)
resume = load(resume_path)
profiles = {
    "workers0": worker0,
    "workers4": worker4,
    "production": production,
    "resume": resume,
}
for name, profile in profiles.items():
    if profile.get("passed_production_gate") is not True:
        raise SystemExit(f"{name} profile did not pass its production gate")
    cache = profile.get("encoded_processor_cache", {}).get("report", {})
    totals = cache.get("totals", {})
    requests = int(totals.get("requests", 0) or 0)
    hits = int(totals.get("hits", 0) or 0)
    misses = int(totals.get("misses", 0) or 0)
    if requests < 1 or hits != requests or misses != 0:
        raise SystemExit(
            f"{name} encoded cache is not a complete hit: "
            f"requests={requests} hits={hits} misses={misses}"
        )
summary = {
    "schema_version": "ifv-qwen35-omp1-closeout-v1",
    "run_tag": run_tag,
    "winner": winner,
    "selection_rule": "workers4 only when >5% faster; otherwise workers0",
    "worker_throughput": {
        "workers0": worker0["throughput"]["unique_samples_per_second"],
        "workers4": worker4["throughput"]["unique_samples_per_second"],
    },
    "production": {
        "profile": production_path,
        "steps": production["steps"],
        "throughput": production["throughput"],
        "validation": production["validation"],
        "checkpoint_save": production["checkpoint_save"],
        "scheduler_order": production["scheduler_order"],
        "checkpoint_io": production.get("checkpoint_io", {}),
        "encoded_processor_cache": production["encoded_processor_cache"],
    },
    "resume": {
        "profile": resume_path,
        "source_checkpoint": checkpoint,
        "steps": resume["steps"],
        "resume": resume["resume"],
        "validation": resume["validation"],
        "checkpoint_save": resume["checkpoint_save"],
        "scheduler_order": resume["scheduler_order"],
        "checkpoint_io": resume.get("checkpoint_io", {}),
        "encoded_processor_cache": resume["encoded_processor_cache"],
    },
    "passed": True,
}
path = Path(output)
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, indent=2, sort_keys=True))
PY
