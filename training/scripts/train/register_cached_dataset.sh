#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"

if [[ "$#" -lt 3 || "$#" -gt 4 ]]; then
  echo "usage: $0 CACHE_DIR MODEL_PROFILE SFT_PROFILE [HISTORICAL_MANIFEST]" >&2
  exit 2
fi

CACHE_DIR="$(cd "$1" && pwd)"
MODEL_PROFILE="$2"
SFT_PROFILE="$3"
HISTORICAL_MANIFEST="${4:-}"
SELECTION="$CACHE_DIR/curriculum-selection.tsv"
FINGERPRINTS="$CACHE_DIR/source-dataset-fingerprints.tsv"
PROFILE="$CACHE_DIR/cache-profile.json"
MANIFEST="$CACHE_DIR/dataset-manifest.json"
VERIFICATION_DIR="$DATA_ROOT/logs/cache-registration"

load_profile "$MODEL_PROFILE"
load_profile "$SFT_PROFILE"
configure_training_runtime
require_model_path

if [[ ! -d "$CACHE_DIR/train" || ! -d "$CACHE_DIR/val" ]]; then
  echo "cached dataset must contain train/ and val/: $CACHE_DIR" >&2
  exit 2
fi
if [[ ! -s "$SELECTION" ]]; then
  echo "curriculum selection is missing: $SELECTION" >&2
  exit 2
fi
if [[ -e "$MANIFEST" ]]; then
  echo "refusing to replace existing cached dataset manifest: $MANIFEST" >&2
  exit 2
fi

mkdir -p "$VERIFICATION_DIR"
HISTORICAL_PREFLIGHT=""
if [[ -n "$HISTORICAL_MANIFEST" ]]; then
  if [[ ! -s "$HISTORICAL_MANIFEST" ]]; then
    echo "historical cached dataset manifest is missing: $HISTORICAL_MANIFEST" >&2
    exit 2
  fi
  HISTORICAL_PREFLIGHT="$VERIFICATION_DIR/$(basename "$CACHE_DIR")-historical-preflight.json"
  python - "$CACHE_DIR" "$HISTORICAL_MANIFEST" "$HISTORICAL_PREFLIGHT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


cache_dir = Path(sys.argv[1]).resolve()
manifest_path = Path(sys.argv[2]).resolve()
output = Path(sys.argv[3])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
errors = []
results = []
for artifact in manifest.get("artifacts") or []:
    relative = str(artifact.get("path") or "")
    path = cache_dir / relative
    result = {
        "path": relative,
        "exists": path.is_file(),
        "bytes_match": False,
        "sha256_match": False,
    }
    if path.is_file():
        result["bytes_match"] = path.stat().st_size == artifact.get("bytes")
        result["sha256_match"] = sha256(path) == artifact.get("sha256")
    if not result["exists"]:
        errors.append(f"missing:{relative}")
    elif not result["bytes_match"]:
        errors.append(f"bytes:{relative}")
    elif not result["sha256_match"]:
        errors.append(f"sha256:{relative}")
    results.append(result)
payload = {
    "schema_version": "ifv-historical-cache-preflight-v1",
    "passed": not errors,
    "cache_dir": str(cache_dir),
    "historical_manifest": str(manifest_path),
    "historical_manifest_sha256": sha256(manifest_path),
    "results": results,
    "errors": errors,
}
output.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
if errors:
    raise SystemExit(1)
PY
fi

source_paths_available=true
while IFS=$'\t' read -r channel _weight train_dataset validation_dataset; do
  if [[ "$channel" == "channel" ]]; then
    continue
  fi
  if [[ ! -s "$train_dataset" || ! -s "$validation_dataset" ]]; then
    source_paths_available=false
  fi
done <"$SELECTION"

REGISTRATION_KIND="existing-cache-original-sources"
if [[ "$source_paths_available" == "true" ]]; then
  {
    printf 'channel\tsplit\tpath\tbytes\tmtime_ns\tsha256\n'
    while IFS=$'\t' read -r channel _weight train_dataset validation_dataset; do
      if [[ "$channel" == "channel" ]]; then
        continue
      fi
      train_dataset="$(realpath "$train_dataset")"
      validation_dataset="$(realpath "$validation_dataset")"
      printf '%s\ttrain\t%s\t%s\t%s\t%s\n' \
        "$channel" \
        "$train_dataset" \
        "$(wc -c <"$train_dataset" | tr -d '[:space:]')" \
        "$(python -c 'import os, sys; print(os.stat(sys.argv[1]).st_mtime_ns)' "$train_dataset")" \
        "$(sha256sum "$train_dataset" | awk '{print $1}')"
      printf '%s\tvalidation\t%s\t%s\t%s\t%s\n' \
        "$channel" \
        "$validation_dataset" \
        "$(wc -c <"$validation_dataset" | tr -d '[:space:]')" \
        "$(python -c 'import os, sys; print(os.stat(sys.argv[1]).st_mtime_ns)' "$validation_dataset")" \
        "$(sha256sum "$validation_dataset" | awk '{print $1}')"
    done <"$SELECTION"
  } >"$FINGERPRINTS"
else
  if [[ -z "$HISTORICAL_PREFLIGHT" ]]; then
    echo "source datasets are missing; a verified historical manifest is required for recovery" >&2
    exit 2
  fi
  REGISTRATION_KIND="existing-cache-recovered-from-verified-cache"
  SOURCE_SNAPSHOT_DIR="$CACHE_DIR/source-snapshot"
  mkdir "$SOURCE_SNAPSHOT_DIR"
  python - "$CACHE_DIR" "$SOURCE_SNAPSHOT_DIR" <<'PY'
import sys
from pathlib import Path

from datasets import load_from_disk


cache_dir = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
for source_name, output_name in (
    ("train", "train.jsonl"),
    ("val", "validation.jsonl"),
):
    dataset = load_from_disk(str(cache_dir / source_name))
    dataset.to_json(
        str(output_dir / output_name),
        orient="records",
        lines=True,
        force_ascii=False,
    )
PY
  {
    printf 'channel\tsplit\tpath\tbytes\tmtime_ns\tsha256\n'
    for split in train validation; do
      if [[ "$split" == "train" ]]; then
        snapshot="$SOURCE_SNAPSHOT_DIR/train.jsonl"
      else
        snapshot="$SOURCE_SNAPSHOT_DIR/validation.jsonl"
      fi
      printf 'registered_cache\t%s\t%s\t%s\t%s\t%s\n' \
        "$split" \
        "$snapshot" \
        "$(wc -c <"$snapshot" | tr -d '[:space:]')" \
        "$(python -c 'import os, sys; print(os.stat(sys.argv[1]).st_mtime_ns)' "$snapshot")" \
        "$(sha256sum "$snapshot" | awk '{print $1}')"
    done
  } >"$FINGERPRINTS"
fi

python - \
  "$MODEL_PROFILE" \
  "$SFT_PROFILE" \
  "$CACHE_DIR" \
  "$PROFILE" \
  "$REGISTRATION_KIND" \
  "$HISTORICAL_MANIFEST" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


model_profile = Path(sys.argv[1]).resolve()
sft_profile = Path(sys.argv[2]).resolve()
cache_dir = Path(sys.argv[3]).resolve()
output = Path(sys.argv[4])
registration_kind = sys.argv[5]
historical_manifest = Path(sys.argv[6]).resolve() if sys.argv[6] else None
payload = {
    "schema_version": "ifv-cached-dataset-profile-v1",
    "cache_id": cache_dir.name,
    "registration": registration_kind,
    "model_profile": {
        "id": model_profile.name,
        "path": str(model_profile),
        "sha256": sha256(model_profile),
    },
    "sft_profile": {
        "id": sft_profile.name,
        "path": str(sft_profile),
        "sha256": sha256(sft_profile),
    },
    "model_id": os.environ["IFV_MODEL_ID"],
    "max_length": int(os.environ["IFV_MAX_LENGTH"]),
    "image_max_token_num": int(os.environ["IFV_IMAGE_MAX_TOKEN_NUM"]),
    "attention_implementation": os.environ["IFV_ATTN_IMPL"],
    "add_non_thinking_prefix": (
        os.environ.get("IFV_ADD_NON_THINKING_PREFIX", "false").lower() == "true"
    ),
    "historical_manifest": (
        {
            "path": str(historical_manifest),
            "sha256": sha256(historical_manifest),
        }
        if historical_manifest is not None
        else None
    ),
}
cache_log = cache_dir / "cache.log"
if not cache_log.is_file():
    raise FileNotFoundError(f"cache log is missing: {cache_log}")
log_text = cache_log.read_text(encoding="utf-8", errors="replace")
import re
import shlex

first_line = log_text.splitlines()[0]
if not first_line.startswith("run sh: `"):
    raise ValueError("cache log does not expose the original export command")
command = first_line.removeprefix("run sh: `").rsplit("`", 1)[0]
tokens = shlex.split(command)


def command_value(name: str) -> str:
    flag = f"--{name}"
    index = tokens.index(flag)
    return tokens[index + 1]


image_match = re.search(r"image_max_token_num:\s*(\d+)", log_text)
if image_match is None:
    raise ValueError("cache log does not expose image_max_token_num")
observed_contract = {
    "max_length": int(command_value("max_length")),
    "attention_implementation": command_value("attn_impl"),
    "image_max_token_num": int(image_match.group(1)),
    "add_non_thinking_prefix": (
        command_value("add_non_thinking_prefix").lower() == "true"
    ),
}
expected_contract = {
    "max_length": payload["max_length"],
    "attention_implementation": payload["attention_implementation"],
    "image_max_token_num": payload["image_max_token_num"],
    "add_non_thinking_prefix": payload["add_non_thinking_prefix"],
}
if observed_contract != expected_contract:
    raise ValueError(
        "cache build profile does not match cache.log: "
        f"expected={expected_contract} observed={observed_contract}"
    )
payload["observed_cache_build_contract"] = observed_contract
output.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY

if [[ -f "$CACHE_DIR/cache.env" ]]; then
  if ! grep -q '^IFV_CACHED_DATASET_MANIFEST=' "$CACHE_DIR/cache.env"; then
    printf 'IFV_CACHED_DATASET_MANIFEST=%s\n' "$MANIFEST" >>"$CACHE_DIR/cache.env"
  fi
  if ! grep -q '^IFV_CACHED_DATASET_VERSION=' "$CACHE_DIR/cache.env"; then
    printf 'IFV_CACHED_DATASET_VERSION=%s\n' "$(basename "$CACHE_DIR")" >>"$CACHE_DIR/cache.env"
  fi
else
  cat >"$CACHE_DIR/cache.env" <<EOF
IFV_CACHED_DATASET=$CACHE_DIR/train
IFV_CACHED_VAL_DATASET=$CACHE_DIR/val
IFV_CACHED_DATASET_MANIFEST=$MANIFEST
IFV_CACHED_DATASET_VERSION=$(basename "$CACHE_DIR")
IFV_LOAD_FROM_CACHE_FILE=true
EOF
fi

python -m ifv_training cached-dataset-manifest \
  --cache-dir "$CACHE_DIR" \
  --output "$MANIFEST"

python -m ifv_training verify-cached-dataset \
  --train-dir "$CACHE_DIR/train" \
  --validation-dir "$CACHE_DIR/val" \
  --manifest "$MANIFEST" \
  --output "$VERIFICATION_DIR/$(basename "$CACHE_DIR").json"

printf 'registered cached dataset: %s\n' "$CACHE_DIR"
