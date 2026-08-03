#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"

if [[ "$#" -ne 3 ]]; then
  echo "usage: $0 CACHE_DIR MODEL_PROFILE SFT_PROFILE" >&2
  exit 2
fi

CACHE_DIR="$(cd "$1" && pwd)"
MODEL_PROFILE="$2"
SFT_PROFILE="$3"
SELECTION="$CACHE_DIR/curriculum-selection.tsv"
FINGERPRINTS="$CACHE_DIR/source-dataset-fingerprints.tsv"
PROFILE="$CACHE_DIR/cache-profile.json"
MANIFEST="$CACHE_DIR/dataset-manifest.json"

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

{
  printf 'channel\tsplit\tpath\tbytes\tmtime_ns\tsha256\n'
  while IFS=$'\t' read -r channel _weight train_dataset validation_dataset; do
    if [[ "$channel" == "channel" ]]; then
      continue
    fi
    require_dataset "$train_dataset"
    require_dataset "$validation_dataset"
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

python - "$MODEL_PROFILE" "$SFT_PROFILE" "$CACHE_DIR" "$PROFILE" <<'PY'
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
payload = {
    "schema_version": "ifv-cached-dataset-profile-v1",
    "cache_id": cache_dir.name,
    "registration": "existing-cache",
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
}
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

VERIFICATION_DIR="$DATA_ROOT/logs/cache-registration"
mkdir -p "$VERIFICATION_DIR"
python -m ifv_training verify-cached-dataset \
  --train-dir "$CACHE_DIR/train" \
  --validation-dir "$CACHE_DIR/val" \
  --manifest "$MANIFEST" \
  --output "$VERIFICATION_DIR/$(basename "$CACHE_DIR").json"

printf 'registered cached dataset: %s\n' "$CACHE_DIR"
