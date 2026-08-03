#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"

if [[ "$#" -ne 5 ]]; then
  echo "usage: $0 MODEL_PROFILE SFT_PROFILE PERCEPTION_DIR POLICY_DIR CACHE_ID" >&2
  exit 2
fi

MODEL_PROFILE="$1"
SFT_PROFILE="$2"
PERCEPTION_DIR="$3"
POLICY_DIR="$4"
CACHE_ID="$5"

load_profile "$MODEL_PROFILE"
load_profile "$SFT_PROFILE"
configure_training_runtime
require_model_path
require_value CACHE_ID

candidate_train_datasets=(
  "$PERCEPTION_DIR/train.jsonl"
  "$POLICY_DIR/train.group-planning.jsonl"
  "$POLICY_DIR/train.group-react.jsonl"
  "$POLICY_DIR/train.group-decision.jsonl"
  "$POLICY_DIR/train.group-reflection.jsonl"
  "$POLICY_DIR/train.group-judgment.jsonl"
)
candidate_validation_datasets=(
  "$PERCEPTION_DIR/validation.jsonl"
  "$POLICY_DIR/validation.group-planning.jsonl"
  "$POLICY_DIR/validation.group-react.jsonl"
  "$POLICY_DIR/validation.group-decision.jsonl"
  "$POLICY_DIR/validation.group-reflection.jsonl"
  "$POLICY_DIR/validation.group-judgment.jsonl"
)
candidate_channels=(perception planning react decision reflection judgment)
candidate_weights=(0.20 0.15 0.35 0.20 0.05 0.05)
train_datasets=()
validation_datasets=()
active_channels=()
active_weights=()
for index in "${!candidate_channels[@]}"; do
  train_dataset="${candidate_train_datasets[$index]}"
  validation_dataset="${candidate_validation_datasets[$index]}"
  if [[ -s "$train_dataset" && -s "$validation_dataset" ]]; then
    train_datasets+=("$train_dataset")
    validation_datasets+=("$validation_dataset")
    active_channels+=("${candidate_channels[$index]}")
    active_weights+=("${candidate_weights[$index]}")
  elif [[ -s "$train_dataset" || -s "$validation_dataset" ]]; then
    echo "curriculum channel has only one non-empty split: ${candidate_channels[$index]}" >&2
    exit 2
  else
    echo "Skipping absent curriculum channel: ${candidate_channels[$index]}" >&2
  fi
done
if [[ "${#active_channels[@]}" -lt 1 ]]; then
  echo "no non-empty curriculum channels are available" >&2
  exit 2
fi
weight_sum="$({ printf '%s\n' "${active_weights[@]}"; } | awk '{sum += $1} END {printf "%.12g", sum}')"
interleave_prob=()
for weight in "${active_weights[@]}"; do
  interleave_prob+=("$(awk -v numerator="$weight" -v denominator="$weight_sum" 'BEGIN {printf "%.12g", numerator / denominator}')")
done

CACHE_ROOT="${IFV_CACHED_DATA_ROOT:-$DATA_ROOT/cache/ms-swift-datasets}"
CACHE_DIR="$CACHE_ROOT/$CACHE_ID"
new_output_dir "$CACHE_DIR"

{
  printf 'channel\tweight\ttrain_dataset\tvalidation_dataset\n'
  for index in "${!active_channels[@]}"; do
    printf '%s\t%s\t%s\t%s\n' \
      "${active_channels[$index]}" \
      "${interleave_prob[$index]}" \
      "${train_datasets[$index]}" \
      "${validation_datasets[$index]}"
  done
} >"$CACHE_DIR/curriculum-selection.tsv"

{
  printf 'channel\tsplit\tpath\tbytes\tsha256\n'
  for index in "${!active_channels[@]}"; do
    train_dataset="${train_datasets[$index]}"
    validation_dataset="${validation_datasets[$index]}"
    printf '%s\ttrain\t%s\t%s\t%s\n' \
      "${active_channels[$index]}" \
      "$train_dataset" \
      "$(wc -c <"$train_dataset" | tr -d '[:space:]')" \
      "$(sha256sum "$train_dataset" | awk '{print $1}')"
    printf '%s\tvalidation\t%s\t%s\t%s\n' \
      "${active_channels[$index]}" \
      "$validation_dataset" \
      "$(wc -c <"$validation_dataset" | tr -d '[:space:]')" \
      "$(sha256sum "$validation_dataset" | awk '{print $1}')"
  done
} >"$CACHE_DIR/source-dataset-fingerprints.tsv"

args=(
  swift export
  --model "$IFV_MODEL_ID"
  --dataset "${train_datasets[@]}"
  --val_dataset "${validation_datasets[@]}"
  --interleave_prob "${interleave_prob[@]}"
  --stopping_strategy all_exhausted
  --split_dataset_ratio 0
  --strict true
  --load_from_cache_file "$IFV_LOAD_FROM_CACHE_FILE"
  --torch_dtype "$IFV_TORCH_DTYPE"
  --max_length "$IFV_MAX_LENGTH"
  --attn_impl "$IFV_ATTN_IMPL"
  --to_cached_dataset true
  --output_dir "$CACHE_DIR"
  --exist_ok true
  --dataset_num_proc "${IFV_DATASET_NUM_PROC:-2}"
)

if [[ "${IFV_ADD_NON_THINKING_PREFIX:-false}" == "true" ]]; then
  args+=(--add_non_thinking_prefix true)
fi

export IMAGE_MAX_TOKEN_NUM="$IFV_IMAGE_MAX_TOKEN_NUM"
print_command "${args[@]}"
"${args[@]}" 2>&1 | tee "$CACHE_DIR/cache.log"

cat >"$CACHE_DIR/cache.env" <<EOF
IFV_CACHED_DATASET=$CACHE_DIR/train
IFV_CACHED_VAL_DATASET=$CACHE_DIR/val
IFV_LOAD_FROM_CACHE_FILE=true
EOF

echo "cached dataset profile: $CACHE_DIR/cache.env"
