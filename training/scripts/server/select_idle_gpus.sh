#!/usr/bin/env bash
set -euo pipefail

mapfile -t rows < <(
  nvidia-smi \
    --query-gpu=index,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader,nounits |
    awk -F',' '$1 + 0 >= 4 && $1 + 0 <= 7 {
      for (i=1; i<=NF; i++) gsub(/^[ \t]+|[ \t]+$/, "", $i);
      print $1 "," $2 "," $3 "," $4
    }' |
    sort -t',' -k2,2n -k4,4n
)

printf '%-6s %-12s %-12s %-12s\n' GPU USED_MIB TOTAL_MIB UTIL_PERCENT
for row in "${rows[@]}"; do
  IFS=',' read -r gpu used total util <<<"$row"
  printf '%-6s %-12s %-12s %-12s\n' "$gpu" "$used" "$total" "$util"
done

if [[ "${#rows[@]}" -ge 1 ]]; then
  candidates=()
  for row in "${rows[@]}"; do
    IFS=',' read -r gpu used _ _ <<<"$row"
    if [[ "$used" -le 1024 ]]; then
      candidates+=("$gpu")
    fi
  done
  echo
  echo "Allowed candidates only; verify process ownership before use:"
  if [[ "${#candidates[@]}" -gt 0 ]]; then
    joined="$(IFS=,; echo "${candidates[*]:0:4}")"
    echo "export CUDA_VISIBLE_DEVICES=$joined"
  else
    echo "No idle GPU in the allowed set 4,5,6,7."
  fi
fi
