#!/usr/bin/env bash
set -euo pipefail

mapfile -t rows < <(
  nvidia-smi \
    --query-gpu=index,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader,nounits |
    awk -F',' '{
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

if [[ "${#rows[@]}" -ge 2 ]]; then
  first="${rows[0]%%,*}"
  second="${rows[1]%%,*}"
  echo
  echo "Candidate only; verify process ownership before use:"
  echo "export CUDA_VISIBLE_DEVICES=$first,$second"
fi
