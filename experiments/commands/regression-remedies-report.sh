#!/usr/bin/env bash
# Gather the PPO regression remedies results under analysis/remedies/ for
# syncing: each run's metrics and status table, and action_values score of
# every evaluated checkpoint. Run from training/.
set -euo pipefail
OUT=analysis/remedies
mkdir -p "$OUT"
checkpoints=(--checkpoint "v9=runs/ygo-policies-setcodes-baseline-seed-2012/checkpoints/model-00009.pt")
for arm in bc dagger anchor-0.1 anchor-1; do
  run=runs/ygo-policies-remedies-$arm
  if [ ! -f "$run/metrics.jsonl" ]; then
    continue
  fi
  cp "$run/metrics.jsonl" "$OUT/$arm-metrics.jsonl"
  uv run python -m ygo_training status "$run/metrics.jsonl" --ppo > "$OUT/$arm-status.txt"
  for version in 19 29 39 49 59; do
    path=$(printf '%s/checkpoints/model-%05d.pt' "$run" "$version")
    if [ -f "$path" ]; then
      checkpoints+=(--checkpoint "$arm-v$version=$path")
    fi
  done
done
uv run python -m ygo_training.action_values score \
  --values analysis/action-values-blackwing-seed-7000.pt "${checkpoints[@]}" \
  > "$OUT/action-values-score.json"
echo "Results are in $OUT"
