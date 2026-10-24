#!/usr/bin/env bash
# The models of the search-imitation experiment, read by the search's own
# worlds. Run from the repository root, after the arms
# (search-imitation-arms.sh).
# The held-out run's log holds how every alternative the search tried did
# against the pilot's answer. Each model's answers are read by that: the seed
# the arms started from, every arm after the updates listed in POINTS (its
# last, unless told otherwise), and the seed run's last checkpoint if that run
# has gone on since. Then one table is written.
# Nothing is trained, and what is measured already is skipped: for other
# POINTS, give another TRIED_OUT. The arms were measured after 4, 8, 16, 32
# and 64 updates, and a checkpoint takes about two minutes:
#   POINTS="4 8 16 32 64" TRIED_OUT=analysis/search-imitation/tried-by-update \
#     bash experiments/commands/search-imitation-tried.sh
set -euo pipefail
DEVICE=${DEVICE:-cuda}
WORKERS=${WORKERS:-24}
SIZES=${SIZES:-960 1920 3840 7680}
POINTS=${POINTS:-64}
GAMES=${GAMES:-analysis/search-imitation}
RUNS=${RUNS:-runs/search-imitation}
ARMS_OUT=${ARMS_OUT:-analysis/search-imitation/arms}
TRIED_OUT=${TRIED_OUT:-analysis/search-imitation/tried}
SEED_RUN=${SEED_RUN:-runs/format13-blackwing-imitation-seed-2012}

pinned=$(git submodule status --recursive)
if grep -q '^[^ ]' <<< "$pinned"; then
  echo "A submodule is not at its pinned revision: git submodule update --init --recursive" >&2
  exit 1
fi
cargo build --release
for needed in "$GAMES/heldout.jsonl" "$ARMS_OUT/seed.json"; do
  if [ ! -f "$needed" ]; then
    echo "Missing $needed" >&2
    exit 1
  fi
done
mkdir -p "$TRIED_OUT"

# The seed the arms started from: update k of an arm wrote version seed + k.
SEED_CHECKPOINT=$(uv run python -c 'import json, sys
print(json.load(open(sys.argv[1], encoding="utf8"))["checkpoints"][0]["checkpoint"])' "$ARMS_OUT/seed.json")
seed=$(basename "$SEED_CHECKPOINT" .pt)
seed=$((10#${seed#model-}))

measure() {
  local output=$1
  shift
  if [ ! -f "$output" ]; then
    uv run python -m ygo_training evaluate-search "$@" --device "$DEVICE" --workers "$WORKERS" \
      --records "$GAMES/heldout.jsonl" --output "$output.part" | tee -a "$TRIED_OUT/tables.txt"
    mv "$output.part" "$output"
  fi
}

measure "$TRIED_OUT/seed.json" "$SEED_CHECKPOINT"
for size in $SIZES; do
  for arm in search pilot; do
    measured=()
    for point in $POINTS; do
      measured+=("$(printf '%s/checkpoints/model-%05d.pt' "$RUNS/$arm-$size" $((seed + point)))")
    done
    measure "$TRIED_OUT/$arm-$size.json" "${measured[@]}"
  done
done
latest=$(ls "$SEED_RUN"/checkpoints/model-*.pt | sort | tail -1)
if [ "$(basename "$latest")" != "$(basename "$SEED_CHECKPOINT")" ]; then
  measure "$TRIED_OUT/seed-continued.json" "$latest"
fi
uv run python tools/search_imitation_report.py "$TRIED_OUT" --records "$GAMES/heldout.jsonl" \
  --output "$TRIED_OUT/report.json" | tee "$TRIED_OUT/report.txt"
