#!/usr/bin/env bash
# The arms of the search-imitation experiment. Run from the repository root,
# after the seed (format13-imitation-seed.sh) and the games
# (search-imitation-games.sh).
# For every set size, two runs from the seed, on the same deals in the same
# updates: one imitates the searched games, the other the pilot's own games.
# Every run does the same number of updates, whatever its size. Then the seed
# and every run are measured on the held-out run, the runs at the updates
# listed in POINTS, and one table is written.
# Finished runs and measurements are skipped; an interrupted run goes on.
set -euo pipefail
DEVICE=${DEVICE:-cuda}
WORKERS=${WORKERS:-24}
SIZES=${SIZES:-960 1920 3840 7680}
UPDATES=${UPDATES:-64}
ARM_DUELS=${ARM_DUELS:-480}
POINTS=${POINTS:-4 8 16 32 64}
OPPONENTS=${OPPONENTS:-burn,crystal,dragunity,gishki,gladiator,heroes,infernity,lightsworn,monarch,morphtronic,rock-block,spellcaster}
GAMES=${GAMES:-analysis/search-imitation}
RUNS=${RUNS:-runs/search-imitation}
ARMS_OUT=${ARMS_OUT:-analysis/search-imitation/arms}
SEED_RUN=${SEED_RUN:-runs/format13-blackwing-imitation-seed-2012}
# The arms start from the seed run's last checkpoint. Once that seed is
# measured, a later run of this script keeps to it, though the seed run may
# have gone on since.
if [ -z "${SEED_CHECKPOINT:-}" ] && [ -f "$ARMS_OUT/seed.json" ]; then
  SEED_CHECKPOINT=$(uv run python -c 'import json, sys
print(json.load(open(sys.argv[1], encoding="utf8"))["checkpoints"][0]["checkpoint"])' "$ARMS_OUT/seed.json")
fi
SEED_CHECKPOINT=${SEED_CHECKPOINT:-$(ls "$SEED_RUN"/checkpoints/model-*.pt | sort | tail -1)}

pinned=$(git submodule status --recursive)
if grep -q '^[^ ]' <<< "$pinned"; then
  echo "A submodule is not at its pinned revision: git submodule update --init --recursive" >&2
  exit 1
fi
cargo build --release
for rows in "$GAMES/train.jsonl" "$GAMES/heldout.jsonl" "$SEED_CHECKPOINT"; do
  if [ ! -f "$rows" ]; then
    echo "Missing $rows" >&2
    exit 1
  fi
done
mkdir -p "$ARMS_OUT"

# The version of the seed: update k of a run writes version seed + k.
seed=$(basename "$SEED_CHECKPOINT" .pt)
seed=$((10#${seed#model-}))
checkpoint() { printf '%s/checkpoints/model-%05d.pt' "$1" $((seed + $2)); }

measure() {
  local output=$1
  shift
  if [ ! -f "$output" ]; then
    uv run python -m ygo_training evaluate-search "$@" --device "$DEVICE" --workers "$WORKERS" \
      --records "$GAMES/heldout.jsonl" --output "$output.part"
    mv "$output.part" "$output"
  fi
}

measure "$ARMS_OUT/seed.json" "$SEED_CHECKPOINT"
for size in $SIZES; do
  for arm in search pilot; do
    run="$RUNS/$arm-$size"
    game=$arm
    if [ "$arm" = pilot ]; then game=baseline; fi
    train=(
      --device "$DEVICE" --workers "$WORKERS" --run "$run" --learner blackwing
      --opponents "$OPPONENTS" --duels "$ARM_DUELS"
      --imitation-only-updates "$UPDATES" --imitation-value-weight 0.5 --ppo-epochs 4
      --imitation-records "$GAMES/train.jsonl" --imitation-game "$game" --imitation-games "$size"
      --eval-every 0 --seed 2012
    )
    if ! ls "$run"/checkpoints/model-*.pt > /dev/null 2>&1; then
      uv run python -m ygo_training train "${train[@]}" --updates "$UPDATES" \
        --resume "$SEED_CHECKPOINT"
    elif [ ! -f "$(checkpoint "$run" "$UPDATES")" ]; then
      latest=$(basename "$(ls "$run"/checkpoints/model-*.pt | sort | tail -1)" .pt)
      uv run python -m ygo_training train "${train[@]}" --continue \
        --updates $((seed + UPDATES - 10#${latest#model-}))
    fi
    cp "$run/metrics.jsonl" "$ARMS_OUT/$arm-$size.metrics.jsonl"
    cp "$run/run.json" "$ARMS_OUT/$arm-$size.run.json"
    measured=()
    for point in $POINTS; do measured+=("$(checkpoint "$run" "$point")"); done
    measure "$ARMS_OUT/$arm-$size.json" "${measured[@]}"
  done
done
uv run python tools/search_imitation_report.py "$ARMS_OUT" --records "$GAMES/heldout.jsonl" \
  --output "$ARMS_OUT/report.json" | tee "$ARMS_OUT/report.txt"
