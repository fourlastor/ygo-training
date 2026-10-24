#!/usr/bin/env bash
# The arms of the search-leads experiment. Run from the repository root,
# after the games (search-leads-games.sh).
# 0. The seed against its pilot on the suite, as the games' script measured
#    the two: the arms start from it up to five points under, and nothing
#    is trained otherwise.
# 1. The weight of the lead loss: a run of four updates from the seed that
#    trains the head with the weight 1, and from its updates but the first
#    the imitation loss's gradient on the parameters the two losses share,
#    over the lead loss's.
# 2. Four runs from the seed, on the same deals in the same updates:
#      pilot       the pilot's games, imitated;
#      search      the searched games, the answers played imitated;
#      leads       the searched games, the pilot's answers imitated and the
#                  lead head taught, with the weight of step 1;
#      leads-read  the same with the weight 1 and no gradient from the head
#                  into what it reads.
# 3. The seed and every run after the updates of POINTS on the validation
#    run, and the choice: the update and the margin at which each of the two
#    last arms gains most by its rule.
# 4. The held-out run, read once: those two arms at their choice, its deals
#    also played by the rule, and the seed and the two other arms at the
#    updates so chosen. Then the readings in one table.
# Finished runs and measurements are skipped and an interrupted run goes on.
# The weight and the choice are kept as first written.
set -euo pipefail
DEVICE=${DEVICE:-cuda}
WORKERS=${WORKERS:-24}
UPDATES=${UPDATES:-128}
ARM_DUELS=${ARM_DUELS:-480}
POINTS=${POINTS:-16 32 64 96 128}
CALIBRATION_UPDATES=${CALIBRATION_UPDATES:-4}
OPPONENTS=${OPPONENTS:-burn,crystal,dragunity,gishki,gladiator,heroes,infernity,lightsworn,monarch,morphtronic,rock-block,spellcaster}
GAMES=${GAMES:-analysis/search-leads}
RUNS=${RUNS:-runs/search-leads}
OUT=${OUT:-analysis/search-leads/arms}
SEED_CHECKPOINT=${SEED_CHECKPOINT:-runs/format13-blackwing-imitation-seed-2012/checkpoints/model-00089.pt}
SEED_SUITE=${SEED_SUITE:-$GAMES/seed-suite.json}

pinned=$(git submodule status --recursive)
if grep -q '^[^ ]' <<< "$pinned"; then
  echo "A submodule is not at its pinned revision: git submodule update --init --recursive" >&2
  exit 1
fi
for needed in "$GAMES/train.jsonl" "$GAMES/validation.jsonl" "$GAMES/heldout.jsonl" \
  "$GAMES/policies-revision.txt" "$GAMES/pilot-suite.json" "$SEED_SUITE" "$SEED_CHECKPOINT"; do
  if [ ! -f "$needed" ]; then
    echo "Missing $needed" >&2
    exit 1
  fi
done
# A recorded duel plays again only on the pilots it was recorded with.
revision=$(git -C native/ygo-policies rev-parse HEAD)
if [ "$(cat "$GAMES/policies-revision.txt")" != "$revision" ]; then
  echo "The games of $GAMES are of ygo-policies $(cat "$GAMES/policies-revision.txt"), not $revision" >&2
  exit 1
fi
cargo build --release
mkdir -p "$OUT"
# The weight and the choice belong to the runs of one seed.
if [ -f "$OUT/seed.txt" ] && [ "$(cat "$OUT/seed.txt")" != "$SEED_CHECKPOINT" ]; then
  echo "$OUT was started from $(cat "$OUT/seed.txt"): give another OUT and RUNS" >&2
  exit 1
fi
echo "$SEED_CHECKPOINT" > "$OUT/seed.txt"
uv run python tools/search_leads_report.py seed "$GAMES/pilot-suite.json" "$SEED_SUITE" \
  --checkpoint "$SEED_CHECKPOINT" | tee "$OUT/seed-check.txt"

# The version of the seed: update k of a run writes version seed + k.
seed=$(basename "$SEED_CHECKPOINT" .pt)
seed=$((10#${seed#model-}))
checkpoint() { printf '%s/checkpoints/model-%05d.pt' "$1" $((seed + $2)); }

# train RUN UPDATES OPTION...: the run from the seed, to that many updates.
train() {
  local run=$1 updates=$2 latest
  shift 2
  local arguments=(
    --device "$DEVICE" --workers "$WORKERS" --run "$run" --learner blackwing
    --opponents "$OPPONENTS" --duels "$ARM_DUELS"
    --imitation-only-updates "$updates" --imitation-value-weight 0.5 --ppo-epochs 4
    --imitation-records "$GAMES/train.jsonl" --eval-every 0 --seed 2012 "$@"
  )
  if ! ls "$run"/checkpoints/model-*.pt > /dev/null 2>&1; then
    uv run python -m ygo_training train "${arguments[@]}" --updates "$updates" \
      --resume "$SEED_CHECKPOINT"
  elif [ ! -f "$(checkpoint "$run" "$updates")" ]; then
    latest=$(basename "$(ls "$run"/checkpoints/model-*.pt | sort | tail -1)" .pt)
    uv run python -m ygo_training train "${arguments[@]}" --continue \
      --updates $((seed + updates - 10#${latest#model-}))
  fi
}

# measure ROWS OUTPUT CHECKPOINT... [OPTION...]: the checkpoints on a run.
measure() {
  local rows=$1 output=$2
  shift 2
  if [ ! -f "$output" ]; then
    mkdir -p "$(dirname "$output")"
    uv run python -m ygo_training evaluate-search "$@" --device "$DEVICE" --workers "$WORKERS" \
      --records "$GAMES/$rows.jsonl" --output "$output.part"
    mv "$output.part" "$output"
  fi
}

# chosen ARM FIELD: of the arm's choice; nothing where it has none.
chosen() {
  uv run python -c 'import json, sys
choice = json.load(open(sys.argv[1], encoding="utf8")).get(sys.argv[2])
print("" if choice is None else choice[sys.argv[3]])' "$OUT/choice.json" "$1" "$2"
}

# 1. The weight of the lead loss.
if [ ! -f "$OUT/lead-weight.txt" ]; then
  train "$RUNS/calibration" "$CALIBRATION_UPDATES" \
    --imitation-game search --imitation-label pilot --imitation-lead-weight 1
  uv run python tools/search_leads_report.py weight "$RUNS/calibration/metrics.jsonl" \
    > "$OUT/lead-weight.txt.part"
  mv "$OUT/lead-weight.txt.part" "$OUT/lead-weight.txt"
fi
cat "$OUT/lead-weight.txt"
weight=$(awk '$1 == "weight:" { print $2 }' "$OUT/lead-weight.txt")

# 2. The arms.
arm() {
  local name=$1
  shift
  train "$RUNS/$name" "$UPDATES" "$@"
  cp "$RUNS/$name/metrics.jsonl" "$OUT/$name.metrics.jsonl"
  cp "$RUNS/$name/run.json" "$OUT/$name.run.json"
}
arm pilot --imitation-game baseline
arm search --imitation-game search
arm leads --imitation-game search --imitation-label pilot --imitation-lead-weight "$weight"
arm leads-read --imitation-game search --imitation-label pilot --imitation-lead-weight 1 \
  --imitation-lead-detach

# 3. The validation run, and the choice.
measure validation "$OUT/validation/seed.json" "$SEED_CHECKPOINT"
for name in pilot search leads leads-read; do
  measured=()
  for point in $POINTS; do measured+=("$(checkpoint "$RUNS/$name" "$point")"); done
  measure validation "$OUT/validation/$name.json" "${measured[@]}"
done
if [ ! -f "$OUT/choice.json" ]; then
  uv run python tools/search_leads_report.py read "$OUT" --choice "$OUT/choice.json.part" \
    > /dev/null
  mv "$OUT/choice.json.part" "$OUT/choice.json"
fi

# 4. The held-out run, at the choice.
updates=""
for name in leads leads-read; do
  update=$(chosen "$name" update)
  if [ -n "$update" ]; then
    measure heldout "$OUT/heldout/$name.json" "$(checkpoint "$RUNS/$name" "$update")" \
      --lead-margin "$(chosen "$name" margin)"
    updates="$updates $update"
  fi
done
if [ -n "$updates" ]; then
  measure heldout "$OUT/heldout/seed.json" "$SEED_CHECKPOINT"
  for name in pilot search; do
    measured=()
    for update in $(printf '%s\n' $updates | sort -n -u); do
      measured+=("$(checkpoint "$RUNS/$name" "$update")")
    done
    measure heldout "$OUT/heldout/$name.json" "${measured[@]}"
  done
fi
uv run python tools/search_leads_report.py read "$OUT" --output "$OUT/report.json" \
  | tee "$OUT/report.txt"
