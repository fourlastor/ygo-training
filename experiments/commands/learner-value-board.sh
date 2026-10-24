#!/usr/bin/env bash
# The learner itself: the value head and the board. Run from the repository
# root on the CUDA machine; the record is
# experiments/2026-10-08-value-head-and-board.md.
# 0. What can be told of a result at the decisions the search examined, and
#    what a small network tells of it: the references the value heads are
#    read beside.
# 1. Imitation of the Blackwing pilot from scratch, on the same games in the
#    same order, with one thing changed in each run:
#      again          nothing: the seed's recipe, four passes through an
#                     update's games;
#      once           one pass, every game seen once;
#      once-few       one pass, and a game's result the value head's target
#                     at four of its transactions;
#      once-reading   one pass, and a value head that reads the shared state
#                     without shaping it;
#      once-cards     one pass, and the board as its cards.
#    A run of one pass makes four times the updates of `again`, which are as
#    many optimizer steps. The suite is played, and a checkpoint kept, every
#    EVERY updates (AGAIN_EVERY for `again`); the others are removed as the
#    run goes.
# 2. From the last checkpoints of `once` and `once-cards`, one pass through
#    the logged games of the search-leads experiment with the lead head
#    taught: what it explains before each update on games new to it.
# 3. The seed as it is, and the runs after the updates of POINTS, on the
#    held-out logged games: the deals they win, and their value heads
#    against the worlds. The two runs of step 2 on the validation games.
# Finished runs and measurements are skipped and an interrupted run goes on.
# About five and a half hours.
set -euo pipefail
DEVICE=${DEVICE:-cuda}
WORKERS=${WORKERS:-24}
DUELS=${DUELS:-520}
AGAIN_UPDATES=${AGAIN_UPDATES:-90}
AGAIN_EVERY=${AGAIN_EVERY:-10}
UPDATES=${UPDATES:-360}
FEW_UPDATES=${FEW_UPDATES:-120}
EVERY=${EVERY:-40}
POINTS=${POINTS:-120 240 360}
LEAD_UPDATES=${LEAD_UPDATES:-16}
LEAD_DUELS=${LEAD_DUELS:-480}
LEAD_WEIGHT=${LEAD_WEIGHT:-75}
OPPONENTS=${OPPONENTS:-blackwing,burn,crystal,dragunity,gishki,gladiator,heroes,infernity,lightsworn,monarch,morphtronic,rock-block,spellcaster}
LOGGED_OPPONENTS=${LOGGED_OPPONENTS:-burn,crystal,dragunity,gishki,gladiator,heroes,infernity,lightsworn,monarch,morphtronic,rock-block,spellcaster}
GAMES=${GAMES:-analysis/search-leads}
RUNS=${RUNS:-runs/learner}
OUT=${OUT:-analysis/learner}
SEED_CHECKPOINT=${SEED_CHECKPOINT:-runs/format13-blackwing-imitation-seed-2012/checkpoints/model-00089.pt}
SUITE=${SUITE:-eval_suites/ygo-policies-blackwing-screen.json}

pinned=$(git submodule status --recursive)
if grep -q '^[^ ]' <<< "$pinned"; then
  echo "A submodule is not at its pinned revision: git submodule update --init --recursive" >&2
  exit 1
fi
for needed in "$GAMES/train.jsonl" "$GAMES/validation.jsonl" "$GAMES/heldout.jsonl" \
  "$GAMES/policies-revision.txt" "$SEED_CHECKPOINT" "$SUITE"; do
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
for point in $POINTS; do
  if ((point % EVERY || point > UPDATES)); then
    echo "No checkpoint is kept after $point updates: POINTS are multiples of EVERY up to UPDATES" >&2
    exit 1
  fi
done
if ((AGAIN_UPDATES % AGAIN_EVERY || UPDATES % EVERY || FEW_UPDATES % EVERY)); then
  echo "A run's last update must be one the suite is played after" >&2
  exit 1
fi
cargo build --release
mkdir -p "$OUT"

# 0. The references.
if [ ! -f "$OUT/references.json" ]; then
  uv run python tools/policy_search_predictability.py "$GAMES/train.jsonl" "$GAMES/heldout.jsonl" \
    --output "$OUT/references.json.part" | tee "$OUT/references.txt"
  mv "$OUT/references.json.part" "$OUT/references.json"
fi

# A run from scratch after so many updates: its versions count from 0.
checkpoint() { printf '%s/checkpoints/model-%05d.pt' "$RUNS/$1" $(($2 - 1)); }

# keep NAME: the run's figures beside the measurements.
keep() {
  cp "$RUNS/$1/metrics.jsonl" "$OUT/$1.metrics.jsonl"
  cp "$RUNS/$1/run.json" "$OUT/$1.run.json"
}

# imitate NAME UPDATES EVERY OPTION...: the run from scratch, to that many
# updates of the pilot's games.
imitate() {
  local name=$1 updates=$2 every=$3 latest
  shift 3
  local arguments=(
    --device "$DEVICE" --workers "$WORKERS" --run "$RUNS/$name" --learner blackwing
    --opponents "$OPPONENTS" --duels "$DUELS"
    --imitation-only-updates 1000 --imitation-value-weight 0.5
    --eval-every "$every" --eval-suite "$SUITE" --keep-checkpoints "$every" --seed 2012 "$@"
  )
  if ! ls "$RUNS/$name"/checkpoints/model-*.pt > /dev/null 2>&1; then
    uv run python -m ygo_training train "${arguments[@]}" --updates "$updates"
  elif [ ! -f "$(checkpoint "$name" "$updates")" ]; then
    latest=$(basename "$(ls "$RUNS/$name"/checkpoints/model-*.pt | sort | tail -1)" .pt)
    uv run python -m ygo_training train "${arguments[@]}" --continue \
      --updates $((updates - 1 - 10#${latest#model-}))
  fi
  keep "$name"
}

# 1. The runs. The seed's own value head first, on games of the present
#    pilots that are new to it: the first of four updates as it was trained.
if [ ! -f "$RUNS/seed-value/metrics.jsonl" ]; then
  uv run python -m ygo_training train --device "$DEVICE" --workers "$WORKERS" \
    --run "$RUNS/seed-value" --learner blackwing --opponents "$OPPONENTS" --duels "$DUELS" \
    --imitation-only-updates 1000 --imitation-value-weight 0.5 --eval-every 0 --seed 2012 \
    --updates 4 --resume "$SEED_CHECKPOINT"
fi
keep seed-value
imitate again "$AGAIN_UPDATES" "$AGAIN_EVERY"
imitate once "$UPDATES" "$EVERY" --ppo-epochs 1
imitate once-few "$FEW_UPDATES" "$EVERY" --ppo-epochs 1 --imitation-value-positions 4
imitate once-reading "$UPDATES" "$EVERY" --ppo-epochs 1 --imitation-value-detach
imitate once-cards "$UPDATES" "$EVERY" --ppo-epochs 1 --board-representation individual-tokens

# 2. leads NAME FROM OPTION...: one pass through the logged games from the
#    last checkpoint of FROM, the pilot's answers imitated and the lead head
#    taught. Its versions go on from FROM's.
taught() { printf '%s/checkpoints/model-%05d.pt' "$RUNS/$1" $((UPDATES - 1 + LEAD_UPDATES)); }
leads() {
  local name=$1 from=$2 latest
  shift 2
  local arguments=(
    --device "$DEVICE" --workers "$WORKERS" --run "$RUNS/$name" --learner blackwing
    --opponents "$LOGGED_OPPONENTS" --duels "$LEAD_DUELS" --ppo-epochs 1
    --imitation-only-updates "$LEAD_UPDATES" --imitation-value-weight 0.5
    --imitation-records "$GAMES/train.jsonl" --imitation-game search --imitation-label pilot
    --imitation-lead-weight "$LEAD_WEIGHT" --eval-every 0 --seed 2012
    --keep-checkpoints "$LEAD_UPDATES" "$@"
  )
  if ! ls "$RUNS/$name"/checkpoints/model-*.pt > /dev/null 2>&1; then
    uv run python -m ygo_training train "${arguments[@]}" --updates "$LEAD_UPDATES" \
      --resume "$(checkpoint "$from" "$UPDATES")"
  elif [ ! -f "$(taught "$name")" ]; then
    latest=$(basename "$(ls "$RUNS/$name"/checkpoints/model-*.pt | sort | tail -1)" .pt)
    uv run python -m ygo_training train "${arguments[@]}" --continue \
      --updates $((UPDATES - 1 + LEAD_UPDATES - 10#${latest#model-}))
  fi
  keep "$name"
}
leads once-leads once
leads once-cards-leads once-cards --board-representation individual-tokens

# 3. measure ROWS NAME CHECKPOINT... [OPTION...]: the checkpoints on a run
#    of logged games.
measure() {
  local rows=$1 name=$2
  shift 2
  if [ ! -f "$OUT/$rows/$name.json" ]; then
    mkdir -p "$OUT/$rows"
    uv run python -m ygo_training evaluate-search "$@" --device "$DEVICE" --workers "$WORKERS" \
      --records "$GAMES/$rows.jsonl" --output "$OUT/$rows/$name.json.part" \
      | tee -a "$OUT/$rows/tables.txt"
    mv "$OUT/$rows/$name.json.part" "$OUT/$rows/$name.json"
  fi
}
# points NAME: the run's checkpoints after the updates of POINTS.
points() {
  local point
  for point in $POINTS; do
    checkpoint "$1" "$point"
    echo
  done
}
measure heldout seed "$SEED_CHECKPOINT"
measure heldout again "$(checkpoint again "$AGAIN_UPDATES")"
mapfile -t found < <(points once)
measure heldout once "${found[@]}"
measure heldout once-few "$(checkpoint once-few "$FEW_UPDATES")"
mapfile -t found < <(points once-reading)
measure heldout once-reading "${found[@]}"
mapfile -t found < <(points once-cards)
measure heldout once-cards "${found[@]}" --board-representation individual-tokens
measure validation once-leads "$(taught once-leads)"
measure validation once-cards-leads "$(taught once-cards-leads)" \
  --board-representation individual-tokens

cat "$OUT/references.txt"
echo "The runs' figures and the measurements are in $OUT: copy it back."
