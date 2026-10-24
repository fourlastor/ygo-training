#!/usr/bin/env bash
# The learner itself, the follow-up: the head that predicts and the best copy
# in one network. Run from the repository root on the CUDA machine, after
# learner-value-board.sh, whose runs it goes on from; the record is
# experiments/2026-10-08-value-head-and-board.md.
# 1. Four runs of one pass on the same games, board by positions:
#      once             the pooled board, a game's result at every transaction;
#      once-few         the pooled board, the result at four of them;
#      once-cards       the board as its cards, at every transaction;
#      once-cards-few   the board as its cards, at four: the new one.
#    `once` and `once-cards` are there from the first script, `once-few` is
#    taken on from its 120 updates, `once-cards-few` starts from scratch. A
#    run that is complete is left as it is.
# 2. Each run after the updates of POINTS on the held-out logged games: the
#    deals it wins, and its value head against the worlds.
# Finished runs and measurements are skipped and an interrupted run goes on.
# About two and a half hours.
set -euo pipefail
DEVICE=${DEVICE:-cuda}
WORKERS=${WORKERS:-24}
DUELS=${DUELS:-520}
UPDATES=${UPDATES:-360}
EVERY=${EVERY:-40}
POINTS=${POINTS:-240 280 320 360}
OPPONENTS=${OPPONENTS:-blackwing,burn,crystal,dragunity,gishki,gladiator,heroes,infernity,lightsworn,monarch,morphtronic,rock-block,spellcaster}
GAMES=${GAMES:-analysis/search-leads}
RUNS=${RUNS:-runs/learner}
OUT=${OUT:-analysis/learner-follow-up}
SUITE=${SUITE:-eval_suites/ygo-policies-blackwing-screen.json}

pinned=$(git submodule status --recursive)
if grep -q '^[^ ]' <<< "$pinned"; then
  echo "A submodule is not at its pinned revision: git submodule update --init --recursive" >&2
  exit 1
fi
for needed in "$GAMES/heldout.jsonl" "$GAMES/policies-revision.txt" "$SUITE"; do
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
if ((UPDATES % EVERY)); then
  echo "A run's last update must be one the suite is played after" >&2
  exit 1
fi
cargo build --release
mkdir -p "$OUT"

# A run from scratch after so many updates: its versions count from 0.
checkpoint() { printf '%s/checkpoints/model-%05d.pt' "$RUNS/$1" $(($2 - 1)); }

# imitate NAME OPTION...: the run from scratch, or from where it stands, to
# UPDATES updates of the pilot's games; its figures beside the measurements.
imitate() {
  local name=$1 latest
  shift
  local arguments=(
    --device "$DEVICE" --workers "$WORKERS" --run "$RUNS/$name" --learner blackwing
    --opponents "$OPPONENTS" --duels "$DUELS"
    --imitation-only-updates 1000 --imitation-value-weight 0.5
    --eval-every "$EVERY" --eval-suite "$SUITE" --keep-checkpoints "$EVERY" --seed 2012 "$@"
  )
  if ! ls "$RUNS/$name"/checkpoints/model-*.pt > /dev/null 2>&1; then
    uv run python -m ygo_training train "${arguments[@]}" --updates "$UPDATES"
  elif [ ! -f "$(checkpoint "$name" "$UPDATES")" ]; then
    latest=$(basename "$(ls "$RUNS/$name"/checkpoints/model-*.pt | sort | tail -1)" .pt)
    uv run python -m ygo_training train "${arguments[@]}" --continue \
      --updates $((UPDATES - 1 - 10#${latest#model-}))
  fi
  cp "$RUNS/$name/metrics.jsonl" "$OUT/$name.metrics.jsonl"
  cp "$RUNS/$name/run.json" "$OUT/$name.run.json"
}

# 1. The runs.
imitate once --ppo-epochs 1
imitate once-few --ppo-epochs 1 --imitation-value-positions 4
imitate once-cards --ppo-epochs 1 --board-representation individual-tokens
imitate once-cards-few --ppo-epochs 1 --imitation-value-positions 4 \
  --board-representation individual-tokens

# 2. measure NAME [OPTION...]: the run's checkpoints after the updates of
#    POINTS on the held-out logged games.
measure() {
  local name=$1 point found=()
  shift
  for point in $POINTS; do
    found+=("$(checkpoint "$name" "$point")")
  done
  if [ ! -f "$OUT/heldout/$name.json" ]; then
    mkdir -p "$OUT/heldout"
    uv run python -m ygo_training evaluate-search "${found[@]}" "$@" --device "$DEVICE" \
      --workers "$WORKERS" --records "$GAMES/heldout.jsonl" \
      --output "$OUT/heldout/$name.json.part" | tee -a "$OUT/heldout/tables.txt"
    mv "$OUT/heldout/$name.json.part" "$OUT/heldout/$name.json"
  fi
}
measure once
measure once-few
measure once-cards --board-representation individual-tokens
measure once-cards-few --board-representation individual-tokens

uv run python tools/learner_follow_up_report.py "$OUT" --output "$OUT/report.json" | tee "$OUT/report.txt"
echo "The runs' figures, the measurements and the readings are in $OUT: copy it back."
