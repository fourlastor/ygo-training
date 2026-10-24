#!/usr/bin/env bash
# PPO with a head that predicts: does it still undo the copy, and does it add
# to it? Run from the repository root on the CUDA machine, after
# learner-follow-up.sh, whose run `once-few` it starts from; the record is
# experiments/2026-10-09-ppo-with-a-head-that-predicts.md.
# 1. Five runs of PPO from the same checkpoint, without an anchor unless
#    said, their games through the library's pool:
#      as-before            four passes, the head taught PPO's return at
#                           every transaction: PPO as it was run so far;
#      one-pass             one pass, the same lesson;
#      head-kept            one pass, the duel's result at four transactions
#                           of a seat: the lesson that keeps a head predicting;
#      head-kept-plain      the same, with an answer's credit the result
#                           minus the head's estimate (--duel-gae-lambda 1);
#      head-kept-anchored   as head-kept, with the pilot's answers as an
#                           anchor of 0.1.
# 2. Each run after the updates of POINTS on the held-out logged games: the
#    deals it wins, where its first choice leaves the pilot's answer, and
#    its value head against the worlds.
# 3. The readings.
# Finished runs and measurements are skipped and an interrupted run goes on.
# About four hours.
set -euo pipefail
DEVICE=${DEVICE:-cuda}
WORKERS=${WORKERS:-24}
THREADS=${THREADS:-20}
SLOTS=${SLOTS:-16}
DUELS=${DUELS:-520}
UPDATES=${UPDATES:-120}
EVERY=${EVERY:-30}
POINTS=${POINTS:-30 60 90 120}
OPPONENTS=${OPPONENTS:-blackwing,burn,crystal,dragunity,gishki,gladiator,heroes,infernity,lightsworn,monarch,morphtronic,rock-block,spellcaster}
GAMES=${GAMES:-analysis/search-leads}
START=${START:-runs/learner/once-few/checkpoints/model-00359.pt}
START_HELD=${START_HELD:-analysis/learner-follow-up/heldout/once-few.json}
RUNS=${RUNS:-runs/ppo-head}
OUT=${OUT:-analysis/ppo-head}
SUITE=${SUITE:-eval_suites/ygo-policies-blackwing-screen.json}

pinned=$(git submodule status --recursive)
if grep -q '^[^ ]' <<< "$pinned"; then
  echo "A submodule is not at its pinned revision: git submodule update --init --recursive" >&2
  exit 1
fi
for needed in "$GAMES/heldout.jsonl" "$GAMES/policies-revision.txt" "$SUITE" "$START" "$START_HELD"; do
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
# The updates the start has had: a run from it counts its versions on.
started=$(basename "$START" .pt)
STARTED=$((10#${started#model-} + 1))
if ((STARTED % EVERY)); then
  echo "The start has had $STARTED updates: EVERY must divide them, for the suite to be played after EVERY more" >&2
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
mkdir -p "$OUT/heldout"
# The start as it was measured, beside what is measured here.
cp "$START_HELD" "$OUT/heldout/start.json"

# A run after so many updates of its own.
checkpoint() { printf '%s/checkpoints/model-%05d.pt' "$RUNS/$1" $((STARTED + $2 - 1)); }

# ppo NAME OPTION...: the run from the start, or from where it stands, to
# UPDATES updates of games the model plays; its figures beside the measurements.
ppo() {
  local name=$1 latest
  shift
  local arguments=(
    --device "$DEVICE" --workers "$WORKERS" --pool-threads "$THREADS" --pool-slots "$SLOTS"
    --run "$RUNS/$name" --learner blackwing --opponents "$OPPONENTS" --duels "$DUELS"
    --eval-every "$EVERY" --eval-suite "$SUITE" --keep-checkpoints "$EVERY" --seed 2012 "$@"
  )
  if ! ls "$RUNS/$name"/checkpoints/model-*.pt > /dev/null 2>&1; then
    uv run python -m ygo_training train "${arguments[@]}" --resume "$START" --updates "$UPDATES"
  elif [ ! -f "$(checkpoint "$name" "$UPDATES")" ]; then
    latest=$(basename "$(ls "$RUNS/$name"/checkpoints/model-*.pt | sort | tail -1)" .pt)
    uv run python -m ygo_training train "${arguments[@]}" --continue \
      --updates $((STARTED + UPDATES - 1 - 10#${latest#model-}))
  fi
  cp "$RUNS/$name/metrics.jsonl" "$OUT/$name.metrics.jsonl"
  cp "$RUNS/$name/run.json" "$OUT/$name.run.json"
}

# 1. The runs.
ppo as-before --ppo-epochs 4
ppo one-pass --ppo-epochs 1
ppo head-kept --ppo-epochs 1 --ppo-value-positions 4 --ppo-value-target result
ppo head-kept-plain --ppo-epochs 1 --ppo-value-positions 4 --ppo-value-target result \
  --duel-gae-lambda 1
ppo head-kept-anchored --ppo-epochs 1 --ppo-value-positions 4 --ppo-value-target result \
  --teacher-imitation-weight 0.1

# 2. measure NAME: the run's checkpoints after the updates of POINTS on the
#    held-out logged games. The deals a checkpoint plays itself go through
#    the pool, which plays them as the workers do; a recorded duel it
#    watches stays with the workers.
measure() {
  local name=$1 point found=()
  for point in $POINTS; do
    found+=("$(checkpoint "$name" "$point")")
  done
  if [ ! -f "$OUT/heldout/$name.json" ]; then
    uv run python -m ygo_training evaluate-search "${found[@]}" --device "$DEVICE" \
      --workers "$WORKERS" --pool-threads "$THREADS" --pool-slots "$SLOTS" \
      --records "$GAMES/heldout.jsonl" \
      --output "$OUT/heldout/$name.json.part" | tee -a "$OUT/heldout/tables.txt"
    mv "$OUT/heldout/$name.json.part" "$OUT/heldout/$name.json"
  fi
}
for name in as-before one-pass head-kept head-kept-plain head-kept-anchored; do
  measure "$name"
done

# 3. The readings.
uv run python tools/ppo_head_report.py "$OUT" --output "$OUT/report.json" | tee "$OUT/report.txt"
echo "The runs' figures, the measurements and the readings are in $OUT: copy it back."
