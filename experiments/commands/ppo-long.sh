#!/usr/bin/env bash
# PPO in one pass, further: was its gain one run's luck, and does it go on
# adding? The follow-up of the experiment on PPO with a head that predicts
# (experiments/2026-10-09-ppo-with-a-head-that-predicts.md). Run from the
# repository root on the CUDA machine, after ppo-head.sh, whose run
# `one-pass` it starts beside and goes on from.
# 1. Two runs as long as `one-pass` and from the checkpoint it started from,
#    their games through the library's pool as its were:
#      one-pass-seed    the same, its games drawn with another seed;
#      one-pass-rate    the same at another learning rate.
#    Each after the updates of POINTS on the held-out logged games.
# 2. one-pass-long: `one-pass` on from its last checkpoint to LONG updates in
#    all, with a checkpoint every STEP of them, each on the held-out games.
# 3. The readings.
# PARTS="short" or PARTS="long" makes one of the two parts, and SHORT names
# the short runs to make: with RUNS and OUT of its own,
# PARTS=short SHORT=one-pass-seed SEED=2014 is a third seed, set beside the
# first run as the second was. Finished runs and measurements are skipped
# and an interrupted run goes on. About nine and a half hours: 75 minutes
# for the two short runs, the rest for the long one.
# Another run than `one-pass` is made long with NAME (the run's name), FIRST
# (its folder) and FIRST_OUT (where its figures and measurements are), with
# RUNS and OUT of its own; OPTIONS are added to its command line, and BESIDE
# is the folder of a long run its checkpoints are set beside. A second long
# run at the other learning rate, beside the first:
#   PARTS=long NAME=one-pass-rate FIRST=runs/ppo-long/one-pass-rate \
#     FIRST_OUT=analysis/ppo-long BESIDE=analysis/ppo-long \
#     RUNS=runs/ppo-long-rate OUT=analysis/ppo-long-rate
set -euo pipefail
DEVICE=${DEVICE:-cuda}
WORKERS=${WORKERS:-24}
THREADS=${THREADS:-20}
SLOTS=${SLOTS:-16}
DUELS=${DUELS:-520}
UPDATES=${UPDATES:-120}
EVERY=${EVERY:-30}
POINTS=${POINTS:-30 60 90 120}
LONG=${LONG:-1920}
STEP=${STEP:-120}
SEED=${SEED:-2013}
RATE=${RATE:-1e-4}
PARTS=${PARTS:-short long}
SHORT=${SHORT:-one-pass-seed one-pass-rate}
OPPONENTS=${OPPONENTS:-blackwing,burn,crystal,dragunity,gishki,gladiator,heroes,infernity,lightsworn,monarch,morphtronic,rock-block,spellcaster}
GAMES=${GAMES:-analysis/search-leads}
START=${START:-runs/learner/once-few/checkpoints/model-00359.pt}
NAME=${NAME:-one-pass}
FIRST=${FIRST:-runs/ppo-head/one-pass}
OPTIONS=${OPTIONS:-}
BESIDE=${BESIDE:-}
FIRST_OUT=${FIRST_OUT:-analysis/ppo-head}
RUNS=${RUNS:-runs/ppo-long}
OUT=${OUT:-analysis/ppo-long}
SUITE=${SUITE:-eval_suites/ygo-policies-blackwing-screen.json}

for part in $PARTS; do
  if [ "$part" != short ] && [ "$part" != long ]; then
    echo "PARTS names short, long or both, not $part" >&2
    exit 1
  fi
done
for name in $SHORT; do
  if [ "$name" != one-pass-seed ] && [ "$name" != one-pass-rate ]; then
    echo "SHORT names one-pass-seed, one-pass-rate or both, not $name" >&2
    exit 1
  fi
done
pinned=$(git submodule status --recursive)
if grep -q '^[^ ]' <<< "$pinned"; then
  echo "A submodule is not at its pinned revision: git submodule update --init --recursive" >&2
  exit 1
fi
# The updates the start has had: a run from it counts its versions on.
started=$(basename "$START" .pt)
STARTED=$((10#${started#model-} + 1))
# A run after so many updates of its own, and the first run after its last.
checkpoint() { printf '%s/checkpoints/model-%05d.pt' "$RUNS/$1" $((STARTED + $2 - 1)); }
first=$(printf '%s/checkpoints/model-%05d.pt' "$FIRST" $((STARTED + UPDATES - 1)))
for needed in "$GAMES/heldout.jsonl" "$GAMES/policies-revision.txt" "$SUITE" "$START" "$first" \
  "$FIRST_OUT/heldout/start.json" "$FIRST_OUT/heldout/$NAME.json" \
  "$FIRST_OUT/$NAME.metrics.jsonl" "$FIRST_OUT/$NAME.run.json"; do
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
if ((STARTED % EVERY || UPDATES % EVERY)); then
  echo "EVERY must divide the start's $STARTED updates and the runs' $UPDATES, for the suite to be played after EVERY more" >&2
  exit 1
fi
for point in $POINTS; do
  if ((point % EVERY || point > UPDATES)); then
    echo "No checkpoint is kept after $point updates: POINTS are multiples of EVERY up to UPDATES" >&2
    exit 1
  fi
done
if ((STARTED % STEP || UPDATES % STEP || LONG % STEP || LONG < UPDATES + 3 * STEP)); then
  echo "STEP must divide the start's $STARTED updates, the first run's $UPDATES and LONG, and the long run is read by three checkpoints after the first run's" >&2
  exit 1
fi
cargo build --release
mkdir -p "$OUT/heldout"
# The first run and its start as they were measured, beside what is measured here.
cp "$FIRST_OUT/heldout/start.json" "$FIRST_OUT/heldout/$NAME.json" "$OUT/heldout/"
cp "$FIRST_OUT/$NAME.metrics.jsonl" "$FIRST_OUT/$NAME.run.json" "$OUT/"

# ppo NAME FROM UPDATES KEPT OPTION...: the run from the checkpoint FROM,
# or from where it stands, to UPDATES updates counted from the start, with a
# checkpoint and the suite every KEPT of them; its figures beside the
# measurements.
ppo() {
  local name=$1 from=$2 updates=$3 kept=$4 began latest
  shift 4
  local arguments=(
    --device "$DEVICE" --workers "$WORKERS" --pool-threads "$THREADS" --pool-slots "$SLOTS"
    --run "$RUNS/$name" --learner blackwing --opponents "$OPPONENTS" --duels "$DUELS"
    --eval-every "$kept" --eval-suite "$SUITE" --keep-checkpoints "$kept" --seed 2012
    --ppo-epochs 1 "$@"
  )
  if ! ls "$RUNS/$name"/checkpoints/model-*.pt > /dev/null 2>&1; then
    began=$(basename "$from" .pt)
    uv run python -m ygo_training train "${arguments[@]}" --resume "$from" \
      --updates $((STARTED + updates - 1 - 10#${began#model-}))
  elif [ ! -f "$(checkpoint "$name" "$updates")" ]; then
    latest=$(basename "$(ls "$RUNS/$name"/checkpoints/model-*.pt | sort | tail -1)" .pt)
    uv run python -m ygo_training train "${arguments[@]}" --continue \
      --updates $((STARTED + updates - 1 - 10#${latest#model-}))
  fi
  cp "$RUNS/$name/metrics.jsonl" "$OUT/$name.metrics.jsonl"
  cp "$RUNS/$name/run.json" "$OUT/$name.run.json"
}

# measure FILE CHECKPOINT...: those checkpoints on the held-out logged games.
# The deals a checkpoint plays itself go through the pool, which plays them
# as the workers do; a recorded duel it watches stays with the workers.
measure() {
  local file=$1
  shift
  if [ ! -f "$OUT/heldout/$file.json" ]; then
    uv run python -m ygo_training evaluate-search "$@" --device "$DEVICE" \
      --workers "$WORKERS" --pool-threads "$THREADS" --pool-slots "$SLOTS" \
      --records "$GAMES/heldout.jsonl" \
      --output "$OUT/heldout/$file.json.part" | tee -a "$OUT/heldout/tables.txt"
    mv "$OUT/heldout/$file.json.part" "$OUT/heldout/$file.json"
  fi
}

report() {
  local beside=()
  if [ -n "$BESIDE" ]; then
    beside=(--beside "$BESIDE")
  fi
  uv run python tools/ppo_long_report.py "$OUT" --long "$LONG" --first "$NAME" \
    ${beside[@]+"${beside[@]}"} \
    --output "$OUT/report.json" | tee "$OUT/report.txt"
}

# 1. The two runs as long as the first, and each on the held-out games.
if [[ " $PARTS " == *" short "* ]]; then
  for name in $SHORT; do
    if [ "$name" = one-pass-seed ]; then
      ppo one-pass-seed "$START" "$UPDATES" "$EVERY" --training-seed "$SEED"
    else
      ppo one-pass-rate "$START" "$UPDATES" "$EVERY" --resume-learning-rate "$RATE"
    fi
  done
  for name in $SHORT; do
    found=()
    for point in $POINTS; do
      found+=("$(checkpoint "$name" "$point")")
    done
    measure "$name" "${found[@]}"
  done
  # What the short runs say, while the long one is made.
  if [[ " $PARTS " == *" long "* ]]; then
    report
  fi
fi

# 2. The first run made long, and each of its checkpoints on the held-out
#    games: a file a checkpoint, so that a run made longer is measured on.
if [[ " $PARTS " == *" long "* ]]; then
  # shellcheck disable=SC2086
  ppo "$NAME-long" "$first" "$LONG" "$STEP" $OPTIONS
  for ((point = UPDATES + STEP; point <= LONG; point += STEP)); do
    measure "$NAME-long.$(printf '%04d' "$point")" "$(checkpoint "$NAME-long" "$point")"
  done
fi

# 3. The readings.
report
echo "The runs' figures, the measurements and the readings are in $OUT: copy it back."
