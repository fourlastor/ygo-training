#!/usr/bin/env bash
# A copy of the specialist: is what PPO added to the pilot's copy taken into
# a new network by imitating the checkpoint, in a fraction of the games it
# took to learn? Run from the repository root on the CUDA machine, after
# ppo-long.sh, whose run `one-pass-long` ends in the teacher, and
# learner-follow-up.sh, whose run `once-few` is the pilot's copy the new
# ones are set beside.
# 1. On the held-out logged games: the teacher, and the pilot's copy after
#    the updates of POINTS, where its checkpoints are kept.
# 2. Two runs from scratch that imitate the teacher as `once-few` imitated
#    the pilot: one pass, a game's result at four of its positions, from the
#    same first network and on the same deals, with the teacher in the
#    pilot's seat taking its first choices:
#      choice         taught its first choice at each decision, as a
#                     pilot's answer is taught;
#      distribution   taught what it gives every answer.
#    COPIES names the ones to make. Each after the updates of POINTS on the
#    held-out games.
# 3. The readings, after each run.
# Finished runs and measurements are skipped and an interrupted run goes on.
# About two hours and a quarter: 45 minutes a run, the rest to measure.
set -euo pipefail
DEVICE=${DEVICE:-cuda}
WORKERS=${WORKERS:-24}
THREADS=${THREADS:-20}
SLOTS=${SLOTS:-16}
DUELS=${DUELS:-520}
UPDATES=${UPDATES:-200}
EVERY=${EVERY:-40}
POINTS=${POINTS:-40 80 120 160 200}
COPIES=${COPIES:-choice distribution}
OPPONENTS=${OPPONENTS:-blackwing,burn,crystal,dragunity,gishki,gladiator,heroes,infernity,lightsworn,monarch,morphtronic,rock-block,spellcaster}
GAMES=${GAMES:-analysis/search-leads}
TEACHER=${TEACHER:-runs/ppo-long/one-pass-long/checkpoints/model-02279.pt}
# What the runs the teacher came from left, and the pilot's copy's.
LINE_OUT=${LINE_OUT:-analysis/ppo-long}
CONTROL=${CONTROL:-runs/learner/once-few}
CONTROL_OUT=${CONTROL_OUT:-analysis/learner-follow-up}
RUNS=${RUNS:-runs/specialist-copy}
OUT=${OUT:-analysis/specialist-copy}
SUITE=${SUITE:-eval_suites/ygo-policies-blackwing-screen.json}

for name in $COPIES; do
  if [ "$name" != choice ] && [ "$name" != distribution ]; then
    echo "COPIES names choice, distribution or both, not $name" >&2
    exit 1
  fi
done
pinned=$(git submodule status --recursive)
if grep -q '^[^ ]' <<< "$pinned"; then
  echo "A submodule is not at its pinned revision: git submodule update --init --recursive" >&2
  exit 1
fi
for needed in "$GAMES/heldout.jsonl" "$GAMES/policies-revision.txt" "$SUITE" "$TEACHER" \
  "$CONTROL_OUT/once-few.metrics.jsonl" "$CONTROL_OUT/once-few.run.json" \
  "$CONTROL_OUT/heldout/once-few.json"; do
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
mkdir -p "$OUT/heldout"
# The pilot's copy as it was made and measured, beside what is measured here;
# and, where they are there, the runs the teacher came from.
cp "$CONTROL_OUT/once-few.metrics.jsonl" "$OUT/pilot-copy.metrics.jsonl"
cp "$CONTROL_OUT/once-few.run.json" "$OUT/pilot-copy.run.json"
cp "$CONTROL_OUT/heldout/once-few.json" "$OUT/heldout/pilot-copy-later.json"
for found in "$LINE_OUT"/heldout/one-pass.json "$LINE_OUT"/heldout/one-pass-long.*.json; do
  if [ -f "$found" ]; then
    cp "$found" "$OUT/heldout/line.$(basename "$found")"
  fi
done
if [ -f "$LINE_OUT/one-pass-long.metrics.jsonl" ]; then
  cp "$LINE_OUT/one-pass-long.metrics.jsonl" "$OUT/teacher.metrics.jsonl"
fi

# A run from scratch after so many updates: its versions count from 0.
checkpoint() { printf '%s/checkpoints/model-%05d.pt' "$1" $(($2 - 1)); }

# copy NAME OPTION...: the run from scratch, or from where it stands, to
# UPDATES updates of the teacher's games; its figures beside the measurements.
copy() {
  local name=$1 latest
  shift
  local arguments=(
    --device "$DEVICE" --workers "$WORKERS" --pool-threads "$THREADS" --pool-slots "$SLOTS"
    --run "$RUNS/$name" --learner blackwing --opponents "$OPPONENTS" --duels "$DUELS"
    --imitation-only-updates 1000 --imitation-value-weight 0.5 --imitation-value-positions 4
    --ppo-epochs 1 --imitation-teacher "$TEACHER"
    --eval-every "$EVERY" --eval-suite "$SUITE" --keep-checkpoints "$EVERY" --seed 2012 "$@"
  )
  if ! ls "$RUNS/$name"/checkpoints/model-*.pt > /dev/null 2>&1; then
    uv run python -m ygo_training train "${arguments[@]}" --updates "$UPDATES"
  elif [ ! -f "$(checkpoint "$RUNS/$name" "$UPDATES")" ]; then
    latest=$(basename "$(ls "$RUNS/$name"/checkpoints/model-*.pt | sort | tail -1)" .pt)
    uv run python -m ygo_training train "${arguments[@]}" --continue \
      --updates $((UPDATES - 1 - 10#${latest#model-}))
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
  uv run python tools/specialist_copy_report.py "$OUT" --updates "$UPDATES" \
    --output "$OUT/report.json" | tee "$OUT/report.txt"
}

# 1. The teacher, and the pilot's copy after the updates of POINTS.
measure teacher "$TEACHER"
found=()
for point in $POINTS; do
  kept=$(checkpoint "$CONTROL" "$point")
  if [ -f "$kept" ]; then
    found+=("$kept")
  else
    echo "The pilot's copy after $point updates is not kept ($kept): it is not measured" >&2
  fi
done
if ((${#found[@]})); then
  measure pilot-copy "${found[@]}"
fi

# 2. The runs, each on the held-out games, and 3. what the copies made so
#    far say, while the next is made.
for name in $COPIES; do
  copy "$name" --imitation-teacher-targets "$name"
  found=()
  for point in $POINTS; do
    found+=("$(checkpoint "$RUNS/$name" "$point")")
  done
  measure "$name" "${found[@]}"
  report
done
echo "The runs' figures, the measurements and the readings are in $OUT: copy it back."
