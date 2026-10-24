#!/usr/bin/env bash
# The copy of the specialist with decks-01 of more decks as its teacher
# (experiments/2026-10-09-ppo-with-a-head-that-predicts.md, its section on the
# teacher chosen on 2026-10-20). Run from the repository root on the CUDA
# machine. The teacher is of format 14, so the run is made by the script of
# 05d27141 (experiments/commands/specialist-copy.sh there) in a checkout of
# that revision beside this one, OLD, made by tools/old_checkout.sh when it is
# not there; on this checkout's files (the held-out games, the pilot's copy,
# the teacher), into RUNS and OUT here. What else the old script reads
# (DEVICE, UPDATES, COPIES...) is passed on as it is. Then the fourth
# reading, fixed on 2026-10-20 before any run, with this checkout's code: each
# copy after UPDATES updates on decks-01's curve in the Blackwing mirror
# (experiments/commands/blackwing-mirror-curve.sh, into OUT/mirror/COPY), the
# copy and decks-01's checkpoints folded for the mirror at OLD where they are
# not yet (tools/fold_deck_lists.py); `choice` is read, `distribution` set
# beside it. About three hours on the CUDA machine.
#   bash experiments/commands/specialist-copy-decks-01.sh
set -euo pipefail
MAIN=$(pwd)
REVISION=05d27141
OLD=${OLD:-$MAIN/../ygo-$REVISION}
[ -d "$OLD" ] || bash tools/old_checkout.sh "$REVISION" "$OLD"
if [ "$(git -C "$OLD" rev-parse --short=8 HEAD)" != "$REVISION" ]; then
  echo "$OLD is not at $REVISION" >&2
  exit 1
fi
export GAMES=${GAMES:-$MAIN/analysis/search-leads}
export TEACHER=${TEACHER:-$MAIN/runs/more-decks/decks-01/checkpoints/model-02159.pt}
export CONTROL=${CONTROL:-$MAIN/runs/learner/once-few}
export CONTROL_OUT=${CONTROL_OUT:-$MAIN/analysis/learner-follow-up}
# The teacher's own run is not one the old report names: nothing is set
# beside it from there.
export LINE_OUT=${LINE_OUT:-$MAIN/analysis/more-decks}
export RUNS=${RUNS:-$MAIN/runs/specialist-copy-decks-01}
export OUT=${OUT:-$MAIN/analysis/specialist-copy-decks-01}
(cd "$OLD" && bash experiments/commands/specialist-copy.sh)

# 4. On decks-01's curve in the mirror.
python=(uv run python)
[ -z "${PYTHON:-}" ] || python=("$PYTHON")
MIRROR_POINTS=${MIRROR_POINTS:-0 120 240 360 480 600 720 840 960 1080 1200 1320 1440 1560 1680 1800 1920}
# fold CHECKPOINT DIRECTORY: the checkpoint folded for the mirror, once.
fold() {
  if [ ! -f "$2/$(basename "$1")" ]; then
    "${python[@]}" tools/fold_deck_lists.py "$1" --own blackwing --opponent blackwing \
      --old-checkout "$OLD" --output-dir "$2"
  fi
}
for point in $MIRROR_POINTS; do
  if ((point == 0)); then
    fold runs/more-decks/decks-01-copy/checkpoints/model-00239.pt \
      runs/more-decks/decks-01-copy/blackwing-mirror
  else
    fold "$(printf 'runs/more-decks/decks-01/checkpoints/model-%05d.pt' $((239 + point)))" \
      runs/more-decks/decks-01/blackwing-mirror
  fi
done
for name in ${COPIES:-choice distribution}; do
  copy=$(printf '%s/%s/checkpoints/model-%05d.pt' "$RUNS" "$name" $((${UPDATES:-200} - 1)))
  fold "$copy" "$RUNS/$name/blackwing-mirror"
  NAME=$name CHECKPOINTS="$RUNS/$name/blackwing-mirror/$(basename "$copy")" OUT="$OUT/mirror" \
    DEVICE=${DEVICE:-cuda} POINTS=$MIRROR_POINTS SUITE=eval_suites/blackwing-against-checkpoint.json \
    bash experiments/commands/blackwing-mirror-curve.sh
done
if [ -d "$OUT/mirror/choice" ]; then
  "${python[@]}" tools/blackwing_mirror_curve.py "$OUT/mirror/choice" --keeps 720 --little 120 \
    | tee "$OUT/mirror/reading.txt"
fi
