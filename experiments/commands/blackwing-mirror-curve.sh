#!/usr/bin/env bash
# The Blackwing mirror along the training of the network of Blackwing alone
# (decks-01 of experiments/2026-10-14-more-decks.md, "The mirror along the
# training of Blackwing alone"). Run from the repository root. Each of
# CHECKPOINTS (comma-separated, of today's format) against decks-01 at its
# copy's end (point 0) and after each of POINTS updates of its PPO, every deal
# of the mirror (SUITE) from either seat, into OUT/NAME; then the report
# (tools/blackwing_mirror_curve.py): the share of the mirror they win against
# decks-01 at each point, and where they stand on its curve. decks-01's
# checkpoints are of format 14, played as tools/fold_deck_lists.py writes them
# for this mirror (FOLDED, FOLDED_COPY): they are good for nothing else. A
# finished measurement is not made again. On the development machine's
# processor 2,048 games take some 45 s: a checkpoint at the seventeen points,
# 25 minutes.
#   NAME=decks-13 CHECKPOINTS=runs/more-decks/decks-13/blackwing-mirror/model-05039.pt \
#     bash experiments/commands/blackwing-mirror-curve.sh
set -euo pipefail
NAME=${NAME:?NAME names what is measured}
CHECKPOINTS=${CHECKPOINTS:?CHECKPOINTS lists the checkpoints, comma-separated}
POINTS=${POINTS:-0 120 240 360 480 600 720 840 960 1080 1200 1320 1440 1560 1680 1800 1920}
FOLDED=${FOLDED:-runs/more-decks/decks-01/blackwing-mirror}
FOLDED_COPY=${FOLDED_COPY:-runs/more-decks/decks-01-copy/blackwing-mirror/model-00239.pt}
# decks-01's copy's updates: after P updates of its PPO its checkpoint is
# model-(COPIED + P - 1).
COPIED=${COPIED:-240}
DEVICE=${DEVICE:-cpu}
WORKERS=${WORKERS:-24}
THREADS=${THREADS:-20}
SLOTS=${SLOTS:-16}
SEED=${SEED:-2012}
SUITE=${SUITE:-eval_suites/blackwing-against-checkpoint.json}
OUT=${OUT:-analysis/blackwing-mirror-curve}
# The interpreter of a machine that has no uv: the benchmark's setup leaves one.
PYTHON=${PYTHON:-}
# Evaluations on a processor run six times slower with every thread.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

IFS=, read -r -a checkpoints <<< "$CHECKPOINTS"
# decks01 POINT: decks-01's checkpoint at that point, as folded.
decks01() {
  if (($1 == 0)); then
    echo "$FOLDED_COPY"
  else
    printf '%s/model-%05d.pt\n' "$FOLDED" $((COPIED + $1 - 1))
  fi
}
against=()
for point in $POINTS; do
  against+=("$(decks01 "$point")")
done
for path in "$SUITE" "${checkpoints[@]}" "${against[@]}"; do
  if [ ! -f "$path" ]; then
    echo "Missing $path (decks-01's are written by tools/fold_deck_lists.py)" >&2
    exit 1
  fi
done
pinned=$(git submodule status --recursive)
if grep -q '^[^ ]' <<< "$pinned"; then
  echo "A submodule is not at its pinned revision: git submodule update --init --recursive" >&2
  exit 1
fi
python=(uv run python)
[ -z "$PYTHON" ] || python=("$PYTHON")
cargo build --release
mkdir -p "$OUT/$NAME"

# play FILE CHECKPOINT HELD: CHECKPOINT plays the mirror, HELD in the other seat.
play() {
  if [ ! -f "$OUT/$NAME/$1.json" ]; then
    "${python[@]}" -m ygo_training evaluate-suite "$2" --device "$DEVICE" \
      --workers "$WORKERS" --pool-threads "$THREADS" --pool-slots "$SLOTS" --seed "$SEED" \
      --suite "$SUITE" --opponent-checkpoint "$3" \
      --output "$OUT/$NAME/$1.json.part" > /dev/null
    mv "$OUT/$NAME/$1.json.part" "$OUT/$NAME/$1.json"
    echo "$(date +%H:%M) $NAME: $1"
  fi
}
for checkpoint in "${checkpoints[@]}"; do
  stem=$(basename "$checkpoint" .pt)
  for point in $POINTS; do
    play "$stem.plays-$point" "$checkpoint" "$(decks01 "$point")"
    play "$stem.held-$point" "$(decks01 "$point")" "$checkpoint"
  done
done
"${python[@]}" tools/blackwing_mirror_curve.py "$OUT/$NAME" --output "$OUT/$NAME/report.json" \
  | tee "$OUT/$NAME/report.txt"
