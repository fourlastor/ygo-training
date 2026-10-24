#!/usr/bin/env bash
# What this card would take for wider calls of the model and a wider pass:
# whether many duels in flight, an update every so many of their decisions,
# could be much faster here, and the trainer's graphs at other members widths.
# A measure of speed, not an experiment: nothing is trained. About 25 minutes
# on the CUDA machine. Run from the repository root when nothing else uses the
# card: with another run beside them the times are not this card's.
#
# Into OUT, with the copy's last checkpoint of thirteen decks (CHECKPOINT):
# calls-WIDE.txt         tools/call_timing.py on an update of WIDE duels,
#                        THREADS threads each with its share in flight at
#                        once, half of them against itself: its calls timed
#                        by their size, to about a thousand decisions. An
#                        update holds 1,024 duels at most, the run of seeds
#                        each update owns (ygo_training/schedules.py).
# calls-members-N.txt    the same on an update of DUELS duels as a run plays
#                        them, the trainer's graphs fixed at N members of an
#                        answer, for N of 2 (the trainer's), 4 and 8.
# pass.txt               tools/update_timing.py: a pass over an update's
#                        decisions by the size of its minibatches (SIZES).
# pass-tf32.txt          the same with float32 matrix products in TF32
#                        (--matmul-precision high), and how far it moves an
#                        answer; pass-fused-adam.txt with Adam's step fused
#                        into one kernel. Two cheap changes of the numbers.
#
# PYTHON names an interpreter to use where uv is not there.
set -euo pipefail
CHECKPOINT=${CHECKPOINT:-runs/thirteen-decks/copy/checkpoints/model-03119.pt}
OUT=${OUT:-analysis/wide-calls}
DEVICE=${DEVICE:-cuda}
THREADS=${THREADS:-20}
WIDE=${WIDE:-1020}
DUELS=${DUELS:-520}
REPEATS=${REPEATS:-50}
SIZES=${SIZES:-256,512,1024,2048,4096}
if [ ! -f "$CHECKPOINT" ]; then
  echo "Missing $CHECKPOINT" >&2
  exit 1
fi
if ((WIDE % THREADS)); then
  echo "WIDE is duels THREADS share evenly, not $WIDE among $THREADS" >&2
  exit 1
fi
if [[ "$DEVICE" == cuda* ]] && [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
  echo "Something else uses the card (nvidia-smi): its times would not be the card's" >&2
  exit 1
fi
pinned=$(git submodule status --recursive)
if grep -q '^[^ ]' <<< "$pinned"; then
  echo "A submodule is not at its pinned revision: git submodule update --init --recursive" >&2
  exit 1
fi
python=(uv run python)
[ -z "${PYTHON:-}" ] || python=("$PYTHON")
cargo build --release
mkdir -p "$OUT"
common=(--device "$DEVICE" --pool-threads "$THREADS" --repeats "$REPEATS")
echo "$(date '+%H:%M') calls of $WIDE duels in flight"
"${python[@]}" tools/call_timing.py "$CHECKPOINT" "${common[@]}" --pool-slots $((WIDE / THREADS)) \
  --duels "$WIDE" --itself $((WIDE / 2)) --output "$OUT/calls-$WIDE.txt" > /dev/null
for members in 2 4 8; do
  echo "$(date '+%H:%M') calls of $DUELS duels, the graphs at $members members"
  "${python[@]}" tools/call_timing.py "$CHECKPOINT" "${common[@]}" --duels "$DUELS" \
    --itself $((DUELS / 2)) --members-width "$members" --output "$OUT/calls-members-$members.txt" > /dev/null
done
passes=(--device "$DEVICE" --pool-threads "$THREADS" --duels "$DUELS" --sizes "$SIZES")
echo "$(date '+%H:%M') a pass by the size of its minibatches"
"${python[@]}" tools/update_timing.py "$CHECKPOINT" "${passes[@]}" --output "$OUT/pass.txt" > /dev/null
echo "$(date '+%H:%M') the same, in TF32 and with Adam fused"
"${python[@]}" tools/update_timing.py "$CHECKPOINT" "${passes[@]}" --matmul-precision high \
  --output "$OUT/pass-tf32.txt" > /dev/null
"${python[@]}" tools/update_timing.py "$CHECKPOINT" "${passes[@]}" --fused-adam \
  --output "$OUT/pass-fused-adam.txt" > /dev/null
echo "$(date '+%H:%M') measured: $OUT"
