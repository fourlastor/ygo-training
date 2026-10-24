#!/usr/bin/env bash
# What answering the small calls of an update's games from CUDA graphs does
# on this machine, with every duel taken forward together and with a
# thread's duels in two groups; and a step of a pass as a graph. A measure of
# speed, not an experiment: nothing it writes is a result. Run from the
# repository root, when nothing else runs there.
#
# First tools/call_timing.py: a call of the model taken apart, the trainer's
# own way of answering calls at fixed sizes run on the batches of an update
# (it holds each graph against the model's own answer), and a step of a pass
# recorded whole as a graph. Three minutes. Where the trainer's way fails
# there, nothing more is made.
#
# Then experiments/commands/side-by-side.sh for each of WAYS: the same short
# run of PPO alone and COUNTS times at once, as processes have the card and
# through NVIDIA's MPS (PARTS). About a quarter of an hour a way.
#
# one-group          every call as it is, every duel together: what the
#                    trainer does. Made only where no report of it is in
#                    OUT/one-group, nor in BASE, which an earlier run left.
# graphs             calls of up to GRAPHS decisions from CUDA graphs
#                    (--inference-graphs).
# graphs-two-groups  the same with a thread's duels in two groups while
#                    their rounds hold more than OVER decisions
#                    (--pool-groups 2 --pool-groups-over OVER): the engines
#                    of one group run while the model answers the other.
# two-groups         the two groups with every call as it is, while their
#                    rounds hold more than OVER_PLAIN decisions. Not made
#                    unless named.
#
# The reports are in OUT: call-timing.txt, a folder for each way, and
# plain.txt and mps.txt, which set the ways beside each other and say whether
# the run alone played the same games every way. Answered at fixed sizes a
# decision's scores differ in their last digits, so that the games can be
# other games from some update on.
set -euo pipefail
WAYS=${WAYS:-graphs graphs-two-groups}
COUNTS=${COUNTS:-1 3}
UPDATES=${UPDATES:-7}
PARTS=${PARTS:-plain mps}
GRAPHS=${GRAPHS:-160}
OVER=${OVER:-64}
OVER_PLAIN=${OVER_PLAIN:-176}
ITSELF=${ITSELF:-260}
TIMING=${TIMING:-yes}
OUT=${OUT:-analysis/call-graphs}
RUNS=${RUNS:-runs/call-graphs}
BASE=${BASE:-analysis/two-groups/one-group}
CHECKPOINT=${CHECKPOINT:-runs/self-play/with-itself/checkpoints/model-03239.pt}
DEVICE=${DEVICE:-cuda}
THREADS=${THREADS:-20}
SLOTS=${SLOTS:-16}
DUELS=${DUELS:-520}
PYTHON=${PYTHON:-}
export COUNTS UPDATES PARTS CHECKPOINT PYTHON ITSELF DEVICE THREADS SLOTS DUELS

for way in $WAYS; do
  case "$way" in
    one-group | graphs | graphs-two-groups | two-groups) ;;
    *)
      echo "WAYS names one-group, graphs, graphs-two-groups and two-groups, not $way" >&2
      exit 1
      ;;
  esac
done
if [ ! -f "$CHECKPOINT" ]; then
  echo "No checkpoint at $CHECKPOINT" >&2
  exit 1
fi
mkdir -p "$OUT"
python=(uv run python)
[ -z "$PYTHON" ] || python=("$PYTHON")
cargo build --release

if [ "$TIMING" = yes ]; then
  echo "== a call of the model, the trainer's own way at fixed sizes, and a step of a pass as a graph"
  "${python[@]}" tools/call_timing.py "$CHECKPOINT" --device "$DEVICE" --itself "$ITSELF" \
    --duels "$DUELS" --pool-threads "$THREADS" --pool-slots "$SLOTS" --output "$OUT/call-timing.txt"
  if grep -q "IT FAILED" "$OUT/call-timing.txt"; then
    echo "The trainer's own way at fixed sizes failed: $OUT/call-timing.txt says how. Nothing more is made." >&2
    exit 1
  fi
fi

# The trainer as it is: made where no report of it is at hand.
if [[ " $WAYS " != *" one-group "* ]]; then
  for part in $PARTS; do
    if [ ! -f "$OUT/one-group/$part.json" ] && [ ! -f "$BASE/$part.json" ]; then
      WAYS="one-group $WAYS"
      break
    fi
  done
fi

for way in $WAYS; do
  echo "== $way"
  case "$way" in
    one-group)
      GRAPHS=0 POOL_GROUPS=1 POOL_GROUPS_OVER=0 OUT=$OUT/$way RUNS=$RUNS/$way \
        bash experiments/commands/side-by-side.sh
      ;;
    graphs)
      GRAPHS=$GRAPHS POOL_GROUPS=1 POOL_GROUPS_OVER=0 OUT=$OUT/$way RUNS=$RUNS/$way \
        bash experiments/commands/side-by-side.sh
      ;;
    graphs-two-groups)
      GRAPHS=$GRAPHS POOL_GROUPS=2 POOL_GROUPS_OVER=$OVER OUT=$OUT/$way RUNS=$RUNS/$way \
        bash experiments/commands/side-by-side.sh
      ;;
    two-groups)
      GRAPHS=0 POOL_GROUPS=2 POOL_GROUPS_OVER=$OVER_PLAIN OUT=$OUT/$way RUNS=$RUNS/$way \
        bash experiments/commands/side-by-side.sh
      ;;
  esac
done

for part in $PARTS; do
  # The trainer as it is first: this run's, or the one an earlier run left.
  reports=()
  if [ -f "$OUT/one-group/$part.json" ]; then
    reports+=("$OUT/one-group/$part.json")
  elif [ -f "$BASE/$part.json" ]; then
    reports+=("$BASE/$part.json")
  fi
  for way in graphs graphs-two-groups two-groups; do
    [ ! -f "$OUT/$way/$part.json" ] || reports+=("$OUT/$way/$part.json")
  done
  if [ "${#reports[@]}" -lt 2 ]; then
    echo "Fewer than two reports of $part to set beside each other" >&2
    continue
  fi
  echo "== $part: the ways beside each other"
  "${python[@]}" tools/side_by_side.py "$CHECKPOINT" --beside "${reports[@]}" | tee "$OUT/$part.txt"
done
echo "The reports are in $OUT"
