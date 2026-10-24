#!/usr/bin/env bash
# How many runs of the trainer a rented machine carries at once: run on the
# machine itself, from the repository root, with a checkpoint sent there.
# From this checkout that is:
#
#     export RENTED=root@HOST:PORT
#     bash tools/rented.sh send
#     bash tools/rented.sh put runs/model-02279-rate.pt
#     bash tools/rented.sh start capacity bash experiments/commands/rented-capacity.sh
#     bash tools/rented.sh watch capacity
#     INTO=analysis/rented bash tools/rented.sh fetch analysis/rented-capacity
#
# It makes the benchmark's setup where it is missing (tools/benchmark.sh
# with ONLY=setup: the toolchain, the build and a Python environment, about
# ten minutes the first time), and then experiments/commands/side-by-side.sh
# with what the machine has: its share of the processor's threads for a
# run's pool (THREADS names another count), as many duels a thread as keep
# FLIGHT duels in flight (320, the CUDA machine's; an even count, which two
# groups share out), COUNTS runs at once, and the runs through MPS as well
# where its daemon is on the PATH (PARTS=plain goes without). A measure of
# speed, not an experiment: nothing it writes is a result.
#
# The runs are made for each of WAYS, as experiments/commands/call-graphs.sh
# makes them on the CUDA machine:
#
# one-group          every call as it is, every duel together.
# graphs             calls of up to GRAPHS decisions from CUDA graphs: half
#                    of the duels in flight unless named.
# graphs-two-groups  the same with a thread's duels in two groups while
#                    their rounds hold more than OVER decisions.
#
# The reports are in OUT: a folder for each way, and plain.txt and mps.txt,
# which set the ways beside each other.
#
# A machine can name a card that no program can compute on: the trainer
# would then go on the processor without a word, and this would measure
# that. It asks the driver first, before anything is set up, and stops where
# the card does not start; ON_THE_PROCESSOR=1 goes on all the same.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
FOLDER=${FOLDER:-benchmark}
CHECKPOINT=${CHECKPOINT:-runs/model-02279-rate.pt}
COUNTS=${COUNTS:-1 2 3 4 6}
FLIGHT=${FLIGHT:-320}
WORKERS=${WORKERS:-24}
WAYS=${WAYS:-one-group graphs}
OVER=${OVER:-64}
OUT=${OUT:-analysis/rented-capacity}
RUNS=${RUNS:-runs/rented-capacity}
named=${THREADS:-}
for way in $WAYS; do
  case "$way" in
    one-group | graphs | graphs-two-groups) ;;
    *)
      echo "WAYS names one-group, graphs and graphs-two-groups, not $way" >&2
      exit 1
      ;;
  esac
done
if [ ! -f "$CHECKPOINT" ]; then
  echo "No checkpoint at $CHECKPOINT: tools/rented.sh put sends one" >&2
  exit 1
fi
if [ -z "${ON_THE_PROCESSOR:-}" ]; then
  started=$(python3 -c 'import ctypes; print(ctypes.CDLL("libcuda.so.1").cuInit(0))' 2> /dev/null || echo none)
  if [ "$started" != 0 ]; then
    echo "No program can compute on a card here (CUDA's start: $started): this would measure the" \
      "processor. ON_THE_PROCESSOR=1 goes on all the same" >&2
    exit 1
  fi
fi
ONLY=setup FOLDER=$FOLDER bash tools/benchmark.sh
case "$FOLDER" in
  /*) ;;
  *) FOLDER=$PWD/$FOLDER ;;
esac
# What the setup left: PYTHON, DEVICE, the machine's THREADS, the way to cargo.
# shellcheck disable=SC1091
. "$FOLDER/setup.sh"
THREADS=${named:-$THREADS}
if [ "$DEVICE" != cuda ] && [ -z "${ON_THE_PROCESSOR:-}" ]; then
  echo "PyTorch computes on the $DEVICE here, not on a card: ON_THE_PROCESSOR=1 goes on all the same" >&2
  exit 1
fi
SLOTS=$(((FLIGHT + THREADS - 1) / THREADS))
SLOTS=$((SLOTS + SLOTS % 2))
GRAPHS=${GRAPHS:-$((THREADS * SLOTS / 2))}
if [ -z "${PARTS:-}" ]; then
  PARTS=plain
  if command -v nvidia-cuda-mps-control > /dev/null; then
    PARTS="plain mps"
  fi
fi
echo "A run's pool: $THREADS threads of $SLOTS duels. Runs at once: $COUNTS. Parts: $PARTS." \
  "Ways: $WAYS, the graphs for calls of up to $GRAPHS decisions"
export PYTHON DEVICE THREADS SLOTS COUNTS PARTS CHECKPOINT WORKERS
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
  esac
done
for part in $PARTS; do
  reports=()
  for way in one-group graphs graphs-two-groups; do
    [ ! -f "$OUT/$way/$part.json" ] || reports+=("$OUT/$way/$part.json")
  done
  if [ "${#reports[@]}" -ge 2 ]; then
    echo "== $part: the ways beside each other"
    "$PYTHON" tools/side_by_side.py "$CHECKPOINT" --beside "${reports[@]}" | tee "$OUT/$part.txt"
  fi
done
echo "The reports are in $OUT"
