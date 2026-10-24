#!/usr/bin/env bash
# A command on a rented machine, with what the benchmark's setup leaves
# there. Run on the machine itself, from anywhere:
#
#   bash experiments/commands/rented-setup.sh bash experiments/commands/more-decks.sh
#
# It asks the driver for the card first, since a machine can name a card
# that no program can compute on and a run would then go on the processor
# without a word (ON_THE_PROCESSOR=1 goes on all the same). It makes the
# benchmark's setup where that is missing (`ONLY=setup bash
# tools/benchmark.sh`: the toolchain, the build and a Python environment in
# FOLDER), and runs the command with what the setup left in its
# environment: the interpreter (PYTHON), the device (DEVICE), the machine's
# share of its processor's threads (THREADS), as many duels a thread as keep
# FLIGHT in flight, an even count (SLOTS), and the way to cargo. A THREADS
# or SLOTS that is named stays.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
if ! (($#)); then
  echo "Name the command: bash experiments/commands/rented-setup.sh COMMAND..." >&2
  exit 2
fi
FOLDER=${FOLDER:-benchmark}
FLIGHT=${FLIGHT:-320}
named=${THREADS:-}
slots=${SLOTS:-}
if [ -z "${ON_THE_PROCESSOR:-}" ]; then
  started=$(python3 -c 'import ctypes; print(ctypes.CDLL("libcuda.so.1").cuInit(0))' 2> /dev/null || echo none)
  if [ "$started" != 0 ]; then
    echo "No program can compute on a card here (CUDA's start: $started)." \
      "ON_THE_PROCESSOR=1 goes on all the same" >&2
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
if [ -z "$slots" ]; then
  slots=$(((FLIGHT + THREADS - 1) / THREADS))
  slots=$((slots + slots % 2))
fi
SLOTS=$slots
export PYTHON DEVICE THREADS SLOTS
echo "On the $DEVICE, a pool of $THREADS threads with $SLOTS duels each: $*"
exec "$@"
