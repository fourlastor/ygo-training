#!/usr/bin/env bash
# Self-play, a second seed, and what is left for an exploiter after 1,920
# updates (experiments/2026-10-13-self-play-second-seed.md). Run from the
# repository root on the CUDA machine, with nothing else running there. One
# after the other, each by experiments/commands/self-play.sh:
#
#   1. A second pair of arms, with-itself and pilots-alone of the training
#      seed SEED, 960 updates on from the start of the first pair, the two
#      at once through MPS, and their measurements. About six hours.
#   2. Two exploiters of each arm's checkpoint after 1,920 updates, of the
#      first pair, three at once, and their measurements. About four hours.
#
# The second is made whatever came of the first, and what failed is told at
# the end. Started again, each takes its runs on from where they stand and
# skips what is there.
#
# What is copied back: analysis/self-play-SEED and analysis/self-play-further.
set -uo pipefail
SEED=${SEED:-2013}
failed=()

echo "== 1. The arms of training seed $SEED"
AT_ONCE=2 PARTS=arms ARM=960 SINCE= ARM_SEED=$SEED \
  RUNS=runs/self-play-$SEED OUT=analysis/self-play-$SEED \
  bash experiments/commands/self-play.sh || failed+=("the arms of seed $SEED")

echo "== 2. The exploiters of the first pair's checkpoints after 1,920 updates"
AT_ONCE=3 PARTS=exploiters ARM=1920 SINCE=960 ARM_SEED= \
  RUNS=runs/self-play OUT=analysis/self-play-further \
  bash experiments/commands/self-play.sh || failed+=("the exploiters")

if ((${#failed[@]})); then
  printf 'Not to its end: %s\n' "${failed[@]}" >&2
  exit 1
fi
echo "Both are made: copy back analysis/self-play-$SEED and analysis/self-play-further."
