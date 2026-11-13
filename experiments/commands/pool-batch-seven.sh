#!/usr/bin/env bash
# Batch seven of the expanded pool (experiments/2026-11-02-expanded-pool.md), from batch
# six's pooled, its mean at 7,680 updates, the arm the pool goes on from: every usable list
# not yet taken, 113 new game Decks, game-0370 to game-0484 but game-0379 and game-0397,
# with none held out, at half batch six's length, 3,840 updates. Everything else as pooled:
# the old decks, the seventy-one and the 362 game Decks of the batches before, play their
# duels at 0.4 of the first batch's weights, and at each decision of an old deck's seat in
# a duel between two old decks the divergence from a frozen network's answers is added to
# PPO's loss, seventy-one-champions' in a duel between two of the seventy-one and otherwise
# the start's. The new decks' pairings are measured at 16 games each, read at 2 points; the
# mirrors deal by deal against seventy-one-champions' check and beside pooled's.
# game-0037, game-0065, game-0093, game-0167, game-0171, game-0335 and game-0397 hold a
# card that names a card, a question of 4,096 answers, and game-0233 Last Turn, whose
# forced battle inside an attack takes the engine down: they are left out. game-0379 waits
# for the fix of the seat's tribute selection, which counts a tribute a card where its
# Flame Ruler counts as two.
#
# A batch of experiments/commands/pool-batch.sh, whose settings stay open here for a smoke
# test; run from the repository root on the CUDA machine, some 11 hours for the run and 7
# for its measure.
#   AT_ONCE=4 bash experiments/commands/pool-batch-seven.sh
set -euo pipefail
START=${START:-runs/expanded-pool/batch-six-pooled-mean/7680.pt}
RUNS=${RUNS:-runs/expanded-pool}
ANALYSIS=${ANALYSIS:-analysis/expanded-pool}
RUN=batch-seven
OUT=${OUT:-$ANALYSIS/$RUN}
if [ ! -f "$START" ]; then
  echo "Missing $START" >&2
  exit 1
fi
RUN=$RUN RUNS=$RUNS OUT=$OUT NETWORK=$START \
  FIRST=${FIRST:-78960} UPDATES=${UPDATES:-3840} POINTS=${POINTS:-1920 3840} \
  OLD_POINTS=${OLD_POINTS:-3840} \
  NEW=$(printf "game-%04d," $(seq 370 378) $(seq 380 396) $(seq 398 484) | sed "s/,$//") \
  EARLIER=$(printf "game-%04d," $(seq 1 36) $(seq 38 64) $(seq 66 92) $(seq 94 166) \
    $(seq 168 170) $(seq 172 232) $(seq 234 334) $(seq 336 369) | sed "s/,$//") \
  OLD_SCALE=0.4 DISTILL_REFERENCE=$START \
  DISTILL_SECOND=${DISTILL_SECOND:-runs/seventy-one-champions/ppo-mean/21900.pt} \
  PAIRINGS=${PAIRINGS:-eval_suites/pool-batch-seven-pairings-checkpoint.json} \
  HELD_OUT= \
  PILOTS=${PILOTS:-eval_suites/pool-batch-seven-against-pilots.json} \
  OLD=${OLD:-eval_suites/pool-batch-seven-old-pairings-checkpoint.json} \
  MIRRORS_REFERENCE=${MIRRORS_REFERENCE:-analysis/seventy-one-champions/checks/20160} \
  MIRRORS_BESIDE=${MIRRORS_BESIDE:-analysis/expanded-pool/batch-six-pooled/7680/mirrors} \
  LEARNS=0.02 DOES_NOT=0.01 \
  bash "$(dirname "${BASH_SOURCE[0]}")/pool-batch.sh"
