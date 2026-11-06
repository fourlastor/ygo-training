#!/usr/bin/env bash
# Batch two of the expanded pool (experiments/2026-11-02-expanded-pool.md): from
# distilled's mean at 9,000 updates, fifty new game Decks with distilled's term and share.
# The old decks, the seventy-one and the sixty-two game Decks of the batch again, play their
# duels at 0.4 of the first batch's weights, and at each decision of an old deck's seat in a
# duel between two old decks the divergence from the start's answers, frozen, is added to
# PPO's loss. game-0065 and game-0093 hold a card that names a card, a question of 4,096
# answers, as game-0037 does among the old. A batch of experiments/commands/pool-batch.sh,
# whose settings stay open here for a smoke test; run from the repository root on the CUDA
# machine, some 18 hours for the run and 2 for the measure.
#   AT_ONCE=4 bash experiments/commands/pool-batch-two.sh
set -euo pipefail
NETWORK=${NETWORK:-runs/expanded-pool/replay-distilled-mean/9000.pt}
RUN=${RUN:-batch-two} OUT=${OUT:-analysis/expanded-pool/batch-two} NETWORK=$NETWORK \
  FIRST=${FIRST:-42120} UPDATES=${UPDATES:-6840} POINTS=${POINTS:-3420 6840} \
  NEW=$(printf "game-%04d," 64 $(seq 66 92) $(seq 94 115) | sed "s/,$//") \
  EARLIER=$(printf "game-%04d," $(seq 1 36) $(seq 38 63) | sed "s/,$//") \
  OLD_SCALE=0.4 DISTILL_REFERENCE=$NETWORK \
  PAIRINGS=${PAIRINGS:-eval_suites/pool-batch-two-pairings-checkpoint.json} \
  HELD_OUT=${HELD_OUT:-eval_suites/pool-batch-two-held-out-checkpoint.json} \
  PILOTS=${PILOTS:-eval_suites/pool-batch-two-against-pilots.json} \
  OLD=${OLD:-eval_suites/pool-batch-two-old-pairings-checkpoint.json} \
  bash "$(dirname "${BASH_SOURCE[0]}")/pool-batch.sh"
