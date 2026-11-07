#!/usr/bin/env bash
# Batch three of the expanded pool (experiments/2026-11-02-expanded-pool.md): from batch
# two's mean at 6,840 updates, fifty new game Decks with distilled's term and share. The old
# decks, the seventy-one and the 112 game Decks of the batches before, play their duels at
# 0.4 of the first batch's weights, and at each decision of an old deck's seat in a duel
# between two old decks the divergence from the start's answers, frozen, is added to PPO's
# loss. The old decks are measured at the end alone; the mirrors are read deal by deal
# against seventy-one-champions' check at 20,160 updates and told beside batch two's.
# game-0037, game-0065 and game-0093 among the old decks and game-0167 and game-0171 among
# the held out hold a card that names a card, a question of 4,096 answers. A batch of
# experiments/commands/pool-batch.sh, whose settings stay open here for a smoke test; run
# from the repository root on the CUDA machine, some 19 hours for the run and 3.5 for the
# measure.
#   AT_ONCE=4 bash experiments/commands/pool-batch-three.sh
set -euo pipefail
NETWORK=${NETWORK:-runs/expanded-pool/batch-two-mean/6840.pt}
RUN=${RUN:-batch-three} OUT=${OUT:-analysis/expanded-pool/batch-three} NETWORK=$NETWORK \
  FIRST=${FIRST:-48960} UPDATES=${UPDATES:-7200} POINTS=${POINTS:-3600 7200} \
  OLD_POINTS=${OLD_POINTS:-7200} \
  NEW=$(printf "game-%04d," $(seq 116 165) | sed "s/,$//") \
  EARLIER=$(printf "game-%04d," $(seq 1 36) $(seq 38 64) $(seq 66 92) $(seq 94 115) | sed "s/,$//") \
  OLD_SCALE=0.4 DISTILL_REFERENCE=$NETWORK \
  PAIRINGS=${PAIRINGS:-eval_suites/pool-batch-three-pairings-checkpoint.json} \
  HELD_OUT=${HELD_OUT:-eval_suites/pool-batch-three-held-out-checkpoint.json} \
  PILOTS=${PILOTS:-eval_suites/pool-batch-three-against-pilots.json} \
  OLD=${OLD:-eval_suites/pool-batch-three-old-pairings-checkpoint.json} \
  MIRRORS_REFERENCE=${MIRRORS_REFERENCE:-analysis/seventy-one-champions/checks/20160} \
  MIRRORS_BESIDE=${MIRRORS_BESIDE:-analysis/expanded-pool/batch-two/6840/mirrors} \
  bash "$(dirname "${BASH_SOURCE[0]}")/pool-batch.sh"
