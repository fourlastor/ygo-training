#!/usr/bin/env bash
# Batch four of the expanded pool (experiments/2026-11-02-expanded-pool.md): from batch
# three's mean at 7,200 updates, fifty new game Decks with distilled's term and share. The
# old decks, the seventy-one and the 162 game Decks of the batches before, play their duels at
# 0.4 of the first batch's weights, and at each decision of an old deck's seat in a duel
# between two old decks the divergence from a frozen network's answers is added to PPO's
# loss: in a duel between two of the seventy-one from seventy-one-champions', the network
# distilled began from and was held to, otherwise from the start's. The old decks are
# measured at the end alone; the mirrors are read deal by deal against seventy-one-champions'
# check at 20,160 updates and told beside batch three's. game-0037, game-0065 and game-0093
# among the old decks and game-0167 and game-0171, left out, hold a card that names a card, a
# question of 4,096 answers. A batch of experiments/commands/pool-batch.sh, whose settings
# stay open here for a smoke test; run from the repository root on the CUDA machine, some 20
# hours for the run and 3 for the measure.
#   AT_ONCE=4 bash experiments/commands/pool-batch-four.sh
set -euo pipefail
NETWORK=${NETWORK:-runs/expanded-pool/batch-three-mean/7200.pt}
RUN=${RUN:-batch-four} OUT=${OUT:-analysis/expanded-pool/batch-four} NETWORK=$NETWORK \
  FIRST=${FIRST:-56160} UPDATES=${UPDATES:-7440} POINTS=${POINTS:-3720 7440} \
  OLD_POINTS=${OLD_POINTS:-7440} \
  NEW=$(printf "game-%04d," 166 $(seq 168 170) $(seq 172 217) | sed "s/,$//") \
  EARLIER=$(printf "game-%04d," $(seq 1 36) $(seq 38 64) $(seq 66 92) $(seq 94 165) | sed "s/,$//") \
  OLD_SCALE=0.4 DISTILL_REFERENCE=$NETWORK \
  DISTILL_SECOND=${DISTILL_SECOND:-runs/seventy-one-champions/ppo-mean/21900.pt} \
  PAIRINGS=${PAIRINGS:-eval_suites/pool-batch-four-pairings-checkpoint.json} \
  HELD_OUT=${HELD_OUT:-eval_suites/pool-batch-four-held-out-checkpoint.json} \
  PILOTS=${PILOTS:-eval_suites/pool-batch-four-against-pilots.json} \
  OLD=${OLD:-eval_suites/pool-batch-four-old-pairings-checkpoint.json} \
  MIRRORS_REFERENCE=${MIRRORS_REFERENCE:-analysis/seventy-one-champions/checks/20160} \
  MIRRORS_BESIDE=${MIRRORS_BESIDE:-analysis/expanded-pool/batch-three/7200/mirrors} \
  bash "$(dirname "${BASH_SOURCE[0]}")/pool-batch.sh"
