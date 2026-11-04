#!/usr/bin/env bash
# The batch again in two arms (experiments/2026-11-02-expanded-pool.md): from
# seventy-one-champions with the sixty-two Decks of the first batch test and the first batch,
# a batch of experiments/commands/pool-batch.sh for each ARM, `quarter` first. `quarter` has
# the first batch's weights; `distilled` has the old decks' duels at 0.4 of them, and at each
# decision of an old deck's seat in a duel between two old decks the divergence from
# seventy-one-champions' answers, frozen, added to PPO's loss. The cards' identity tables are
# as the fast arm reads (FAST): ten times the learning rate where it learns faster and the old
# decks are no worse, the same rate otherwise. `distilled` is read against `quarter`'s games at
# 0, and at the end its new decks against `quarter`'s (tools/pool_batch_compare.py). Run from
# the repository root on the CUDA machine, some 23 hours an arm.
#   ARM=quarter AT_ONCE=4 bash experiments/commands/pool-replay.sh
#   ARM=distilled AT_ONCE=4 bash experiments/commands/pool-replay.sh
set -euo pipefail
ARM=${ARM:?ARM names quarter or distilled}
NETWORK=${NETWORK:-runs/seventy-one-champions/ppo-mean/21900.pt}
FAST=${FAST:-analysis/expanded-pool/batch-one-fast/compare.json}
UPDATES=${UPDATES:-9000}
POINTS=${POINTS:-4500 9000}
PYTHON=${PYTHON:-}
python=(uv run python)
[ -z "$PYTHON" ] || python=("$PYTHON")
case $ARM in
  quarter) scale=1 reference= ;;
  distilled) scale=0.4 reference=$NETWORK ;;
  *)
    echo "ARM names quarter or distilled, not $ARM" >&2
    exit 1
    ;;
esac
if [ ! -f "$FAST" ]; then
  echo "Missing $FAST: the fast arm is read first" >&2
  exit 1
fi
identity=$("${python[@]}" - "$FAST" << 'SCRIPT'
import json
import sys

read = json.load(open(sys.argv[1], encoding="utf8"))
print(10 if read["faster"] == "faster" and read["old decks"] == "no worse" else 1)
SCRIPT
)
echo "The fast arm read: identity tables at $identity times the learning rate"
QUARTER=analysis/expanded-pool/replay-quarter
OUT=analysis/expanded-pool/replay-$ARM
if [ "$ARM" = distilled ]; then
  if [ ! -f "$QUARTER/0/pairings.json" ]; then
    echo "Missing $QUARTER/0: quarter is measured first" >&2
    exit 1
  fi
  # Both read against the same games at 0, seventy-one-champions'.
  mkdir -p "$OUT/0"
  cp --update=none "$QUARTER"/0/*.json "$OUT/0/"
fi
RUN=replay-$ARM OUT=$OUT NETWORK=$NETWORK FIRST=33120 UPDATES=$UPDATES POINTS=$POINTS \
  NEW=$(printf "game-%04d," $(seq 1 36) $(seq 38 63) | sed "s/,$//") EARLIER= \
  OLD_SCALE=$scale DISTILL_REFERENCE=$reference IDENTITY_SCALE=$identity \
  PAIRINGS=eval_suites/pool-batch-replay-pairings-checkpoint.json \
  HELD_OUT=eval_suites/pool-batch-replay-held-out-checkpoint.json \
  PILOTS=eval_suites/pool-batch-replay-against-pilots.json \
  OLD=eval_suites/pool-batch-replay-old-pairings-checkpoint.json \
  bash experiments/commands/pool-batch.sh
if [ "$ARM" = distilled ]; then
  "${python[@]}" tools/pool_batch_compare.py "$OUT" "$QUARTER" --point "${POINTS##* }" |
    tee "$OUT/compare.txt"
fi
