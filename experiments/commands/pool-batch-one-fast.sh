#!/usr/bin/env bash
# The first batch's fast arm (experiments/2026-11-02-expanded-pool.md, the card identities,
# step 2): the first batch again, from the same network on the same decks, weights and seed,
# with the cards' identity tables at SCALE times the learning rate, to HALF updates, and read
# against the first batch there (tools/pool_batch_compare.py). Run after the first batch is
# measured, from the repository root on the CUDA machine. With GO_ON=1, once its reading lets
# it go on: the fast arm to the end of the batch and measured as a batch is.
#   AT_ONCE=4 bash experiments/commands/pool-batch-one-fast.sh
set -euo pipefail
CONTROL_RUN=${CONTROL_RUN:-batch-one}
FAST_RUN=${FAST_RUN:-batch-one-fast}
CONTROL=${CONTROL:-analysis/expanded-pool/$CONTROL_RUN}
FAST=${FAST:-analysis/expanded-pool/$FAST_RUN}
AT_HALF=${AT_HALF:-analysis/expanded-pool/$CONTROL_RUN-4020}
HALF=${HALF:-4020}
SCALE=${SCALE:-10}
PYTHON=${PYTHON:-}
python=(uv run python)
[ -z "$PYTHON" ] || python=("$PYTHON")
for path in "$CONTROL/0/pairings.json" "$CONTROL/$HALF/pairings.json" "$CONTROL/$HALF/old.json"; do
  if [ ! -f "$path" ]; then
    echo "Missing $path: the first batch is measured first" >&2
    exit 1
  fi
done
# Both are read against the first batch's games at 0, the same frozen network's.
mkdir -p "$FAST/0" "$AT_HALF/0" "$AT_HALF/$HALF"
cp --update=none "$CONTROL"/0/*.json "$FAST/0/"
cp --update=none "$CONTROL"/0/*.json "$AT_HALF/0/"
cp --update=none "$CONTROL/$HALF/pairings.json" "$CONTROL/$HALF/old.json" "$AT_HALF/$HALF/"
if [ "${GO_ON:-0}" = 1 ]; then
  RUN=$FAST_RUN OUT=$FAST IDENTITY_SCALE=$SCALE bash experiments/commands/pool-batch.sh
  exit
fi
RUN=$FAST_RUN OUT=$FAST IDENTITY_SCALE=$SCALE UPDATES=$HALF POINTS=$HALF \
  bash experiments/commands/pool-batch.sh
# The first batch's mirrors, held-out Decks and pilots at the same point.
RUN=$CONTROL_RUN OUT=$AT_HALF UPDATES=$HALF POINTS=$HALF PARTS=measure \
  bash experiments/commands/pool-batch.sh
"${python[@]}" tools/pool_batch_compare.py "$FAST" "$AT_HALF" --point "$HALF" | tee "$FAST/compare.txt"
