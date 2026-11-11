#!/usr/bin/env bash
# Batch six of the expanded pool (experiments/2026-11-02-expanded-pool.md), in two arms from
# batch five's mean at 7,680 updates: ARM=pooled, that network as it is, and ARM=switched,
# the same network given the board as its cards with board_fusion at zero
# (tools/switch_board.py), which answers as it did. A hundred new game Decks, game-0269 to
# game-0369 but game-0335, at batch five's length, everything else as batch five: the old
# decks, the seventy-one and the 262 game Decks of the batches before, play their duels at
# 0.4 of the first batch's weights, and at each decision of an old deck's seat in a duel
# between two old decks the divergence from a frozen network's answers is added to PPO's
# loss, seventy-one-champions' in a duel between two of the seventy-one and otherwise batch
# five's mean, the same two for both arms, each answering in its own shape. The new decks'
# pairings are measured at 16 games each, read at 3 points; the mirrors deal by deal against
# seventy-one-champions' check and beside batch five's. game-0037, game-0065, game-0093,
# game-0167, game-0171 and game-0335 hold a card that names a card, a question of 4,096
# answers, and game-0233 Last Turn, whose forced battle inside an attack takes the engine
# down: they are left out. game-0379, held out, is left out of the held out since 2026-11-11:
# its Flame Ruler counts as two tributes, which the seat's card-by-card tribute selection
# does not count, and a selection with no answer fails the round. A held-out measure made on
# other decks than the suite's is set aside and played again.
#
# switched takes pooled's measures at 0 where pooled has made them, its start answering the
# same, but for the pilots', which it plays itself: the same games, one by one
# (tools/pool_arms_report.py tells). A batch of experiments/commands/pool-batch.sh, whose
# settings stay open here for a smoke test; run from the repository root on the CUDA
# machine, pooled first, some 20 hours for an arm's run and 4 to 5 for its measure. Then
# experiments/commands/pool-batch-six-arms.sh plays the two head to head.
#   ARM=pooled AT_ONCE=4 bash experiments/commands/pool-batch-six.sh
#   ARM=switched AT_ONCE=4 bash experiments/commands/pool-batch-six.sh
set -euo pipefail
ARM=${ARM:-}
case $ARM in
  pooled | switched) ;;
  *)
    echo "ARM is pooled or switched, not '$ARM'" >&2
    exit 1
    ;;
esac
START=${START:-runs/expanded-pool/batch-five-mean/7680.pt}
HELD_OUT=${HELD_OUT:-eval_suites/pool-batch-six-held-out-checkpoint.json}
RUNS=${RUNS:-runs/expanded-pool}
ANALYSIS=${ANALYSIS:-analysis/expanded-pool}
RUN=batch-six-$ARM
OUT=${OUT:-$ANALYSIS/$RUN}
python=(uv run python)
[ -z "${PYTHON:-}" ] || python=("$PYTHON")
if [ ! -f "$START" ]; then
  echo "Missing $START" >&2
  exit 1
fi
board=
network=$START
if [ "$ARM" = switched ]; then
  board=individual-tokens
  network=$RUNS/$RUN.begun.pt
  if [ ! -f "$network" ]; then
    mkdir -p "$RUNS"
    "${python[@]}" tools/switch_board.py "$START" --output "$network"
  fi
  # pooled's measures at 0, played by the same answers, but the pilots'.
  for part in pairings held-out; do
    made=$ANALYSIS/batch-six-pooled/0/$part.json
    if [ -f "$made" ] && [ ! -f "$OUT/0/$part.json" ]; then
      mkdir -p "$OUT/0"
      cp "$made" "$OUT/0/$part.json"
    fi
  done
fi
for made in "$OUT"/*/held-out.json; do
  [ -f "$made" ] || continue
  if ! "${python[@]}" - "$made" "$HELD_OUT" << 'SCRIPT'; then
import json
import sys

made, suite = sys.argv[1:3]
(record,) = json.load(open(made, encoding="utf8"))["records"]
asked = {case["name"] for case in json.load(open(suite, encoding="utf8"))["cases"]}
sys.exit(0 if set(record["evaluation_suite"]) == asked else 1)
SCRIPT
    echo "$made was played on other decks than $HELD_OUT: set aside, to be played again" >&2
    mv "$made" "$made.other-decks"
  fi
done
RUN=$RUN RUNS=$RUNS OUT=$OUT NETWORK=$network BOARD=$board \
  FIRST=${FIRST:-71280} UPDATES=${UPDATES:-7680} POINTS=${POINTS:-3840 7680} \
  OLD_POINTS=${OLD_POINTS:-7680} \
  NEW=$(printf "game-%04d," $(seq 269 334) $(seq 336 369) | sed "s/,$//") \
  EARLIER=$(printf "game-%04d," $(seq 1 36) $(seq 38 64) $(seq 66 92) $(seq 94 166) \
    $(seq 168 170) $(seq 172 232) $(seq 234 268) | sed "s/,$//") \
  OLD_SCALE=0.4 DISTILL_REFERENCE=$START \
  DISTILL_SECOND=${DISTILL_SECOND:-runs/seventy-one-champions/ppo-mean/21900.pt} \
  PAIRINGS=${PAIRINGS:-eval_suites/pool-batch-six-pairings-checkpoint.json} \
  HELD_OUT=$HELD_OUT \
  PILOTS=${PILOTS:-eval_suites/pool-batch-six-against-pilots.json} \
  OLD=${OLD:-eval_suites/pool-batch-six-old-pairings-checkpoint.json} \
  MIRRORS_REFERENCE=${MIRRORS_REFERENCE:-analysis/seventy-one-champions/checks/20160} \
  MIRRORS_BESIDE=${MIRRORS_BESIDE:-analysis/expanded-pool/batch-five/7680/mirrors} \
  LEARNS=0.03 DOES_NOT=0.01 \
  bash "$(dirname "${BASH_SOURCE[0]}")/pool-batch.sh"
