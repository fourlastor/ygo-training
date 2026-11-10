#!/usr/bin/env bash
# Batch six's two arms head to head (experiments/2026-11-02-expanded-pool.md), once both are
# made by experiments/commands/pool-batch-six.sh: at each of POINTS, switched's mean against
# pooled's on the arms' suite, each of the batch's 433 decks against 32 others drawn at
# random, two games each from both seats, played once with each arm in the learner's seat
# (ARMS/POINT/arms.plays.json, then arms.held.json); a deal's two games are read together.
# Then the two arms side by side (tools/pool_arms_report.py): reading 4 at the last point,
# and beside it the arms' differences deal by deal on the measures they share. Some half an
# hour on the CUDA machine.
#   AT_ONCE=4 bash experiments/commands/pool-batch-six-arms.sh
set -euo pipefail
DEVICE=${DEVICE:-cuda}
WORKERS=${WORKERS:-24}
THREADS=${THREADS:-20}
SLOTS=${SLOTS:-16}
SEED=${SEED:-2012}
RUNS=${RUNS:-runs/expanded-pool}
ANALYSIS=${ANALYSIS:-analysis/expanded-pool}
POINTS=${POINTS:-3840 7680}
SUITE=${SUITE:-eval_suites/pool-batch-six-arms.json}
ARMS=${ARMS:-$ANALYSIS/batch-six-arms}
LOGS=${LOGS:-$ARMS/logs}
PYTHON=${PYTHON:-}
# shellcheck source=experiments/commands/at-once.sh
source "$(dirname "${BASH_SOURCE[0]}")/at-once.sh"

python=(uv run python)
[ -z "$PYTHON" ] || python=("$PYTHON")
for point in $POINTS; do
  for arm in pooled switched; do
    if [ ! -f "$RUNS/batch-six-$arm-mean/$point.pt" ]; then
      echo "Missing $RUNS/batch-six-$arm-mean/$point.pt" >&2
      exit 1
    fi
  done
done
for arm in pooled switched; do
  if [ ! -f "$ANALYSIS/batch-six-$arm/report.json" ]; then
    echo "batch-six-$arm is not measured: $ANALYSIS/batch-six-$arm/report.json" >&2
    exit 1
  fi
done
cargo build --release
mkdir -p "$ARMS"

failed() {
  grep -q '"error_type"' "$1"
}
# playing FILE OPTION...: the arms' suite as the options name, into FILE.json.
playing() {
  local file=$1
  shift
  mkdir -p "$(dirname "$file")"
  "${python[@]}" -m ygo_training evaluate-suite "$@" --device "$DEVICE" \
    --workers "$WORKERS" --pool-threads "$THREADS" --pool-slots "$SLOTS" --seed "$SEED" \
    --suite "$SUITE" --output "$file.json.part" > /dev/null
  if failed "$file.json.part"; then
    echo "A case of $file failed: $file.json.part" >&2
    return 1
  fi
  mv "$file.json.part" "$file.json"
  echo "$(date +%H:%M) $file"
}
for point in $POINTS; do
  pooled=$RUNS/batch-six-pooled-mean/$point.pt
  switched=$RUNS/batch-six-switched-mean/$point.pt
  if [ ! -f "$ARMS/$point/arms.plays.json" ]; then
    at_once "arms-$point-plays" playing "$ARMS/$point/arms.plays" "$switched" \
      --board-representation individual-tokens --opponent-checkpoint "$pooled"
  fi
  if [ ! -f "$ARMS/$point/arms.held.json" ]; then
    at_once "arms-$point-held" playing "$ARMS/$point/arms.held" "$pooled" \
      --opponent-checkpoint "$switched"
  fi
done
all_done
"${python[@]}" tools/pool_arms_report.py "$ANALYSIS/batch-six-pooled" \
  "$ANALYSIS/batch-six-switched" "$ARMS" --points "${POINTS// /,}" | tee "$ARMS/report.txt"
