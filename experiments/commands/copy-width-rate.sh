#!/usr/bin/env bash
# The copy's width and rate at seventy-one decks (experiments/2026-11-02-expanded-pool.md,
# its section beside the batches): four copies of the decks of LIST from their recordings,
# each made as seventy-one-champions' copy was (experiments/commands/seventy-one-champions.sh,
# its copy), at once on a card through MPS (experiments/commands/at-once.sh):
#
#   control  as that copy was made;
#   wide     twice as wide: --embedding-dim 192 --hidden-dim 384;
#   narrow   half as wide: --embedding-dim 48 --hidden-dim 96;
#   slow     control from its checkpoint after SLOW_FROM updates, on at SLOW_RATE from
#            the same state, on the same games after it; it begins once control is there.
#
# Each runs UPDATES updates of DUELS duels from scratch at SEED, ITSELF of them in the
# mirrors with both seats imitated and the others against the thirteen decks' pilots, in
# minibatches of 256 with no duels in flight, every game seen once and its result at four
# of its positions; a checkpoint every STEP. Then their metrics into OUT, and the report
# (tools/copy_width_rate_report.py) over WINDOW, beside REFERENCE (seventy-one-champions'
# copy's metrics) and FOURTEEN (the fourteen decks' copy's) where they are there. Run from
# the repository root on a rented card, with the recordings in RECORDS and what the
# benchmark's setup leaves there:
#
#   AT_ONCE=4 bash experiments/commands/rented-setup.sh \
#     bash experiments/commands/copy-width-rate.sh
#
# THREADS is the machine's share of threads, as the setup leaves it: each copy takes a
# quarter of them (ARM_THREADS), with the pool's slots to hold an update's DUELS duels.
# A run that was stopped goes on from its last checkpoint when this is started again.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export DEVICE=${DEVICE:-cuda}
WORKERS=${WORKERS:-24}
total=${THREADS:-32}
ARM_THREADS=${ARM_THREADS:-$((total / 4))}
DUELS=${DUELS:-520}
ITSELF=${ITSELF:-260}
LIST=${LIST:-experiments/commands/champion-recordings.list}
RECORDS=${RECORDS:-records/champions}
THIRTEEN=${THIRTEEN:-blackwing,lightsworn,gishki,burn,crystal,dragunity,gladiator,heroes,infernity,monarch,morphtronic,rock-block,spellcaster}
UPDATES=${UPDATES:-1420}
STEP=${STEP:-142}
SLOW_FROM=${SLOW_FROM:-710}
SLOW_RATE=${SLOW_RATE:-1e-4}
SEED=${SEED:-2012}
# The updates read, the first and the last; stretches of so many updates for the errors.
WINDOW=${WINDOW:-711,1420}
STRETCH=${STRETCH:-71}
RUNS=${RUNS:-runs/copy-width-rate}
OUT=${OUT:-analysis/copy-width-rate}
REFERENCE=${REFERENCE:-analysis/seventy-one-champions/taught.metrics.jsonl}
FOURTEEN=${FOURTEEN:-analysis/fourteen-decks/taught.metrics.jsonl}
# How long slow waits for control's checkpoint, in seconds.
PATIENCE=${PATIENCE:-86400}
LOGS=${LOGS:-$RUNS/logs}
PYTHON=${PYTHON:-}
python=(uv run python)
[ -z "$PYTHON" ] || python=("$PYTHON")
# shellcheck source=experiments/commands/at-once.sh
. experiments/commands/at-once.sh

if ((SLOW_FROM % STEP || SLOW_FROM >= UPDATES || UPDATES % STEP)); then
  echo "SLOW_FROM and UPDATES are multiples of STEP, SLOW_FROM before UPDATES" >&2
  exit 1
fi
mapfile -t decks < <(grep -v '^#' "$LIST" | awk 'NF {print $1}')
DECKS=$(IFS=,; echo "${decks[*]}")
for deck in "${decks[@]}"; do
  if [ ! -f "$RECORDS/$deck/about.json" ] && [ ! -f "$RECORDS/$deck.tar" ]; then
    echo "No recording of $deck in $RECORDS" >&2
    exit 1
  fi
done
pinned=$(git submodule status --recursive)
if grep -q '^[^ ]' <<< "$pinned"; then
  echo "A submodule is not at its pinned revision: git submodule update --init --recursive" >&2
  exit 1
fi
cargo build --release
mkdir -p "$RUNS" "$OUT"
# Each thread's share of an update's duels, an even count.
slots=$(((DUELS + ARM_THREADS - 1) / ARM_THREADS))
SLOTS_ARM=$((slots + slots % 2))
echo "$(date +%H:%M) ${#decks[@]} decks, four copies of $ARM_THREADS threads with $SLOTS_ARM duels each"

# checkpoint RUN FIRST POINT: the checkpoint of the run at RUN after POINT updates past
# the FIRST it began with.
checkpoint() {
  printf '%s/checkpoints/model-%05d.pt' "$1" $(($2 + $3 - 1))
}
# unpacked: each deck's recording a folder of RECORDS, its archive unpacked beside
# itself where it is not yet, whole or not at all.
unpacked() {
  local deck into
  for deck in "${decks[@]}"; do
    [ ! -f "$RECORDS/$deck/about.json" ] || continue
    into=$(mktemp -d "$RECORDS/.unpacking.XXXXXX")
    tar -xf "$RECORDS/$deck.tar" -C "$into"
    mv "$into/$deck" "$RECORDS/$deck"
    rmdir "$into"
  done
}
# What every copy is told but its run.
copying=(
  --device "$DEVICE" --workers "$WORKERS" --pool-threads "$ARM_THREADS"
  --learner "$DECKS" --opponents "$THIRTEEN" --duels "$DUELS"
  --eval-every 0 --keep-checkpoints "$STEP" --seed "$SEED" --ppo-epochs 1
  --pool-slots "$SLOTS_ARM" --imitation-only-updates 1000000 --imitation-value-weight 0.5
  --imitation-value-positions 4 --minibatch 256 --duels-in-flight 0
  --self-play-duels "$ITSELF" --self-play-pairs mirrors --copy-records "$RECORDS"
)
# went NAME: the update the run NAME stands at, its latest checkpoint's.
went() {
  local latest
  latest=$(basename "$(ls "$RUNS/$1"/checkpoints/model-*.pt | sort | tail -1)" .pt)
  echo $((10#${latest#model-} + 1))
}
# made NAME OPTION...: the copy NAME to UPDATES updates, from where it stands or from its
# start.
made() {
  local name=$1
  shift
  if ! ls "$RUNS/$name"/checkpoints/model-*.pt > /dev/null 2>&1; then
    "${python[@]}" -m ygo_training train "${copying[@]}" --run "$RUNS/$name" "$@" \
      --updates "$UPDATES"
  elif [ ! -f "$(checkpoint "$RUNS/$name" 0 "$UPDATES")" ]; then
    "${python[@]}" -m ygo_training train "${copying[@]}" --run "$RUNS/$name" "$@" \
      --continue --updates $((UPDATES - $(went "$name")))
  fi
}
# slowed: slow, control's run as it stood after SLOW_FROM updates gone on at SLOW_RATE,
# once that checkpoint is there. A copy that goes on deals the recordings from where it
# stood (`Orchestrator._records_caught_up`), so the two imitate the same games after it;
# one begun anew from that checkpoint (--resume) would deal them from their first.
slowed() {
  local from waited=0
  from=$(checkpoint "$RUNS/control" 0 "$SLOW_FROM")
  while [ ! -f "$from" ] || [ ! -f "$from.json" ]; do
    if ((waited >= PATIENCE)); then
      echo "control has no checkpoint after $SLOW_FROM updates after ${PATIENCE}s: $from" >&2
      return 1
    fi
    sleep 30
    waited=$((waited + 30))
  done
  if ! ls "$RUNS/slow"/checkpoints/model-*.pt > /dev/null 2>&1; then
    mkdir -p "$RUNS/slow/checkpoints"
    cp "$from" "$from.json" "$RUNS/slow/checkpoints/"
    cp "$RUNS/control/run.json" "$RUNS/slow/run.json"
    # control's metrics to that checkpoint, which slow's go on from.
    "${python[@]}" - "$RUNS/control/metrics.jsonl" "$RUNS/slow/metrics.jsonl" "$SLOW_FROM" << 'KEPT'
import json, sys
lines = open(sys.argv[1], encoding="utf8").read().splitlines()
kept = [line for line in lines if line and int(json.loads(line)["version"]) < int(sys.argv[3])]
open(sys.argv[2], "w", encoding="utf8").write("".join(line + "\n" for line in kept))
KEPT
  fi
  if [ ! -f "$(checkpoint "$RUNS/slow" 0 "$UPDATES")" ]; then
    "${python[@]}" -m ygo_training train "${copying[@]}" --run "$RUNS/slow" --continue \
      --updates $((UPDATES - $(went slow))) --resume-learning-rate "$SLOW_RATE"
  fi
}

unpacked
at_once control made control
at_once wide made wide --embedding-dim 192 --hidden-dim 384
at_once narrow made narrow --embedding-dim 48 --hidden-dim 96
at_once slow slowed
all_done
echo "$(date +%H:%M) the four copies are made"
for run in control wide narrow slow; do
  cp "$RUNS/$run/metrics.jsonl" "$OUT/$run.metrics.jsonl"
  cp "$RUNS/$run/run.json" "$OUT/$run.run.json"
done
beside=()
[ ! -f "$REFERENCE" ] || beside+=(--reference "$REFERENCE")
[ ! -f "$FOURTEEN" ] || beside+=(--fourteen "$FOURTEEN")
"${python[@]}" tools/copy_width_rate_report.py "$OUT" --window "$WINDOW" --stretch "$STRETCH" \
  --decks "$DECKS" "${beside[@]}" --output "$OUT/report.json" | tee "$OUT/report.txt"
echo "The copies are read: $OUT/report.txt"
