#!/usr/bin/env bash
# The expanded pool's first batch test (experiments/2026-11-02-expanded-pool.md):
# twelve new decks taught by PPO from where the network of the seventy-one decks
# stands, and read against that network frozen. Run from the repository root on
# the CUDA machine. PARTS names what is made, in its order:
#
# run. PPO from NETWORK at RATE for UPDATES updates of DUELS duels, every duel
# against itself: three quarters of them with a new deck in one seat and any of
# the decks in the other, a quarter between the old decks (--self-play-weights
# from RUN.weights.json: a new deck against an old one 38, against a new one or
# itself 76, an old deck against an old one or itself 5), with the settings of
# fast; a checkpoint every STEP.
#
# measure. Into OUT, at each of POINTS by the mean of the last MEAN_KEPT
# checkpoints the run kept by then, and at 0 by NETWORK: each new deck against
# NETWORK at each old deck (PAIRINGS), and every ordered pairing of the old decks
# head to head against it (OLD). After the last point, into OUT/LAST: the next
# twelve decks against NETWORK (HELD_OUT), at 0 too; the new decks against the
# thirteen decks' pilots (PILOTS), at 0 too; and each old mirror head to head
# against its champion. Then the run's metrics and the report
# (tools/first_batch_report.py).
#
# Some 12 hours for the run and 2 for the measure, AT_ONCE=4.
#   PARTS="run measure" AT_ONCE=4 bash experiments/commands/first-batch.sh
set -euo pipefail
DEVICE=${DEVICE:-cuda}
WORKERS=${WORKERS:-24}
THREADS=${THREADS:-20}
SLOTS=${SLOTS:-16}
DUELS=${DUELS:-520}
PARTS=${PARTS:-run measure}
NETWORK=${NETWORK:-runs/seventy-one-champions/ppo-mean/21900.pt}
RUNS=${RUNS:-runs/expanded-pool}
OUT=${OUT:-analysis/expanded-pool/first-batch}
LIST=${LIST:-experiments/commands/champion-recordings.list}
THIRTEEN=${THIRTEEN:-blackwing,lightsworn,gishki,burn,crystal,dragunity,gladiator,heroes,infernity,monarch,morphtronic,rock-block,spellcaster}
NEW=${NEW:-game-0001,game-0002,game-0003,game-0004,game-0005,game-0006,game-0007,game-0008,game-0009,game-0010,game-0011,game-0012}
FIRST=${FIRST:-33120}
UPDATES=${UPDATES:-4320}
POINTS=${POINTS:-540 1080 2160 4320}
STEP=${STEP:-60}
MEAN_KEPT=${MEAN_KEPT:-6}
RATE=${RATE:-1e-4}
SEED=${SEED:-2012}
WEIGHT_NEW_OLD=${WEIGHT_NEW_OLD:-38}
WEIGHT_NEW_NEW=${WEIGHT_NEW_NEW:-76}
WEIGHT_OLD_OLD=${WEIGHT_OLD_OLD:-5}
IN_FLIGHT=${IN_FLIGHT:-1040}
FAST_SLOTS=${FAST_SLOTS:-52}
MINIBATCH=${MINIBATCH:-512}
PAIRINGS=${PAIRINGS:-eval_suites/first-batch-pairings-checkpoint.json}
HELD_OUT=${HELD_OUT:-eval_suites/next-batch-pairings-checkpoint.json}
PILOTS=${PILOTS:-eval_suites/first-batch-against-pilots.json}
OLD=${OLD:-eval_suites/seventy-one-decks-check-pairings-checkpoint.json}
MIRRORS=${MIRRORS:-analysis/seventy-one-champions/checks/suites}
PYTHON=${PYTHON:-}
LOGS=${LOGS:-$OUT/logs}
# shellcheck source=experiments/commands/at-once.sh
source "$(dirname "${BASH_SOURCE[0]}")/at-once.sh"

for part in $PARTS; do
  case $part in
    run | measure) ;;
    *)
      echo "PARTS names run and measure, not $part" >&2
      exit 1
      ;;
  esac
done
mapfile -t olds < <(grep -v '^#' "$LIST" | awk 'NF {print $1}')
IFS=, read -r -a news <<< "$NEW"
for path in "$NETWORK" "$PAIRINGS" "$HELD_OUT" "$PILOTS" "$OLD"; do
  if [ ! -f "$path" ]; then
    echo "Missing $path" >&2
    exit 1
  fi
done
last_point=${POINTS##* }
if ((last_point != UPDATES)); then
  echo "The last of POINTS is UPDATES" >&2
  exit 1
fi
for point in $POINTS; do
  if (((FIRST + point) % STEP)); then
    echo "POINTS name checkpoints the run keeps, not $point" >&2
    exit 1
  fi
done
python=(uv run python)
[ -z "$PYTHON" ] || python=("$PYTHON")
cargo build --release
mkdir -p "$RUNS" "$OUT"

# checkpoint POINT: the checkpoint the run kept after POINT updates.
checkpoint() {
  printf '%s/first-batch/checkpoints/model-%05d.pt' "$RUNS" $((FIRST + $1 - 1))
}
failed() {
  grep -q '"error_type"' "$1"
}
declare -A queued
# played FILE SUITE OPTION...: what the options name on a suite, with how each
# game ended, into FILE.json, at once with others where AT_ONCE says.
played() {
  if { [ ! -f "$1.json" ] || failed "$1.json"; } && [ -z "${queued[$1]:-}" ]; then
    queued[$1]=1
    at_once "$(basename "$(dirname "$1")")-$(basename "$1")" playing "$@"
  fi
}
playing() {
  local file=$1 suite=$2
  shift 2
  mkdir -p "$(dirname "$file")"
  "${python[@]}" -m ygo_training evaluate-suite "$@" --device "$DEVICE" \
    --workers "$WORKERS" --pool-threads "$THREADS" --pool-slots "$SLOTS" --seed "$SEED" \
    --suite "$suite" --output "$file.json.part" > /dev/null
  if failed "$file.json.part"; then
    echo "A case of $file failed: $file.json.part" >&2
    return 1
  fi
  mv "$file.json.part" "$file.json"
  echo "$(date +%H:%M) $file"
}
# weights FILE: the weights of the duels, into FILE.
weights() {
  "${python[@]}" - "$1" "$NEW" "$(IFS=,; echo "${olds[*]}")" "$WEIGHT_NEW_OLD" "$WEIGHT_NEW_NEW" "$WEIGHT_OLD_OLD" << 'SCRIPT'
import json
import sys

file, new, old, new_old, new_new, old_old = sys.argv[1:7]
new, old = new.split(","), old.split(",")
told = {f"{one}>{two}": float(old_old) for one in old for two in old}
for one in new:
    for two in old:
        told[f"{one}>{two}"] = told[f"{two}>{one}"] = float(new_old)
    for two in new:
        told[f"{one}>{two}"] = float(new_new)
with open(file, "w", encoding="utf8") as out:
    json.dump(told, out)
SCRIPT
}

if [[ " $PARTS " == *" run "* ]]; then
  decks=$(IFS=,; echo "${news[*]},${olds[*]}")
  weights "$RUNS/first-batch.weights.json"
  arguments=(
    --device "$DEVICE" --workers "$WORKERS" --pool-threads "$THREADS"
    --run "$RUNS/first-batch" --learner "$decks" --opponents "$THIRTEEN" --duels "$DUELS"
    --eval-every 0 --keep-checkpoints "$STEP" --seed "$SEED" --ppo-epochs 1
    --pool-slots "$FAST_SLOTS" --rollout-estimates kept
    --duels-in-flight "$IN_FLIGHT" --minibatch "$MINIBATCH" --inference-graphs 0 --pool-groups 2
    --self-play-duels "$DUELS" --self-play-weights "$RUNS/first-batch.weights.json"
  )
  if ! ls "$RUNS/first-batch"/checkpoints/model-*.pt > /dev/null 2>&1; then
    "${python[@]}" -m ygo_training train "${arguments[@]}" --resume "$NETWORK" \
      --resume-learning-rate "$RATE" --updates "$UPDATES"
  elif [ ! -f "$(checkpoint "$UPDATES")" ]; then
    latest=$(basename "$(ls "$RUNS/first-batch"/checkpoints/model-*.pt | sort | tail -1)" .pt)
    "${python[@]}" -m ygo_training train "${arguments[@]}" --continue \
      --updates $((FIRST + UPDATES - 1 - 10#${latest#model-}))
  fi
fi

if [[ " $PARTS " == *" measure "* ]]; then
  # The mean of the last checkpoints kept by each point.
  for point in $POINTS; do
    mean=$RUNS/first-batch-mean/$point.pt
    if [ ! -f "$mean" ]; then
      kept=()
      for ((back = MEAN_KEPT - 1; back >= 0; back--)); do
        kept+=("$(checkpoint $((point - back * STEP)))")
      done
      mkdir -p "$(dirname "$mean")"
      "${python[@]}" tools/mean_checkpoint.py "${kept[@]}" --output "$mean"
    fi
  done
  played "$OUT/0/pairings" "$PAIRINGS" "$NETWORK" --opponent-checkpoint "$NETWORK"
  played "$OUT/0/held-out" "$HELD_OUT" "$NETWORK" --opponent-checkpoint "$NETWORK"
  played "$OUT/0/pilots" "$PILOTS" "$NETWORK"
  for point in $POINTS; do
    mean=$RUNS/first-batch-mean/$point.pt
    played "$OUT/$point/pairings" "$PAIRINGS" "$mean" --opponent-checkpoint "$NETWORK"
    played "$OUT/$point/old" "$OLD" "$mean" --opponent-checkpoint "$NETWORK"
  done
  mean=$RUNS/first-batch-mean/$UPDATES.pt
  played "$OUT/$UPDATES/held-out" "$HELD_OUT" "$mean" --opponent-checkpoint "$NETWORK"
  played "$OUT/$UPDATES/pilots" "$PILOTS" "$mean"
  for deck in "${olds[@]}"; do
    teacher=$(awk -v deck="$deck" '$1 == deck {print $2}' "$LIST")
    played "$OUT/$UPDATES/mirrors/$deck.plays" "$MIRRORS/$deck.json" "$mean" --opponent-checkpoint "$teacher"
    played "$OUT/$UPDATES/mirrors/$deck.held" "$MIRRORS/$deck.json" "$teacher" --opponent-checkpoint "$mean"
  done
  all_done
  cp "$RUNS/first-batch/metrics.jsonl" "$OUT/first-batch.metrics.jsonl"
  "${python[@]}" tools/first_batch_report.py "$OUT" --points "${POINTS// /,}" | tee "$OUT/report.txt"
fi
