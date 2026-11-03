#!/usr/bin/env bash
# A batch of the expanded pool (experiments/2026-11-02-expanded-pool.md): new decks
# taught by PPO from where the network stands after the batch before, and read
# against that network frozen. Run from the repository root on the CUDA machine.
# PARTS names what is made, in its order:
#
# run. PPO from NETWORK at RATE for UPDATES updates of DUELS duels, every duel
# against itself (--self-play-weights from RUN.weights.json): MIRRORS_SHARE of them
# in the mirrors of all the decks, OLD_SHARE between two old decks, and the rest
# with a new deck in one seat and any other deck in the other; with the settings of
# fast, a checkpoint every STEP. UPDATES gives a new deck some 80,000 games.
#
# measure. Into OUT, at each of POINTS by the mean of the last MEAN_KEPT
# checkpoints the run kept by then, and at 0 by NETWORK: each new deck against
# NETWORK at each old deck (PAIRINGS), and at the points every ordered pairing of
# the old decks head to head against it (OLD). After the last point: the next decks
# against NETWORK (HELD_OUT), at 0 too; the new decks against the thirteen decks'
# pilots (PILOTS), at 0 too; and each of MIRROR_DECKS head to head in its mirror
# against its champion. Then the run's metrics and the report
# (tools/pool_batch_report.py).
#
# Batch one: some 19 hours for the run and 1.5 for the measure, AT_ONCE=4.
#   PARTS="run measure" AT_ONCE=4 bash experiments/commands/pool-batch.sh
set -euo pipefail
DEVICE=${DEVICE:-cuda}
WORKERS=${WORKERS:-24}
THREADS=${THREADS:-20}
SLOTS=${SLOTS:-16}
DUELS=${DUELS:-520}
PARTS=${PARTS:-run measure}
NETWORK=${NETWORK:-runs/expanded-pool/first-batch-mean/4320.pt}
RUNS=${RUNS:-runs/expanded-pool}
RUN=${RUN:-batch-one}
OUT=${OUT:-analysis/expanded-pool/$RUN}
LIST=${LIST:-experiments/commands/champion-recordings.list}
THIRTEEN=${THIRTEEN:-blackwing,lightsworn,gishki,burn,crystal,dragunity,gladiator,heroes,infernity,monarch,morphtronic,rock-block,spellcaster}
# Batch one leaves out game-0037: its Cloak and Dagger names a card, a decision of
# 4,096 answers that the pool pads every decision in flight to.
NEW=${NEW:-$(printf "game-%04d," $(seq 13 36) $(seq 38 63) | sed "s/,$//")}
# The game decks of the batches before, old decks here as the seventy-one are.
EARLIER=${EARLIER:-$(printf "game-%04d," $(seq 1 12) | sed "s/,$//")}
FIRST=${FIRST:-37440}
UPDATES=${UPDATES:-8040}
POINTS=${POINTS:-4020 8040}
STEP=${STEP:-60}
MEAN_KEPT=${MEAN_KEPT:-6}
RATE=${RATE:-1e-4}
SEED=${SEED:-2012}
MIRRORS_SHARE=${MIRRORS_SHARE:-0.25}
OLD_SHARE=${OLD_SHARE:-0.1875}
IN_FLIGHT=${IN_FLIGHT:-1040}
FAST_SLOTS=${FAST_SLOTS:-52}
MINIBATCH=${MINIBATCH:-512}
PAIRINGS=${PAIRINGS:-eval_suites/pool-batch-one-pairings-checkpoint.json}
HELD_OUT=${HELD_OUT:-eval_suites/pool-batch-one-held-out-checkpoint.json}
PILOTS=${PILOTS:-eval_suites/pool-batch-one-against-pilots.json}
OLD=${OLD:-eval_suites/pool-batch-one-old-pairings-checkpoint.json}
MIRRORS=${MIRRORS:-analysis/seventy-one-champions/checks/suites}
# The decks whose mirrors are read, of the champions' list: all of them where unset.
MIRROR_DECKS=${MIRROR_DECKS:-}
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
mapfile -t seventy_one < <(grep -v '^#' "$LIST" | awk 'NF {print $1}')
IFS=, read -r -a earlier <<< "$EARLIER"
olds=("${seventy_one[@]}" "${earlier[@]}")
IFS=, read -r -a news <<< "$NEW"
if [ -n "$MIRROR_DECKS" ]; then
  IFS=, read -r -a mirrored <<< "$MIRROR_DECKS"
else
  mirrored=("${seventy_one[@]}")
fi
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
  printf '%s/%s/checkpoints/model-%05d.pt' "$RUNS" "$RUN" $((FIRST + $1 - 1))
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
# weights FILE: the weights of the duels, into FILE, and what they give a deck.
weights() {
  "${python[@]}" - "$1" "$NEW" "$(IFS=,; echo "${olds[*]}")" "$MIRRORS_SHARE" "$OLD_SHARE" "$DUELS" << 'SCRIPT'
import json
import sys

file, new, old, mirrors, old_share, duels = sys.argv[1:7]
new, old = new.split(","), old.split(",")
decks = new + old
mirrors, old_share, duels = float(mirrors), float(old_share), int(duels)
new_share = 1 - mirrors - old_share
# A new deck meets any other deck as often from either seat, so that a pair of new
# decks is met from both: twice the weight of a new deck and an old one.
pairs = 2 * len(new) * len(old) + 2 * len(new) * (len(new) - 1)
told = {f"{one}>{one}": mirrors / len(decks) for one in decks}
for one in old:
    for two in old:
        if one != two:
            told[f"{one}>{two}"] = old_share / (len(old) * (len(old) - 1))
for one in new:
    for two in old:
        told[f"{one}>{two}"] = told[f"{two}>{one}"] = new_share / pairs
    for two in new:
        if one != two:
            told[f"{one}>{two}"] = 2 * new_share / pairs
with open(file, "w", encoding="utf8") as out:
    json.dump(told, out)
seats = {deck: 0.0 for deck in decks}
for pair, weight in told.items():
    for deck in pair.split(">"):
        seats[deck] += duels * weight
print(
    f"An update gives a new deck {sum(seats[d] for d in new) / len(new):.2f} games,"
    f" an old one {sum(seats[d] for d in old) / len(old):.2f},"
    f" a mirror {duels * mirrors / len(decks):.2f} duels"
)
SCRIPT
}

if [[ " $PARTS " == *" run "* ]]; then
  decks=$(IFS=,; echo "${news[*]},${olds[*]}")
  weights "$RUNS/$RUN.weights.json"
  arguments=(
    --device "$DEVICE" --workers "$WORKERS" --pool-threads "$THREADS"
    --run "$RUNS/$RUN" --learner "$decks" --opponents "$THIRTEEN" --duels "$DUELS"
    --eval-every 0 --keep-checkpoints "$STEP" --seed "$SEED" --ppo-epochs 1
    --pool-slots "$FAST_SLOTS" --rollout-estimates kept
    --duels-in-flight "$IN_FLIGHT" --minibatch "$MINIBATCH" --inference-graphs 0 --pool-groups 2
    --self-play-duels "$DUELS" --self-play-weights "$RUNS/$RUN.weights.json"
  )
  if ! ls "$RUNS/$RUN"/checkpoints/model-*.pt > /dev/null 2>&1; then
    "${python[@]}" -m ygo_training train "${arguments[@]}" --resume "$NETWORK" \
      --resume-learning-rate "$RATE" --updates "$UPDATES"
  elif [ ! -f "$(checkpoint "$UPDATES")" ]; then
    latest=$(basename "$(ls "$RUNS/$RUN"/checkpoints/model-*.pt | sort | tail -1)" .pt)
    "${python[@]}" -m ygo_training train "${arguments[@]}" --continue \
      --updates $((FIRST + UPDATES - 1 - 10#${latest#model-}))
  fi
fi

if [[ " $PARTS " == *" measure "* ]]; then
  # The mean of the last checkpoints kept by each point.
  for point in $POINTS; do
    mean=$RUNS/$RUN-mean/$point.pt
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
    mean=$RUNS/$RUN-mean/$point.pt
    played "$OUT/$point/pairings" "$PAIRINGS" "$mean" --opponent-checkpoint "$NETWORK"
    played "$OUT/$point/old" "$OLD" "$mean" --opponent-checkpoint "$NETWORK"
  done
  mean=$RUNS/$RUN-mean/$UPDATES.pt
  played "$OUT/$UPDATES/held-out" "$HELD_OUT" "$mean" --opponent-checkpoint "$NETWORK"
  played "$OUT/$UPDATES/pilots" "$PILOTS" "$mean"
  for deck in "${mirrored[@]}"; do
    teacher=$(awk -v deck="$deck" '$1 == deck {print $2}' "$LIST")
    played "$OUT/$UPDATES/mirrors/$deck.plays" "$MIRRORS/$deck.json" "$mean" --opponent-checkpoint "$teacher"
    played "$OUT/$UPDATES/mirrors/$deck.held" "$MIRRORS/$deck.json" "$teacher" --opponent-checkpoint "$mean"
  done
  all_done
  cp "$RUNS/$RUN/metrics.jsonl" "$OUT/$RUN.metrics.jsonl"
  "${python[@]}" tools/pool_batch_report.py "$OUT" --points "${POINTS// /,}" --run "$RUN" | tee "$OUT/report.txt"
fi
