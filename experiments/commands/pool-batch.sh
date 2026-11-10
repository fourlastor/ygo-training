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
# fast, a checkpoint every STEP. UPDATES gives a new deck some 80,000 games. The old decks'
# duels, their mirrors and their pairings, are at OLD_SCALE of those weights, what that leaves
# going to the new decks' pairings; with DISTILL_REFERENCE, at each decision of an old deck's
# seat in a duel between two old decks the divergence from that frozen network's answers is
# added to PPO's loss at DISTILL_WEIGHT (train --distill-reference); with DISTILL_SECOND too,
# in a duel between two of the seventy-one it is from that second network's answers instead
# (train --distill-second-reference). With BOARD, the network trained and the networks
# of the run measured are of that shape (train --board-representation): individual-tokens
# for the board as its cards, a NETWORK of the pooled board given it with
# tools/switch_board.py. An interrupted run goes on from its last checkpoint;
# with NEW_PINS=1 it goes on although the pins moved since (train --new-pins), for a move
# whose duels are the old ones.
#
# measure. Into OUT, at each of POINTS by the mean of the last MEAN_KEPT
# checkpoints the run kept by then, and at 0 by NETWORK: each new deck against
# NETWORK at each old deck (PAIRINGS), and at each of OLD_POINTS, the points where
# unset, every ordered pairing of the old decks head to head against it (OLD). After
# the last point: the next decks against NETWORK (HELD_OUT), at 0 too; the new decks
# against the thirteen decks' pilots (PILOTS), at 0 too; and each of MIRROR_DECKS head
# to head in its mirror against its champion. Then the run's metrics and the report
# (tools/pool_batch_report.py), the mirrors read deal by deal against those in
# MIRRORS_REFERENCE where set and told beside those in MIRRORS_BESIDE, the new decks read at
# the marks LEARNS and DOES_NOT where set (shares). What fails beside
# the others is played again once they are over, alone: they share the card's memory,
# which four of them can run out of together (the fast arm's pilots at 4,020, on
# 2026-11-04).
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
# The game decks of the batches before, old decks here as the seventy-one are: none where
# EARLIER is set empty.
EARLIER=${EARLIER-$(printf "game-%04d," $(seq 1 12) | sed "s/,$//")}
FIRST=${FIRST:-37440}
UPDATES=${UPDATES:-8040}
POINTS=${POINTS:-4020 8040}
# The points the old decks are measured at, the last of POINTS among them.
OLD_POINTS=${OLD_POINTS:-$POINTS}
STEP=${STEP:-60}
MEAN_KEPT=${MEAN_KEPT:-6}
RATE=${RATE:-1e-4}
# The cards' identity tables learn at this many times RATE.
IDENTITY_SCALE=${IDENTITY_SCALE:-1}
SEED=${SEED:-2012}
MIRRORS_SHARE=${MIRRORS_SHARE:-0.25}
OLD_SHARE=${OLD_SHARE:-0.1875}
OLD_SCALE=${OLD_SCALE:-1}
DISTILL_REFERENCE=${DISTILL_REFERENCE:-}
DISTILL_WEIGHT=${DISTILL_WEIGHT:-1}
# The network the seventy-one are held to in their duels with each other, where it is
# not DISTILL_REFERENCE.
DISTILL_SECOND=${DISTILL_SECOND:-}
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
# Another network's mirrors on the same deals, the reading's reference, and one told beside.
MIRRORS_REFERENCE=${MIRRORS_REFERENCE:-}
MIRRORS_BESIDE=${MIRRORS_BESIDE:-}
# The shape of the network trained and measured, where it is not the pooled board.
BOARD=${BOARD:-}
# Reading 1's marks, where a batch fixed others than the first batch's.
LEARNS=${LEARNS:-}
DOES_NOT=${DOES_NOT:-}
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
for path in "$NETWORK" "$PAIRINGS" "$HELD_OUT" "$PILOTS" "$OLD" $DISTILL_SECOND; do
  if [ ! -f "$path" ]; then
    echo "Missing $path" >&2
    exit 1
  fi
done
if [ -n "$DISTILL_SECOND" ] && [ -z "$DISTILL_REFERENCE" ]; then
  echo "DISTILL_SECOND goes with DISTILL_REFERENCE" >&2
  exit 1
fi
for path in $MIRRORS_REFERENCE $MIRRORS_BESIDE; do
  if [ ! -d "$path" ]; then
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
for point in $OLD_POINTS; do
  if [[ " $POINTS " != *" $point "* ]]; then
    echo "OLD_POINTS are among POINTS, not $point" >&2
    exit 1
  fi
done
if [ "${OLD_POINTS##* }" != "$last_point" ]; then
  echo "The last of OLD_POINTS is UPDATES" >&2
  exit 1
fi
python=(uv run python)
[ -z "$PYTHON" ] || python=("$PYTHON")
# The run's own networks are played in their shape; another network, an opponent or a
# champion, is read in its own.
board=()
[ -z "$BOARD" ] || board=(--board-representation "$BOARD")
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
# measures: the measure's games, a file at most once until `queued` is emptied.
measures() {
  local point mean deck teacher
  played "$OUT/0/pairings" "$PAIRINGS" "$NETWORK" "${board[@]}" --opponent-checkpoint "$NETWORK"
  played "$OUT/0/held-out" "$HELD_OUT" "$NETWORK" "${board[@]}" --opponent-checkpoint "$NETWORK"
  played "$OUT/0/pilots" "$PILOTS" "$NETWORK" "${board[@]}"
  for point in $POINTS; do
    mean=$RUNS/$RUN-mean/$point.pt
    played "$OUT/$point/pairings" "$PAIRINGS" "$mean" "${board[@]}" --opponent-checkpoint "$NETWORK"
  done
  for point in $OLD_POINTS; do
    mean=$RUNS/$RUN-mean/$point.pt
    played "$OUT/$point/old" "$OLD" "$mean" "${board[@]}" --opponent-checkpoint "$NETWORK"
  done
  mean=$RUNS/$RUN-mean/$UPDATES.pt
  played "$OUT/$UPDATES/held-out" "$HELD_OUT" "$mean" "${board[@]}" --opponent-checkpoint "$NETWORK"
  played "$OUT/$UPDATES/pilots" "$PILOTS" "$mean" "${board[@]}"
  for deck in "${mirrored[@]}"; do
    teacher=$(awk -v deck="$deck" '$1 == deck {print $2}' "$LIST")
    played "$OUT/$UPDATES/mirrors/$deck.plays" "$MIRRORS/$deck.json" "$mean" "${board[@]}" \
      --opponent-checkpoint "$teacher"
    played "$OUT/$UPDATES/mirrors/$deck.held" "$MIRRORS/$deck.json" "$teacher" --opponent-checkpoint "$mean"
  done
}
# weights FILE: the weights of the duels, into FILE, and what they give a deck.
weights() {
  "${python[@]}" - "$1" "$NEW" "$(IFS=,; echo "${olds[*]}")" "$MIRRORS_SHARE" "$OLD_SHARE" "$DUELS" \
    "$OLD_SCALE" << 'SCRIPT'
import json
import sys

file, new, old, mirrors, old_share, duels, old_scale = sys.argv[1:8]
new, old = new.split(","), old.split(",")
decks = new + old
mirrors, old_share, duels, old_scale = float(mirrors), float(old_share), int(duels), float(old_scale)
# The old decks' duels at old_scale of their weights, what that leaves to the new
# decks' pairings.
new_share = 1 - mirrors - old_share + (1 - old_scale) * (mirrors * len(old) / len(decks) + old_share)
# A new deck meets any other deck as often from either seat, so that a pair of new
# decks is met from both: twice the weight of a new deck and an old one.
pairs = 2 * len(new) * len(old) + 2 * len(new) * (len(new) - 1)
told = {f"{one}>{one}": mirrors / len(decks) * (old_scale if one in old else 1) for one in decks}
for one in old:
    for two in old:
        if one != two:
            told[f"{one}>{two}"] = old_scale * old_share / (len(old) * (len(old) - 1))
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
    + (f", an old mirror {duels * mirrors * old_scale / len(decks):.2f}" if old_scale != 1 else "")
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
    --identity-learning-rate-scale "$IDENTITY_SCALE" "${board[@]}"
  )
  if [ -n "$DISTILL_REFERENCE" ]; then
    arguments+=(
      --distill-reference "$DISTILL_REFERENCE" --distill-weight "$DISTILL_WEIGHT"
      --distill-decks "$(IFS=,; echo "${olds[*]}")"
    )
    if [ -n "$DISTILL_SECOND" ]; then
      arguments+=(
        --distill-second-reference "$DISTILL_SECOND"
        --distill-second-decks "$(IFS=,; echo "${seventy_one[*]}")"
      )
    fi
  fi
  if ! ls "$RUNS/$RUN"/checkpoints/model-*.pt > /dev/null 2>&1; then
    "${python[@]}" -m ygo_training train "${arguments[@]}" --resume "$NETWORK" \
      --resume-learning-rate "$RATE" --updates "$UPDATES"
  elif [ ! -f "$(checkpoint "$UPDATES")" ]; then
    latest=$(basename "$(ls "$RUNS/$RUN"/checkpoints/model-*.pt | sort | tail -1)" .pt)
    "${python[@]}" -m ygo_training train "${arguments[@]}" --continue ${NEW_PINS:+--new-pins} \
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
  measures
  if ! all_done; then
    echo "$(date +%H:%M) What failed is played again, alone" >&2
    AT_ONCE=1
    queued=()
    measures
  fi
  cp "$RUNS/$RUN/metrics.jsonl" "$OUT/$RUN.metrics.jsonl"
  report=(--points "${POINTS// /,}" --run "$RUN")
  [ -z "$MIRRORS_REFERENCE" ] || report+=(--reference "$MIRRORS_REFERENCE")
  [ -z "$MIRRORS_BESIDE" ] || report+=(--beside "$MIRRORS_BESIDE")
  [ -z "$LEARNS" ] || report+=(--learns "$LEARNS")
  [ -z "$DOES_NOT" ] || report+=(--does-not "$DOES_NOT")
  "${python[@]}" tools/pool_batch_report.py "$OUT" "${report[@]}" | tee "$OUT/report.txt"
fi
