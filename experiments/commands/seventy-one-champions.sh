#!/usr/bin/env bash
# A network of the seventy-one decks
# (experiments/2026-10-28-seventy-one-champions.md): each deck's run of the
# champions' recipe copied into one network from its recorded games, then
# PPO with every duel against itself, a quarter of them in the mirrors,
# checked as it trains. Run from the repository root on the CUDA machine.
# PARTS names what is made, in its order:
#
# copy. taught: one network of the seventy-one decks from scratch, each
# deck's games its teacher's recorded games played again with its answers
# and imitated, COPY updates of DUELS duels, 160 a deck, ITSELF of an
# update's duels in the mirrors, both seats imitated, the others against
# the thirteen decks' pilots, at 3e-4, in minibatches of 256 with no duels
# in flight, every game seen once and its result at four of its positions;
# a checkpoint every COPY_STEP. The recordings are RECORDS/DECK, each made
# by tools/record_champion.py; an archive RECORDS/DECK.tar is unpacked
# beside itself. First the teachers are held to the record
# (tools/seventy_one_teachers.py): each deck recorded with its run of LIST,
# none of them under all71 in its mirror by the figure of its making
# (CHAMPIONS), RELEASE the decks with none there.
#
# copy-measure. Into OUT: the copy's last checkpoint head to head against
# each teacher in its mirror; it and its checkpoints after each of CURVE
# against all71 in each mirror; it and all71 against the pilots (PILOTS);
# the copy's metrics; then the report (tools/seventy_one_champions_report.py,
# the teachers of RELEASE told by their measurements in TEACHERS_MEASURED).
#
# ppo. PPO on from the copy's last checkpoint for PPO updates at RATE, every
# duel of an update against itself and none against a pilot, a quarter of
# them in the mirrors (--self-play-weights from RUNS/ppo.weights.json: each
# mirror MIRROR_WEIGHT, each other ordered pairing OTHER_WEIGHT), with the
# fourteen decks' arms' settings; a checkpoint every STEP. It is trained in
# stretches to each of CHECKS, and after each its last checkpoint is checked,
# the training paused, on the deals the copy was checked on before it, into
# OUT/checks: head to head against each teacher in its mirror on CHECK_DEALS
# deals from both seats, and against all71 on CHECK_PAIRINGS. Where the
# mirrors or the pairings are under the copy at two checks in a row
# (tools/seventy_one_checks.py), it stops, unless EARLY_STOP is 0.
#
# ppo-measure. Into OUT: PPO's last checkpoint head to head against each
# teacher in its mirror; it and the copy against all71 on every ordered
# pairing (QUICK-1.json to QUICK-QUICK_PARTS.json, the deals of one suite in
# parts that go at once); it against the pilots; its metrics; then the
# report.
#
# AT_ONCE: how many of the measurements go at a time, through an MPS server
# on a card (experiments/commands/at-once.sh), what each writes in
# OUT/logs; 1, the default, one after the other.
#
# By reckoning on the CUDA machine: the copy 20 to 30 hours, its
# measurements 2 to 3 at seven at once; PPO some two days, a check some ten
# minutes at seven at once, its measurements about an hour.
#   PARTS=copy bash experiments/commands/seventy-one-champions.sh
#   PARTS=copy-measure AT_ONCE=7 bash experiments/commands/seventy-one-champions.sh
#   PARTS=ppo AT_ONCE=7 bash experiments/commands/seventy-one-champions.sh
#   PARTS=ppo-measure AT_ONCE=7 bash experiments/commands/seventy-one-champions.sh
set -euo pipefail
DEVICE=${DEVICE:-cuda}
WORKERS=${WORKERS:-24}
THREADS=${THREADS:-20}
SLOTS=${SLOTS:-16}
DUELS=${DUELS:-520}
PARTS=${PARTS:-copy copy-measure ppo ppo-measure}
# Each deck and its run's last checkpoint, which teaches it.
LIST=${LIST:-experiments/commands/champion-recordings.list}
RECORDS=${RECORDS:-../records}
CHAMPIONS=${CHAMPIONS:-analysis/champions}
RELEASE=${RELEASE:-blackwing,cyber,lightsworn,infernity}
TEACHERS_MEASURED=${TEACHERS_MEASURED:-analysis/fourteen-decks/teachers}
THIRTEEN=${THIRTEEN:-blackwing,lightsworn,gishki,burn,crystal,dragunity,gladiator,heroes,infernity,monarch,morphtronic,rock-block,spellcaster}
COPY=${COPY:-11360}
COPY_STEP=${COPY_STEP:-160}
ITSELF=${ITSELF:-260}
SEED=${SEED:-2012}
COPY_SLOTS=${COPY_SLOTS:-26}
# PPO: the fourteen decks' 4,320 updates by the deck, a quarter of its duels
# in the mirrors.
PPO=${PPO:-21900}
STEP=${STEP:-240}
RATE=${RATE:-1e-4}
MIRROR_WEIGHT=${MIRROR_WEIGHT:-70}
OTHER_WEIGHT=${OTHER_WEIGHT:-3}
# fast's settings, the fourteen decks' arms': two updates' duels in flight,
# the pool's slots to hold them, minibatches of 512, the calls as they are in
# two groups.
IN_FLIGHT=${IN_FLIGHT:-1040}
FAST_SLOTS=${FAST_SLOTS:-52}
MINIBATCH=${MINIBATCH:-512}
# PPO's checks: after so many of its updates, on so many deals of each mirror
# and against all71 on these pairings.
CHECKS=${CHECKS:-2880 5760 8640 11520 14400 17280 20160}
CHECK_DEALS=${CHECK_DEALS:-128}
CHECK_PAIRINGS=${CHECK_PAIRINGS:-eval_suites/seventy-one-decks-check-pairings-checkpoint.json}
EARLY_STOP=${EARLY_STOP:-1}
NETWORK=${NETWORK:-runs/seventy-one-decks/all71/checkpoints/model-27599.pt}
RUNS=${RUNS:-runs/seventy-one-champions}
OUT=${OUT:-analysis/seventy-one-champions}
SUITES=${SUITES:-eval_suites}
PILOTS=${PILOTS:-eval_suites/seventy-one-decks-against-pilots.json}
# Every ordered pairing against all71, in parts: QUICK-1.json and on.
QUICK=${QUICK:-eval_suites/seventy-one-decks-pairings-quick-checkpoint}
QUICK_PARTS=${QUICK_PARTS:-7}
# The copy's checkpoints measured in the mirrors: 40, 80, 120 and 160
# updates a deck.
CURVE=${CURVE:-2840 5680 8520 11360}
# The interpreter of a machine that has no uv: the benchmark's setup leaves one.
PYTHON=${PYTHON:-}
LOGS=${LOGS:-$OUT/logs}
# shellcheck source=experiments/commands/at-once.sh
source "$(dirname "${BASH_SOURCE[0]}")/at-once.sh"

for part in $PARTS; do
  case $part in
    copy | copy-measure | ppo | ppo-measure) ;;
    *)
      echo "PARTS names copy, copy-measure, ppo and ppo-measure, not $part" >&2
      exit 1
      ;;
  esac
done
if ((COPY % COPY_STEP || COPY < 3 * COPY_STEP)); then
  echo "COPY is a multiple of COPY_STEP, and three of them at least" >&2
  exit 1
fi
for point in $CURVE; do
  if ((point % COPY_STEP || point > COPY)); then
    echo "CURVE names checkpoints the copy keeps, not $point" >&2
    exit 1
  fi
done
previous=0
for point in $CHECKS; do
  if ((point <= previous || point >= PPO)); then
    echo "CHECKS names points of PPO in their order and before its end, not $point" >&2
    exit 1
  fi
  previous=$point
done
mapfile -t decks < <(grep -v '^#' "$LIST" | awk 'NF {print $1}')
DECKS=$(IFS=,; echo "${decks[*]}")
needed=("$NETWORK" "$PILOTS" "$CHECK_PAIRINGS")
for deck in "${decks[@]}"; do
  needed+=("$SUITES/$deck-against-checkpoint.json")
done
for ((part = 1; part <= QUICK_PARTS; part++)); do
  needed+=("$QUICK-$part.json")
done
for path in "${needed[@]}"; do
  if [ ! -f "$path" ]; then
    echo "Missing $path" >&2
    exit 1
  fi
done
pinned=$(git submodule status --recursive)
if grep -q '^[^ ]' <<< "$pinned"; then
  echo "A submodule is not at its pinned revision: git submodule update --init --recursive" >&2
  exit 1
fi
python=(uv run python)
[ -z "$PYTHON" ] || python=("$PYTHON")
cargo build --release
mkdir -p "$RUNS" "$OUT"

# checkpoint RUN FIRST POINT: the checkpoint of the run at RUN after POINT
# updates past the FIRST it began with.
checkpoint() {
  printf '%s/checkpoints/model-%05d.pt' "$1" $(($2 + $3 - 1))
}
# teacher DECK: the run that teaches DECK.
teacher() {
  awk -v deck="$1" '$1 == deck {print $2}' "$LIST"
}
# unpacked: each deck's recording a folder of RECORDS, its archive unpacked
# beside itself where it is not yet, whole or not at all.
unpacked() {
  local deck into
  for deck in "${decks[@]}"; do
    [ ! -f "$RECORDS/$deck/about.json" ] || continue
    if [ ! -f "$RECORDS/$deck.tar" ]; then
      echo "No recording of $deck in $RECORDS" >&2
      exit 1
    fi
    into=$(mktemp -d "$RECORDS/.unpacking.XXXXXX")
    tar -xf "$RECORDS/$deck.tar" -C "$into"
    mv "$into/$deck" "$RECORDS/$deck"
    rmdir "$into"
  done
}
# played FILE SUITE OPTION...: what the options name on a suite, with how
# each game ended, into FILE.json, at once with others where AT_ONCE says.
played() {
  # A file being played beside others is not there yet: it is not played
  # twice.
  if [ ! -f "$1.json" ] && [ -z "${queued[$1]:-}" ]; then
    queued[$1]=1
    at_once "$(basename "$(dirname "$1")")-$(basename "$1")" playing "$@"
  fi
}
declare -A queued
# playing FILE SUITE OPTION...: what played does, then and there.
playing() {
  local file=$1 suite=$2
  shift 2
  mkdir -p "$(dirname "$file")"
  "${python[@]}" -m ygo_training evaluate-suite "$@" --device "$DEVICE" \
    --workers "$WORKERS" --pool-threads "$THREADS" --pool-slots "$SLOTS" --seed "$SEED" \
    --suite "$suite" --output "$file.json.part" > /dev/null
  mv "$file.json.part" "$file.json"
  echo "$(date +%H:%M) $file"
}
# pair FILE FIRST SECOND SUITE: FIRST in the learner's seat against SECOND,
# then the other way round, into FILE.plays.json and FILE.held.json.
pair() {
  local file=$1 one=$2 other=$3 suite=$4
  played "$file.plays" "$suite" "$one" --opponent-checkpoint "$other"
  played "$file.held" "$suite" "$other" --opponent-checkpoint "$one"
}
# made NAME FIRST UPDATES KEPT OPTION...: the run NAME of the seventy-one
# decks, to UPDATES updates after the FIRST it began with, from where it
# stands or from its start, with a checkpoint every KEPT updates.
made() {
  local name=$1 first=$2 updates=$3 kept=$4 last latest
  shift 4
  last=$(checkpoint "$RUNS/$name" "$first" "$updates")
  local arguments=(
    --device "$DEVICE" --workers "$WORKERS" --pool-threads "$THREADS"
    --run "$RUNS/$name" --learner "$DECKS" --opponents "$THIRTEEN" --duels "$DUELS"
    --eval-every 0 --keep-checkpoints "$kept" --seed "$SEED" --ppo-epochs 1
  )
  if ! ls "$RUNS/$name"/checkpoints/model-*.pt > /dev/null 2>&1; then
    "${python[@]}" -m ygo_training train "${arguments[@]}" "$@" --updates "$updates"
  elif [ ! -f "$last" ]; then
    latest=$(basename "$(ls "$RUNS/$name"/checkpoints/model-*.pt | sort | tail -1)" .pt)
    # A run that goes on is told what it was told, but for where it began.
    local again=() skip=0 option
    for option in "$@"; do
      if ((skip)); then
        skip=0
      elif [ "$option" = --resume ] || [ "$option" = --resume-learning-rate ]; then
        skip=1
      else
        again+=("$option")
      fi
    done
    "${python[@]}" -m ygo_training train "${arguments[@]}" "${again[@]}" \
      --continue --updates $((first + updates - 1 - 10#${latest#model-}))
  fi
}
# weights FILE MIRROR OTHER: the weights of the duels against itself, each
# mirror MIRROR and each other ordered pairing of the decks OTHER, into FILE.
weights() {
  local file=$1 mirror=$2 other=$3 one two entries=()
  for one in "${decks[@]}"; do
    for two in "${decks[@]}"; do
      if [ "$one" = "$two" ]; then
        entries+=("\"$one>$two\": $mirror")
      else
        entries+=("\"$one>$two\": $other")
      fi
    done
  done
  (IFS=,; echo "{${entries[*]}}") > "$file"
}
# checked NAME NETWORK: the network as PPO's checks measure it, into
# OUT/checks/NAME: against all71 on the checks' pairings, and head to head
# against each teacher in its mirror.
checked() {
  local name=$1 network=$2 deck
  played "$OUT/checks/$name/pairings" "$CHECK_PAIRINGS" "$network" --opponent-checkpoint "$NETWORK"
  for deck in "${decks[@]}"; do
    pair "$OUT/checks/$name/$deck" "$network" "$(teacher "$deck")" "$OUT/checks/suites/$deck.json"
  done
  all_done
}
# heeded: what PPO's checks say so far, and PPO stopped where its last two
# checks meet the rule, unless EARLY_STOP is 0.
heeded() {
  local said=0
  "${python[@]}" tools/seventy_one_checks.py read "$OUT/checks" --decks "$DECKS" \
    --output "$OUT/checks/report.json" > "$OUT/checks/report.txt" || said=$?
  cat "$OUT/checks/report.txt"
  if ((said == 3 && EARLY_STOP)); then
    echo "PPO stops by its checks: $OUT/checks/report.txt (EARLY_STOP=0 goes on)" >&2
    exit 1
  elif ((said && said != 3)); then
    exit "$said"
  fi
}
# reported: what the measurements in OUT say.
reported() {
  "${python[@]}" tools/seventy_one_champions_report.py "$OUT" --decks "$DECKS" \
    --champions "$CHAMPIONS" --teachers "$TEACHERS_MEASURED" --output "$OUT/report.json" |
    tee "$OUT/report.txt"
  echo "The measurements and what they say are in $OUT."
}

copied=$(checkpoint "$RUNS/taught" 0 "$COPY")
trained=$(checkpoint "$RUNS/ppo" "$COPY" "$PPO")
fast=(
  --pool-slots "$FAST_SLOTS" --rollout-estimates kept
  --duels-in-flight "$IN_FLIGHT" --minibatch "$MINIBATCH" --inference-graphs 0 --pool-groups 2
  --resume "$copied" --resume-learning-rate "$RATE"
)

if [[ " $PARTS " == *" copy "* ]]; then
  unpacked
  "${python[@]}" tools/seventy_one_teachers.py "$LIST" --records "$RECORDS" \
    --champions "$CHAMPIONS" --release "$RELEASE"
  echo "$(date +%H:%M) taught begins"
  made taught 0 "$COPY" "$COPY_STEP" --pool-slots "$COPY_SLOTS" \
    --imitation-only-updates 1000000 --imitation-value-weight 0.5 \
    --imitation-value-positions 4 --minibatch 256 --duels-in-flight 0 \
    --self-play-duels "$ITSELF" --self-play-pairs mirrors --copy-records "$RECORDS"
  echo "$(date +%H:%M) taught ends"
fi

if [[ " $PARTS " == *" copy-measure "* ]]; then
  for path in "$copied" $(for deck in "${decks[@]}"; do teacher "$deck"; done); do
    if [ ! -f "$path" ]; then
      echo "Missing $path: the copy is measured when it and its teachers are made" >&2
      exit 1
    fi
  done
  # The two longest first, so that the others go beside them.
  played "$OUT/pilots/taught" "$PILOTS" "$copied"
  played "$OUT/pilots/all71" "$PILOTS" "$NETWORK"
  for deck in "${decks[@]}"; do
    pair "$OUT/heads-taught/$deck" "$copied" "$(teacher "$deck")" \
      "$SUITES/$deck-against-checkpoint.json"
  done
  for deck in "${decks[@]}"; do
    for point in $CURVE; do
      pair "$OUT/taught-$deck/$point" "$(checkpoint "$RUNS/taught" 0 "$point")" "$NETWORK" \
        "$SUITES/$deck-against-checkpoint.json"
    done
  done
  all_done
  cp "$RUNS/taught/metrics.jsonl" "$OUT/taught.metrics.jsonl"
  cp "$RUNS/taught/run.json" "$OUT/taught.run.json"
  reported
fi

if [[ " $PARTS " == *" ppo "* ]]; then
  for path in "$copied" $(for deck in "${decks[@]}"; do teacher "$deck"; done); do
    if [ ! -f "$path" ]; then
      echo "Missing $path: PPO goes on from the copy and is checked against its teachers" >&2
      exit 1
    fi
  done
  weights "$RUNS/ppo.weights.json" "$MIRROR_WEIGHT" "$OTHER_WEIGHT"
  "${python[@]}" tools/seventy_one_checks.py suites --decks "$DECKS" --suites "$SUITES" \
    --games "$CHECK_DEALS" --into "$OUT/checks/suites"
  checked copy "$copied"
  for point in $CHECKS; do
    echo "$(date +%H:%M) ppo to $point"
    made ppo "$COPY" "$point" "$STEP" "${fast[@]}" --self-play-duels "$DUELS" \
      --self-play-weights "$RUNS/ppo.weights.json"
    checked "$point" "$(checkpoint "$RUNS/ppo" "$COPY" "$point")"
    heeded
  done
  echo "$(date +%H:%M) ppo to $PPO"
  made ppo "$COPY" "$PPO" "$STEP" "${fast[@]}" --self-play-duels "$DUELS" \
    --self-play-weights "$RUNS/ppo.weights.json"
  echo "$(date +%H:%M) ppo ends"
fi

if [[ " $PARTS " == *" ppo-measure "* ]]; then
  for path in "$trained" "$copied" $(for deck in "${decks[@]}"; do teacher "$deck"; done); do
    if [ ! -f "$path" ]; then
      echo "Missing $path: PPO is measured when it, the copy and the teachers are made" >&2
      exit 1
    fi
  done
  # The longest first, so that the others go beside them.
  for ((part = 1; part <= QUICK_PARTS; part++)); do
    played "$OUT/pairings/ppo-$part" "$QUICK-$part.json" "$trained" --opponent-checkpoint "$NETWORK"
    played "$OUT/pairings/taught-$part" "$QUICK-$part.json" "$copied" \
      --opponent-checkpoint "$NETWORK"
  done
  played "$OUT/pilots/ppo" "$PILOTS" "$trained"
  for deck in "${decks[@]}"; do
    pair "$OUT/heads-ppo/$deck" "$trained" "$(teacher "$deck")" \
      "$SUITES/$deck-against-checkpoint.json"
  done
  all_done
  cp "$RUNS/ppo/metrics.jsonl" "$OUT/ppo.metrics.jsonl"
  cp "$RUNS/ppo/run.json" "$OUT/ppo.run.json"
  reported
fi
