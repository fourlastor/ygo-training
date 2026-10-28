#!/usr/bin/env bash
# A network of the seventy-one decks
# (experiments/2026-10-28-seventy-one-champions.md): each deck's run of the
# champions' recipe copied into one network from its recorded games, then
# PPO, drawn once the fourteen decks' mirror arms are read. Run from the
# repository root on the CUDA machine. PARTS names what is made, in its
# order:
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
# AT_ONCE: how many of the measurements go at a time, through an MPS server
# on a card (experiments/commands/at-once.sh), what each writes in
# OUT/logs; 1, the default, one after the other.
#
# By reckoning on the CUDA machine: the copy 20 to 30 hours, its
# measurements 2 to 3 at seven at once.
#   PARTS=copy bash experiments/commands/seventy-one-champions.sh
#   PARTS=copy-measure AT_ONCE=7 bash experiments/commands/seventy-one-champions.sh
set -euo pipefail
DEVICE=${DEVICE:-cuda}
WORKERS=${WORKERS:-24}
THREADS=${THREADS:-20}
SLOTS=${SLOTS:-16}
DUELS=${DUELS:-520}
PARTS=${PARTS:-copy copy-measure}
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
NETWORK=${NETWORK:-runs/seventy-one-decks/all71/checkpoints/model-27599.pt}
RUNS=${RUNS:-runs/seventy-one-champions}
OUT=${OUT:-analysis/seventy-one-champions}
SUITES=${SUITES:-eval_suites}
PILOTS=${PILOTS:-eval_suites/seventy-one-decks-against-pilots.json}
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
    copy | copy-measure) ;;
    *)
      echo "PARTS names copy and copy-measure, not $part" >&2
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
mapfile -t decks < <(grep -v '^#' "$LIST" | awk 'NF {print $1}')
DECKS=$(IFS=,; echo "${decks[*]}")
for path in "$NETWORK" "$PILOTS" $(printf "$SUITES/%s-against-checkpoint.json " "${decks[@]}"); do
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
    "${python[@]}" -m ygo_training train "${arguments[@]}" "$@" \
      --continue --updates $((first + updates - 1 - 10#${latest#model-}))
  fi
}

copied=$(checkpoint "$RUNS/taught" 0 "$COPY")

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
  "${python[@]}" tools/seventy_one_champions_report.py "$OUT" --decks "$DECKS" \
    --champions "$CHAMPIONS" --teachers "$TEACHERS_MEASURED" --output "$OUT/report.json" |
    tee "$OUT/report.txt"
  echo "The measurements and what they say are in $OUT."
fi
