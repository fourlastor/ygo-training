#!/usr/bin/env bash
# The seventy-one's copy twice as wide (experiments/2026-11-02-expanded-pool.md, its
# section beside the batches): one network of the seventy-one decks from scratch, each
# deck's games its teacher's recorded games played again with its answers and imitated,
# as seventy-one-champions' copy (taught) was made (experiments/commands/
# seventy-one-champions.sh, its copy): COPY updates of DUELS duels, 160 a deck, ITSELF of
# an update's duels in the mirrors, both seats imitated, the others against the thirteen
# decks' pilots, in minibatches of 256 with no duels in flight, every game seen once and
# its result at four of its positions, a checkpoint every COPY_STEP. Two changes: twice
# as wide (--embedding-dim EMBEDDING --hidden-dim HIDDEN), and at 3e-4 to HALF updates and
# at SLOW_RATE after them, from the same state and on the same games. Run from the
# repository root on the CUDA machine, before the pins move: a copy from recordings stops
# at the first game that plays again with another digest. PARTS names what is made, in
# its order:
#
# copy. RUNS/copy, from the recordings in RECORDS, each made by tools/record_champion.py;
# an archive RECORDS/DECK.tar is unpacked beside itself.
#
# measure. Into OUT, at the copy's last checkpoint: head to head against
# seventy-one-champions' copy (REFERENCE_COPY) on the pairings of that record's checks
# (CHECK_PAIRINGS), each game played once with each copy in the learner's seat
# (heads/copy.plays.json and .held.json); each copy as those checks measured that copy
# (checks/wide and checks/taught): head to head against each teacher in its mirror on
# CHECK_DEALS deals from both seats, and against all71 on the checks' pairings; the copy
# head to head against each teacher in its mirror on every deal of its suite
# (teachers/DECK), beside that copy's on the same deals (REFERENCE/heads-taught); the
# copy's metrics; then the report (tools/seventy_one_wide_report.py), beside that copy's
# metrics (REFERENCE/taught.metrics.jsonl). The mirrors of HEAVY, which filled the card's
# memory at seven at once in that copy's measure, are played one at a time after the
# others.
#
# AT_ONCE: how many of the measurements go at a time, through an MPS server on a card
# (experiments/commands/at-once.sh), what each writes in LOGS; 1, the default, one after
# the other.
#
# A copy that was stopped goes on from its last checkpoint when this is started again, at
# the rate of the half it stands in, and a measurement made is not made again; one that
# holds a failed case is played again.
#
# By reckoning on the CUDA machine: the copy 36 to 44 hours, the measure some 2.
#   AT_ONCE=4 bash experiments/commands/seventy-one-wide.sh
set -euo pipefail
DEVICE=${DEVICE:-cuda}
WORKERS=${WORKERS:-24}
THREADS=${THREADS:-20}
SLOTS=${SLOTS:-16}
DUELS=${DUELS:-520}
PARTS=${PARTS:-copy measure}
# Each deck and its run's last checkpoint, which teaches it.
LIST=${LIST:-experiments/commands/champion-recordings.list}
RECORDS=${RECORDS:-../records}
THIRTEEN=${THIRTEEN:-blackwing,lightsworn,gishki,burn,crystal,dragunity,gladiator,heroes,infernity,monarch,morphtronic,rock-block,spellcaster}
COPY=${COPY:-11360}
COPY_STEP=${COPY_STEP:-142}
# The rate falls after HALF updates, to SLOW_RATE.
HALF=${HALF:-5680}
SLOW_RATE=${SLOW_RATE:-1e-4}
# Twice today's widths: a card's identity and a decision.
EMBEDDING=${EMBEDDING:-192}
HIDDEN=${HIDDEN:-384}
ITSELF=${ITSELF:-260}
SEED=${SEED:-2012}
COPY_SLOTS=${COPY_SLOTS:-26}
CHECK_DEALS=${CHECK_DEALS:-128}
# The fit's error from stretches of so many updates, one a deck, and its last window of
# so many updates a deck.
STRETCH=${STRETCH:-71}
LATELY=${LATELY:-20}
CHECK_PAIRINGS=${CHECK_PAIRINGS:-eval_suites/seventy-one-decks-check-pairings-checkpoint.json}
HEAVY=${HEAVY:-cloudian exodia}
NETWORK=${NETWORK:-runs/seventy-one-decks/all71/checkpoints/model-27599.pt}
# seventy-one-champions' copy, and its measurements: its metrics and it head to head
# against each teacher in its mirror.
REFERENCE_COPY=${REFERENCE_COPY:-runs/seventy-one-champions/taught/checkpoints/model-11359.pt}
REFERENCE=${REFERENCE:-analysis/seventy-one-champions}
RUNS=${RUNS:-runs/seventy-one-wide}
OUT=${OUT:-analysis/seventy-one-wide}
SUITES=${SUITES:-eval_suites}
# The interpreter of a machine that has no uv: the benchmark's setup leaves one.
PYTHON=${PYTHON:-}
LOGS=${LOGS:-$OUT/logs}
# shellcheck source=experiments/commands/at-once.sh
source "$(dirname "${BASH_SOURCE[0]}")/at-once.sh"

for part in $PARTS; do
  case $part in
    copy | measure) ;;
    *)
      echo "PARTS names copy and measure, not $part" >&2
      exit 1
      ;;
  esac
done
if ((COPY % COPY_STEP || HALF % COPY_STEP || HALF <= 0 || HALF >= COPY)); then
  echo "COPY and HALF are multiples of COPY_STEP, HALF before COPY" >&2
  exit 1
fi
mapfile -t decks < <(grep -v '^#' "$LIST" | awk 'NF {print $1}')
DECKS=$(IFS=,; echo "${decks[*]}")
needed=()
if [[ " $PARTS " == *" measure "* ]]; then
  needed+=("$NETWORK" "$CHECK_PAIRINGS" "$REFERENCE_COPY" "$REFERENCE/taught.metrics.jsonl")
  for deck in "${decks[@]}"; do
    needed+=("$SUITES/$deck-against-checkpoint.json")
    needed+=("$REFERENCE/heads-taught/$deck.plays.json" "$REFERENCE/heads-taught/$deck.held.json")
  done
fi
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

# checkpoint RUN FIRST POINT: the checkpoint of the run at RUN after POINT updates past
# the FIRST it began with.
checkpoint() {
  printf '%s/checkpoints/model-%05d.pt' "$1" $(($2 + $3 - 1))
}
# teacher DECK: the run that teaches DECK.
teacher() {
  awk -v deck="$1" '$1 == deck {print $2}' "$LIST"
}
# unpacked: each deck's recording a folder of RECORDS, its archive unpacked beside
# itself where it is not yet, whole or not at all.
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
# went: how many updates the copy has made, by its latest checkpoint.
went() {
  local latest
  if ! ls "$RUNS/copy"/checkpoints/model-*.pt > /dev/null 2>&1; then
    echo 0
    return
  fi
  latest=$(basename "$(ls "$RUNS/copy"/checkpoints/model-*.pt | sort | tail -1)" .pt)
  echo $((10#${latest#model-} + 1))
}
# failed FILE: whether evaluate-suite, which goes on past a case that fails, left a
# case's error in FILE where its games would be.
failed() {
  grep -q '"error_type"' "$1"
}
# played FILE SUITE OPTION...: what the options name on a suite, with how each game
# ended, into FILE.json, at once with others where AT_ONCE says, or then and there once
# `alone` is set.
played() {
  # A file being played beside others is not there yet: it is not played twice. One
  # that holds a failed case is played again.
  if { [ ! -f "$1.json" ] || failed "$1.json"; } && [ -z "${queued[$1]:-}" ]; then
    queued[$1]=1
    if ((alone)); then
      playing "$@"
    else
      at_once "$(basename "$(dirname "$1")")-$(basename "$1")" playing "$@"
    fi
  fi
}
declare -A queued
alone=0
# playing FILE SUITE OPTION...: what played does, then and there.
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
# pair FILE FIRST SECOND SUITE: FIRST in the learner's seat against SECOND, then the
# other way round, into FILE.plays.json and FILE.held.json.
pair() {
  local file=$1 one=$2 other=$3 suite=$4
  played "$file.plays" "$suite" "$one" --opponent-checkpoint "$other"
  played "$file.held" "$suite" "$other" --opponent-checkpoint "$one"
}
# mirrors DECK: the copy and seventy-one-champions' copy against DECK's teacher in its
# mirror, on the checks' deals, and the copy on every deal of its suite.
mirrors() {
  local deck=$1 taught
  taught=$(teacher "$deck")
  pair "$OUT/checks/wide/$deck" "$copied" "$taught" "$OUT/checks/suites/$deck.json"
  pair "$OUT/checks/taught/$deck" "$REFERENCE_COPY" "$taught" "$OUT/checks/suites/$deck.json"
  pair "$OUT/teachers/$deck" "$copied" "$taught" "$SUITES/$deck-against-checkpoint.json"
}

copied=$(checkpoint "$RUNS/copy" 0 "$COPY")
copying=(
  --device "$DEVICE" --workers "$WORKERS" --pool-threads "$THREADS"
  --run "$RUNS/copy" --learner "$DECKS" --opponents "$THIRTEEN" --duels "$DUELS"
  --eval-every 0 --keep-checkpoints "$COPY_STEP" --seed "$SEED" --ppo-epochs 1
  --pool-slots "$COPY_SLOTS" --imitation-only-updates 1000000 --imitation-value-weight 0.5
  --imitation-value-positions 4 --minibatch 256 --duels-in-flight 0
  --self-play-duels "$ITSELF" --self-play-pairs mirrors --copy-records "$RECORDS"
  --embedding-dim "$EMBEDDING" --hidden-dim "$HIDDEN"
)

if [[ " $PARTS " == *" copy "* ]]; then
  unpacked
  at=$(went)
  if ((at < HALF)); then
    echo "$(date +%H:%M) the copy's first half begins at $at"
    if ((at == 0)); then
      "${python[@]}" -m ygo_training train "${copying[@]}" --updates "$HALF"
    else
      "${python[@]}" -m ygo_training train "${copying[@]}" --continue --updates $((HALF - at))
    fi
    at=$(went)
  fi
  # A copy that goes on deals the recordings from where it stood
  # (`Orchestrator._records_caught_up`): the same games as seventy-one-champions' copy.
  if ((at < COPY)); then
    echo "$(date +%H:%M) the copy's second half begins at $at, at $SLOW_RATE"
    "${python[@]}" -m ygo_training train "${copying[@]}" --continue \
      --updates $((COPY - at)) --resume-learning-rate "$SLOW_RATE"
  fi
  echo "$(date +%H:%M) the copy is made"
fi

if [[ " $PARTS " == *" measure "* ]]; then
  for path in "$copied" $(for deck in "${decks[@]}"; do teacher "$deck"; done); do
    if [ ! -f "$path" ]; then
      echo "Missing $path: the copy is measured when it and its teachers are made" >&2
      exit 1
    fi
  done
  "${python[@]}" tools/seventy_one_checks.py suites --decks "$DECKS" --suites "$SUITES" \
    --games "$CHECK_DEALS" --into "$OUT/checks/suites"
  pair "$OUT/heads/copy" "$copied" "$REFERENCE_COPY" "$CHECK_PAIRINGS"
  played "$OUT/checks/wide/pairings" "$CHECK_PAIRINGS" "$copied" --opponent-checkpoint "$NETWORK"
  played "$OUT/checks/taught/pairings" "$CHECK_PAIRINGS" "$REFERENCE_COPY" \
    --opponent-checkpoint "$NETWORK"
  for deck in "${decks[@]}"; do
    [[ " $HEAVY " == *" $deck "* ]] || mirrors "$deck"
  done
  all_done
  alone=1
  for deck in "${decks[@]}"; do
    [[ " $HEAVY " != *" $deck "* ]] || mirrors "$deck"
  done
  cp "$RUNS/copy/metrics.jsonl" "$OUT/wide.metrics.jsonl"
  cp "$RUNS/copy/run.json" "$OUT/wide.run.json"
  "${python[@]}" tools/seventy_one_wide_report.py "$OUT" --reference "$REFERENCE" \
    --decks "$DECKS" --half "$HALF" --copy "$COPY" --stretch "$STRETCH" --lately "$LATELY" \
    --output "$OUT/report.json" |
    tee "$OUT/report.txt"
  echo "The copy is measured: $OUT/report.txt"
fi
