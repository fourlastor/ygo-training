#!/usr/bin/env bash
# The board as its cards from scratch, at fourteen decks
# (experiments/2026-11-02-expanded-pool.md, its section beside the batches): two
# networks of the fourteen decks of experiments/2026-10-26-fourteen-decks.md from
# scratch at SEED, pooled and tokens, the same in everything but the board, which
# tokens takes as its cards at both stages (--board-representation
# individual-tokens). Run from the repository root on the CUDA machine. PARTS names
# what is made, in its order:
#
# copy. Each arm's copy, ARM-copy, the two at once: each deck's games its champion's
# recorded games (RECORDS/DECK, the seventy-one's recordings, made by
# tools/record_champion.py; an archive RECORDS/DECK.tar is unpacked beside itself)
# played again with its answers and imitated, COPY updates of DUELS duels, 160 a
# deck, ITSELF of an update's duels in the mirrors, both seats imitated, the others
# against the thirteen decks' pilots, at 3e-4, in minibatches of 256 with no duels in
# flight, every game seen once and its result at four of its positions; a checkpoint
# every COPY_STEP.
#
# ppo. Each arm on from its copy's last checkpoint, the two at once, as the fourteen
# decks' quarter arm: UPDATES updates of DUELS duels at RATE with fast's settings,
# every duel against itself and none against a pilot, a quarter of them in the
# mirrors (--self-play-weights from RUNS/quarter.weights.json: each mirror 13, each
# other ordered pairing 3); a checkpoint where the trainer's count, which goes on
# from the copy's, closes STEP, and the last.
#
# measure. Into OUT, on SUITE (each ordered pairing of the fourteen decks, 64 games),
# each game played once with each arm in the learner's seat (FOLDER/POINT.plays.json
# with tokens there, then FOLDER/POINT.held.json): tokens head to head against pooled
# at the last three checkpoints of PPO (heads/last), at the copies' last checkpoints
# (heads/copy) and after each of ALONG (heads/POINT); each arm's last three
# checkpoints against all71 (ARM-all71/POINT.json); the runs' metrics; then the
# report (tools/fourteen_tokens_report.py), beside the fourteen decks' quarter arm
# against all71 on the same deals (QUARTER).
#
# AT_ONCE: how many of the measurements go at a time, through an MPS server on a card
# (experiments/commands/at-once.sh), what each writes in LOGS; the arms are made
# ARMS_AT_ONCE at a time, two unless said.
#
# The trainer numbers an arm's updates of PPO on from its copy's and keeps a
# checkpoint where that count closes STEP: after 160, 400 and so on to 4,240 updates
# past the copy's 2,240, and after 4,320, its last three after 4,000, 4,240 and 4,320.
# A run that was stopped goes on from its last checkpoint when this is started again,
# and a measurement made is not made again.
#
# By reckoning on the CUDA machine: the two copies some 10 hours, PPO 14, the
# measurements 2.
#   AT_ONCE=4 bash experiments/commands/fourteen-tokens.sh
set -euo pipefail
DEVICE=${DEVICE:-cuda}
WORKERS=${WORKERS:-24}
THREADS=${THREADS:-20}
SLOTS=${SLOTS:-16}
DUELS=${DUELS:-520}
PARTS=${PARTS:-copy ppo measure}
ARMS_AT_ONCE=${ARMS_AT_ONCE:-2}
DECKS=${DECKS:-blackwing,cyber,lightsworn,infernity,gishki,burn,crystal,dragunity,gladiator,heroes,monarch,morphtronic,rock-block,spellcaster}
THIRTEEN=${THIRTEEN:-blackwing,lightsworn,gishki,burn,crystal,dragunity,gladiator,heroes,infernity,monarch,morphtronic,rock-block,spellcaster}
RECORDS=${RECORDS:-../records}
COPY=${COPY:-2240}
COPY_STEP=${COPY_STEP:-80}
COPY_SLOTS=${COPY_SLOTS:-26}
UPDATES=${UPDATES:-4320}
STEP=${STEP:-240}
ITSELF=${ITSELF:-260}
RATE=${RATE:-1e-4}
SEED=${SEED:-2012}
# The quarter arm's weights of the duels against itself: each mirror, each other
# ordered pairing.
MIRROR_WEIGHT=${MIRROR_WEIGHT:-13}
OTHER_WEIGHT=${OTHER_WEIGHT:-3}
# fast's settings, the fourteen decks' arms': two updates' duels in flight, the
# pool's slots to hold them, minibatches of 512, the calls as they are in two groups.
IN_FLIGHT=${IN_FLIGHT:-1040}
FAST_SLOTS=${FAST_SLOTS:-52}
MINIBATCH=${MINIBATCH:-512}
# The head to head along PPO, besides its last three checkpoints: points it keeps.
ALONG=${ALONG:-1360 2800}
NETWORK=${NETWORK:-runs/seventy-one-decks/all71/checkpoints/model-27599.pt}
RUNS=${RUNS:-runs/fourteen-tokens}
OUT=${OUT:-analysis/fourteen-tokens}
SUITE=${SUITE:-eval_suites/fourteen-decks-pairings-quick-checkpoint.json}
# The fourteen decks' quarter arm against all71 on SUITE, its last three checkpoints.
QUARTER=${QUARTER:-analysis/fourteen-decks/quarter-quick}
# The interpreter of a machine that has no uv: the benchmark's setup leaves one.
PYTHON=${PYTHON:-}
LOGS=${LOGS:-$OUT/logs}
# shellcheck source=experiments/commands/at-once.sh
source "$(dirname "${BASH_SOURCE[0]}")/at-once.sh"
MEASURE_AT_ONCE=$AT_ONCE
TOKENS=(--board-representation individual-tokens)

for part in $PARTS; do
  case $part in
    copy | ppo | measure) ;;
    *)
      echo "PARTS names copy, ppo and measure, not $part" >&2
      exit 1
      ;;
  esac
done
if ((UPDATES < 3 * STEP || COPY % COPY_STEP || COPY < 3 * COPY_STEP)); then
  echo "UPDATES is three STEP at least, COPY a multiple of COPY_STEP and three of them" >&2
  exit 1
fi
read -r -a decks <<< "${DECKS//,/ }"
for path in "$NETWORK" "$SUITE"; do
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

# checkpoint RUN FIRST POINT: the checkpoint of the run at RUN after POINT updates
# past the FIRST it began with.
checkpoint() {
  printf '%s/checkpoints/model-%05d.pt' "$1" $(($2 + $3 - 1))
}
# kept FIRST UPDATES EVERY: the points, updates past the FIRST it began with, of the
# checkpoints a run of UPDATES keeps with --keep-checkpoints EVERY: where the
# trainer's count, which went on from FIRST, closes EVERY (RunRepository.thin), and
# the last.
kept() {
  local point
  for ((point = 1; point < $2; point++)); do
    ((($1 + point) % $3)) || echo "$point"
  done
  echo "$2"
}
mapfile -t points < <(kept "$COPY" "$UPDATES" "$STEP")
last=("${points[@]: -3}")
for point in $ALONG; do
  if [[ " ${points[*]} " != *" $point "* ]] || ((point >= ${last[0]})); then
    echo "ALONG names points PPO keeps before its last three, not $point" >&2
    exit 1
  fi
done
# shaped ARM: what the arm's training is told of its shape.
shaped() {
  [ "$1" != tokens ] || echo "${TOKENS[@]}"
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
# made NAME FIRST UPDATES KEPT OPTION...: the run NAME of the fourteen decks, to
# UPDATES updates after the FIRST it began with, from where it stands or from its
# start, with a checkpoint every KEPT updates.
made() {
  local name=$1 first=$2 updates=$3 kept=$4 last_ latest
  shift 4
  last_=$(checkpoint "$RUNS/$name" "$first" "$updates")
  local arguments=(
    --device "$DEVICE" --workers "$WORKERS" --pool-threads "$THREADS"
    --run "$RUNS/$name" --learner "$DECKS" --opponents "$THIRTEEN" --duels "$DUELS"
    --eval-every 0 --keep-checkpoints "$kept" --seed "$SEED" --ppo-epochs 1
  )
  if ! ls "$RUNS/$name"/checkpoints/model-*.pt > /dev/null 2>&1; then
    "${python[@]}" -m ygo_training train "${arguments[@]}" "$@" --updates "$updates"
  elif [ ! -f "$last_" ]; then
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
# weights FILE MIRROR OTHER: the weights of the duels against itself, each mirror
# MIRROR and each other ordered pairing of the decks OTHER, into FILE.
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
# failed FILE: whether evaluate-suite, which goes on past a case that fails, left a
# case's error in FILE where its games would be.
failed() {
  grep -q '"error_type"' "$1"
}
# played FILE OPTION...: SUITE as the options name, with how each game ended, into
# FILE.json, at once with others where AT_ONCE says.
played() {
  # A file being played beside others is not there yet: it is not played twice. One
  # that holds a failed case is played again.
  if { [ ! -f "$1.json" ] || failed "$1.json"; } && [ -z "${queued[$1]:-}" ]; then
    queued[$1]=1
    at_once "$(basename "$(dirname "$1")")-$(basename "$1")" playing "$@"
  fi
}
declare -A queued
# playing FILE OPTION...: what played does, then and there.
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
# heads FILE TOKENS POOLED: the two checkpoints head to head, tokens in the learner's
# seat and then pooled, into FILE.plays.json and FILE.held.json. A checkpoint in the
# other seat is built as it was saved.
heads() {
  played "$1.plays" "$2" "${TOKENS[@]}" --opponent-checkpoint "$3"
  played "$1.held" "$3" --opponent-checkpoint "$2"
}

copied() { checkpoint "$RUNS/$1-copy" 0 "$COPY"; }
trained() { checkpoint "$RUNS/$1" "$COPY" "$2"; }

if [[ " $PARTS " == *" copy "* ]]; then
  unpacked
  AT_ONCE=$ARMS_AT_ONCE
  for arm in pooled tokens; do
    read -r -a shaping <<< "$(shaped "$arm")"
    at_once "$arm-copy" made "$arm-copy" 0 "$COPY" "$COPY_STEP" --pool-slots "$COPY_SLOTS" \
      --imitation-only-updates 1000000 --imitation-value-weight 0.5 \
      --imitation-value-positions 4 --minibatch 256 --duels-in-flight 0 \
      --self-play-duels "$ITSELF" --self-play-pairs mirrors --copy-records "$RECORDS" \
      "${shaping[@]}"
  done
  all_done
  AT_ONCE=$MEASURE_AT_ONCE
  echo "$(date +%H:%M) the copies are made"
fi

if [[ " $PARTS " == *" ppo "* ]]; then
  for arm in pooled tokens; do
    if [ ! -f "$(copied "$arm")" ]; then
      echo "Missing $(copied "$arm"): PPO goes on from the copy" >&2
      exit 1
    fi
  done
  weights "$RUNS/quarter.weights.json" "$MIRROR_WEIGHT" "$OTHER_WEIGHT"
  AT_ONCE=$ARMS_AT_ONCE
  for arm in pooled tokens; do
    read -r -a shaping <<< "$(shaped "$arm")"
    at_once "$arm" made "$arm" "$COPY" "$UPDATES" "$STEP" --pool-slots "$FAST_SLOTS" \
      --rollout-estimates kept --duels-in-flight "$IN_FLIGHT" --minibatch "$MINIBATCH" \
      --inference-graphs 0 --pool-groups 2 --self-play-duels "$DUELS" \
      --self-play-weights "$RUNS/quarter.weights.json" "${shaping[@]}" \
      --resume "$(copied "$arm")" --resume-learning-rate "$RATE"
  done
  all_done
  AT_ONCE=$MEASURE_AT_ONCE
  echo "$(date +%H:%M) PPO is made"
fi

if [[ " $PARTS " == *" measure "* ]]; then
  for arm in pooled tokens; do
    for path in "$(copied "$arm")" "$(trained "$arm" "$UPDATES")"; do
      if [ ! -f "$path" ]; then
        echo "Missing $path: the arms are measured when they are made" >&2
        exit 1
      fi
    done
  done
  for point in "${last[@]}"; do
    heads "$OUT/heads/last/$point" "$(trained tokens "$point")" "$(trained pooled "$point")"
  done
  for point in "${last[@]}"; do
    played "$OUT/pooled-all71/$point" "$(trained pooled "$point")" --opponent-checkpoint "$NETWORK"
    played "$OUT/tokens-all71/$point" "$(trained tokens "$point")" "${TOKENS[@]}" \
      --opponent-checkpoint "$NETWORK"
  done
  heads "$OUT/heads/copy/$COPY" "$(copied tokens)" "$(copied pooled)"
  for point in $ALONG; do
    heads "$OUT/heads/$point/$point" "$(trained tokens "$point")" "$(trained pooled "$point")"
  done
  all_done
  for run in pooled-copy tokens-copy pooled tokens; do
    cp "$RUNS/$run/metrics.jsonl" "$OUT/$run.metrics.jsonl"
    cp "$RUNS/$run/run.json" "$OUT/$run.run.json"
  done
  beside=()
  [ ! -d "$QUARTER" ] || beside=(--quarter "$QUARTER")
  "${python[@]}" tools/fourteen_tokens_report.py "$OUT" --last "$(IFS=,; echo "${last[*]}")" \
    --along "copy,${ALONG// /,}" "${beside[@]}" --output "$OUT/report.json" | tee "$OUT/report.txt"
  echo "The measurements and what they say are in $OUT."
fi
