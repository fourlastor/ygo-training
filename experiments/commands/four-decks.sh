#!/usr/bin/env bash
# A network of four decks (experiments/2026-10-24-four-decks.md, its part 2):
# the four champions imitated into one network of Blackwing, Cyber,
# Lightsworn and Infernity, then its pairings trained in two arms of as many
# updates, one with every pairing as often as any other and one with each
# pairing weighted by how far behind it is against all71. Run from the
# repository root on the CUDA machine. PARTS names what is made, in its
# order:
#
# copy. taught: one network of the four decks from scratch imitating each
# deck's champion (TEACHERS, DECK=CHECKPOINT) by its first choice, COPY
# updates of DUELS duels, ITSELF of an update's duels the champions against
# themselves in their own mirrors, both seats imitated, the others against
# the thirteen decks' pilots, at 3e-4, in minibatches of 256 with no duels in
# flight, every game seen once and its result at four of its positions; the
# teachers' games in two groups and from CUDA graphs; a checkpoint every
# COPY_STEP.
#
# uniform. PPO on from the copy's last checkpoint for UPDATES updates at
# RATE, ITSELF of an update's duels against itself on every ordered pairing
# of its decks as often, the others against the thirteen decks' pilots, with
# fast's settings; a checkpoint every STEP.
#
# weighted. The same, with its duels against itself shared out by weights
# (train --self-play-weights): before every STEP updates its last checkpoint
# plays every ordered pairing against all71 on the other side (QUICK), and
# tools/pairing_weights.py makes the next STEP's weights of that measurement
# and the one before, all kept in OUT/weighted-weights.
#
# measure. Into OUT: each teacher and the copy's last three checkpoints and
# those of CURVE against all71 in each mirror; each arm's last three
# checkpoints on every ordered pairing against all71 (PAIRINGS), and every
# other checkpoint of the uniform arm on QUICK; the uniform arm after FAR
# updates in Blackwing against Cyber both ways (CROSS); the copy's and each
# arm's last checkpoint head to head against each champion in its mirror;
# the runs' metrics; then the report (tools/four_decks_report.py).
#
# short. The copy shorter (experiments/2026-10-26-shorter-copy.md): the
# uniform arm's PPO on from the copy's checkpoint after SHORT updates, as
# many updates and as the uniform arm was made.
#
# short-measure. Into OUT, as the uniform arm was measured: the short arm's
# last three checkpoints on PAIRINGS, every one it keeps and the copy's it
# went on from on QUICK, its last head to head against each champion in its
# mirror, its metrics; then its report beside the uniform arm
# (tools/shorter_copy_report.py). The trainer numbers its updates on from
# the copy's and keeps a checkpoint where that count closes STEP, and the
# last: the short arm's at 80, 320 and so on past the copy's 640, its last
# three after 2,480, 2,720 and 2,880.
#
# About 20 hours on the CUDA machine by reckoning: the copy 5, each arm 7.
#   PARTS=copy bash experiments/commands/four-decks.sh
set -euo pipefail
DEVICE=${DEVICE:-cuda}
WORKERS=${WORKERS:-24}
THREADS=${THREADS:-20}
SLOTS=${SLOTS:-16}
DUELS=${DUELS:-520}
PARTS=${PARTS:-copy uniform weighted measure}
DECKS=${DECKS:-blackwing,cyber,lightsworn,infernity}
TEACHERS=${TEACHERS:-blackwing=runs/two-decks/blackwing/checkpoints/model-00959.pt cyber=runs/cyber-champion/cyber-old/checkpoints/model-00959.pt lightsworn=runs/four-decks/lightsworn/checkpoints/model-00959.pt infernity=runs/four-decks/infernity/checkpoints/model-00959.pt}
THIRTEEN=${THIRTEEN:-blackwing,lightsworn,gishki,burn,crystal,dragunity,gladiator,heroes,infernity,monarch,morphtronic,rock-block,spellcaster}
COPY=${COPY:-960}
COPY_STEP=${COPY_STEP:-80}
UPDATES=${UPDATES:-2880}
STEP=${STEP:-240}
ITSELF=${ITSELF:-260}
RATE=${RATE:-1e-4}
SEED=${SEED:-2012}
# The copy's teachers' games: an update's duels at once, in two groups, the
# calls from CUDA graphs of up to so many decisions (decided on 2026-10-25).
COPY_SLOTS=${COPY_SLOTS:-26}
COPY_GRAPHS=${COPY_GRAPHS:-130}
# fast's settings: two updates' duels in flight, the pool's slots to hold
# them, minibatches of 512, the calls as they are in two groups.
IN_FLIGHT=${IN_FLIGHT:-1040}
FAST_SLOTS=${FAST_SLOTS:-52}
MINIBATCH=${MINIBATCH:-512}
NETWORK=${NETWORK:-runs/seventy-one-decks/all71/checkpoints/model-27599.pt}
RUNS=${RUNS:-runs/four-decks}
OUT=${OUT:-analysis/four-decks/network}
SUITES=${SUITES:-eval_suites}
PAIRINGS=${PAIRINGS:-eval_suites/four-decks-pairings-checkpoint.json}
QUICK=${QUICK:-eval_suites/four-decks-pairings-quick-checkpoint.json}
CROSS=${CROSS:-eval_suites/blackwing-against-cyber-checkpoint.json}
# The uniform arm's updates at which it has played Blackwing against Cyber as
# often as specialist distillation's cross pairings after 240, and where
# those were measured; the copy's earlier checkpoints measured beside.
FAR=${FAR:-1920}
# The copy's checkpoint the short arm goes on from.
SHORT=${SHORT:-640}
DISTILLED=${DISTILLED:-analysis/specialist-distillation/cross}
CURVE=${CURVE:-160 320 480 640}
# The interpreter of a machine that has no uv: the benchmark's setup leaves one.
PYTHON=${PYTHON:-}

for part in $PARTS; do
  case $part in
    copy | uniform | weighted | measure | short | short-measure) ;;
    *)
      echo "PARTS names copy, uniform, weighted, measure, short and short-measure, not $part" >&2
      exit 1
      ;;
  esac
done
if ((UPDATES % STEP || UPDATES < 3 * STEP || COPY % COPY_STEP || COPY < 3 * COPY_STEP)); then
  echo "UPDATES is a multiple of STEP and COPY of COPY_STEP, three of them at least" >&2
  exit 1
fi
read -r -a decks <<< "${DECKS//,/ }"
teach=()
for given in $TEACHERS; do
  teach+=(--imitation-teacher "$given")
  if [ ! -f "${given#*=}" ]; then
    echo "Missing ${given#*=}" >&2
    exit 1
  fi
done
for path in "$NETWORK" "$PAIRINGS" "$QUICK" "$CROSS" \
  $(printf "$SUITES/%s-against-checkpoint.json " "${decks[@]}"); do
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
# kept FIRST UPDATES EVERY: the points, updates past the FIRST it began
# with, of the checkpoints a run of UPDATES keeps with --keep-checkpoints
# EVERY: where the trainer's count, which went on from FIRST, closes EVERY
# (RunRepository.thin), and the last.
kept() {
  local point
  for ((point = 1; point < $2; point++)); do
    ((($1 + point) % $3)) || echo "$point"
  done
  echo "$2"
}
# teacher DECK: the champion that teaches DECK.
teacher() {
  local given
  for given in $TEACHERS; do
    if [ "${given%%=*}" = "$1" ]; then
      echo "${given#*=}"
    fi
  done
}
# played FILE SUITE OPTION...: what the options name on a suite, with how
# each game ended, into FILE.json.
played() {
  local file=$1 suite=$2
  shift 2
  if [ ! -f "$file.json" ]; then
    mkdir -p "$(dirname "$file")"
    "${python[@]}" -m ygo_training evaluate-suite "$@" --device "$DEVICE" \
      --workers "$WORKERS" --pool-threads "$THREADS" --pool-slots "$SLOTS" --seed "$SEED" \
      --suite "$suite" --output "$file.json.part" > /dev/null
    mv "$file.json.part" "$file.json"
    echo "$(date +%H:%M) $file"
  fi
}
# pair FILE FIRST SECOND SUITE: FIRST in the learner's seat against SECOND,
# then the other way round, into FILE.plays.json and FILE.held.json.
pair() {
  local file=$1 one=$2 other=$3 suite=$4
  played "$file.plays" "$suite" "$one" --opponent-checkpoint "$other"
  played "$file.held" "$suite" "$other" --opponent-checkpoint "$one"
}

# made NAME FIRST UPDATES KEPT OPTION...: the run NAME of the four decks, to
# UPDATES updates after the FIRST it began with, from where it stands or from
# its start, with a checkpoint every KEPT updates.
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

copied=$(checkpoint "$RUNS/taught" 0 "$COPY")
arm=(
  --pool-slots "$FAST_SLOTS" --self-play-duels "$ITSELF" --rollout-estimates kept
  --duels-in-flight "$IN_FLIGHT" --minibatch "$MINIBATCH" --inference-graphs 0 --pool-groups 2
  --resume "$copied" --resume-learning-rate "$RATE"
)

if [[ " $PARTS " == *" copy "* ]]; then
  echo "$(date +%H:%M) taught begins"
  made taught 0 "$COPY" "$COPY_STEP" --pool-slots "$COPY_SLOTS" --pool-groups 2 \
    --inference-graphs "$COPY_GRAPHS" --imitation-only-updates 1000000 \
    --imitation-value-weight 0.5 --imitation-value-positions 4 --minibatch 256 \
    --duels-in-flight 0 --self-play-duels "$ITSELF" --self-play-pairs mirrors "${teach[@]}"
  echo "$(date +%H:%M) taught ends"
fi

if [[ " $PARTS " == *" uniform "* ]]; then
  echo "$(date +%H:%M) uniform begins"
  made uniform "$COPY" "$UPDATES" "$STEP" "${arm[@]}"
  echo "$(date +%H:%M) uniform ends"
fi

if [[ " $PARTS " == *" weighted "* ]]; then
  echo "$(date +%H:%M) weighted begins"
  weights=$RUNS/weighted-weights.json
  before=()
  for ((point = 0; point < UPDATES; point += STEP)); do
    at=$copied
    ((point == 0)) || at=$(checkpoint "$RUNS/weighted" "$COPY" "$point")
    measured=$OUT/weighted-weights/$point
    played "$measured" "$QUICK" "$at" --opponent-checkpoint "$NETWORK"
    "${python[@]}" tools/pairing_weights.py "$measured.json" --decks "$DECKS" \
      --output "$measured.weights.json" --shares "$measured.shares.json" "${before[@]}"
    cp "$measured.weights.json" "$weights"
    made weighted "$COPY" $((point + STEP)) "$STEP" "${arm[@]}" --self-play-weights "$weights"
    before=(--before "$measured.shares.json")
  done
  echo "$(date +%H:%M) weighted ends"
fi

if [[ " $PARTS " == *" short "* ]]; then
  echo "$(date +%H:%M) short begins"
  # The uniform arm's options, on from the copy's checkpoint after SHORT.
  shorter=()
  for option in "${arm[@]}"; do
    if [ "$option" = "$copied" ]; then
      option=$(checkpoint "$RUNS/taught" 0 "$SHORT")
    fi
    shorter+=("$option")
  done
  made short "$SHORT" "$UPDATES" "$STEP" "${shorter[@]}"
  echo "$(date +%H:%M) short ends"
fi

if [[ " $PARTS " == *" measure "* ]]; then
  last=$(checkpoint "$RUNS/uniform" "$COPY" "$UPDATES")
  for path in "$copied" "$last" "$(checkpoint "$RUNS/weighted" "$COPY" "$UPDATES")"; do
    if [ ! -f "$path" ]; then
      echo "Missing $path: the runs are measured when they are made" >&2
      exit 1
    fi
  done
  for deck in "${decks[@]}"; do
    mirror=$SUITES/$deck-against-checkpoint.json
    pair "$OUT/teachers/$deck" "$(teacher "$deck")" "$NETWORK" "$mirror"
    for ((point = COPY - 2 * COPY_STEP; point <= COPY; point += COPY_STEP)); do
      pair "$OUT/taught-$deck/$point" "$(checkpoint "$RUNS/taught" 0 "$point")" "$NETWORK" "$mirror"
    done
    for point in $CURVE; do
      pair "$OUT/taught-curve-$deck/$point" "$(checkpoint "$RUNS/taught" 0 "$point")" \
        "$NETWORK" "$mirror"
    done
    for who in taught uniform weighted; do
      first=$COPY
      at=$(checkpoint "$RUNS/$who" "$first" "$UPDATES")
      [ "$who" != taught ] || at=$copied
      pair "$OUT/heads-$who/$deck" "$at" "$(teacher "$deck")" "$mirror"
    done
  done
  for who in uniform weighted; do
    for ((point = UPDATES - 2 * STEP; point <= UPDATES; point += STEP)); do
      played "$OUT/$who-pairings/$point" "$PAIRINGS" "$(checkpoint "$RUNS/$who" "$COPY" "$point")" \
        --opponent-checkpoint "$NETWORK"
    done
  done
  for ((point = STEP; point <= UPDATES; point += STEP)); do
    played "$OUT/uniform-quick/$point" "$QUICK" "$(checkpoint "$RUNS/uniform" "$COPY" "$point")" \
      --opponent-checkpoint "$NETWORK"
  done
  played "$OUT/weighted-weights/$UPDATES" "$QUICK" "$(checkpoint "$RUNS/weighted" "$COPY" "$UPDATES")" \
    --opponent-checkpoint "$NETWORK"
  pair "$OUT/uniform-cross/$FAR" "$(checkpoint "$RUNS/uniform" "$COPY" "$FAR")" "$NETWORK" "$CROSS"
  for name in taught uniform weighted; do
    cp "$RUNS/$name/metrics.jsonl" "$OUT/$name.metrics.jsonl"
    cp "$RUNS/$name/run.json" "$OUT/$name.run.json"
  done
  "${python[@]}" tools/four_decks_report.py "$OUT" --decks "$DECKS" --distilled "$DISTILLED" \
    --far "$FAR" --output "$OUT/report.json" | tee "$OUT/report.txt"
  echo "The measurements and what they say are in $OUT."
fi

if [[ " $PARTS " == *" short-measure "* ]]; then
  last=$(checkpoint "$RUNS/short" "$SHORT" "$UPDATES")
  if [ ! -f "$last" ]; then
    echo "Missing $last: the short arm is measured when it is made" >&2
    exit 1
  fi
  for deck in "${decks[@]}"; do
    pair "$OUT/heads-short/$deck" "$last" "$(teacher "$deck")" "$SUITES/$deck-against-checkpoint.json"
  done
  mapfile -t points < <(kept "$SHORT" "$UPDATES" "$STEP")
  for point in "${points[@]: -3}"; do
    played "$OUT/short-pairings/$point" "$PAIRINGS" "$(checkpoint "$RUNS/short" "$SHORT" "$point")" \
      --opponent-checkpoint "$NETWORK"
  done
  for point in 0 "${points[@]}"; do
    at=$(checkpoint "$RUNS/taught" 0 "$SHORT")
    ((point == 0)) || at=$(checkpoint "$RUNS/short" "$SHORT" "$point")
    played "$OUT/short-quick/$point" "$QUICK" "$at" --opponent-checkpoint "$NETWORK"
  done
  cp "$RUNS/short/metrics.jsonl" "$OUT/short.metrics.jsonl"
  cp "$RUNS/short/run.json" "$OUT/short.run.json"
  "${python[@]}" tools/shorter_copy_report.py "$OUT" --decks "$DECKS" \
    --output "$OUT/short-report.json" | tee "$OUT/short-report.txt"
  echo "The measurements and what they say are in $OUT."
fi
