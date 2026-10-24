#!/usr/bin/env bash
# A network of two decks at the champions' duels of each mirror
# (experiments/2026-10-24-two-decks.md): the champion's recipe (the Cyber
# champion's, decks-01's before it) for one network of Blackwing and Cyber,
# and for Blackwing alone again at today's code. Run from the repository root
# on the CUDA machine. PARTS names what is made, in its order:
#
# arms. both, then blackwing. both: the pilots of both decks imitated from
# scratch, COPY updates a deck of DUELS against the thirteen decks' pilots,
# every game seen once and its result at four of its positions, at 3e-4, in
# minibatches of 256 with no duels in flight; then PPO on from the copy's
# last checkpoint for UPDATES updates for each ordered pair of its decks, at
# RATE, ITSELF of an update's duels against itself on every ordered pair of
# its decks as often (a deck against itself among them) and the others
# against the thirteen decks' pilots, with fast's settings (OPTIONS adds
# more); a checkpoint every STEP updates a pair. Each mirror has then as
# many games as a champion's, the two decks against each other twice that,
# and each pairing with a pilot twice a champion's. blackwing: the
# champion's recipe for Blackwing alone, COPY and UPDATES, a checkpoint
# every STEP. A finished run is skipped, an interrupted one goes on.
#
# measure. Into OUT, each a pair of files with the first network in the
# learner's seat and then the other, by the last three of the first's kept
# checkpoints: both against each deck's champion in its mirror (Blackwing
# alone, CYBER) at as many games of it; both and Blackwing alone against
# all71 (NETWORK) in the mirrors; decks-01 (DECKS01, folded) against
# Blackwing alone; both playing Blackwing against all71 playing Cyber
# (CROSS); the runs' metrics; then the report (tools/two_decks_report.py).
#
# About 6 to 9 hours on the CUDA machine by reckoning.
#   bash experiments/commands/two-decks.sh
set -euo pipefail
DEVICE=${DEVICE:-cuda}
WORKERS=${WORKERS:-24}
THREADS=${THREADS:-20}
SLOTS=${SLOTS:-16}
DUELS=${DUELS:-520}
PARTS=${PARTS:-arms measure}
THIRTEEN=${THIRTEEN:-blackwing,lightsworn,gishki,burn,crystal,dragunity,gladiator,heroes,infernity,monarch,morphtronic,rock-block,spellcaster}
COPY=${COPY:-240}
UPDATES=${UPDATES:-720}
STEP=${STEP:-120}
ITSELF=${ITSELF:-260}
RATE=${RATE:-1e-4}
SEED=${SEED:-2012}
# fast's settings: two updates' duels in flight, the pool's slots to hold
# them, minibatches of 512, the calls as they are in two groups.
IN_FLIGHT=${IN_FLIGHT:-1040}
FAST_SLOTS=${FAST_SLOTS:-52}
MINIBATCH=${MINIBATCH:-512}
OPTIONS=${OPTIONS:-}
NETWORK=${NETWORK:-runs/seventy-one-decks/all71/checkpoints/model-27599.pt}
# The Cyber champion, its copy as long as COPY, and its own games against
# all71 in the Cyber mirror; the two Cyber runs head to head.
CYBER=${CYBER:-runs/cyber-champion/cyber-old}
CYBER_ALONE=${CYBER_ALONE:-analysis/cyber-champion/cyber-old/mirror}
CALIBRATION=${CALIBRATION:-analysis/cyber-champion/calibration}
# decks-01's checkpoints folded for the Blackwing mirror.
DECKS01=${DECKS01:-runs/more-decks/decks-01/blackwing-mirror}
RUNS=${RUNS:-runs/two-decks}
OUT=${OUT:-analysis/two-decks}
BLACKWING_MIRROR=${BLACKWING_MIRROR:-eval_suites/blackwing-against-checkpoint.json}
CYBER_MIRROR=${CYBER_MIRROR:-eval_suites/cyber-against-checkpoint.json}
CROSS=${CROSS:-eval_suites/blackwing-against-cyber-checkpoint.json}
# The interpreter of a machine that has no uv: the benchmark's setup leaves one.
PYTHON=${PYTHON:-}
# The decks of both: a deck's share of an update's pilots falls with their
# number, and a mirror's share of its duels against itself with its square.
BOTH=blackwing,cyber
KINDS=2

for part in $PARTS; do
  if [ "$part" != arms ] && [ "$part" != measure ]; then
    echo "PARTS names arms and measure, not $part" >&2
    exit 1
  fi
done
if ((UPDATES % STEP || UPDATES < 3 * STEP || COPY % (KINDS * STEP))); then
  echo "UPDATES is a multiple of STEP, three of them at least, and COPY of $KINDS of them" >&2
  exit 1
fi
for path in "$NETWORK" "$BLACKWING_MIRROR" "$CYBER_MIRROR" "$CROSS"; do
  if [ ! -f "$path" ]; then
    echo "Missing $path" >&2
    exit 1
  fi
done
# checkpoint RUN FIRST POINT: the checkpoint of the run at RUN after POINT
# updates past the FIRST it began with.
checkpoint() {
  printf '%s/checkpoints/model-%05d.pt' "$1" $(($2 + $3 - 1))
}
points=()
for ((point = UPDATES - 2 * STEP; point <= UPDATES; point += STEP)); do
  points+=("$point")
done
for point in "${points[@]}"; do
  for path in "$(checkpoint "$CYBER" "$COPY" "$point")" \
    "$(printf '%s/model-%05d.pt' "$DECKS01" $((COPY + point - 1)))"; do
    if [ ! -f "$path" ]; then
      echo "Missing $path" >&2
      exit 1
    fi
  done
done
pinned=$(git submodule status --recursive)
if grep -q '^[^ ]' <<< "$pinned"; then
  echo "A submodule is not at its pinned revision: git submodule update --init --recursive" >&2
  exit 1
fi
python=(uv run python)
[ -z "$PYTHON" ] || python=("$PYTHON")
read -r -a options <<< "$OPTIONS"
cargo build --release
mkdir -p "$RUNS" "$OUT"

# made NAME FIRST UPDATES KEPT LEARNER OPTION...: the run NAME of the
# learner's decks LEARNER, to UPDATES updates after the FIRST it began with,
# from where it stands or from its start, with a checkpoint every KEPT
# updates.
made() {
  local name=$1 first=$2 updates=$3 kept=$4 learner=$5 last latest
  shift 5
  last=$(checkpoint "$RUNS/$name" "$first" "$updates")
  local arguments=(
    --device "$DEVICE" --workers "$WORKERS" --pool-threads "$THREADS"
    --run "$RUNS/$name" --learner "$learner" --opponents "$THIRTEEN" --duels "$DUELS"
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

# arm NAME LEARNER KINDS: the copy of the pilots of LEARNER's KINDS decks,
# and PPO on from it, as long as a champion's for each deck and each ordered
# pair of them.
arm() {
  local name=$1 learner=$2 kinds=$3
  local copy=$((kinds * COPY)) updates=$((kinds * kinds * UPDATES)) step=$((kinds * kinds * STEP))
  echo "$(date +%H:%M) $name begins"
  made "$name-copy" 0 "$copy" "$copy" "$learner" --pool-slots "$SLOTS" \
    --imitation-only-updates 1000000 --imitation-value-weight 0.5 \
    --imitation-value-positions 4 --minibatch 256 --duels-in-flight 0
  made "$name" "$copy" "$updates" "$step" "$learner" --pool-slots "$FAST_SLOTS" \
    --self-play-duels "$ITSELF" --rollout-estimates kept \
    --duels-in-flight "$IN_FLIGHT" --minibatch "$MINIBATCH" --inference-graphs 0 \
    --pool-groups 2 "${options[@]}" \
    --resume "$(checkpoint "$RUNS/$name-copy" 0 "$copy")" --resume-learning-rate "$RATE"
  echo "$(date +%H:%M) $name ends"
}

if [[ " $PARTS " == *" arms "* ]]; then
  arm both "$BOTH" "$KINDS"
  arm blackwing blackwing 1
  echo "The arms are made: $RUNS."
fi

if [[ " $PARTS " == *" measure "* ]]; then
  # suite FILE SUITE OPTION...: what the options name on a suite, with how
  # each game ended, into OUT/FILE.json.
  suite() {
    local file=$1 played=$2
    shift 2
    if [ ! -f "$OUT/$file.json" ]; then
      "${python[@]}" -m ygo_training evaluate-suite "$@" --device "$DEVICE" \
        --workers "$WORKERS" --pool-threads "$THREADS" --pool-slots "$SLOTS" --seed "$SEED" \
        --suite "$played" --output "$OUT/$file.json.part" > /dev/null
      mv "$OUT/$file.json.part" "$OUT/$file.json"
      echo "$(date +%H:%M) $file"
    fi
  }
  # pair FOLDER STEM FIRST SECOND SUITE: FIRST in the learner's seat against
  # SECOND, then the other way round.
  pair() {
    local folder=$1 stem=$2 first=$3 second=$4 played=$5
    mkdir -p "$OUT/$folder"
    suite "$folder/$stem.plays" "$played" "$first" --opponent-checkpoint "$second"
    suite "$folder/$stem.held" "$played" "$second" --opponent-checkpoint "$first"
  }
  for point in "${points[@]}"; do
    alone=$(checkpoint "$RUNS/blackwing" "$COPY" "$point")
    within=$((KINDS * KINDS * point))
    both=$(checkpoint "$RUNS/both" $((KINDS * COPY)) "$within")
    for path in "$alone" "$both"; do
      if [ ! -f "$path" ]; then
        echo "Missing $path: the arms are measured when they are made" >&2
        exit 1
      fi
    done
    pair blackwing "$within" "$both" "$alone" "$BLACKWING_MIRROR"
    pair cyber "$within" "$both" "$(checkpoint "$CYBER" "$COPY" "$point")" "$CYBER_MIRROR"
    pair all71-blackwing "$within" "$both" "$NETWORK" "$BLACKWING_MIRROR"
    pair all71-cyber "$within" "$both" "$NETWORK" "$CYBER_MIRROR"
    pair cross "$within" "$both" "$NETWORK" "$CROSS"
    pair all71-alone "$point" "$alone" "$NETWORK" "$BLACKWING_MIRROR"
    pair decks-01 "$point" "$(printf '%s/model-%05d.pt' "$DECKS01" $((COPY + point - 1)))" \
      "$alone" "$BLACKWING_MIRROR"
  done
  for name in both-copy both blackwing-copy blackwing; do
    cp "$RUNS/$name/metrics.jsonl" "$OUT/$name.metrics.jsonl"
    cp "$RUNS/$name/run.json" "$OUT/$name.run.json"
  done
  beside=()
  [ ! -d "$CYBER_ALONE" ] || beside+=(--cyber-alone "$CYBER_ALONE")
  [ ! -d "$CALIBRATION" ] || beside+=(--calibration "$CALIBRATION")
  "${python[@]}" tools/two_decks_report.py "$OUT" "${beside[@]}" --output "$OUT/report.json" \
    | tee "$OUT/report.txt"
  echo "The measurements and what they say are in $OUT."
fi
