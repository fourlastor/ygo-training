#!/usr/bin/env bash
# Specialist distillation (experiments/2026-10-24-specialist-distillation.md):
# the two champions imitated into one network of Blackwing and Cyber, then its
# decks against each other. Run from the repository root on the CUDA machine,
# after experiments/commands/two-decks.sh made and measured its runs. PARTS
# names what is made, in its order:
#
# yardstick. The network of both (BOTH_RUNS/both, from its copy both-copy of
# BOTH_COPY updates) in Blackwing against Cyber (CROSS) against all71
# (NETWORK), by its copy and its checkpoints every 480 updates of PPO, into
# BOTH_OUT/cross, where two-decks.sh measured its last three; then games
# measured before, played again at today's code into OUT/check and set beside
# them: the network of both's there after 1,920 updates, and each teacher's
# against all71 in its mirror. Played on 2026-10-24, before the design was
# fixed.
#
# arms. taught, then cross. taught: one network of both decks from scratch
# imitating each deck's champion (BLACKWING_TEACHER, CYBER_TEACHER) by its
# first choice, COPY updates of DUELS duels, ITSELF of an update's duels the
# champions against themselves in their own mirrors, both seats imitated, and
# the others against the thirteen decks' pilots, at 3e-4, in minibatches of
# 256 with no duels in flight, every game seen once and its result at four of
# its positions; a checkpoint every COPY_STEP. cross: PPO on from the copy's
# last checkpoint for UPDATES updates at RATE, ITSELF of an update's duels
# against itself with the two decks against each other alone and the others
# against the thirteen decks' pilots, with fast's settings (OPTIONS adds
# more); a checkpoint every STEP. A finished run is skipped, an interrupted
# one goes on.
#
# measure. Into OUT, each a pair of files with the first network in the
# learner's seat and then the other: the copy by its last three checkpoints
# against all71 in each mirror and head to head against each teacher in its
# mirror; the copy's last checkpoint and each of cross's against all71 in
# Blackwing against Cyber; each of cross's against all71 in each mirror; the
# runs' metrics; then the report (tools/specialist_distillation_report.py).
#
# About 4.5 to 5 hours on the CUDA machine by reckoning.
#   bash experiments/commands/specialist-distillation.sh
set -euo pipefail
DEVICE=${DEVICE:-cuda}
WORKERS=${WORKERS:-24}
THREADS=${THREADS:-20}
SLOTS=${SLOTS:-16}
DUELS=${DUELS:-520}
PARTS=${PARTS:-arms measure}
THIRTEEN=${THIRTEEN:-blackwing,lightsworn,gishki,burn,crystal,dragunity,gladiator,heroes,infernity,monarch,morphtronic,rock-block,spellcaster}
COPY=${COPY:-480}
COPY_STEP=${COPY_STEP:-80}
UPDATES=${UPDATES:-1440}
STEP=${STEP:-240}
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
# The champions, each by its last checkpoint, after 720 updates of PPO.
BLACKWING_TEACHER=${BLACKWING_TEACHER:-runs/two-decks/blackwing/checkpoints/model-00959.pt}
CYBER_TEACHER=${CYBER_TEACHER:-runs/cyber-champion/cyber-old/checkpoints/model-00959.pt}
# Where two-decks.sh made and measured the network of both, how long its copy
# was, and where the Cyber champion was measured.
BOTH_RUNS=${BOTH_RUNS:-runs/two-decks}
BOTH_COPY=${BOTH_COPY:-480}
BOTH_OUT=${BOTH_OUT:-analysis/two-decks}
CYBER_OUT=${CYBER_OUT:-analysis/cyber-champion}
RUNS=${RUNS:-runs/specialist-distillation}
OUT=${OUT:-analysis/specialist-distillation}
BLACKWING_MIRROR=${BLACKWING_MIRROR:-eval_suites/blackwing-against-checkpoint.json}
CYBER_MIRROR=${CYBER_MIRROR:-eval_suites/cyber-against-checkpoint.json}
CROSS=${CROSS:-eval_suites/blackwing-against-cyber-checkpoint.json}
# The interpreter of a machine that has no uv: the benchmark's setup leaves one.
PYTHON=${PYTHON:-}
BOTH=blackwing,cyber

for part in $PARTS; do
  if [ "$part" != yardstick ] && [ "$part" != arms ] && [ "$part" != measure ]; then
    echo "PARTS names yardstick, arms and measure, not $part" >&2
    exit 1
  fi
done
if ((UPDATES % STEP || COPY % COPY_STEP || COPY < 3 * COPY_STEP)); then
  echo "UPDATES is a multiple of STEP, and COPY of COPY_STEP, three of them at least" >&2
  exit 1
fi
for path in "$NETWORK" "$BLACKWING_TEACHER" "$CYBER_TEACHER" "$BLACKWING_MIRROR" \
  "$CYBER_MIRROR" "$CROSS"; do
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
read -r -a options <<< "$OPTIONS"
cargo build --release
mkdir -p "$RUNS" "$OUT"

# checkpoint RUN FIRST POINT: the checkpoint of the run at RUN after POINT
# updates past the FIRST it began with.
checkpoint() {
  printf '%s/checkpoints/model-%05d.pt' "$1" $(($2 + $3 - 1))
}
# suite FILE SUITE OPTION...: what the options name on a suite, with how each
# game ended, into FILE.json.
suite() {
  local file=$1 played=$2
  shift 2
  if [ ! -f "$file.json" ]; then
    "${python[@]}" -m ygo_training evaluate-suite "$@" --device "$DEVICE" \
      --workers "$WORKERS" --pool-threads "$THREADS" --pool-slots "$SLOTS" --seed "$SEED" \
      --suite "$played" --output "$file.json.part" > /dev/null
    mv "$file.json.part" "$file.json"
    echo "$(date +%H:%M) $file"
  fi
}
# pair FOLDER STEM FIRST SECOND SUITE: FIRST in the learner's seat against
# SECOND, then the other way round.
pair() {
  local folder=$1 stem=$2 first=$3 second=$4 played=$5
  mkdir -p "$folder"
  suite "$folder/$stem.plays" "$played" "$first" --opponent-checkpoint "$second"
  suite "$folder/$stem.held" "$played" "$second" --opponent-checkpoint "$first"
}
# The report's account of where the runs it reads are.
beside=(--both "$BOTH_OUT" --cyber-champion "$CYBER_OUT")

if [[ " $PARTS " == *" yardstick "* ]]; then
  pair "$BOTH_OUT/cross" 0 "$(checkpoint "$BOTH_RUNS/both-copy" 0 "$BOTH_COPY")" \
    "$NETWORK" "$CROSS"
  for ((point = 480; point <= 2880; point += 480)); do
    pair "$BOTH_OUT/cross" "$point" "$(checkpoint "$BOTH_RUNS/both" "$BOTH_COPY" "$point")" \
      "$NETWORK" "$CROSS"
  done
  pair "$OUT/check" cross-1920 "$(checkpoint "$BOTH_RUNS/both" "$BOTH_COPY" 1920)" \
    "$NETWORK" "$CROSS"
  pair "$OUT/check" blackwing-720 "$BLACKWING_TEACHER" "$NETWORK" "$BLACKWING_MIRROR"
  pair "$OUT/check" cyber-720 "$CYBER_TEACHER" "$NETWORK" "$CYBER_MIRROR"
  "${python[@]}" tools/specialist_distillation_report.py check "$OUT" "${beside[@]}"
fi

# made NAME FIRST UPDATES KEPT OPTION...: the run NAME of both decks, to
# UPDATES updates after the FIRST it began with, from where it stands or
# from its start, with a checkpoint every KEPT updates.
made() {
  local name=$1 first=$2 updates=$3 kept=$4 last latest
  shift 4
  last=$(checkpoint "$RUNS/$name" "$first" "$updates")
  local arguments=(
    --device "$DEVICE" --workers "$WORKERS" --pool-threads "$THREADS"
    --run "$RUNS/$name" --learner "$BOTH" --opponents "$THIRTEEN" --duels "$DUELS"
    --eval-every 0 --keep-checkpoints "$kept" --seed "$SEED" --ppo-epochs 1
  )
  echo "$(date +%H:%M) $name begins"
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
  echo "$(date +%H:%M) $name ends"
}

copied=$(checkpoint "$RUNS/taught" 0 "$COPY")
if [[ " $PARTS " == *" arms "* ]]; then
  made taught 0 "$COPY" "$COPY_STEP" --pool-slots "$SLOTS" \
    --imitation-only-updates 1000000 --imitation-value-weight 0.5 \
    --imitation-value-positions 4 --minibatch 256 --duels-in-flight 0 \
    --self-play-duels "$ITSELF" --self-play-pairs mirrors \
    --imitation-teacher "blackwing=$BLACKWING_TEACHER" --imitation-teacher "cyber=$CYBER_TEACHER"
  made cross "$COPY" "$UPDATES" "$STEP" --pool-slots "$FAST_SLOTS" \
    --self-play-duels "$ITSELF" --self-play-pairs cross --rollout-estimates kept \
    --duels-in-flight "$IN_FLIGHT" --minibatch "$MINIBATCH" --inference-graphs 0 \
    --pool-groups 2 "${options[@]}" --resume "$copied" --resume-learning-rate "$RATE"
  echo "The arms are made: $RUNS."
fi

if [[ " $PARTS " == *" measure "* ]]; then
  for path in "$copied" "$(checkpoint "$RUNS/cross" "$COPY" "$UPDATES")"; do
    if [ ! -f "$path" ]; then
      echo "Missing $path: the arms are measured when they are made" >&2
      exit 1
    fi
  done
  for ((point = COPY - 2 * COPY_STEP; point <= COPY; point += COPY_STEP)); do
    taught=$(checkpoint "$RUNS/taught" 0 "$point")
    pair "$OUT/taught-blackwing" "$point" "$taught" "$NETWORK" "$BLACKWING_MIRROR"
    pair "$OUT/taught-cyber" "$point" "$taught" "$NETWORK" "$CYBER_MIRROR"
    pair "$OUT/teacher-blackwing" "$point" "$taught" "$BLACKWING_TEACHER" "$BLACKWING_MIRROR"
    pair "$OUT/teacher-cyber" "$point" "$taught" "$CYBER_TEACHER" "$CYBER_MIRROR"
  done
  pair "$OUT/cross" 0 "$copied" "$NETWORK" "$CROSS"
  for ((point = STEP; point <= UPDATES; point += STEP)); do
    crossed=$(checkpoint "$RUNS/cross" "$COPY" "$point")
    pair "$OUT/cross" "$point" "$crossed" "$NETWORK" "$CROSS"
    pair "$OUT/cross-blackwing" "$point" "$crossed" "$NETWORK" "$BLACKWING_MIRROR"
    pair "$OUT/cross-cyber" "$point" "$crossed" "$NETWORK" "$CYBER_MIRROR"
  done
  for name in taught cross; do
    cp "$RUNS/$name/metrics.jsonl" "$OUT/$name.metrics.jsonl"
    cp "$RUNS/$name/run.json" "$OUT/$name.run.json"
  done
  "${python[@]}" tools/specialist_distillation_report.py report "$OUT" "${beside[@]}" \
    --output "$OUT/report.json" | tee "$OUT/report.txt"
  echo "The measurements and what they say are in $OUT."
fi
