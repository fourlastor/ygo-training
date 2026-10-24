#!/usr/bin/env bash
# The network of two decks with its duels against itself in the mirrors only
# (experiments/2026-10-24-two-decks.md, "Follow-up: the mirrors only"): one
# network of Blackwing and Cyber with as many games of each mirror, and of
# each pairing with a pilot, as the two champions together, and no more. Run
# from the repository root on the CUDA machine, after
# experiments/commands/two-decks.sh made and measured its runs. PARTS names
# what is made, in its order:
#
# arm. mirrors: PPO on from the last checkpoint of the network of both's copy
# (both-copy, the same copy) for UPDATES updates for each of its decks, at
# RATE, ITSELF of an update's duels against itself, each deck's mirror as
# often as the other's (--self-play-pairs mirrors), and the others against
# the thirteen decks' pilots, with fast's settings (OPTIONS adds more); a
# checkpoint every STEP updates a deck. A finished run is skipped, an
# interrupted one goes on.
#
# measure. Into OUT, each a pair of files with mirrors in the learner's seat
# and then in the other, by its last three kept checkpoints: against all71
# (NETWORK) in each deck's mirror and in Blackwing against Cyber (CROSS);
# against each deck's champion (Blackwing alone, CYBER) in its mirror and
# against the network of both, at as many games of it; the run's metrics;
# then the report (tools/two_decks_mirrors_report.py), beside what
# two-decks.sh measured of the network of both and of Blackwing alone
# (BESIDE) and cyber-old's own games against all71 (CYBER_ALONE).
#
# About 3 to 3.5 hours on the CUDA machine by reckoning.
#   bash experiments/commands/two-decks-mirrors.sh
set -euo pipefail
DEVICE=${DEVICE:-cuda}
WORKERS=${WORKERS:-24}
THREADS=${THREADS:-20}
SLOTS=${SLOTS:-16}
DUELS=${DUELS:-520}
PARTS=${PARTS:-arm measure}
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
# all71 in the Cyber mirror.
CYBER=${CYBER:-runs/cyber-champion/cyber-old}
CYBER_ALONE=${CYBER_ALONE:-analysis/cyber-champion/cyber-old/mirror}
# Where two-decks.sh made its runs and measured them.
RUNS=${RUNS:-runs/two-decks}
BESIDE=${BESIDE:-analysis/two-decks}
OUT=${OUT:-analysis/two-decks/mirrors}
BLACKWING_MIRROR=${BLACKWING_MIRROR:-eval_suites/blackwing-against-checkpoint.json}
CYBER_MIRROR=${CYBER_MIRROR:-eval_suites/cyber-against-checkpoint.json}
CROSS=${CROSS:-eval_suites/blackwing-against-cyber-checkpoint.json}
# The interpreter of a machine that has no uv: the benchmark's setup leaves one.
PYTHON=${PYTHON:-}
# The decks: a deck's share of an update falls with their number, of the
# pilots' duels and, in the mirrors only, of those against itself.
BOTH=blackwing,cyber
KINDS=2

for part in $PARTS; do
  if [ "$part" != arm ] && [ "$part" != measure ]; then
    echo "PARTS names arm and measure, not $part" >&2
    exit 1
  fi
done
if ((UPDATES % STEP || UPDATES < 3 * STEP || COPY % STEP)); then
  echo "UPDATES is a multiple of STEP, three of them at least, and COPY of STEP" >&2
  exit 1
fi
# checkpoint RUN FIRST POINT: the checkpoint of the run at RUN after POINT
# updates past the FIRST it began with.
checkpoint() {
  printf '%s/checkpoints/model-%05d.pt' "$1" $(($2 + $3 - 1))
}
copied=$(checkpoint "$RUNS/both-copy" 0 $((KINDS * COPY)))
for path in "$NETWORK" "$BLACKWING_MIRROR" "$CYBER_MIRROR" "$CROSS" "$copied"; do
  if [ ! -f "$path" ]; then
    echo "Missing $path" >&2
    exit 1
  fi
done
points=()
for ((point = UPDATES - 2 * STEP; point <= UPDATES; point += STEP)); do
  points+=("$point")
done
for point in "${points[@]}"; do
  if [ ! -f "$(checkpoint "$CYBER" "$COPY" "$point")" ]; then
    echo "Missing $(checkpoint "$CYBER" "$COPY" "$point")" >&2
    exit 1
  fi
done
for folder in all71-blackwing all71-cyber cross all71-alone; do
  if [ ! -d "$BESIDE/$folder" ]; then
    echo "Missing $BESIDE/$folder: two-decks.sh measures the network of both first" >&2
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

if [[ " $PARTS " == *" arm "* ]]; then
  first=$((KINDS * COPY))
  updates=$((KINDS * UPDATES))
  last=$(checkpoint "$RUNS/mirrors" "$first" "$updates")
  arguments=(
    --device "$DEVICE" --workers "$WORKERS" --pool-threads "$THREADS"
    --run "$RUNS/mirrors" --learner "$BOTH" --opponents "$THIRTEEN" --duels "$DUELS"
    --eval-every 0 --keep-checkpoints $((KINDS * STEP)) --seed "$SEED" --ppo-epochs 1
    --pool-slots "$FAST_SLOTS" --self-play-duels "$ITSELF" --self-play-pairs mirrors
    --rollout-estimates kept --duels-in-flight "$IN_FLIGHT" --minibatch "$MINIBATCH"
    --inference-graphs 0 --pool-groups 2 "${options[@]}"
  )
  echo "$(date +%H:%M) mirrors begins"
  if ! ls "$RUNS/mirrors"/checkpoints/model-*.pt > /dev/null 2>&1; then
    "${python[@]}" -m ygo_training train "${arguments[@]}" --updates "$updates" \
      --resume "$copied" --resume-learning-rate "$RATE"
  elif [ ! -f "$last" ]; then
    latest=$(basename "$(ls "$RUNS/mirrors"/checkpoints/model-*.pt | sort | tail -1)" .pt)
    "${python[@]}" -m ygo_training train "${arguments[@]}" --continue \
      --updates $((first + updates - 1 - 10#${latest#model-}))
  fi
  echo "$(date +%H:%M) mirrors ends"
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
    # At the same games of each mirror: a champion after POINT updates, the
    # mirrors after KINDS times as many, the network of both after KINDS
    # squared times.
    within=$((KINDS * point))
    mirrors=$(checkpoint "$RUNS/mirrors" $((KINDS * COPY)) "$within")
    both=$(checkpoint "$RUNS/both" $((KINDS * COPY)) $((KINDS * KINDS * point)))
    alone=$(checkpoint "$RUNS/blackwing" "$COPY" "$point")
    for path in "$mirrors" "$both" "$alone"; do
      if [ ! -f "$path" ]; then
        echo "Missing $path: the runs are measured when they are made" >&2
        exit 1
      fi
    done
    pair all71-blackwing "$within" "$mirrors" "$NETWORK" "$BLACKWING_MIRROR"
    pair all71-cyber "$within" "$mirrors" "$NETWORK" "$CYBER_MIRROR"
    pair cross "$within" "$mirrors" "$NETWORK" "$CROSS"
    pair blackwing "$within" "$mirrors" "$alone" "$BLACKWING_MIRROR"
    pair cyber "$within" "$mirrors" "$(checkpoint "$CYBER" "$COPY" "$point")" "$CYBER_MIRROR"
    pair both-blackwing "$within" "$mirrors" "$both" "$BLACKWING_MIRROR"
    pair both-cyber "$within" "$mirrors" "$both" "$CYBER_MIRROR"
  done
  cp "$RUNS/mirrors/metrics.jsonl" "$OUT/mirrors.metrics.jsonl"
  cp "$RUNS/mirrors/run.json" "$OUT/mirrors.run.json"
  beside=()
  [ ! -d "$CYBER_ALONE" ] || beside+=(--cyber-alone "$CYBER_ALONE")
  "${python[@]}" tools/two_decks_mirrors_report.py "$OUT" --both "$BESIDE" "${beside[@]}" \
    --output "$OUT/report.json" | tee "$OUT/report.txt"
  echo "The measurements and what they say are in $OUT."
fi
