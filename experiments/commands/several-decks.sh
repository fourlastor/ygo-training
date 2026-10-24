#!/usr/bin/env bash
# One network, several decks: does a network that imitates the pilots of
# several decks copy each of them as well as a network that imitates one?
# Run from the repository root on the CUDA machine.
# 1. A network of its own for each deck of DECKS, from scratch: the deck's
#    pilot imitated as `once-few` imitated Blackwing's (one pass, a game's
#    result at four of its positions), UPDATES updates of DUELS games.
#      alone-<deck>
# 2. One network of all of them, from the same first network: the same
#    lesson, each deck in an equal share of an update's games, for as many
#    updates as there are decks times UPDATES, so that it has had as many
#    games of each deck as that deck's own network.
#      together
# 3. On the screen of each deck, game by game: the deck's pilot, its own
#    network after the updates of POINTS, and the network of all the decks
#    after as many games of the deck.
# 4. The readings.
# Finished runs and measurements are skipped and an interrupted run goes on.
# So a deck more needs only its own network and the network of all of them:
# name another folder for that one (JOINT) and for the measurements (OUT),
# and the networks of the decks that are there are read as they are.
#   DECKS="blackwing lightsworn gishki" JOINT=together-three OUT=analysis/three-decks bash ...
# About two hours with two decks, three with three: half an hour for each
# deck's own network and as much again a deck for the network of all.
set -euo pipefail
DEVICE=${DEVICE:-cuda}
WORKERS=${WORKERS:-24}
THREADS=${THREADS:-20}
SLOTS=${SLOTS:-16}
DUELS=${DUELS:-520}
UPDATES=${UPDATES:-200}
EVERY=${EVERY:-40}
POINTS=${POINTS:-40 80 120 160 200}
DECKS=${DECKS:-blackwing lightsworn}
OPPONENTS=${OPPONENTS:-blackwing,burn,crystal,dragunity,gishki,gladiator,heroes,infernity,lightsworn,monarch,morphtronic,rock-block,spellcaster}
SEED=${SEED:-2012}
# The Blackwing pilot's copy as it was made before, to check this one against.
BEFORE=${BEFORE:-analysis/learner-follow-up/once-few.metrics.jsonl}
RUNS=${RUNS:-runs/several-decks}
OUT=${OUT:-analysis/several-decks}
# The folder under RUNS of the network of all the decks.
JOINT=${JOINT:-together}

decks=($DECKS)
count=${#decks[@]}
if ((count < 2)); then
  echo "DECKS names at least two decks" >&2
  exit 1
fi
if [ "$(printf '%s\n' "${decks[@]}" | sort -u | wc -l)" -ne "$count" ]; then
  echo "DECKS names a deck twice" >&2
  exit 1
fi
pinned=$(git submodule status --recursive)
if grep -q '^[^ ]' <<< "$pinned"; then
  echo "A submodule is not at its pinned revision: git submodule update --init --recursive" >&2
  exit 1
fi
screen() { printf 'eval_suites/ygo-policies-%s-screen.json' "$1"; }
for deck in "${decks[@]}"; do
  if [ ! -f "$(screen "$deck")" ]; then
    echo "Missing $(screen "$deck")" >&2
    exit 1
  fi
done
for point in $POINTS; do
  if ((point % EVERY || point > UPDATES)); then
    echo "No checkpoint is kept after $point updates: POINTS are multiples of EVERY up to UPDATES" >&2
    exit 1
  fi
done
if ((UPDATES % EVERY)); then
  echo "A run's last update must be one the suite is played after" >&2
  exit 1
fi
if [ -f "$RUNS/$JOINT/run.json" ]; then
  has=$(uv run python -c 'import json, sys; print(" ".join(json.load(open(sys.argv[1])).get("learners") or []))' "$RUNS/$JOINT/run.json")
  if [ "$has" != "${decks[*]}" ]; then
    echo "$RUNS/$JOINT is a network of ${has:-one deck}, not of ${decks[*]}: name another folder with JOINT" >&2
    exit 1
  fi
fi
cargo build --release
mkdir -p "$OUT/screens"
if [ -f "$BEFORE" ]; then
  cp "$BEFORE" "$OUT/before.metrics.jsonl"
fi

# A run from scratch after so many updates: its versions count from 0.
checkpoint() { printf '%s/checkpoints/model-%05d.pt' "$RUNS/$1" $(($2 - 1)); }

# imitate NAME LEARNER UPDATES EVERY [AS]: the run from scratch, or from
# where it stands, to UPDATES updates of the pilots' games, with a checkpoint
# and its decks' screens every EVERY of them; its figures beside the
# measurements, under the name AS where that is not the run's.
imitate() {
  local name=$1 learner=$2 updates=$3 every=$4 as=${5:-$1} latest
  local arguments=(
    --device "$DEVICE" --workers "$WORKERS" --pool-threads "$THREADS" --pool-slots "$SLOTS"
    --run "$RUNS/$name" --learner "$learner" --opponents "$OPPONENTS" --duels "$DUELS"
    --imitation-only-updates 1000000 --imitation-value-weight 0.5 --imitation-value-positions 4
    --ppo-epochs 1 --eval-every "$every" --keep-checkpoints "$every" --seed "$SEED"
  )
  if ! ls "$RUNS/$name"/checkpoints/model-*.pt > /dev/null 2>&1; then
    uv run python -m ygo_training train "${arguments[@]}" --updates "$updates"
  elif [ ! -f "$(checkpoint "$name" "$updates")" ]; then
    latest=$(basename "$(ls "$RUNS/$name"/checkpoints/model-*.pt | sort | tail -1)" .pt)
    uv run python -m ygo_training train "${arguments[@]}" --continue \
      --updates $((updates - 1 - 10#${latest#model-}))
  fi
  cp "$RUNS/$name/metrics.jsonl" "$OUT/$as.metrics.jsonl"
  cp "$RUNS/$name/run.json" "$OUT/$as.run.json"
}

# measure FILE DECK [CHECKPOINT...]: those checkpoints on the deck's screen,
# or its pilot when none is given, with how each game ended.
measure() {
  local file=$1 deck=$2
  shift 2
  if [ ! -f "$OUT/screens/$file.json" ]; then
    uv run python -m ygo_training evaluate-suite "${@:---pilot}" --device "$DEVICE" \
      --workers "$WORKERS" --pool-threads "$THREADS" --pool-slots "$SLOTS" --seed "$SEED" \
      --suite "$(screen "$deck")" --output "$OUT/screens/$file.json.part" > /dev/null
    mv "$OUT/screens/$file.json.part" "$OUT/screens/$file.json"
  fi
}

report() {
  uv run python tools/several_decks_report.py "$OUT" --decks "${decks[@]}" \
    --updates "$UPDATES" --output "$OUT/report.json" | tee "$OUT/report.txt"
}

# 1. Each deck's pilot and its own network, on its screen.
for deck in "${decks[@]}"; do
  measure "$deck.pilot" "$deck"
  imitate "alone-$deck" "$deck" "$UPDATES" "$EVERY"
  found=()
  for point in $POINTS; do
    found+=("$(checkpoint "alone-$deck" "$point")")
  done
  measure "$deck.alone" "$deck" "${found[@]}"
done

# 2. The network of all the decks, and 3. each deck's screen after as many
#    games of the deck as its own network had at the updates of POINTS.
imitate "$JOINT" "$(IFS=,; echo "${decks[*]}")" $((count * UPDATES)) $((count * EVERY)) together
found=()
for point in $POINTS; do
  found+=("$(checkpoint "$JOINT" $((count * point)))")
done
for deck in "${decks[@]}"; do
  measure "$deck.together" "$deck" "${found[@]}"
done

# 4. The readings.
report
echo "The runs' figures, the measurements and the readings are in $OUT: copy it back."
