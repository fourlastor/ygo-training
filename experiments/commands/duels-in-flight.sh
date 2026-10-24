#!/usr/bin/env bash
# Duels in flight and larger minibatches (experiments/2026-10-23-duels-in-
# flight.md): a run. Run from the repository root on the machine that holds
# the run of seventy-one decks (SEVENTY_ONE, made by
# experiments/commands/seventy-one-decks.sh). PARTS names what is made, in
# its order:
#
# trial. PLAIN updates of PPO on from all71's copy at this code with all71's
# own settings, into RUNS/plain: the speed the arm is set beside. Then, for
# each way of answering the model's calls of WAYS, TRIAL updates of the
# arm's settings into RUNS/trial-WAY; the way whose updates after their
# first two take the least time is the arm's (OUT/trial.json). A finished
# trial is read again, not made again.
#
# arm. Under RUNS, the run fast: PPO on from all71's copy for UPDATES
# updates as all71 went on from it, its settings and seed, with IN_FLIGHT
# duels kept in flight (FAST_SLOTS slots a thread) and minibatches of
# MINIBATCH, its calls answered the trial's way; a checkpoint every STEP. A
# finished run is skipped, and an interrupted run goes on, its duels in
# flight begun again.
#
# measure. Into OUT: fast and all71 by their checkpoints after the last
# three STEP of UPDATES updates of PPO, on the screen of each deck of
# THIRTEEN and on OLD and NEW, both at this code. Then the report
# (tools/duels_in_flight_report.py), which takes both readings. A finished
# measurement is not made again.
#
# About 20 minutes for the trial, four hours for the arm by reckoning, and an
# hour and a half to measure.
#   bash experiments/commands/duels-in-flight.sh
set -euo pipefail
DEVICE=${DEVICE:-cuda}
WORKERS=${WORKERS:-24}
THREADS=${THREADS:-20}
SLOTS=${SLOTS:-16}
DUELS=${DUELS:-520}
PARTS=${PARTS:-trial arm measure}
THIRTEEN=${THIRTEEN:-blackwing,lightsworn,gishki,burn,crystal,dragunity,gladiator,heroes,infernity,monarch,morphtronic,rock-block,spellcaster}
# The 71 decks of seventy-one decks, in the order it named them.
DECKS=${DECKS:-$THIRTEEN,arcana,draconic-might,harpie,karakuri,pyramid,quickdraw-plant,six-samurai,tele-dad,x-saber,countdown,destiny-hero,fortune-lady,gravekeeper,machina,ojama,toon,verdict,watt,agents,alien,amazoness,batteryman,benkei,chain-burn,chaos,cloudian,counter-fairy,cyber,dark-world,deckout,demise,destiny-board,exodia,fabled,fish,flamvell,garden,gem-knight,gemini,genex,gusto,herald,ice-barrier,iron-chain,jurrac,macro-dd,malefic,naturia,nordic,nurse,psychic,reptilianne,scrap,spirit,venom,volcanic,worm,zombie}
# The updates of all71's copy, 240 a deck.
COPY=${COPY:-240}
# R1 of seventy-one decks.
UPDATES=${UPDATES:-1920}
STEP=${STEP:-120}
ITSELF=${ITSELF:-260}
RATE=${RATE:-1e-4}
SEED=${SEED:-2012}
# The arm: two updates' duels in flight, the pool's slots to hold them.
IN_FLIGHT=${IN_FLIGHT:-1040}
FAST_SLOTS=${FAST_SLOTS:-52}
MINIBATCH=${MINIBATCH:-512}
# The trial: all71's own settings, and the arm's for each way of its calls.
PLAIN=${PLAIN:-20}
TRIAL=${TRIAL:-8}
WAYS=${WAYS:-graphs-two graphs64-two eager-one eager-two}
SEVENTY_ONE=${SEVENTY_ONE:-runs/seventy-one-decks}
RUNS=${RUNS:-runs/duels-in-flight}
OUT=${OUT:-analysis/duels-in-flight/run}
SCREENS=${SCREENS:-eval_suites/ygo-policies-%s-screen.json}
OLD=${OLD:-eval_suites/ygo-policies-unseen-played.json}
NEW=${NEW:-eval_suites/ygo-policies-forty-played.json}
# The interpreter of a machine that has no uv: the benchmark's setup leaves one.
PYTHON=${PYTHON:-}

# How each way answers the model's calls: on a card, graphs up to half the
# pool's duels and two groups unless told.
declare -A ways=(
  [graphs-two]=""
  [graphs64-two]="--inference-graphs 64 --pool-groups 2"
  [eager-one]="--inference-graphs 0 --pool-groups 1"
  [eager-two]="--inference-graphs 0 --pool-groups 2"
)

for part in $PARTS; do
  if [ "$part" != trial ] && [ "$part" != arm ] && [ "$part" != measure ]; then
    echo "PARTS names trial, arm and measure, not $part" >&2
    exit 1
  fi
done
for way in $WAYS; do
  if [ -z "${ways[$way]+named}" ]; then
    echo "WAYS names ${!ways[*]}, not $way" >&2
    exit 1
  fi
done
IFS=, read -r -a thirteen <<< "$THIRTEEN"
IFS=, read -r -a decks <<< "$DECKS"
if ((UPDATES % STEP || UPDATES < 3 * STEP)); then
  echo "UPDATES is a multiple of STEP, three of them at least: not $UPDATES" >&2
  exit 1
fi
for suite in "$OLD" "$NEW"; do
  if [ ! -f "$suite" ]; then
    echo "Missing $suite" >&2
    exit 1
  fi
done
copied=$((${#decks[@]} * COPY))
copy=$(printf '%s/all71-copy/checkpoints/model-%05d.pt' "$SEVENTY_ONE" $((copied - 1)))
if [ ! -f "$copy" ]; then
  echo "Missing all71's copy: $copy" >&2
  exit 1
fi
pinned=$(git submodule status --recursive)
if grep -q '^[^ ]' <<< "$pinned"; then
  echo "A submodule is not at its pinned revision: git submodule update --init --recursive" >&2
  exit 1
fi
python=(uv run python)
[ -z "$PYTHON" ] || python=("$PYTHON")
cargo build --release
mkdir -p "$RUNS" "$OUT"

# made NAME UPDATES SLOTS OPTION...: the run NAME of DECKS from all71's copy
# to UPDATES updates of PPO as all71 went on from it, SLOTS duels a thread,
# from where it stands or from the copy, a checkpoint kept every STEP.
made() {
  local name=$1 updates=$2 slots=$3 last latest
  shift 3
  last=$(printf '%s/checkpoints/model-%05d.pt' "$RUNS/$name" $((copied + updates - 1)))
  local arguments=(
    --device "$DEVICE" --workers "$WORKERS" --pool-threads "$THREADS" --pool-slots "$slots"
    --run "$RUNS/$name" --opponents "$DECKS" --duels "$DUELS" --eval-every 0
    --keep-checkpoints "$STEP" --seed "$SEED" --ppo-epochs 1 --learner "$DECKS"
    --self-play-duels "$ITSELF" --rollout-estimates kept "$@"
  )
  if ! ls "$RUNS/$name"/checkpoints/model-*.pt > /dev/null 2>&1; then
    "${python[@]}" -m ygo_training train "${arguments[@]}" \
      --resume "$copy" --resume-learning-rate "$RATE" --updates "$updates"
  elif [ ! -f "$last" ]; then
    latest=$(basename "$(ls "$RUNS/$name"/checkpoints/model-*.pt | sort | tail -1)" .pt)
    "${python[@]}" -m ygo_training train "${arguments[@]}" --continue \
      --updates $((copied + updates - 1 - 10#${latest#model-}))
  fi
}

fast=(--duels-in-flight "$IN_FLIGHT" --minibatch "$MINIBATCH")

if [[ " $PARTS " == *" trial "* ]]; then
  made plain "$PLAIN" "$SLOTS"
  for way in $WAYS; do
    read -r -a flags <<< "${ways[$way]}"
    made "trial-$way" "$TRIAL" "$FAST_SLOTS" "${fast[@]}" "${flags[@]}"
  done
  "${python[@]}" - "$RUNS" "$OUT/trial.json" $WAYS << 'EOF'
import json
import sys
from pathlib import Path

runs, out, names = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]


def seconds(name: str) -> float:
    """An update's mean time after the run's first two."""
    rows = [json.loads(line) for line in (runs / name / "metrics.jsonl").open() if line.strip()]
    kept = rows[2:] or rows
    return sum(row["wall_seconds"] - row.get("evaluation_seconds", 0.0) for row in kept) / len(kept)


found = {name: seconds(f"trial-{name}") for name in names}
chosen = min(found, key=found.get)
plain = seconds("plain")
out.write_text(json.dumps({"ways": found, "plain": plain, "chosen": chosen}, indent=2))
for name, value in found.items():
    print(f"trial {name}: {value:.2f} s an update{'  <- the arm' if name == chosen else ''}")
print(f"all71's settings at this code: {plain:.2f} s an update")
EOF
fi

if [[ " $PARTS " == *" arm "* ]]; then
  if [ ! -f "$OUT/trial.json" ]; then
    echo "Missing $OUT/trial.json: the arm's calls are the trial's" >&2
    exit 1
  fi
  chosen=$("${python[@]}" -c 'import json, sys; print(json.load(open(sys.argv[1]))["chosen"])' "$OUT/trial.json")
  read -r -a flags <<< "${ways[$chosen]}"
  made fast "$UPDATES" "$FAST_SLOTS" "${fast[@]}" "${flags[@]}"
  echo "The run is made: $RUNS/fast."
fi

if [[ " $PARTS " == *" measure "* ]]; then
  mkdir -p "$OUT/screens" "$OUT/old" "$OUT/new"
  # shellcheck disable=SC2059
  screen() { printf "$SCREENS" "$1"; }
  # three FOLDER: a run's checkpoints after the last three STEP of UPDATES
  # updates of PPO, after the copy.
  three() {
    local point
    for ((point = UPDATES - 2 * STEP; point <= UPDATES; point += STEP)); do
      printf '%s/checkpoints/model-%05d.pt\n' "$1" $((copied + point - 1))
    done
  }
  # suite FOLDER FILE SUITE CHECKPOINT...: those checkpoints on a suite, with
  # how each game ended, into OUT/FOLDER/FILE.json.
  suite() {
    local folder=$1 file=$2 played=$3
    shift 3
    if [ ! -f "$OUT/$folder/$file.json" ]; then
      "${python[@]}" -m ygo_training evaluate-suite "$@" --device "$DEVICE" \
        --workers "$WORKERS" --pool-threads "$THREADS" --pool-slots "$SLOTS" --seed "$SEED" \
        --suite "$played" --output "$OUT/$folder/$file.json.part" > /dev/null
      mv "$OUT/$folder/$file.json.part" "$OUT/$folder/$file.json"
    fi
  }
  declare -A folders=([fast]="$RUNS/fast" ["all71-$UPDATES"]="$SEVENTY_ONE/all71")
  for run in fast "all71-$UPDATES"; do
    mapfile -t found < <(three "${folders[$run]}")
    for path in "${found[@]}"; do
      if [ ! -f "$path" ]; then
        echo "Missing $path: $run is measured when it is made" >&2
        exit 1
      fi
    done
    for deck in "${thirteen[@]}"; do
      suite screens "$run.$deck" "$(screen "$deck")" "${found[@]}"
    done
    suite old "$run" "$OLD" "${found[@]}"
    suite new "$run" "$NEW" "${found[@]}"
  done
  # The runs' own figures: the arm's and the trial's whole, all71's over as
  # many updates.
  cp "$RUNS/fast/metrics.jsonl" "$OUT/fast.metrics.jsonl"
  cp "$RUNS/fast/run.json" "$OUT/fast.run.json"
  cp "$RUNS/plain/metrics.jsonl" "$OUT/plain.metrics.jsonl"
  head -n "$UPDATES" "$SEVENTY_ONE/all71/metrics.jsonl" > "$OUT/all71.metrics.jsonl"
  "${python[@]}" tools/duels_in_flight_report.py "$OUT" --thirteen "${thirteen[@]}" \
    --copied "$copied" --point "$UPDATES" --step "$STEP" --first "$PLAIN" \
    --output "$OUT/report.json" | tee "$OUT/report.txt"
  echo "The measurements and what they say are in $OUT."
fi
