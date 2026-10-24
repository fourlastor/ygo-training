#!/usr/bin/env bash
# Seventy-one decks: every deck of the pool
# (experiments/2026-10-19-seventy-one-decks.md). Run from the repository root
# on a machine with a card. PARTS names what is made:
#
# arm (the default). Under RUNS, one network of DECKS, the 71 of the pin:
# its copy (all71-copy: the decks' pilots imitated from scratch, COPY updates
# a deck at 3e-4, every game seen once, its result at four positions, seed
# SEED; a checkpoint kept every COPY_KEPT updates and the last), then PPO on
# from the copy's last checkpoint (all71: UPDATES updates at RATE, one pass,
# ITSELF of an update's DUELS duels against itself, any two of its decks, and
# the others its decks against their pilots, with the games' own estimates;
# a checkpoint every STEP). No run is evaluated as it goes. A finished run is
# skipped and an interrupted run goes on; with NEW_PINS=1 it goes on although
# the pins moved since (train --new-pins), for a move whose duels are the old
# ones. About two days and a quarter by reckoning.
#   bash experiments/commands/seventy-one-decks.sh
#
# measure. Once the run is made, on the machine it is on, into OUT: all71 at
# each point of POINTS, by its checkpoints after the last three STEP of its
# updates of PPO there, and the controls HALF and ALL (half of the experiment
# on thirteen decks, all of thirty-one decks, after the last three STEP of
# CONTROL_POINT), on the screen of each deck of THIRTEEN and on the decks of
# OLD (the eighteen other decks of thirty-one, each against the pilots of the
# thirteen); all71 and ALL on the decks of NEW (the forty new to the pool, the
# same way); all71's copy at its last checkpoint and the decks' own pilots on
# OLD and NEW too. Then the report (tools/seventy_one_decks_report.py), which
# takes the readings. A point not reached is left for later, and a finished
# measurement is not made again. About two hours by reckoning.
#   PARTS=measure bash experiments/commands/seventy-one-decks.sh
set -euo pipefail
DEVICE=${DEVICE:-cuda}
WORKERS=${WORKERS:-24}
THREADS=${THREADS:-20}
SLOTS=${SLOTS:-16}
DUELS=${DUELS:-520}
PARTS=${PARTS:-arm}
THIRTEEN=${THIRTEEN:-blackwing,lightsworn,gishki,burn,crystal,dragunity,gladiator,heroes,infernity,monarch,morphtronic,rock-block,spellcaster}
# The 71 decks of the pin (the policies' 82a7c61): the thirteen, the eighteen
# other decks of thirty-one decks, and the forty new to the pool.
DECKS=${DECKS:-$THIRTEEN,arcana,draconic-might,harpie,karakuri,pyramid,quickdraw-plant,six-samurai,tele-dad,x-saber,countdown,destiny-hero,fortune-lady,gravekeeper,machina,ojama,toon,verdict,watt,agents,alien,amazoness,batteryman,benkei,chain-burn,chaos,cloudian,counter-fairy,cyber,dark-world,deckout,demise,destiny-board,exodia,fabled,fish,flamvell,garden,gem-knight,gemini,genex,gusto,herald,ice-barrier,iron-chain,jurrac,macro-dd,malefic,naturia,nordic,nurse,psychic,reptilianne,scrap,spirit,venom,volcanic,worm,zombie}
COPY=${COPY:-240}
COPY_KEPT=${COPY_KEPT:-2400}
UPDATES=${UPDATES:-10560}
STEP=${STEP:-120}
# R1, R2 and R3: as many updates as half and all, about as many games of
# each deck as all had, and as half had.
POINTS=${POINTS:-1920,4440,10560}
ITSELF=${ITSELF:-260}
RATE=${RATE:-1e-4}
SEED=${SEED:-2012}
RUNS=${RUNS:-runs/seventy-one-decks}
OUT=${OUT:-analysis/seventy-one-decks}
# The controls, each with the updates of its copy, read after CONTROL_POINT
# updates of PPO.
HALF=${HALF:-runs/thirteen-decks/half}
HALF_COPIED=${HALF_COPIED:-3120}
ALL=${ALL:-runs/thirty-one-decks/all}
ALL_COPIED=${ALL_COPIED:-7440}
CONTROL_POINT=${CONTROL_POINT:-1920}
# A deck's screen (its name in place of %s), the eighteen and the forty
# against the thirteen decks' pilots: smaller ones try this part small.
SCREENS=${SCREENS:-eval_suites/ygo-policies-%s-screen.json}
OLD=${OLD:-eval_suites/ygo-policies-unseen-played.json}
NEW=${NEW:-eval_suites/ygo-policies-forty-played.json}
# The interpreter of a machine that has no uv: the benchmark's setup leaves one.
PYTHON=${PYTHON:-}

for part in $PARTS; do
  if [ "$part" != arm ] && [ "$part" != measure ]; then
    echo "PARTS names arm and measure, not $part" >&2
    exit 1
  fi
done
IFS=, read -r -a thirteen <<< "$THIRTEEN"
IFS=, read -r -a decks <<< "$DECKS"
IFS=, read -r -a points <<< "$POINTS"
if ((${#points[@]} != 3)); then
  echo "POINTS names three points, R1, R2 and R3" >&2
  exit 1
fi
for point in "${points[@]}" "$CONTROL_POINT"; do
  if ((point % STEP || point < 3 * STEP)); then
    echo "A point of reading is a multiple of STEP, three of them at least: not $point" >&2
    exit 1
  fi
done
if ((points[2] != UPDATES)); then
  echo "R3 is the end of the run: POINTS ends at UPDATES" >&2
  exit 1
fi
if [ "$(printf '%s\n' "${decks[@]}" | sort | uniq -d)" ]; then
  echo "DECKS names a deck twice" >&2
  exit 1
fi
for deck in "${thirteen[@]}"; do
  if ! printf '%s\n' "${decks[@]}" | grep -qx "$deck"; then
    echo "$deck is one of THIRTEEN and not of DECKS" >&2
    exit 1
  fi
done
for suite in "$OLD" "$NEW"; do
  if [ ! -f "$suite" ]; then
    echo "Missing $suite" >&2
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
"${python[@]}" - "$DECKS" << 'EOF'
import sys
from ygo_training.environment import pool
missing = sorted(set(sys.argv[1].split(",")) - set(pool(".")))
if missing:
    sys.exit(f"Not in the pool of the pin: {', '.join(missing)}")
EOF
copied=$((${#decks[@]} * COPY))

# made NAME FIRST UPDATES KEPT OPTION...: the run NAME of DECKS (learners and
# opponents both) to UPDATES updates after the FIRST it began with, from where
# it stands or from its start, which the options name, a checkpoint kept every
# KEPT updates.
made() {
  local name=$1 first=$2 updates=$3 kept=$4 last latest
  shift 4
  last=$(printf '%s/checkpoints/model-%05d.pt' "$RUNS/$name" $((first + updates - 1)))
  local arguments=(
    --device "$DEVICE" --workers "$WORKERS" --pool-threads "$THREADS" --pool-slots "$SLOTS"
    --run "$RUNS/$name" --opponents "$DECKS" --duels "$DUELS" --eval-every 0
    --keep-checkpoints "$kept" --seed "$SEED" --ppo-epochs 1 --learner "$DECKS"
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
    "${python[@]}" -m ygo_training train "${arguments[@]}" "${again[@]}" --continue \
      ${NEW_PINS:+--new-pins} --updates $((first + updates - 1 - 10#${latest#model-}))
  fi
}

if [[ " $PARTS " == *" arm "* ]]; then
  mkdir -p "$RUNS"
  made all71-copy 0 "$copied" "$COPY_KEPT" \
    --imitation-only-updates 1000000 --imitation-value-weight 0.5 --imitation-value-positions 4
  made all71 "$copied" "$UPDATES" "$STEP" --self-play-duels "$ITSELF" --rollout-estimates kept \
    --resume "$(printf '%s/checkpoints/model-%05d.pt' "$RUNS/all71-copy" $((copied - 1)))" \
    --resume-learning-rate "$RATE"
  echo "The run is made: $RUNS/all71."
fi

if [[ " $PARTS " == *" measure "* ]]; then
  mkdir -p "$OUT/screens" "$OUT/old" "$OUT/new"
  # shellcheck disable=SC2059
  screen() { printf "$SCREENS" "$1"; }
  # three FOLDER COPIED POINT: a run's checkpoints after the last three STEP
  # of POINT updates of PPO, after a copy of COPIED updates.
  three() {
    local point
    for ((point = $3 - 2 * STEP; point <= $3; point += STEP)); do
      printf '%s/checkpoints/model-%05d.pt\n' "$1" $(($2 + point - 1))
    done
  }
  # suite FOLDER FILE SUITE CHECKPOINT... [OPTION...]: those checkpoints on a
  # suite, with how each game ended, into OUT/FOLDER/FILE.json.
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

  declare -A folders=([half]="$HALF" [all]="$ALL")
  declare -A before=([half]="$HALF_COPIED" [all]="$ALL_COPIED")
  declare -A at=([half]="$CONTROL_POINT" [all]="$CONTROL_POINT")
  runs=()
  for point in "${points[@]}"; do
    runs+=("all71-$point")
    folders[all71-$point]="$RUNS/all71"
    before[all71-$point]=$copied
    at[all71-$point]=$point
  done
  runs+=(half all)
  for run in "${runs[@]}"; do
    mapfile -t found < <(three "${folders[$run]}" "${before[$run]}" "${at[$run]}")
    ready=1
    for path in "${found[@]}"; do
      [ -f "$path" ] || ready=0
    done
    if ((!ready)); then
      echo "$run is not made to its point: it is measured when it is"
      continue
    fi
    for deck in "${thirteen[@]}"; do
      suite screens "$run.$deck" "$(screen "$deck")" "${found[@]}"
    done
    suite old "$run" "$OLD" "${found[@]}"
    if [ "$run" != half ]; then
      suite new "$run" "$NEW" "${found[@]}"
    fi
  done
  # The run's own figures, and the controls' over as many updates as read.
  for run in all71 half all; do
    case $run in
      all71) folder="$RUNS/all71" updates=$UPDATES ;;
      half) folder="$HALF" updates=$CONTROL_POINT ;;
      all) folder="$ALL" updates=$CONTROL_POINT ;;
    esac
    if [ -f "$folder/metrics.jsonl" ]; then
      head -n "$updates" "$folder/metrics.jsonl" > "$OUT/$run.metrics.jsonl"
      cp "$folder/run.json" "$OUT/$run.run.json"
    fi
  done
  copy=$(printf '%s/checkpoints/model-%05d.pt' "$RUNS/all71-copy" $((copied - 1)))
  if [ -f "$copy" ]; then
    suite old all71-copy "$OLD" "$copy"
    suite new all71-copy "$NEW" "$copy"
  fi
  suite old pilot "$OLD" --pilot
  suite new pilot "$NEW" --pilot
  "${python[@]}" tools/seventy_one_decks_report.py "$OUT" --thirteen "${thirteen[@]}" \
    --points "${points[@]}" --copied "$copied" --half-copied "$HALF_COPIED" \
    --all-copied "$ALL_COPIED" --control-point "$CONTROL_POINT" --step "$STEP" \
    --output "$OUT/report.json" | tee "$OUT/report.txt"
  echo "The measurements and what they say are in $OUT."
fi
