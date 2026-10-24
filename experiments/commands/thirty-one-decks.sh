#!/usr/bin/env bash
# Thirty-one decks: more decks for the decks never trained on
# (experiments/2026-10-18-thirty-one-decks.md). Run from the repository root
# on a machine with a card. PARTS names what is made:
#
# arms (the default). Under RUNS, three networks each made as half of
# experiments/commands/thirteen-decks.sh was, of more decks:
#      plus-a  the thirteen decks of training (DECKS) and the nine of HALF_A
#              of the eighteen outside them;
#      plus-b  the thirteen and the nine of HALF_B;
#      all     the thirty-one.
# Each plays its own decks and meets their pilots alone, so that the decks
# of the other half are never seen by plus-a or by plus-b. Each is its copy
# (NAME-copy: its decks' pilots imitated from scratch, COPY updates a deck at
# 3e-4, every game seen once, its result at four positions, seed SEED), then
# PPO on from the copy's last checkpoint (NAME: UPDATES updates at RATE, one
# pass, ITSELF of an update's DUELS duels against itself, any two of its
# decks, and the others its decks against their pilots). The three go
# AT_ONCE at a time through an MPS server on a card
# (experiments/commands/at-once.sh), each its copy and then its PPO. A copy
# keeps a checkpoint every COPY updates and PPO every STEP. No run is
# evaluated as it goes: the checkpoints are measured afterwards. Finished
# runs are skipped and an interrupted run goes on. About a day by reckoning.
#   AT_ONCE=3 bash experiments/commands/thirty-one-decks.sh
#
# measure. Once the arms are made, on the machine their runs are on, into
# OUT: each arm, and the control (CONTROL, half of the experiment on
# thirteen decks), by its last three checkpoints on the screen of each deck
# of DECKS and on the decks outside them (UNSEEN, each against the pilots of
# DECKS); each one's copy at its last checkpoint (the control's
# CONTROL_COPY) and the decks' own pilots on those decks too. Then the
# report (tools/thirty_one_decks_report.py), which takes the readings. A run
# that is not made to its end is left for later, and a finished measurement
# is not made again. About three hours by reckoning.
#   PARTS=measure bash experiments/commands/thirty-one-decks.sh
set -euo pipefail
DEVICE=${DEVICE:-cuda}
WORKERS=${WORKERS:-24}
THREADS=${THREADS:-20}
SLOTS=${SLOTS:-16}
DUELS=${DUELS:-520}
PARTS=${PARTS:-arms}
DECKS=${DECKS:-blackwing,lightsworn,gishki,burn,crystal,dragunity,gladiator,heroes,infernity,monarch,morphtronic,rock-block,spellcaster}
# The eighteen decks outside DECKS in two halves alike in what their pilots,
# half and their cards' nearness say of them (the record).
HALF_A=${HALF_A:-arcana,draconic-might,harpie,karakuri,pyramid,quickdraw-plant,six-samurai,tele-dad,x-saber}
HALF_B=${HALF_B:-countdown,destiny-hero,fortune-lady,gravekeeper,machina,ojama,toon,verdict,watt}
COPY=${COPY:-240}
UPDATES=${UPDATES:-1920}
STEP=${STEP:-120}
ITSELF=${ITSELF:-260}
RATE=${RATE:-1e-4}
SEED=${SEED:-2012}
RUNS=${RUNS:-runs/thirty-one-decks}
OUT=${OUT:-analysis/thirty-one-decks}
# The control: half of the experiment on thirteen decks, made at the
# policies' 1b85b63, whose games are those of the engine pinned since
# (experiments/2026-10-16-thirteen-decks.md), and its copy.
CONTROL=${CONTROL:-runs/thirteen-decks/half}
CONTROL_COPY=${CONTROL_COPY:-runs/thirteen-decks/copy/checkpoints/model-03119.pt}
# A deck's screen (its name in place of %s), and the decks outside DECKS
# against their pilots: smaller ones try this part small.
SCREENS=${SCREENS:-eval_suites/ygo-policies-%s-screen.json}
UNSEEN=${UNSEEN:-eval_suites/ygo-policies-unseen-played.json}
# The interpreter of a machine that has no uv: the benchmark's setup leaves one.
PYTHON=${PYTHON:-}

for part in $PARTS; do
  if [ "$part" != arms ] && [ "$part" != measure ]; then
    echo "PARTS names arms and measure, not $part" >&2
    exit 1
  fi
done
IFS=, read -r -a decks <<< "$DECKS"
IFS=, read -r -a half_a <<< "$HALF_A"
IFS=, read -r -a half_b <<< "$HALF_B"
if ((COPY % STEP || UPDATES % STEP || UPDATES < 3 * STEP)); then
  echo "STEP must divide a deck's $COPY updates of a copy and the $UPDATES after them" >&2
  exit 1
fi
if ((ITSELF < 1 || ITSELF >= DUELS)); then
  echo "ITSELF is the duels of an update against itself, some and not all of its $DUELS" >&2
  exit 1
fi
every=$(printf '%s\n' "${decks[@]}" "${half_a[@]}" "${half_b[@]}")
if [ "$(sort <<< "$every" | uniq -d)" ]; then
  echo "DECKS, HALF_A and HALF_B name a deck twice between them" >&2
  exit 1
fi
if [ ! -f "$UNSEEN" ]; then
  echo "Missing $UNSEEN" >&2
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
mkdir -p "$RUNS"
LOGS=$RUNS
# shellcheck source=experiments/commands/at-once.sh
source "$(dirname "${BASH_SOURCE[0]}")/at-once.sh"

ARMS=(plus-a plus-b all)
declare -A played=(
  [plus-a]="$DECKS,$HALF_A"
  [plus-b]="$DECKS,$HALF_B"
  [all]="$DECKS,$HALF_A,$HALF_B"
)
# copied ARM: the updates of the arm's copy, COPY a deck.
copied() {
  local those
  IFS=, read -r -a those <<< "${played[$1]}"
  echo $((${#those[@]} * COPY))
}

# made NAME FIRST UPDATES KEPT DECKS OPTION...: the run NAME of DECKS
# (learners and opponents both) to UPDATES updates after the FIRST it began
# with, from where it stands or from its start, which the options name
# (nothing for a network from scratch), with a checkpoint every KEPT updates.
made() {
  local name=$1 first=$2 updates=$3 kept=$4 those=$5 last latest
  shift 5
  last=$(printf '%s/checkpoints/model-%05d.pt' "$RUNS/$name" $((first + updates - 1)))
  local arguments=(
    --device "$DEVICE" --workers "$WORKERS" --pool-threads "$THREADS" --pool-slots "$SLOTS"
    --run "$RUNS/$name" --opponents "$those" --duels "$DUELS" --eval-every 0
    --keep-checkpoints "$kept" --seed "$SEED" --ppo-epochs 1 --learner "$those"
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
      --updates $((first + updates - 1 - 10#${latest#model-}))
  fi
}

# whole NAME: the arm's copy, then its PPO on from the copy's last checkpoint.
whole() {
  local name=$1 copy
  copy=$(copied "$name")
  made "$name-copy" 0 "$copy" "$COPY" "${played[$name]}" \
    --imitation-only-updates 1000000 --imitation-value-weight 0.5 --imitation-value-positions 4
  made "$name" "$copy" "$UPDATES" "$STEP" "${played[$name]}" --self-play-duels "$ITSELF" \
    --resume "$(printf '%s/checkpoints/model-%05d.pt' "$RUNS/$name-copy" $((copy - 1)))" \
    --resume-learning-rate "$RATE"
}

if [[ " $PARTS " == *" arms "* ]]; then
  for arm in "${ARMS[@]}"; do
    at_once "$arm" whole "$arm"
  done
  all_done
  echo "The arms are made: their runs are in $RUNS."
fi

if [[ " $PARTS " == *" measure "* ]]; then
  mkdir -p "$OUT/screens" "$OUT/unseen"
  # shellcheck disable=SC2059
  screen() { printf "$SCREENS" "$1"; }
  # three FOLDER COPIED: a run's checkpoints after the last three STEP of
  # its UPDATES of PPO, after a copy of COPIED updates.
  three() {
    local point
    for ((point = UPDATES - 2 * STEP; point <= UPDATES; point += STEP)); do
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

  declare -A folders=([control]="$CONTROL")
  declare -A copies=([control]="$CONTROL_COPY")
  declare -A before=([control]=$((${#decks[@]} * COPY)))
  for arm in "${ARMS[@]}"; do
    folders[$arm]="$RUNS/$arm"
    before[$arm]=$(copied "$arm")
    copies[$arm]=$(printf '%s/checkpoints/model-%05d.pt' "$RUNS/$arm-copy" $((${before[$arm]} - 1)))
  done
  for run in "${ARMS[@]}" control; do
    mapfile -t found < <(three "${folders[$run]}" "${before[$run]}")
    ready=1
    for path in "${found[@]}" "${copies[$run]}"; do
      [ -f "$path" ] || ready=0
    done
    if ((!ready)); then
      echo "$run is not made to its end: it is measured when it is"
      continue
    fi
    # The control's own figures over as many updates as an arm's.
    head -n "$UPDATES" "${folders[$run]}/metrics.jsonl" > "$OUT/$run.metrics.jsonl"
    cp "${folders[$run]}/run.json" "$OUT/$run.run.json"
    for deck in "${decks[@]}"; do
      suite screens "$run.$deck" "$(screen "$deck")" "${found[@]}"
    done
    suite unseen "$run" "$UNSEEN" "${found[@]}"
    suite unseen "$run-copy" "$UNSEEN" "${copies[$run]}"
  done
  suite unseen pilot "$UNSEEN" --pilot
  "${python[@]}" tools/thirty_one_decks_report.py "$OUT" --decks "${decks[@]}" \
    --half-a "${half_a[@]}" --half-b "${half_b[@]}" --copy "$COPY" --updates "$UPDATES" \
    --step "$STEP" --output "$OUT/report.json" | tee "$OUT/report.txt"
  echo "The measurements and what they say are in $OUT."
fi
