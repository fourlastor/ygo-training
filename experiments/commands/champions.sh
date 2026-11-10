#!/usr/bin/env bash
# Champions of single decks by the recipe (experiments/2026-10-24-four-decks.md;
# the Cyber champion's and Blackwing alone's before it). Run from the
# repository root on the CUDA machine. PARTS names what is made, in its
# order:
#
# arms. For each run of RUNNING, DECK, DECK:SEED (SEED unless said) or either
# with @SHAPE: the deck's pilot imitated from scratch for COPY updates of DUELS
# duels against the thirteen decks' pilots, every game seen once and its
# result at four of its positions, at 3e-4, in minibatches of 256 with no
# duels in flight; then PPO on from the copy's last checkpoint for UPDATES
# updates at RATE, ITSELF of an update's duels the mirror against itself and
# the others against the thirteen decks' pilots, with fast's settings
# (OPTIONS adds more); a checkpoint every STEP. A run of another seed is named
# DECK-SEED. A SHAPE makes the network another shape from the first, at both
# stages: tokens, the board as its cards (--board-representation
# individual-tokens), or no-history, without its own earlier answers
# (--leave-out history); the run is named with -SHAPE after. The shape
# switched has no copy of its own: it is its control at SWITCHED updates of
# PPO given the board as its cards from there on, board_fusion at zero
# (tools/switch_board.py), and goes on as the control does to UPDATES; its
# control is made that far first. A finished run is skipped, an interrupted
# one goes on.
#
# measure. Into OUT, each a pair of files with the first network in the
# learner's seat and then the other, by each run's last three checkpoints:
# against all71 (NETWORK) in its deck's mirror
# (eval_suites/DECK-against-checkpoint.json, every deal from both seats); for
# a run of another seed, the deck's first run (FIRST, DECK=RUN) against all71
# too, and the two head to head in the mirror, checkpoint for checkpoint; for
# a run of a SHAPE, it and its control, the run of RUNNING of the same deck
# and seed without one, head to head in the mirror, checkpoint for
# checkpoint; the runs' metrics; then the report (tools/champions_report.py).
#
# About 2 hours a run on the CUDA machine.
#   bash experiments/commands/champions.sh
set -euo pipefail
DEVICE=${DEVICE:-cuda}
WORKERS=${WORKERS:-24}
THREADS=${THREADS:-20}
SLOTS=${SLOTS:-16}
DUELS=${DUELS:-520}
PARTS=${PARTS:-arms measure}
RUNNING=${RUNNING:-lightsworn infernity cyber:2013}
THIRTEEN=${THIRTEEN:-blackwing,lightsworn,gishki,burn,crystal,dragunity,gladiator,heroes,infernity,monarch,morphtronic,rock-block,spellcaster}
COPY=${COPY:-240}
UPDATES=${UPDATES:-720}
STEP=${STEP:-120}
ITSELF=${ITSELF:-260}
RATE=${RATE:-1e-4}
SEED=${SEED:-2012}
# The control's update of PPO a run of the shape switched begins from.
SWITCHED=${SWITCHED:-720}
# fast's settings: two updates' duels in flight, the pool's slots to hold
# them, minibatches of 512, the calls as they are in two groups.
IN_FLIGHT=${IN_FLIGHT:-1040}
FAST_SLOTS=${FAST_SLOTS:-52}
MINIBATCH=${MINIBATCH:-512}
OPTIONS=${OPTIONS:-}
NETWORK=${NETWORK:-runs/seventy-one-decks/all71/checkpoints/model-27599.pt}
# The first run of a deck made again at another seed: the Cyber champion's
# arm that copied the pilot as pinned.
FIRST=${FIRST:-cyber=runs/cyber-champion/cyber-old}
RUNS=${RUNS:-runs/four-decks}
OUT=${OUT:-analysis/four-decks}
SUITES=${SUITES:-eval_suites}
# The interpreter of a machine that has no uv: the benchmark's setup leaves one.
PYTHON=${PYTHON:-}

for part in $PARTS; do
  if [ "$part" != arms ] && [ "$part" != measure ]; then
    echo "PARTS names arms and measure, not $part" >&2
    exit 1
  fi
done
if ((UPDATES % STEP || UPDATES < 3 * STEP)); then
  echo "UPDATES is a multiple of STEP, three of them at least" >&2
  exit 1
fi
# deck RUN, seed RUN, shape RUN, name RUN: what a run of RUNNING names.
deck() {
  local run=${1%%@*}
  echo "${run%%:*}"
}
seed() {
  local run=${1%%@*}
  [[ $run == *:* ]] && echo "${run#*:}" || echo "$SEED"
}
shape() { [[ $1 == *@* ]] && echo "${1#*@}" || true; }
name() {
  local named
  if [ "$(seed "$1")" = "$SEED" ]; then named=$(deck "$1"); else named="$(deck "$1")-$(seed "$1")"; fi
  if [ -n "$(shape "$1")" ]; then named="$named-$(shape "$1")"; fi
  echo "$named"
}
# shaped RUN: the options of a run's shape, at both stages of its training.
# learning RUN: what a suite is told of a checkpoint of the run's in the
# learner's seat (a checkpoint in the other seat is built as it was saved,
# and one that leaves out a part of its input is loaded leaving it out).
shaped() {
  case "$(shape "$1")" in
    tokens | switched) echo --board-representation individual-tokens ;;
    no-history) echo --leave-out history ;;
  esac
}
learning() {
  case "$(shape "$1")" in
    tokens | switched) echo --board-representation individual-tokens ;;
  esac
}
# control RUN: the run of the same deck and seed without a shape.
control() { echo "${1%%@*}"; }
# first DECK: the first run of DECK, where FIRST names one.
first() {
  local given
  for given in $FIRST; do
    if [ "${given%%=*}" = "$1" ]; then
      echo "${given#*=}"
      return
    fi
  done
}
for run in $RUNNING; do
  for path in "$NETWORK" "$SUITES/$(deck "$run")-against-checkpoint.json"; do
    if [ ! -f "$path" ]; then
      echo "Missing $path" >&2
      exit 1
    fi
  done
  if [ -z "$(shape "$run")" ] && [ "$(seed "$run")" != "$SEED" ] \
    && [ -z "$(first "$(deck "$run")")" ]; then
    echo "A run of another seed is set beside the deck's first run: FIRST names none for $run" >&2
    exit 1
  fi
  if [ -n "$(shape "$run")" ]; then
    if [ -z "$(shaped "$run")" ]; then
      echo "A shape is tokens, no-history or switched, not $(shape "$run") ($run)" >&2
      exit 1
    fi
    if [ "$(shape "$run")" = switched ] && ((SWITCHED % STEP || SWITCHED >= UPDATES - 2 * STEP)); then
      echo "SWITCHED is a multiple of STEP before the last three checkpoints, not $SWITCHED" >&2
      exit 1
    fi
    if [[ " $PARTS " == *" measure "* ]] && [[ " $RUNNING " != *" $(control "$run") "* ]]; then
      echo "A run of a shape is measured beside its control: RUNNING names no $(control "$run")" \
        "for $run" >&2
      exit 1
    fi
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

# made NAME FIRST UPDATES KEPT LEARNER SEED OPTION...: the run NAME of the
# deck LEARNER at SEED, to UPDATES updates after the FIRST it began with,
# from where it stands or from its start, with a checkpoint every KEPT
# updates.
made() {
  local name=$1 first=$2 updates=$3 kept=$4 learner=$5 seeded=$6 last latest
  shift 6
  last=$(checkpoint "$RUNS/$name" "$first" "$updates")
  local arguments=(
    --device "$DEVICE" --workers "$WORKERS" --pool-threads "$THREADS"
    --run "$RUNS/$name" --learner "$learner" --opponents "$THIRTEEN" --duels "$DUELS"
    --eval-every 0 --keep-checkpoints "$kept" --seed "$seeded" --ppo-epochs 1
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

if [[ " $PARTS " == *" arms "* ]]; then
  for run in $RUNNING; do
    named=$(name "$run")
    read -r -a shaping <<< "$(shaped "$run")"
    echo "$(date +%H:%M) $named begins"
    if [ "$(shape "$run")" = switched ]; then
      begun="$RUNS/$named.begun.pt"
      if [ ! -f "$begun" ]; then
        from=$(checkpoint "$RUNS/$(name "$(control "$run")")" "$COPY" "$SWITCHED")
        if [ ! -f "$from" ]; then
          echo "Missing $from: a switched run begins once its control is made that far" >&2
          exit 1
        fi
        "${python[@]}" tools/switch_board.py "$from" --output "$begun"
      fi
      made "$named" $((COPY + SWITCHED)) $((UPDATES - SWITCHED)) "$STEP" "$(deck "$run")" \
        "$(seed "$run")" --pool-slots "$FAST_SLOTS" --self-play-duels "$ITSELF" \
        --rollout-estimates kept --duels-in-flight "$IN_FLIGHT" --minibatch "$MINIBATCH" \
        --inference-graphs 0 --pool-groups 2 "${shaping[@]}" "${options[@]}" \
        --resume "$begun" --resume-learning-rate "$RATE"
      echo "$(date +%H:%M) $named ends"
      continue
    fi
    made "$named-copy" 0 "$COPY" "$COPY" "$(deck "$run")" "$(seed "$run")" \
      --pool-slots "$SLOTS" --imitation-only-updates 1000000 --imitation-value-weight 0.5 \
      --imitation-value-positions 4 --minibatch 256 --duels-in-flight 0 "${shaping[@]}"
    made "$named" "$COPY" "$UPDATES" "$STEP" "$(deck "$run")" "$(seed "$run")" \
      --pool-slots "$FAST_SLOTS" --self-play-duels "$ITSELF" --rollout-estimates kept \
      --duels-in-flight "$IN_FLIGHT" --minibatch "$MINIBATCH" --inference-graphs 0 \
      --pool-groups 2 "${shaping[@]}" "${options[@]}" \
      --resume "$(checkpoint "$RUNS/$named-copy" 0 "$COPY")" --resume-learning-rate "$RATE"
    echo "$(date +%H:%M) $named ends"
  done
  echo "The champions are made: $RUNS."
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
  # pair FOLDER STEM FIRST SECOND SUITE [FIRST'S [SECOND'S]]: FIRST in the
  # learner's seat against SECOND, then the other way round, each told in the
  # learner's seat what its run's shape asks (learning).
  pair() {
    local folder=$1 stem=$2 one=$3 other=$4 played=$5 ones=() others=()
    read -r -a ones <<< "${6:-}"
    read -r -a others <<< "${7:-}"
    mkdir -p "$OUT/$folder"
    suite "$folder/$stem.plays" "$played" "$one" --opponent-checkpoint "$other" "${ones[@]}"
    suite "$folder/$stem.held" "$played" "$other" --opponent-checkpoint "$one" "${others[@]}"
  }
  told=()
  for run in $RUNNING; do
    named=$(name "$run")
    played="$SUITES/$(deck "$run")-against-checkpoint.json"
    for ((point = UPDATES - 2 * STEP; point <= UPDATES; point += STEP)); do
      champion=$(checkpoint "$RUNS/$named" "$COPY" "$point")
      if [ ! -f "$champion" ]; then
        echo "Missing $champion: a run is measured when it is made" >&2
        exit 1
      fi
      pair "$named" "$point" "$champion" "$NETWORK" "$played" "$(learning "$run")"
      if [ -n "$(shape "$run")" ]; then
        controlled=$(name "$(control "$run")")
        before=$(checkpoint "$RUNS/$controlled" "$COPY" "$point")
        if [ ! -f "$before" ]; then
          echo "Missing $before: a run of a shape is measured once its control is made" >&2
          exit 1
        fi
        pair "$named-against-$controlled" "$point" "$champion" "$before" "$played" \
          "$(learning "$run")"
      elif [ "$(seed "$run")" != "$SEED" ]; then
        firsts=$(first "$(deck "$run")")
        before=$(checkpoint "$firsts" "$COPY" "$point")
        pair "$(basename "$firsts")" "$point" "$before" "$NETWORK" "$played"
        pair "$named-against-$(basename "$firsts")" "$point" "$champion" "$before" "$played"
      fi
    done
    parts=("$named-copy" "$named")
    [ "$(shape "$run")" != switched ] || parts=("$named")
    for part in "${parts[@]}"; do
      cp "$RUNS/$part/metrics.jsonl" "$OUT/$part.metrics.jsonl"
      cp "$RUNS/$part/run.json" "$OUT/$part.run.json"
    done
    if [ -n "$(shape "$run")" ]; then
      told+=("$named=$(name "$(control "$run")")")
    elif [ "$(seed "$run")" = "$SEED" ]; then
      told+=("$named")
    else
      told+=("$named=$(basename "$(first "$(deck "$run")")")")
    fi
  done
  "${python[@]}" tools/champions_report.py "$OUT" "${told[@]}" --output "$OUT/report.json" \
    | tee "$OUT/report.txt"
  echo "The measurements and what they say are in $OUT."
fi
