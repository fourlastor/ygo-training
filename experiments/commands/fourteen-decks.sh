#!/usr/bin/env bash
# A network of fourteen decks (experiments/2026-10-26-fourteen-decks.md):
# the thirteen decks of training and Cyber, each deck's champion imitated
# into one network, then PPO with every duel against itself on the ordered
# pairings of its decks, none against a pilot; the four decks' recipe, with
# its duels against the pilots, made only where that arm falls short. Run
# from the repository root on the CUDA machine. PARTS names what is made, in
# its order:
#
# copy. taught: one network of the fourteen decks from scratch imitating
# each deck's champion (TEACHERS, DECK=CHECKPOINT) by its first choice, COPY
# updates of DUELS duels, 160 a deck, ITSELF of an update's duels the
# champions against themselves in their own mirrors, both seats imitated,
# the others against the thirteen decks' pilots, at 3e-4, in minibatches of
# 256 with no duels in flight, every game seen once and its result at four
# of its positions; the teachers' games in two groups and from CUDA graphs;
# a checkpoint every COPY_STEP.
#
# selfplay. PPO on from the copy's last checkpoint for UPDATES updates at
# RATE, every duel of an update against itself on the ordered pairings of
# its decks, each as often as any other, with fast's settings; a checkpoint
# every STEP.
#
# recipe. The four decks' recipe, made only where the selfplay arm falls
# short: ITSELF of an update's duels against itself, the others against the
# thirteen decks' pilots, for RECIPE updates, so as many duels against
# itself as the selfplay arm.
#
# measure. Into OUT: each teacher and the copy after each of CURVE against
# all71 in each mirror, and the copy's last checkpoint on QUICK; all71
# against the pilots (PILOTS); then the selfplay arm: its last three
# checkpoints on every ordered pairing against all71 (PAIRINGS), every one
# it keeps on QUICK, its last head to head against each champion in its
# mirror, against the pilots, and head to head against the network of four
# decks on its pairings (FOUR, FOUR_PAIRINGS); the runs' metrics; then the
# report (tools/fourteen_decks_report.py).
#
# recipe-measure. The recipe arm measured as the selfplay arm, then the
# report again, the two arms compared.
#
# copy-heads. The copy's last checkpoint head to head against each champion
# in its mirror, on the deals the selfplay arm played them, then the report
# again: where the arm's mirrors were lost, in the copy or in PPO.
#
# AT_ONCE: how many of the measurements go at a time, through an MPS server
# on a card (experiments/commands/at-once.sh), what each writes in
# OUT/logs; 1, the default, one after the other. With more than one, the
# first teacher's games in its mirror are then played again alone and held
# to those played beside the others, game for game.
#
# LEVELS: a measure of how many at a time pay. The decks' own measurements
# in their mirrors (each teacher, the copy along its updates and the arm
# head to head, twelve files of the same deals for every deck) go in a group
# of decks for each number named, as many decks as the number and as many
# at a time, the first group's decks first; after each group its first
# deck's teacher is played again alone, held to its games and timed. The
# rest go as many at a time as the last number names. The card's and the
# processor's load every ten seconds are kept in OUT/logs/load.txt, each
# group's files and times in OUT/logs/levels.jsonl, and what they come to
# in OUT/at-once.txt (tools/at_once_report.py). LEVELS="3 5 6" measures
# fourteen decks so.
#
# The trainer numbers an arm's updates on from the copy's and keeps a
# checkpoint where that count closes STEP, and the last: the selfplay arm's
# after 160, 400 and so on to 4,240 updates past the copy's 2,240, and after
# 4,320, its last three after 4,000, 4,240 and 4,320.
#
# About 25 hours on the CUDA machine by reckoning: the copy 11, the selfplay
# arm 10, the measurements 4.
#   PARTS=copy bash experiments/commands/fourteen-decks.sh
set -euo pipefail
DEVICE=${DEVICE:-cuda}
WORKERS=${WORKERS:-24}
THREADS=${THREADS:-20}
SLOTS=${SLOTS:-16}
DUELS=${DUELS:-520}
PARTS=${PARTS:-copy selfplay measure}
DECKS=${DECKS:-blackwing,cyber,lightsworn,infernity,gishki,burn,crystal,dragunity,gladiator,heroes,monarch,morphtronic,rock-block,spellcaster}
TEACHERS=${TEACHERS:-"blackwing=runs/two-decks/blackwing/checkpoints/model-00959.pt
cyber=runs/cyber-champion/cyber-old/checkpoints/model-00959.pt
lightsworn=runs/four-decks/lightsworn/checkpoints/model-00959.pt
infernity=runs/four-decks/infernity/checkpoints/model-00959.pt
gishki=runs/champions/gishki/checkpoints/model-00959.pt
burn=runs/champions/burn/checkpoints/model-00959.pt
crystal=runs/champions/crystal/checkpoints/model-00959.pt
dragunity=runs/champions/dragunity/checkpoints/model-00959.pt
gladiator=runs/champions/gladiator/checkpoints/model-00959.pt
heroes=runs/champions/heroes/checkpoints/model-00959.pt
monarch=runs/champions/monarch/checkpoints/model-00959.pt
morphtronic=runs/champions/morphtronic/checkpoints/model-00959.pt
rock-block=runs/champions/rock-block/checkpoints/model-00959.pt
spellcaster=runs/champions/spellcaster/checkpoints/model-00959.pt"}
THIRTEEN=${THIRTEEN:-blackwing,lightsworn,gishki,burn,crystal,dragunity,gladiator,heroes,infernity,monarch,morphtronic,rock-block,spellcaster}
COPY=${COPY:-2240}
COPY_STEP=${COPY_STEP:-80}
UPDATES=${UPDATES:-4320}
RECIPE=${RECIPE:-8640}
STEP=${STEP:-240}
ITSELF=${ITSELF:-260}
RATE=${RATE:-1e-4}
SEED=${SEED:-2012}
# The copy's teachers' games: an update's duels at once, in two groups, the
# calls from CUDA graphs of up to so many decisions, as for four decks.
COPY_SLOTS=${COPY_SLOTS:-26}
COPY_GRAPHS=${COPY_GRAPHS:-130}
# fast's settings: two updates' duels in flight, the pool's slots to hold
# them, minibatches of 512, the calls as they are in two groups.
IN_FLIGHT=${IN_FLIGHT:-1040}
FAST_SLOTS=${FAST_SLOTS:-52}
MINIBATCH=${MINIBATCH:-512}
NETWORK=${NETWORK:-runs/seventy-one-decks/all71/checkpoints/model-27599.pt}
# The network of four decks, its uniform arm's last checkpoint.
FOUR=${FOUR:-runs/four-decks/uniform/checkpoints/model-03839.pt}
RUNS=${RUNS:-runs/fourteen-decks}
OUT=${OUT:-analysis/fourteen-decks}
SUITES=${SUITES:-eval_suites}
PAIRINGS=${PAIRINGS:-eval_suites/fourteen-decks-pairings-checkpoint.json}
QUICK=${QUICK:-eval_suites/fourteen-decks-pairings-quick-checkpoint.json}
PILOTS=${PILOTS:-eval_suites/fourteen-decks-against-pilots.json}
FOUR_PAIRINGS=${FOUR_PAIRINGS:-eval_suites/four-decks-pairings-checkpoint.json}
# The copy's checkpoints measured in the mirrors: 40, 80, 120 and 160
# updates a deck.
CURVE=${CURVE:-560 1120 1680 2240}
# The interpreter of a machine that has no uv: the benchmark's setup leaves one.
PYTHON=${PYTHON:-}
LOGS=${LOGS:-$OUT/logs}
LEVELS=${LEVELS:-}
# shellcheck source=experiments/commands/at-once.sh
source "$(dirname "${BASH_SOURCE[0]}")/at-once.sh"

for part in $PARTS; do
  case $part in
    copy | selfplay | recipe | measure | recipe-measure | copy-heads) ;;
    *)
      echo "PARTS names copy, selfplay, recipe, measure, recipe-measure and copy-heads, not $part" >&2
      exit 1
      ;;
  esac
done
if ((UPDATES < 3 * STEP || RECIPE < 3 * STEP || COPY % COPY_STEP || COPY < 3 * COPY_STEP)); then
  echo "UPDATES and RECIPE are three STEP at least, COPY a multiple of COPY_STEP and three of them" >&2
  exit 1
fi
for point in $CURVE; do
  if ((point % COPY_STEP || point > COPY)); then
    echo "CURVE names checkpoints the copy keeps, not $point" >&2
    exit 1
  fi
done
read -r -a decks <<< "${DECKS//,/ }"
teach=()
for given in $TEACHERS; do
  teach+=(--imitation-teacher "$given")
  if [ ! -f "${given#*=}" ]; then
    echo "Missing ${given#*=}" >&2
    exit 1
  fi
done
for path in "$NETWORK" "$FOUR" "$PAIRINGS" "$QUICK" "$PILOTS" "$FOUR_PAIRINGS" \
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
# each game ended, into FILE.json, at once with others where AT_ONCE says.
played() {
  # A file being played beside others is not there yet: it is not played
  # twice.
  if [ ! -f "$1.json" ] && [ -z "${queued[$1]:-}" ]; then
    queued[$1]=1
    planned+=("$1.json")
    at_once "$(basename "$(dirname "$1")")-$(basename "$1")" playing "$@"
  fi
}
declare -A queued
planned=()
# playing FILE SUITE OPTION...: what played does, then and there.
playing() {
  local file=$1 suite=$2
  shift 2
  mkdir -p "$(dirname "$file")"
  "${python[@]}" -m ygo_training evaluate-suite "$@" --device "$DEVICE" \
    --workers "$WORKERS" --pool-threads "$THREADS" --pool-slots "$SLOTS" --seed "$SEED" \
    --suite "$suite" --output "$file.json.part" > /dev/null
  mv "$file.json.part" "$file.json"
  echo "$(date +%H:%M) $file"
}
# pair FILE FIRST SECOND SUITE: FIRST in the learner's seat against SECOND,
# then the other way round, into FILE.plays.json and FILE.held.json.
pair() {
  local file=$1 one=$2 other=$3 suite=$4
  played "$file.plays" "$suite" "$one" --opponent-checkpoint "$other"
  played "$file.held" "$suite" "$other" --opponent-checkpoint "$one"
}

# made NAME FIRST UPDATES KEPT OPTION...: the run NAME of the fourteen
# decks, to UPDATES updates after the FIRST it began with, from where it
# stands or from its start, with a checkpoint every KEPT updates.
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

# measured ARM UPDATES: the arm ARM, of UPDATES updates on from the copy,
# into OUT: its last three checkpoints on PAIRINGS, every one it keeps on
# QUICK, its last head to head against each champion in its mirror, against
# the pilots and against the network of four decks, and its metrics.
measured() {
  local arm=$1 updates=$2 last point deck
  local points=()
  last=$(checkpoint "$RUNS/$arm" "$COPY" "$updates")
  if [ ! -f "$last" ]; then
    echo "Missing $last: an arm is measured when it is made" >&2
    exit 1
  fi
  mapfile -t points < <(kept "$COPY" "$updates" "$STEP")
  for point in "${points[@]: -3}"; do
    played "$OUT/$arm-pairings/$point" "$PAIRINGS" "$(checkpoint "$RUNS/$arm" "$COPY" "$point")" \
      --opponent-checkpoint "$NETWORK"
  done
  for point in "${points[@]}"; do
    played "$OUT/$arm-quick/$point" "$QUICK" "$(checkpoint "$RUNS/$arm" "$COPY" "$point")" \
      --opponent-checkpoint "$NETWORK"
  done
  for deck in "${decks[@]}"; do
    pair "$OUT/heads-$arm/$deck" "$last" "$(teacher "$deck")" "$SUITES/$deck-against-checkpoint.json"
  done
  played "$OUT/pilots/$arm" "$PILOTS" "$last"
  pair "$OUT/four-decks/$arm" "$last" "$FOUR" "$FOUR_PAIRINGS"
  cp "$RUNS/$arm/metrics.jsonl" "$OUT/$arm.metrics.jsonl"
  cp "$RUNS/$arm/run.json" "$OUT/$arm.run.json"
}
# mirrored DECK: the deck's own measurements in its mirror: its teacher, the
# copy after each of CURVE, and the arm's last checkpoint head to head
# against its teacher.
mirrored() {
  local mirror=$SUITES/$1-against-checkpoint.json point
  pair "$OUT/teachers/$1" "$(teacher "$1")" "$NETWORK" "$mirror"
  for point in $CURVE; do
    pair "$OUT/taught-$1/$point" "$(checkpoint "$RUNS/taught" 0 "$point")" "$NETWORK" "$mirror"
  done
  pair "$OUT/heads-selfplay/$1" "$(checkpoint "$RUNS/selfplay" "$COPY" "$UPDATES")" \
    "$(teacher "$1")" "$mirror"
}
# timed LEVEL FROM PART: the files planned since FROM, played LEVEL at a
# time, kept in LOGS/levels.jsonl as PART: the decks' own measurements in
# their mirrors, one of them alone, or the rest.
timed() {
  local files=() file
  for file in "${planned[@]}"; do
    files+=("\"$file\"")
  done
  if ((${#files[@]})); then
    local joined
    joined=$(IFS=,; echo "${files[*]}")
    echo "{\"part\": \"$3\", \"at_once\": $1, \"from\": $2, \"to\": $(date +%s), \"files\": [$joined]}" \
      >> "$LOGS/levels.jsonl"
  fi
  planned=()
}
# alone DECK: the deck's teacher in its mirror played again alone, timed, and
# held to the games it played beside the others.
alone() {
  local started
  if [ ! -f "$OUT/alone/$1.plays.json" ]; then
    started=$(date +%s)
    planned=("$OUT/alone/$1.plays.json")
    playing "$OUT/alone/$1.plays" "$SUITES/$1-against-checkpoint.json" "$(teacher "$1")" \
      --opponent-checkpoint "$NETWORK"
    timed 1 "$started" alone
    "${python[@]}" tools/same_games.py "$OUT/teachers/$1.plays.json" "$OUT/alone/$1.plays.json"
  fi
}
# leveled LEVEL DECK...: the decks' own measurements, LEVEL at a time, then
# the first of them alone.
leveled() {
  local level=$1 started deck
  shift
  AT_ONCE=$level
  started=$(date +%s)
  planned=()
  for deck in "$@"; do
    mirrored "$deck"
  done
  all_done
  timed "$level" "$started" mirrors
  alone "$1"
}
# watched: the card's busy share and memory and the processor's counters
# every ten seconds into LOGS/load.txt, while the script goes.
watched() {
  mkdir -p "$LOGS"
  (
    set +e
    while kill -0 $$ 2> /dev/null; do
      card="- -"
      if [[ "$DEVICE" == cuda* ]] && command -v nvidia-smi > /dev/null; then
        card=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits \
          2> /dev/null | head -1 | tr -d ,)
      fi
      { read -r _ counters; } < /proc/stat
      echo "$(date +%s) ${card:-- -} $counters"
      sleep 10
    done >> "$LOGS/load.txt"
  ) &
  disown $!
}
# reported: the report of what OUT holds, once every measurement is over;
# where they went at once, the first teacher's games played again alone and
# held to them.
reported() {
  all_done
  [ -z "${measuring:-}" ] || timed "$AT_ONCE" "$measuring" rest
  if ((AT_ONCE > 1)); then
    alone "${decks[0]}"
  fi
  "${python[@]}" tools/fourteen_decks_report.py "$OUT" --decks "$DECKS" \
    --output "$OUT/report.json" | tee "$OUT/report.txt"
  echo "The measurements and what they say are in $OUT."
}

copied=$(checkpoint "$RUNS/taught" 0 "$COPY")
fast=(
  --pool-slots "$FAST_SLOTS" --rollout-estimates kept
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

if [[ " $PARTS " == *" selfplay "* ]]; then
  echo "$(date +%H:%M) selfplay begins"
  made selfplay "$COPY" "$UPDATES" "$STEP" "${fast[@]}" --self-play-duels "$DUELS"
  echo "$(date +%H:%M) selfplay ends"
fi

if [[ " $PARTS " == *" recipe "* ]]; then
  echo "$(date +%H:%M) recipe begins"
  made recipe "$COPY" "$RECIPE" "$STEP" "${fast[@]}" --self-play-duels "$ITSELF"
  echo "$(date +%H:%M) recipe ends"
fi

if [[ " $PARTS " == *" measure "* ]]; then
  for path in "$copied" "$(checkpoint "$RUNS/selfplay" "$COPY" "$UPDATES")"; do
    if [ ! -f "$path" ]; then
      echo "Missing $path: the copy and the arm are measured when they are made" >&2
      exit 1
    fi
  done
  read -r -a levels <<< "$LEVELS"
  place=0
  for level in "${levels[@]}"; do
    place=$((place + level))
  done
  if ((place > ${#decks[@]})); then
    echo "LEVELS names groups of $place decks in all, of ${#decks[@]}" >&2
    exit 1
  fi
  watched
  place=0
  for level in "${levels[@]}"; do
    leveled "$level" "${decks[@]:place:level}"
    place=$((place + level))
  done
  ((${#levels[@]} == 0)) || AT_ONCE=${levels[-1]}
  measuring=$(date +%s)
  for deck in "${decks[@]:place}"; do
    mirrored "$deck"
  done
  played "$OUT/copy-quick/0" "$QUICK" "$copied" --opponent-checkpoint "$NETWORK"
  played "$OUT/pilots/all71" "$PILOTS" "$NETWORK"
  cp "$RUNS/taught/metrics.jsonl" "$OUT/taught.metrics.jsonl"
  cp "$RUNS/taught/run.json" "$OUT/taught.run.json"
  measured selfplay "$UPDATES"
  reported
  if ((${#levels[@]})); then
    "${python[@]}" tools/at_once_report.py "$OUT" | tee "$OUT/at-once.txt"
  fi
fi

if [[ " $PARTS " == *" recipe-measure "* ]]; then
  measured recipe "$RECIPE"
  reported
fi

if [[ " $PARTS " == *" copy-heads "* ]]; then
  if [ ! -f "$copied" ]; then
    echo "Missing $copied: the copy is measured when it is made" >&2
    exit 1
  fi
  for deck in "${decks[@]}"; do
    pair "$OUT/heads-taught/$deck" "$copied" "$(teacher "$deck")" \
      "$SUITES/$deck-against-checkpoint.json"
  done
  reported
fi
