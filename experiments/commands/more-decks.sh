#!/usr/bin/env bash
# One network for more decks, half of its duels against itself: what a deck
# more costs the decks a network already plays, at the same number of games
# in all. Run from the repository root on a machine with a card. PARTS names
# what is made:
#
# arms (the default). A network for each of SIZES, which plays the first so
# many decks of DECKS, so that the decks of a smaller one are among those of
# a larger and Blackwing is in every one. Each is made in two runs:
#      decks-NN-copy  the pilots of its decks imitated from scratch, as the
#                     networks of experiments/commands/several-decks.sh were
#                     (every game seen once, its result at four positions),
#                     COPY updates for each of its decks;
#      decks-NN       PPO on from the copy's last checkpoint, UPDATES
#                     updates at the learning rate RATE: ITSELF of an
#                     update's DUELS against itself, any two of its decks
#                     with one in each seat, and the others its decks against
#                     the pilots of OPPONENTS.
# A copy keeps a checkpoint every COPY updates and the other run every STEP.
# No run is evaluated as it goes: the checkpoints are measured afterwards,
# where they are brought (a network of thirteen decks has thirteen screens,
# 21,632 games each time).
# AT_ONCE arms go at the same time, through an MPS server on a card
# (experiments/commands/at-once.sh), each its copy and then its other run:
# a rented machine of 32 threads carried six runs at once at a quarter more
# an update than one alone. Finished runs are skipped and an interrupted run
# goes on.
#
# measure. Once the arms are made, on the machine their runs are on: each
# network's copy at its last checkpoint and its last three checkpoints, on
# the held-out deals of GAMES and on the screen of each of its decks, and
# each deck's pilot on its screen, into OUT; then the report
# (tools/more_decks_report.py), which takes the readings. Beside them, every
# other checkpoint each network kept of its PPO, on the held-out deals: how
# fast each learns (CURVE=0 leaves it out). A network that is not made to
# its end is left for later, and a finished measurement is not made again.
# About two hours on the CUDA machine by reckoning, and two more for the
# curves.
#   PARTS=measure bash experiments/commands/more-decks.sh
#
# beyond. After the experiment was read and closed (2026-10-16), on the
# machine the runs are on: the networks beyond their own decks, into
# OUT/beyond, then that report (tools/more_decks_beyond_report.py), which
# takes its four readings. Each network's copy at its last checkpoint and
# its last three checkpoints playing the eighteen decks outside training
# themselves, against the pilots of the thirteen of training (UNSEEN), and
# each of those decks' own pilots on the same games; and the network of the
# most decks against the network of the fewest in the mirror of the first
# deck (MIRROR), their last three checkpoints matched, each deal from either
# seat. About an hour on the CUDA machine by reckoning.
#   PARTS=beyond bash experiments/commands/more-decks.sh
#
# kept. Names the files under RUNS that stay, one a line, and does nothing
# else: every file but a checkpoint a run only passes through on its way to
# the next it keeps. What tools/rented.sh follow brings from a rented
# machine:
#   bash tools/rented.sh follow NAME env PARTS=kept bash experiments/commands/more-decks.sh
# Its networks read the deck lists (format 14), which the input dropped in
# format 15: they are made and measured at 05d27141, the revision before.
set -euo pipefail
DEVICE=${DEVICE:-cuda}
WORKERS=${WORKERS:-24}
THREADS=${THREADS:-20}
SLOTS=${SLOTS:-16}
DUELS=${DUELS:-520}
PARTS=${PARTS:-arms}
SIZES=${SIZES:-1 2 3 5 8 13}
# Blackwing, the two decks a network of several was first made of, and the
# other decks of training by their names.
DECKS=${DECKS:-blackwing lightsworn gishki burn crystal dragunity gladiator heroes infernity monarch morphtronic rock-block spellcaster}
OPPONENTS=${OPPONENTS:-blackwing,burn,crystal,dragunity,gishki,gladiator,heroes,infernity,lightsworn,monarch,morphtronic,rock-block,spellcaster}
COPY=${COPY:-240}
UPDATES=${UPDATES:-1920}
STEP=${STEP:-120}
ITSELF=${ITSELF:-260}
RATE=${RATE:-1e-4}
SEED=${SEED:-2012}
RUNS=${RUNS:-runs/more-decks}
OUT=${OUT:-analysis/more-decks}
GAMES=${GAMES:-analysis/search-leads}
# A deck's screen, the deck's name in place of %s: smaller suites try the
# measuring part at a small size.
SCREENS=${SCREENS:-eval_suites/ygo-policies-%s-screen.json}
# What PARTS=beyond plays: the decks outside training, each against the
# pilots of the thirteen of training, and the mirror of the first deck
# against a checkpoint held in the other seat.
UNSEEN=${UNSEEN:-eval_suites/ygo-policies-unseen-played.json}
MIRROR=${MIRROR:-eval_suites/blackwing-against-checkpoint.json}
CURVE=${CURVE:-1}
# The interpreter of a machine that has no uv: the benchmark's setup leaves one.
PYTHON=${PYTHON:-}

for part in $PARTS; do
  if [ "$part" != arms ] && [ "$part" != measure ] && [ "$part" != beyond ] \
    && [ "$part" != kept ]; then
    echo "PARTS names arms, measure, beyond and kept, not $part" >&2
    exit 1
  fi
done
if ! [[ "$STEP" =~ ^[1-9][0-9]*$ ]]; then
  echo "STEP is the updates between the checkpoints a run keeps, one at least, not $STEP" >&2
  exit 1
fi

if [[ " $PARTS " == *" kept "* ]]; then
  # A run keeps the checkpoints whose updates in all are a multiple of what
  # it was told (COPY for a copy, STEP for the other run, and STEP divides
  # COPY), and passes through one after every other update.
  [ ! -d "$RUNS" ] || find "$RUNS" -type f ! -name '*-tmp' | awk -F/ -v step="$STEP" '
    $NF ~ /^model-[0-9]+\.pt(\.json)?$/ {
      updates = substr($NF, 7) + 1
      if (updates % step) next
    }
    { print }' | sort
  [ "$PARTS" != kept ] || exit 0
fi

if ((COPY % STEP || UPDATES % STEP || UPDATES < 3 * STEP)); then
  echo "STEP must divide a deck's $COPY updates of its copy and the $UPDATES after them, and an arm is read by its last three checkpoints" >&2
  exit 1
fi
decks=($DECKS)
for size in $SIZES; do
  if ! [[ "$size" =~ ^[1-9][0-9]*$ ]] || ((size > ${#decks[@]})); then
    echo "SIZES names how many of the ${#decks[@]} decks of DECKS a network plays, not $size" >&2
    exit 1
  fi
done
if [ "$(printf '%s\n' "${decks[@]}" | sort -u | wc -l)" -ne "${#decks[@]}" ]; then
  echo "DECKS names a deck twice" >&2
  exit 1
fi
if ((ITSELF < 1 || ITSELF >= DUELS)); then
  echo "ITSELF is the duels of an update against itself, some and not all of its $DUELS" >&2
  exit 1
fi
pinned=$(git submodule status --recursive)
if grep -q '^[^ ]' <<< "$pinned"; then
  echo "A submodule is not at its pinned revision: git submodule update --init --recursive" >&2
  exit 1
fi
if [[ " $PARTS " == *" measure "* ]]; then
  for file in "$GAMES/heldout.jsonl" "$GAMES/policies-revision.txt"; do
    if [ ! -f "$file" ]; then
      echo "Missing $file" >&2
      exit 1
    fi
  done
  # A recorded duel plays again only on the pilots it was recorded with.
  revision=$(git -C native/ygo-policies rev-parse HEAD)
  if [ "$(cat "$GAMES/policies-revision.txt")" != "$revision" ]; then
    echo "The games of $GAMES are of ygo-policies $(cat "$GAMES/policies-revision.txt"), not $revision" >&2
    exit 1
  fi
fi
python=(uv run python)
[ -z "$PYTHON" ] || python=("$PYTHON")
cargo build --release
mkdir -p "$RUNS"
LOGS=$RUNS
# shellcheck source=experiments/commands/at-once.sh
source "$(dirname "${BASH_SOURCE[0]}")/at-once.sh"

# The first SIZE decks, with commas between them.
decks_of() {
  local IFS=,
  echo "${decks[*]:0:$1}"
}

# made NAME FIRST UPDATES KEPT OPTION...: the run NAME to UPDATES updates
# after the FIRST it began with, from where it stands or from its start,
# which the options name (nothing for a network from scratch), with a
# checkpoint every KEPT updates.
made() {
  local name=$1 first=$2 updates=$3 kept=$4 last latest
  shift 4
  last=$(printf '%s/checkpoints/model-%05d.pt' "$RUNS/$name" $((first + updates - 1)))
  local arguments=(
    --device "$DEVICE" --workers "$WORKERS" --pool-threads "$THREADS" --pool-slots "$SLOTS"
    --run "$RUNS/$name" --opponents "$OPPONENTS" --duels "$DUELS" --eval-every 0
    --keep-checkpoints "$kept" --seed "$SEED" --ppo-epochs 1
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

# arm SIZE: the network of the first SIZE decks: its copy, and PPO on from it.
arm() {
  local size=$1 name learners copied
  name=$(printf 'decks-%02d' "$size")
  learners=$(decks_of "$size")
  copied=$((size * COPY))
  made "$name-copy" 0 "$copied" "$COPY" --learner "$learners" \
    --imitation-only-updates 1000000 --imitation-value-weight 0.5 --imitation-value-positions 4
  made "$name" "$copied" "$UPDATES" "$STEP" --learner "$learners" \
    --self-play-duels "$ITSELF" \
    --resume "$(printf '%s/checkpoints/model-%05d.pt' "$RUNS/$name-copy" $((copied - 1)))" \
    --resume-learning-rate "$RATE"
}

if [[ " $PARTS " == *" arms "* ]]; then
  for size in $SIZES; do
    at_once "$(printf 'decks-%02d' "$size")" arm "$size"
  done
  all_done
  echo "The arms are made: their runs are in $RUNS."
fi

if [[ " $PARTS " == *" measure "* ]]; then
  mkdir -p "$OUT/heldout" "$OUT/screens"
  # shellcheck disable=SC2059
  screen() { printf "$SCREENS" "$1"; }

  # held FILE CHECKPOINT...: those checkpoints on the held-out deals.
  held() {
    local file=$1
    shift
    if [ ! -f "$OUT/heldout/$file.json" ]; then
      "${python[@]}" -m ygo_training evaluate-search "$@" --device "$DEVICE" \
        --workers "$WORKERS" --pool-threads "$THREADS" --pool-slots "$SLOTS" \
        --records "$GAMES/heldout.jsonl" \
        --output "$OUT/heldout/$file.json.part" >> "$OUT/heldout/tables.txt"
      mv "$OUT/heldout/$file.json.part" "$OUT/heldout/$file.json"
    fi
  }

  # screened FILE DECK [CHECKPOINT...]: those checkpoints on the deck's
  # screen, or its pilot where none is given, with how each game ended.
  screened() {
    local file=$1 deck=$2
    shift 2
    if [ ! -f "$OUT/screens/$file.json" ]; then
      "${python[@]}" -m ygo_training evaluate-suite "${@:---pilot}" --device "$DEVICE" \
        --workers "$WORKERS" --pool-threads "$THREADS" --pool-slots "$SLOTS" --seed "$SEED" \
        --suite "$(screen "$deck")" --output "$OUT/screens/$file.json.part" > /dev/null
      mv "$OUT/screens/$file.json.part" "$OUT/screens/$file.json"
    fi
  }

  largest=0
  for size in $SIZES; do
    name=$(printf 'decks-%02d' "$size")
    copied=$((size * COPY))
    found=("$(printf '%s/checkpoints/model-%05d.pt' "$RUNS/$name-copy" $((copied - 1)))")
    for ((point = UPDATES - 2 * STEP; point <= UPDATES; point += STEP)); do
      found+=("$(printf '%s/checkpoints/model-%05d.pt' "$RUNS/$name" $((copied + point - 1)))")
    done
    whole=1
    for checkpoint in "${found[@]}"; do
      [ -f "$checkpoint" ] || whole=0
    done
    if ((!whole)); then
      echo "$name is not made to its end: it is measured when it is"
      continue
    fi
    ((size > largest)) && largest=$size
    for run in "$name-copy" "$name"; do
      cp "$RUNS/$run/metrics.jsonl" "$OUT/$run.metrics.jsonl"
      cp "$RUNS/$run/run.json" "$OUT/$run.run.json"
    done
    held "$name" "${found[@]}"
    if [ "$CURVE" != 0 ]; then
      # The other checkpoints it kept of its PPO, for its curve.
      along=()
      for ((point = STEP; point < UPDATES - 2 * STEP; point += STEP)); do
        along+=("$(printf '%s/checkpoints/model-%05d.pt' "$RUNS/$name" $((copied + point - 1)))")
      done
      ((${#along[@]} == 0)) || held "$name.curve" "${along[@]}"
    fi
    for deck in "${decks[@]:0:size}"; do
      screened "$name.$deck" "$deck" "${found[@]}"
    done
  done
  for deck in "${decks[@]:0:largest}"; do
    screened "pilot.$deck" "$deck"
  done
  # shellcheck disable=SC2086
  "${python[@]}" tools/more_decks_report.py "$OUT" --sizes $SIZES --decks "${decks[@]}" \
    --copy "$COPY" --updates "$UPDATES" --step "$STEP" \
    --output "$OUT/report.json" | tee "$OUT/report.txt"
  echo "The measurements and what they say are in $OUT."
fi

if [[ " $PARTS " == *" beyond "* ]]; then
  mkdir -p "$OUT/beyond/unseen" "$OUT/beyond/head-to-head"
  sizes=($SIZES)
  # checkpoints SIZE: the network of the first SIZE decks, its copy at its
  # last checkpoint and its last three of PPO, one a line.
  checkpoints() {
    local name copied point
    name=$(printf 'decks-%02d' "$1")
    copied=$(($1 * COPY))
    printf '%s/checkpoints/model-%05d.pt\n' "$RUNS/$name-copy" $((copied - 1))
    for ((point = UPDATES - 2 * STEP; point <= UPDATES; point += STEP)); do
      printf '%s/checkpoints/model-%05d.pt\n' "$RUNS/$name" $((copied + point - 1))
    done
  }
  # played FILE [CHECKPOINT...]: those checkpoints playing the decks outside
  # training, or those decks' own pilots where none is given, with how each
  # game ended.
  played() {
    local file=$1
    shift
    if [ ! -f "$OUT/beyond/unseen/$file.json" ]; then
      "${python[@]}" -m ygo_training evaluate-suite "${@:---pilot}" --device "$DEVICE" \
        --workers "$WORKERS" --pool-threads "$THREADS" --pool-slots "$SLOTS" --seed "$SEED" \
        --suite "$UNSEEN" --output "$OUT/beyond/unseen/$file.json.part" > /dev/null
      mv "$OUT/beyond/unseen/$file.json.part" "$OUT/beyond/unseen/$file.json"
    fi
  }
  # against FILE CHECKPOINT HELD: the checkpoint in the mirror, HELD in the
  # other seat.
  against() {
    if [ ! -f "$OUT/beyond/head-to-head/$1.json" ]; then
      "${python[@]}" -m ygo_training evaluate-suite "$2" --device "$DEVICE" \
        --workers "$WORKERS" --pool-threads "$THREADS" --pool-slots "$SLOTS" --seed "$SEED" \
        --suite "$MIRROR" --opponent-checkpoint "$3" \
        --output "$OUT/beyond/head-to-head/$1.json.part" > /dev/null
      mv "$OUT/beyond/head-to-head/$1.json.part" "$OUT/beyond/head-to-head/$1.json"
    fi
  }
  # whole SIZE: whether the network is made to its end.
  whole() {
    local checkpoint
    while read -r checkpoint; do
      [ -f "$checkpoint" ] || return 1
    done < <(checkpoints "$1")
  }
  for size in "${sizes[@]}"; do
    if ! whole "$size"; then
      echo "$(printf 'decks-%02d' "$size") is not made to its end: it is measured when it is"
      continue
    fi
    mapfile -t found < <(checkpoints "$size")
    played "$(printf 'decks-%02d' "$size")" "${found[@]}"
  done
  played pilot
  # The network of the most decks against that of the fewest, each deal
  # from either seat: their checkpoints after as many updates of PPO.
  fewest=${sizes[0]}
  most=${sizes[$((${#sizes[@]} - 1))]}
  if whole "$fewest" && whole "$most" && ((fewest != most)); then
    mapfile -t few < <(checkpoints "$fewest")
    mapfile -t many < <(checkpoints "$most")
    for index in 1 2 3; do
      point=$((UPDATES - (3 - index) * STEP))
      against "mirror-$point" "${many[$index]}" "${few[$index]}"
      against "mirror-$point.back" "${few[$index]}" "${many[$index]}"
    done
  fi
  # shellcheck disable=SC2086
  "${python[@]}" tools/more_decks_beyond_report.py "$OUT/beyond" --sizes $SIZES \
    --copy "$COPY" --updates "$UPDATES" --step "$STEP" \
    --output "$OUT/beyond/report.json" | tee "$OUT/beyond/report.txt"
  echo "The measurements beyond the networks' decks and what they say are in $OUT/beyond."
fi
