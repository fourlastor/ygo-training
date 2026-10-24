#!/usr/bin/env bash
# One network of the thirteen decks of training on the model's new input
# (format 15, no deck list), longer, with UPGO beside it, and against itself
# alone (experiments/2026-10-16-thirteen-decks.md). Run from the repository
# root on a machine with a card. PARTS names what is made:
#
# arms (the default). Under RUNS:
#      copy        the pilots of the thirteen decks imitated from scratch, as
#                  decks-13-copy of experiments/commands/more-decks.sh was
#                  (every game seen once, its result at four positions),
#                  COPY updates a deck at 3e-4, seed SEED;
#      half        PPO on from the copy's last checkpoint, UPDATES updates at
#                  RATE, one pass, ITSELF of an update's DUELS duels against
#                  itself (any two of its decks, one in each seat) and the
#                  others its decks against the pilots of OPPONENTS, as
#                  decks-13 was;
#      half-upgo   the same with UPGO (`--upgo`);
#      itself      the same with every duel against itself.
# The copy is made first, then the three AT_ONCE at a time through an MPS
# server on a card (experiments/commands/at-once.sh). A copy keeps a
# checkpoint every COPY updates and the other runs every STEP. No run is
# evaluated as it goes: the checkpoints are measured afterwards. Finished
# runs are skipped and an interrupted run goes on.
#   AT_ONCE=3 bash experiments/commands/thirteen-decks.sh
#
# measure. Once the arms are made, on the machine their runs are on, into
# OUT: the copy at its last checkpoint and each arm at its last three (and
# half also after its first MIDWAY updates, the last three of them, as
# decks-13 was measured) on the screen of each deck; the same checkpoints
# but half's at MIDWAY on the held-out deals of GAMES and on the decks
# outside training (UNSEEN); and itself against half, their last three
# checkpoints matched, on every pairing of the decks (AGAINST), each in
# turn the one held. Then the report (tools/thirteen_decks_report.py),
# which takes the readings, with decks-13 and the pilots as BEFORE measured
# them on the same games. A run that is not made to its end is left for
# later, and a finished measurement is not made again. About two and a
# half hours on the CUDA machine by reckoning.
#   PARTS=measure bash experiments/commands/thirteen-decks.sh
set -euo pipefail
DEVICE=${DEVICE:-cuda}
WORKERS=${WORKERS:-24}
THREADS=${THREADS:-20}
SLOTS=${SLOTS:-16}
DUELS=${DUELS:-520}
PARTS=${PARTS:-arms}
DECKS=${DECKS:-blackwing,lightsworn,gishki,burn,crystal,dragunity,gladiator,heroes,infernity,monarch,morphtronic,rock-block,spellcaster}
OPPONENTS=${OPPONENTS:-blackwing,burn,crystal,dragunity,gishki,gladiator,heroes,infernity,lightsworn,monarch,morphtronic,rock-block,spellcaster}
COPY=${COPY:-240}
UPDATES=${UPDATES:-3840}
STEP=${STEP:-120}
ITSELF=${ITSELF:-260}
RATE=${RATE:-1e-4}
SEED=${SEED:-2012}
RUNS=${RUNS:-runs/thirteen-decks}
OUT=${OUT:-analysis/thirteen-decks}
# The updates of PPO decks-13 had made, by which half is set beside it.
MIDWAY=${MIDWAY:-1920}
# Where the experiment on more decks left decks-13's measurements and the
# pilots' (its measure and beyond parts), the held-out deals, a deck's
# screen (the deck's name in place of %s), the decks outside training
# against the pilots of training, and every pairing of the decks of
# training against a checkpoint held: smaller ones try this part small.
BEFORE=${BEFORE:-analysis/more-decks}
GAMES=${GAMES:-analysis/search-leads}
SCREENS=${SCREENS:-eval_suites/ygo-policies-%s-screen.json}
# UNSEEN set empty leaves the decks outside training unmeasured. On
# 2026-10-17 a stall duel of Countdown's against Infernity filled its duel's
# 256 MB arena and the engine's exception aborted the process; the engine
# reuses what Lua frees since, and a duel whose arena still fills is cut off.
UNSEEN=${UNSEEN-eval_suites/ygo-policies-unseen-played.json}
AGAINST=${AGAINST:-eval_suites/thirteen-decks-head-to-head.json}
# The interpreter of a machine that has no uv: the benchmark's setup leaves one.
PYTHON=${PYTHON:-}

for part in $PARTS; do
  if [ "$part" != arms ] && [ "$part" != measure ]; then
    echo "PARTS names arms and measure, not $part" >&2
    exit 1
  fi
done
IFS=, read -r -a decks <<< "$DECKS"
if ((COPY % STEP || UPDATES % STEP || UPDATES < 3 * STEP)); then
  echo "STEP must divide a deck's $COPY updates of the copy and the $UPDATES after them" >&2
  exit 1
fi
if ((ITSELF < 1 || ITSELF >= DUELS)); then
  echo "ITSELF is the duels of an update of half against itself, some and not all of its $DUELS" >&2
  exit 1
fi
if ((MIDWAY % STEP || MIDWAY < 3 * STEP || MIDWAY >= UPDATES)); then
  echo "MIDWAY is a number of updates of half, three STEP at least and fewer than $UPDATES, that STEP divides" >&2
  exit 1
fi
if [[ " $PARTS " == *" measure "* ]]; then
  for file in "$GAMES/heldout.jsonl" "$GAMES/policies-revision.txt" ${UNSEEN:+"$UNSEEN"} "$AGAINST"; do
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

copied=$((${#decks[@]} * COPY))

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
      --updates $((first + updates - 1 - 10#${latest#model-}))
  fi
}

# arm NAME OPTION...: PPO on from the copy's last checkpoint.
arm() {
  local name=$1
  shift
  made "$name" "$copied" "$UPDATES" "$STEP" "$@" \
    --resume "$(printf '%s/checkpoints/model-%05d.pt' "$RUNS/copy" $((copied - 1)))" \
    --resume-learning-rate "$RATE"
}

if [[ " $PARTS " == *" arms "* ]]; then
  made copy 0 "$copied" "$COPY" \
    --imitation-only-updates 1000000 --imitation-value-weight 0.5 --imitation-value-positions 4
  at_once half arm half --self-play-duels "$ITSELF"
  at_once half-upgo arm half-upgo --self-play-duels "$ITSELF" --upgo
  at_once itself arm itself --self-play-duels "$DUELS"
  all_done
  echo "The arms are made: their runs are in $RUNS."
fi

if [[ " $PARTS " == *" measure "* ]]; then
  mkdir -p "$OUT/screens" "$OUT/heldout" "$OUT/unseen" "$OUT/head-to-head"
  # shellcheck disable=SC2059
  screen() { printf "$SCREENS" "$1"; }
  # checkpoint RUN UPDATES: a checkpoint of an arm after so many updates of
  # PPO, or of the copy after so many in all.
  checkpoint() {
    if [ "$1" = copy ]; then
      printf '%s/copy/checkpoints/model-%05d.pt' "$RUNS" $(($2 - 1))
    else
      printf '%s/%s/checkpoints/model-%05d.pt' "$RUNS" "$1" $((copied + $2 - 1))
    fi
  }
  # three RUN UPDATES: its checkpoints after the last three STEP up to UPDATES.
  three() {
    local point
    for ((point = $2 - 2 * STEP; point <= $2; point += STEP)); do
      echo "$(checkpoint "$1" "$point")"
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

  declare -A measured=(
    [copy]="$(checkpoint copy "$copied")"
    [half]="$(three half "$MIDWAY")"$'\n'"$(three half "$UPDATES")"
    [half-upgo]="$(three half-upgo "$UPDATES")"
    [itself]="$(three itself "$UPDATES")"
  )
  made_runs=()
  for run in copy half half-upgo itself; do
    mapfile -t found <<< "${measured[$run]}"
    whole=1
    for path in "${found[@]}"; do
      [ -f "$path" ] || whole=0
    done
    if ((!whole)); then
      echo "$run is not made to its end: it is measured when it is"
      continue
    fi
    made_runs+=("$run")
    cp "$RUNS/$run/metrics.jsonl" "$OUT/$run.metrics.jsonl"
    cp "$RUNS/$run/run.json" "$OUT/$run.run.json"
    for deck in "${decks[@]}"; do
      suite screens "$run.$deck" "$(screen "$deck")" "${found[@]}"
    done
    # Beside the readings: the end of each run, but for half after MIDWAY.
    [ "$run" != half ] || mapfile -t found < <(three half "$UPDATES")
    held "$run" "${found[@]}"
    [ -z "$UNSEEN" ] || suite unseen "$run" "$UNSEEN" "${found[@]}"
  done
  if [[ " ${made_runs[*]} " == *" half "* && " ${made_runs[*]} " == *" itself "* ]]; then
    mapfile -t mine < <(three itself "$UPDATES")
    mapfile -t theirs < <(three half "$UPDATES")
    for index in 0 1 2; do
      point=$((UPDATES - (2 - index) * STEP))
      suite head-to-head "itself-$point" "$AGAINST" "${mine[$index]}" \
        --opponent-checkpoint "${theirs[$index]}"
      suite head-to-head "itself-$point.back" "$AGAINST" "${theirs[$index]}" \
        --opponent-checkpoint "${mine[$index]}"
    done
  fi
  "${python[@]}" tools/thirteen_decks_report.py "$OUT" --before "$BEFORE" --decks "${decks[@]}" \
    --copy "$COPY" --updates "$UPDATES" --midway "$MIDWAY" --step "$STEP" \
    --output "$OUT/report.json" | tee "$OUT/report.txt"
  echo "The measurements and what they say are in $OUT."
fi
