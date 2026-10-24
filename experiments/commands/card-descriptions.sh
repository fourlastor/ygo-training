#!/usr/bin/env bash
# Card descriptions and hidden identities, for decks outside training
# (experiments/2026-10-17-card-descriptions.md). Run from the repository
# root on a machine with a card. PARTS names what is made:
#
# arms (the default). Under RUNS, three networks of the thirteen decks of
# training, each made as half of experiments/commands/thirteen-decks.sh was
# and told more of a card than its input gives, by the table TABLE
# (checkpoint format 16):
#      both          the cards' descriptions read, and the identities of a
#                    HIDE share of the cards hidden in each update;
#      descriptions  the descriptions alone;
#      hidden        the identities hidden alone.
# Each is its copy (NAME-copy: the pilots of the thirteen decks imitated
# from scratch, COPY updates a deck at 3e-4, every game seen once, its
# result at four positions, seed SEED), then PPO on from the copy's last
# checkpoint (NAME: UPDATES updates at RATE, one pass, ITSELF of an
# update's DUELS duels against itself, any two of its decks, and the others
# its decks against the pilots of OPPONENTS). The three go AT_ONCE at a
# time through an MPS server on a card (experiments/commands/at-once.sh),
# each its copy and then its PPO. A copy keeps a checkpoint every COPY
# updates and PPO every STEP. No run is evaluated as it goes: the
# checkpoints are measured afterwards. Finished runs are skipped and an
# interrupted run goes on. About thirteen and a half hours by reckoning.
#   AT_ONCE=3 bash experiments/commands/card-descriptions.sh
#
# measure. Once the arms are made, on the machine their runs are on, into
# OUT: each arm, and the control (CONTROL, half of the experiment on
# thirteen decks), by its last three checkpoints on the screen of each deck
# of training and on the decks outside training (UNSEEN); each one's copy at
# its last checkpoint (the control's CONTROL_COPY) and the decks' own pilots
# on those decks too. Then the report (tools/card_descriptions_report.py),
# which takes the readings. A run that is not made to its end is left for
# later, and a finished measurement is not made again. About two and a half
# hours by reckoning.
#   PARTS=measure bash experiments/commands/card-descriptions.sh
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
UPDATES=${UPDATES:-1920}
STEP=${STEP:-120}
ITSELF=${ITSELF:-260}
RATE=${RATE:-1e-4}
SEED=${SEED:-2012}
HIDE=${HIDE:-0.25}
TABLE=${TABLE:-data/card-descriptions.npz}
RUNS=${RUNS:-runs/card-descriptions}
OUT=${OUT:-analysis/card-descriptions}
# The control: half of the experiment on thirteen decks, made at the
# policies' 1b85b63, whose games are those of the engine pinned since
# (experiments/2026-10-16-thirteen-decks.md), and its copy.
CONTROL=${CONTROL:-runs/thirteen-decks/half}
CONTROL_COPY=${CONTROL_COPY:-runs/thirteen-decks/copy/checkpoints/model-03119.pt}
# A deck's screen (its name in place of %s), and the decks outside training
# against the pilots of training: smaller ones try this part small.
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
if ((COPY % STEP || UPDATES % STEP || UPDATES < 3 * STEP)); then
  echo "STEP must divide a deck's $COPY updates of a copy and the $UPDATES after them" >&2
  exit 1
fi
if ((ITSELF < 1 || ITSELF >= DUELS)); then
  echo "ITSELF is the duels of an update against itself, some and not all of its $DUELS" >&2
  exit 1
fi
for file in "$TABLE" "$UNSEEN"; do
  if [ ! -f "$file" ]; then
    echo "Missing $file" >&2
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
mkdir -p "$RUNS"
LOGS=$RUNS
# shellcheck source=experiments/commands/at-once.sh
source "$(dirname "${BASH_SOURCE[0]}")/at-once.sh"

copied=$((${#decks[@]} * COPY))
ARMS=(both descriptions hidden)
declare -A told=(
  [both]="--card-table $TABLE --card-descriptions --hide-identities $HIDE"
  [descriptions]="--card-table $TABLE --card-descriptions"
  [hidden]="--card-table $TABLE --hide-identities $HIDE"
)

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

# whole NAME: the arm's copy, then its PPO on from the copy's last
# checkpoint, both told of a card what the arm is.
whole() {
  local name=$1
  local cards
  read -r -a cards <<< "${told[$name]}"
  made "$name-copy" 0 "$copied" "$COPY" "${cards[@]}" \
    --imitation-only-updates 1000000 --imitation-value-weight 0.5 --imitation-value-positions 4
  made "$name" "$copied" "$UPDATES" "$STEP" "${cards[@]}" --self-play-duels "$ITSELF" \
    --resume "$(printf '%s/checkpoints/model-%05d.pt' "$RUNS/$name-copy" $((copied - 1)))" \
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
  # three FOLDER: a run's checkpoints after the last three STEP of its UPDATES of PPO.
  three() {
    local point
    for ((point = UPDATES - 2 * STEP; point <= UPDATES; point += STEP)); do
      printf '%s/checkpoints/model-%05d.pt\n' "$1" $((copied + point - 1))
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

  declare -A folders=([both]="$RUNS/both" [descriptions]="$RUNS/descriptions" [hidden]="$RUNS/hidden" [control]="$CONTROL")
  declare -A copies=([control]="$CONTROL_COPY")
  for arm in "${ARMS[@]}"; do
    copies[$arm]=$(printf '%s/checkpoints/model-%05d.pt' "$RUNS/$arm-copy" $((copied - 1)))
  done
  for run in "${ARMS[@]}" control; do
    mapfile -t found < <(three "${folders[$run]}")
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
  "${python[@]}" tools/card_descriptions_report.py "$OUT" --decks "${decks[@]}" \
    --table "$TABLE" --copy "$COPY" --updates "$UPDATES" --step "$STEP" \
    --output "$OUT/report.json" | tee "$OUT/report.txt"
  echo "The measurements and what they say are in $OUT."
fi
