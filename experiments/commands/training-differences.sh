#!/usr/bin/env bash
# Three of the ways ygo-agent's trainer differs from this one, each tried on
# its own against a run already made (experiments/2026-10-16-training-differences.md).
# Each is PPO of Blackwing alone on from the copy that CONTROL began from
# (decks-01 of experiments/commands/more-decks.sh: its learning rate, one
# pass, ITSELF of DUELS duels an update against itself and the others
# against the pilots of OPPONENTS, seed SEED), with one thing changed:
#
#   small-updates  an update of a SMALLER-th of the duels, as many of them
#                  against itself: SMALLER updates for each of the control's,
#                  the same games and the same steps of the optimizer a game,
#                  and the policy the games are played by new SMALLER times
#                  as often (theirs: some twenty games an update);
#   their-step     their minibatch and clip: `--minibatch 512 --clip 0.25`;
#   upgo           their UPGO added to each transaction's advantage: `--upgo`.
#
# Run from the repository root, on the machine CONTROL is on. PARTS names
# what is made:
#
# arms (the default). The three runs, each as far as the games of the last
# of READ of the control's updates, AT_ONCE of them at the same time
# (experiments/commands/at-once.sh). A finished run is skipped and an
# interrupted one goes on.
#
# measure. Each run's checkpoints after the games of each of READ of the
# control's updates, on the held-out deals of GAMES, and the control's own
# checkpoints there again, under the rule against repeated commands the runs
# trained under where the control did not; with each run's figures, into
# OUT, then the report (tools/training_differences_report.py), which takes
# the readings, and sets the control's checkpoints as they were measured
# before the rule (BEFORE) beside them.
#   PARTS=measure bash experiments/commands/training-differences.sh
# Its control and copy read the deck lists (format 14), which the input
# dropped in format 15: it runs at 05d27141, the revision before.
set -euo pipefail
DEVICE=${DEVICE:-cuda}
WORKERS=${WORKERS:-24}
THREADS=${THREADS:-20}
SLOTS=${SLOTS:-16}
DUELS=${DUELS:-520}
ITSELF=${ITSELF:-260}
PARTS=${PARTS:-arms}
SEED=${SEED:-2012}
RATE=${RATE:-1e-4}
LEARNER=${LEARNER:-blackwing}
OPPONENTS=${OPPONENTS:-blackwing,burn,crystal,dragunity,gishki,gladiator,heroes,infernity,lightsworn,monarch,morphtronic,rock-block,spellcaster}
# The run the others are set against, and the copy it began from.
CONTROL=${CONTROL:-runs/more-decks/decks-01}
COPIED=${COPIED:-runs/more-decks/decks-01-copy/checkpoints/model-00239.pt}
# The control's updates whose games each run is read at: its last three; the
# runs stop at the last.
READ=${READ:-720 840 960}
# How many updates of small-updates make one of the control's.
SMALLER=${SMALLER:-4}
# A run keeps a checkpoint every STEP of its updates (and its last).
STEP=${STEP:-120}
RUNS=${RUNS:-runs/training-differences}
OUT=${OUT:-analysis/training-differences}
GAMES=${GAMES:-analysis/search-leads}
# The control's checkpoints on the held-out deals as the experiment on more
# decks measured them, before the rule against repeated commands.
BEFORE=${BEFORE:-analysis/more-decks/heldout/decks-01.curve.json}
# The interpreter of a machine that has no uv: the benchmark's setup leaves one.
PYTHON=${PYTHON:-}

for part in $PARTS; do
  if [ "$part" != arms ] && [ "$part" != measure ]; then
    echo "PARTS names arms and measure, not $part" >&2
    exit 1
  fi
done
read -r -a points <<< "$READ"
if ((${#points[@]} != 3)); then
  echo "READ names the three updates of the control a run is read at, not $READ" >&2
  exit 1
fi
if ((DUELS % SMALLER || ITSELF % SMALLER || ITSELF < 1 || ITSELF >= DUELS)); then
  echo "SMALLER must divide the $DUELS duels of an update and the $ITSELF against itself, some and not all" >&2
  exit 1
fi
began=$(basename "$COPIED" .pt)
began=$((10#${began#model-} + 1))
if ((began % STEP)); then
  echo "STEP must divide the $began updates the copy had made" >&2
  exit 1
fi
for point in "${points[@]}"; do
  if ((point % STEP)); then
    echo "STEP must divide the control's updates the runs are read at, not $point" >&2
    exit 1
  fi
done
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
if [ ! -f "$COPIED" ]; then
  echo "Missing $COPIED, the copy the runs begin from" >&2
  exit 1
fi
python=(uv run python)
[ -z "$PYTHON" ] || python=("$PYTHON")
cargo build --release
mkdir -p "$RUNS"
LOGS=$RUNS
# shellcheck source=experiments/commands/at-once.sh
source "$(dirname "${BASH_SOURCE[0]}")/at-once.sh"

names=(small-updates their-step upgo)
# How many of a run's updates make one of the control's, and what it is told
# beside what the control was.
declare -A scales=([small-updates]=$SMALLER [their-step]=1 [upgo]=1)
declare -A changes=([small-updates]="" [their-step]="--minibatch 512 --clip 0.25" [upgo]=--upgo)

# checkpoint FOLDER SCALE POINT: a run's checkpoint after the games of POINT
# of the control's updates.
checkpoint() {
  printf '%s/checkpoints/model-%05d.pt' "$1" $((began + $2 * $3 - 1))
}

# run NAME: the run, from the copy or from where it stands.
run() {
  local name=$1 scale=${scales[$1]} updates latest
  local -a change
  read -r -a change <<< "${changes[$1]}"
  updates=$((scale * points[2]))
  local arguments=(
    --device "$DEVICE" --workers "$WORKERS" --pool-threads "$THREADS" --pool-slots "$SLOTS"
    --run "$RUNS/$name" --opponents "$OPPONENTS" --duels $((DUELS / scale)) --eval-every 0
    --keep-checkpoints "$STEP" --seed "$SEED" --ppo-epochs 1 --learner "$LEARNER"
    --self-play-duels $((ITSELF / scale)) ${change[@]+"${change[@]}"}
  )
  if ! ls "$RUNS/$name"/checkpoints/model-*.pt > /dev/null 2>&1; then
    "${python[@]}" -m ygo_training train "${arguments[@]}" --resume "$COPIED" \
      --resume-learning-rate "$RATE" --updates "$updates"
  elif [ ! -f "$(checkpoint "$RUNS/$name" "$scale" "${points[2]}")" ]; then
    latest=$(basename "$(ls "$RUNS/$name"/checkpoints/model-*.pt | sort | tail -1)" .pt)
    "${python[@]}" -m ygo_training train "${arguments[@]}" --continue \
      --updates $((began + updates - 1 - 10#${latest#model-}))
  fi
}

if [[ " $PARTS " == *" arms "* ]]; then
  for name in "${names[@]}"; do
    at_once "$name" run "$name"
  done
  all_done
  echo "The runs are made: they are in $RUNS."
fi

if [[ " $PARTS " == *" measure "* ]]; then
  mkdir -p "$OUT/heldout" "$OUT/runs"
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
  for name in control "${names[@]}"; do
    if [ "$name" = control ]; then
      folder=$CONTROL scale=1
    else
      folder=$RUNS/$name scale=${scales[$name]}
    fi
    found=()
    for point in "${points[@]}"; do
      found+=("$(checkpoint "$folder" "$scale" "$point")")
    done
    if [ ! -f "${found[2]}" ]; then
      echo "$name is not made to its end (${found[2]}): left for later"
      continue
    fi
    cp "$folder/run.json" "$OUT/runs/$name.run.json"
    cp "$folder/metrics.jsonl" "$OUT/runs/$name.metrics.jsonl"
    held "$name" "${found[@]}"
  done
  before=()
  [ ! -f "$BEFORE" ] || before=(--before "$BEFORE")
  "${python[@]}" tools/training_differences_report.py "$OUT" --began "$began" --read "${points[@]}" \
    --smaller "$SMALLER" ${before[@]+"${before[@]}"} --output "$OUT/report.json" | tee "$OUT/report.txt"
fi
