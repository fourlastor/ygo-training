#!/usr/bin/env bash
# A Cyber champion (experiments/2026-10-23-cyber-champion.md): the recipe of
# the network of Blackwing alone (decks-01 of more decks) for Cyber, in two
# arms: cyber-new from a copy of the improved pilot (the policies at PILOTS,
# the branch BRANCH, in a worktree of its own at WORKTREE), cyber-old from a
# copy of the pilot as pinned. Run from the repository root on the CUDA
# machine, the branch in this checkout's policies repository. PARTS names
# what is made, in its order:
#
# arms. cyber-old, then cyber-new: the pilot of Cyber imitated from scratch,
# COPY updates of DUELS against the thirteen decks' pilots, every game seen
# once and its result at four of its positions, at 3e-4, in minibatches of
# 256 with no duels in flight; then PPO on from the copy's last checkpoint
# for UPDATES updates at RATE, ITSELF of an update's duels the mirror against
# itself and the others against the thirteen decks' pilots, with fast's
# settings (OPTIONS adds more: the overlap's, for one); a checkpoint every
# STEP. A finished run is skipped, an interrupted one goes on.
#
# measure. Into OUT: each arm's checkpoints after every STEP of PPO on
# Cyber's yardstick (eval_suites/cyber-yardstick.json, the deals all71 and
# the pilots were measured on by seventy-one decks); cyber-new's last three
# against all71 (NETWORK) in the Cyber mirror
# (eval_suites/cyber-against-checkpoint.json), from both seats; the runs'
# metrics; then the report (tools/cyber_champion_report.py).
#
# About 1.5 hours an arm by reckoning, and 20 minutes to measure.
#   bash experiments/commands/cyber-champion.sh
set -euo pipefail
DEVICE=${DEVICE:-cuda}
WORKERS=${WORKERS:-24}
THREADS=${THREADS:-20}
SLOTS=${SLOTS:-16}
DUELS=${DUELS:-520}
PARTS=${PARTS:-arms measure}
THIRTEEN=${THIRTEEN:-blackwing,lightsworn,gishki,burn,crystal,dragunity,gladiator,heroes,infernity,monarch,morphtronic,rock-block,spellcaster}
COPY=${COPY:-240}
UPDATES=${UPDATES:-720}
STEP=${STEP:-120}
ITSELF=${ITSELF:-260}
RATE=${RATE:-1e-4}
SEED=${SEED:-2012}
# fast's settings: two updates' duels in flight, the pool's slots to hold
# them, minibatches of 512, the calls as they are in two groups.
IN_FLIGHT=${IN_FLIGHT:-1040}
FAST_SLOTS=${FAST_SLOTS:-52}
MINIBATCH=${MINIBATCH:-512}
OPTIONS=${OPTIONS:-}
# The updates of PPO reading 2 is taken at.
READ_AT=${READ_AT:-120 720}
BRANCH=${BRANCH:-pilots-against-the-network}
PILOTS=${PILOTS:-d696a4b}
WORKTREE=${WORKTREE:-../ygo-pilots}
NETWORK=${NETWORK:-runs/seventy-one-decks/all71/checkpoints/model-27599.pt}
REFERENCES=${REFERENCES:-analysis/seventy-one-decks}
NEW_PILOT=${NEW_PILOT:-analysis/improved-pilots/branch-$PILOTS/new.json}
RUNS=${RUNS:-runs/cyber-champion}
OUT=${OUT:-analysis/cyber-champion}
YARDSTICK=${YARDSTICK:-eval_suites/cyber-yardstick.json}
MIRROR=${MIRROR:-eval_suites/cyber-against-checkpoint.json}
# The interpreter of a machine that has no uv: the benchmark's setup leaves one.
PYTHON=${PYTHON:-}
MAIN=$PWD
VENDOR=$MAIN/native/ygo-policies/vendor
# The runs are made from two checkouts: their places name the same ones.
[[ "$RUNS" == /* ]] || RUNS=$MAIN/$RUNS
[[ "$OUT" == /* ]] || OUT=$MAIN/$OUT

for part in $PARTS; do
  if [ "$part" != arms ] && [ "$part" != measure ]; then
    echo "PARTS names arms and measure, not $part" >&2
    exit 1
  fi
done
if ((UPDATES % STEP || UPDATES < 3 * STEP)); then
  echo "UPDATES is a multiple of STEP, three of them at least: not $UPDATES" >&2
  exit 1
fi
for path in "$NETWORK" "$YARDSTICK" "$MIRROR"; do
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
[[ "${python[0]}" == /* ]] || [ "${python[0]}" = uv ] || python[0]=$MAIN/${python[0]}
read -r -a options <<< "$OPTIONS"
cargo build --release
mkdir -p "$RUNS" "$OUT"

# The improved pilot: a worktree of this repository at its HEAD, its
# submodules filled from this checkout's, the policies at the branch.
if [ ! -d "$WORKTREE" ]; then
  git worktree add --detach "$WORKTREE" HEAD
fi
git -C "$WORKTREE" checkout -q --detach "$(git rev-parse HEAD)"
(
  cd "$WORKTREE"
  git -c protocol.file.allow=always \
    -c submodule.native/ygo-policies.url="$MAIN/native/ygo-policies" \
    -c submodule.native/ygo-model-input.url="$MAIN/native/ygo-model-input" \
    submodule update --init native/ygo-policies native/ygo-model-input
  cd native/ygo-policies
  git fetch -q "$MAIN/native/ygo-policies" "$BRANCH"
  git checkout -q --detach FETCH_HEAD
  git -c protocol.file.allow=always \
    -c submodule.vendor/ocgcore.url="$VENDOR/ocgcore" \
    -c submodule.vendor/CardScripts.url="$VENDOR/CardScripts" \
    -c submodule.vendor/BabelCdb.url="$VENDOR/BabelCdb" \
    submodule update --init vendor/ocgcore vendor/CardScripts vendor/BabelCdb
  git submodule init > /dev/null
  cd vendor/ocgcore
  git -c protocol.file.allow=always -c submodule.lua/src.url="$VENDOR/ocgcore/lua/src" \
    submodule update --init lua/src
  git submodule init > /dev/null
)
found=$(git -C "$WORKTREE/native/ygo-policies" rev-parse --short=7 HEAD)
if [ "$found" != "${PILOTS:0:7}" ]; then
  echo "The branch is at $found, not at the pilots measured ($PILOTS)" >&2
  exit 1
fi
(cd "$WORKTREE" && cargo build --release)

# made ROOT NAME FIRST UPDATES KEPT OPTION...: the run NAME, played by the
# checkout at ROOT, to UPDATES updates after the FIRST it began with, from
# where it stands or from its start, with a checkpoint every KEPT updates.
made() {
  local root=$1 name=$2 first=$3 updates=$4 kept=$5 last latest
  shift 5
  last=$(printf '%s/checkpoints/model-%05d.pt' "$RUNS/$name" $((first + updates - 1)))
  local arguments=(
    --device "$DEVICE" --workers "$WORKERS" --pool-threads "$THREADS"
    --run "$RUNS/$name" --learner cyber --opponents "$THIRTEEN" --duels "$DUELS"
    --eval-every 0 --keep-checkpoints "$kept" --seed "$SEED" --ppo-epochs 1
  )
  if ! ls "$RUNS/$name"/checkpoints/model-*.pt > /dev/null 2>&1; then
    (cd "$root" && "${python[@]}" -m ygo_training train "${arguments[@]}" "$@" --updates "$updates")
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
    (cd "$root" && "${python[@]}" -m ygo_training train "${arguments[@]}" "${again[@]}" \
      --continue --updates $((first + updates - 1 - 10#${latest#model-})))
  fi
}

# arm NAME ROOT: the copy of the pilot of the checkout at ROOT, and PPO on
# from it.
arm() {
  local name=$1 root=$2
  echo "$(date +%H:%M) $name begins"
  made "$root" "$name-copy" 0 "$COPY" "$COPY" --pool-slots "$SLOTS" \
    --imitation-only-updates 1000000 --imitation-value-weight 0.5 \
    --imitation-value-positions 4 --minibatch 256 --duels-in-flight 0
  made "$root" "$name" "$COPY" "$UPDATES" "$STEP" --pool-slots "$FAST_SLOTS" \
    --self-play-duels "$ITSELF" --rollout-estimates kept \
    --duels-in-flight "$IN_FLIGHT" --minibatch "$MINIBATCH" --inference-graphs 0 \
    --pool-groups 2 "${options[@]}" \
    --resume "$(printf '%s/checkpoints/model-%05d.pt' "$RUNS/$name-copy" $((COPY - 1)))" \
    --resume-learning-rate "$RATE"
  echo "$(date +%H:%M) $name ends"
}

if [[ " $PARTS " == *" arms "* ]]; then
  arm cyber-old "$MAIN"
  arm cyber-new "$WORKTREE"
  echo "The arms are made: $RUNS."
fi

if [[ " $PARTS " == *" measure "* ]]; then
  mkdir -p "$OUT/yardstick" "$OUT/mirror"
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
  for name in cyber-old cyber-new; do
    points=()
    for ((point = STEP; point <= UPDATES; point += STEP)); do
      path=$(printf '%s/checkpoints/model-%05d.pt' "$RUNS/$name" $((COPY + point - 1)))
      if [ ! -f "$path" ]; then
        echo "Missing $path: $name is measured when it is made" >&2
        exit 1
      fi
      points+=("$path")
    done
    suite "yardstick/$name" "$YARDSTICK" "${points[@]}"
    cp "$RUNS/$name-copy/metrics.jsonl" "$OUT/$name-copy.metrics.jsonl"
    cp "$RUNS/$name/metrics.jsonl" "$OUT/$name.metrics.jsonl"
    cp "$RUNS/$name/run.json" "$OUT/$name.run.json"
  done
  for ((point = UPDATES - 2 * STEP; point <= UPDATES; point += STEP)); do
    path=$(printf '%s/checkpoints/model-%05d.pt' "$RUNS/cyber-new" $((COPY + point - 1)))
    stem=$(basename "$path" .pt)
    suite "mirror/$stem.plays" "$MIRROR" "$path" --opponent-checkpoint "$NETWORK"
    suite "mirror/$stem.held" "$MIRROR" "$NETWORK" --opponent-checkpoint "$path"
  done
  pilot=()
  [ ! -f "$NEW_PILOT" ] || pilot=(--new-pilot "$NEW_PILOT")
  read -r -a read_at <<< "$READ_AT"
  "${python[@]}" tools/cyber_champion_report.py "$OUT" --references "$REFERENCES" \
    --copied "$COPY" "${pilot[@]}" --read-at "${read_at[@]}" --output "$OUT/report.json" \
    | tee "$OUT/report.txt"
  echo "The measurements and what they say are in $OUT."
fi
