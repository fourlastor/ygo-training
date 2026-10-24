#!/usr/bin/env bash
# Self-play, and what sees the plays the rate against pilots does not
# (experiments/2026-10-11-self-play.md). Run from the repository root on
# the CUDA machine. PARTS names what is made:
#
# base (the default). Nothing in it has a reading of its own: it measures
# START, the strongest checkpoint, with what the arms are then measured with.
#   1. Head to head: the checkpoints of START's run after the updates of
#      EARLIER, START itself and the deck's pilot, each against START held
#      in the other seat, on the 2,048 deals of the mirror in AGAINST.
#   2. Exploiters of START, one for each training seed of SEEDS: a learner
#      that starts as START and plays every duel of EXPLOIT updates against
#      START held as it is, measured every EVERY updates against it and
#      against the pilots (SUITE).
# About three hours, by what the same games take with a pilot in the other
# seat: ten minutes head to head, the rest the two exploiters.
#
# arms. The two arms of the experiment, each ARM updates of DUELS duels on
# from START, measured every STEP of them:
#      with-itself    ITSELF of an update's duels against itself, the
#                     others against the pilots;
#      pilots-alone   every duel against the pilots.
#   1. START and each arm's checkpoints on the held-out logged games.
#   2. Head to head: the last three checkpoints of with-itself against the
#      last of pilots-alone, each in turn the one held.
#   3. START and the last three checkpoints of each arm against the decks
#      outside training (UNSEEN).
#   4. The named plays, by START and the last checkpoint of each
#      arm, on the processor (PLAYS: options of tools/named_plays.py, to try
#      it at a smaller size).
# Twelve hours as it ran: seven for with-itself, four for pilots-alone, the
# rest the measurements.
# The arms go on from where they stand: with a longer ARM they are taken to
# it, and are then measured in a folder of their own (OUT), since some
# measurements are named by the arm and not by its length. With SINCE,
# with-itself's last three checkpoints are also measured with its own
# checkpoint after SINCE updates, each in turn the one held: whether it has
# gained on itself.
#   ARM=1920 SINCE=960 OUT=analysis/self-play-further PARTS=arms bash ...
# ARM_SEED gives both arms another training seed than the experiment's 2012:
# other deals and other draws from the same start, measured on the same
# deals, in folders of their own.
#   ARM_SEED=2013 RUNS=runs/self-play-2013 OUT=analysis/self-play-2013 \
#     PARTS=arms bash ...
#
# exploiters. What a learner that looks for its habits wins against the
# last checkpoint of each arm, as the base has it for START: the arms come
# first.
#   1. Exploiters of each arm's last checkpoint, one for each training seed
#      of SEEDS, made as the base's of START (which are made here too where
#      they are not there).
#   2. Every exploiter's checkpoints after its last three EVERY updates and
#      its victim on the mirror's deals, each in turn the one held: every
#      deal from either seat.
#   3. What the exploiters of START found, against each arm's last
#      checkpoint: START and the last checkpoint of each of its exploiters
#      with that checkpoint, each in turn the one held.
# Seven to nine hours: 1.6 an exploiter by the base and 2.1 for the first of
# with-itself, four of them, and 48 measurements of under a minute.
# With a longer ARM the exploiters are of the arms' checkpoints after it, in
# run folders of their own (exploiter-ARM-SEED.LENGTH), and are measured in
# OUT as the others:
#   ARM=1920 OUT=analysis/self-play-further PARTS=exploiters bash ...
#
# What is there is told at the end. Finished measurements and runs are
# skipped and an interrupted run goes on.
#
# AT_ONCE has so many of a part's trainings go at the same time, through an
# MPS server on a card (experiments/commands/at-once.sh): the two arms, or
# the exploiters. Each then writes to a log beside its run (RUNS/NAME.log),
# and takes half as long again as with the card to itself; the measurements
# come after them, one at a time, as ever.
#   AT_ONCE=2 PARTS=arms bash experiments/commands/self-play.sh
set -euo pipefail
DEVICE=${DEVICE:-cuda}
WORKERS=${WORKERS:-24}
THREADS=${THREADS:-20}
SLOTS=${SLOTS:-16}
DUELS=${DUELS:-520}
PARTS=${PARTS:-base}
EXPLOIT=${EXPLOIT:-240}
EVERY=${EVERY:-30}
SEEDS=${SEEDS:-2012 2013}
ARM=${ARM:-960}
STEP=${STEP:-120}
ITSELF=${ITSELF:-260}
RATE=${RATE:-1e-4}
OPPONENTS=${OPPONENTS:-blackwing,burn,crystal,dragunity,gishki,gladiator,heroes,infernity,lightsworn,monarch,morphtronic,rock-block,spellcaster}
START=${START:-runs/ppo-long-rate/one-pass-long/checkpoints/model-02279.pt}
EARLIER=${EARLIER:-839 1319 1799 2039 2159}
GAMES=${GAMES:-analysis/search-leads}
RUNS=${RUNS:-runs/self-play}
OUT=${OUT:-analysis/self-play}
SUITE=${SUITE:-eval_suites/ygo-policies-blackwing-screen-and-checkpoint.json}
AGAINST=${AGAINST:-eval_suites/blackwing-against-checkpoint.json}
UNSEEN=${UNSEEN:-eval_suites/ygo-policies-blackwing-unseen.json}
PLAYS=${PLAYS:-}
SINCE=${SINCE:-}
ARM_SEED=${ARM_SEED:-}
LOGS=$RUNS
# shellcheck source=experiments/commands/at-once.sh
source "$(dirname "${BASH_SOURCE[0]}")/at-once.sh"

for part in $PARTS; do
  if [ "$part" != base ] && [ "$part" != arms ] && [ "$part" != exploiters ]; then
    echo "PARTS names base, arms and exploiters, not $part" >&2
    exit 1
  fi
done
pinned=$(git submodule status --recursive)
if grep -q '^[^ ]' <<< "$pinned"; then
  echo "A submodule is not at its pinned revision: git submodule update --init --recursive" >&2
  exit 1
fi
# The updates the start has had: a run from it counts its versions on.
started=$(basename "$START" .pt)
STARTED=$((10#${started#model-} + 1))
needed=("$START" "$SUITE" "$AGAINST")
earlier=()
if [[ " $PARTS " == *" base "* ]]; then
  for version in $EARLIER; do
    earlier+=("$(printf '%s/model-%05d.pt' "$(dirname "$START")" "$((10#$version))")")
  done
  needed+=(${earlier[@]+"${earlier[@]}"})
  if ((STARTED % EVERY || EXPLOIT % EVERY)); then
    echo "EVERY must divide the start's $STARTED updates and the exploiters' $EXPLOIT, for the suite to be played after EVERY more" >&2
    exit 1
  fi
fi
if [[ " $PARTS " == *" arms "* ]]; then
  needed+=("$UNSEEN" "$GAMES/heldout.jsonl" "$GAMES/policies-revision.txt")
  if ((STARTED % STEP || ARM % STEP || ARM < 3 * STEP)); then
    echo "STEP must divide the start's $STARTED updates and the arms' $ARM, and an arm is read by its last three checkpoints" >&2
    exit 1
  fi
  if ((ITSELF < 1 || ITSELF >= DUELS)); then
    echo "ITSELF is the duels of an update against itself, some and not all of its $DUELS" >&2
    exit 1
  fi
  if [ -n "$SINCE" ] && ((SINCE < STEP || SINCE % STEP || SINCE > ARM - 3 * STEP)); then
    echo "SINCE is an earlier checkpoint of with-itself: a multiple of STEP, before the last three of its $ARM updates" >&2
    exit 1
  fi
  # An arm that is there is taken on or measured as it is: one of another
  # training seed is not the arm asked for.
  for name in with-itself pilots-alone; do
    if [ -f "$RUNS/$name/run.json" ] &&
      ! grep -q "\"training_seed\": ${ARM_SEED:-2012}," "$RUNS/$name/run.json"; then
      echo "$RUNS/$name is an arm of another training seed than ${ARM_SEED:-2012}: a seed's arms have folders of their own (RUNS, OUT)" >&2
      exit 1
    fi
  done
fi
if [[ " $PARTS " == *" exploiters "* ]]; then
  if ((STARTED % EVERY || (STARTED + ARM) % EVERY || EXPLOIT % EVERY || EXPLOIT < 3 * EVERY)); then
    echo "EVERY must divide the $STARTED updates of the start, the $((STARTED + ARM)) of an arm's last checkpoint and the exploiters' $EXPLOIT, and an exploiter is read by its last three checkpoints" >&2
    exit 1
  fi
fi
for file in "${needed[@]}"; do
  if [ ! -f "$file" ]; then
    echo "Missing $file" >&2
    exit 1
  fi
done
if [[ " $PARTS " == *" arms "* ]]; then
  # A recorded duel plays again only on the pilots it was recorded with.
  revision=$(git -C native/ygo-policies rev-parse HEAD)
  if [ "$(cat "$GAMES/policies-revision.txt")" != "$revision" ]; then
    echo "The games of $GAMES are of ygo-policies $(cat "$GAMES/policies-revision.txt"), not $revision" >&2
    exit 1
  fi
fi
cargo build --release
mkdir -p "$OUT/head-to-head" "$OUT/heldout"

# An arm after so many updates of its own.
checkpoint() { printf '%s/checkpoints/model-%05d.pt' "$RUNS/$1" $((STARTED + $2 - 1)); }

# home NAME: the folder of the run NAME. An exploiter of an arm's last
# checkpoint has a folder for each length of the arm: those of the arms'
# first 960 updates keep the folders they were made in, and those of a
# longer arm are beside them.
home() {
  if [[ "$1" == exploiter-*-* ]] && ((ARM != 960)); then
    printf '%s/%s.%s' "$RUNS" "$1" "$ARM"
  else
    printf '%s/%s' "$RUNS" "$1"
  fi
}

# later NAME FROM UPDATES: the run NAME, which began as the checkpoint FROM,
# after so many updates of its own.
later() {
  local began
  began=$(basename "$2" .pt)
  printf '%s/checkpoints/model-%05d.pt' "$(home "$1")" $((10#${began#model-} + $3))
}

# against FILE HELD OPTION...: the mirror against the checkpoint HELD in the
# other seat, by the checkpoints given or by the deck's pilot.
against() {
  local file=$1 held=$2
  shift 2
  if [ ! -f "$OUT/head-to-head/$file.json" ]; then
    uv run python -m ygo_training evaluate-suite "$@" --device "$DEVICE" \
      --workers "$WORKERS" --pool-threads "$THREADS" --pool-slots "$SLOTS" --seed 2012 \
      --suite "$AGAINST" --opponent-checkpoint "$held" \
      --output "$OUT/head-to-head/$file.json.part" > /dev/null
    mv "$OUT/head-to-head/$file.json.part" "$OUT/head-to-head/$file.json"
  fi
}

# pair FILE ONE OTHER: two checkpoints on the mirror's deals, each in turn
# the one held, so that every deal is played from either seat: ONE against
# OTHER held in FILE.json, OTHER against ONE held in FILE.back.json.
pair() {
  against "$1" "$3" "$2"
  against "$1.back" "$2" "$3"
}

# train NAME FROM UPDATES KEPT OPTION...: the run NAME from the checkpoint
# FROM, or from where it stands, to UPDATES updates of its own, with a
# checkpoint and the suite every KEPT of them; its figures beside the
# measurements.
train() {
  local name=$1 from=$2 updates=$3 kept=$4 began last latest folder
  shift 4
  folder=$(home "$name")
  began=$(basename "$from" .pt)
  last=$(printf '%s/checkpoints/model-%05d.pt' "$folder" $((10#${began#model-} + updates)))
  local arguments=(
    --device "$DEVICE" --workers "$WORKERS" --pool-threads "$THREADS" --pool-slots "$SLOTS"
    --run "$folder" --learner blackwing --opponents "$OPPONENTS" --duels "$DUELS"
    --eval-every "$kept" --eval-suite "$SUITE" --keep-checkpoints "$kept" --seed 2012
    --ppo-epochs 1 "$@"
  )
  if ! ls "$folder"/checkpoints/model-*.pt > /dev/null 2>&1; then
    uv run python -m ygo_training train "${arguments[@]}" --resume "$from" \
      --resume-learning-rate "$RATE" --updates "$updates"
  elif [ ! -f "$last" ]; then
    latest=$(basename "$(ls "$folder"/checkpoints/model-*.pt | sort | tail -1)" .pt)
    uv run python -m ygo_training train "${arguments[@]}" --continue \
      --updates $((10#${began#model-} + updates - 10#${latest#model-}))
  fi
  cp "$folder/metrics.jsonl" "$OUT/$name.metrics.jsonl"
  cp "$folder/run.json" "$OUT/$name.run.json"
}

# exploiters LABEL VICTIM: an exploiter of the checkpoint VICTIM for each
# training seed, exploiter-LABELSEED: a learner that starts as VICTIM and
# plays every duel of its updates against VICTIM held as it is, which the
# suite's case against a checkpoint is played against too. They are over
# after all_done.
exploiters() {
  local seed
  for seed in $SEEDS; do
    at_once "$(basename "$(home "exploiter-$1$seed")")" train "exploiter-$1$seed" "$2" \
      "$EXPLOIT" "$EVERY" --training-seed "$seed" --opponent-checkpoint "$2" \
      --opponent-checkpoint-duels "$DUELS"
  done
}

# measure FILE CHECKPOINT...: those checkpoints on the held-out logged games.
# The deals a checkpoint plays itself go through the pool, which plays them
# as the workers do; a recorded duel it watches stays with the workers.
measure() {
  local file=$1
  shift
  if [ ! -f "$OUT/heldout/$file.json" ]; then
    uv run python -m ygo_training evaluate-search "$@" --device "$DEVICE" \
      --workers "$WORKERS" --pool-threads "$THREADS" --pool-slots "$SLOTS" \
      --records "$GAMES/heldout.jsonl" \
      --output "$OUT/heldout/$file.json.part" | tee -a "$OUT/heldout/tables.txt"
    mv "$OUT/heldout/$file.json.part" "$OUT/heldout/$file.json"
  fi
}

if [[ " $PARTS " == *" base "* ]]; then
  # 1. Head to head against the start.
  against start "$START" ${earlier[@]+"${earlier[@]}"} "$START"
  against pilot "$START" --pilot
  # 2. The exploiters of the start.
  exploiters "" "$START"
  all_done
fi

if [[ " $PARTS " == *" arms "* ]]; then
  # The arms: the start is held for the suite alone, in both.
  seeded=()
  [ -z "$ARM_SEED" ] || seeded=(--training-seed "$ARM_SEED")
  at_once with-itself train with-itself "$START" "$ARM" "$STEP" \
    --opponent-checkpoint "$START" --self-play-duels "$ITSELF" ${seeded[@]+"${seeded[@]}"}
  at_once pilots-alone train pilots-alone "$START" "$ARM" "$STEP" \
    --opponent-checkpoint "$START" ${seeded[@]+"${seeded[@]}"}
  all_done
  # 1. The held-out games: a file a checkpoint.
  measure start "$START"
  for name in with-itself pilots-alone; do
    for ((point = STEP; point <= ARM; point += STEP)); do
      measure "$name.$(printf '%04d' "$point")" "$(checkpoint "$name" "$point")"
    done
  done
  # 2. Head to head: every deal from either seat.
  mine=()
  theirs=()
  for ((point = ARM - 2 * STEP; point <= ARM; point += STEP)); do
    mine+=("$(checkpoint with-itself "$point")")
    theirs+=("$(checkpoint pilots-alone "$point")")
  done
  against with-itself "$(checkpoint pilots-alone "$ARM")" "${mine[@]}"
  for ((point = ARM - 2 * STEP; point <= ARM; point += STEP)); do
    against "pilots-alone.$(printf '%04d' "$point")" "$(checkpoint with-itself "$point")" \
      "$(checkpoint pilots-alone "$ARM")"
  done
  if [ -n "$SINCE" ]; then
    # With itself and its own earlier checkpoint, every deal from either seat.
    for ((point = ARM - 2 * STEP; point <= ARM; point += STEP)); do
      pair "with-itself.$(printf '%04d' "$point").since" "$(checkpoint with-itself "$point")" \
        "$(checkpoint with-itself "$SINCE")"
    done
  fi
  # 3. The decks outside training.
  if [ ! -f "$OUT/unseen.json" ]; then
    uv run python -m ygo_training evaluate-suite "$START" "${mine[@]}" "${theirs[@]}" \
      --device "$DEVICE" --workers "$WORKERS" --pool-threads "$THREADS" --pool-slots "$SLOTS" \
      --seed 2012 --suite "$UNSEEN" --output "$OUT/unseen.json.part" > /dev/null
    mv "$OUT/unseen.json.part" "$OUT/unseen.json"
  fi
  # 4. The named plays.
  if [ ! -f "$OUT/named-plays.json" ]; then
    # shellcheck disable=SC2086
    uv run python tools/named_plays.py "start=$START" \
      "with-itself=$(checkpoint with-itself "$ARM")" \
      "pilots-alone=$(checkpoint pilots-alone "$ARM")" $PLAYS \
      --output "$OUT/named-plays.json.part" | tee "$OUT/named-plays.txt"
    mv "$OUT/named-plays.json.part" "$OUT/named-plays.json"
  fi
fi

if [[ " $PARTS " == *" exploiters "* ]]; then
  for name in with-itself pilots-alone; do
    if [ ! -f "$(checkpoint "$name" "$ARM")" ]; then
      echo "Missing $(checkpoint "$name" "$ARM"): the arms come first (PARTS=arms)" >&2
      exit 1
    fi
  done
  # 1. The exploiters: of the start as the base makes them, and of each
  #    arm's last checkpoint.
  exploiters "" "$START"
  for name in with-itself pilots-alone; do
    exploiters "$name-" "$(checkpoint "$name" "$ARM")"
  done
  all_done
  # 2. An exploiter and its victim, every deal from either seat.
  for label in "" with-itself- pilots-alone-; do
    victim=$START
    if [ -n "$label" ]; then
      victim=$(checkpoint "${label%-}" "$ARM")
    fi
    for seed in $SEEDS; do
      for ((point = EXPLOIT - 2 * EVERY; point <= EXPLOIT; point += EVERY)); do
        pair "exploiter-$label$seed.$(printf '%04d' "$point")" \
          "$(later "exploiter-$label$seed" "$victim" "$point")" "$victim"
      done
    done
  done
  # 3. What the start's exploiters found, against each arm's last checkpoint.
  for name in with-itself pilots-alone; do
    pair "start.at-$name" "$START" "$(checkpoint "$name" "$ARM")"
    for seed in $SEEDS; do
      pair "exploiter-$seed.at-$name" "$(later "exploiter-$seed" "$START" "$EXPLOIT")" \
        "$(checkpoint "$name" "$ARM")"
    done
  done
fi

# What is there says.
uv run python tools/self_play_report.py "$OUT" --arm "$ARM" --step "$STEP" \
  --exploit "$EXPLOIT" --every "$EVERY" ${SINCE:+--since "$SINCE"} \
  --output "$OUT/report.json" | tee "$OUT/report.txt"
echo "The runs' figures, the measurements and what they say are in $OUT: copy it back."
