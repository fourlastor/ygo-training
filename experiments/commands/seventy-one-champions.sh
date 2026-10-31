#!/usr/bin/env bash
# A network of the seventy-one decks
# (experiments/2026-10-28-seventy-one-champions.md): each deck's run of the
# champions' recipe copied into one network from its recorded games, then
# PPO with every duel against itself, half of them in the mirrors from its
# 5,761st update, a quarter before it, checked as it trains. Run from the
# repository root on the CUDA machine.
# PARTS names what is made, in its order:
#
# copy. taught: one network of the seventy-one decks from scratch, each
# deck's games its teacher's recorded games played again with its answers
# and imitated, COPY updates of DUELS duels, 160 a deck, ITSELF of an
# update's duels in the mirrors, both seats imitated, the others against
# the thirteen decks' pilots, at 3e-4, in minibatches of 256 with no duels
# in flight, every game seen once and its result at four of its positions;
# a checkpoint every COPY_STEP. The recordings are RECORDS/DECK, each made
# by tools/record_champion.py; an archive RECORDS/DECK.tar is unpacked
# beside itself. First the teachers are held to the record
# (tools/seventy_one_teachers.py): each deck recorded with its run of LIST,
# none of them under all71 in its mirror by the figure of its making
# (CHAMPIONS), RELEASE the decks with none there.
#
# copy-measure. Into OUT: the copy's last checkpoint head to head against
# each teacher in its mirror; it and its checkpoints after each of CURVE
# against all71 in each mirror; it and all71 against the pilots (PILOTS);
# the copy's metrics; then the report (tools/seventy_one_champions_report.py,
# the teachers of RELEASE told by their measurements in TEACHERS_MEASURED).
#
# copy-check. Into OUT/checks/copy: the copy's last checkpoint as PPO's
# checks measure it (below), so that it is known before PPO is begun; ppo
# begins by the same and finds it made.
#
# ppo. PPO on from the copy's last checkpoint for PPO updates at RATE, every
# duel of an update against itself and none against a pilot, half of them
# in the mirrors (--self-play-weights from RUNS/ppo.weights.json: each
# mirror MIRROR_WEIGHT, each other ordered pairing OTHER_WEIGHT, which was
# 3, a quarter of them, for its first 5,760 updates), with the fourteen
# decks' arms' settings; a checkpoint every STEP. It is trained in
# stretches to each of CHECKS, and after each the mean of the weights of its
# last MEAN_KEPT kept checkpoints (RUNS/ppo-mean/POINT.pt) is checked, the
# training paused, on the deals the copy was checked on before it, into
# OUT/checks: head to head against each teacher in its mirror on CHECK_DEALS
# deals from both seats, and against all71 on CHECK_PAIRINGS. Where the
# mirrors or the pairings are under the copy at two of these checks in a row
# (tools/seventy_one_checks.py), it stops, unless EARLY_STOP is 0. Its checks
# after 2,880 and 5,760 updates read its last checkpoint, and it stopped by
# them on 2026-10-31: it goes on from there, and they do not count.
#
# ppo-measure. Into OUT: the mean of PPO's last MEAN_KEPT kept checkpoints head
# to head against each teacher in its mirror; it and the copy against all71 on
# every ordered pairing (QUICK-1.json to QUICK-QUICK_PARTS.json, the deals of
# one suite in parts that go at once); it against the pilots; PPO's metrics;
# then the report.
#
# along. Into OUT/along, on checkpoints already kept and with no training, how
# the mirrors moved head to head against each teacher: each of the fourteen
# decks' ARMS after each of ARMS_ALONG, on the deals they were read on, and PPO
# after each of ALONG, on the checks' deals; then what they say, beside the
# fourteen decks' measurements in FOURTEEN_OUT and PPO's checks
# (tools/seventy_one_checks.py along).
#
# mean. The mean of the weights of PPO's checkpoints from MEAN_FROM to MEAN_TO
# updates, every STEP, into RUNS/ppo-mean (tools/mean_checkpoint.py), measured
# as a check into OUT/checks/mean, then what it says against PPO's check after
# MEAN_AGAINST (tools/seventy_one_checks.py mean).
#
# means. The same along PPO: for each of WINDOWS, the first and last of its
# updates, the mean into RUNS/ppo-mean/WINDOW.pt, measured as a check into
# OUT/checks/mean-WINDOW, then what they say (tools/seventy_one_checks.py means).
#
# AT_ONCE: how many of the measurements go at a time, through an MPS server
# on a card (experiments/commands/at-once.sh), what each writes in
# OUT/logs; 1, the default, one after the other.
#
# By reckoning on the CUDA machine: the copy 20 to 30 hours, its
# measurements 2 to 3 at seven at once; PPO some two days, a check some ten
# minutes at seven at once, its measurements about an hour.
#   PARTS=copy bash experiments/commands/seventy-one-champions.sh
#   PARTS=copy-measure AT_ONCE=7 bash experiments/commands/seventy-one-champions.sh
#   PARTS=copy-check AT_ONCE=7 bash experiments/commands/seventy-one-champions.sh
#   PARTS=ppo AT_ONCE=7 bash experiments/commands/seventy-one-champions.sh
#   PARTS=ppo-measure AT_ONCE=7 bash experiments/commands/seventy-one-champions.sh
#   PARTS=along AT_ONCE=7 bash experiments/commands/seventy-one-champions.sh
#   PARTS=mean AT_ONCE=7 bash experiments/commands/seventy-one-champions.sh
#   PARTS=means AT_ONCE=7 bash experiments/commands/seventy-one-champions.sh
set -euo pipefail
DEVICE=${DEVICE:-cuda}
WORKERS=${WORKERS:-24}
THREADS=${THREADS:-20}
SLOTS=${SLOTS:-16}
DUELS=${DUELS:-520}
PARTS=${PARTS:-copy copy-measure copy-check ppo ppo-measure}
# Each deck and its run's last checkpoint, which teaches it.
LIST=${LIST:-experiments/commands/champion-recordings.list}
RECORDS=${RECORDS:-../records}
CHAMPIONS=${CHAMPIONS:-analysis/champions}
RELEASE=${RELEASE:-blackwing,cyber,lightsworn,infernity}
TEACHERS_MEASURED=${TEACHERS_MEASURED:-analysis/fourteen-decks/teachers}
THIRTEEN=${THIRTEEN:-blackwing,lightsworn,gishki,burn,crystal,dragunity,gladiator,heroes,infernity,monarch,morphtronic,rock-block,spellcaster}
COPY=${COPY:-11360}
# Two updates a deck: each point of CURVE is a checkpoint kept.
COPY_STEP=${COPY_STEP:-142}
ITSELF=${ITSELF:-260}
SEED=${SEED:-2012}
COPY_SLOTS=${COPY_SLOTS:-26}
# PPO: the fourteen decks' 4,320 updates by the deck, half of its duels in
# the mirrors.
PPO=${PPO:-21900}
STEP=${STEP:-240}
RATE=${RATE:-1e-4}
MIRROR_WEIGHT=${MIRROR_WEIGHT:-70}
OTHER_WEIGHT=${OTHER_WEIGHT:-1}
# fast's settings, the fourteen decks' arms': two updates' duels in flight,
# the pool's slots to hold them, minibatches of 512, the calls as they are in
# two groups.
IN_FLIGHT=${IN_FLIGHT:-1040}
FAST_SLOTS=${FAST_SLOTS:-52}
MINIBATCH=${MINIBATCH:-512}
# PPO's checks: after so many of its updates, of the mean of so many of its
# kept checkpoints, on so many deals of each mirror and against all71 on these
# pairings.
CHECKS=${CHECKS:-8640 11520 14400 17280 20160}
MEAN_KEPT=${MEAN_KEPT:-13}
CHECK_DEALS=${CHECK_DEALS:-128}
CHECK_PAIRINGS=${CHECK_PAIRINGS:-eval_suites/seventy-one-decks-check-pairings-checkpoint.json}
EARLY_STOP=${EARLY_STOP:-1}
# The mirrors along the runs: the points of PPO and of the fourteen decks' arms,
# which began after FOURTEEN_COPY updates.
ALONG=${ALONG:-160 640 1360 2080}
ARMS=${ARMS:-quarter selfplay half}
ARMS_ALONG=${ARMS_ALONG:-160 640 1120 2080}
FOURTEEN=${FOURTEEN:-cyber,$THIRTEEN}
FOURTEEN_COPY=${FOURTEEN_COPY:-2240}
FOURTEEN_RUNS=${FOURTEEN_RUNS:-runs/fourteen-decks}
FOURTEEN_OUT=${FOURTEEN_OUT:-analysis/fourteen-decks}
# The mean of PPO's weights: from and to these of its updates, against its check
# after MEAN_AGAINST.
MEAN_FROM=${MEAN_FROM:-2800}
MEAN_TO=${MEAN_TO:-5680}
MEAN_AGAINST=${MEAN_AGAINST:-5760}
WINDOWS=${WINDOWS:-160-1360 1600-2800 3040-4240 4480-5680}
NETWORK=${NETWORK:-runs/seventy-one-decks/all71/checkpoints/model-27599.pt}
RUNS=${RUNS:-runs/seventy-one-champions}
OUT=${OUT:-analysis/seventy-one-champions}
SUITES=${SUITES:-eval_suites}
PILOTS=${PILOTS:-eval_suites/seventy-one-decks-against-pilots.json}
# Every ordered pairing against all71, in parts: QUICK-1.json and on.
QUICK=${QUICK:-eval_suites/seventy-one-decks-pairings-quick-checkpoint}
QUICK_PARTS=${QUICK_PARTS:-7}
# The copy's checkpoints measured in the mirrors: 40, 80, 120 and 160
# updates a deck.
CURVE=${CURVE:-2840 5680 8520 11360}
# The interpreter of a machine that has no uv: the benchmark's setup leaves one.
PYTHON=${PYTHON:-}
LOGS=${LOGS:-$OUT/logs}
# shellcheck source=experiments/commands/at-once.sh
source "$(dirname "${BASH_SOURCE[0]}")/at-once.sh"

for part in $PARTS; do
  case $part in
    copy | copy-measure | copy-check | ppo | ppo-measure | along | mean | means) ;;
    *)
      echo "PARTS names copy, copy-measure, copy-check, ppo, ppo-measure, along, mean and means, not $part" >&2
      exit 1
      ;;
  esac
done
if ((COPY % COPY_STEP || COPY < 3 * COPY_STEP)); then
  echo "COPY is a multiple of COPY_STEP, and three of them at least" >&2
  exit 1
fi
for point in $CURVE; do
  if ((point % COPY_STEP || point > COPY)); then
    echo "CURVE names checkpoints the copy keeps, not $point" >&2
    exit 1
  fi
done
previous=0
for point in $CHECKS; do
  if ((point <= previous || point >= PPO)); then
    echo "CHECKS names points of PPO in their order and before its end, not $point" >&2
    exit 1
  fi
  previous=$point
done
mapfile -t decks < <(grep -v '^#' "$LIST" | awk 'NF {print $1}')
DECKS=$(IFS=,; echo "${decks[*]}")
needed=("$NETWORK" "$PILOTS" "$CHECK_PAIRINGS")
for deck in "${decks[@]}"; do
  needed+=("$SUITES/$deck-against-checkpoint.json")
done
for ((part = 1; part <= QUICK_PARTS; part++)); do
  needed+=("$QUICK-$part.json")
done
for path in "${needed[@]}"; do
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
# teacher DECK: the run that teaches DECK.
teacher() {
  awk -v deck="$1" '$1 == deck {print $2}' "$LIST"
}
# unpacked: each deck's recording a folder of RECORDS, its archive unpacked
# beside itself where it is not yet, whole or not at all.
unpacked() {
  local deck into
  for deck in "${decks[@]}"; do
    [ ! -f "$RECORDS/$deck/about.json" ] || continue
    if [ ! -f "$RECORDS/$deck.tar" ]; then
      echo "No recording of $deck in $RECORDS" >&2
      exit 1
    fi
    into=$(mktemp -d "$RECORDS/.unpacking.XXXXXX")
    tar -xf "$RECORDS/$deck.tar" -C "$into"
    mv "$into/$deck" "$RECORDS/$deck"
    rmdir "$into"
  done
}
# failed FILE: whether evaluate-suite, which goes on past a case that fails,
# left a case's error in FILE where its games would be.
failed() {
  grep -q '"error_type"' "$1"
}
# played FILE SUITE OPTION...: what the options name on a suite, with how
# each game ended, into FILE.json, at once with others where AT_ONCE says.
played() {
  # A file being played beside others is not there yet: it is not played
  # twice. One that holds a failed case is played again.
  if { [ ! -f "$1.json" ] || failed "$1.json"; } && [ -z "${queued[$1]:-}" ]; then
    queued[$1]=1
    at_once "$(basename "$(dirname "$1")")-$(basename "$1")" playing "$@"
  fi
}
declare -A queued
# playing FILE SUITE OPTION...: what played does, then and there.
playing() {
  local file=$1 suite=$2
  shift 2
  mkdir -p "$(dirname "$file")"
  "${python[@]}" -m ygo_training evaluate-suite "$@" --device "$DEVICE" \
    --workers "$WORKERS" --pool-threads "$THREADS" --pool-slots "$SLOTS" --seed "$SEED" \
    --suite "$suite" --output "$file.json.part" > /dev/null
  if failed "$file.json.part"; then
    echo "A case of $file failed: $file.json.part" >&2
    return 1
  fi
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
# made NAME FIRST UPDATES KEPT OPTION...: the run NAME of the seventy-one
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
# weights FILE MIRROR OTHER: the weights of the duels against itself, each
# mirror MIRROR and each other ordered pairing of the decks OTHER, into FILE.
weights() {
  local file=$1 mirror=$2 other=$3 one two entries=()
  for one in "${decks[@]}"; do
    for two in "${decks[@]}"; do
      if [ "$one" = "$two" ]; then
        entries+=("\"$one>$two\": $mirror")
      else
        entries+=("\"$one>$two\": $other")
      fi
    done
  done
  (IFS=,; echo "{${entries[*]}}") > "$file"
}
# suited: each deck's mirror suite of the checks, CHECK_DEALS deals, into
# OUT/checks/suites.
suited() {
  "${python[@]}" tools/seventy_one_checks.py suites --decks "$DECKS" --suites "$SUITES" \
    --games "$CHECK_DEALS" --into "$OUT/checks/suites"
}
# checked NAME NETWORK: the network as PPO's checks measure it, into
# OUT/checks/NAME: against all71 on the checks' pairings, and head to head
# against each teacher in its mirror.
checked() {
  local name=$1 network=$2 deck
  played "$OUT/checks/$name/pairings" "$CHECK_PAIRINGS" "$network" --opponent-checkpoint "$NETWORK"
  for deck in "${decks[@]}"; do
    pair "$OUT/checks/$name/$deck" "$network" "$(teacher "$deck")" "$OUT/checks/suites/$deck.json"
  done
  all_done
}
# meaned NETWORK FROM TO: the mean of the weights of PPO's checkpoints from FROM
# to TO updates, every STEP, into NETWORK.
meaned() {
  local network=$1 point path kept=()
  for ((point = $2; point <= $3; point += STEP)); do
    kept+=("$(checkpoint "$RUNS/ppo" "$COPY" "$point")")
  done
  for path in "${kept[@]}"; do
    if [ ! -f "$path" ]; then
      echo "Missing $path: the mean is of checkpoints PPO kept" >&2
      exit 1
    fi
  done
  [ -f "$network" ] || "${python[@]}" tools/mean_checkpoint.py "${kept[@]}" --output "$network"
}
# lately POINT: the mean of the last MEAN_KEPT checkpoints PPO kept by POINT
# updates, which it keeps at multiples of STEP counted from its copy's start,
# into RUNS/ppo-mean/POINT.pt.
lately() {
  local last=$(($1 - (COPY + $1) % STEP))
  meaned "$RUNS/ppo-mean/$1.pt" $((last - (MEAN_KEPT - 1) * STEP)) "$last"
}
# heeded: what PPO's checks of a mean say so far, and PPO stopped where its last
# two meet the rule, unless EARLY_STOP is 0.
heeded() {
  local said=0
  "${python[@]}" tools/seventy_one_checks.py read "$OUT/checks" --decks "$DECKS" \
    --first "${CHECKS%% *}" --output "$OUT/checks/report.json" > "$OUT/checks/report.txt" ||
    said=$?
  cat "$OUT/checks/report.txt"
  if ((said == 3 && EARLY_STOP)); then
    echo "PPO stops by its checks: $OUT/checks/report.txt (EARLY_STOP=0 goes on)" >&2
    exit 1
  elif ((said && said != 3)); then
    exit "$said"
  fi
}
# reported: what the measurements in OUT say.
reported() {
  "${python[@]}" tools/seventy_one_champions_report.py "$OUT" --decks "$DECKS" \
    --champions "$CHAMPIONS" --teachers "$TEACHERS_MEASURED" --output "$OUT/report.json" |
    tee "$OUT/report.txt"
  echo "The measurements and what they say are in $OUT."
}

copied=$(checkpoint "$RUNS/taught" 0 "$COPY")
trained=$RUNS/ppo-mean/$PPO.pt
fast=(
  --pool-slots "$FAST_SLOTS" --rollout-estimates kept
  --duels-in-flight "$IN_FLIGHT" --minibatch "$MINIBATCH" --inference-graphs 0 --pool-groups 2
  --resume "$copied" --resume-learning-rate "$RATE"
)

if [[ " $PARTS " == *" copy "* ]]; then
  unpacked
  "${python[@]}" tools/seventy_one_teachers.py "$LIST" --records "$RECORDS" \
    --champions "$CHAMPIONS" --release "$RELEASE"
  echo "$(date +%H:%M) taught begins"
  made taught 0 "$COPY" "$COPY_STEP" --pool-slots "$COPY_SLOTS" \
    --imitation-only-updates 1000000 --imitation-value-weight 0.5 \
    --imitation-value-positions 4 --minibatch 256 --duels-in-flight 0 \
    --self-play-duels "$ITSELF" --self-play-pairs mirrors --copy-records "$RECORDS"
  echo "$(date +%H:%M) taught ends"
fi

if [[ " $PARTS " == *" copy-measure "* ]]; then
  for path in "$copied" $(for deck in "${decks[@]}"; do teacher "$deck"; done); do
    if [ ! -f "$path" ]; then
      echo "Missing $path: the copy is measured when it and its teachers are made" >&2
      exit 1
    fi
  done
  # The two longest first, so that the others go beside them.
  played "$OUT/pilots/taught" "$PILOTS" "$copied"
  played "$OUT/pilots/all71" "$PILOTS" "$NETWORK"
  for deck in "${decks[@]}"; do
    pair "$OUT/heads-taught/$deck" "$copied" "$(teacher "$deck")" \
      "$SUITES/$deck-against-checkpoint.json"
  done
  for deck in "${decks[@]}"; do
    for point in $CURVE; do
      pair "$OUT/taught-$deck/$point" "$(checkpoint "$RUNS/taught" 0 "$point")" "$NETWORK" \
        "$SUITES/$deck-against-checkpoint.json"
    done
  done
  all_done
  cp "$RUNS/taught/metrics.jsonl" "$OUT/taught.metrics.jsonl"
  cp "$RUNS/taught/run.json" "$OUT/taught.run.json"
  reported
fi

if [[ " $PARTS " == *" copy-check "* ]]; then
  for path in "$copied" $(for deck in "${decks[@]}"; do teacher "$deck"; done); do
    if [ ! -f "$path" ]; then
      echo "Missing $path: the copy is checked when it and its teachers are made" >&2
      exit 1
    fi
  done
  suited
  checked copy "$copied"
  heeded
fi

if [[ " $PARTS " == *" ppo "* ]]; then
  for path in "$copied" $(for deck in "${decks[@]}"; do teacher "$deck"; done); do
    if [ ! -f "$path" ]; then
      echo "Missing $path: PPO goes on from the copy and is checked against its teachers" >&2
      exit 1
    fi
  done
  weights "$RUNS/ppo.weights.json" "$MIRROR_WEIGHT" "$OTHER_WEIGHT"
  suited
  checked copy "$copied"
  for point in $CHECKS; do
    echo "$(date +%H:%M) ppo to $point"
    made ppo "$COPY" "$point" "$STEP" "${fast[@]}" --self-play-duels "$DUELS" \
      --self-play-weights "$RUNS/ppo.weights.json"
    lately "$point"
    checked "$point" "$RUNS/ppo-mean/$point.pt"
    heeded
  done
  echo "$(date +%H:%M) ppo to $PPO"
  made ppo "$COPY" "$PPO" "$STEP" "${fast[@]}" --self-play-duels "$DUELS" \
    --self-play-weights "$RUNS/ppo.weights.json"
  lately "$PPO"
  echo "$(date +%H:%M) ppo ends"
fi

if [[ " $PARTS " == *" ppo-measure "* ]]; then
  for path in "$trained" "$copied" $(for deck in "${decks[@]}"; do teacher "$deck"; done); do
    if [ ! -f "$path" ]; then
      echo "Missing $path: PPO is measured when it, the copy and the teachers are made" >&2
      exit 1
    fi
  done
  # The longest first, so that the others go beside them.
  for ((part = 1; part <= QUICK_PARTS; part++)); do
    played "$OUT/pairings/ppo-$part" "$QUICK-$part.json" "$trained" --opponent-checkpoint "$NETWORK"
    played "$OUT/pairings/taught-$part" "$QUICK-$part.json" "$copied" \
      --opponent-checkpoint "$NETWORK"
  done
  played "$OUT/pilots/ppo" "$PILOTS" "$trained"
  for deck in "${decks[@]}"; do
    pair "$OUT/heads-ppo/$deck" "$trained" "$(teacher "$deck")" \
      "$SUITES/$deck-against-checkpoint.json"
  done
  all_done
  cp "$RUNS/ppo/metrics.jsonl" "$OUT/ppo.metrics.jsonl"
  cp "$RUNS/ppo/run.json" "$OUT/ppo.run.json"
  reported
fi

if [[ " $PARTS " == *" along "* ]]; then
  suited
  for arm in $ARMS; do
    for point in $ARMS_ALONG; do
      for deck in ${FOURTEEN//,/ }; do
        pair "$OUT/along/$arm/$point/$deck" \
          "$(checkpoint "$FOURTEEN_RUNS/$arm" "$FOURTEEN_COPY" "$point")" "$(teacher "$deck")" \
          "$SUITES/$deck-against-checkpoint.json"
      done
    done
  done
  for point in $ALONG; do
    for deck in "${decks[@]}"; do
      pair "$OUT/along/ppo/$point/$deck" "$(checkpoint "$RUNS/ppo" "$COPY" "$point")" \
        "$(teacher "$deck")" "$OUT/checks/suites/$deck.json"
    done
  done
  all_done
  "${python[@]}" tools/seventy_one_checks.py along "$OUT" --decks "$DECKS" \
    --fourteen "$FOURTEEN_OUT" --fourteen-decks "$FOURTEEN" | tee "$OUT/along/report.txt"
fi

if [[ " $PARTS " == *" mean "* ]]; then
  suited
  network=$RUNS/ppo-mean/$(basename "$(checkpoint "$RUNS/ppo" "$COPY" "$MEAN_TO")")
  meaned "$network" "$MEAN_FROM" "$MEAN_TO"
  checked mean "$network"
  "${python[@]}" tools/seventy_one_checks.py mean "$OUT/checks" --decks "$DECKS" \
    --against "$MEAN_AGAINST" | tee "$OUT/checks/mean/report.txt"
fi

if [[ " $PARTS " == *" means "* ]]; then
  suited
  for window in $WINDOWS; do
    meaned "$RUNS/ppo-mean/$window.pt" "${window%-*}" "${window#*-}"
    checked "mean-$window" "$RUNS/ppo-mean/$window.pt"
  done
  "${python[@]}" tools/seventy_one_checks.py means "$OUT/checks" --decks "$DECKS" \
    --windows "${WINDOWS// /,}" | tee "$OUT/checks/means-report.txt"
fi
