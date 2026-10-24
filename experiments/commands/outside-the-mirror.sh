#!/usr/bin/env bash
# A single-deck champion outside its mirror (experiments/2026-10-24-two-decks.md,
# "Follow-up: a single-deck champion outside its mirror"): Blackwing alone and
# cyber-old in Blackwing against Cyber, beside the network of both that played
# the pairing against itself, and each champion against all71 playing each deck
# outside the thirteen. Run from the repository root once
# experiments/commands/two-decks.sh has measured the network of both (BESIDE),
# on any device: a processor plays the card's games.
#
# 1. The check: the network of both's games of Blackwing against Cyber after
#    1,920 updates, played again here, are BESIDE's, game for game.
# 2. Into OUT/cross: Blackwing alone playing Blackwing against all71 (NETWORK)
#    playing Cyber, and all71 playing Blackwing against cyber-old playing
#    Cyber, on the deals of the network of both's (CROSS), each champion by its
#    checkpoints after the last three STEP of UPDATES updates of PPO.
# 3. Into OUT/panel: each champion playing its deck against all71 playing each
#    deck outside the thirteen (BLACKWING_PANEL, CYBER_PANEL), and all71
#    playing the champion's deck in the same games.
# 4. The report (tools/outside_the_mirror_report.py).
#
# About 15 minutes on the development machine's processor.
#   bash experiments/commands/outside-the-mirror.sh
set -euo pipefail
DEVICE=${DEVICE:-cpu}
WORKERS=${WORKERS:-24}
THREADS=${THREADS:-20}
SLOTS=${SLOTS:-16}
SEED=${SEED:-2012}
COPY=${COPY:-240}
UPDATES=${UPDATES:-720}
STEP=${STEP:-120}
NETWORK=${NETWORK:-runs/seventy-one-decks/all71/checkpoints/model-27599.pt}
ALONE=${ALONE:-runs/two-decks/blackwing}
CYBER=${CYBER:-runs/cyber-champion/cyber-old}
BOTH=${BOTH:-runs/two-decks/both}
BESIDE=${BESIDE:-analysis/two-decks}
# cyber-old's own games against all71 in the Cyber mirror.
CYBER_ALONE=${CYBER_ALONE:-analysis/cyber-champion/cyber-old/mirror}
OUT=${OUT:-analysis/outside-the-mirror}
CROSS=${CROSS:-eval_suites/blackwing-against-cyber-checkpoint.json}
BLACKWING_PANEL=${BLACKWING_PANEL:-eval_suites/blackwing-against-others-checkpoint.json}
CYBER_PANEL=${CYBER_PANEL:-eval_suites/cyber-against-others-checkpoint.json}
PYTHON=${PYTHON:-}
# The network of both's decks: its mirrors had as many games after KINDS
# squared times a champion's updates.
KINDS=2

python=(uv run python)
[ -z "$PYTHON" ] || python=("$PYTHON")
# checkpoint RUN FIRST POINT: the checkpoint of the run at RUN after POINT
# updates past the FIRST it began with.
checkpoint() {
  printf '%s/checkpoints/model-%05d.pt' "$1" $(($2 + $3 - 1))
}
points=()
for ((point = UPDATES - 2 * STEP; point <= UPDATES; point += STEP)); do
  points+=("$point")
done
first=$((KINDS * KINDS * points[0]))
check=$(checkpoint "$BOTH" $((KINDS * COPY)) "$first")
for path in "$NETWORK" "$CROSS" "$BLACKWING_PANEL" "$CYBER_PANEL" "$check" \
  "$BESIDE/cross/$first.plays.json"; do
  if [ ! -f "$path" ]; then
    echo "Missing $path" >&2
    exit 1
  fi
done
for point in "${points[@]}"; do
  for path in "$(checkpoint "$ALONE" "$COPY" "$point")" "$(checkpoint "$CYBER" "$COPY" "$point")"; do
    if [ ! -f "$path" ]; then
      echo "Missing $path" >&2
      exit 1
    fi
  done
done
mkdir -p "$OUT/check" "$OUT/cross" "$OUT/panel"

# suite FILE SUITE OPTION...: what the options name on a suite, with how each
# game ended, into OUT/FILE.json.
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

suite "check/$first.plays" "$CROSS" "$check" --opponent-checkpoint "$NETWORK"
"${python[@]}" tools/outside_the_mirror_report.py check "$OUT/check/$first.plays.json" \
  "$BESIDE/cross/$first.plays.json"
for point in "${points[@]}"; do
  alone=$(checkpoint "$ALONE" "$COPY" "$point")
  cyber=$(checkpoint "$CYBER" "$COPY" "$point")
  suite "cross/blackwing-$point" "$CROSS" "$alone" --opponent-checkpoint "$NETWORK"
  suite "cross/cyber-$point" "$CROSS" "$NETWORK" --opponent-checkpoint "$cyber"
  suite "panel/blackwing-$point" "$BLACKWING_PANEL" "$alone" --opponent-checkpoint "$NETWORK"
  suite "panel/cyber-$point" "$CYBER_PANEL" "$cyber" --opponent-checkpoint "$NETWORK"
done
suite panel/blackwing-all71 "$BLACKWING_PANEL" "$NETWORK" --opponent-checkpoint "$NETWORK"
suite panel/cyber-all71 "$CYBER_PANEL" "$NETWORK" --opponent-checkpoint "$NETWORK"
beside=()
[ ! -d "$CYBER_ALONE" ] || beside+=(--cyber-alone "$CYBER_ALONE")
"${python[@]}" tools/outside_the_mirror_report.py report "$OUT" --both "$BESIDE" \
  --points "${points[@]}" --kinds "$KINDS" "${beside[@]}" --output "$OUT/report.json" \
  | tee "$OUT/report.txt"
echo "The measurements and what they say are in $OUT."
