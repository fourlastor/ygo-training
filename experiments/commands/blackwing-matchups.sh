#!/usr/bin/env bash
# Blackwing against another deck, head to head (experiments/2026-10-14-more-
# decks.md, "Blackwing against another deck, head to head"). Run from the
# repository root. For each deck of OTHERS, Blackwing against it on the 2,048
# deals of eval_suites/blackwing-against-DECK-checkpoint.json: each network of
# PLAYERS playing Blackwing, on the same deals, against each of HELD playing
# the other deck in the other seat, into OUT/HELD/DECK.json; `pilots` in HELD,
# against that deck's pilot on the 2,048 of blackwing-against-DECK-pilot.json
# instead. Then the report (tools/matchups_report.py). Every network is the
# last checkpoint of its run of more decks, folded for its pairing by
# tools/fold_deck_lists.py into runs/more-decks/NETWORK/OWN-against-OPPONENT.
# A finished measurement is not made again. On the development machine's
# processor some 45 s a network and deck against a network, 25 s against a
# pilot: half an hour as it stands.
#   bash experiments/commands/blackwing-matchups.sh
set -euo pipefail
OTHERS=${OTHERS:-lightsworn gishki burn crystal}
PLAYERS=${PLAYERS:-decks-01 decks-05 decks-08 decks-13}
HELD=${HELD:-decks-05 decks-13 pilots}
DEVICE=${DEVICE:-cpu}
WORKERS=${WORKERS:-24}
THREADS=${THREADS:-20}
SLOTS=${SLOTS:-16}
SEED=${SEED:-2012}
OUT=${OUT:-analysis/matchups}
RUNS=${RUNS:-runs/more-decks}
# The interpreter of a machine that has no uv: the benchmark's setup leaves one.
PYTHON=${PYTHON:-}
# Evaluations on a processor run six times slower with every thread.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

# last NETWORK: the name of its run's last checkpoint (1,920 updates of PPO
# after a copy of 240 a deck).
last() {
  local decks=${1#decks-}
  printf 'model-%05d' $((10#$decks * 240 + 1920 - 1))
}
for other in $OTHERS; do
  for network in $PLAYERS; do
    path="$RUNS/$network/blackwing-against-$other/$(last "$network").pt"
    [ -f "$path" ] || { echo "Missing $path (tools/fold_deck_lists.py)" >&2; exit 1; }
  done
  for network in $HELD; do
    [ "$network" != pilots ] || continue
    path="$RUNS/$network/$other-against-blackwing/$(last "$network").pt"
    [ -f "$path" ] || { echo "Missing $path (tools/fold_deck_lists.py)" >&2; exit 1; }
  done
done
pinned=$(git submodule status --recursive)
if grep -q '^[^ ]' <<< "$pinned"; then
  echo "A submodule is not at its pinned revision: git submodule update --init --recursive" >&2
  exit 1
fi
python=(uv run python)
[ -z "$PYTHON" ] || python=("$PYTHON")
cargo build --release

for held in $HELD; do
  mkdir -p "$OUT/$held"
  for other in $OTHERS; do
    file="$OUT/$held/$other.json"
    [ ! -f "$file" ] || continue
    players=()
    for network in $PLAYERS; do
      players+=("$RUNS/$network/blackwing-against-$other/$(last "$network").pt")
    done
    if [ "$held" = pilots ]; then
      against=(--suite "eval_suites/blackwing-against-$other-pilot.json")
    else
      against=(--suite "eval_suites/blackwing-against-$other-checkpoint.json"
        --opponent-checkpoint "$RUNS/$held/$other-against-blackwing/$(last "$held").pt")
    fi
    "${python[@]}" -m ygo_training evaluate-suite "${players[@]}" --device "$DEVICE" \
      --workers "$WORKERS" --pool-threads "$THREADS" --pool-slots "$SLOTS" --seed "$SEED" \
      "${against[@]}" --output "$file.part" > /dev/null
    mv "$file.part" "$file"
    echo "$(date +%H:%M) Blackwing against $held playing $other"
  done
done
"${python[@]}" tools/matchups_report.py "$OUT" --output "$OUT/report.json" | tee "$OUT/report.txt"
