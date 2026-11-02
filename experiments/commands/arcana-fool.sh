#!/usr/bin/env bash
# The Fool's habit in `arcana`'s games, as
# experiments/2026-10-28-seventy-one-champions.md fixes it: PPO's last mean at
# the deck from both seats, as it plays and with the habit held out, on the same
# deals. Run from the repository root.
#
# Three arms: plain; held, the network made to summon or set a monster wherever
# its pilot would while The Fool stands in Defense on its side; no-set, the
# network answering otherwise where it would Set The Fool. Each against three
# sets of opponents, into OUT/ARM-SET.jsonl: mirror, its champion, MIRROR_DEALS
# deals; decks, `all71` at each of the seventy other decks, PAIRING_DEALS deals;
# pilots, each of the thirteen decks' pilots, PILOT_DEALS deals. Parts that are
# there are not played again.
#
# Read with tools/deck_replays.py: `wall` on each plain file, `paired` of an arm's
# file and the plain file of the same set.
#   AT_ONCE=18 bash experiments/commands/arcana-fool.sh
set -euo pipefail
NETWORK=${NETWORK:-runs/seventy-one-champions/ppo-mean/21900.pt}
CHAMPION=${CHAMPION:-runs/champions/arcana/checkpoints/model-00959.pt}
ALL71=${ALL71:-runs/seventy-one-decks/all71/checkpoints/model-27599.pt}
OUT=${OUT:-analysis/seventy-one-champions/fool}
LIST=${LIST:-experiments/commands/champion-recordings.list}
THIRTEEN=${THIRTEEN:-blackwing,lightsworn,gishki,burn,crystal,dragunity,gladiator,heroes,infernity,monarch,morphtronic,rock-block,spellcaster}
FOOL=62892347
MIRROR_DEALS=${MIRROR_DEALS:-2048}
MIRROR_PARTS=${MIRROR_PARTS:-16}
PAIRING_DEALS=${PAIRING_DEALS:-32}
PAIRING_PART=${PAIRING_PART:-14}
PILOT_DEALS=${PILOT_DEALS:-64}
PYTHON=${PYTHON:-}
DEVICE=cpu
LOGS=$OUT/logs
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
# shellcheck source=experiments/commands/at-once.sh
source experiments/commands/at-once.sh

for path in "$NETWORK" "$CHAMPION" "$ALL71"; do
  if [ ! -f "$path" ]; then
    echo "Missing $path" >&2
    exit 1
  fi
done
python=(uv run python)
[ -z "$PYTHON" ] || python=("$PYTHON")
cargo build --release
mkdir -p "$OUT"
mapfile -t others < <(awk '!/^#/ && NF {print $1}' "$LIST" | grep -vx arcana)
arms=(plain held no-set)

# played ARM NAME OPTIONS...: a part of an arm's games, into OUT/ARM/NAME.jsonl.
played() {
  local arm=$1 name=$2
  shift 2
  local file=$OUT/$arm/$name.jsonl
  [ ! -s "$file" ] || return 0
  local holds=()
  case $arm in
    held) holds=(--plays-on) ;;
    no-set) holds=(--never-set) ;;
  esac
  mkdir -p "$OUT/$arm"
  at_once "$arm-$name" "${python[@]}" tools/deck_replays.py play "$NETWORK" --deck arcana \
    --wall "$FOOL" --brief "${holds[@]}" "$@" --output "$file"
}

for arm in "${arms[@]}"; do
  each=$((MIRROR_DEALS / MIRROR_PARTS))
  for ((part = 0; part < MIRROR_PARTS; part++)); do
    played "$arm" "mirror-$part" --other "$CHAMPION" --deals "$each" \
      --first-deal $((part * each))
  done
  played "$arm" pilots-0 --pilots --against "$THIRTEEN" --deals "$PILOT_DEALS"
  for ((first = 0; first < ${#others[@]}; first += PAIRING_PART)); do
    some=$(IFS=,; echo "${others[*]:first:PAIRING_PART}")
    played "$arm" "decks-$first" --other "$ALL71" --against "$some" --deals "$PAIRING_DEALS"
  done
done
all_done
for arm in "${arms[@]}"; do
  for set in mirror decks pilots; do
    cat "$OUT/$arm/$set"-*.jsonl > "$OUT/$arm-$set.jsonl"
  done
done
for set in mirror decks pilots; do
  "${python[@]}" tools/deck_replays.py wall "$OUT/plain-$set.jsonl"
  for arm in held no-set; do
    "${python[@]}" tools/deck_replays.py paired "$OUT/$arm-$set.jsonl" "$OUT/plain-$set.jsonl"
  done
done | tee "$OUT/report.txt"
