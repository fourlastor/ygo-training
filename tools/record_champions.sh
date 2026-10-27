#!/usr/bin/env bash
# Champions' recordings one deck after another (tools/record_champion.py),
# on a machine beside the one that makes the champions. Each line of LIST
# names a deck and its champion's checkpoint on the CUDA machine (FROM):
#
#   burn runs/champions/burn/checkpoints/model-00959.pt
#
# A deck whose archive OUT/DECK.tar is there is done. Each look goes down
# the list and records every deck whose champion is there to be fetched into
# OUT/champions/DECK.pt; a champion still being made is passed over and tried
# on the next look, and so is a deck whose recording fails, its chunks
# already there kept. The list is read again at each look, so a line added
# while it goes is recorded in its turn. With WAIT, a look that recorded
# nothing is followed by another WAIT seconds later, until every deck listed
# is done or it is stopped.
#
# OPTIONS are given to each recording (`--mirrors 44 --against 4` and the
# rest of tools/record_champion.py). With TO, each deck's archive is sent
# there once it is made (`daniele@luna:ygo-training/records`, where a copy
# runs); with PULL as well, each look first copies the archives there into
# OUT, so a deck another machine recorded is done here too. With REVERSE the
# list is gone through from its end: two machines on one list meet in its
# middle.
#
#   WAIT=600 bash tools/record_champions.sh experiments/commands/champion-recordings.list
set -euo pipefail
LIST=${1:?the list of decks and their champions}
OUT=${OUT:-records/champions}
FROM=${FROM:-daniele@luna:ygo-training/ygo}
WAIT=${WAIT:-0}
OPTIONS=${OPTIONS:-}
TO=${TO:-}
PULL=${PULL:-}
REVERSE=${REVERSE:-}
# The interpreter of a machine that has no uv: the benchmark's setup leaves one.
PYTHON=${PYTHON:-}
python=(uv run python)
[ -z "$PYTHON" ] || python=("$PYTHON")
mkdir -p "$OUT/champions"

# fetched DECK CHECKPOINT: the deck's champion in OUT/champions, fetched
# from FROM where it is not here yet; fails where FROM has none.
fetched() {
  local kept=$OUT/champions/$1.pt
  if [ ! -f "$kept" ]; then
    rsync -q "$FROM/$2" "$kept.part" 2> /dev/null || return 1
    mv "$kept.part" "$kept"
  fi
}

while :; do
  left=0
  recorded=0
  if [ -n "$PULL" ] && [ -n "$TO" ]; then
    # A remote shell expands the archives' pattern there; a folder here, this
    # one. Only those not here are copied: one sent from here is its own.
    case $TO in
      *:*) rsync -q --ignore-existing "$TO/*.tar" "$OUT/" < /dev/null 2> /dev/null || true ;;
      *) rsync -q --ignore-existing "$TO"/*.tar "$OUT/" < /dev/null 2> /dev/null || true ;;
    esac
  fi
  looked=("$LIST")
  if [ -n "$REVERSE" ]; then
    reversed=$(mktemp)
    tac "$LIST" > "$reversed"
    looked=("$reversed")
  fi
  # The list on a descriptor of its own: what the recording runs, and the
  # remote shell of a fetch, read their own input and not the list.
  while read -r deck checkpoint _ <&3; do
    if [ -z "$deck" ] || [ "${deck:0:1}" = "#" ] || [ -f "$OUT/$deck.tar" ]; then
      continue
    fi
    left=$((left + 1))
    if ! fetched "$deck" "$checkpoint" < /dev/null; then
      continue
    fi
    echo "$(date '+%F %H:%M') $deck begins"
    if "${python[@]}" tools/record_champion.py "$deck" "$OUT/champions/$deck.pt" \
      --out "$OUT" $OPTIONS < /dev/null; then
      recorded=$((recorded + 1))
      left=$((left - 1))
      if [ -n "$TO" ]; then
        rsync -q "$OUT/$deck.tar" "$TO/" < /dev/null
        echo "$(date '+%F %H:%M') $deck.tar sent to $TO"
      fi
    else
      # A champion fetched while it was being written is fetched again.
      echo "$(date '+%F %H:%M') $deck FAILED: tried again on the next look" >&2
      rm -f "$OUT/champions/$deck.pt"
    fi
  done 3< "${looked[0]}"
  [ -z "$REVERSE" ] || rm -f "${looked[0]}"
  if ((left == 0)); then
    echo "$(date '+%F %H:%M') every deck of $LIST is recorded"
    break
  fi
  if ((!recorded)); then
    ((WAIT > 0)) || break
    sleep "$WAIT"
  fi
done
