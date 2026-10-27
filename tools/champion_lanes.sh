#!/usr/bin/env bash
# Champions of the decks of LIST, LANES of them at a time, on a machine beside
# the CUDA machine that makes the others: experiments/commands/champions.sh
# for each, as the queue runs it there (RUNNING=DECK, RUNS=runs/champions,
# OUT=analysis/champions/DECK), from the checkout CHECKOUT at the commit the
# queue's champions are made at. Each line of LIST names a deck:
#
#   zombie
#   x-saber
#
# A lane takes the first deck of LIST that no lane has taken (the folder
# TAKEN/DECK, which only one lane can make) and makes its champion, then the
# next. The list is read again for each deck, so a line added while it goes is
# made in its turn, and a deck taken out before its turn is not. A champion
# made is marked TAKEN/DECK/done; a deck whose script fails stays taken and is
# not tried again, its log in LOGS/champion-DECK.log. A run that was stopped is
# taken again by removing its folder: the script goes on from its last
# checkpoint.
#
# The script's own variables go through as they are. On a machine of more
# threads than a lane should have, THREADS and FAST_SLOTS share them out: the
# pool must hold the IN_FLIGHT duels of the PPO arm, THREADS x FAST_SLOTS of
# them, 1,040 by default.
#
#   THREADS=10 FAST_SLOTS=104 OMP_NUM_THREADS=2 LANES=2 \
#     bash tools/champion_lanes.sh decks.txt
set -uo pipefail
LIST=${1:?the decks, one a line, in the order they are made}
LANES=${LANES:-2}
CHECKOUT=${CHECKOUT:-$(cd "$(dirname "$0")/.." && pwd)}
LOGS=${LOGS:-$CHECKOUT/../logs}
TAKEN=${TAKEN:-$CHECKOUT/../taken}
mkdir -p "$LOGS" "$TAKEN"
LIST=$(realpath "$LIST")

# taken: the next deck of LIST that no lane has taken, now taken by this one;
# nothing where every deck is.
taken() {
  local deck _
  while read -r deck _; do
    if [ -z "$deck" ] || [ "${deck:0:1}" = "#" ]; then
      continue
    fi
    if mkdir "$TAKEN/$deck" 2> /dev/null; then
      echo "$deck"
      return
    fi
  done < "$LIST"
}

# lane N: champions one after another, until no deck is left.
lane() {
  local deck code
  while deck=$(taken) && [ -n "$deck" ]; do
    echo "$(date '+%F %T') lane $1: $deck begins"
    (
      cd "$CHECKOUT" || exit 1
      echo "$(date '+%F %T') $deck at $(git rev-parse --short HEAD), lane $1"
      RUNNING=$deck RUNS=runs/champions OUT=analysis/champions/$deck \
        bash experiments/commands/champions.sh
    ) >> "$LOGS/champion-$deck.log" 2>&1 < /dev/null
    code=$?
    if ((code == 0)); then
      touch "$TAKEN/$deck/done"
      echo "$(date '+%F %T') lane $1: $deck made"
    else
      echo "$(date '+%F %T') lane $1: $deck FAILED ($code), its log in $LOGS/champion-$deck.log" >&2
    fi
  done
  echo "$(date '+%F %T') lane $1: no deck left"
}

for ((number = 1; number <= LANES; number++)); do
  lane "$number" &
  # The lanes begin apart, so that their first updates do not fall together.
  sleep 30
done
wait
