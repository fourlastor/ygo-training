#!/usr/bin/env bash
# What a card the network never trained on costs at play (experiments/2026-11-02-expanded-pool.md,
# the card identities beside the batches): NETWORK with its identity rows no update moved
# rewritten (tools/identity_keys.py), blank, the mean of the trained rows, and drawn afresh on
# two seeds, against NETWORK as it is. Into OUT: as-is/, the network against itself; plays/ARM/,
# the arm at each of DECKS against the network as it is at each old deck; faces/ARM/, the network
# as it is at DECKS against the arm. Each DEALS deals from both seats from FIRST_DEAL. Runs on
# this machine's cores 8-23, two plays at a time; a play already there is kept.
#   bash experiments/commands/identity-keys.sh
set -euo pipefail
NETWORK=${NETWORK:-runs/expanded-pool/first-batch-mean/4320.pt}
ARMS=${ARMS:-runs/identity-keys}
OUT=${OUT:-analysis/identity-keys}
DECKS=${DECKS:-game-0064,$(printf "game-%04d," $(seq 66 76) | sed "s/,$//")}
EARLIER=${EARLIER:-$(printf "game-%04d," $(seq 1 12) | sed "s/,$//")}
LIST=${LIST:-experiments/commands/champion-recordings.list}
DEALS=${DEALS:-8}
FIRST_DEAL=${FIRST_DEAL:-60000}
THREADS=${THREADS:-8}
PYTHON=${PYTHON:-.venv/bin/python}
old=$( (grep -v '^#' "$LIST" | awk 'NF {print $1}'; tr , '\n' <<< "$EARLIER") | paste -sd,)
IFS=, read -r -a decks <<< "$DECKS"
mkdir -p "$ARMS" "$OUT"
for arm in blank mean redraw-1 redraw-2; do
  if [ ! -f "$ARMS/$arm.pt" ]; then
    keys=${arm%%-*}
    seed=${arm#redraw-}
    [ "$keys" = redraw ] || seed=0
    "$PYTHON" tools/identity_keys.py make "$NETWORK" --keys "$keys" --seed "$seed" --output "$ARMS/$arm.pt"
  fi
done
# play NETWORK OTHER FOLDER: NETWORK at each of DECKS against OTHER at each old deck.
play() {
  mkdir -p "$3"
  for deck in "${decks[@]}"; do
    [ ! -f "$3/$deck.jsonl" ] || continue
    OMP_NUM_THREADS=$THREADS taskset -c 8-23 nice -n 10 "$PYTHON" tools/deck_replays.py play "$1" \
      --deck "$deck" --against "$old" --other "$2" --deals "$DEALS" --first-deal "$FIRST_DEAL" \
      --brief --output "$3/$deck.jsonl.part" > /dev/null
    mv "$3/$deck.jsonl.part" "$3/$deck.jsonl"
  done
  echo "$(date +%H:%M) $3"
}
(
  play "$NETWORK" "$NETWORK" "$OUT/as-is"
  for arm in blank mean redraw-1 redraw-2; do play "$ARMS/$arm.pt" "$NETWORK" "$OUT/plays/$arm"; done
) &
(
  for arm in blank mean redraw-1 redraw-2; do play "$NETWORK" "$ARMS/$arm.pt" "$OUT/faces/$arm"; done
) &
wait
"$PYTHON" tools/identity_keys.py read "$OUT" | tee "$OUT/report.txt"
