#!/usr/bin/env bash
# Where what a batch learns of its new decks lives (experiments/2026-11-02-expanded-pool.md, the
# card identities, step 2): each of NETWORKS, NAME=CHECKPOINT, with the identity rows of the cards
# its run met first, those it moved where BEGUN, the network the run began from, had them
# untrained, put back to their untrained values and drawn afresh (tools/identity_keys.py make
# --begun), at the run's new decks (DECKS) against BEGUN at each old deck. Into OUT: start/, BEGUN
# against itself; NAME/as-is/, NAME/initial/ and NAME/redraw/, the network and its two arms. DEALS
# deals from both seats from FIRST_DEAL. Runs on this machine's cores 8-23, two plays at a time; a
# play already there is kept.
#   bash experiments/commands/identity-rows.sh
set -euo pipefail
BEGUN=${BEGUN:-runs/expanded-pool/first-batch-mean/4320.pt}
NETWORKS=${NETWORKS:-fast=runs/expanded-pool/batch-one-fast-mean/4020.pt,control=runs/expanded-pool/batch-one-mean/4020.pt}
ARMS=${ARMS:-runs/identity-rows}
OUT=${OUT:-analysis/identity-rows}
DECKS=${DECKS:-$(printf "game-%04d," $(seq 13 36) $(seq 38 63) | sed "s/,$//")}
EARLIER=${EARLIER:-$(printf "game-%04d," $(seq 1 12) | sed "s/,$//")}
LIST=${LIST:-experiments/commands/champion-recordings.list}
DEALS=${DEALS:-1}
FIRST_DEAL=${FIRST_DEAL:-70000}
THREADS=${THREADS:-8}
PYTHON=${PYTHON:-.venv/bin/python}
old=$( (grep -v '^#' "$LIST" | awk 'NF {print $1}'; tr , '\n' <<< "$EARLIER") | paste -sd,)
IFS=, read -r -a decks <<< "$DECKS"
IFS=, read -r -a networks <<< "$NETWORKS"
mkdir -p "$ARMS" "$OUT"
# What is played: a network, and the folder its games go into.
plays=("$BEGUN" "$OUT/start")
for named in "${networks[@]}"; do
  name=${named%%=*}
  for keys in initial redraw; do
    if [ ! -f "$ARMS/$name/$keys.pt" ]; then
      "$PYTHON" tools/identity_keys.py make "${named#*=}" --begun "$BEGUN" --keys "$keys" \
        --seed 1 --output "$ARMS/$name/$keys.pt"
    fi
  done
  plays+=("${named#*=}" "$OUT/$name/as-is")
  plays+=("$ARMS/$name/initial.pt" "$OUT/$name/initial" "$ARMS/$name/redraw.pt" "$OUT/$name/redraw")
done
# play NETWORK FOLDER: NETWORK at each of DECKS against BEGUN at each old deck.
play() {
  mkdir -p "$2"
  for deck in "${decks[@]}"; do
    [ ! -f "$2/$deck.jsonl" ] || continue
    OMP_NUM_THREADS=$THREADS taskset -c 8-23 nice -n 10 "$PYTHON" tools/deck_replays.py play "$1" \
      --deck "$deck" --against "$old" --other "$BEGUN" --deals "$DEALS" --first-deal "$FIRST_DEAL" \
      --brief --output "$2/$deck.jsonl.part" > /dev/null
    mv "$2/$deck.jsonl.part" "$2/$deck.jsonl"
  done
  echo "$(date +%H:%M) $2"
}
pids=()
for half in 0 1; do
  (
    for ((index = half; 2 * index < ${#plays[@]}; index += 2)); do
      play "${plays[2 * index]}" "${plays[2 * index + 1]}"
    done
  ) &
  pids+=($!)
done
for pid in "${pids[@]}"; do
  wait "$pid"
done
"$PYTHON" tools/identity_keys.py rows "$OUT" "$ARMS" | tee "$OUT/report.txt"
