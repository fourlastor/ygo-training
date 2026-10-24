#!/usr/bin/env bash
# The model without the deck lists (experiments/2026-10-15-without-deck-lists.md):
# copies that leave parts of their input out (`train --leave-out`) against
# the copies of experiments/commands/more-decks.sh of the same seed. An
# imitating run's games are its pilots', so each plays the very games its
# control did and differs from it only in what its model reads. Run from the
# repository root, on the machine the copies of more decks are on. PARTS
# names what is made:
#
# arms (the default). Two copies under RUNS, made as more-decks.sh makes
# its copies (COPY updates a deck at 3e-4, every game seen once with its
# result at four positions, DUELS duels an update against the pilots of
# OPPONENTS, seed SEED):
#      blackwing-own-list    Blackwing, reading its own list and not the
#                            other player's; its control is decks-01-copy;
#      three-no-lists        Blackwing, Lightsworn and Gishki, reading
#                            neither list; its control is decks-03-copy;
#      blackwing-no-history  Blackwing, reading both lists and not its own
#                            earlier answers; its control is decks-01-copy.
# AT_ONCE of them go at the same time (experiments/commands/at-once.sh). An
# interrupted copy goes on, and a finished one is skipped.
#
# measure. Each copy and its control (under CONTROLS) at its last
# checkpoint: on the held-out deals of GAMES, on the eighteen decks outside
# training (UNSEEN), and the copy of three decks and its control on the
# screen of each of its decks (decks-01-copy, the control of two, once); with each run's figures, by which the report
# checks that a copy and its control began as one network and played the
# same games. Into OUT, then the report (tools/deck_lists_report.py), which
# takes the readings. A finished measurement is not made again. About half
# an hour on the CUDA machine by reckoning.
#   PARTS=measure bash experiments/commands/without-deck-lists.sh
# Its copies leave out deck lists, which the input dropped in format 15:
# they are made and measured at 05d27141, the revision before.
set -euo pipefail
DEVICE=${DEVICE:-cuda}
WORKERS=${WORKERS:-24}
THREADS=${THREADS:-20}
SLOTS=${SLOTS:-16}
DUELS=${DUELS:-520}
PARTS=${PARTS:-arms}
COPY=${COPY:-240}
SEED=${SEED:-2012}
OPPONENTS=${OPPONENTS:-blackwing,burn,crystal,dragunity,gishki,gladiator,heroes,infernity,lightsworn,monarch,morphtronic,rock-block,spellcaster}
RUNS=${RUNS:-runs/without-deck-lists}
CONTROLS=${CONTROLS:-runs/more-decks}
OUT=${OUT:-analysis/without-deck-lists}
GAMES=${GAMES:-analysis/search-leads}
UNSEEN=${UNSEEN:-eval_suites/ygo-policies-blackwing-unseen.json}
# A deck's screen, the deck's name in place of %s: smaller suites try the
# measuring part at a small size.
SCREENS=${SCREENS:-eval_suites/ygo-policies-%s-screen.json}
# The interpreter of a machine that has no uv: the benchmark's setup leaves one.
PYTHON=${PYTHON:-}

for part in $PARTS; do
  if [ "$part" != arms ] && [ "$part" != measure ]; then
    echo "PARTS names arms and measure, not $part" >&2
    exit 1
  fi
done
pinned=$(git submodule status --recursive)
if grep -q '^[^ ]' <<< "$pinned"; then
  echo "A submodule is not at its pinned revision: git submodule update --init --recursive" >&2
  exit 1
fi
if [[ " $PARTS " == *" measure "* ]]; then
  for file in "$GAMES/heldout.jsonl" "$GAMES/policies-revision.txt"; do
    if [ ! -f "$file" ]; then
      echo "Missing $file" >&2
      exit 1
    fi
  done
  # A recorded duel plays again only on the pilots it was recorded with.
  revision=$(git -C native/ygo-policies rev-parse HEAD)
  if [ "$(cat "$GAMES/policies-revision.txt")" != "$revision" ]; then
    echo "The games of $GAMES are of ygo-policies $(cat "$GAMES/policies-revision.txt"), not $revision" >&2
    exit 1
  fi
fi
python=(uv run python)
[ -z "$PYTHON" ] || python=("$PYTHON")
cargo build --release
mkdir -p "$RUNS"
LOGS=$RUNS
# shellcheck source=experiments/commands/at-once.sh
source "$(dirname "${BASH_SOURCE[0]}")/at-once.sh"

# The copies: name, its decks, what its model leaves out, its control.
names=(blackwing-own-list three-no-lists blackwing-no-history)
declare -A learners=(
  [blackwing-own-list]=blackwing [three-no-lists]=blackwing,lightsworn,gishki
  [blackwing-no-history]=blackwing
)
declare -A leaves=(
  [blackwing-own-list]=opponent-deck [three-no-lists]=own-deck,opponent-deck
  [blackwing-no-history]=history
)
declare -A controls=(
  [blackwing-own-list]=decks-01-copy [three-no-lists]=decks-03-copy
  [blackwing-no-history]=decks-01-copy
)

# The last checkpoint of a copy of NAME's decks under FOLDER.
last() {
  local decks
  IFS=, read -r -a decks <<< "${learners[$1]}"
  printf '%s/checkpoints/model-%05d.pt' "$2" $((${#decks[@]} * COPY - 1))
}

# copy NAME: the copy, from its start or from where it stands.
copy() {
  local name=$1 decks updates latest
  IFS=, read -r -a decks <<< "${learners[$name]}"
  updates=$((${#decks[@]} * COPY))
  local arguments=(
    --device "$DEVICE" --workers "$WORKERS" --pool-threads "$THREADS" --pool-slots "$SLOTS"
    --run "$RUNS/$name" --opponents "$OPPONENTS" --duels "$DUELS" --eval-every 0
    --keep-checkpoints "$COPY" --seed "$SEED" --ppo-epochs 1 --learner "${learners[$name]}"
    --imitation-only-updates 1000000 --imitation-value-weight 0.5 --imitation-value-positions 4
    --leave-out "${leaves[$name]}"
  )
  if ! ls "$RUNS/$name"/checkpoints/model-*.pt > /dev/null 2>&1; then
    "${python[@]}" -m ygo_training train "${arguments[@]}" --updates "$updates"
  elif [ ! -f "$(last "$name" "$RUNS/$name")" ]; then
    latest=$(basename "$(ls "$RUNS/$name"/checkpoints/model-*.pt | sort | tail -1)" .pt)
    "${python[@]}" -m ygo_training train "${arguments[@]}" --continue \
      --updates $((updates - 1 - 10#${latest#model-}))
  fi
}

if [[ " $PARTS " == *" arms "* ]]; then
  for name in "${names[@]}"; do
    at_once "$name" copy "$name"
  done
  all_done
  echo "The copies are made: their runs are in $RUNS."
fi

if [[ " $PARTS " == *" measure "* ]]; then
  mkdir -p "$OUT/heldout" "$OUT/unseen" "$OUT/screens" "$OUT/runs"
  # shellcheck disable=SC2059
  screen() { printf "$SCREENS" "$1"; }
  # suited KIND FILE SUITE CHECKPOINT: the checkpoint on a suite.
  suited() {
    if [ ! -f "$OUT/$1/$2.json" ]; then
      "${python[@]}" -m ygo_training evaluate-suite "$4" --device "$DEVICE" \
        --workers "$WORKERS" --pool-threads "$THREADS" --pool-slots "$SLOTS" --seed "$SEED" \
        --suite "$3" --output "$OUT/$1/$2.json.part" > /dev/null
      mv "$OUT/$1/$2.json.part" "$OUT/$1/$2.json"
    fi
  }
  for name in "${names[@]}"; do
    control=${controls[$name]}
    for pair in "$name=$RUNS/$name" "$control=$CONTROLS/$control"; do
      file=${pair%%=*}
      folder=${pair#*=}
      checkpoint=$(last "$name" "$folder")
      if [ ! -f "$checkpoint" ]; then
        echo "$file is not made to its end ($checkpoint): left for later"
        continue
      fi
      cp "$folder/run.json" "$OUT/runs/$file.run.json"
      cp "$folder/metrics.jsonl" "$OUT/runs/$file.metrics.jsonl"
      if [ ! -f "$OUT/heldout/$file.json" ]; then
        "${python[@]}" -m ygo_training evaluate-search "$checkpoint" --device "$DEVICE" \
          --workers "$WORKERS" --pool-threads "$THREADS" --pool-slots "$SLOTS" \
          --records "$GAMES/heldout.jsonl" \
          --output "$OUT/heldout/$file.json.part" >> "$OUT/heldout/tables.txt"
        mv "$OUT/heldout/$file.json.part" "$OUT/heldout/$file.json"
      fi
      suited unseen "$file" "$UNSEEN" "$checkpoint"
      if [ "$name" = three-no-lists ]; then
        IFS=, read -r -a decks <<< "${learners[$name]}"
        for deck in "${decks[@]}"; do
          suited screens "$file.$deck" "$(screen "$deck")" "$checkpoint"
        done
      fi
    done
  done
  "${python[@]}" tools/deck_lists_report.py "$OUT" --runs "$OUT/runs" \
    --output "$OUT/report.json" | tee "$OUT/report.txt"
fi
