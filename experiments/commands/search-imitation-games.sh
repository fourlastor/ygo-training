#!/usr/bin/env bash
# The searched games of the search-imitation experiment: Blackwing against
# each of the 12 other decks, the pilot's game and the searched game of every
# deal. Run from the repository root.
# 1. The held-out games, with the search's log, and from it how firmly the
#    search decided where it left its pilot.
# 2. The games to train on, on a seed far from the first.
# Each deck is a run of its own, so an interrupted script goes on with the
# decks it had not finished; a game's seed depends on the two decks' names and
# on the run's seed, not on which decks are played together.
set -euo pipefail
WORKERS=${WORKERS:-24}
TRAIN_GAMES=${TRAIN_GAMES:-640}
TRAIN_SEED=${TRAIN_SEED:-800000}
HELDOUT_GAMES=${HELDOUT_GAMES:-128}
HELDOUT_SEED=${HELDOUT_SEED:-900000}
OPPONENTS=${OPPONENTS:-burn,crystal,dragunity,gishki,gladiator,heroes,infernity,lightsworn,monarch,morphtronic,rock-block,spellcaster}
GAMES=${GAMES:-analysis/search-imitation}

# The bench numbers a matchup's games from the run's seed on: two runs share
# deals unless their seeds lie further apart than their games per matchup.
apart=$((TRAIN_SEED > HELDOUT_SEED ? TRAIN_SEED - HELDOUT_SEED : HELDOUT_SEED - TRAIN_SEED))
if ((apart < TRAIN_GAMES || apart < HELDOUT_GAMES)); then
  echo "The two seeds are too close: the runs would share deals" >&2
  exit 1
fi
pinned=$(git submodule status --recursive)
if grep -q '^[^ ]' <<< "$pinned"; then
  echo "A submodule is not at its pinned revision: git submodule update --init --recursive" >&2
  exit 1
fi
(cd native/ygo-policies && cargo build --release -p ygo-policies-bench -p ygo-policies-ffi)
BENCH=native/ygo-policies/target/release/policy-bench
CARDS=native/ygo-policies/vendor/BabelCdb/cards.cdb

# search NAME GAMES SEED [OPTIONS]: NAME.jsonl, every deck's rows in turn.
search() {
  local name=$1 games=$2 seed=$3 deck part
  shift 3
  mkdir -p "$GAMES/$name"
  for deck in ${OPPONENTS//,/ }; do
    part="$GAMES/$name/$deck.jsonl"
    if [ ! -f "$part.done" ]; then
      "$BENCH" search --policies blackwing --opponents "$deck" --games "$games" --seed "$seed" \
        --workers "$WORKERS" --record true "$@" --output "$part"
      touch "$part.done"
    fi
  done
  for deck in ${OPPONENTS//,/ }; do cat "$GAMES/$name/$deck.jsonl"; done > "$GAMES/$name.jsonl"
  uv run python tools/policy_search_summary.py --cards "$CARDS" "$GAMES/$name.jsonl" \
    --output "$GAMES/$name.summary.json" | tee "$GAMES/$name.summary.txt"
}

search heldout "$HELDOUT_GAMES" "$HELDOUT_SEED" --log true
uv run python tools/policy_search_firmness.py "$GAMES/heldout.jsonl" \
  --output "$GAMES/heldout.firmness.json" | tee "$GAMES/heldout.firmness.txt"
search train "$TRAIN_GAMES" "$TRAIN_SEED"
git -C native/ygo-policies rev-parse HEAD > "$GAMES/policies-revision.txt"
