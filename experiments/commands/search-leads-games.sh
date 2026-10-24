#!/usr/bin/env bash
# The searched games of the search-leads experiment: Blackwing against each
# of the 12 other decks, the pilot's game and the searched game of every
# deal, with the search's full log: how the pilot's answer and every
# alternative it tried did in the worlds of its first stage, also where the
# pilot's answer won them all. Run from the repository root.
# 0. The seed the arms will start from and its pilot, on the evaluation
#    suite. The seed learned that pilot against the pilots the other decks
#    had before ygo-policies 75e4b37, which changed ten of the twelve.
# 1. Games to choose a margin on, on a seed of their own.
# 2. The held-out games, and
# 3. the games to train on. Runs 2 and 3 have the deals of the
#    search-imitation experiment and are other games: the opponents answer
#    otherwise now.
# Each deck is a run of its own, so an interrupted script goes on with the
# decks it had not finished; a game's seed depends on the two decks' names and
# on the run's seed, not on which decks are played together.
set -euo pipefail
DEVICE=${DEVICE:-cuda}
WORKERS=${WORKERS:-24}
TRAIN_GAMES=${TRAIN_GAMES:-640}
TRAIN_SEED=${TRAIN_SEED:-800000}
VALIDATION_GAMES=${VALIDATION_GAMES:-32}
VALIDATION_SEED=${VALIDATION_SEED:-850000}
HELDOUT_GAMES=${HELDOUT_GAMES:-128}
HELDOUT_SEED=${HELDOUT_SEED:-900000}
OPPONENTS=${OPPONENTS:-burn,crystal,dragunity,gishki,gladiator,heroes,infernity,lightsworn,monarch,morphtronic,rock-block,spellcaster}
GAMES=${GAMES:-analysis/search-leads}
SEED_CHECKPOINT=${SEED_CHECKPOINT:-runs/format13-blackwing-imitation-seed-2012/checkpoints/model-00089.pt}
SUITE=${SUITE:-eval_suites/ygo-policies-blackwing-screen.json}

# The bench numbers a matchup's games from the run's seed on: two runs share
# deals unless their seeds lie further apart than their games per matchup.
apart() {
  local distance=$(($1 > $3 ? $1 - $3 : $3 - $1))
  if ((distance < $2 || distance < $4)); then
    echo "The seeds $1 and $3 are too close: the runs would share deals" >&2
    exit 1
  fi
}
apart "$TRAIN_SEED" "$TRAIN_GAMES" "$HELDOUT_SEED" "$HELDOUT_GAMES"
apart "$TRAIN_SEED" "$TRAIN_GAMES" "$VALIDATION_SEED" "$VALIDATION_GAMES"
apart "$VALIDATION_SEED" "$VALIDATION_GAMES" "$HELDOUT_SEED" "$HELDOUT_GAMES"
pinned=$(git submodule status --recursive)
if grep -q '^[^ ]' <<< "$pinned"; then
  echo "A submodule is not at its pinned revision: git submodule update --init --recursive" >&2
  exit 1
fi
if [ ! -f "$SEED_CHECKPOINT" ]; then
  echo "Missing $SEED_CHECKPOINT" >&2
  exit 1
fi
cargo build --release
(cd native/ygo-policies && cargo build --release -p ygo-policies-bench -p ygo-policies-ffi)
BENCH=native/ygo-policies/target/release/policy-bench
CARDS=native/ygo-policies/vendor/BabelCdb/cards.cdb

# search NAME GAMES SEED: NAME.jsonl, every deck's rows in turn.
search() {
  local name=$1 games=$2 seed=$3 deck part
  mkdir -p "$GAMES/$name"
  for deck in ${OPPONENTS//,/ }; do
    part="$GAMES/$name/$deck.jsonl"
    if [ ! -f "$part.done" ]; then
      rm -f "$part" "${part%.jsonl}.metadata.json"
      "$BENCH" search --policies blackwing --opponents "$deck" --games "$games" --seed "$seed" \
        --workers "$WORKERS" --record true --log full --output "$part"
      touch "$part.done"
    fi
  done
  for deck in ${OPPONENTS//,/ }; do cat "$GAMES/$name/$deck.jsonl"; done > "$GAMES/$name.jsonl"
  uv run python tools/policy_search_summary.py --cards "$CARDS" "$GAMES/$name.jsonl" \
    --output "$GAMES/$name.summary.json" | tee "$GAMES/$name.summary.txt"
}

mkdir -p "$GAMES"
revision=$(git -C native/ygo-policies rev-parse HEAD)
if [ -f "$GAMES/policies-revision.txt" ] && [ "$(cat "$GAMES/policies-revision.txt")" != "$revision" ]; then
  echo "$GAMES holds games of ygo-policies $(cat "$GAMES/policies-revision.txt"): give another GAMES" >&2
  exit 1
fi
echo "$revision" > "$GAMES/policies-revision.txt"
if [ ! -f "$GAMES/seed-suite.txt" ]; then
  uv run python -m ygo_training evaluate-suite --pilot --workers "$WORKERS" --suite "$SUITE" \
    --output "$GAMES/pilot-suite.json" > /dev/null
  uv run python -m ygo_training evaluate-suite "$SEED_CHECKPOINT" --device "$DEVICE" \
    --workers "$WORKERS" --suite "$SUITE" --output "$GAMES/seed-suite.json" > /dev/null
  uv run python - "$GAMES/pilot-suite.json" "$GAMES/seed-suite.json" <<'EOF' > "$GAMES/seed-suite.txt.part"
import json
import sys

from ygo_training.metrics import format_evaluation_suite_status

records = []
for path in sys.argv[1:]:
    with open(path, encoding="utf8") as saved:
        records += [
            {**record, "version": record["checkpoint_label"]}
            for record in json.load(saved)["records"]
        ]
print(format_evaluation_suite_status(records))
for record in records:
    agreed = [case.get("pilot_agreement") for case in record["evaluation_suite"].values()]
    if None not in agreed:
        share = sum(agreed) / len(agreed)
        print(f"{record['version']}: first choice the pilot's answer at {share:.1%}, matchups averaged")
EOF
  mv "$GAMES/seed-suite.txt.part" "$GAMES/seed-suite.txt"
fi
cat "$GAMES/seed-suite.txt"
search validation "$VALIDATION_GAMES" "$VALIDATION_SEED"
search heldout "$HELDOUT_GAMES" "$HELDOUT_SEED"
search train "$TRAIN_GAMES" "$TRAIN_SEED"
