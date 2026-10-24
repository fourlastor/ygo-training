#!/usr/bin/env bash
# A floor under each deck's ceiling: `policy-bench search` plays every game
# twice on the same seed, the pilot alone and the pilot with a one-step search
# at each of its decisions. CPU only (about 40 s per searched game per core).
# The training protocol's runs use this checkout's engine build, pilots, deck
# lists, scripts and card database; the updated ones use another checkout of
# ygo-policies for the pilots and lists, on the same engine build.
# Run from training/:
#   bash experiments/commands/policy-search-floor.sh POLICIES_CHECKOUT OUTPUT_DIRECTORY
# POLICIES_CHECKOUT is a ygo-policies checkout with the `search` command
# (branch search-player), built with
#   cargo build --release -p ygo-policies-ffi -p ygo-policies-bench
set -euo pipefail
UPDATED=$(cd "$1" && pwd)
OUT=$2
ROOT=$(cd .. && pwd)
BENCH=$UPDATED/target/release/policy-bench
CORE=$ROOT/YGO.TrainingWorker/bin/Release/net9.0/libocgcore.so
OLD=(
  --library "$ROOT/native/ygo-policies/target/release/libygo_policies.so" --core "$CORE"
  --cards "$ROOT/native/cdb/cards.cdb" --scripts "$ROOT/native/scripts"
  --decks "$ROOT/native/ygo-policies/decks"
)
NEW=(
  --library "$UPDATED/target/release/libygo_policies.so" --core "$CORE"
  --cards "$UPDATED/vendor/BabelCdb/cards.cdb" --scripts "$UPDATED/vendor/CardScripts"
  --decks "$UPDATED/decks"
)
mkdir -p "$OUT"
search() {
  local name=$1
  shift
  if [ ! -f "$OUT/$name.jsonl" ]; then
    "$BENCH" search --opponents existing --workers 22 --output "$OUT/$name.jsonl" "$@"
  fi
}
search monarch-old-strict --policies monarch --games 64 --strict true "${OLD[@]}"
search blackwing-old-strict --policies blackwing --games 64 --strict true "${OLD[@]}"
search blackwing-new-strict --policies blackwing --games 48 --strict true "${NEW[@]}"
search monarch-new-strict --policies monarch --games 32 --strict true "${NEW[@]}"
search monarch-old-loose --policies monarch --games 32 --strict false "${OLD[@]}"
search blackwing-old-loose --policies blackwing --games 32 --strict false "${OLD[@]}"
# Twice the worlds at every stage.
DEEP=(--games 32 --strict true --worlds 16 --confirm 64 --final 192)
search blackwing-old-strict-deep --policies blackwing "${DEEP[@]}" "${OLD[@]}"
search monarch-old-strict-deep --policies monarch "${DEEP[@]}" "${OLD[@]}"
# Not a player: what luck leaves when every hidden card and draw is known.
search blackwing-old-foresight --policies blackwing --games 64 --foresight true "${OLD[@]}"
search monarch-old-foresight --policies monarch --games 64 --foresight true "${OLD[@]}"
uv run python tools/policy_search_summary.py --cards "$ROOT/native/cdb/cards.cdb" \
  "$OUT"/*.jsonl --output "$OUT/summary.json"
