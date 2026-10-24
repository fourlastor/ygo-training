#!/usr/bin/env bash
# More of the held-out run for the arms of the search-leads experiment. Run
# from the repository root, after the arms and their readings
# (search-leads-arms.sh). Nothing is trained and no reading changes.
# Every arm is measured on the held-out deals after each update of POINTS,
# where the readings took one update of each. That says whether an arm wins
# more deals than the pilot arm beyond that update, and at the update of the
# readings it gives the rule's changes apart by where they start: from the
# pilot's answer, back to it, or from one other answer to another. An arm's
# five checkpoints take about ten minutes.
# What is measured already is skipped.
set -euo pipefail
DEVICE=${DEVICE:-cuda}
WORKERS=${WORKERS:-24}
POINTS=${POINTS:-16 32 64 96 128}
ARMS=${ARMS:-pilot search leads leads-read}
GAMES=${GAMES:-analysis/search-leads}
RUNS=${RUNS:-runs/search-leads}
OUT=${OUT:-analysis/search-leads/arms}

pinned=$(git submodule status --recursive)
if grep -q '^[^ ]' <<< "$pinned"; then
  echo "A submodule is not at its pinned revision: git submodule update --init --recursive" >&2
  exit 1
fi
for needed in "$GAMES/heldout.jsonl" "$GAMES/policies-revision.txt" "$OUT/seed.txt" \
  "$OUT/choice.json" "$OUT/heldout/seed.json"; do
  if [ ! -f "$needed" ]; then
    echo "Missing $needed" >&2
    exit 1
  fi
done
# A recorded duel plays again only on the pilots it was recorded with.
revision=$(git -C native/ygo-policies rev-parse HEAD)
if [ "$(cat "$GAMES/policies-revision.txt")" != "$revision" ]; then
  echo "The games of $GAMES are of ygo-policies $(cat "$GAMES/policies-revision.txt"), not $revision" >&2
  exit 1
fi
cargo build --release
mkdir -p "$OUT/more"

# The version of the seed the arms started from: update k of a run wrote
# version seed + k.
seed=$(basename "$(cat "$OUT/seed.txt")" .pt)
seed=$((10#${seed#model-}))

for name in $ARMS; do
  measured=()
  for point in $POINTS; do
    measured+=("$(printf '%s/checkpoints/model-%05d.pt' "$RUNS/$name" $((seed + point)))")
  done
  if [ ! -f "$OUT/more/$name.json" ]; then
    uv run python -m ygo_training evaluate-search "${measured[@]}" --device "$DEVICE" \
      --workers "$WORKERS" --records "$GAMES/heldout.jsonl" --output "$OUT/more/$name.json.part" \
      | tee -a "$OUT/more/tables.txt"
    mv "$OUT/more/$name.json.part" "$OUT/more/$name.json"
  fi
done
uv run python tools/search_leads_report.py more "$OUT" --output "$OUT/more/report.json" \
  | tee "$OUT/more/report.txt"
