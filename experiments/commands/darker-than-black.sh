#!/usr/bin/env bash
# Darker Than Black: does a network learn decks, or cards?
# (experiments/2026-10-19-darker-than-black.md). Run from the repository root
# on the machine the runs are on. Into OUT, on the wide Blackwing screen
# (SUITE: the thirteen decks' pilots, 512 games each), with Blackwing's own
# list and with LIST (Darker Than Black, a variant of it, which no pilot was
# written for): Blackwing's pilot, and the networks' checkpoints after 1,680,
# 1,800 and 1,920 updates of PPO, half's of the experiment on thirteen decks
# (HALF_CHECKPOINTS) and all's of the experiment on thirty-one decks
# (ALL_CHECKPOINTS). Then the report (tools/darker_than_black_report.py),
# which takes the reading. A finished measurement is not made again. About
# half an hour by reckoning.
#   bash experiments/commands/darker-than-black.sh
set -euo pipefail
DEVICE=${DEVICE:-cuda}
WORKERS=${WORKERS:-24}
THREADS=${THREADS:-20}
SLOTS=${SLOTS:-16}
SEED=${SEED:-2012}
SUITE=${SUITE:-eval_suites/ygo-policies-blackwing-wide.json}
LIST=${LIST:-data/decks/darker-than-black.ydk}
HALF_CHECKPOINTS=${HALF_CHECKPOINTS:-runs/thirteen-decks/half/checkpoints/model-04799.pt,runs/thirteen-decks/half/checkpoints/model-04919.pt,runs/thirteen-decks/half/checkpoints/model-05039.pt}
ALL_CHECKPOINTS=${ALL_CHECKPOINTS:-runs/thirty-one-decks/all/checkpoints/model-09119.pt,runs/thirty-one-decks/all/checkpoints/model-09239.pt,runs/thirty-one-decks/all/checkpoints/model-09359.pt}
OUT=${OUT:-analysis/darker-than-black}
# The interpreter of a machine that has no uv: the benchmark's setup leaves one.
PYTHON=${PYTHON:-}

IFS=, read -r -a half <<< "$HALF_CHECKPOINTS"
IFS=, read -r -a all <<< "$ALL_CHECKPOINTS"
for path in "$SUITE" "$LIST" "${half[@]}" "${all[@]}"; do
  if [ ! -f "$path" ]; then
    echo "Missing $path" >&2
    exit 1
  fi
done
if ((${#half[@]} != 3 || ${#all[@]} != 3)); then
  echo "A network is read by three checkpoints" >&2
  exit 1
fi
pinned=$(git submodule status --recursive)
if grep -q '^[^ ]' <<< "$pinned"; then
  echo "A submodule is not at its pinned revision: git submodule update --init --recursive" >&2
  exit 1
fi
python=(uv run python)
[ -z "$PYTHON" ] || python=("$PYTHON")
cargo build --release
mkdir -p "$OUT"

# measure FILE OPTION...: the suite played with Blackwing's list and with
# LIST, by the pilot or the checkpoints the options name, into
# OUT/FILE.blackwing.json and OUT/FILE.darker-than-black.json.
measure() {
  local file=$1 listed
  shift
  for listed in blackwing darker-than-black; do
    if [ -f "$OUT/$file.$listed.json" ]; then
      continue
    fi
    local played=()
    [ "$listed" = blackwing ] || played=(--learner-list "$LIST")
    "${python[@]}" -m ygo_training evaluate-suite "$@" "${played[@]}" --device "$DEVICE" \
      --workers "$WORKERS" --pool-threads "$THREADS" --pool-slots "$SLOTS" --seed "$SEED" \
      --suite "$SUITE" --output "$OUT/$file.$listed.json.part" > /dev/null
    mv "$OUT/$file.$listed.json.part" "$OUT/$file.$listed.json"
  done
}
measure pilot --pilot
measure half "${half[@]}"
measure all "${all[@]}"
"${python[@]}" tools/darker_than_black_report.py "$OUT" --output "$OUT/report.json" \
  | tee "$OUT/report.txt"
echo "The measurements and what they say are in $OUT."
