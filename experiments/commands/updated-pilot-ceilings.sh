#!/usr/bin/env bash
# Ceiling reference: each deck's pilot from another build of ygo-policies (an
# updated checkout) against this checkout's opponents, on that deck's suite and
# this checkout's deck lists. CPU only, about 20 s per suite with 20 workers.
# Run from training/:
#   bash experiments/commands/updated-pilot-ceilings.sh LIBRARY OUTPUT_DIRECTORY
# LIBRARY is the other build's libygo_policies.so
# (cargo build --release -p ygo-policies-ffi in that checkout).
set -euo pipefail
LIBRARY=$1
OUT=$2
mkdir -p "$OUT"
for deck in blackwing burn crystal dragunity gishki gladiator heroes infernity lightsworn monarch morphtronic rock-block spellcaster; do
  uv run python -m ygo_training.teacher_evaluation --root .. \
    --suite "eval_suites/ygo-policies-$deck-screen.json" --workers 20 \
    --learner-library "$LIBRARY" --output "$OUT/$deck.jsonl" |
    sed -n "2s/^teacher  aggregate/$deck/p"
done
