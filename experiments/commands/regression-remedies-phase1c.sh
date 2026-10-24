#!/usr/bin/env bash
# PPO regression remedies, phase 1c. Run from training/.
# 1. Does imitation distill other decks? For Monarch (the strongest teacher,
#    its seed within 5 points of it) and Spellcaster (the weakest, its seed at
#    about half its win rate): the warm-up seed (v9) evaluated against the
#    current opponents, then 40 more teacher-executed imitation updates from
#    it (v10-v49). DECKS=monarch leaves Spellcaster out.
# 2. Does the 0.1 anchor have to stay? Three branches from Anchor 0.1 at v59,
#    20 updates each (v60-v79): the anchor kept, removed, and kept on a tenth
#    of the learner's decisions. The imitation term averages over the labelled
#    decisions, so a tenth of the labels pulls as hard, on less evidence.
# Finished runs are skipped and interrupted ones continued, so the script can
# be re-run. Results are gathered under analysis/remedies/.
set -euo pipefail
RUNTIME=(--root .. --device cuda --workers 24 --inference-batch 64 --batch-wait-ms 2 --limit 4096)
COMMON=(
  "${RUNTIME[@]}" --duels 520 --ppo-epochs 4 --minibatch 256
  --learning-rate 3e-4 --rollout-temperature 1.0
  --opponents 'blackwing,burn,crystal,dragunity,gishki,gladiator,heroes,infernity,lightsworn,monarch,morphtronic,rock-block,spellcaster'
  --deck-pool ygo-policies --opponent-policy ygo-policies --teacher-profile ygo-policies
  --teacher-imitation-decay-updates 0 --puzzle-training 0 --optimization-mode source-aware
  --duel-advantage-normalization centered --puzzle-loss-weight 0.25 --duel-gae-lambda 0.95
  --duel-deck-mix 'declared=1' --attack-target-mode joint-staged --monster-identity-weight 1
  --no-randomize-monster-metadata --action-value-auxiliary-weight 0 --board-representation mean-pool
  --eval-every 10 --seed 2012
)
DECKS=${DECKS:-monarch spellcaster}
ANCHOR=runs/ygo-policies-remedies-anchor-0.1/checkpoints/model-00059.pt
VALUES=analysis/action-values-blackwing-seed-7000.pt
OUT=analysis/remedies

seed_of() {
  echo "runs/ygo-policies-setcodes-baseline-$1-seed-2012/checkpoints/model-00009.pt"
}

# Branch a run from a checkpoint up to a version, or continue it if it began.
branch() {
  local run=$1 target=$2 source=$3 latest
  shift 3
  if ls "$run"/checkpoints/model-*.pt > /dev/null 2>&1; then
    latest=$(basename "$(ls "$run"/checkpoints/model-*.pt | sort | tail -1)" .pt)
    latest=$((10#${latest#model-}))
    if ((latest < target)); then
      uv run python -m ygo_training train "${COMMON[@]}" --run "$run" \
        --updates "$((target - latest))" --continue "$@"
    fi
  else
    latest=$(basename "$source" .pt)
    uv run python -m ygo_training train "${COMMON[@]}" --run "$run" \
      --updates "$((target - 10#${latest#model-}))" --resume "$source" "$@"
  fi
}

for input in "$ANCHOR" "$VALUES" $(for deck in $DECKS; do seed_of "$deck"; done); do
  [ -f "$input" ] || { echo "Missing $input" >&2; exit 1; }
done
mkdir -p "$OUT"

for deck in $DECKS; do
  seed=$(seed_of "$deck")
  suite=eval_suites/ygo-policies-$deck-screen.json
  if [ ! -f "$OUT/$deck-v9-evaluation.json" ]; then
    uv run python -m ygo_training evaluate-suite "${RUNTIME[@]}" --seed 2012 \
      --attack-target-mode joint-staged --board-representation mean-pool --deck-pool ygo-policies \
      "$seed" --suite "$suite" --output "$OUT/$deck-v9-evaluation.json"
  fi
  branch "runs/ygo-policies-remedies-$deck-imitation" 49 "$seed" \
    --learner "$deck" --eval-suite "$suite" \
    --imitation-only-updates 40 --warmup-execution teacher --imitation-value-weight 0.5
done

BLACKWING=(--learner blackwing --eval-suite eval_suites/ygo-policies-blackwing-screen.json)
branch runs/ygo-policies-remedies-anchor-kept 79 "$ANCHOR" "${BLACKWING[@]}" \
  --teacher-imitation-weight 0.1
branch runs/ygo-policies-remedies-anchor-off 79 "$ANCHOR" "${BLACKWING[@]}" \
  --teacher-imitation-weight 0
branch runs/ygo-policies-remedies-anchor-thin 79 "$ANCHOR" "${BLACKWING[@]}" \
  --teacher-imitation-weight 0.1 --teacher-label-fraction 0.1

# Each run's metrics and status table, and the Blackwing branches' decisions
# on the 274 branched states next to the checkpoint they started from.
checkpoints=(--checkpoint "anchor-0.1-v59=$ANCHOR")
for arm in anchor-kept anchor-off anchor-thin; do
  for version in 64 69 74 79; do
    checkpoint=$(printf 'runs/ygo-policies-remedies-%s/checkpoints/model-%05d.pt' "$arm" "$version")
    checkpoints+=(--checkpoint "$arm-v$version=$checkpoint")
  done
done
for arm in anchor-kept anchor-off anchor-thin $(printf '%s-imitation ' $DECKS); do
  run=runs/ygo-policies-remedies-$arm
  cp "$run/metrics.jsonl" "$OUT/$arm-metrics.jsonl"
  uv run python -m ygo_training status "$run/metrics.jsonl" --ppo > "$OUT/$arm-status.txt"
done
uv run python -m ygo_training.action_values score --values "$VALUES" "${checkpoints[@]}" \
  > "$OUT/action-values-score-phase1c.json"
echo "Results are in $OUT"
