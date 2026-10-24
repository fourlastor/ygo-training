#!/usr/bin/env bash
# Attribute one PPO update's effect on the fixed Blackwing branched states to
# each objective term (update_attribution), from the warm-up seed (v9) and
# versions 49 and 89 of the long control branch. Each run reproduces that
# branch's next update exactly and writes no checkpoint.
set -euo pipefail
R=runs/ygo-policies-setcodes-branch-control-long/checkpoints
for pair in \
  "9 runs/ygo-policies-setcodes-baseline-seed-2012/checkpoints/model-00009.pt" \
  "49 $R/model-00049.pt" \
  "89 $R/model-00089.pt"; do
  set -- $pair
  uv run python -m ygo_training.update_attribution --values analysis/action-values-blackwing-seed-7000.pt --output analysis/update-attribution-v$1.json -- --root .. --device cuda --workers 24 --duels 520 --inference-batch 64 --batch-wait-ms 2 --limit 4096 --ppo-epochs 4 --minibatch 256 --learning-rate 3e-4 --rollout-temperature 1.0 --learner blackwing --opponents 'blackwing,burn,crystal,dragunity,gishki,gladiator,heroes,infernity,lightsworn,monarch,morphtronic,rock-block,spellcaster' --deck-pool ygo-policies --opponent-policy ygo-policies --teacher-profile ygo-policies --teacher-imitation-weight 0 --teacher-imitation-decay-updates 0 --puzzle-training 0 --optimization-mode source-aware --duel-advantage-normalization centered --puzzle-loss-weight 0.25 --duel-gae-lambda 0.95 --duel-deck-mix 'declared=1' --attack-target-mode joint-staged --monster-identity-weight 1 --no-randomize-monster-metadata --action-value-auxiliary-weight 0 --board-representation mean-pool --seed 2012 --resume $2
done
