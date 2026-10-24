#!/usr/bin/env bash
# Three PPO updates from versions 49 and 89 of the long control branch, logging
# the advantages PPO assigns to chain pass and activation (chain_* metrics).
set -euo pipefail
R=runs/ygo-policies-setcodes-branch-control-long/checkpoints
for v in 49 89; do
  uv run python -m ygo_training train --root .. --device cuda --run runs/chain-credit-probe-v$v --workers 24 --duels 520 --updates 3 --inference-batch 64 --batch-wait-ms 2 --limit 4096 --ppo-epochs 4 --minibatch 256 --learning-rate 3e-4 --rollout-temperature 1.0 --learner blackwing --opponents 'blackwing,burn,crystal,dragunity,gishki,gladiator,heroes,infernity,lightsworn,monarch,morphtronic,rock-block,spellcaster' --deck-pool ygo-policies --opponent-policy ygo-policies --teacher-profile ygo-policies --teacher-imitation-weight 0 --teacher-imitation-decay-updates 0 --puzzle-training 0 --optimization-mode source-aware --duel-advantage-normalization centered --puzzle-loss-weight 0.25 --duel-gae-lambda 0.95 --duel-deck-mix 'declared=1' --attack-target-mode joint-staged --monster-identity-weight 1 --no-randomize-monster-metadata --action-value-auxiliary-weight 0 --board-representation mean-pool --eval-every 0 --eval-suite eval_suites/ygo-policies-blackwing-screen.json --seed 2012 --resume $(printf '%s/model-%05d.pt' $R $v)
done
