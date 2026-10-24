#!/usr/bin/env bash
# PPO regression remedies, phase 1: DAgger and imitation-anchored PPO branches
# from the Blackwing warm-up seed (v9), 20 updates each (v10-v29), with the
# long control branch's protocol otherwise. Teacher labels come from the
# learner deck's own ygo-policies policy watching the learner seat.
set -euo pipefail
BASE=(
  --root .. --device cuda --workers 24 --duels 520 --updates 20
  --inference-batch 64 --batch-wait-ms 2 --limit 4096 --ppo-epochs 4 --minibatch 256
  --learning-rate 3e-4 --rollout-temperature 1.0 --learner blackwing
  --opponents 'blackwing,burn,crystal,dragunity,gishki,gladiator,heroes,infernity,lightsworn,monarch,morphtronic,rock-block,spellcaster'
  --deck-pool ygo-policies --opponent-policy ygo-policies --teacher-profile ygo-policies
  --teacher-imitation-decay-updates 0 --puzzle-training 0 --optimization-mode source-aware
  --duel-advantage-normalization centered --puzzle-loss-weight 0.25 --duel-gae-lambda 0.95
  --duel-deck-mix 'declared=1' --attack-target-mode joint-staged --monster-identity-weight 1
  --no-randomize-monster-metadata --action-value-auxiliary-weight 0 --board-representation mean-pool
  --eval-every 10 --eval-suite eval_suites/ygo-policies-blackwing-screen.json --seed 2012
  --resume runs/ygo-policies-setcodes-baseline-seed-2012/checkpoints/model-00009.pt
)
uv run python -m ygo_training train "${BASE[@]}" --run runs/ygo-policies-remedies-dagger \
  --teacher-imitation-weight 1 --policy-loss-weight 0 --entropy-coefficient 0
uv run python -m ygo_training train "${BASE[@]}" --run runs/ygo-policies-remedies-anchor-0.1 \
  --teacher-imitation-weight 0.1
uv run python -m ygo_training train "${BASE[@]}" --run runs/ygo-policies-remedies-anchor-1 \
  --teacher-imitation-weight 1
