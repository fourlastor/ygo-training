#!/usr/bin/env bash
# PPO regression remedies, phase 1b. Run from training/.
# 1. Imitation control: 20 more teacher-executed imitation updates from the
#    warm-up seed (v10-v29), the warm-up's own recipe against the current
#    opponents and teacher. It separates DAgger's learner-state labels from
#    more imitation.
# 2. The three phase 1 arms continued in place to v59.
# Finished runs are skipped, so the script can be re-run after an
# interruption. An interrupted imitation control must be deleted first:
# continuing it in place would move its imitation schedule.
set -euo pipefail
BASE=(
  --root .. --device cuda --workers 24 --duels 520
  --inference-batch 64 --batch-wait-ms 2 --limit 4096 --ppo-epochs 4 --minibatch 256
  --learning-rate 3e-4 --rollout-temperature 1.0 --learner blackwing
  --opponents 'blackwing,burn,crystal,dragunity,gishki,gladiator,heroes,infernity,lightsworn,monarch,morphtronic,rock-block,spellcaster'
  --deck-pool ygo-policies --opponent-policy ygo-policies --teacher-profile ygo-policies
  --teacher-imitation-decay-updates 0 --puzzle-training 0 --optimization-mode source-aware
  --duel-advantage-normalization centered --puzzle-loss-weight 0.25 --duel-gae-lambda 0.95
  --duel-deck-mix 'declared=1' --attack-target-mode joint-staged --monster-identity-weight 1
  --no-randomize-monster-metadata --action-value-auxiliary-weight 0 --board-representation mean-pool
  --eval-every 10 --eval-suite eval_suites/ygo-policies-blackwing-screen.json --seed 2012
)

# Continue a run in place up to a checkpoint version.
continue_to() {
  local run=$1 target=$2 latest remaining
  shift 2
  if [ ! -d "$run/checkpoints" ]; then
    echo "Missing $run: run regression-remedies-phase1.sh first" >&2
    exit 1
  fi
  latest=$(basename "$(ls "$run"/checkpoints/model-*.pt | sort | tail -1)" .pt)
  remaining=$((target - 10#${latest#model-}))
  if ((remaining > 0)); then
    uv run python -m ygo_training train "${BASE[@]}" --run "$run" --updates "$remaining" --continue "$@"
  fi
}

BC=runs/ygo-policies-remedies-bc
if [ ! -f "$BC/checkpoints/model-00029.pt" ]; then
  uv run python -m ygo_training train "${BASE[@]}" --run "$BC" --updates 20 \
    --resume runs/ygo-policies-setcodes-baseline-seed-2012/checkpoints/model-00009.pt \
    --imitation-only-updates 20 --warmup-execution teacher --imitation-value-weight 0.5
fi
continue_to runs/ygo-policies-remedies-dagger 59 \
  --teacher-imitation-weight 1 --policy-loss-weight 0 --entropy-coefficient 0
continue_to runs/ygo-policies-remedies-anchor-0.1 59 --teacher-imitation-weight 0.1
continue_to runs/ygo-policies-remedies-anchor-1 59 --teacher-imitation-weight 1
bash experiments/commands/regression-remedies-report.sh
