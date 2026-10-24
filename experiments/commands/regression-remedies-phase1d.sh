#!/usr/bin/env bash
# PPO regression remedies, phase 1d. Run from training/.
# 1. Does the anchored arm keep rising past its teacher, and is the rise
#    PPO's? Anchor kept continues in place to v119 (40 updates), and DAgger,
#    the same labels without the PPO term, continues in place over the same
#    versions (v60-v119) as its control.
# 2. Does Spellcaster's imitation reach its teacher, and whose states have to
#    be labelled? Two branches from the imitation run's v49, 40 updates each
#    (v50-v89): more teacher-executed imitation, and DAgger on the learner's
#    own states. SPELLCASTER_DAGGER=0 leaves the second out.
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
BLACKWING=(--learner blackwing --eval-suite eval_suites/ygo-policies-blackwing-screen.json)
SPELLCASTER=(--learner spellcaster --eval-suite eval_suites/ygo-policies-spellcaster-screen.json)
DAGGER=(--teacher-imitation-weight 1 --policy-loss-weight 0 --entropy-coefficient 0)
SPELLCASTER_DAGGER=${SPELLCASTER_DAGGER:-1}
KEPT=runs/ygo-policies-remedies-anchor-kept
CONTROL=runs/ygo-policies-remedies-dagger
IMITATED=runs/ygo-policies-remedies-spellcaster-imitation/checkpoints/model-00049.pt
VALUES=analysis/action-values-blackwing-seed-7000.pt
OUT=analysis/remedies

latest() {
  local name
  name=$(basename "$(ls "$1"/checkpoints/model-*.pt | sort | tail -1)" .pt)
  echo "$((10#${name#model-}))"
}

# Continue an existing run in place up to a version.
continue_to() {
  local run=$1 target=$2 from
  shift 2
  from=$(latest "$run")
  if ((from < target)); then
    uv run python -m ygo_training train "${COMMON[@]}" --run "$run" \
      --updates "$((target - from))" --continue "$@"
  fi
}

# Branch a run from a checkpoint up to a version, or continue it if it began.
branch() {
  local run=$1 target=$2 source=$3 from
  shift 3
  if ls "$run"/checkpoints/model-*.pt > /dev/null 2>&1; then
    continue_to "$run" "$target" "$@"
  else
    from=$(basename "$source" .pt)
    uv run python -m ygo_training train "${COMMON[@]}" --run "$run" \
      --updates "$((target - 10#${from#model-}))" --resume "$source" "$@"
  fi
}

for input in "$KEPT/checkpoints/model-00079.pt" "$CONTROL/checkpoints/model-00059.pt" "$IMITATED" "$VALUES"; do
  [ -f "$input" ] || { echo "Missing $input" >&2; exit 1; }
done
mkdir -p "$OUT"

continue_to "$KEPT" 119 "${BLACKWING[@]}" --teacher-imitation-weight 0.1
continue_to "$CONTROL" 119 "${BLACKWING[@]}" "${DAGGER[@]}"

# A continuation cannot lengthen an imitation schedule, so more imitation is
# a branch from the last checkpoint: it restores the optimizer and goes on
# with the same rollout seeds.
arms=(anchor-kept dagger spellcaster-bc)
branch runs/ygo-policies-remedies-spellcaster-bc 89 "$IMITATED" "${SPELLCASTER[@]}" \
  --imitation-only-updates 40 --warmup-execution teacher --imitation-value-weight 0.5
if [ "$SPELLCASTER_DAGGER" != 0 ]; then
  arms+=(spellcaster-dagger)
  branch runs/ygo-policies-remedies-spellcaster-dagger 89 "$IMITATED" "${SPELLCASTER[@]}" \
    "${DAGGER[@]}"
fi

# Each run's metrics and status table, and the two Blackwing arms' decisions
# on the 274 branched states.
checkpoints=()
for version in 79 99 119; do
  checkpoints+=(--checkpoint "anchor-kept-v$version=$(printf '%s/checkpoints/model-%05d.pt' "$KEPT" "$version")")
done
for version in 59 79 99 119; do
  checkpoints+=(--checkpoint "dagger-v$version=$(printf '%s/checkpoints/model-%05d.pt' "$CONTROL" "$version")")
done
for arm in "${arms[@]}"; do
  run=runs/ygo-policies-remedies-$arm
  cp "$run/metrics.jsonl" "$OUT/$arm-metrics.jsonl"
  uv run python -m ygo_training status "$run/metrics.jsonl" --ppo > "$OUT/$arm-status.txt"
done
uv run python -m ygo_training.action_values score --values "$VALUES" "${checkpoints[@]}" \
  > "$OUT/action-values-score-phase1d.json"
echo "Results are in $OUT"
