#!/usr/bin/env bash
# The value head of the imitation seed, measured where it is trained. Run from
# the repository root.
# A few imitation-only updates of a run branched from a checkpoint of the seed
# run, on the games that run would play next. Each update reports, before it
# optimizes, the head's explained variance at the first step of every
# transaction of its games, which are new to the model: the first row is the
# checkpoint's own. The seed run is left as it is. About 25 seconds an update
# on the CUDA machine.
set -euo pipefail
DEVICE=${DEVICE:-cuda}
WORKERS=${WORKERS:-24}
SEED_DUELS=${SEED_DUELS:-520}
UPDATES=${UPDATES:-4}
SEED_RUN=${SEED_RUN:-runs/format13-blackwing-imitation-seed-2012}
SEED_OUT=${SEED_OUT:-analysis/format13-seed}
CHECKPOINT=${CHECKPOINT:-$(ls "$SEED_RUN"/checkpoints/model-*.pt | sort | tail -1)}
version=$(basename "$CHECKPOINT" .pt)
version=${version#model-}
RUN=${RUN:-runs/format13-value-check-$version}

pinned=$(git submodule status --recursive)
if grep -q '^[^ ]' <<< "$pinned"; then
  echo "A submodule is not at its pinned revision: git submodule update --init --recursive" >&2
  exit 1
fi
cargo build --release

if [ ! -f "$RUN/metrics.jsonl" ]; then
  uv run python -m ygo_training train --device "$DEVICE" --run "$RUN" --learner blackwing \
    --opponents 'blackwing,burn,crystal,dragunity,gishki,gladiator,heroes,infernity,lightsworn,monarch,morphtronic,rock-block,spellcaster' \
    --workers "$WORKERS" --duels "$SEED_DUELS" \
    --imitation-only-updates 1000 --imitation-value-weight 0.5 --ppo-epochs 4 \
    --eval-every 0 --seed 2012 --updates "$UPDATES" --resume "$CHECKPOINT"
fi
mkdir -p "$SEED_OUT"
uv run python -m ygo_training status "$RUN/metrics.jsonl" --ppo | tee "$SEED_OUT/value-check-$version.txt"
