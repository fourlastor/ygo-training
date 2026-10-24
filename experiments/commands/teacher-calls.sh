#!/usr/bin/env bash
# How fast the teachers' games of a copy go in each way of asking them
# (experiments/2026-10-24-specialist-distillation.md, its section on the
# copy's speed): the copy of specialist distillation, the two champions
# teaching one network of both decks, for UPDATES updates in each setting of
# SETTINGS, each NAME:SLOTS:GROUPS:GRAPHS (0 for neither named), into
# RUNS/NAME; then each setting's seconds an update after its first, and where
# they went. Run from the repository root on the CUDA machine, with nothing
# else running there.
#   bash experiments/commands/teacher-calls.sh
set -euo pipefail
DEVICE=${DEVICE:-cuda}
WORKERS=${WORKERS:-24}
THREADS=${THREADS:-20}
DUELS=${DUELS:-520}
UPDATES=${UPDATES:-12}
ITSELF=${ITSELF:-260}
SEED=${SEED:-2012}
THIRTEEN=${THIRTEEN:-blackwing,lightsworn,gishki,burn,crystal,dragunity,gladiator,heroes,infernity,monarch,morphtronic,rock-block,spellcaster}
BLACKWING_TEACHER=${BLACKWING_TEACHER:-runs/two-decks/blackwing/checkpoints/model-00959.pt}
CYBER_TEACHER=${CYBER_TEACHER:-runs/cyber-champion/cyber-old/checkpoints/model-00959.pt}
# As the run was begun; an update's duels at once; and those in two groups,
# from graphs, and both.
SETTINGS=${SETTINGS:-as-begun:16:0:0 at-once:26:0:0 groups:26:2:0 graphs:26:0:260 both:26:2:130}
RUNS=${RUNS:-runs/teacher-calls}
PYTHON=${PYTHON:-}
python=(uv run python)
[ -z "$PYTHON" ] || python=("$PYTHON")
for path in "$BLACKWING_TEACHER" "$CYBER_TEACHER"; do
  if [ ! -f "$path" ]; then
    echo "Missing $path" >&2
    exit 1
  fi
done
cargo build --release
mkdir -p "$RUNS"
for setting in $SETTINGS; do
  IFS=: read -r name slots groups graphs <<< "$setting"
  if [ -f "$RUNS/$name/metrics.jsonl" ]; then
    continue
  fi
  named=()
  ((groups == 0)) || named+=(--pool-groups "$groups")
  ((graphs == 0)) || named+=(--inference-graphs "$graphs")
  echo "$(date +%H:%M) $name"
  "${python[@]}" -m ygo_training train --device "$DEVICE" --workers "$WORKERS" \
    --pool-threads "$THREADS" --pool-slots "$slots" "${named[@]}" --run "$RUNS/$name" \
    --learner blackwing,cyber --opponents "$THIRTEEN" --duels "$DUELS" --updates "$UPDATES" \
    --eval-every 0 --keep-checkpoints "$UPDATES" --seed "$SEED" --ppo-epochs 1 \
    --imitation-only-updates 1000000 --imitation-value-weight 0.5 \
    --imitation-value-positions 4 --minibatch 256 --duels-in-flight 0 \
    --self-play-duels "$ITSELF" --self-play-pairs mirrors \
    --imitation-teacher "blackwing=$BLACKWING_TEACHER" \
    --imitation-teacher "cyber=$CYBER_TEACHER" > /dev/null
done
"${python[@]}" - "$RUNS" $SETTINGS <<'EOF'
import json
import sys
from pathlib import Path

runs = Path(sys.argv[1])
print("setting      s/update  games  teachers  engines  update  calls  graphed")
for setting in sys.argv[2:]:
    name = setting.split(":")[0]
    rows = [json.loads(line) for line in (runs / name / "metrics.jsonl").read_text().splitlines()]
    after = rows[1:] or rows

    def mean(key: str) -> float:
        return sum(float(row.get(key, 0.0)) for row in after) / len(after)

    print(
        f"{name:<12} {mean('wall_seconds'):8.1f} {mean('seconds'):6.1f} "
        f"{mean('inference_seconds'):9.1f} {mean('pool_threads_seconds'):8.1f} "
        f"{mean('ppo_update_seconds'):7.1f} {mean('inference_batches'):6.0f} "
        f"{mean('pool_graph_batches'):8.0f}"
    )
EOF
