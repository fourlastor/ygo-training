#!/usr/bin/env bash
# The library's pool beside the workers on the CUDA machine: how long an
# evaluation and an update's games take by each route, and whether the two
# agree. A check of the batched environment, not an experiment: nothing it
# writes is a result. Run from the repository root; about ten minutes.
# 1. One checkpoint on the suite, by the workers and through the pool.
# 2. Updates that imitate the pilot, by each route: the same model to the
#    last digit is expected.
# 3. Updates whose games the model plays, from that checkpoint, one pass
#    each: by the workers, and through pools of each of SLOTS duels a thread.
# Its runs go to a folder of their own under RUNS and its report to
# OUT/report.txt.  With BEFORE, the folder an earlier check left its runs in,
# the report also says of every run through the pool whether it is that
# check's run again: the same figures and the same model.  That is what a
# change meant to alter nothing is held to.
set -euo pipefail
DEVICE=${DEVICE:-cuda}
WORKERS=${WORKERS:-24}
THREADS=${THREADS:-20}
SLOTS=${SLOTS:-8 16 32}
DUELS=${DUELS:-520}
UPDATES=${UPDATES:-3}
OPPONENTS=${OPPONENTS:-blackwing,burn,crystal,dragunity,gishki,gladiator,heroes,infernity,lightsworn,monarch,morphtronic,rock-block,spellcaster}
CHECKPOINT=${CHECKPOINT:-runs/learner/once/checkpoints/model-00359.pt}
SUITE=${SUITE:-eval_suites/ygo-policies-blackwing-screen.json}
RUNS=${RUNS:-runs/pool-check}/$(date +%Y%m%d-%H%M%S)
OUT=${OUT:-analysis/pool-check}
BEFORE=${BEFORE:-}
if [ -n "$BEFORE" ] && [ ! -d "$BEFORE" ]; then
  echo "No earlier check in $BEFORE" >&2
  exit 1
fi

pinned=$(git submodule status --recursive)
if grep -q '^[^ ]' <<< "$pinned"; then
  echo "A submodule is not at its pinned revision: git submodule update --init --recursive" >&2
  exit 1
fi
for needed in "$CHECKPOINT" "$SUITE"; do
  if [ ! -f "$needed" ]; then
    echo "Missing $needed" >&2
    exit 1
  fi
done
if ((UPDATES < 2)); then
  echo "The first update of a run warms up and is not timed: UPDATES is 2 at least" >&2
  exit 1
fi
cargo build --release
mkdir -p "$RUNS" "$OUT"
first=${SLOTS%% *}

echo "1. The suite, by the workers and through the pool"
suite() {
  uv run python -m ygo_training evaluate-suite "$CHECKPOINT" --device "$DEVICE" \
    --workers "$WORKERS" --seed 2012 --suite "$SUITE" --output "$RUNS/suite-$1.json" "${@:2}" \
    > "$RUNS/suite-$1.log"
}
suite workers
suite pool --pool-threads "$THREADS" --pool-slots "$first"

# train NAME OPTION...: UPDATES updates of a run of its own.
train() {
  echo "   $1"
  uv run python -m ygo_training train --device "$DEVICE" --workers "$WORKERS" \
    --run "$RUNS/$1" --learner blackwing --opponents "$OPPONENTS" --duels "$DUELS" \
    --eval-every 0 --keep-checkpoints "$UPDATES" --seed 2012 --updates "$UPDATES" "${@:2}" \
    > "$RUNS/$1.log"
}

echo "2. Updates that imitate the pilot"
imitate() {
  train "$@" --imitation-only-updates 1000 --imitation-value-weight 0.5 \
    --imitation-value-positions 4 --ppo-epochs 1
}
imitate imitation-workers
imitate imitation-pool --pool-threads "$THREADS" --pool-slots "$first"

echo "3. Updates whose games the model plays"
play() { train "$@" --resume "$CHECKPOINT" --ppo-epochs 1; }
play ppo-workers
for slots in $SLOTS; do
  play "ppo-pool-$slots" --pool-threads "$THREADS" --pool-slots "$slots"
done

uv run python - "$RUNS" "$UPDATES" "$THREADS" "$BEFORE" <<'EOF' | tee "$OUT/report.txt"
import json
import sys
from pathlib import Path

from ygo_training.checkpoints import training_state_fingerprints

runs, updates, threads = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
before = Path(sys.argv[4]) if sys.argv[4] else None


def rows(name):
    lines = (runs / name / "metrics.jsonl").read_text(encoding="utf8").splitlines()
    return [json.loads(line) for line in lines]


def mean(found, key):
    """Over the updates after the first, which warms up."""
    return sum(row[key] for row in found[1:]) / (len(found) - 1)


def told(row):
    """A row's figures but for its seconds and where its checkpoint is."""
    return {key: value for key, value in row.items() if "second" not in key and key != "checkpoint"}


def as_before(name):
    """Whether a run is the earlier check's run of that name again: every
    figure that check has of every update (a later trainer may report
    more), and the model and optimizer after the last."""
    if before is None or not (before / name / "metrics.jsonl").exists():
        return
    lines = (before / name / "metrics.jsonl").read_text(encoding="utf8").splitlines()
    earlier, now = [json.loads(line) for line in lines], rows(name)
    figures = len(earlier) == len(now) and all(
        key in b and b[key] == value
        for a, b in zip(earlier, now, strict=True)
        for key, value in told(a).items()
    )
    # The last checkpoint of each: a run taken on from a checkpoint goes on
    # from its number.
    newest = [sorted((folder / name / "checkpoints").glob("model-*.pt")) for folder in (before, runs)]
    model = (
        all(newest)
        and newest[0][-1].name == newest[1][-1].name
        and training_state_fingerprints(newest[0][-1]) == training_state_fingerprints(newest[1][-1])
    )
    print(
        f"            as in {before}: the same figures {'yes' if figures else 'NO'}, "
        f"the same model {'yes' if model else 'NO'}"
    )


print(f"The pool beside the workers: {runs}")
print()
print("1. One checkpoint on the suite")
for name in ("workers", "pool"):
    suite = json.loads((runs / f"suite-{name}.json").read_text(encoding="utf8"))
    cases = suite["records"][0]["evaluation_suite"].values()
    wins, games = sum(case["wins"] for case in cases), sum(case["games"] for case in cases)
    print(f"   {name:<8} {suite['seconds']:7.1f} s   wins {wins} of {games}")
print()
print(f"2. Updates that imitate the pilot, the mean of updates 2 to {updates}")
found = {name: rows(f"imitation-{name}") for name in ("workers", "pool")}
for name, measured in found.items():
    print(
        f"   {name:<8} games {mean(measured, 'seconds'):6.2f} s   "
        f"update {mean(measured, 'ppo_update_seconds'):6.2f} s   "
        f"all of it {mean(measured, 'wall_seconds'):6.2f} s   "
        f"decisions {mean(measured, 'samples'):7.0f}"
    )
    if name == "pool":
        as_before("imitation-pool")
same_games = all(
    ours["rollout_training_fingerprint"] == theirs["rollout_training_fingerprint"]
    for ours, theirs in zip(found["pool"], found["workers"], strict=True)
)
last = f"checkpoints/model-{updates - 1:05d}.pt"
states = [training_state_fingerprints(runs / f"imitation-{name}" / last) for name in found]
print(f"   the same games: {'yes' if same_games else 'NO'}")
print(f"   the same model after {updates} updates: {'yes' if states[0] == states[1] else 'NO'}")
print()
print(f"3. Updates whose games the model plays, the mean of updates 2 to {updates}")
print("   (all of it: with the games' estimates worked out again before the update, and the saving)")
names = sorted(
    (path.name for path in runs.glob("ppo-*") if path.is_dir()),
    key=lambda name: (name != "ppo-workers", int(name.rsplit("-", 1)[1]) if "pool" in name else 0),
)
for name in names:
    measured = rows(name)
    label = "workers" if name == "ppo-workers" else f"{threads} x {name.rsplit('-', 1)[1]}"
    print(
        f"   {label:<8} games {mean(measured, 'seconds'):6.2f} s   "
        f"{mean(measured, 'mean_inference_batch'):5.0f} a batch   "
        f"model {mean(measured, 'inference_seconds'):6.2f} s   "
        f"update {mean(measured, 'ppo_update_seconds'):6.2f} s   "
        f"all of it {mean(measured, 'wall_seconds'):6.2f} s   "
        f"decisions {mean(measured, 'decisions'):7.0f}   "
        f"wins {mean(measured, 'learner_win_rate'):.3f}"
    )
    if "pool_threads_seconds" in measured[-1]:
        # Of the games' seconds: the engines and pilots with the packing of
        # each decision, the building of the batches, their copy to Python.
        # Of the update's: the building of its minibatches.
        print(
            f"            of the games: engines {mean(measured, 'pool_threads_seconds'):6.2f} s, "
            f"batches {mean(measured, 'pool_batch_seconds'):6.2f} s, "
            f"copies {mean(measured, 'pool_copy_seconds'):6.2f} s;   "
            f"of the update: minibatches {mean(measured, 'pool_minibatch_seconds'):6.2f} s"
        )
        as_before(name)
EOF
echo "Report: $OUT/report.txt"
