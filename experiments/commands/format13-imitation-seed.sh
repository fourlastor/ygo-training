#!/usr/bin/env bash
# The first full-size run on the ported setup (checkpoint format 13), and the
# seed of the search-imitation experiment. Run from the repository root.
# 1. The reference: the Blackwing pilot's own games on the evaluation suite.
# 2. Imitation: the pilot plays, the learner imitates, 520 duels an update,
#    evaluated on the suite every 10 updates, up to version SEED_TARGET - 1.
# 3. To its plateau: 20 more updates at a time, up to version SEED_MAX - 1,
#    while the last two evaluations are on average more than SEED_FLAT above
#    the two before. One evaluation is good to a point, and the first form of
#    this rule, which compared the last with the one before, stopped the seed
#    of the search-imitation experiment while it was still learning.
# It needs no search. Run again, it continues the run, finished or
# interrupted: every update of the run is an imitation update. It fails if
# the seed still rises at SEED_MAX, or levels off more than SEED_SHORT under
# its pilot, so that nothing is built on such a seed without a look.
set -euo pipefail
DEVICE=${DEVICE:-cuda}
WORKERS=${WORKERS:-24}
SEED_TARGET=${SEED_TARGET:-50}
SEED_STEP=${SEED_STEP:-20}
SEED_MAX=${SEED_MAX:-250}
SEED_FLAT=${SEED_FLAT:-0.01}
SEED_SHORT=${SEED_SHORT:-0.03}
SEED_DUELS=${SEED_DUELS:-520}
SEED_EVERY=${SEED_EVERY:-10}
SEED_SUITE=${SEED_SUITE:-eval_suites/ygo-policies-blackwing-screen.json}
SEED_RUN=${SEED_RUN:-runs/format13-blackwing-imitation-seed-2012}
SEED_OUT=${SEED_OUT:-analysis/format13-seed}

# The duels are what the checked-out pilots, engine, scripts and cards make
# of a seed, and the engine is compiled into the environment's library.
pinned=$(git submodule status --recursive)
if grep -q '^[^ ]' <<< "$pinned"; then
  echo "A submodule is not at its pinned revision: git submodule update --init --recursive" >&2
  exit 1
fi
cargo build --release

mkdir -p "$SEED_OUT"
uv run python -m ygo_training evaluate-suite --pilot --workers "$WORKERS" --seed 2012 \
  --suite "$SEED_SUITE" --output "$SEED_OUT/pilot-reference.json" > /dev/null

TRAIN=(
  --device "$DEVICE" --run "$SEED_RUN" --learner blackwing
  --opponents 'blackwing,burn,crystal,dragunity,gishki,gladiator,heroes,infernity,lightsworn,monarch,morphtronic,rock-block,spellcaster'
  --workers "$WORKERS" --duels "$SEED_DUELS"
  --imitation-only-updates 1000 --imitation-value-weight 0.5 --ppo-epochs 4
  --eval-every "$SEED_EVERY" --eval-suite "$SEED_SUITE" --seed 2012
)

# Train up to version $1 - 1, from wherever the run stands.
train_to() {
  local latest remaining
  if ls "$SEED_RUN"/checkpoints/model-*.pt > /dev/null 2>&1; then
    latest=$(basename "$(ls "$SEED_RUN"/checkpoints/model-*.pt | sort | tail -1)" .pt)
    remaining=$(($1 - 1 - 10#${latest#model-}))
    if ((remaining > 0)); then
      uv run python -m ygo_training train "${TRAIN[@]}" --updates "$remaining" --continue
    fi
  else
    uv run python -m ygo_training train "${TRAIN[@]}" --updates "$1"
  fi
}

# The seed's evaluations beside its pilot's reference. `plateau`: whether the
# last two are on average no more than SEED_FLAT above the two before.
# `short`: whether they are more than SEED_SHORT under the pilot.
seed_is() {
  uv run python - "$1" "$SEED_RUN/metrics.jsonl" "$SEED_OUT/pilot-reference.json" \
    "$SEED_FLAT" "$SEED_SHORT" << 'PY'
import json
import sys


def rate(suite):
    cases = [case for case in suite.values() if isinstance(case, dict) and "games" in case]
    return sum(case["wins"] for case in cases) / sum(case["games"] for case in cases)


question, metrics, reference, flat, short = sys.argv[1:]
rates = []
for line in open(metrics, encoding="utf8"):
    suite = json.loads(line).get("evaluation_suite")
    if suite:
        rates.append(rate(suite))
pilot = rate(json.load(open(reference, encoding="utf8"))["records"][0]["evaluation_suite"])
if question == "plateau":
    print(
        "held-out evaluations of the seed:",
        " ".join(f"{value:.1%}" for value in rates),
        f"(its pilot {pilot:.1%})",
    )
    if len(rates) < 4:
        raise SystemExit(1)
    raise SystemExit(0 if sum(rates[-2:]) / 2 - sum(rates[-4:-2]) / 2 <= float(flat) else 1)
level = sum(rates[-2:]) / 2
if pilot - level > float(short):
    print(f"The seed levels off at {level:.1%}, {100 * (pilot - level):.1f} points under its pilot")
    raise SystemExit(0)
raise SystemExit(1)
PY
}

target=$SEED_TARGET
train_to "$target"
until seed_is plateau; do
  target=$((target + SEED_STEP))
  if ((target > SEED_MAX)); then
    echo "The seed still rises at version $((target - SEED_STEP - 1)): look at it before going on" >&2
    exit 1
  fi
  train_to "$target"
done

uv run python -m ygo_training status "$SEED_RUN/metrics.jsonl" > "$SEED_OUT/status.txt"
uv run python -m ygo_training status "$SEED_RUN/metrics.jsonl" --ppo > "$SEED_OUT/status-ppo.txt"
uv run python -m ygo_training status "$SEED_RUN/metrics.jsonl" --evaluation-suite > "$SEED_OUT/status-suite.txt"
cp "$SEED_RUN/metrics.jsonl" "$SEED_OUT/metrics.jsonl"
cp "$SEED_RUN/run.json" "$SEED_OUT/run.json"
cat "$SEED_OUT/status-suite.txt"
if seed_is short >&2; then
  echo "Look at it before going on" >&2
  exit 1
fi
