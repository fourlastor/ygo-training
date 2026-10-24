#!/usr/bin/env bash
# Overlap: the pass beside the next update's games
# (experiments/2026-10-23-overlap.md): a run. Run from the repository root on
# the machine that holds the runs of seventy-one decks (SEVENTY_ONE) and of
# duels in flight (FAST_RUNS, measured into FAST_OUT by
# experiments/commands/duels-in-flight.sh). PARTS names what is made, in its
# order:
#
# trial. TRIAL updates of PPO on from all71's copy at this code with fast's
# settings, into RUNS/trial-fast: the speed the arm is set beside; and as
# many with the next update's games beside the pass, into
# RUNS/trial-overlap. A finished trial is not made again.
#
# arm. Under RUNS, the run overlap: PPO on from all71's copy for UPDATES
# updates as all71 went on from it, its settings and seed, with fast's
# settings (IN_FLIGHT duels in flight, FAST_SLOTS slots a thread,
# minibatches of MINIBATCH, the calls answered as they are in two groups)
# and the next update's games played beside the pass by a second thread; a
# checkpoint every STEP. A finished run is skipped, and an interrupted run
# goes on, its duels in flight begun again.
#
# measure. First fast's last checkpoint on the screen of CHECK again, at
# this code: its games must be those duels in flight measured, or nothing
# is measured (fast's measurements would not stand). Then into OUT: overlap
# by its checkpoints after the last three STEP of UPDATES updates of PPO, on
# the screen of each deck of THIRTEEN and on OLD and NEW. Then the report
# (tools/overlap_report.py), which takes both readings. A finished
# measurement is not made again.
#
# About 10 minutes for the trial, three hours for the arm by reckoning, and
# 50 minutes to measure.
#   bash experiments/commands/overlap.sh
set -euo pipefail
DEVICE=${DEVICE:-cuda}
WORKERS=${WORKERS:-24}
THREADS=${THREADS:-20}
SLOTS=${SLOTS:-16}
DUELS=${DUELS:-520}
PARTS=${PARTS:-trial arm measure}
THIRTEEN=${THIRTEEN:-blackwing,lightsworn,gishki,burn,crystal,dragunity,gladiator,heroes,infernity,monarch,morphtronic,rock-block,spellcaster}
# The 71 decks of seventy-one decks, in the order it named them.
DECKS=${DECKS:-$THIRTEEN,arcana,draconic-might,harpie,karakuri,pyramid,quickdraw-plant,six-samurai,tele-dad,x-saber,countdown,destiny-hero,fortune-lady,gravekeeper,machina,ojama,toon,verdict,watt,agents,alien,amazoness,batteryman,benkei,chain-burn,chaos,cloudian,counter-fairy,cyber,dark-world,deckout,demise,destiny-board,exodia,fabled,fish,flamvell,garden,gem-knight,gemini,genex,gusto,herald,ice-barrier,iron-chain,jurrac,macro-dd,malefic,naturia,nordic,nurse,psychic,reptilianne,scrap,spirit,venom,volcanic,worm,zombie}
# The updates of all71's copy, 240 a deck.
COPY=${COPY:-240}
# R1 of seventy-one decks.
UPDATES=${UPDATES:-1920}
STEP=${STEP:-120}
ITSELF=${ITSELF:-260}
RATE=${RATE:-1e-4}
SEED=${SEED:-2012}
# fast's settings: two updates' duels in flight, the pool's slots to hold
# them, minibatches of 512, the calls as they are in two groups.
IN_FLIGHT=${IN_FLIGHT:-1040}
FAST_SLOTS=${FAST_SLOTS:-52}
MINIBATCH=${MINIBATCH:-512}
TRIAL=${TRIAL:-20}
# The screen fast's last checkpoint is measured on again.
CHECK=${CHECK:-blackwing}
SEVENTY_ONE=${SEVENTY_ONE:-runs/seventy-one-decks}
FAST_RUNS=${FAST_RUNS:-runs/duels-in-flight}
FAST_OUT=${FAST_OUT:-analysis/duels-in-flight/run}
RUNS=${RUNS:-runs/overlap}
OUT=${OUT:-analysis/overlap/run}
SCREENS=${SCREENS:-eval_suites/ygo-policies-%s-screen.json}
OLD=${OLD:-eval_suites/ygo-policies-unseen-played.json}
NEW=${NEW:-eval_suites/ygo-policies-forty-played.json}
# The interpreter of a machine that has no uv: the benchmark's setup leaves one.
PYTHON=${PYTHON:-}

for part in $PARTS; do
  if [ "$part" != trial ] && [ "$part" != arm ] && [ "$part" != measure ]; then
    echo "PARTS names trial, arm and measure, not $part" >&2
    exit 1
  fi
done
IFS=, read -r -a thirteen <<< "$THIRTEEN"
IFS=, read -r -a decks <<< "$DECKS"
if ((UPDATES % STEP || UPDATES < 3 * STEP)); then
  echo "UPDATES is a multiple of STEP, three of them at least: not $UPDATES" >&2
  exit 1
fi
for suite in "$OLD" "$NEW"; do
  if [ ! -f "$suite" ]; then
    echo "Missing $suite" >&2
    exit 1
  fi
done
copied=$((${#decks[@]} * COPY))
copy=$(printf '%s/all71-copy/checkpoints/model-%05d.pt' "$SEVENTY_ONE" $((copied - 1)))
if [ ! -f "$copy" ]; then
  echo "Missing all71's copy: $copy" >&2
  exit 1
fi
pinned=$(git submodule status --recursive)
if grep -q '^[^ ]' <<< "$pinned"; then
  echo "A submodule is not at its pinned revision: git submodule update --init --recursive" >&2
  exit 1
fi
python=(uv run python)
[ -z "$PYTHON" ] || python=("$PYTHON")
cargo build --release
mkdir -p "$RUNS" "$OUT"

# made NAME UPDATES OPTION...: the run NAME of DECKS from all71's copy to
# UPDATES updates of PPO as all71 went on from it, with fast's settings,
# from where it stands or from the copy, a checkpoint kept every STEP.
made() {
  local name=$1 updates=$2 last latest
  shift 2
  last=$(printf '%s/checkpoints/model-%05d.pt' "$RUNS/$name" $((copied + updates - 1)))
  local arguments=(
    --device "$DEVICE" --workers "$WORKERS" --pool-threads "$THREADS" --pool-slots "$FAST_SLOTS"
    --run "$RUNS/$name" --opponents "$DECKS" --duels "$DUELS" --eval-every 0
    --keep-checkpoints "$STEP" --seed "$SEED" --ppo-epochs 1 --learner "$DECKS"
    --self-play-duels "$ITSELF" --rollout-estimates kept
    --duels-in-flight "$IN_FLIGHT" --minibatch "$MINIBATCH" --inference-graphs 0 --pool-groups 2
    "$@"
  )
  if ! ls "$RUNS/$name"/checkpoints/model-*.pt > /dev/null 2>&1; then
    "${python[@]}" -m ygo_training train "${arguments[@]}" \
      --resume "$copy" --resume-learning-rate "$RATE" --updates "$updates"
  elif [ ! -f "$last" ]; then
    latest=$(basename "$(ls "$RUNS/$name"/checkpoints/model-*.pt | sort | tail -1)" .pt)
    "${python[@]}" -m ygo_training train "${arguments[@]}" --continue \
      --updates $((copied + updates - 1 - 10#${latest#model-}))
  fi
}

if [[ " $PARTS " == *" trial "* ]]; then
  made trial-fast "$TRIAL"
  made trial-overlap "$TRIAL" --overlap thread
  echo "The trial is made: $RUNS/trial-fast and $RUNS/trial-overlap."
fi

if [[ " $PARTS " == *" arm "* ]]; then
  made overlap "$UPDATES" --overlap thread
  echo "The run is made: $RUNS/overlap."
fi

if [[ " $PARTS " == *" measure "* ]]; then
  mkdir -p "$OUT/check" "$OUT/screens" "$OUT/old" "$OUT/new"
  # shellcheck disable=SC2059
  screen() { printf "$SCREENS" "$1"; }
  # suite FOLDER FILE SUITE CHECKPOINT...: those checkpoints on a suite, with
  # how each game ended, into OUT/FOLDER/FILE.json.
  suite() {
    local folder=$1 file=$2 played=$3
    shift 3
    if [ ! -f "$OUT/$folder/$file.json" ]; then
      "${python[@]}" -m ygo_training evaluate-suite "$@" --device "$DEVICE" \
        --workers "$WORKERS" --pool-threads "$THREADS" --pool-slots "$SLOTS" --seed "$SEED" \
        --suite "$played" --output "$OUT/$folder/$file.json.part" > /dev/null
      mv "$OUT/$folder/$file.json.part" "$OUT/$folder/$file.json"
    fi
  }
  # fast's measurements stand only if its games are the same at this code.
  last=$(printf '%s/fast/checkpoints/model-%05d.pt' "$FAST_RUNS" $((copied + UPDATES - 1)))
  if [ ! -f "$last" ]; then
    echo "Missing $last: fast's last checkpoint" >&2
    exit 1
  fi
  suite check "fast.$CHECK" "$(screen "$CHECK")" "$last"
  "${python[@]}" - "$OUT/check/fast.$CHECK.json" "$FAST_OUT/screens/fast.$CHECK.json" \
    $((copied + UPDATES)) << 'EOF'
import json
import sys

again, before, step = sys.argv[1], sys.argv[2], int(sys.argv[3])


def games(path: str) -> dict:
    records = json.load(open(path))["records"]
    found = [record for record in records if int(record["checkpoint_step"]) == step]
    if len(found) != 1:
        raise SystemExit(f"{path}: no single checkpoint after {step} updates in all")
    return {name: case["outcomes"] for name, case in found[0]["evaluation_suite"].items()}


if games(again) != games(before):
    raise SystemExit(
        f"fast's games on the screen differ at this code ({again} against {before}): "
        "its measurements do not stand, and both runs are to be measured again"
    )
print(f"fast's games at this code are those duels in flight measured ({again})")
EOF
  points=()
  for ((point = UPDATES - 2 * STEP; point <= UPDATES; point += STEP)); do
    path=$(printf '%s/overlap/checkpoints/model-%05d.pt' "$RUNS" $((copied + point - 1)))
    if [ ! -f "$path" ]; then
      echo "Missing $path: overlap is measured when it is made" >&2
      exit 1
    fi
    points+=("$path")
  done
  for deck in "${thirteen[@]}"; do
    suite screens "overlap.$deck" "$(screen "$deck")" "${points[@]}"
  done
  suite old overlap "$OLD" "${points[@]}"
  suite new overlap "$NEW" "${points[@]}"
  # The runs' own figures: the arm's whole, fast's and the trial's.
  cp "$RUNS/overlap/metrics.jsonl" "$OUT/overlap.metrics.jsonl"
  cp "$RUNS/overlap/run.json" "$OUT/overlap.run.json"
  cp "$RUNS/trial-fast/metrics.jsonl" "$OUT/trial-fast.metrics.jsonl"
  cp "$RUNS/trial-overlap/metrics.jsonl" "$OUT/trial-overlap.metrics.jsonl"
  cp "$FAST_OUT/fast.metrics.jsonl" "$OUT/fast.metrics.jsonl"
  "${python[@]}" tools/overlap_report.py "$OUT" --base "$FAST_OUT" --thirteen "${thirteen[@]}" \
    --copied "$copied" --point "$UPDATES" --step "$STEP" --first "$TRIAL" \
    --output "$OUT/report.json" | tee "$OUT/report.txt"
  echo "The measurements and what they say are in $OUT."
fi
