#!/usr/bin/env bash
# How cards work: a head that predicts what the engine does after an answer
# (experiments/2026-09-08-engine-transition-prediction.md). Run from the
# repository root on the machine that holds the run of seventy-one decks
# (SEVENTY_ONE, made by experiments/commands/seventy-one-decks.sh). PARTS
# names what is made, in its order:
#
# probe. The head taught alone on all71's copy, its network held as it is,
# on the games of PROBE updates of all71's PPO from it, and read on games it
# never trained on against the frequencies (tools/card_head.py probe), into
# OUT/probe: what the copy's network already tells of the cards. The run's
# head begins as the probe leaves it.
#
# weight. On the games of the copy's first update, the size of PPO's
# gradient on the parameters it shares with the head and of the probe's
# head's: the weight at which the head's is as large as PPO's
# (OUT/weight.json). WEIGHT, where it is set, is taken instead.
#
# check. CHECK updates of PPO on from all71's copy at this code, without the
# head, as all71 went on from it, into RUNS/check; each update's
# fingerprints of every decision of its games and of what it trains on are
# set beside all71's own, and it stops where one differs.
#
# arm. Under RUNS, the run cards: PPO on from all71's copy for UPDATES
# updates as all71 went on from it, its settings and seed, with the head
# taught beside it (train --card-head, at the weight, begun as the probe
# left it); a checkpoint every STEP, the head kept beside each. A finished
# run is skipped and an interrupted run goes on.
#
# read. The first reading: the arm's last checkpoint and the head beside it
# on games far from the run's, against the frequencies counted on games of
# its own (tools/card_head.py read), into OUT/reading-1.json.
#
# measure. Into OUT: the arm by its checkpoints after the last three STEP of
# UPDATES updates of PPO, on the screen of each deck of THIRTEEN and on OLD
# and NEW, as seventy-one decks measured all71 there; all71 there as BASE
# measured it at this code (the critic's run), or measured into OUT where
# BASE does not hold it. Then the report
# (tools/privileged_critic_run_report.py --arm cards), which takes both
# readings. A finished measurement is not made again.
#
# About seven to eight hours for the arm by reckoning, an hour for the probe
# and the first reading, and an hour to measure.
#   bash experiments/commands/card-head.sh
set -euo pipefail
DEVICE=${DEVICE:-cuda}
WORKERS=${WORKERS:-24}
THREADS=${THREADS:-20}
SLOTS=${SLOTS:-16}
DUELS=${DUELS:-520}
PARTS=${PARTS:-probe weight check arm read measure}
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
CHECK=${CHECK:-3}
PROBE=${PROBE:-40}
WEIGHT=${WEIGHT:-}
SEVENTY_ONE=${SEVENTY_ONE:-runs/seventy-one-decks}
RUNS=${RUNS:-runs/card-head}
OUT=${OUT:-analysis/card-head/run}
# Where all71 at R1 was measured at this code: by the critic's run.
BASE=${BASE:-analysis/privileged-critic/run}
SCREENS=${SCREENS:-eval_suites/ygo-policies-%s-screen.json}
OLD=${OLD:-eval_suites/ygo-policies-unseen-played.json}
NEW=${NEW:-eval_suites/ygo-policies-forty-played.json}
# The interpreter of a machine that has no uv: the benchmark's setup leaves one.
PYTHON=${PYTHON:-}

for part in $PARTS; do
  case $part in
    probe | weight | check | arm | read | measure) ;;
    *)
      echo "PARTS names probe, weight, check, arm, read and measure, not $part" >&2
      exit 1
      ;;
  esac
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
described="$SEVENTY_ONE/all71/run.json"
for path in "$copy" "$described"; do
  if [ ! -f "$path" ]; then
    echo "Missing $path, of seventy-one decks" >&2
    exit 1
  fi
done
pinned=$(git submodule status --recursive)
if grep -q '^[^ ]' <<< "$pinned"; then
  echo "A submodule is not at its pinned revision: git submodule update --init --recursive" >&2
  exit 1
fi
python=(uv run python)
[ -z "$PYTHON" ] || python=("$PYTHON")
cargo build --release
mkdir -p "$RUNS" "$OUT"
tools=(--device "$DEVICE" --threads "$THREADS" --slots "$SLOTS")
last=$(printf '%s/cards/checkpoints/model-%05d.pt' "$RUNS" $((copied + UPDATES - 1)))

# made NAME UPDATES OPTION...: the run NAME of DECKS from all71's copy to
# UPDATES updates of PPO as all71 went on from it, from where it stands or
# from the copy, a checkpoint kept every STEP.
made() {
  local name=$1 updates=$2 last latest
  shift 2
  last=$(printf '%s/checkpoints/model-%05d.pt' "$RUNS/$name" $((copied + updates - 1)))
  local arguments=(
    --device "$DEVICE" --workers "$WORKERS" --pool-threads "$THREADS" --pool-slots "$SLOTS"
    --run "$RUNS/$name" --opponents "$DECKS" --duels "$DUELS" --eval-every 0
    --keep-checkpoints "$STEP" --seed "$SEED" --ppo-epochs 1 --learner "$DECKS"
    --self-play-duels "$ITSELF" --rollout-estimates kept "$@"
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

if [[ " $PARTS " == *" probe "* ]] && [ ! -f "$OUT/probe/probe.json" ]; then
  "${python[@]}" tools/card_head.py probe "$copy" --run-json "$described" "${tools[@]}" \
    --updates "$PROBE" --output "$OUT/probe" | tee "$OUT/probe.txt"
fi

if [[ " $PARTS " == *" weight "* ]] && [ -z "$WEIGHT" ] && [ ! -f "$OUT/weight.json" ]; then
  "${python[@]}" tools/card_head.py weight "$copy" --run-json "$described" "${tools[@]}" \
    --chunk 256 --head "$OUT/probe/card-head.pt" --output "$OUT/weight.json"
fi

if [[ " $PARTS " == *" check "* ]]; then
  made check "$CHECK"
  "${python[@]}" - "$RUNS/check/metrics.jsonl" "$SEVENTY_ONE/all71/metrics.jsonl" << 'EOF'
import json
import sys

KEYS = ("rollout_decision_fingerprint", "rollout_training_fingerprint")
checked, theirs = (
    [json.loads(line) for line in open(path, encoding="utf8") if line.strip()]
    for path in sys.argv[1:]
)
by_version = {row["version"]: row for row in theirs}
differs = False
for row in checked:
    other = by_version.get(row["version"])
    if other is None:
        sys.exit(f"all71 has no update of version {row['version']}")
    same = all(row[key] == other[key] for key in KEYS)
    differs |= not same
    print(f"version {row['version']}: {'the same games' if same else 'OTHER GAMES'} as all71's")
if differs:
    sys.exit("This code does not play all71's games: the arm is not made")
EOF
fi

if [[ " $PARTS " == *" arm "* ]]; then
  if [ -z "$WEIGHT" ]; then
    WEIGHT=$("${python[@]}" -c 'import json, sys; print(json.load(open(sys.argv[1]))["weight"])' \
      "$OUT/weight.json")
  fi
  made cards "$UPDATES" --card-head --card-head-weight "$WEIGHT" \
    --card-head-begun "$OUT/probe/card-head.pt"
  echo "The run is made: $RUNS/cards."
fi

if [[ " $PARTS " == *" read "* ]] && [ ! -f "$OUT/reading-1.json" ]; then
  if [ ! -f "$last" ]; then
    echo "Missing $last: the first reading is taken when the run is made" >&2
    exit 1
  fi
  head=$(dirname "$last")/card-head-$(basename "$last" | sed 's/^model-//')
  "${python[@]}" tools/card_head.py read "$last" --run-json "$RUNS/cards/run.json" "${tools[@]}" \
    --head "$head" --output "$OUT/reading-1.json" | tee "$OUT/reading-1.txt"
fi

if [[ " $PARTS " == *" measure "* ]]; then
  mkdir -p "$OUT/screens" "$OUT/old" "$OUT/new"
  # shellcheck disable=SC2059
  screen() { printf "$SCREENS" "$1"; }
  # three FOLDER: a run's checkpoints after the last three STEP of UPDATES
  # updates of PPO, after the copy.
  three() {
    local point
    for ((point = UPDATES - 2 * STEP; point <= UPDATES; point += STEP)); do
      printf '%s/checkpoints/model-%05d.pt\n' "$1" $((copied + point - 1))
    done
  }
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
  runs=(cards)
  base=$BASE
  if [ ! -f "$BASE/old/all71-$UPDATES.json" ]; then
    runs+=("all71-$UPDATES")
    base=$OUT
  fi
  declare -A folders=([cards]="$RUNS/cards" ["all71-$UPDATES"]="$SEVENTY_ONE/all71")
  for run in "${runs[@]}"; do
    mapfile -t found < <(three "${folders[$run]}")
    for path in "${found[@]}"; do
      if [ ! -f "$path" ]; then
        echo "Missing $path: $run is measured when it is made" >&2
        exit 1
      fi
    done
    for deck in "${thirteen[@]}"; do
      suite screens "$run.$deck" "$(screen "$deck")" "${found[@]}"
    done
    suite old "$run" "$OLD" "${found[@]}"
    suite new "$run" "$NEW" "${found[@]}"
  done
  # The runs' own figures over the same updates.
  cp "$RUNS/cards/metrics.jsonl" "$OUT/cards.metrics.jsonl"
  cp "$RUNS/cards/run.json" "$OUT/cards.run.json"
  head -n "$UPDATES" "$SEVENTY_ONE/all71/metrics.jsonl" > "$OUT/all71.metrics.jsonl"
  cards=()
  [ ! -f "$OUT/reading-1.json" ] || cards=(--cards "$OUT/reading-1.json")
  "${python[@]}" tools/privileged_critic_run_report.py "$OUT" --base "$base" --arm cards \
    --thirteen "${thirteen[@]}" --copied "$copied" --point "$UPDATES" --step "$STEP" \
    "${cards[@]}" --output "$OUT/report.json" | tee "$OUT/report.txt"
  echo "The measurements and what they say are in $OUT."
fi
