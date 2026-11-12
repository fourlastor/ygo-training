#!/usr/bin/env bash
# The copy's width and rate at seventy-one decks, beside its readings
# (experiments/2026-11-02-expanded-pool.md, its section beside the batches): each copy's
# last checkpoint, as experiments/commands/copy-width-rate.sh leaves them in RUNS, head to
# head against the teacher of each of the fourteen decks in its mirror, every deal of the
# deck's suite (SUITES/DECK-against-checkpoint.json) at SEED from both seats: into
# OUT/DECK/RUN.plays.json with the copy in the learner's seat and OUT/DECK/RUN.held.json
# with the teacher there, the teachers as LIST names them. Then the report again, with
# them (tools/copy_width_rate_report.py --heads), into METRICS/report.json and .txt.
# Run from the repository root, on the CUDA machine or on the processor, whose suites
# play the same games:
#
#   DEVICE=cpu OMP_NUM_THREADS=8 AT_ONCE=2 bash experiments/commands/copy-width-rate-heads.sh
#
# A file already played is not played again.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export DEVICE=${DEVICE:-cuda}
WORKERS=${WORKERS:-8}
THREADS=${THREADS:-8}
SLOTS=${SLOTS:-16}
SEED=${SEED:-2012}
LIST=${LIST:-experiments/commands/champion-recordings.list}
SUITES=${SUITES:-eval_suites}
RUNS=${RUNS:-runs/copy-width-rate}
LAST=${LAST:-1419}
METRICS=${METRICS:-analysis/copy-width-rate}
OUT=${OUT:-$METRICS/heads}
LOGS=${LOGS:-$OUT/logs}
FOURTEEN=${FOURTEEN:-blackwing cyber lightsworn infernity gishki burn crystal dragunity gladiator heroes monarch morphtronic rock-block spellcaster}
# The report's own: its window, stretches and mark, and what it tells beside.
WINDOW=${WINDOW:-711,1420}
STRETCH=${STRETCH:-71}
MARK=${MARK:-0.0025}
REFERENCE=${REFERENCE:-analysis/seventy-one-champions/taught.metrics.jsonl}
FOURTEEN_COPY=${FOURTEEN_COPY:-analysis/fourteen-decks/taught.metrics.jsonl}
PYTHON=${PYTHON:-}
python=(uv run python)
[ -z "$PYTHON" ] || python=("$PYTHON")
# shellcheck source=experiments/commands/at-once.sh
. experiments/commands/at-once.sh

read -r -a decks <<< "$FOURTEEN"
arms=(control wide narrow slow)
# checkpoint RUN: the copy's last checkpoint.
checkpoint() {
  printf '%s/%s/checkpoints/model-%05d.pt' "$RUNS" "$1" "$LAST"
}
# teacher DECK: the run that teaches DECK.
teacher() {
  awk -v deck="$1" '$1 == deck {print $2}' "$LIST"
}
for deck in "${decks[@]}"; do
  if [ ! -f "$(teacher "$deck")" ] || [ ! -f "$SUITES/$deck-against-checkpoint.json" ]; then
    echo "No teacher of $deck or no suite of its mirror: $(teacher "$deck")" >&2
    exit 1
  fi
done
for arm in "${arms[@]}"; do
  if [ ! -f "$(checkpoint "$arm")" ]; then
    echo "No checkpoint of $arm: $(checkpoint "$arm")" >&2
    exit 1
  fi
done

# failed FILE: whether evaluate-suite, which goes on past a case that fails, left a
# case's error in FILE where its games would be.
failed() {
  grep -q '"error_type"' "$1"
}
# playing FILE SUITE LEARNER OPPONENT: the suite with LEARNER in the learner's seat
# against OPPONENT, into FILE.json, where it is not there.
playing() {
  local file=$1 suite=$2 learner=$3 opponent=$4
  if [ -f "$file.json" ] && ! failed "$file.json"; then
    return 0
  fi
  mkdir -p "$(dirname "$file")"
  "${python[@]}" -m ygo_training evaluate-suite "$learner" --opponent-checkpoint "$opponent" \
    --device "$DEVICE" --workers "$WORKERS" --pool-threads "$THREADS" --pool-slots "$SLOTS" \
    --seed "$SEED" --suite "$suite" --output "$file.json.part" > /dev/null
  if failed "$file.json.part"; then
    echo "A case of $file failed: $file.json.part" >&2
    return 1
  fi
  mv "$file.json.part" "$file.json"
  echo "$(date +%H:%M) $file"
}

for deck in "${decks[@]}"; do
  suite=$SUITES/$deck-against-checkpoint.json
  taught=$(teacher "$deck")
  for arm in "${arms[@]}"; do
    at_once "$deck-$arm-plays" playing "$OUT/$deck/$arm.plays" "$suite" "$(checkpoint "$arm")" "$taught"
    at_once "$deck-$arm-held" playing "$OUT/$deck/$arm.held" "$suite" "$taught" "$(checkpoint "$arm")"
  done
done
all_done
echo "$(date +%H:%M) the copies are played against the teachers"
mapfile -t listed < <(grep -v '^#' "$LIST" | awk 'NF {print $1}')
beside=()
[ ! -f "$REFERENCE" ] || beside+=(--reference "$REFERENCE")
[ ! -f "$FOURTEEN_COPY" ] || beside+=(--fourteen "$FOURTEEN_COPY")
"${python[@]}" tools/copy_width_rate_report.py "$METRICS" --window "$WINDOW" --stretch "$STRETCH" \
  --decks "$(IFS=,; echo "${listed[*]}")" --mark "$MARK" "${beside[@]}" --heads "$OUT" \
  --output "$METRICS/report.json" | tee "$METRICS/report.txt"
echo "The copies are read with their head to head: $METRICS/report.txt"
