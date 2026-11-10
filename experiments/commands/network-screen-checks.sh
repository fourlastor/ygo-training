#!/usr/bin/env bash
# The checks with no training of the screen's second session
# (experiments/2026-11-02-expanded-pool.md): on Cyber's mirror suite, every
# deal from both seats, the screen's two copies' last checkpoints head to
# head, and cyber-tokens and cyber at 720 updates of PPO each against
# cyber-old at 720 (the Cyber champion's arm made by the same recipe at the
# same seed); then read (tools/network_screen_checks.py, into OUT/checks.txt).
# Run from the repository root, on the development machine's processor:
#
#   bash experiments/commands/network-screen-checks.sh
#
# A suite played is not played again.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
RUNS=${RUNS:-runs/network-screen}
OLD=${OLD:-runs/cyber-champion/cyber-old/checkpoints/model-00959.pt}
OUT=${OUT:-analysis/network-screen/checks}
SUITE=${SUITE:-eval_suites/cyber-against-checkpoint.json}
DEVICE=${DEVICE:-cpu}
WORKERS=${WORKERS:-8}
THREADS=${THREADS:-8}
SLOTS=${SLOTS:-16}
SEED=${SEED:-2012}
# On a processor more threads of PyTorch's own make a suite slower.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
PYTHON=${PYTHON:-}
python=(uv run python)
[ -z "$PYTHON" ] || python=("$PYTHON")
tokens="--board-representation individual-tokens"

# suite FILE FIRST SECOND OPTION...: FIRST in the learner's seat against
# SECOND, told what the options say of FIRST, into OUT/FILE.json.
suite() {
  local file=$1 one=$2 other=$3
  shift 3
  if [ ! -f "$OUT/$file.json" ]; then
    "${python[@]}" -m ygo_training evaluate-suite "$one" --opponent-checkpoint "$other" "$@" \
      --device "$DEVICE" --workers "$WORKERS" --pool-threads "$THREADS" --pool-slots "$SLOTS" \
      --seed "$SEED" --suite "$SUITE" --output "$OUT/$file.json.part" > /dev/null
    mv "$OUT/$file.json.part" "$OUT/$file.json"
    echo "$(date +%H:%M) $file"
  fi
}
# pair FOLDER STEM FIRST SECOND [FIRST'S [SECOND'S]]: FIRST in the learner's
# seat against SECOND, then the other way round, as champions.sh pairs them.
pair() {
  local folder=$1 stem=$2 one=$3 other=$4 ones=() others=()
  read -r -a ones <<< "${5:-}"
  read -r -a others <<< "${6:-}"
  mkdir -p "$OUT/$folder"
  suite "$folder/$stem.plays" "$one" "$other" "${ones[@]}"
  suite "$folder/$stem.held" "$other" "$one" "${others[@]}"
}

for path in "$OLD" "$SUITE" "$RUNS/cyber-copy/checkpoints/model-00239.pt" \
  "$RUNS/cyber-tokens-copy/checkpoints/model-00239.pt" \
  "$RUNS/cyber/checkpoints/model-00959.pt" "$RUNS/cyber-tokens/checkpoints/model-00959.pt"; do
  if [ ! -f "$path" ]; then
    echo "Missing $path" >&2
    exit 1
  fi
done
pair copies 240 "$RUNS/cyber-tokens-copy/checkpoints/model-00239.pt" \
  "$RUNS/cyber-copy/checkpoints/model-00239.pt" "$tokens"
pair cyber-tokens-against-cyber-old 720 "$RUNS/cyber-tokens/checkpoints/model-00959.pt" "$OLD" \
  "$tokens"
pair cyber-against-cyber-old 720 "$RUNS/cyber/checkpoints/model-00959.pt" "$OLD"
"${python[@]}" tools/network_screen_checks.py "$OUT" --output "$OUT/checks.json" \
  | tee "$OUT/checks.txt"
