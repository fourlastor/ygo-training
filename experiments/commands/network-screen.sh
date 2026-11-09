#!/usr/bin/env bash
# The screen of the network under PPO (experiments/2026-11-02-expanded-pool.md,
# its section beside the batches): CONTROL's deck by the champions' recipe from
# scratch at seed 2012 in three shapes, as the network is (cyber), the board as
# its cards (cyber@tokens) and without its own earlier answers
# (cyber@no-history), AT_ONCE of them at a time (experiments/commands/at-once.sh,
# through MPS on a card); then each measured as champions.sh measures runs of a
# shape, every variant head to head against the control, and read
# (tools/network_screen_report.py, into OUT/screen.txt). Run from the
# repository root on a rented card, with what the benchmark's setup leaves
# there:
#
#   AT_ONCE=3 bash experiments/commands/rented-setup.sh \
#     bash experiments/commands/network-screen.sh
#
# THREADS is a run's share of the machine's threads (10 unless named, three
# runs on 32); FAST_SLOTS is set from it so that the pool holds the IN_FLIGHT
# duels of the PPO stage, THREADS x FAST_SLOTS of them. What else is named of
# champions.sh's variables goes through to it as it is. A run that was stopped
# goes on from its last checkpoint when this is started again, and a measure
# made is not made again.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export RUNS=${RUNS:-runs/network-screen}
export OUT=${OUT:-analysis/network-screen}
export DEVICE=${DEVICE:-cuda}
LOGS=${LOGS:-$RUNS/logs}
CONTROL=${CONTROL:-cyber}
export THREADS=${THREADS:-10}
IN_FLIGHT=${IN_FLIGHT:-1040}
slots=$(((IN_FLIGHT + THREADS - 1) / THREADS))
export IN_FLIGHT FAST_SLOTS=${FAST_SLOTS:-$((slots + slots % 2))}
PYTHON=${PYTHON:-}
python=(uv run python)
[ -z "$PYTHON" ] || python=("$PYTHON")
arms=("$CONTROL" "$CONTROL@tokens" "$CONTROL@no-history")
# shellcheck source=experiments/commands/at-once.sh
. experiments/commands/at-once.sh

for arm in "${arms[@]}"; do
  at_once "${arm/@/-}" env RUNNING="$arm" PARTS=arms bash experiments/commands/champions.sh
done
all_done
env RUNNING="${arms[*]}" PARTS=measure bash experiments/commands/champions.sh
"${python[@]}" tools/network_screen_report.py "$OUT" "$CONTROL" "$CONTROL-tokens" \
  "$CONTROL-no-history" --output "$OUT/screen.json" | tee "$OUT/screen.txt"
echo "The screen is read: $OUT/screen.txt"
