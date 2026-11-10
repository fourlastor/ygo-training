#!/usr/bin/env bash
# The second session of the screen of the network under PPO
# (experiments/2026-11-02-expanded-pool.md): Blackwing from scratch by the
# champions' recipe as the network is and with the board as its cards
# (blackwing, blackwing@tokens, to 720 updates of PPO); the screen's Cyber
# control and tokens runs gone on from 720 to 1,440 updates of PPO, and the
# control at 720 given the board as its cards from there on (cyber,
# cyber@tokens, cyber@switched). The five go at once
# (experiments/commands/at-once.sh, through MPS on a card); then each deck's
# runs are measured as champions.sh measures runs of a shape, the two decks at
# once, and read (tools/network_screen_report.py, into OUT/DECK/screen.txt).
# Run from the repository root on a rented card, with the screen's Cyber runs
# in RUNS (the copies' last checkpoints and the runs' at 720 updates of PPO,
# with their run.json and metrics) and what the benchmark's setup leaves
# there:
#
#   AT_ONCE=5 bash experiments/commands/rented-setup.sh \
#     bash experiments/commands/network-screen-two.sh
#
# THREADS is the machine's share of threads, as the setup leaves it; a
# Blackwing run, which copies as well, takes 4 of 17 of them
# (BLACKWING_THREADS) and a Cyber run a third of the rest (CYBER_THREADS),
# each with the pool's slots to hold the PPO stage's IN_FLIGHT duels, and a
# deck's measure half of them (MEASURE_THREADS). A run that was stopped goes
# on from its last checkpoint when this is started again, and a measure made
# is not made again.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export RUNS=${RUNS:-runs/network-screen}
export OUT=${OUT:-analysis/network-screen-two}
export DEVICE=${DEVICE:-cuda}
LOGS=${LOGS:-$RUNS/logs-two}
total=${THREADS:-36}
BLACKWING_THREADS=${BLACKWING_THREADS:-$((total * 4 / 17))}
CYBER_THREADS=${CYBER_THREADS:-$(((total - 2 * BLACKWING_THREADS) / 3))}
MEASURE_THREADS=${MEASURE_THREADS:-$((total / 2))}
IN_FLIGHT=${IN_FLIGHT:-1040}
# The copy's updates and the screen's of PPO, Blackwing's too; the Cyber
# runs go on to LONGER, the switched one from the control's at UPDATES.
export COPY=${COPY:-240}
UPDATES=${UPDATES:-720}
LONGER=${LONGER:-1440}
export IN_FLIGHT SWITCHED=${SWITCHED:-$UPDATES}
PYTHON=${PYTHON:-}
python=(uv run python)
[ -z "$PYTHON" ] || python=("$PYTHON")
# shellcheck source=experiments/commands/at-once.sh
. experiments/commands/at-once.sh

# slots THREADS: a thread's share of the duels in flight, an even count.
slots() {
  local each=$(((IN_FLIGHT + $1 - 1) / $1))
  echo $((each + each % 2))
}
for run in cyber cyber-tokens; do
  for path in "$(printf '%s/%s-copy/checkpoints/model-%05d.pt' "$RUNS" "$run" $((COPY - 1)))" \
    "$(printf '%s/%s/checkpoints/model-%05d.pt' "$RUNS" "$run" $((COPY + UPDATES - 1)))" \
    "$RUNS/$run/run.json"; do
    if [ ! -f "$path" ]; then
      echo "Missing $path: the screen's Cyber runs go on from where they stand" >&2
      exit 1
    fi
  done
done
echo "Blackwing runs on $BLACKWING_THREADS threads each, Cyber runs on $CYBER_THREADS"
for arm in blackwing blackwing@tokens; do
  at_once "${arm/@/-}" env RUNNING="$arm" PARTS=arms UPDATES="$UPDATES" \
    THREADS="$BLACKWING_THREADS" FAST_SLOTS="$(slots "$BLACKWING_THREADS")" \
    bash experiments/commands/champions.sh
done
for arm in cyber cyber@tokens cyber@switched; do
  at_once "${arm/@/-}" env RUNNING="$arm" PARTS=arms UPDATES="$LONGER" \
    THREADS="$CYBER_THREADS" FAST_SLOTS="$(slots "$CYBER_THREADS")" \
    bash experiments/commands/champions.sh
done
all_done
at_once measure-blackwing env RUNNING="blackwing blackwing@tokens" PARTS=measure \
  UPDATES="$UPDATES" OUT="$OUT/blackwing" THREADS="$MEASURE_THREADS" \
  bash experiments/commands/champions.sh
at_once measure-cyber env RUNNING="cyber cyber@tokens cyber@switched" PARTS=measure \
  UPDATES="$LONGER" OUT="$OUT/cyber" THREADS="$MEASURE_THREADS" \
  bash experiments/commands/champions.sh
all_done
"${python[@]}" tools/network_screen_report.py "$OUT/blackwing" blackwing blackwing-tokens \
  --output "$OUT/blackwing/screen.json" | tee "$OUT/blackwing/screen.txt"
"${python[@]}" tools/network_screen_report.py "$OUT/cyber" cyber cyber-tokens cyber-switched \
  --output "$OUT/cyber/screen.json" | tee "$OUT/cyber/screen.txt"
echo "The second session is read: $OUT/blackwing/screen.txt, $OUT/cyber/screen.txt"
