#!/usr/bin/env bash
# Runs the three corrected-engine baselines one after another, from training/.
# A failed run is logged and the queue continues with the next one.
set -u
cd "$(dirname "$0")/../.."
mkdir -p analysis/overnight
for name in setcodes-baseline setcodes-baseline-monarch setcodes-baseline-spellcaster; do
    log="analysis/overnight/${name}.log"
    echo "$(date -Is) start ${name}" | tee -a analysis/overnight/queue.log
    if bash "experiments/commands/${name}.txt" > "${log}" 2>&1; then
        echo "$(date -Is) done ${name}" | tee -a analysis/overnight/queue.log
    else
        echo "$(date -Is) FAILED ${name} (exit $?), see ${log}" | tee -a analysis/overnight/queue.log
    fi
done
