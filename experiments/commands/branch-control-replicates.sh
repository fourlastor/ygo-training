#!/usr/bin/env bash
# Replicate control branches: same warm-up seed and evaluation games, different
# training randomness. Run from training/.
set -u
cd "$(dirname "$0")/../.."
mkdir -p analysis/replicates
for seed in 2013 2014 2015; do
    name="setcodes-branch-control-t${seed}"
    echo "$(date -Is) start ${name}" | tee -a analysis/replicates/queue.log
    if bash "experiments/commands/${name}.txt" > "analysis/replicates/${name}.log" 2>&1; then
        echo "$(date -Is) done ${name}" | tee -a analysis/replicates/queue.log
    else
        echo "$(date -Is) FAILED ${name} (exit $?)" | tee -a analysis/replicates/queue.log
    fi
done
