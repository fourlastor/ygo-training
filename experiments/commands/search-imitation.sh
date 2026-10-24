#!/usr/bin/env bash
# The whole search-imitation experiment on one machine. Run from the
# repository root:
# 1. the seed, the pilot imitated to its plateau (format13-imitation-seed.sh);
# 2. the searched games, held out and to train on (search-imitation-games.sh);
# 3. the arms, their measurements and the report (search-imitation-arms.sh).
# Each step skips what is finished and goes on with what is not, so the
# script can be run again after an interruption. What it leaves to copy back
# is under analysis/format13-seed and analysis/search-imitation, the rows of
# the two search runs excepted: they are large and stay where they are made.
set -euo pipefail
here=$(dirname "$0")
bash "$here/format13-imitation-seed.sh"
bash "$here/search-imitation-games.sh"
bash "$here/search-imitation-arms.sh"
