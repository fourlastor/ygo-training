#!/usr/bin/env bash
# What sharing a thread's duels out among two groups does on this machine:
# one group's engines run while the model answers the other's decisions
# (--pool-groups 2, "double-buffered sampling"). A measure of speed, not an
# experiment: nothing it writes is a result. Run from the repository root,
# when nothing else runs there.
#
# experiments/commands/side-by-side.sh is made twice, with one group and
# with two: the same short run of PPO alone and COUNTS times at once, as
# processes have the card and through NVIDIA's MPS (PARTS). Then the two are
# set beside each other for each part: what an update takes in each run, the
# duels an hour in all, and whether the run alone played the same games with
# two groups as with one.
#
# OVER has the two groups go on together once their rounds hold so many
# decisions or fewer between them (--pool-groups-over): a call of the model
# costs much the same whatever it holds, and a round split in two pays only
# while it is large. 0 keeps the groups to an update's last duel.
#
# HALVES names what is made anew, "one" group and "two": the other is read
# where an earlier run left it in OUT.
#
# The reports are in OUT: one-group/, two-groups/ (or two-groups-over-OVER/),
# and plain.txt and mps.txt (or plain-over-OVER.txt and mps-over-OVER.txt),
# which set the two beside each other.
set -euo pipefail
COUNTS=${COUNTS:-1 3}
UPDATES=${UPDATES:-7}
PARTS=${PARTS:-plain mps}
OVER=${OVER:-0}
HALVES=${HALVES:-one two}
OUT=${OUT:-analysis/two-groups}
RUNS=${RUNS:-runs/two-groups}
CHECKPOINT=${CHECKPOINT:-runs/self-play/with-itself/checkpoints/model-03239.pt}
PYTHON=${PYTHON:-}
export COUNTS UPDATES PARTS CHECKPOINT PYTHON

two=two-groups
suffix=
if [ "$OVER" -gt 0 ]; then
  two=two-groups-over-$OVER
  suffix=-over-$OVER
fi
for half in $HALVES; do
  case "$half" in
    one)
      POOL_GROUPS=1 OUT=$OUT/one-group RUNS=$RUNS/one-group bash experiments/commands/side-by-side.sh
      ;;
    two)
      POOL_GROUPS=2 POOL_GROUPS_OVER=$OVER OUT=$OUT/$two RUNS=$RUNS/$two \
        bash experiments/commands/side-by-side.sh
      ;;
    *)
      echo "HALVES names one and two, not $half" >&2
      exit 1
      ;;
  esac
done

python=(uv run python)
[ -z "$PYTHON" ] || python=("$PYTHON")
for part in $PARTS; do
  if [ ! -f "$OUT/one-group/$part.json" ] || [ ! -f "$OUT/$two/$part.json" ]; then
    echo "No report of $part for both: $OUT/one-group/$part.json and $OUT/$two/$part.json" >&2
    continue
  fi
  echo "== $part: one group, two groups"
  "${python[@]}" tools/side_by_side.py "$CHECKPOINT" \
    --beside "$OUT/one-group/$part.json" "$OUT/$two/$part.json" | tee "$OUT/$part$suffix.txt"
done
echo "The reports are in $OUT"
