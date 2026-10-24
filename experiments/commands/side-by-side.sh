#!/usr/bin/env bash
# What this machine does with several runs of the trainer at once: as
# processes have a card, by turns, and through NVIDIA's MPS, which lets them
# use it at the same moment. A measure of speed, not an experiment: nothing
# it writes is a result, and the runs' checkpoints are removed. Run from the
# repository root, when nothing else runs there.
#
# PARTS names what is made, in this order:
# plain   tools/side_by_side.py: the same short run of PPO alone and then
#         COUNTS times at once, ITSELF of an update's duels against itself.
# mps     The same with the runs going through an MPS server. Its control
#         daemon is started here, with its pipes and logs in a folder of its
#         own under /tmp, so that only what this script starts goes through
#         it; it is told to quit when the part ends, however it ends. The
#         daemon comes with the card's driver (nvidia-cuda-mps-control).
#
# POOL_GROUPS shares every thread's duels out among so many groups in every
# run (--pool-groups; 1 is none), which go on together once their rounds hold
# POOL_GROUPS_OVER decisions or fewer (--pool-groups-over; 0: never). GRAPHS
# has every run answer its calls of up to so many decisions from CUDA graphs
# (--inference-graphs; 0: none).
#
# PYTHON names an interpreter to use where uv is not there (a machine set
# up by tools/benchmark.sh has one in benchmark/venv), and WORKERS a run's
# worker processes, which with the pool only draw the seats.
#
# The reports go to OUT: plain.txt, mps.txt and, where both were made,
# beside.txt, which sets the two beside each other and says whether the run
# alone played the same games both ways. What the runs leave is in RUNS.
set -euo pipefail
DEVICE=${DEVICE:-cuda}
THREADS=${THREADS:-20}
SLOTS=${SLOTS:-16}
DUELS=${DUELS:-520}
ITSELF=${ITSELF:-260}
COUNTS=${COUNTS:-1 2 3}
UPDATES=${UPDATES:-5}
WORKERS=${WORKERS:-24}
POOL_GROUPS=${POOL_GROUPS:-1}
POOL_GROUPS_OVER=${POOL_GROUPS_OVER:-0}
GRAPHS=${GRAPHS:-0}
PYTHON=${PYTHON:-}
PARTS=${PARTS:-plain mps}
CHECKPOINT=${CHECKPOINT:-runs/self-play/with-itself/checkpoints/model-03239.pt}
STAMP=$(date +%Y%m%d-%H%M%S)
RUNS=${RUNS:-runs/side-by-side}/$STAMP
OUT=${OUT:-analysis/side-by-side}

for part in $PARTS; do
  case "$part" in
    plain | mps) ;;
    *)
      echo "PARTS names plain and mps, not $part" >&2
      exit 1
      ;;
  esac
done
pinned=$(git submodule status --recursive)
if grep -q '^[^ ]' <<< "$pinned"; then
  echo "A submodule is not at its pinned revision: git submodule update --init --recursive" >&2
  exit 1
fi
if [ ! -f "$CHECKPOINT" ]; then
  echo "No checkpoint at $CHECKPOINT" >&2
  exit 1
fi
if [[ " $PARTS " == *" mps "* ]] && ! command -v nvidia-cuda-mps-control > /dev/null; then
  echo "No nvidia-cuda-mps-control on the PATH: PARTS=plain goes without it" >&2
  exit 1
fi
mkdir -p "$RUNS" "$OUT"
cargo build --release
python=(uv run python)
[ -z "$PYTHON" ] || python=("$PYTHON")

# side NAME: the runs alone and at once, left in RUNS/NAME, the report in OUT.
side() {
  # shellcheck disable=SC2086
  "${python[@]}" tools/side_by_side.py "$CHECKPOINT" --itself "$ITSELF" --device "$DEVICE" \
    --duels "$DUELS" --threads "$THREADS" --slots "$SLOTS" --updates "$UPDATES" \
    --workers "$WORKERS" --groups "$POOL_GROUPS" --groups-over "$POOL_GROUPS_OVER" \
    --graphs "$GRAPHS" --counts $COUNTS --folder "$RUNS/$1" || true
  cp "$RUNS/$1"/*/report.txt "$OUT/$1.txt"
  cp "$RUNS/$1"/*/report.json "$OUT/$1.json"
}

# Tell the daemon to quit, keep what it and its server wrote, and remove
# its folder under /tmp.
quit_mps() {
  echo quit | nvidia-cuda-mps-control > /dev/null 2>&1 || true
  mkdir -p "$RUNS/mps-logs"
  cp "$CUDA_MPS_LOG_DIRECTORY"/*.log "$RUNS/mps-logs/" 2> /dev/null || true
  rm -rf "$served"
}

for part in $PARTS; do
  echo "== $part"
  case "$part" in
    plain)
      side plain
      ;;
    mps)
      # Short paths: the daemon's pipes are sockets, whose names have a limit.
      served=$(mktemp -d /tmp/ygo-mps.XXXXXX)
      export CUDA_MPS_PIPE_DIRECTORY=$served/pipe CUDA_MPS_LOG_DIRECTORY=$served/log
      mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
      nvidia-cuda-mps-control -d
      trap quit_mps EXIT
      side mps
      quit_mps
      trap - EXIT
      # A run that went through the server is a client in the daemon's log.
      clients=$(grep -c "NEW CLIENT" "$RUNS/mps-logs/control.log" 2> /dev/null || true)
      if [ "${clients:-0}" -gt 0 ]; then
        echo "The MPS daemon's log names $clients clients ($RUNS/mps-logs)" | tee -a "$OUT/mps.txt"
      else
        echo "The MPS daemon's log names no client: whether the runs went through it is to be" \
          "read in $RUNS/mps-logs" | tee -a "$OUT/mps.txt"
      fi
      unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
      ;;
  esac
done
if [ -f "$OUT/plain.json" ] && [ -f "$OUT/mps.json" ] && [[ " $PARTS " == *" mps "* ]]; then
  "${python[@]}" tools/side_by_side.py "$CHECKPOINT" --beside "$OUT/plain.json" "$OUT/mps.json" \
    | tee "$OUT/beside.txt"
fi
echo "The reports are in $OUT, and the rest in $RUNS"
