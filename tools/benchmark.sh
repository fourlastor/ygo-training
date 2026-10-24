#!/usr/bin/env bash
# How many duels an hour this machine trains, in every mode of the trainer,
# starting from nothing: no checkpoint, no recorded duels and no Python
# environment are needed beforehand. A measure of speed, not an experiment:
# nothing it writes is a result.
#
#     bash tools/benchmark.sh
#
# It sets itself up, does a fixed exercise, writes a report and stops.
# Everything it installs and writes goes to one folder (FOLDER, `benchmark`
# under the repository root), but for the engine's build, which stays where
# every command looks for it (`target/` and `native/ygo-policies/target/`).
#
# 1. Setup, each step only where it is missing:
#    - the submodules (`git submodule update --init --recursive`);
#    - a Rust toolchain, installed by rustup into FOLDER when `cargo` is not
#      on the PATH. A C++ compiler cannot be installed from here: it is
#      asked for;
#    - the engine, the environment and the bench (`cargo build --release`);
#    - a Python environment in FOLDER/venv, made with `python -m venv` and
#      pip, with the torch that `uv.lock` names (TORCH names another). uv is
#      not used, and where the system's Python comes without pip (Debian and
#      Ubuntu, until python3-venv is installed) pip's own installer is
#      fetched for the environment. PYTHON names an interpreter to use
#      instead, one that has torch, numpy and rich: `.venv/bin/python` where
#      uv made one.
#    What it fetches: Rust's installer and toolchain (rustup.rs), the
#    crates of the build (crates.io), pip's installer (bootstrap.pypa.io)
#    and the packages (pypi.org), 2.75 GB of them: torch with its libraries
#    for the card.
# 2. The exercise, the same games on every machine (DUELS duels an update,
#    the deals of seed 2012):
#    - a network imitates the Blackwing pilot from nothing for WARM updates.
#      That is the first mode's measure, and it gives the others a player
#      whose games are of ordinary length: after 40 such updates a copy wins
#      about half of the suite in games of 12 to 13 turns, where a new
#      network's games are not games anyone trains on;
#    - recorded duels for the modes that imitate them: the pilot with the
#      cheapest search there is, against each other deck;
#    - TIMED updates in each other mode, on from that copy, after one update
#      that warms up and is not timed;
#    - the copy measured by its first choice: the suite against the pilots,
#      and the mirror against itself held in the other seat.
# 3. The report (tools/benchmark_report.py): for every mode what an update
#    takes and where its seconds go, duels an hour and decisions a second.
#    It is FOLDER/report.txt and report.json, and FOLDER/STAMP holds what it
#    was made from. The exercise's checkpoints are removed at the end
#    (KEEP=1 keeps them).
#
# ONLY=setup makes the setup and stops: FOLDER/setup.sh then holds what
# another command needs of it, to be read by that command's shell
# (`. FOLDER/setup.sh`): the interpreter (PYTHON), the device, the threads
# this machine may run at once (THREADS) and, where Rust was installed here,
# the way to cargo.
#
# A mode that fails is told in the report and the others go on. About
# twenty-five minutes on a machine like the present CUDA one by reckoning,
# after a setup of a few minutes the first time; FOLDER comes to 5.4 GB and
# the build to 2.8 GB.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT=$PWD
FOLDER=${FOLDER:-benchmark}
case "$FOLDER" in
  /*) ;;
  *) FOLDER=$ROOT/$FOLDER ;;
esac
DEVICE=${DEVICE:-}
# The threads this machine may run at once. A container is held to a share
# of the processor that nproc does not tell: a rented one saw 128 threads
# and was allowed 13.6 of them.
share() {
  local seen quota=0 period=1 held
  seen=$(nproc)
  if [ -r /sys/fs/cgroup/cpu.max ]; then
    read -r quota period < /sys/fs/cgroup/cpu.max
    [ "$quota" != max ] || quota=0
  elif [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]; then
    quota=$(< /sys/fs/cgroup/cpu/cpu.cfs_quota_us)
    period=$(< /sys/fs/cgroup/cpu/cpu.cfs_period_us)
  fi
  held=$(((quota + period - 1) / period))
  echo $((quota > 0 && held < seen ? held : seen))
}
SHARE=$(share)
THREADS=${THREADS:-$SHARE}
if ((SHARE < $(nproc))); then
  # Torch and cargo count the processor's threads for themselves, and would
  # start more than the share.
  export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$SHARE} CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-$SHARE}
fi
SLOTS=${SLOTS:-16}
WORKERS=${WORKERS:-24}
DUELS=${DUELS:-520}
WARM=${WARM:-40}
TIMED=${TIMED:-4}
RATE=${RATE:-1e-4}
PYTHON=${PYTHON:-}
TORCH=${TORCH:-}
KEEP=${KEEP:-}
OTHERS=${OTHERS:-burn,crystal,dragunity,gishki,gladiator,heroes,infernity,lightsworn,monarch,morphtronic,rock-block,spellcaster}
OPPONENTS=${OPPONENTS:-blackwing,$OTHERS}
DECKS=${DECKS:-blackwing,lightsworn}
SUITE=${SUITE:-eval_suites/ygo-policies-blackwing-screen.json}
AGAINST=${AGAINST:-eval_suites/blackwing-against-checkpoint.json}

if ((WARM < 2 || TIMED < 1)); then
  echo "WARM is two updates at least and TIMED one: the first update of a run warms up and is not timed" >&2
  exit 1
fi
have() { command -v "$1" > /dev/null 2>&1; }

# The Python an environment is made with: the newest on the PATH of those
# the trainer runs on.
base=
if [ -z "$PYTHON" ] && [ ! -x "$FOLDER/venv/bin/python" ]; then
  for candidate in python3.13 python3.12 python3.11 python3; do
    if have "$candidate" && "$candidate" -c \
      'import sys; sys.exit(not (3, 11) <= sys.version_info[:2] <= (3, 13))'; then
      base=$candidate
      break
    fi
  done
fi

# What cannot be brought from here is asked for before anything is made.
missing=()
if ! git rev-parse --git-dir > /dev/null 2>&1; then
  missing+=("a git checkout of the repository: its submodules are fetched by git")
fi
if ! have c++ && ! have g++ && ! have clang++; then
  missing+=("a C++ compiler, to build the engine (Debian or Ubuntu: sudo apt install build-essential)")
fi
if [ -z "$PYTHON" ] && [ ! -x "$FOLDER/venv/bin/python" ] && [ -z "$base" ]; then
  missing+=("Python 3.11, 3.12 or 3.13 on the PATH, or PYTHON=/path/to/python of one that has torch, numpy and rich")
fi
if ! have curl; then
  if ! have cargo && [ ! -x "$FOLDER/cargo/bin/cargo" ]; then
    missing+=("curl, to fetch Rust's installer; or Rust itself (https://rustup.rs)")
  elif [ -n "$base" ] && ! "$base" -c 'import ensurepip' > /dev/null 2>&1; then
    missing+=("curl, to fetch pip's installer; or the venv package of $base (Debian or Ubuntu: sudo apt install python3-venv)")
  fi
fi
if ((${#missing[@]})); then
  echo "This machine lacks what the benchmark cannot install by itself:" >&2
  printf '  - %s\n' "${missing[@]}" >&2
  exit 1
fi

STAMP=$(date +%Y%m%d-%H%M%S)
RUNS=$FOLDER/$STAMP
mkdir -p "$RUNS"
began=$(date +%s)
failed=()

# stage NAME COMMAND...: a step of the setup, with its seconds for the report.
stage() {
  local name=$1 start
  shift
  start=$(date +%s)
  echo "== $name"
  "$@"
  printf '%s\t%s\n' "$name" $(($(date +%s) - start)) >> "$RUNS/setup.tsv"
}

submodules() {
  local pinned
  pinned=$(git submodule status --recursive)
  if grep -q '^-' <<< "$pinned"; then
    git submodule update --init --recursive
    pinned=$(git submodule status --recursive)
  fi
  if grep -q '^[^ ]' <<< "$pinned"; then
    echo "A submodule is not at its pinned revision: git submodule update --init --recursive" >&2
    exit 1
  fi
}

toolchain() {
  if ! have cargo; then
    export RUSTUP_HOME=$FOLDER/rustup CARGO_HOME=$FOLDER/cargo
    if [ ! -x "$CARGO_HOME/bin/cargo" ]; then
      curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --no-modify-path --profile minimal
    fi
    export PATH=$CARGO_HOME/bin:$PATH
  fi
  cargo --version
}

build() {
  cargo build --release
  (cd native/ygo-policies && cargo build --release -p ygo-policies-bench -p ygo-policies-ffi)
}

environment() {
  if [ -z "$PYTHON" ]; then
    if [ ! -x "$FOLDER/venv/bin/python" ]; then
      if ! "$base" -m venv "$FOLDER/venv" > /dev/null 2>&1; then
        # A Python without ensurepip makes an environment without pip.
        "$base" -m venv --clear --without-pip "$FOLDER/venv"
      fi
    fi
    PYTHON=$FOLDER/venv/bin/python
    if ! "$PYTHON" -m pip --version > /dev/null 2>&1; then
      curl --proto '=https' --tlsv1.2 -sSf https://bootstrap.pypa.io/get-pip.py -o "$FOLDER/get-pip.py"
      "$PYTHON" "$FOLDER/get-pip.py" --no-cache-dir
    fi
    if ! "$PYTHON" -c 'import torch, numpy, rich' > /dev/null 2>&1; then
      if [ -z "$TORCH" ]; then
        TORCH=$("$PYTHON" -c '
import tomllib
with open("uv.lock", "rb") as lock:
    packages = tomllib.load(lock)["package"]
print(next(package["version"] for package in packages if package["name"] == "torch"))')
      fi
      "$PYTHON" -m pip install --no-cache-dir "torch==$TORCH" numpy rich
    fi
  fi
  if ! "$PYTHON" -c 'import torch, numpy, rich'; then
    echo "$PYTHON lacks torch, numpy or rich" >&2
    exit 1
  fi
  if [ -z "$DEVICE" ]; then
    # A card torch names and cannot compute on is no card.
    DEVICE=$("$PYTHON" -W ignore -c '
import torch
try:
    assert torch.cuda.is_available()
    (torch.zeros(4, device="cuda") + 1).sum().item()
    print("cuda")
except Exception:
    print("cpu")')
  fi
  echo "$("$PYTHON" --version), device $DEVICE"
}

stage "submodules" submodules
stage "toolchain" toolchain
stage "build" build
stage "python" environment

if [ "${ONLY:-}" = setup ]; then
  {
    printf 'export PYTHON=%q DEVICE=%q THREADS=%q\n' "$PYTHON" "$DEVICE" "$SHARE"
    if [ "${CARGO_HOME:-}" = "$FOLDER/cargo" ]; then
      printf 'export RUSTUP_HOME=%q CARGO_HOME=%q\n' "$RUSTUP_HOME" "$CARGO_HOME"
      printf 'export PATH=%q:"$PATH"\n' "$CARGO_HOME/bin"
    fi
    if [ -n "${OMP_NUM_THREADS:-}" ]; then
      printf 'export OMP_NUM_THREADS=%q CARGO_BUILD_JOBS=%q\n' "$OMP_NUM_THREADS" "${CARGO_BUILD_JOBS:-$SHARE}"
    fi
  } > "$FOLDER/setup.sh"
  rm -f "$RUNS/setup.tsv"
  rmdir "$RUNS" 2> /dev/null || true
  echo "The setup is made, in $(($(date +%s) - began)) s: . $FOLDER/setup.sh"
  exit 0
fi

"$PYTHON" - "$RUNS/settings.json" <<EOF
import json
import sys

json.dump(
    {
        "stamp": "$STAMP",
        "device": "$DEVICE",
        "threads": $THREADS,
        "slots": $SLOTS,
        "workers": $WORKERS,
        "duels": $DUELS,
        "warm": $WARM,
        "timed": $TIMED,
        "rate": "$RATE",
        "decks": "$DECKS",
        "python": "$PYTHON",
        "cargo": "$(cargo --version)",
        "commit": "$(git rev-parse HEAD)",
    },
    open(sys.argv[1], "w", encoding="utf8"),
    indent=2,
)
EOF

# part NAME COMMAND...: a part of the exercise, its output in NAME.log. One
# that fails is told, and the others go on.
part() {
  local name=$1 start
  shift
  start=$(date +%s)
  printf '   %-22s' "$name"
  if "$@" > "$RUNS/$name.log" 2>&1; then
    echo "$(($(date +%s) - start)) s"
  else
    echo "FAILED: $RUNS/$name.log"
    failed+=("$name")
  fi
  printf '%s\t%s\n' "$name" $(($(date +%s) - start)) >> "$RUNS/exercise.tsv"
}

# train NAME OPTION...: a run of its own, one pass an update, no evaluation.
train() {
  local name=$1
  shift
  "$PYTHON" -m ygo_training train --device "$DEVICE" --workers "$WORKERS" --run "$RUNS/$name" \
    --duels "$DUELS" --eval-every 0 --seed 2012 --ppo-epochs 1 "$@"
}
pool=(--pool-threads "$THREADS" --pool-slots "$SLOTS")
one=(--learner blackwing --opponents "$OPPONENTS")
imitating=(--imitation-only-updates 1000 --imitation-value-weight 0.5 --imitation-value-positions 4)
COPY=$(printf '%s/imitation/checkpoints/model-%05d.pt' "$RUNS" $((WARM - 1)))
# On from the copy: the timed updates and the one before them.
on=(--resume "$COPY" --updates $((TIMED + 1)) --keep-checkpoints $((TIMED + 1)))
playing=("${on[@]}" --resume-learning-rate "$RATE")
RECORDS=$RUNS/records.jsonl
BENCH=native/ygo-policies/target/release/policy-bench

# An update's worth of recorded duels: as many deals against each other deck,
# an even number, the pilot's seat being each in turn.
records() {
  local others=${OTHERS//,/ }
  local decks
  decks=$(wc -w <<< "$others")
  "$BENCH" search --policies blackwing --opponents "$OTHERS" \
    --games $((2 * ((DUELS + 2 * decks - 1) / (2 * decks)))) --seed 800000 --workers "$THREADS" \
    --worlds 1 --confirm 1 --final 1 --record true --log true --output "$RECORDS"
}

measure() {
  local name=$1
  shift
  "$PYTHON" -m ygo_training evaluate-suite "$COPY" --device "$DEVICE" --workers "$WORKERS" \
    "${pool[@]}" --seed 2012 --output "$RUNS/$name.json" "$@"
}

echo "== the exercise: $RUNS"
part imitation train imitation "${one[@]}" "${pool[@]}" "${imitating[@]}" \
  --updates "$WARM" --keep-checkpoints "$WARM"
if [ ! -f "$COPY" ]; then
  tail -n 20 "$RUNS/imitation.log" >&2
  echo "The copy the other modes go on from was not made: nothing more to measure" >&2
  exit 1
fi
part imitation-decks train imitation-decks --learner "$DECKS" --opponents "$OPPONENTS" \
  "${pool[@]}" "${imitating[@]}" --updates $((TIMED + 1)) --keep-checkpoints $((TIMED + 1))
part imitation-checkpoint train imitation-checkpoint "${one[@]}" "${pool[@]}" "${imitating[@]}" \
  "${on[@]}" --imitation-teacher "$COPY"
part records records
recorded=(--learner blackwing --opponents "$OTHERS" "${pool[@]}" "${imitating[@]}" "${on[@]}"
  --imitation-records "$RECORDS")
part imitation-records train imitation-records "${recorded[@]}"
part imitation-leads train imitation-leads "${recorded[@]}" --imitation-lead-weight 0.5
part ppo train ppo "${one[@]}" "${pool[@]}" "${playing[@]}"
part ppo-anchor train ppo-anchor "${one[@]}" "${pool[@]}" "${playing[@]}" \
  --teacher-imitation-weight 0.1
part ppo-half train ppo-half "${one[@]}" "${pool[@]}" "${playing[@]}" \
  --self-play-duels $((DUELS / 2))
part ppo-itself train ppo-itself "${one[@]}" "${pool[@]}" "${playing[@]}" \
  --self-play-duels "$DUELS"
part ppo-checkpoint train ppo-checkpoint "${one[@]}" "${pool[@]}" "${playing[@]}" \
  --opponent-checkpoint "$COPY" --opponent-checkpoint-duels "$DUELS"
part suite measure suite --suite "$SUITE"
part mirror measure mirror --suite "$AGAINST" --opponent-checkpoint "$COPY"

if [ -z "$KEEP" ]; then
  # Only what this exercise made: the checkpoints of its own runs.
  rm -rf "$RUNS"/*/checkpoints
fi
printf '%s\t%s\n' "all of it" $(($(date +%s) - began)) >> "$RUNS/exercise.tsv"
echo
"$PYTHON" tools/benchmark_report.py "$RUNS" --output "$RUNS/report.json" | tee "$RUNS/report.txt"
cp "$RUNS/report.txt" "$RUNS/report.json" "$FOLDER/"
echo
echo "The report: $FOLDER/report.txt"
if ((${#failed[@]})); then
  echo "Failed: ${failed[*]}" >&2
  exit 1
fi
