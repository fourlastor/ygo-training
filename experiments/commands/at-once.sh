# shellcheck shell=bash
# Runs of a script that go at the same time. Sourced by the scripts whose
# trainings do not wait for each other. AT_ONCE says how many go at a time;
# with 1, the default, they go one after the other in the terminal, as ever.
#
# On a card (DEVICE) more than one go through NVIDIA's MPS, which lets
# several processes work on the card at the same time where they otherwise
# take it by turns: three runs of 520 duels, half of them against itself,
# took 23.0 s an update each through it where one alone took 15.5 s, twice
# the duels an hour of the one (docs/commands.md, its section on calls from
# CUDA graphs).
# Each run is as long as it would be with the card to itself and half as
# long again; what is gained is the runs that are made beside it.
#
# The daemon is started here and told to quit when the runs are over, or the
# script is. Its control program comes with the card's driver
# (nvidia-cuda-mps-control); where it is not there, or with MPS=0, the runs
# go at once all the same and take the card by turns. What the daemon and
# its server wrote is kept in LOGS/mps.
#
# Ctrl-C stops the runs that are going as it stops a training in the
# terminal: each where it stands, to be taken on from its last checkpoint
# when the script is started again. A second one stops them at once, and
# the script's end while they go stops them too. Each run is a group of
# processes of its own for that, told by the script: what bash 5.3 puts in
# the background of a script does not hear Ctrl-C (5.2's did), and the runs
# would go on without it.
#
# A run that fails does not stop the others: the script stops when all of
# them are over, and names it. Under MPS a process that the card faults can
# take the server and the other runs with it; a script that takes its runs
# on from where they stand is then started again.
#
#   at_once NAME COMMAND...   COMMAND, once fewer than AT_ONCE are going:
#                             with more than one at a time, in the
#                             background, what it writes added to
#                             LOGS/NAME.log
#   all_done                  waits for them, and fails where one did
#
# The script that sources this names DEVICE and LOGS.

AT_ONCE=${AT_ONCE:-1}
MPS=${MPS:-1}
if ! [[ "$AT_ONCE" =~ ^[1-9][0-9]*$ ]]; then
  echo "AT_ONCE is how many runs go at a time, one at least, not $AT_ONCE" >&2
  exit 1
fi
at_once_pids=()
at_once_names=()
at_once_served=

# Tell the daemon to quit, keep what it and its server wrote, and remove its
# folder under /tmp.
at_once_quit() {
  [ -n "$at_once_served" ] || return 0
  echo quit | nvidia-cuda-mps-control > /dev/null 2>&1 || true
  mkdir -p "$LOGS/mps"
  cp "$CUDA_MPS_LOG_DIRECTORY"/*.log "$LOGS/mps/" 2> /dev/null || true
  rm -rf "$at_once_served"
  at_once_served=
  unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
}

# at_once_signal SIGNAL: to every process of the runs that are going.
at_once_signal() {
  local pid
  for pid in "${at_once_pids[@]}"; do
    kill -s "$1" -- "-$pid" 2> /dev/null || true
  done
}

# Stop the runs that are going, wait for them, and tell the daemon to quit:
# on Ctrl-C, or when the script ends before them. A run that has not stopped
# in half a minute (one started where Ctrl-C was ignored from the first,
# which no shell can undo) is stopped at once.
at_once_stop() {
  local waited=0
  if [ -n "$(jobs -rp)" ]; then
    echo "$(date '+%H:%M') Stopping ${at_once_names[*]}: Ctrl-C again stops them at once" >&2 || true
    trap 'at_once_signal TERM' INT TERM HUP
    at_once_signal INT
    while [ -n "$(jobs -rp)" ] && ((waited < 60)); do
      sleep 0.5 || true
      waited=$((waited + 1))
    done
    [ -z "$(jobs -rp)" ] || at_once_signal TERM
    while [ -n "$(jobs -rp)" ]; do
      wait || true
    done
  fi
  at_once_pids=()
  at_once_names=()
  at_once_quit
}

# An MPS server for the runs that follow, where there is a card and the
# daemon's control program.
at_once_serve() {
  [ -z "$at_once_served" ] || return 0
  if [[ "$DEVICE" != cuda* ]] || [ "$MPS" = 0 ]; then
    return 0
  fi
  if ! command -v nvidia-cuda-mps-control > /dev/null; then
    echo "No nvidia-cuda-mps-control on the PATH: the runs take the card by turns"
    return 0
  fi
  # Short paths: the daemon's pipes are sockets, whose names have a limit.
  at_once_served=$(mktemp -d /tmp/ygo-mps.XXXXXX)
  export CUDA_MPS_PIPE_DIRECTORY=$at_once_served/pipe CUDA_MPS_LOG_DIRECTORY=$at_once_served/log
  mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
  nvidia-cuda-mps-control -d
  echo "The runs go through an MPS server ($at_once_served)"
}

at_once() {
  local name=$1
  shift
  if ((AT_ONCE == 1)); then
    "$@"
    return
  fi
  at_once_serve
  trap at_once_stop EXIT
  trap 'at_once_stop; exit 130' INT TERM HUP
  while (($(jobs -rp | wc -l) >= AT_ONCE)); do
    sleep 2
  done
  mkdir -p "$LOGS"
  echo "$(date '+%H:%M') $name begins: tail -f $LOGS/$name.log"
  # With job control the run is a group of its own, which hears Ctrl-C.
  set -m
  ("$@") < /dev/null >> "$LOGS/$name.log" 2>&1 &
  set +m
  at_once_pids+=($!)
  at_once_names+=("$name")
}

all_done() {
  local index failed=()
  for index in "${!at_once_pids[@]}"; do
    if wait "${at_once_pids[$index]}"; then
      echo "$(date '+%H:%M') ${at_once_names[$index]} is over"
    else
      failed+=("${at_once_names[$index]}")
      echo "$(date '+%H:%M') ${at_once_names[$index]} FAILED: $LOGS/${at_once_names[$index]}.log" >&2
      tail -n 5 "$LOGS/${at_once_names[$index]}.log" >&2 || true
    fi
  done
  at_once_pids=()
  at_once_names=()
  at_once_quit
  trap - EXIT INT TERM HUP
  if ((${#failed[@]})); then
    echo "Failed: ${failed[*]}. Started again, the script takes its runs on from where they stand." >&2
    return 1
  fi
}
