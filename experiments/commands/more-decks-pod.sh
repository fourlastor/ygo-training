#!/usr/bin/env bash
# The arms of experiments/commands/more-decks.sh on a rented pod of
# Runpod's, from this machine to the end: the checkout sent there, the arms
# started, what they keep brought here as they go, and the pod stopped and
# deleted when they have ended, or failed, and all of it is here. Run from
# the repository root of the machine the checkpoints are to be on:
#
#   RENTED=root@HOST:PORT POD=ID bash experiments/commands/more-decks-pod.sh
#
# RENTED is where the pod answers by SSH with this machine's key, and POD
# its id at Runpod. Before anything is started there, runpodctl must name
# that pod (set up with the account's key: `runpodctl doctor`), since it is
# what stops the pod at the end, and the pod's card must start.
#
# What it does, each step only where it is still to do, so that it is
# started again at any time (a terminal that closed, this machine
# restarted, the pod stopped and started again) and goes on from where
# things stand:
#   1. Sends this checkout's HEAD (tools/rented.sh send), and puts the
#      pod's runs folder on its volume (VOLUME, /workspace): a pod's own
#      disk is cleared when the pod stops, and its volume is not. What the
#      arms write is then still there after a stop or a restart of the
#      pod, and the arms go on from it.
#   2. Starts the arms there, left running, on the benchmark's setup, which
#      is made there first (experiments/commands/rented-setup.sh): AT_ONCE
#      arms at a time, six where it is not named.
#   3. Follows them (tools/rented.sh follow): every EVERY seconds what the
#      arms keep is brought to the same paths here (RUNS, and the arms' own
#      output in runs/rented), and once more when they have ended; then
#      what is here is held against what is there, file by file.
#   4. With every file kept there here as it is there, whatever came of the
#      arms: stops the pod and deletes it (`runpodctl pod stop`, `runpodctl
#      pod delete`), and asks runpodctl whether it still names the pod. It
#      ends with 0 when the arms came to their end, and with 1 when they
#      did not.
# A pod that is deleted takes what it holds with it. Where not everything
# could be brought, the pod is stopped and NOT deleted: what the arms wrote
# stays on its volume, which is paid for until the pod is deleted, and this
# ends with 3 and says so. The pod is then started again where it was
# rented, and this is run again with the address it has then: it brings
# what is there and deletes the pod. A pod without a volume is left running
# in that case, since a stop would clear what it holds. Where runpodctl
# still names the pod after it was told to delete it, this ends with 4: the
# pod is to be deleted where it was rented.
#
# LEAVE=1 leaves the pod as it is at the end, and needs no POD. What the
# arms are told there is what is named here of SIZES, DECKS, OPPONENTS,
# COPY, UPDATES, STEP, ITSELF, RATE, SEED, DUELS, WORKERS, THREADS, SLOTS,
# FLIGHT and MPS; COMMAND is what is started there, where it is not the
# arms on the benchmark's setup.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
if [ -z "${RENTED:-}" ]; then
  echo "Name where the pod answers: RENTED=root@HOST:PORT" >&2
  exit 2
fi
POD=${POD:-}
LEAVE=${LEAVE:-}
export AT=${AT:-/root/ygo}
NAME=${NAME:-more-decks}
RUNS=${RUNS:-runs/more-decks}
STEP=${STEP:-120}
AT_ONCE=${AT_ONCE:-6}
export EVERY=${EVERY:-600}
COMMAND=${COMMAND:-bash experiments/commands/rented-setup.sh bash experiments/commands/more-decks.sh}
# Where a pod's volume is, which a stop leaves; nothing for a pod without one.
VOLUME=${VOLUME-/workspace}
# Here: where the pod keeps its runs, once they are on its volume.
marker=runs/rented/$NAME.volume
if ! command -v rsync > /dev/null; then
  echo "There is no rsync here to bring what the arms keep" >&2
  exit 1
fi

if [ -z "$LEAVE" ]; then
  if [ -z "$POD" ]; then
    echo "Name the pod that is stopped and deleted at the end (POD=its id), or LEAVE=1 to leave it as it is" >&2
    exit 2
  fi
  if ! command -v runpodctl > /dev/null; then
    echo "There is no runpodctl here to stop the pod with at the end. LEAVE=1 leaves the pod as it is" >&2
    exit 1
  fi
fi
# Whether runpodctl names the pod as one that is there: by the pod's id in
# what it tells of it, whatever status the program ends with, and not as a
# pod that is terminated.
there() {
  local told
  told=$(runpodctl pod get "$POD" 2> /dev/null) || true
  grep -Eq "\"id\": ?\"$POD\"" <<< "$told" \
    && ! grep -Eiq '"(runtime|desired)Status": ?"terminated"' <<< "$told"
}
if [ -z "$LEAVE" ] && ! there; then
  echo "runpodctl does not name the pod $POD (its key: runpodctl doctor). LEAVE=1 leaves the pod as it is" >&2
  exit 1
fi
rented() { bash tools/rented.sh "$@"; }
# Whether runpodctl tells the pod as one that is stopped.
stopped() {
  runpodctl pod get "$POD" 2> /dev/null \
    | grep -Eiq '"runtimeStatus": ?"stopped"|"desiredStatus": ?"exited"'
}

# The pod's runs folder put on its volume where it is not, and where it is
# kept told: the volume's folder, or none for a pod without a volume.
link='
volume=$1
if [ -L runs ]; then readlink runs; exit 0; fi
if [ -z "$volume" ] || [ ! -d "$volume" ]; then echo none; exit 0; fi
if [ -e runs ]; then
  echo "The checkout there has a runs folder of its own: it is not put on the volume over it" >&2
  exit 1
fi
mkdir -p "$volume/ygo-runs" && ln -s "$volume/ygo-runs" runs && echo "$volume/ygo-runs"
'
keeps() {
  local where
  where=$(rented run bash -c "$link" link "$1")
  mkdir -p runs/rented
  if [ "$where" = none ]; then
    rm -f "$marker"
    echo "The pod keeps its runs on its own disk, which a stop clears: it has no volume at ${VOLUME:-any place}"
  else
    echo "$where" > "$marker"
    echo "The pod keeps its runs on its volume, which a stop leaves: $where"
  fi
}

if ! found=$(rented state "$NAME"); then
  echo "The pod does not answer at $RENTED. A pod that was stopped is started again where it was" \
    "rented, and has another address then" >&2
  exit 1
fi
if [ "$found" = none ] || [ "$found" = gone ]; then
  told=$(rented machine)
  echo "$told"
  if [ -z "${ON_THE_PROCESSOR:-}" ] && ! grep -q '^cuda: starts' <<< "$told"; then
    echo "The pod's card does not start: nothing is sent or started" >&2
    exit 1
  fi
  rented send
  keeps "$VOLUME"
  # A pod that was stopped and started again has a new disk of its own, and
  # on its volume what the arms had written: how they stand is told there.
  found=$(rented state "$NAME")
else
  # Arms that are there are left where they write: it is only asked.
  keeps ""
fi
case "$found" in
  running) echo "$NAME is running there: followed from here" ;;
  ended*) echo "$NAME has $found there: what it kept is brought" ;;
  none | gone)
    # What was brought here before and is no longer there was lost with the
    # pod's disk: runs started again from nothing would be brought over it.
    if [ -n "$(ls -A "$RUNS" 2> /dev/null)" ] && ! rented run test -d "$RUNS"; then
      echo "$RUNS is here and not on the pod, which has lost what it held. Name another RUNS to" \
        "start the arms again from nothing, or put what is here back there first" \
        "(bash tools/rented.sh put $RUNS; a run goes on from its last checkpoint once the rows" \
        "after it are taken out of its metrics.jsonl)" >&2
      exit 1
    fi
    [ "$found" = none ] || echo "$NAME stopped there and left no status: it is started again and goes on"
    told=(AT_ONCE="$AT_ONCE" RUNS="$RUNS" STEP="$STEP")
    for name in SIZES DECKS OPPONENTS COPY UPDATES ITSELF RATE SEED DUELS WORKERS THREADS SLOTS \
      FLIGHT MPS ON_THE_PROCESSOR PYTHON DEVICE; do
      [ -z "${!name:-}" ] || told+=("$name=${!name}")
    done
    # shellcheck disable=SC2086
    rented start "$NAME" env "${told[@]}" $COMMAND
    ;;
  *)
    echo "The pod does not say how $NAME stands: $found" >&2
    exit 1
    ;;
esac

# What the arms keep, and their own output.
kept="PARTS=kept RUNS=$(printf '%q' "$RUNS") STEP=$(printf '%q' "$STEP") bash experiments/commands/more-decks.sh; echo runs/rented/$NAME.log"
followed=0
rented follow "$NAME" bash -c "$kept" || followed=$?
if ((followed == 3)); then
  echo "Not everything the arms kept is here." >&2
  if [ -n "$LEAVE" ]; then
    echo "The pod is left as it is (LEAVE). Started again, this tries again" >&2
  elif [ ! -s "$marker" ]; then
    echo "The pod $POD is left as it is, RUNNING at its price: it has no volume, and a stop would" \
      "clear what it holds. Started again, this tries again" >&2
  else
    runpodctl pod stop "$POD" > /dev/null 2>&1 || true
    halted=
    for ((look = 0; look < 5; look++)); do
      if stopped; then
        halted=1
        break
      fi
      sleep "${AGAIN:-60}"
    done
    if [ -n "$halted" ]; then
      echo "The pod $POD is STOPPED and not deleted: what the arms wrote is on its volume" \
        "($(cat "$marker")), which is paid for until the pod is deleted. Start the pod again" \
        "where it was rented and run this again with the address it has then: it brings what" \
        "is there and deletes the pod" >&2
    else
      echo "runpodctl does not tell the pod $POD as stopped: look at it where it was rented, it" \
        "may be RUNNING at its price. What the arms wrote is on its volume ($(cat "$marker"))" >&2
    fi
  fi
  exit 3
elif ((followed)); then
  echo "The arms were not followed to their end (tools/rented.sh follow ended with $followed):" \
    "the pod is as it was" >&2
  exit "$followed"
fi
found=$(rented state "$NAME")
echo "$NAME has $found, and what it kept is here under $RUNS"
if [ -z "$LEAVE" ]; then
  runpodctl pod stop "$POD" > /dev/null 2>&1 || echo "runpodctl did not stop the pod $POD" >&2
  runpodctl pod delete "$POD" > /dev/null 2>&1 || echo "runpodctl did not delete the pod $POD" >&2
  if there; then
    echo "runpodctl STILL NAMES the pod $POD: delete it where it was rented, it may be running at its price" >&2
    exit 4
  fi
  echo "The pod $POD is deleted: runpodctl no longer names it"
else
  echo "The pod is left as it is (LEAVE): it runs at its price until it is stopped"
fi
[ "$found" = "ended 0" ]
