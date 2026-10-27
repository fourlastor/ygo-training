#!/usr/bin/env bash
# The champions a machine makes with tools/champion_lanes.sh, brought to the
# CUDA machine through this one: each deck marked done there, its two runs
# and its measures (runs/champions/DECK, runs/champions/DECK-copy,
# analysis/champions/DECK), streamed from the one checkout into the other and
# held there to their checksums. A deck brought is marked in RELAYED, here,
# and not brought again. With WAIT, it looks again every WAIT seconds, until
# it is stopped.
#
# FROM and TO are what ssh is given to reach each machine, its options first,
# FROM_AT and TO_AT the checkouts there. FROM_BASH is how a script read from
# its input runs on the first: `bash -s` where the login shell is bash,
# `wsl.exe -d Ubuntu-24.04 -e bash -s` for a Windows machine's WSL. The other
# runs what it is given as any shell of a Unix machine does; with TO empty it
# is this machine, where the CUDA machine reaches the other itself.
#
#   FROM='-J daniele@luna alelk@100.93.143.204' FROM_AT=ygo-training/ygo \
#   FROM_BASH='wsl.exe -d Ubuntu-24.04 -e bash -s' TO=daniele@luna \
#   WAIT=1800 bash tools/relay_champions.sh
#
# and on the CUDA machine, from its checkout:
#
#   FROM=alelk@100.93.143.204 FROM_BASH='wsl.exe -d Ubuntu-24.04 -e bash -s' \
#   TO= TO_AT=. WAIT=1800 bash relay_champions.sh
set -euo pipefail
FROM=${FROM:?what ssh is given to reach the machine that makes the champions}
FROM_AT=${FROM_AT:-ygo-training/ygo}
FROM_BASH=${FROM_BASH:-bash -s}
TO=${TO-daniele@luna}
TO_AT=${TO_AT:-ygo-training/ygo}
TAKEN=${TAKEN:-../taken}
RELAYED=${RELAYED:-runs/relayed}
WAIT=${WAIT:-0}
mkdir -p "$RELAYED"

# from SCRIPT: the script run in the checkout there, its output here. The
# script is the input, so that nothing in it is read by a shell of another
# kind on the way (cmd.exe in front of WSL), and FROM_AT is from the home
# folder (WSL begins in the Windows folder it is started from). to COMMAND:
# the command run in the other checkout, with this one's input.
from() {
  # shellcheck disable=SC2086
  ssh -o BatchMode=yes $FROM "$FROM_BASH" <<< "cd && cd $FROM_AT && $1"
}
to() {
  if [ -z "$TO" ]; then
    (cd "$TO_AT" && bash -c "$1")
    return
  fi
  # shellcheck disable=SC2086
  ssh -o BatchMode=yes $TO "cd $TO_AT && $1"
}
# folders DECK: what a deck's champion is, in a checkout.
folders() {
  echo "runs/champions/$1 runs/champions/$1-copy analysis/champions/$1"
}
# sums DECK: each file of the deck's folders with its checksum, in order.
sums() {
  echo "find $(folders "$1") -type f -print0 | sort -z | xargs -0 sha256sum"
}
# bring DECK: the deck's folders streamed there, then held to their
# checksums; fails where a step does, or a file is not as it is here.
bring() {
  local made held
  to "mkdir -p runs/champions analysis/champions" < /dev/null || return 1
  from "tar -cf - $(folders "$1")" | to "tar -xf -" || return 1
  made=$(from "$(sums "$1")") || return 1
  held=$(to "$(sums "$1")" < /dev/null) || return 1
  [ -n "$made" ] && [ "$made" = "$held" ]
}

while :; do
  for deck in $(from "ls $TAKEN" || true); do
    if [ -f "$RELAYED/$deck" ] || ! from "[ -f $TAKEN/$deck/done ]"; then
      continue
    fi
    echo "$(date '+%F %T') $deck: bringing it"
    if bring "$deck"; then
      touch "$RELAYED/$deck"
      echo "$(date '+%F %T') $deck: brought, every file as it is there"
    else
      echo "$(date '+%F %T') $deck: NOT brought whole, brought again on the next look" >&2
    fi
  done
  ((WAIT > 0)) || break
  sleep "$WAIT"
done
