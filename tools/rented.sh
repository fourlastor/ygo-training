#!/usr/bin/env bash
# A rented machine, driven from this checkout over SSH: the repository is
# sent there, a command is left running, its output is looked at and what it
# wrote is brought back. Nothing here knows a provider. The machine is
# whatever answers at RENTED with a login by key, bash, git and tar; renting
# it, stopping it and deleting it are done where it was rented.
#
#     export RENTED=root@HOST:PORT
#     bash tools/rented.sh machine
#     bash tools/rented.sh send
#     bash tools/rented.sh put runs/a-checkpoint.pt
#     bash tools/rented.sh start benchmark bash tools/benchmark.sh
#     bash tools/rented.sh watch benchmark
#     INTO=benchmark/rented bash tools/rented.sh fetch benchmark/report.txt
#
# machine  what answers there: its processor, the threads a program sees and
#          how many of them the machine is allowed to use at once (a
#          container's share, which is what THREADS should be), its memory,
#          its card and driver and whether CUDA starts on it (a machine
#          can name a card that no program can compute on), the room on its
#          disks and which of the tools the repository's scripts ask for are
#          there.
# send     this checkout's HEAD, to the checkout there (AT). It is pushed
#          over SSH into a bare repository beside that checkout, so the
#          machine needs no key of GitHub's for a private repository; the
#          submodules it fetches itself. What is not committed is not sent.
# put      PATH...: files or folders of this checkout that git does not
#          hold (a checkpoint), to the same paths in the checkout there.
#          Folders are merged, as fetch merges them here.
# start    NAME COMMAND...: COMMAND in the checkout there, left running when
#          the connection closes. Its output is in runs/rented/NAME.log
#          there, and its exit status in NAME.exit beside it once it ends.
# watch    NAME [LINES]: the end of that output, and whether NAME still runs.
# wait     NAME: stays until NAME has ended, and ends with its status.
# state    NAME: one word of it, for a script: running, ended with its
#          status after it ("ended 0"), gone (it stopped and left no
#          status) or none (it was never started).
# follow   NAME COMMAND...: stays until NAME has ended, bringing what it
#          writes here as it goes. COMMAND is run in the checkout there and
#          names the files to bring, one a line, by their paths in the
#          checkout: every EVERY seconds (600) while NAME runs, and once
#          more when it has ended, they are brought to the same paths under
#          INTO by rsync, which takes only what is new and takes a file half
#          brought on. Then what is here is held against what is there, file
#          by file by its checksum. It ends with 0 when every file named is
#          here as it is there, whatever came of NAME, and with 3 when not:
#          what to do with the machine then is not to be decided on what is
#          here. Needs rsync here and there.
# fetch    PATH...: files or folders of the checkout there, to the same
#          paths under INTO here (this checkout unless said). Folders are
#          merged: a file of the same name is replaced, nothing is removed.
# run      COMMAND...: a command in the checkout there, its output here.
#          Without a command, a shell there.
#
# The machine's host key is accepted the first time and kept in KNOWN
# (runs/rented-hosts), apart from your own known hosts: a rented address
# passes to another machine sooner or later, and its line is then removed
# from that file.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT=$PWD
AT=${AT:-/workspace/ygo}
INTO=${INTO:-.}
KNOWN=${KNOWN:-$ROOT/runs/rented-hosts}

usage() {
  echo "RENTED=user@host:port bash tools/rented.sh machine | send | put PATH... | start NAME COMMAND... | watch NAME [LINES] | wait NAME | state NAME | follow NAME COMMAND... | fetch PATH... | run [COMMAND...]" >&2
  exit 2
}
(($#)) || usage
if [ -z "${RENTED:-}" ]; then
  echo "Name the machine: RENTED=user@host:port" >&2
  exit 2
fi
case "$RENTED" in
  *:*) target=${RENTED%:*} port=${RENTED##*:} ;;
  *) target=$RENTED port=22 ;;
esac
case "$AT" in
  /*) ;;
  *)
    echo "AT is a path from the root there, not $AT" >&2
    exit 2
    ;;
esac
mkdir -p "$(dirname "$KNOWN")"
options=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o "UserKnownHostsFile=$KNOWN"
  -o ServerAliveInterval=30 -o ConnectTimeout=20)

remote() { ssh "${options[@]}" -p "$port" "$target" "$@"; }
# A script for the machine's bash, read from standard input.
there() { remote bash -s; }
quoted() { printf '%q ' "$@"; }
named() {
  if [[ ! ${1:-} =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "A name of letters, digits, dots and dashes, not '${1:-}'" >&2
    exit 2
  fi
}

machine() {
  there << EOF
grep -m1 'model name' /proc/cpuinfo | sed 's/.*: /processor: /'
threads=\$(nproc)
allowed=\$threads
# A container is held to a share of the processor that nproc does not tell.
quota=0 period=1
if [ -r /sys/fs/cgroup/cpu.max ]; then
  read -r quota period < /sys/fs/cgroup/cpu.max
  [ "\$quota" != max ] || quota=0
elif [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]; then
  quota=\$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us)
  period=\$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)
fi
held=\$(((quota + period - 1) / period))
[ "\$quota" -gt 0 ] && [ "\$held" -lt "\$threads" ] && allowed=\$held
echo "threads: \$threads seen, \$allowed allowed at once"
memory=\$(awk '/MemTotal/ {printf "%.0f", \$2 / 1048576}' /proc/meminfo)
limit=\$(cat /sys/fs/cgroup/memory.max 2> /dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2> /dev/null || echo max)
case "\$limit" in
  max | 9223372036854771712) echo "memory: \$memory GiB" ;;
  *) echo "memory: \$memory GiB seen, \$((limit / 1073741824)) GiB allowed" ;;
esac
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2> /dev/null | sed 's/^/card: /' \
  || echo "card: none that nvidia-smi names"
# The driver's own answer to a program that asks for the card: 0 where it starts.
python3 - 2> /dev/null << 'ASKED' || echo "cuda: not asked (no python3)"
import ctypes
try:
    started = ctypes.CDLL("libcuda.so.1").cuInit(0)
except OSError:
    print("cuda: no driver library for it")
else:
    print("cuda: starts" if started == 0 else f"cuda: DOES NOT START (error {started}): no program can compute on the card")
ASKED
place=$(quoted "$(dirname "$AT")")
[ -d "\$place" ] || place=/
df -h --output=avail,target "\$place" / | awk 'NR > 1 && !seen[\$2]++ {print "room: " \$1 " at " \$2}'
for tool in git tar curl c++ python3 uv cargo rsync; do
  command -v "\$tool" > /dev/null 2>&1 && there="\$there \$tool" || lacking="\$lacking \$tool"
done
echo "tools:\${there:- none}\${lacking:+ (not there:\$lacking)}"
python3 --version 2> /dev/null || true
EOF
}

send() {
  local changed
  changed=$(git status --porcelain --untracked-files=no | wc -l)
  ((changed == 0)) || echo "$changed changed files here are not committed: they are not sent" >&2
  there << EOF
set -e
command -v git > /dev/null || { echo "There is no git there" >&2; exit 1; }
mkdir -p $(quoted "$AT.git")
git init -q --bare $(quoted "$AT.git")
EOF
  GIT_SSH_COMMAND="ssh $(quoted "${options[@]}")" \
    git push --quiet --force "ssh://$target:$port$AT.git" HEAD:refs/heads/sent
  there << EOF
set -e
at=$(quoted "$AT")
[ -d "\$at/.git" ] || git clone -q --no-checkout "\$at.git" "\$at" 2> /dev/null
cd "\$at"
git fetch -q origin sent
git checkout -q --detach FETCH_HEAD
git submodule update --init --recursive
echo "there: \$(git log --oneline -1)"
EOF
}

start() {
  named "${1:-}"
  local name=$1
  shift
  (($#)) || usage
  local command inner
  command=$(quoted "$@")
  inner="echo \$\$ > runs/rented/$name.pid; { $command; } > runs/rented/$name.log 2>&1; echo \$? > runs/rented/$name.exit"
  there << EOF
set -e
cd $(quoted "$AT")
mkdir -p runs/rented
if [ -e runs/rented/$name.pid ] && [ ! -e runs/rented/$name.exit ] \
  && kill -0 "\$(cat runs/rented/$name.pid)" 2> /dev/null; then
  echo "$name is still running there" >&2
  exit 1
fi
rm -f runs/rented/$name.exit
# A session of its own, and nothing of the connection's held open: what a
# closing connection ends is its own session.
setsid bash -c $(quoted "$inner") < /dev/null > /dev/null 2>&1 &
EOF
  echo "$name started there: $command"
}

watch() {
  named "${1:-}"
  local name=$1 lines=${2:-20}
  [[ $lines =~ ^[0-9]+$ ]] || usage
  there << EOF
cd $(quoted "$AT/runs/rented") || exit 1
tail -n $lines $name.log
if [ -e $name.exit ]; then
  echo "-- $name ended with status \$(cat $name.exit)"
elif kill -0 "\$(cat $name.pid 2> /dev/null)" 2> /dev/null; then
  echo "-- $name is running"
else
  echo "-- $name is gone, and left no status"
fi
EOF
}

await() {
  named "${1:-}"
  local name=$1
  there << EOF
cd $(quoted "$AT/runs/rented") || exit 1
while [ ! -e $name.exit ]; do
  if ! kill -0 "\$(cat $name.pid 2> /dev/null)" 2> /dev/null; then
    sleep 2
    [ -e $name.exit ] && break
    echo "$name is gone, and left no status" >&2
    exit 1
  fi
  sleep 5
done
exit "\$(cat $name.exit)"
EOF
}

state() {
  named "${1:-}"
  local name=$1
  there << EOF
cd $(quoted "$AT/runs/rented") 2> /dev/null || { echo none; exit 0; }
if [ -e $name.exit ]; then
  echo "ended \$(cat $name.exit)"
elif [ ! -e $name.pid ]; then
  echo none
elif kill -0 "\$(cat $name.pid)" 2> /dev/null; then
  echo running
else
  # It may have ended between the two looks.
  sleep 2
  if [ -e $name.exit ]; then echo "ended \$(cat $name.exit)"; else echo gone; fi
fi
EOF
}

follow() {
  named "${1:-}"
  local name=$1
  shift
  (($#)) || usage
  local every=${EVERY:-600} lister list shell found waited tries differing
  [[ $every =~ ^[0-9]+$ ]] || usage
  if ! command -v rsync > /dev/null; then
    echo "There is no rsync here" >&2
    exit 1
  fi
  lister=$(quoted "$@")
  list=$(mktemp)
  # The connection rsync makes is this script's own: its options do not
  # survive rsync's splitting of a command at spaces, so they are in a file.
  shell=$(mktemp)
  # shellcheck disable=SC2064
  trap "rm -f $(quoted "$list" "$shell")" EXIT
  printf '#!/usr/bin/env bash\nexec ssh %s-p %q "$@"\n' "$(quoted "${options[@]}")" "$port" > "$shell"
  chmod +x "$shell"
  mkdir -p "$INTO"
  # A folder on the way to a file may be a link there to where it is kept (a
  # disk that outlasts the machine): the files are brought, and the folders
  # here stay what they are.
  local brought=(-a -s --copy-dirlinks --no-implied-dirs)
  # The files named there now, brought here: 0 when all of them are here.
  bring() {
    remote "cd $(quoted "$AT") && $lister" > "$list" || return 1
    local status=0
    rsync "${brought[@]}" --partial --files-from="$list" -e "$shell" "$target:$AT/" "$INTO/" || status=$?
    # 24: a file named was gone when its turn came, which the next time settles.
    ((status == 0 || status == 24))
  }
  while :; do
    found=$(state "$name") || found=unreachable
    [ "$found" = running ] || [ "$found" = unreachable ] || break
    if [ "$found" = running ]; then
      bring || echo "$(date '+%H:%M') Not all was brought this time: the next goes on from it" >&2
    else
      echo "$(date '+%H:%M') The machine does not answer: asked again later" >&2
    fi
    # The end of NAME is seen within half a minute, whatever EVERY is.
    for ((waited = 0; waited < every; waited += 30)); do
      sleep $((every - waited < 30 ? every - waited : 30))
      found=$(state "$name") || found=unreachable
      [ "$found" = running ] || [ "$found" = unreachable ] || break
    done
  done
  echo "$(date '+%H:%M') $name: $found"
  for ((tries = 1; tries <= 5; tries++)); do
    if bring; then
      differing=$(rsync "${brought[@]}" --checksum --dry-run --itemize-changes \
        --files-from="$list" -e "$shell" "$target:$AT/" "$INTO/") \
        || differing="rsync could not compare them"
      if [ -z "$differing" ]; then
        echo "Every file named there is here as it is there: $(wc -l < "$list") files under $INTO"
        return 0
      fi
      echo "Not the same here as there:" >&2
      echo "$differing" | head -n 20 >&2
    fi
    echo "$(date '+%H:%M') Try $tries of 5 to bring the last of it did not come to its end" >&2
    ((tries == 5)) || sleep "${AGAIN:-60}"
  done
  echo "What $name wrote there is NOT all here: the machine still holds it" >&2
  return 3
}

put() {
  (($#)) || usage
  local path
  for path; do
    path=${path%/}
    case "/$path/" in
      //* | */../*)
        echo "A path inside the checkout, not $path" >&2
        exit 2
        ;;
    esac
    if [ ! -e "$path" ]; then
      echo "Nothing at $path here" >&2
      exit 1
    fi
    tar -C "$ROOT" -cf - "$path" | remote "mkdir -p $(quoted "$AT") && tar -C $(quoted "$AT") -xf -"
    echo "there: $AT/$path"
  done
}

fetch() {
  (($#)) || usage
  local path
  mkdir -p "$INTO"
  for path; do
    path=${path%/}
    case "/$path/" in
      //* | */../*)
        echo "A path inside the checkout, not $path" >&2
        exit 2
        ;;
    esac
    remote "tar -C $(quoted "$AT") -cf - $(quoted "$path")" | tar -C "$INTO" -xf -
    echo "$INTO/$path"
  done
}

run() {
  if (($#)); then
    remote "cd $(quoted "$AT") && $(quoted "$@")"
  else
    ssh "${options[@]}" -p "$port" -t "$target" "cd $(quoted "$AT") 2> /dev/null; exec bash -l"
  fi
}

command=$1
shift
case "$command" in
  machine) machine ;;
  send) send ;;
  put) put "$@" ;;
  start) start "$@" ;;
  watch) watch "$@" ;;
  wait) await "$@" ;;
  state) state "$@" ;;
  follow) follow "$@" ;;
  fetch) fetch "$@" ;;
  run) run "$@" ;;
  *) usage ;;
esac
