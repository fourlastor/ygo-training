#!/usr/bin/env bash
# The improved pilots on all71's yardstick
# (experiments/2026-10-23-improved-pilots.md), told without a mark. Run from
# the repository root on the development machine, with the policies at their
# pin:
#
# 1. the pilots as the trainer pins them: the forty decks' and the eighteen
#    decks' suites as seventy-one decks played them, and the thirteen decks'
#    screens (evaluate-suite --pilot, seed 2012, 24 workers), into
#    OUT/pinned-COMMIT, once a pin;
# 2. the branch's: a worktree of this repository at its HEAD (WORKTREE), its
#    submodules filled from this checkout's, the policies at BRANCH as REMOTE
#    has it, built on its own, nothing of this checkout touched; the same
#    suites into OUT/branch-COMMIT, and the files the branch changed;
# 3. the report (tools/improved_pilots.py), all71's games those of
#    seventy-one decks (NETWORK): a deck the branch left as it was must win
#    the same games.
#
# PARTS names what is made: pinned (1), branch (2 and 3). Minutes: pilot
# games only.
#   bash experiments/commands/improved-pilots.sh
set -euo pipefail
BRANCH=${BRANCH:-pilots-against-the-network}
REMOTE=${REMOTE:-https://github.com/fourlastor/ygo-policies}
WORKTREE=${WORKTREE:-../ygo-pilots}
OUT=${OUT:-analysis/improved-pilots}
NETWORK=${NETWORK:-analysis/seventy-one-decks}
WORKERS=${WORKERS:-24}
THREADS=${THREADS:-20}
SLOTS=${SLOTS:-16}
SEED=${SEED:-2012}
PYTHON=${PYTHON:-$PWD/.venv/bin/python}
PARTS=${PARTS:-pinned branch}
MAIN=$PWD
VENDOR=$MAIN/native/ygo-policies/vendor
declare -A suites=(
  [new]=$MAIN/eval_suites/ygo-policies-forty-played.json
  [old]=$MAIN/eval_suites/ygo-policies-unseen-played.json
)
# Each of the thirteen decks' own screen, against the thirteen decks'
# pilots, where an improved pilot of theirs plays its own deck.
THIRTEEN=${THIRTEEN:-blackwing lightsworn gishki burn crystal dragunity gladiator heroes infernity monarch morphtronic rock-block spellcaster}
kinds=(new old)
for deck in $THIRTEEN; do
  suites[screen-$deck]=$MAIN/eval_suites/ygo-policies-$deck-screen.json
  kinds+=("screen-$deck")
done

pinned=$(git submodule status --recursive)
if grep -q '^[^ ]' <<< "$pinned"; then
  echo "A submodule is not at its pinned revision: git submodule update --init --recursive" >&2
  exit 1
fi
if [ -n "${YGO_ENV_LIBRARY:-}" ]; then
  echo "YGO_ENV_LIBRARY names another library: each checkout plays with its own" >&2
  exit 1
fi

# played ROOT FOLDER: every suite with the pilots of the checkout at ROOT,
# into OUT/FOLDER.
played() {
  local root=$1 folder=$2 kind
  mkdir -p "$MAIN/$OUT/$folder"
  for kind in "${kinds[@]}"; do
    if [ ! -f "$MAIN/$OUT/$folder/$kind.json" ]; then
      (cd "$root" && "$PYTHON" -m ygo_training evaluate-suite --pilot --device cpu \
        --workers "$WORKERS" --pool-threads "$THREADS" --pool-slots "$SLOTS" --seed "$SEED" \
        --suite "${suites[$kind]}" --output "$MAIN/$OUT/$folder/$kind.json.part" > /dev/null)
      mv "$MAIN/$OUT/$folder/$kind.json.part" "$MAIN/$OUT/$folder/$kind.json"
    fi
  done
}

cargo build --release
pin=$(git -C native/ygo-policies rev-parse --short HEAD)
played "$MAIN" "pinned-$pin"
if [[ " $PARTS " != *" branch "* ]]; then
  exit 0
fi

if [ ! -d "$WORKTREE" ]; then
  git worktree add --detach "$WORKTREE" HEAD
fi
git -C "$WORKTREE" checkout -q --detach "$(git rev-parse HEAD)"
(
  cd "$WORKTREE"
  git -c protocol.file.allow=always \
    -c submodule.native/ygo-policies.url="$MAIN/native/ygo-policies" \
    -c submodule.native/ygo-model-input.url="$MAIN/native/ygo-model-input" \
    submodule update --init native/ygo-policies native/ygo-model-input
  cd native/ygo-policies
  git fetch -q "$REMOTE" "$BRANCH"
  git checkout -q --detach FETCH_HEAD
  git -c protocol.file.allow=always \
    -c submodule.vendor/ocgcore.url="$VENDOR/ocgcore" \
    -c submodule.vendor/CardScripts.url="$VENDOR/CardScripts" \
    -c submodule.vendor/BabelCdb.url="$VENDOR/BabelCdb" \
    submodule update --init vendor/ocgcore vendor/CardScripts vendor/BabelCdb
  git submodule init > /dev/null
  cd vendor/ocgcore
  git -c protocol.file.allow=always -c submodule.lua/src.url="$VENDOR/ocgcore/lua/src" \
    submodule update --init lua/src
  git submodule init > /dev/null
)
branch=$(git -C "$WORKTREE/native/ygo-policies" rev-parse --short HEAD)
echo "The branch's pilots at $branch, against the pin $pin:"
git -C "$WORKTREE/native/ygo-policies" diff --stat "$pin" "$branch" | tail -25
(cd "$WORKTREE" && cargo build --release)
played "$WORKTREE" "branch-$branch"
git -C "$WORKTREE/native/ygo-policies" diff --stat "$pin" "$branch" > "$OUT/branch-$branch/changed.txt"
"$PYTHON" tools/improved_pilots.py "$OUT/pinned-$pin" "$OUT/branch-$branch" --network "$NETWORK" \
  --output "$OUT/branch-$branch/report.json" | tee "$OUT/branch-$branch/report.txt"
