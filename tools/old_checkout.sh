#!/usr/bin/env bash
# A checkout of an older revision beside this one, for what only that revision
# runs: checkpoints of format 13 and 14 and the scripts that made them load at
# 05d27141 and before (docs/commands.md, "A checkpoint of format 14, played by
# today's code"). A worktree of this repository at REVISION in DIRECTORY, its
# submodules and theirs filled from this checkout's without the network, then
# built, with an environment of its own. Run from this checkout:
#   bash tools/old_checkout.sh 05d27141 ../ygo-05d27141
set -euo pipefail
REVISION=$1
DIRECTORY=$2
MAIN=$(git rev-parse --show-toplevel)
VENDOR=$MAIN/native/ygo-policies/vendor
git -C "$MAIN" worktree add --detach "$DIRECTORY" "$REVISION"
cd "$DIRECTORY"
git -c protocol.file.allow=always \
  -c submodule.native/ygo-policies.url="$MAIN/native/ygo-policies" \
  -c submodule.native/ygo-model-input.url="$MAIN/native/ygo-model-input" \
  submodule update native/ygo-policies native/ygo-model-input
(
  cd native/ygo-policies
  git -c protocol.file.allow=always \
    -c submodule.vendor/ocgcore.url="$VENDOR/ocgcore" \
    -c submodule.vendor/CardScripts.url="$VENDOR/CardScripts" \
    -c submodule.vendor/BabelCdb.url="$VENDOR/BabelCdb" \
    submodule update --init vendor/ocgcore vendor/CardScripts vendor/BabelCdb
  # Registered too, so that a script's check of the pins sees them.
  git submodule init > /dev/null
  cd vendor/ocgcore
  git -c protocol.file.allow=always -c submodule.lua/src.url="$VENDOR/ocgcore/lua/src" \
    submodule update --init lua/src
  git submodule init > /dev/null
)
git submodule status --recursive
cargo build --release
uv sync --frozen --offline
