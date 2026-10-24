#!/usr/bin/env bash
# The trainer's library with the engine's C and C++ (the core and Lua) built
# from a profile of this machine's own games: a tenth faster between pilots,
# the same games (docs/commands.md, "The engine built from a profile of its
# own games"). Run from the repository root with nothing else using the
# build, on each machine, and again after the pins move. By flags alone:
#   1. the library built as it is, and a check suite of pilot games played
#      with it, the reference;
#   2. the library built to count what it runs, and the pool's 71 decks
#      against the thirteen decks' pilots played with it, two games a
#      pairing: the profile, into PROFILE/data;
#   3. .cargo/config.toml written (machine-local, not in git) so that every
#      `cargo build --release` here builds the engine from that profile, and
#      the library built so;
#   4. the check suite played again: its games the same as the reference's,
#      or the config is moved aside and the library built as it was.
# The wide Blackwing screen (6,656 games) is timed with both, its start-up of
# some 2 s with it. The models play no part: pilot games only, on the
# processor. On the development machine on 2026-10-20 its games took a
# twentieth less than as built now, those of policy-bench on one thread a
# tenth (docs/commands.md).
#   bash tools/engine_pgo_build.sh
set -euo pipefail
PROFILE=${PROFILE:-$PWD/target/engine-pgo}
WORKERS=${WORKERS:-24}
THREADS=${THREADS:-20}
SLOTS=${SLOTS:-16}
# The interpreter of a machine that has no uv: the benchmark's setup leaves one.
PYTHON=${PYTHON:-}
CONFIG=.cargo/config.toml
# The Rust flags of every build from here: the instrumented one needs the
# coverage library, and GCC names a profile after its object's path, whose
# build directory holds a hash of these flags.
RUSTFLAG="-C link-arg=-lgcov"

pinned=$(git submodule status --recursive)
if grep -q '^[^ ]' <<< "$pinned"; then
  echo "A submodule is not at its pinned revision: git submodule update --init --recursive" >&2
  exit 1
fi
python=(uv run python)
[ -z "$PYTHON" ] || python=("$PYTHON")
stamp=$(date +%Y%m%d-%H%M%S)
mkdir -p "$PROFILE"
[ ! -d "$PROFILE/data" ] || mv "$PROFILE/data" "$PROFILE/data.before-$stamp"
[ ! -f "$CONFIG" ] || mv "$CONFIG" "$CONFIG.before-$stamp"

# The suites: every deck of the suites of the pool's decks, against the
# thirteen decks' pilots.
"${python[@]}" - "$PROFILE" <<'EOF'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
learners, opponents = set(), set()
for name in ("ygo-policies-forty-played", "ygo-policies-unseen-played"):
    for case in json.loads(Path(f"eval_suites/{name}.json").read_text())["cases"]:
        learners.add(case["learner"])
        opponents.add(case["opponent"])
learners |= opponents
for name, games in (("profile", 2), ("check", 4)):
    cases = [
        {"name": f"{learner}-vs-{opponent}", "learner": learner, "opponent": opponent, "games": games}
        for learner in sorted(learners)
        for opponent in sorted(opponents)
    ]
    suite = {"name": f"engine-pgo-{name}", "primary": cases[0]["name"], "cases": cases}
    (out / f"{name}-suite.json").write_text(json.dumps(suite, indent=1))
print(f"{len(learners)} decks against {len(opponents)} pilots")
EOF

TIMED=eval_suites/ygo-policies-blackwing-wide.json
# play SUITE SEED OUTPUT: the suite's pilot games, with how long they took.
play() {
  local start end
  start=$(date +%s.%N)
  "${python[@]}" -m ygo_training evaluate-suite --pilot --device cpu --workers "$WORKERS" \
    --pool-threads "$THREADS" --pool-slots "$SLOTS" --seed "$2" --suite "$1" --output "$3" > /dev/null
  end=$(date +%s.%N)
  echo "$end - $start" | bc
}

echo "1. the library as it is built"
CFLAGS="" CXXFLAGS="" RUSTFLAGS="" cargo build --release
play "$PROFILE/check-suite.json" 2013 "$PROFILE/check-plain.json" > /dev/null
plain=$(play "$TIMED" 2012 "$PROFILE/timed-plain.json")
echo "   the check suite played; the wide screen in $plain s"

echo "2. the library that counts what it runs, and the profile's games"
CFLAGS="-fprofile-generate=$PROFILE/data -fprofile-update=atomic" \
  CXXFLAGS="-fprofile-generate=$PROFILE/data -fprofile-update=atomic" \
  RUSTFLAGS="$RUSTFLAG" cargo build --release
counted=$(play "$PROFILE/profile-suite.json" 2012 "$PROFILE/profile-games.json")
echo "   played in $counted s: $(find "$PROFILE/data" -name '*.gcda' | wc -l) profiles"

echo "3. the library built from the profile"
mkdir -p .cargo
cat > "$CONFIG" <<EOF
# Written by tools/engine_pgo_build.sh on $stamp: the engine's C and C++ built
# from this machine's profile ($PROFILE/data). Machine-local, not in git;
# move it aside to build the engine as it is.
[env]
CFLAGS = "-fprofile-use=$PROFILE/data -fprofile-correction -Wno-missing-profile"
CXXFLAGS = "-fprofile-use=$PROFILE/data -fprofile-correction -Wno-missing-profile"

[build]
rustflags = ["-C", "link-arg=-lgcov"]
EOF
cargo build --release
# Every object with code built where the profile was made (the engine, and
# any other C a crate builds) has its profile under the path GCC looks for,
# and was built again from it just now. An object of data alone (Lua's
# lctype and lopcodes) runs nothing and has none.
missing=0
objects=0
while read -r out; do
  for object in "$out"/*.o; do
    nm --defined-only "$object" | grep -q ' [TtWw] ' || continue
    objects=$((objects + 1))
    if [ ! -f "$PROFILE/data$out/$(basename "$object" .o).gcda" ] || [ ! "$object" -nt "$CONFIG" ]; then
      missing=$((missing + 1))
      echo "   not built from a profile: $object"
    fi
  done
done < <(find "$PROFILE/data" -name '*.gcda' -printf '%h\n' | sort -u | sed "s|^$PROFILE/data||")
echo "   $objects objects with code, $missing of them not built from a profile"

echo "4. the check suite again"
play "$PROFILE/check-suite.json" 2013 "$PROFILE/check-pgo.json" > /dev/null
built=$(play "$TIMED" 2012 "$PROFILE/timed-pgo.json")
echo "   the check suite played; the wide screen in $built s, for $plain as built now"
# same BEFORE AFTER: whether two suites' records tell the same games.
same() {
  "${python[@]}" - "$1" "$2" <<'EOF'
import json
import sys
from pathlib import Path

TIMES = {"seconds", "duels_per_second", "decisions_per_second", "inference_seconds"}


def games(path: str) -> list:
    """The records without what only tells how long they took."""
    def strip(value):
        if isinstance(value, dict):
            return {k: strip(v) for k, v in value.items() if k not in TIMES}
        if isinstance(value, list):
            return [strip(v) for v in value]
        return value

    return strip(json.loads(Path(path).read_text())["records"])


before, after = games(sys.argv[1]), games(sys.argv[2])
cases = before[0]["evaluation_suite"]
print(f"   {sum(case['games'] for case in cases.values())} games in {len(cases)} cases:",
      "the same" if before == after else "NOT the same")
sys.exit(0 if before == after else 1)
EOF
}
if ! same "$PROFILE/check-plain.json" "$PROFILE/check-pgo.json" \
  || ! same "$PROFILE/timed-plain.json" "$PROFILE/timed-pgo.json"; then
  mv "$CONFIG" "$CONFIG.failed-$stamp"
  CFLAGS="" CXXFLAGS="" RUSTFLAGS="" cargo build --release
  echo "The games differ: the config is moved aside and the library built as it was" >&2
  exit 1
fi
if ((missing)); then
  echo "Some of the engine was built without its profile: see above" >&2
fi
echo "Done: every cargo build --release here builds the engine from the profile ($CONFIG)."
