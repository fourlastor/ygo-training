"""A single-deck champion outside its mirror (experiments/2026-10-24-two-decks.md,
its follow-up on a single-deck champion outside its mirror).

    uv run python tools/outside_the_mirror_report.py report analysis/outside-the-mirror \\
        --both analysis/two-decks --points 480 600 720 --kinds 2
    uv run python tools/outside_the_mirror_report.py check HERE.json THERE.json

The directory is what `experiments/commands/outside-the-mirror.sh` leaves:

- `cross/blackwing-POINT.json`: Blackwing alone, after POINT updates of PPO,
  playing Blackwing against `all71` playing Cyber; `cross/cyber-POINT.json`:
  `all71` playing Blackwing against `cyber-old` playing Cyber. Both on the
  deals the network of both was played on in `--both`'s `cross/`, where
  `STEM.plays.json` has it playing Blackwing against `all71` and
  `STEM.held.json` playing Cyber, STEM its updates of PPO, `--kinds` squared
  times a champion's at as many games of each mirror;
- `panel/blackwing-POINT.json` and `panel/cyber-POINT.json`: each champion
  playing its deck against `all71` playing each deck outside the thirteen;
  `panel/blackwing-all71.json` and `panel/cyber-all71.json`: `all71` playing
  the champion's deck in the same games.

A deal's games at the three points are read together. `check` is held to
the first: the network of both's games played again, which must end as they
did.

The readings, fixed on 2026-10-24 before anything was played, each side of
Blackwing against Cyber against `all71`, checkpoint for checkpoint, the
deals paired:

1. Blackwing's side: Blackwing alone against the network of both, each
   playing Blackwing: the single-deck champion plays it as well when its
   share less the network of both's, less two standard errors, is above -2
   points; worse when that difference plus two standard errors is below -2;
   otherwise not decided.
2. Cyber's side: `cyber-old` against the network of both, each playing Cyber,
   the same.

Beside them, without a mark: each champion's edge over `all71` playing its
deck against the decks outside the thirteen, the games paired, by its
edge in its own mirror (`--both`'s `all71-alone/` and `--cyber-alone`).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

ERRORS = 2.0
# Readings 1 and 2: so many points under the network of both.
UNDER = -0.02
# Each side: its deck, its champion, whether the champion is in the
# learner's seat in its files, and the network of both's files of it.
SIDES = (
    ("Blackwing", "Blackwing alone", "blackwing", True, "plays"),
    ("Cyber", "cyber-old", "cyber", False, "held"),
)


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def error(values: list[float]) -> float:
    centre = mean(values)
    return math.sqrt(sum((value - centre) ** 2 for value in values) / (len(values) - 1)) / (
        math.sqrt(len(values))
    )


def cases(path: Path) -> dict[str, list[str]]:
    if not path.is_file():
        raise SystemExit(f"Missing {path}")
    (record,) = json.loads(path.read_text(encoding="utf8"))["records"]
    return {name: case["outcomes"] for name, case in record["evaluation_suite"].items()}


def won(path: Path, *, learner: bool = True) -> list[float]:
    """Each game of a file, its cases in the order of their names: 1 where the
    network in the learner's seat wins it, or with `learner` false the one in
    the other seat."""
    told = "win" if learner else "loss"
    found = cases(path)
    return [float(outcome == told) for name in sorted(found) for outcome in found[name]]


def together(games: list[list[float]]) -> list[float]:
    """A deal's games at each point, together."""
    if len({len(played) for played in games}) != 1:
        raise SystemExit("The points were played on other deals")
    return [mean(list(told)) for told in zip(*games, strict=True)]


def paired(ours: list[float], theirs: list[float]) -> dict[str, float]:
    if len(ours) != len(theirs):
        raise SystemExit("The two were played on other deals")
    gap = [one - other for one, other in zip(ours, theirs, strict=True)]
    return {"ours": mean(ours), "theirs": mean(theirs), "gap": mean(gap), "error": error(gap)}


def verdict(gap: float, error_: float) -> str:
    """Readings 1 and 2."""
    if gap - ERRORS * error_ > UNDER:
        return "as well"
    if gap + ERRORS * error_ < UNDER:
        return "worse"
    return "not decided"


def mirror(folder: Path) -> float:
    """A network's share against all71 in its mirror, every deal from both
    seats (two_decks_report's files)."""
    games = []
    for plays in sorted(folder.glob("*.plays.json")):
        held = plays.with_name(plays.name.replace(".plays.", ".held."))
        games.append(
            [
                (one + other) / 2
                for one, other in zip(won(plays), won(held, learner=False), strict=True)
            ]
        )
    if not games:
        raise SystemExit(f"Nothing is measured in {folder}")
    return mean(together(games))


def report(
    directory: Path,
    both: Path,
    points: list[int],
    kinds: int,
    cyber_alone: Path | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {"readings": {}, "panel": {}}
    for deck, champion, part, seated, kind in SIDES:
        ours = together(
            [won(directory / "cross" / f"{part}-{point}.json", learner=seated) for point in points]
        )
        theirs = together(
            [
                won(both / "cross" / f"{kinds * kinds * point}.{kind}.json", learner=seated)
                for point in points
            ]
        )
        found = paired(ours, theirs)
        found["champion"] = champion
        found["verdict"] = verdict(found["gap"], found["error"])
        result["readings"][deck] = found
    mirrors = {"blackwing": both / "all71-alone", "cyber": cyber_alone}
    for deck, champion, part, _, _ in SIDES:
        ours = together([won(directory / "panel" / f"{part}-{point}.json") for point in points])
        found = paired(ours, won(directory / "panel" / f"{part}-all71.json"))
        found["champion"] = champion
        found["decks"] = len(cases(directory / "panel" / f"{part}-all71.json"))
        if mirrors[part] is not None:
            found["mirror edge"] = mirror(mirrors[part]) - 0.5
        result["panel"][deck] = found
    return result


def shown(result: dict[str, Any]) -> str:
    lines = []
    for number, (deck, found) in enumerate(result["readings"].items(), start=1):
        lines.append(
            f"{number}. {deck}'s side of Blackwing against Cyber: {found['verdict']}. "
            f"{found['champion']} {found['ours']:.1%}, the network of both {found['theirs']:.1%}, "
            f"against all71: {100 * found['gap']:+.1f} ± {100 * found['error']:.1f} points, "
            "the deals paired"
        )
    lines.append("Beside them, against all71 playing each deck outside the thirteen:")
    for deck, found in result["panel"].items():
        line = (
            f"  {found['champion']} playing {deck}: {found['ours']:.1%}, all71 playing it "
            f"{found['theirs']:.1%}: {100 * found['gap']:+.1f} ± {100 * found['error']:.1f} "
            f"points over {found['decks']} decks"
        )
        if "mirror edge" in found:
            line += f"; in its own mirror {100 * found['mirror edge']:+.1f}"
        lines.append(line)
    return "\n".join(lines)


def check(here: Path, there: Path) -> None:
    """The same games played again end as they did."""
    ours, theirs = cases(here), cases(there)
    if ours != theirs:
        raise SystemExit(f"{here} does not end its games as {there} did")
    print(f"{sum(len(told) for told in ours.values()):,} games end as they did ({there})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    told = commands.add_parser("report")
    told.add_argument("directory", type=Path)
    told.add_argument("--both", type=Path, required=True, help="where both was measured")
    told.add_argument("--points", type=int, nargs="+", required=True)
    told.add_argument("--kinds", type=int, default=2, help="the network of both's decks")
    told.add_argument("--cyber-alone", type=Path, help="cyber-old against all71 in its mirror")
    told.add_argument("--output", type=Path)
    again = commands.add_parser("check")
    again.add_argument("here", type=Path)
    again.add_argument("there", type=Path)
    args = parser.parse_args()
    if args.command == "check":
        check(args.here, args.there)
        return
    result = report(args.directory, args.both, args.points, args.kinds, args.cyber_alone)
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2))
    print(shown(result))


if __name__ == "__main__":
    main()
