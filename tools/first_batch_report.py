"""The readings of the expanded pool's first batch test
(experiments/2026-11-02-expanded-pool.md).

    uv run python tools/first_batch_report.py analysis/expanded-pool/first-batch \\
        --points 540,1080,2160,4320

The directory is what `experiments/commands/first-batch.sh` leaves, for each of the
points and at 0: `pairings.json`, each new deck against the frozen network at each
old deck; for each of the points `old.json`, every ordered pairing of the old decks
head to head against it; at 0 and at the last point `held-out.json` and
`pilots.json`, and at the last point `mirrors/`.

The readings, fixed on 2026-11-02 before anything was built:

1. A new deck learns from here: the mean over the new decks of their share against
   the frozen network at the third point, less at the start, the deals paired:
   learns when at least 5 points by two standard errors, does not when at most 2
   points by two standard errors, otherwise not decided.
2. How many games a deck takes: the first of the points before the last whose gain
   is within 2 points of the last point's by two standard errors, the deals paired.
   Where none is, the decks are still rising at the last point.
3. The old decks hold: the network head to head against the frozen one on the old
   decks' pairings at the last point: hold when at least 49% by two standard errors,
   slip when at most 49% by two standard errors, otherwise not decided.

Told beside them, without a mark: each deck's gain along the points; the old decks
along them; the next twelve decks, trained on nothing; the new decks against the
pilots; the old mirrors head to head against their champions; the hours.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fourteen_decks_report import ERRORS, number
from specialist_distillation_report import shares, spent
from two_decks_report import mean, paired, told

# Reading 1.
LEARNS = 0.05
DOES_NOT = 0.02
# Reading 2: within so much of the last point's gain.
WITHIN = 0.02
# Reading 3.
HOLDS = 0.49


def cases(path: Path) -> dict[str, list[float]]:
    """Each case's games in a file, won or not, in their order."""
    (record,) = json.loads(path.read_text(encoding="utf8"))["records"]
    return {
        name: [float(game == "win") for game in case["outcomes"]]
        for name, case in record["evaluation_suite"].items()
    }


def games(table: dict[str, list[float]]) -> list[float]:
    """All the games of a file, in the order of its cases' names."""
    return [game for name in sorted(table) for game in table[name]]


def learns(gain: float, error_: float) -> str:
    if gain - ERRORS * error_ >= LEARNS:
        return "learns"
    if gain + ERRORS * error_ <= DOES_NOT:
        return "does not"
    return "not decided"


def budget(points: list[int], gaps: dict[int, dict[str, float]]) -> int | None:
    """The first point whose gain is within `WITHIN` of the last's, ``gaps`` being
    each point's gain less the last's, or none."""
    for point in points[:-1]:
        if -gaps[point]["gap"] + ERRORS * gaps[point]["error"] <= WITHIN:
            return point
    return None


def holds(share: float, error_: float) -> str:
    if share - ERRORS * error_ >= HOLDS:
        return "hold"
    if share + ERRORS * error_ <= HOLDS:
        return "slip"
    return "not decided"


def report(directory: Path, points: list[int]) -> dict[str, Any]:
    start = cases(directory / "0" / "pairings.json")
    tables = {point: cases(directory / str(point) / "pairings.json") for point in points}
    third, last = points[2], points[-1]
    gains = {point: paired(games(tables[point]), games(start)) for point in points}
    each = {
        name: {point: mean(tables[point][name]) - mean(start[name]) for point in points}
        for name in sorted(start)
    }
    behind = {point: paired(games(tables[point]), games(tables[last])) for point in points[:-1]}
    old = {point: told(games(cases(directory / str(point) / "old.json"))) for point in points}
    found: dict[str, Any] = {
        "start": mean(games(start)),
        "gains": gains,
        "each": each,
        "learns": learns(gains[third]["gap"], gains[third]["error"]),
        "behind the last": behind,
        "budget": budget(points, behind),
        "old": old,
        "holds": holds(old[last]["share"], old[last]["error"]),
    }
    held = directory / str(last) / "held-out.json"
    if held.is_file():
        found["held out"] = {
            "start": mean(games(cases(directory / "0" / "held-out.json"))),
            **paired(games(cases(held)), games(cases(directory / "0" / "held-out.json"))),
        }
    pilots = directory / str(last) / "pilots.json"
    if pilots.is_file():
        found["pilots"] = {
            "start": mean(games(cases(directory / "0" / "pilots.json"))),
            **paired(games(cases(pilots)), games(cases(directory / "0" / "pilots.json"))),
        }
    mirrors = sorted((directory / str(last) / "mirrors").glob("*.held.json"))
    if mirrors:
        rows = {
            path.name.split(".")[0]: told(shares(path.with_name(path.name.split(".")[0])))
            for path in mirrors
        }
        found["mirrors"] = {
            "mean": mean([row["share"] for row in rows.values()]),
            "each": rows,
        }
    found["hours"] = spent(directory, "first-batch")
    return found


def shown(found: dict[str, Any], points: list[int]) -> str:
    third, last = points[2], points[-1]
    gain = found["gains"][third]
    lines = [
        (
            f"1. A new deck {found['learns']}: at {third:,} updates "
            f"{100 * gain['gap']:+.1f} ± {100 * gain['error']:.1f} points against the frozen "
            f"network, from {100 * found['start']:.1f}%"
        ),
    ]
    if found["budget"] is None:
        lines.append(f"2. Still rising at {last:,} updates: no point is within 2 points of it")
    else:
        lines.append(f"2. {found['budget']:,} updates are within 2 points of {last:,}")
    old = found["old"][last]
    lines.append(
        f"3. The old decks {found['holds']}: {100 * old['share']:.1f}% ± {100 * old['error']:.1f}% "
        f"head to head against the frozen network at {last:,} updates"
    )
    lines.append("Beside them:")
    lines.append(
        "  The gain along the run: "
        + ", ".join(
            f"{point:,}: {100 * found['gains'][point]['gap']:+.1f} ± "
            f"{100 * found['gains'][point]['error']:.1f}"
            for point in points
        )
    )
    lines.append(
        "  The old decks along it: "
        + ", ".join(f"{point:,}: {100 * found['old'][point]['share']:.1f}%" for point in points)
    )
    for name, row in found["each"].items():
        lines.append(
            f"  {name.split('-vs-')[0]}: "
            + ", ".join(f"{100 * row[point]:+.1f}" for point in points)
        )
    if "held out" in found:
        told_ = found["held out"]
        lines.append(
            f"  The next twelve decks, trained on nothing: {100 * told_['gap']:+.1f} ± "
            f"{100 * told_['error']:.1f} points, from {100 * told_['start']:.1f}%"
        )
    if "pilots" in found:
        told_ = found["pilots"]
        lines.append(
            f"  The new decks against the pilots: {100 * (told_['start'] + told_['gap']):.1f}%, "
            f"from {100 * told_['start']:.1f}%, {100 * told_['gap']:+.1f} ± {100 * told_['error']:.1f}"
        )
    if "mirrors" in found:
        each = found["mirrors"]["each"]
        lowest = sorted(each, key=lambda deck: each[deck]["share"])[:5]
        lines.append(
            f"  The old mirrors against their champions: {100 * found['mirrors']['mean']:.1f}%, the "
            "lowest " + ", ".join(f"{deck} {100 * each[deck]['share']:.1f}%" for deck in lowest)
        )
    lines.append(f"  The run took {number(found['hours'])}.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--points", required=True)
    args = parser.parse_args()
    points = [int(point) for point in args.points.split(",")]
    found = report(args.directory, points)
    (args.directory / "report.json").write_text(
        json.dumps(found, indent=1, default=str), encoding="utf8"
    )
    print(shown(found, points))


if __name__ == "__main__":
    main()
