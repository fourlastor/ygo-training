"""The readings of a batch of the expanded pool (experiments/2026-11-02-expanded-pool.md).

    uv run python tools/pool_batch_report.py analysis/expanded-pool/batch-one \\
        --points 4020,8040

The directory is what `experiments/commands/pool-batch.sh` leaves: at 0 and at each of
the points `pairings.json`, each new deck against the frozen network at each old deck;
at each point `old.json`, every ordered pairing of the old decks head to head against
it; at 0 and at the last point `held-out.json` and `pilots.json`; at the last point
`mirrors/`, each of the seventy-one mirrors head to head against its champion.

The readings, fixed on 2026-11-03 before anything was built:

1. The new decks learn: the mean over the new decks of their share against the frozen
   network at the last point, less at the start, the deals paired: learns when at
   least 5 points by two standard errors, does not when at most 2 points by two
   standard errors, otherwise not decided.
2. The old decks hold: the network head to head against the frozen one on the old
   decks' pairings at the last point: hold when at least 49% by two standard errors,
   slip when at most 49% by two standard errors, otherwise not decided.
3. The mirrors come back: the seventy-one mirrors head to head against their
   champions at the last point: kept when their mean is at least 48% by two standard
   errors and none is at most 45% by two standard errors, lost when the mean is at
   most 48% by two standard errors or one is at most 45% by two standard errors,
   otherwise not decided.

Told beside them, without a mark: the gain and the old decks at the points before the
last; each new deck's gain; the next decks, trained on nothing; the new decks against
the pilots; the hours.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from first_batch_report import cases, games, holds, learns
from fourteen_decks_report import kept_mirrors, number
from specialist_distillation_report import shares, spent
from two_decks_report import mean, paired, told


def report(directory: Path, points: list[int], run: str) -> dict[str, Any]:
    start = cases(directory / "0" / "pairings.json")
    tables = {point: cases(directory / str(point) / "pairings.json") for point in points}
    last = points[-1]
    gains = {point: paired(games(tables[point]), games(start)) for point in points}
    old = {point: told(games(cases(directory / str(point) / "old.json"))) for point in points}
    found: dict[str, Any] = {
        "start": mean(games(start)),
        "gains": gains,
        "each": {name: mean(tables[last][name]) - mean(start[name]) for name in sorted(start)},
        "learns": learns(gains[last]["gap"], gains[last]["error"]),
        "old": old,
        "holds": holds(old[last]["share"], old[last]["error"]),
    }
    mirrors = sorted((directory / str(last) / "mirrors").glob("*.held.json"))
    if mirrors:
        each = {
            path.name.split(".")[0]: told(shares(path.with_name(path.name.split(".")[0])))
            for path in mirrors
        }
        found["mirrors"] = {**kept_mirrors(each), "each": each}
    for part in ("held-out", "pilots"):
        at_last = directory / str(last) / f"{part}.json"
        if at_last.is_file():
            before = games(cases(directory / "0" / f"{part}.json"))
            found[part] = {"start": mean(before), **paired(games(cases(at_last)), before)}
    found["hours"] = spent(directory, run)
    return found


def shown(found: dict[str, Any], points: list[int]) -> str:
    last = points[-1]
    gain = found["gains"][last]
    old = found["old"][last]
    lines = [
        (
            f"1. A new deck {found['learns']}: at {last:,} updates "
            f"{100 * gain['gap']:+.1f} ± {100 * gain['error']:.1f} points against the frozen "
            f"network, from {100 * found['start']:.1f}%"
        ),
        (
            f"2. The old decks {found['holds']}: {100 * old['share']:.1f}% ± "
            f"{100 * old['error']:.1f}% head to head against the frozen network"
        ),
    ]
    if "mirrors" in found:
        mirrors = found["mirrors"]
        lines.append(
            f"3. The mirrors {mirrors['verdict']}: {100 * mirrors['mean']:.1f}% ± "
            f"{100 * mirrors['error']:.1f}% against their champions"
            + (
                f", under 45%: {', '.join(mirrors['under the floor'])}"
                if mirrors["under the floor"]
                else ""
            )
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
    ranked = sorted(found["each"].items(), key=lambda item: item[1])
    lines.append(
        "  The new decks' least gains: "
        + ", ".join(f"{name.split('-vs-')[0]} {100 * gap:+.1f}" for name, gap in ranked[:5])
        + "; most: "
        + ", ".join(f"{name.split('-vs-')[0]} {100 * gap:+.1f}" for name, gap in ranked[-3:])
    )
    if "mirrors" in found:
        each = found["mirrors"]["each"]
        lowest = sorted(each, key=lambda deck: each[deck]["share"])[:5]
        lines.append(
            "  The lowest mirrors: "
            + ", ".join(f"{deck} {100 * each[deck]['share']:.1f}%" for deck in lowest)
        )
    if "held-out" in found:
        told_ = found["held-out"]
        lines.append(
            f"  The next decks, trained on nothing: {100 * told_['gap']:+.1f} ± "
            f"{100 * told_['error']:.1f} points, from {100 * told_['start']:.1f}%"
        )
    if "pilots" in found:
        told_ = found["pilots"]
        lines.append(
            f"  The new decks against the pilots: {100 * (told_['start'] + told_['gap']):.1f}%, "
            f"from {100 * told_['start']:.1f}%, {100 * told_['gap']:+.1f} ± {100 * told_['error']:.1f}"
        )
    lines.append(f"  The run took {number(found['hours'])}.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--points", required=True)
    parser.add_argument("--run", default="batch-one", help="the run whose metrics give the hours")
    args = parser.parse_args()
    points = [int(point) for point in args.points.split(",")]
    found = report(args.directory, points, args.run)
    (args.directory / "report.json").write_text(
        json.dumps(found, indent=1, default=str), encoding="utf8"
    )
    print(shown(found, points))


if __name__ == "__main__":
    main()
