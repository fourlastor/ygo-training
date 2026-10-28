"""The readings of the network of the seventy-one decks
(experiments/2026-10-28-seventy-one-champions.md).

    uv run python tools/seventy_one_champions_report.py analysis/seventy-one-champions \\
        --decks blackwing,cyber,... --champions analysis/champions \\
        --teachers analysis/fourteen-decks/teachers

The directory is what `PARTS=copy-measure` of
`experiments/commands/seventy-one-champions.sh` leaves. A pair
`STEM.plays.json` and `STEM.held.json` is a suite played with the first
network named in the learner's seat and then the second.

- `heads-taught/DECK`: the copy's last checkpoint head to head against each
  teacher in its mirror;
- `taught-DECK/POINT`: the copy after POINT updates against `all71` in each
  mirror;
- `pilots/taught.json`, `pilots/all71.json`: the copy's last checkpoint and
  `all71`, each deck against each of the thirteen decks' pilots;
- `taught.metrics.jsonl`.

Each teacher's own figure against `all71` in its mirror is its run's, from
its making (`CHAMPIONS/DECK/report.json`, tools/champions_report.py); for
the decks of the champions' release made before that report, its figure as
the fourteen decks measured it (`TEACHERS/DECK`, the same suites and seed).

The reading, fixed on 2026-10-28 before anything was built:

1. The copy keeps the teachers: head to head against each teacher in its
   mirror, by the last checkpoint, every deal from both seats: kept when the
   mean over the seventy-one mirrors is at least 48% by two standard errors
   and no mirror is at most 45% by two standard errors, lost when the mean
   is at most 48% by two standard errors or a mirror at most 45%, and
   otherwise not decided.

Told beside it, without a mark: each mirror against `all71` less its
teacher's figure; the copy along its updates; against the pilots beside
`all71`; the hours.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fourteen_decks_report import against_pilots, joined, kept_mirrors, number
from specialist_distillation_report import shares, spent, stems
from two_decks_report import mean, paired, told


def teacher_figure(deck: str, champions: Path, teachers: Path) -> dict[str, Any] | None:
    """A teacher's own share against `all71` in its mirror and its error:
    its run's, from its making, or as the fourteen decks measured it."""
    report_ = champions / deck / "report.json"
    if report_.is_file():
        found = json.loads(report_.read_text(encoding="utf8"))[deck]
        return {"share": found["share"], "error": found["error"], "from": "its making"}
    if (teachers / f"{deck}.held.json").is_file():
        return {**told(shares(teachers / deck)), "from": "the fourteen decks"}
    return None


def curve(directory: Path, decks: list[str]) -> list[dict[str, Any]]:
    """The copy along its updates against `all71` in each mirror: each
    deck's share, their mean and the weakest."""
    found = []
    for point in stems(directory / f"taught-{decks[0]}"):
        each = {deck: mean(shares(directory / f"taught-{deck}" / str(point))) for deck in decks}
        weakest = min(each, key=lambda deck: each[deck])
        found.append(
            {
                "updates": point,
                "mean": mean(list(each.values())),
                "weakest": weakest,
                "shares": each,
            }
        )
    return found


def against_teachers(
    directory: Path, decks: list[str], champions: Path, teachers: Path, last: int
) -> dict[str, Any]:
    """Each mirror against `all71` by the copy's last checkpoint, less its
    teacher's figure, the two measured apart."""
    each = {}
    for deck in decks:
        ours = told(shares(directory / f"taught-{deck}" / str(last)))
        theirs = teacher_figure(deck, champions, teachers)
        each[deck] = {**ours, "teacher": theirs}
        if theirs is not None:
            each[deck]["gap"] = ours["share"] - theirs["share"]
            each[deck]["gap error"] = math.hypot(ours["error"], theirs["error"])
    known = [one for one in each.values() if "gap" in one]
    return {
        "each": each,
        "gap": mean([one["gap"] for one in known]) if known else None,
        "error": joined([one["gap error"] for one in known]) if known else None,
        "unknown": [deck for deck, one in each.items() if "gap" not in one],
    }


def report(directory: Path, decks: list[str], champions: Path, teachers: Path) -> dict[str, Any]:
    heads = {deck: told(shares(directory / "heads-taught" / deck)) for deck in decks}
    along = curve(directory, decks)
    result: dict[str, Any] = {
        "teachers": {**kept_mirrors(heads), "each": heads},
        "curve": along,
        "against all71": against_teachers(
            directory, decks, champions, teachers, along[-1]["updates"]
        ),
    }
    pilots = directory / "pilots"
    if (pilots / "taught.json").is_file() and (pilots / "all71.json").is_file():
        ours = against_pilots(pilots / "taught.json", decks)
        theirs = against_pilots(pilots / "all71.json", decks)
        result["pilots"] = {
            "mean": ours["mean"],
            "decks": ours["decks"],
            "all71": theirs["mean"],
            "all71 decks": theirs["decks"],
            **paired(ours["games"], theirs["games"]),
        }
    result["hours"] = spent(directory, "taught")
    return result


def shown(result: dict[str, Any], decks: list[str]) -> str:
    teachers = result["teachers"]
    each = teachers["each"]
    lowest = min(each, key=lambda deck: each[deck]["share"])
    last = result["curve"][-1]["updates"]
    floor = teachers["under the floor"]
    lines = [
        f"The copy of the seventy-one decks, by its checkpoint after {last:,} updates:",
        (
            f"  1. The teachers: {teachers['verdict']}. Head to head against each teacher in its "
            f"mirror, the mean {teachers['mean']:.1%} ± {teachers['error']:.1%}, the lowest "
            f"{lowest} {each[lowest]['share']:.1%} ± {each[lowest]['error']:.1%}"
            + (f"; at most 45% by two standard errors: {', '.join(floor)}" if floor else "")
        ),
        "Beside it:",
        "  Head to head against each teacher: "
        + ", ".join(f"{deck} {each[deck]['share']:.1%}" for deck in decks),
        "  Against all71 in each mirror along the copy, the mean over the decks (the weakest): "
        + ", ".join(
            f"{point['updates']:,}: {point['mean']:.1%} ({point['weakest']} "
            f"{point['shares'][point['weakest']]:.0%})"
            for point in result["curve"]
        ),
    ]
    against = result["against all71"]
    if against["gap"] is not None:
        lines.append(
            f"  Against all71 in each mirror after {last:,} updates, less its teacher's figure: "
            f"the mean {100 * against['gap']:+.1f} ± {100 * against['error']:.1f} points"
        )
        lines.append(
            "    "
            + ", ".join(
                f"{deck} {100 * one['gap']:+.1f} ({one['share']:.1%} to {one['teacher']['share']:.1%})"
                for deck, one in against["each"].items()
                if "gap" in one
            )
        )
    if against["unknown"]:
        lines.append(f"    No teacher's figure: {', '.join(against['unknown'])}")
    if "pilots" in result:
        pilots = result["pilots"]
        weakest = sorted(pilots["decks"], key=lambda deck: pilots["decks"][deck])[:5]
        lines.append(
            f"  Against the thirteen decks' pilots {pilots['mean']:.1%}, all71 {pilots['all71']:.1%} "
            f"on the same games: {100 * pilots['gap']:+.1f} ± {100 * pilots['error']:.1f} points; "
            "the weakest decks "
            + ", ".join(
                f"{deck} {pilots['decks'][deck]:.1%} ({pilots['all71 decks'][deck]:.1%})"
                for deck in weakest
            )
        )
    lines.append(f"  The copy took {number(result['hours'])}.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--decks", required=True)
    parser.add_argument("--champions", type=Path, default=Path("analysis/champions"))
    parser.add_argument("--teachers", type=Path, default=Path("analysis/fourteen-decks/teachers"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    decks = [deck.strip() for deck in args.decks.split(",") if deck.strip()]
    result = report(args.directory, decks, args.champions, args.teachers)
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf8")
    print(shown(result, decks))


if __name__ == "__main__":
    main()
