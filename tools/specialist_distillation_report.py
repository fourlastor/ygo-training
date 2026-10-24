"""The readings of specialist distillation
(experiments/2026-10-24-specialist-distillation.md).

    uv run python tools/specialist_distillation_report.py report \\
        analysis/specialist-distillation --both analysis/two-decks \\
        --cyber-champion analysis/cyber-champion
    uv run python tools/specialist_distillation_report.py check \\
        analysis/specialist-distillation --both analysis/two-decks \\
        --cyber-champion analysis/cyber-champion

The directory is what `experiments/commands/specialist-distillation.sh`
leaves, in the files of `tools/two_decks_report.py`: pairs `STEM.plays.json`
and `STEM.held.json` on the 2,048 deals of a suite, the first network named
in the learner's seat and then the second.

- `taught-blackwing/` and `taught-cyber/`: the copy (`taught`) against
  `all71` in each mirror, STEM its updates (320, 400 and 480);
- `teacher-blackwing/` and `teacher-cyber/`: the copy against each deck's
  teacher in its mirror, head to head;
- `taught-curve-blackwing/` and `taught-curve-cyber/`: the copy's
  checkpoints before its last three against `all71` in each mirror, beside
  the readings;
- `cross/`: the copy's last checkpoint (STEM 0) and those of `cross`, STEM
  its updates of PPO, playing either deck against `all71` playing the
  other; `cross-blackwing/` and `cross-cyber/`: `cross` against `all71` in
  each mirror;
- `check/`: games played again at the run's code before it: the network of
  both in Blackwing against Cyber after 1,920 updates of PPO
  (`cross-1920`), and each teacher against `all71` in its mirror
  (`blackwing-720`, `cyber-720`);
- `taught.metrics.jsonl`, `taught.run.json`, `cross.metrics.jsonl` and
  `cross.run.json`: the runs.

`--both` is where `tools/two_decks_report.py` read the network of both: its
curve in Blackwing against Cyber (`cross/`, STEM its updates of PPO), the
yardstick, Blackwing alone in its mirror (`all71-alone/`) and the runs.
`--cyber-champion` is where the Cyber champion's runs are, and `cyber-old`'s
games in its mirror (`cyber-old/mirror/`). `check` says whether the games
of `check/` end as those measured before.

The readings, fixed on 2026-10-24 before anything was built:

1. The copy keeps each champion's mirror: for each deck, the copy's share
   against `all71` in that deck's mirror, by its checkpoints after 320, 400
   and 480 updates, a deal's games at the three together, less its
   teacher's share on the same deals: it keeps the champion's mirror when
   the gap less two standard errors is above -2 points, loses strength when
   the gap plus two standard errors is below -2, and is otherwise not
   decided.
2. The games of the two decks against each other it needs: the first of the
   copy's last checkpoint and those of `cross` at which its share of
   Blackwing against Cyber against `all71`, less the network of both's (its
   last three checkpoints, the deals paired), less two standard errors, is
   above -2 points: the games of the two decks against each other it had
   played by then, beside the network of both's by the same rule. Where
   none meets it, it is not closed within the network of both's games.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from two_decks_report import error, mean, outcomes, paired, told

# Both readings: so many points, by two standard errors.
MARGIN = 0.02
ERRORS = 2.0
# The network of both's last three checkpoints, by its updates of PPO: what
# the second reading measures against.
BOTH_LAST = (1920, 2400, 2880)
# Each deck: its name, its folders' part, its teacher and where the
# teacher's own games against all71 in its mirror were measured before.
DECKS = (
    ("Blackwing", "blackwing", "Blackwing alone", "all71-alone/720"),
    ("Cyber", "cyber", "cyber-old", "cyber-old/mirror/model-00959"),
)


def shares(stem: Path) -> list[float]:
    """The first network's share of each deal of a pair, its two games
    together."""
    won = [float(game == "win") for game in outcomes(stem.with_name(f"{stem.name}.plays.json"))]
    # The second network plays these in the learner's seat: the first wins
    # what it loses.
    lost = [float(game == "loss") for game in outcomes(stem.with_name(f"{stem.name}.held.json"))]
    if len(won) != len(lost):
        raise SystemExit(f"{stem}'s two files play other deals")
    return [(one + other) / 2 for one, other in zip(won, lost, strict=True)]


def stems(folder: Path) -> list[int]:
    """The points measured in a folder, in their order."""
    found = sorted(int(path.name.split(".")[0]) for path in folder.glob("*.held.json"))
    if not found:
        raise SystemExit(f"Nothing is measured in {folder}")
    return found


def together(folder: Path, points: list[int] | tuple[int, ...]) -> list[float]:
    """Each deal's share at the points, their games together."""
    found = [shares(folder / str(point)) for point in points]
    return [mean(list(deal)) for deal in zip(*found, strict=True)]


def crossed(run: Path) -> float:
    """Of an update's duels, the games of two different decks against each
    other, by what the run says of its duels against itself."""
    told_ = json.loads(run.read_text(encoding="utf8"))
    decks = len(told_["learners"])
    pairs = told_.get("self_play_pairs", "all")
    share = {"all": (decks - 1) / decks, "mirrors": 0.0, "cross": 1.0}[pairs]
    return told_["self_play_duels"] * share


def keeps(gap: float, error_: float) -> str:
    """Reading 1, for a deck."""
    if gap - ERRORS * error_ > -MARGIN:
        return "keeps"
    if gap + ERRORS * error_ < -MARGIN:
        return "loses strength"
    return "not decided"


def curve(folder: Path, games: float, reference: list[float]) -> list[dict[str, Any]]:
    """Each point of a curve in Blackwing against Cyber: the games of the two
    decks against each other played by then, the share, the gap to the
    reference with the deals paired, and whether it is within the reading's
    margin."""
    points = []
    for point in stems(folder):
        found = shares(folder / str(point))
        gap = paired(found, reference)
        points.append(
            {
                "updates": point,
                "games": round(point * games),
                **told(found),
                "gap": gap["gap"],
                "gap error": gap["error"],
                "within": gap["gap"] - ERRORS * gap["error"] > -MARGIN,
            }
        )
    return points


def spent(directory: Path, run: str, updates: int | None = None) -> float | None:
    """The hours of a run's updates: all of them, or its first so many."""
    path = directory / f"{run}.metrics.jsonl"
    if not path.is_file():
        return None
    rows = [json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line]
    return sum(float(row["wall_seconds"]) for row in rows[:updates]) / 3600


def summed(*parts: float | None) -> float | None:
    """Hours added up, where all of them are known."""
    known = [part for part in parts if part is not None]
    return None if len(known) < len(parts) else sum(known)


def report(directory: Path, both: Path, cyber_champion: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"mirrors": {}, "cross": {}, "beside": {}}
    teachers = {}
    for deck, part, teacher, _ in DECKS:
        teachers[part] = shares(directory / "check" / f"{part}-720")
        copy = together(directory / f"taught-{part}", stems(directory / f"taught-{part}"))
        found = paired(copy, teachers[part])
        result["mirrors"][deck] = {
            "teacher": teacher,
            "copy": told(copy)["share"],
            "its teacher": told(teachers[part])["share"],
            **found,
            "verdict": keeps(found["gap"], found["error"]),
        }
        head = directory / f"teacher-{part}"
        result["beside"][f"{deck}, head to head"] = told(together(head, stems(head)))
    reference = together(both / "cross", BOTH_LAST)
    ours = curve(directory / "cross", crossed(directory / "cross.run.json"), reference)
    theirs = curve(both / "cross", crossed(both / "both.run.json"), reference)
    met = next((point for point in ours if point["within"]), None)
    known = next((point for point in theirs if point["within"]), None)
    result["cross"] = {
        "reference": told(reference)["share"],
        "reference error": error(reference),
        "games": None if met is None else met["games"],
        "updates": None if met is None else met["updates"],
        "games of both": None if known is None else known["games"],
        "updates of both": None if known is None else known["updates"],
        "verdict": (
            "not closed within the network of both's games"
            if met is None
            else f"closed after {met['games']:,} games of the two decks against each other"
        ),
        "curve": ours,
        "curve of both": theirs,
    }
    along = {}
    for deck, part, _, _ in DECKS:
        folder = directory / f"cross-{part}"
        if not folder.is_dir():
            continue
        points = []
        for point in stems(folder):
            found = shares(folder / str(point))
            points.append(
                {"updates": point, **told(found), "over its teacher": paired(found, teachers[part])}
            )
        along[deck] = points
    result["beside"]["each mirror along cross"] = along
    copied = {}
    for deck, part, _, _ in DECKS:
        points = []
        for folder in (directory / f"taught-curve-{part}", directory / f"taught-{part}"):
            if not folder.is_dir():
                continue
            for point in stems(folder):
                found = shares(folder / str(point))
                points.append(
                    {
                        "updates": point,
                        **told(found),
                        "over its teacher": paired(found, teachers[part]),
                    }
                )
        copied[deck] = sorted(points, key=lambda point: point["updates"])
    result["beside"]["the copy along its updates"] = copied
    champions = {
        "Blackwing alone": summed(spent(both, "blackwing-copy"), spent(both, "blackwing")),
        "cyber-old": summed(
            spent(cyber_champion, "cyber-old-copy"), spent(cyber_champion, "cyber-old")
        ),
    }
    taught = spent(directory, "taught")
    whole = spent(directory, "cross")
    reached = None if met is None else spent(directory, "cross", met["updates"])
    made = [value for value in champions.values() if value is not None]
    result["hours"] = {
        **champions,
        "taught": taught,
        "cross": whole,
        "cross, to the second reading": reached,
        "the whole way, the champions one after the other": summed(
            *champions.values(), taught, reached if met is not None else whole
        ),
        "the whole way, the champions at once": (
            summed(max(made), taught, reached if met is not None else whole)
            if len(made) == len(champions)
            else None
        ),
        "the network of both": summed(spent(both, "both-copy"), spent(both, "both")),
    }
    return result


def check(directory: Path, both: Path, cyber_champion: Path) -> list[tuple[str, int, int]]:
    """How many games of each file of `check/` end as those measured
    before them."""
    before = {
        "cross-1920": both / "cross" / "1920",
        "blackwing-720": both / DECKS[0][3],
        "cyber-720": cyber_champion / DECKS[1][3],
    }
    found = []
    for name, stored in before.items():
        for side in ("plays", "held"):
            again = outcomes(directory / "check" / f"{name}.{side}.json")
            then = outcomes(stored.with_name(f"{stored.name}.{side}.json"))
            if len(again) != len(then):
                raise SystemExit(f"{name}.{side} plays other deals than {stored}")
            same = sum(one == other for one, other in zip(again, then, strict=True))
            found.append((f"{name}.{side}", same, len(again)))
    return found


def number(value: float | None) -> str:
    return "not measured" if value is None else f"{value:.2f} h"


def shown(result: dict[str, Any]) -> str:
    lines = ["1. The copy keeps each champion's mirror, against all71, less its teacher:"]
    for deck, found in result["mirrors"].items():
        lines.append(
            f"   {deck}: {found['verdict']}. The copy {found['copy']:.1%}, {found['teacher']} "
            f"{found['its teacher']:.1%}: {100 * found['gap']:+.1f} ± "
            f"{100 * found['error']:.1f} points, the deals paired"
        )
    cross = result["cross"]
    lines.append(
        "2. The games of the two decks against each other it needs, within 2 points of the "
        f"network of both's {cross['reference']:.1%} by two standard errors: {cross['verdict']}"
        + ("" if cross["updates"] is None else f" (after {cross['updates']:,} updates of cross)")
    )
    if cross["games of both"] is not None:
        lines.append(
            f"   The network of both by the same rule: after {cross['games of both']:,} games "
            f"({cross['updates of both']:,} updates of its PPO)"
        )
    lines.append("Beside them:")
    lines.append("  Blackwing against Cyber, against all71, by the games of the two decks:")
    theirs = {point["games"]: point for point in cross["curve of both"]}
    for point in cross["curve"]:
        line = (
            f"    {point['games']:>7,}: {point['share']:.1%} ± {point['error']:.1%} "
            f"({100 * point['gap']:+.1f} ± {100 * point['gap error']:.1f} to the mark's "
            f"{cross['reference']:.1%})"
        )
        if point["games"] in theirs:
            line += f"; the network of both {theirs[point['games']]['share']:.1%}"
        lines.append(line)
    for deck, _, _, _ in DECKS:
        head = result["beside"][f"{deck}, head to head"]
        lines.append(
            f"  The copy against its {deck} teacher head to head: {head['share']:.1%} ± "
            f"{head['error']:.1%}"
        )
    for deck, points in result["beside"]["the copy along its updates"].items():
        told_ = ", ".join(
            f"{point['updates']}: {point['share']:.1%} "
            f"({100 * point['over its teacher']['gap']:+.1f})"
            for point in points
        )
        lines.append(f"  The copy along its updates, the {deck} mirror, over its teacher: {told_}")
    for deck, points in result["beside"]["each mirror along cross"].items():
        told_ = ", ".join(
            f"{point['updates']}: {point['share']:.1%} "
            f"({100 * point['over its teacher']['gap']:+.1f})"
            for point in points
        )
        lines.append(f"  The {deck} mirror against all71 along cross, over its teacher: {told_}")
    spent_ = "; ".join(f"{name} {number(value)}" for name, value in result["hours"].items())
    lines.append(f"  Hours: {spent_}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("what", choices=("report", "check"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("--both", type=Path, required=True)
    parser.add_argument("--cyber-champion", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.what == "check":
        found = check(args.directory, args.both, args.cyber_champion)
        for name, same, games in found:
            print(f"{name}: {same:,} of {games:,} end as they did")
        if any(same != games for _, same, games in found):
            raise SystemExit(
                "Some games end otherwise: the measurements made before are not paired"
            )
        return
    result = report(args.directory, args.both, args.cyber_champion)
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(shown(result))


if __name__ == "__main__":
    main()
