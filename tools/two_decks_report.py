"""The readings of a network of two decks at the champions' duels of each
mirror (experiments/2026-10-24-two-decks.md).

    uv run python tools/two_decks_report.py analysis/two-decks \\
        --cyber-alone analysis/cyber-champion/cyber-old/mirror \\
        --calibration analysis/cyber-champion/calibration

The directory is what `PARTS=measure` of `experiments/commands/two-decks.sh`
leaves. Each measurement is a pair of files on the 2,048 deals of a suite:
`STEM.plays.json`, the first network named in the learner's seat and the
second in the other, and `STEM.held.json`, the other way round, STEM the
first network's updates of PPO. A deal's two games are read together, and
its games at a folder's three points too.

- `blackwing/` and `cyber/`: the network of both decks (`both`) against each
  deck's champion in its mirror, `both` after 1,920, 2,400 and 2,880 updates
  of PPO and the champion after 480, 600 and 720, as many games of the
  mirror: Blackwing alone (`blackwing`) and `cyber-old`;
- `all71-blackwing/` and `all71-cyber/`: `both` against `all71` in each
  mirror; `all71-alone/`: Blackwing alone against `all71` in its mirror;
- `decks-01/`: `decks-01` of More decks, folded, against Blackwing alone in
  its mirror, both after 480, 600 and 720 updates of PPO;
- `cross/`: `both` playing Blackwing against `all71` playing Cyber, then
  `all71` playing Blackwing against `both` playing Cyber;
- `both-copy.metrics.jsonl`, `both.metrics.jsonl`,
  `blackwing-copy.metrics.jsonl` and `blackwing.metrics.jsonl`: the runs.

`--cyber-alone` is where `cyber-old` was played against `all71` in the Cyber
mirror, for the Cyber champion; `--calibration`, where `cyber-new` and
`cyber-old` were played head to head: two runs of the recipe.

The readings, fixed on 2026-10-24 before anything was built:

1. Blackwing: the network of both against Blackwing alone in the Blackwing
   mirror, by their checkpoints at as many games of it (the network of both
   after 1,920, 2,400 and 2,880 updates of PPO, Blackwing alone after 480,
   600 and 720), every deal from both seats, a deal's games together: it
   matches when its share less two standard errors is at least 48%, is
   weaker when its share plus two standard errors is at most 48%, and is
   otherwise not decided.
2. Cyber: the same in the Cyber mirror, against `cyber-old`.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

MATCHES = 0.48
ERRORS = 2.0
# Each reading: its folder, its deck and the champion there.
READINGS = (("blackwing", "Blackwing", "Blackwing alone"), ("cyber", "Cyber", "cyber-old"))
RUNS = ("both-copy", "both", "blackwing-copy", "blackwing")


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def error(values: list[float]) -> float:
    centre = mean(values)
    return math.sqrt(sum((value - centre) ** 2 for value in values) / (len(values) - 1)) / (
        math.sqrt(len(values))
    )


def outcomes(path: Path) -> list[str]:
    if not path.is_file():
        raise SystemExit(f"Missing {path}")
    (record,) = json.loads(path.read_text(encoding="utf8"))["records"]
    (case,) = record["evaluation_suite"].values()
    return case["outcomes"]


def deals(folder: Path) -> list[float]:
    """The first network's share of each deal, its two games at each of the
    folder's points together."""
    found: list[list[float]] = []
    for plays in sorted(folder.glob("*.plays.json")):
        held = plays.with_name(plays.name.replace(".plays.", ".held."))
        won = [float(told == "win") for told in outcomes(plays)]
        # The second network plays these in the learner's seat: the first
        # wins what it loses.
        lost = [float(told == "loss") for told in outcomes(held)]
        if len(won) != len(lost):
            raise SystemExit(f"{plays.name} and {held.name} play other deals")
        found.append([(one + other) / 2 for one, other in zip(won, lost, strict=True)])
    if not found:
        raise SystemExit(f"Nothing is measured in {folder}")
    return [mean(list(told)) for told in zip(*found, strict=True)]


def told(shares: list[float]) -> dict[str, Any]:
    return {"share": mean(shares), "error": error(shares), "deals": len(shares)}


def paired(ours: list[float], theirs: list[float]) -> dict[str, float]:
    """Our share less theirs, deal by deal."""
    if len(ours) != len(theirs):
        raise SystemExit("The two were played on other deals")
    gap = [one - other for one, other in zip(ours, theirs, strict=True)]
    return {"gap": mean(gap), "error": error(gap)}


def verdict(share: float, error_: float) -> str:
    """Readings 1 and 2."""
    if share - ERRORS * error_ >= MATCHES:
        return "matches"
    if share + ERRORS * error_ <= MATCHES:
        return "weaker"
    return "not decided"


def hours(directory: Path, run: str) -> float | None:
    path = directory / f"{run}.metrics.jsonl"
    if not path.is_file():
        return None
    rows = [json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line]
    return sum(float(row["wall_seconds"]) for row in rows) / 3600


def report(
    directory: Path, cyber_alone: Path | None = None, calibration: Path | None = None
) -> dict[str, Any]:
    result: dict[str, Any] = {"readings": {}, "beside": {}}
    for folder, deck, champion in READINGS:
        found = told(deals(directory / folder))
        found["champion"] = champion
        found["verdict"] = verdict(found["share"], found["error"])
        result["readings"][deck] = found
    beside = result["beside"]
    ours, theirs = deals(directory / "all71-blackwing"), deals(directory / "all71-alone")
    beside["Blackwing mirror against all71"] = {
        "both": told(ours),
        "alone": told(theirs),
        "paired": paired(ours, theirs),
    }
    ours = deals(directory / "all71-cyber")
    beside["Cyber mirror against all71"] = {"both": told(ours)}
    if cyber_alone is not None:
        theirs = deals(cyber_alone)
        beside["Cyber mirror against all71"].update(alone=told(theirs), paired=paired(ours, theirs))
    beside["Blackwing against Cyber, against all71"] = told(deals(directory / "cross"))
    beside["decks-01 against Blackwing alone"] = told(deals(directory / "decks-01"))
    if calibration is not None:
        beside["cyber-new against cyber-old"] = told(deals(calibration))
    result["hours"] = {run: spent for run in RUNS if (spent := hours(directory, run)) is not None}
    return result


def shown(result: dict[str, Any]) -> str:
    lines = []
    for number, (deck, found) in enumerate(result["readings"].items(), start=1):
        lines.append(
            f"{number}. {deck}: {found['verdict']}. The network of both wins "
            f"{found['share']:.1%} ± {found['error']:.1%} of the {deck} mirror against "
            f"{found['champion']} ({found['deals']:,} deals, each from both seats)."
        )
    lines.append("Beside them:")
    for name, found in result["beside"].items():
        if "both" in found:
            line = f"  the {name}: the network of both {found['both']['share']:.1%}"
            if "alone" in found:
                gap = found["paired"]
                line += (
                    f", the deck alone {found['alone']['share']:.1%}: "
                    f"{100 * gap['gap']:+.1f} ± {100 * gap['error']:.1f} points, the deals paired"
                )
            lines.append(line)
        else:
            lines.append(f"  {name}: {found['share']:.1%} ± {found['error']:.1%}")
    spent = ", ".join(f"{run} {value:.2f} h" for run, value in result["hours"].items())
    if spent:
        lines.append(f"  hours: {spent}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--cyber-alone", type=Path, help="cyber-old against all71 in its mirror")
    parser.add_argument("--calibration", type=Path, help="cyber-new against cyber-old")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = report(args.directory, args.cyber_alone, args.calibration)
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2))
    print(shown(result))


if __name__ == "__main__":
    main()
