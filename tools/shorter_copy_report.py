"""The reading of a shorter copy (experiments/2026-10-26-shorter-copy.md): the
network of four decks trained on from its copy's checkpoint after 640
updates, beside the uniform arm, trained on from the copy's last.

    uv run python tools/shorter_copy_report.py analysis/four-decks/network \\
        --decks blackwing,cyber,lightsworn,infernity

The directory is what `PARTS=short-measure` of
`experiments/commands/four-decks.sh` leaves beside the uniform arm's
measurements (`tools/four_decks_report.py`):

- `short-pairings/POINT.json`, `uniform-pairings/POINT.json`: each arm's
  last three checkpoints on every ordered pairing against `all71`, 1,024
  games each;
- `short-quick/POINT.json`, `uniform-quick/POINT.json`: every checkpoint of
  each arm on 128 games of each, and the copy's (`weighted-weights/0.json`);
- `heads-short/DECK`, `heads-uniform/DECK`: each arm's last checkpoint
  against each champion in its mirror;
- `taught.metrics.jsonl`, `uniform.metrics.jsonl`, `short.metrics.jsonl`.

The reading, fixed on 2026-10-26 before anything was built: the shorter
copy costs nothing. The short arm against the uniform arm, by their last
three checkpoints, by the mean over the ten pairings of the learner's share
against `all71` (a mirror from both seats, two decks the mean of their two
seatings), the deals paired: as good when the two are within 1 point of
each other by two standard errors, worse when the short arm is at least 1
point under by two standard errors, and otherwise not decided.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from four_decks_report import ERRORS, averaged, compared, over_points, pairings, quick_curve
from specialist_distillation_report import shares, spent
from two_decks_report import mean, told

# So many points between the arms, either way.
WITHIN = 0.01
SHORT = 640


def verdict(gap: float, error_: float) -> str:
    if gap - ERRORS * error_ >= -WITHIN and gap + ERRORS * error_ <= WITHIN:
        return "as good"
    if gap + ERRORS * error_ <= -WITHIN:
        return "worse"
    return "not decided"


def report(directory: Path, decks: list[str]) -> dict[str, Any]:
    arms = {}
    for arm in ("short", "uniform"):
        folder = directory / f"{arm}-pairings"
        points = sorted(int(path.stem) for path in folder.glob("*.json"))
        if len(points) != 3:
            raise SystemExit(f"{folder} holds {len(points)} checkpoints, not the last three")
        arms[arm] = over_points(folder, points, decks)
    found = compared(arms["short"], arms["uniform"])
    result: dict[str, Any] = {
        "shorter": {
            **found,
            "short": averaged(arms["short"]),
            "uniform": averaged(arms["uniform"]),
            "verdict": verdict(found["gap"], found["error"]),
            "by pairing": {
                f"{one}|{other}": {arm: mean(arms[arm][(one, other)]) for arm in arms}
                for one, other in pairings(decks)
            },
        },
        "beside": {},
    }
    zero = directory / "weighted-weights"
    start = quick_curve(zero, decks)[:1] if (zero / "0.json").is_file() else []
    result["beside"]["curves"] = {
        arm: start + quick_curve(directory / f"{arm}-quick", decks) for arm in ("short", "uniform")
    }
    result["beside"]["head to head"] = {
        arm: {deck: told(shares(directory / f"heads-{arm}" / deck))["share"] for deck in decks}
        for arm in ("short", "uniform")
        if (directory / f"heads-{arm}").is_dir()
    }
    result["hours"] = {
        "the copy to 640": spent(directory, "taught", SHORT),
        "the copy to its end": spent(directory, "taught"),
        "short": spent(directory, "short"),
        "uniform": spent(directory, "uniform"),
    }
    return result


def number(value: float | None) -> str:
    return "not measured" if value is None else f"{value:.2f} h"


def shown(result: dict[str, Any]) -> str:
    found = result["shorter"]
    lines = [
        (
            f"The shorter copy: {found['verdict']}. The mean over the pairings against all71: "
            f"from 640 {found['short']:.1%}, from 960 (uniform) {found['uniform']:.1%}: "
            f"{100 * found['gap']:+.1f} ± {100 * found['error']:.1f} points, the deals paired"
        ),
        "Beside it:",
        "  Each pairing against all71 by the arms' last three checkpoints:",
    ]
    for pairing, shares_ in found["by pairing"].items():
        lines.append(
            f"    {pairing.replace('|', ' against '):<28} from 640 {shares_['short']:.1%}, "
            f"from 960 {shares_['uniform']:.1%}"
        )
    for arm, points in result["beside"]["curves"].items():
        told_ = ", ".join(
            f"{point['updates']}: {point['mean']:.1%} ({point['weakest']} "
            f"{point['shares'][point['weakest']]:.0%})"
            for point in points
        )
        lines.append(f"  The {arm} arm along its updates, the mean (the weakest): {told_}")
    for arm, heads in result["beside"]["head to head"].items():
        told_ = ", ".join(f"{deck} {share:.1%}" for deck, share in heads.items())
        lines.append(f"  Head to head against the champions, {arm}: {told_}")
    spent_ = "; ".join(f"{name} {number(value)}" for name, value in result["hours"].items())
    lines.append(f"  Hours: {spent_}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--decks", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    decks = [deck.strip() for deck in args.decks.split(",") if deck.strip()]
    result = report(args.directory, decks)
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf8")
    print(shown(result))


if __name__ == "__main__":
    main()
