"""The readings of three of the ways ygo-agent's trainer differs from this
one, each tried on its own (experiments/2026-10-16-training-differences.md).

    uv run python tools/training_differences_report.py analysis/training-differences \\
        --began 240 --read 720 840 960 --smaller 4 \\
        --before analysis/more-decks/heldout/decks-01.curve.json

The directory is what `PARTS=measure` of
`experiments/commands/training-differences.sh` leaves: each run's
checkpoints after the games of each of `--read` of the control's updates,
and the control's own, on the held-out deals (`heldout/NAME.json`), with
each run's figures (`runs/NAME.metrics.jsonl`).  The runs began from a copy
after `--began` updates; `small-updates` makes `--smaller` updates for each
of the control's.  A deal is read by the share of a run's three checkpoints
that win it, and a won game is a win (a drawn one is not).

The readings, fixed on 2026-10-16, each run on its own, against the control
on the same deals, deal by deal: better when it wins at least 2 points and
two standard errors more, worse at 2 points and two standard errors fewer,
otherwise not told apart.

Beside them, without a mark: each checkpoint; the control as it was
measured before the rule against repeated commands the runs trained under
(`--before`), against itself under the rule; and what each run's updates
were like.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ppo_head_report import difference
from search_leads_report import percent, points

CONTROL = "control"
RUNS = ("small-updates", "their-step", "upgo")
TITLES = {
    "small-updates": "Smaller updates",
    "their-step": "Their minibatch and clip",
    "upgo": "UPGO",
}
# The mark of the readings: so far over or under, and in standard errors.
MARK = 0.02
ERRORS = 2.0
# What each run's updates were like: these figures, averaged over its run.
FIGURES = ("duel_approximate_kl", "duel_clip_fraction", "duel_entropy", "upgo_advantage_mean")


def scale(name: str, smaller: int) -> int:
    return smaller if name == "small-updates" else 1


def held(path: Path, wanted: dict[int, int]) -> tuple[dict[int, list[float]], Any]:
    """Checkpoints on the held-out deals, by the control's updates whose
    games they had played, and what tells the deals."""
    found = json.loads(path.read_text(encoding="utf8"))
    by_step = {
        int(point["checkpoint_step"]): [float(o == "win") for o in point["deals"]["outcomes"]]
        for point in found["checkpoints"]
    }
    games = {}
    for point, step in wanted.items():
        if step not in by_step:
            raise SystemExit(f"{path}: no checkpoint after {step} updates in all")
        games[point] = by_step[step]
    return games, (found.get("fingerprint"), found.get("games"))


def shares(games: dict[int, list[float]]) -> list[float]:
    """For each deal, the share of the checkpoints that win it."""
    return [sum(deal) / len(deal) for deal in zip(*games.values(), strict=True)]


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def verdict(gap: float, error: float) -> str:
    if gap >= MARK and gap >= ERRORS * error:
        return "better"
    if gap <= -MARK and -gap >= ERRORS * error:
        return "worse"
    return "not told apart"


def figures(path: Path, smaller: int, updates: int) -> dict[str, float]:
    """A run's first `updates` updates, those its checkpoints read had made:
    the seconds of each of the control's updates' worth, and the figures of
    FIGURES where it reports them."""
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line.strip()
    ][:updates]
    found = {"updates": float(len(rows))}
    if rows:
        found["seconds"] = smaller * mean([float(row["wall_seconds"]) for row in rows])
        for key in FIGURES:
            values = [float(row[key]) for row in rows if key in row]
            if values:
                found[key] = mean(values)
    return found


def report(
    directory: Path, began: int, read: list[int], smaller: int, before: Path | None
) -> dict[str, Any]:
    result: dict[str, Any] = {"read": read, "runs": {}, "readings": {}}
    control_path = directory / "heldout" / f"{CONTROL}.json"
    if not control_path.is_file():
        raise SystemExit(f"{control_path}: the control is not measured")
    control, dealt = held(control_path, {point: began + point for point in read})
    result["control"] = {str(point): mean(games) for point, games in control.items()}
    result["control"]["three"] = mean(shares(control))
    if before is not None:
        earlier, told = held(before, {point: began + point for point in read})
        if told[1] != dealt[1]:
            raise SystemExit(f"{before}: not the same held-out deals")
        gap, error = difference(shares(control), shares(earlier))
        result["control_before"] = {"three": mean(shares(earlier)), "gap": gap, "error": error}
    for name in RUNS:
        path = directory / "heldout" / f"{name}.json"
        if not path.is_file():
            continue
        times = scale(name, smaller)
        games, told = held(path, {point: began + times * point for point in read})
        if told != dealt:
            raise SystemExit(f"{name} played other deals than the control")
        gap, error = difference(shares(games), shares(control))
        result["runs"][name] = {str(point): mean(found) for point, found in games.items()}
        result["runs"][name]["three"] = mean(shares(games))
        metrics = directory / "runs" / f"{name}.metrics.jsonl"
        if metrics.is_file():
            result["runs"][name]["figures"] = figures(metrics, times, times * read[-1])
        result["readings"][name] = {"gap": gap, "error": error, "verdict": verdict(gap, error)}
    metrics = directory / "runs" / f"{CONTROL}.metrics.jsonl"
    if metrics.is_file():
        result["control"]["figures"] = figures(metrics, 1, read[-1])
    return result


def show(result: dict[str, Any]) -> str:
    read = result["read"]
    header = "  ".join(f"{point:>6}" for point in read)
    lines = [
        "The held-out deals, by the games of the control's updates each checkpoint had played:",
        f"  {'Run':<26}{header}  The three",
    ]
    rows = [("The control", result["control"])] + [
        (TITLES[name], found) for name, found in result["runs"].items()
    ]
    for title, found in rows:
        cells = "  ".join(f"{percent(found[str(point)]):>6}" for point in read)
        lines.append(f"  {title:<26}{cells}  {percent(found['three']):>9}")
    if "control_before" in result:
        found = result["control_before"]
        lines.append(
            f"The control as measured before the rule against repeated commands: "
            f"{percent(found['three'])}; under it {points(found['gap'], found['error'])}."
        )
    lines.append(
        "What each run's updates were like (the seconds of each of the control's updates' worth):"
    )
    for title, found in rows:
        if "figures" not in found:
            continue
        figures = found["figures"]
        parts = [f"{figures['updates']:.0f} updates", f"{figures.get('seconds', 0):.1f} s"]
        parts += [f"{key} {figures[key]:.4f}" for key in FIGURES if key in figures]
        lines.append(f"  {title:<26}" + ", ".join(parts))
    lines.append("The readings, each run against the control, deal by deal:")
    for index, (name, found) in enumerate(result["readings"].items(), start=1):
        lines.append(
            f"  {index}. {TITLES[name]}: {points(found['gap'], found['error'])}: {found['verdict']}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("directory", type=Path)
    parser.add_argument("--began", type=int, required=True, help="the copy's updates")
    parser.add_argument("--read", type=int, nargs=3, required=True)
    parser.add_argument("--smaller", type=int, default=4)
    parser.add_argument("--before", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = report(args.directory, args.began, args.read, args.smaller, args.before)
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(show(result))


if __name__ == "__main__":
    main()
