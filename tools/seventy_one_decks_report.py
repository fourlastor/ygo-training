"""The readings of seventy-one decks (experiments/2026-10-19-seventy-one-decks.md).

    uv run python tools/seventy_one_decks_report.py analysis/seventy-one-decks \\
        --output analysis/seventy-one-decks/report.json

The directory is what `PARTS=measure` of
`experiments/commands/seventy-one-decks.sh` leaves: the network of the 71
decks at each point of `--points` (`all71-POINT`, by its checkpoints after
the last three `--step` of its updates of PPO there) and the controls (half
of the experiment on thirteen decks, all of thirty-one decks, after the last
three steps of `--control-point`), on the screen of each of the thirteen
decks (`screens/RUN.DECK.json`) and on the eighteen other decks of thirty-one
(`old/RUN.json`); the network and all on the forty decks new to the pool
(`new/RUN.json`); the network's copy at its last checkpoint and the decks' own
pilots on the eighteen and the forty (`RUN-copy.json`, `pilot.json`); and the
runs' own figures (`RUN.metrics.jsonl`).

A game is read by the share of a run's three checkpoints that win it, a won
game a win (a drawn one is not).

The readings, fixed on 2026-10-19 before anything was built, R1, R2 and R3
the three points:

1. The thirteen kept: the network at R3 against half over the thirteen
   screens: equivalent when the gap less two standard errors is above -2
   points, worse when the gap and two standard errors are below -2 points,
   otherwise not decided.
2. The eighteen kept: the network at R2 against all on the eighteen decks
   against the thirteen decks' pilots, by the rule of 1.
3. The new decks: the network at R3 with the forty against those decks' own
   pilots on the same games: over at 2 points and two standard errors more,
   under at as many fewer, otherwise not told apart.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from card_descriptions_report import kept_rule
from ppo_head_report import difference
from search_leads_report import percent, points
from thirteen_decks_report import DECKS, MARK, Games, figures, mean, points_of, verdict

NETWORK = "all71"
HALF = "half"
ALL = "all"
PILOT = "pilot"
COPY = f"{NETWORK}-copy"


def runs_of(
    point_list: list[int], copied: int, half_copied: int, all_copied: int, control_point: int
) -> dict[str, tuple[int, int]]:
    """Each run read, by what it is called: the updates of its copy and the
    point of PPO it is read at."""
    found = {f"{NETWORK}-{point}": (copied, point) for point in point_list}
    found[HALF] = (half_copied, control_point)
    found[ALL] = (all_copied, control_point)
    return found


def report(
    directory: Path,
    thirteen: list[str],
    point_list: list[int],
    copied: int,
    half_copied: int = 3120,
    all_copied: int = 7440,
    control_point: int = 1920,
    step: int = 120,
) -> dict[str, Any]:
    if len(point_list) != 3:
        raise SystemExit("Three points are read: R1, R2 and R3")
    runs = runs_of(point_list, copied, half_copied, all_copied, control_point)
    r1, r2, r3 = (f"{NETWORK}-{point}" for point in point_list)
    result: dict[str, Any] = {
        "thirteen": thirteen,
        "points": point_list,
        "readings": {},
        "beside": {},
    }
    readings, beside = result["readings"], result["beside"]

    def steps(run: str) -> dict[str, int]:
        before, point = runs[run]
        return {f"{run}@{at}": before + at for at in points_of(point, step)}

    def three(found: Games, run: str) -> list[float] | None:
        return found.three(run, points_of(runs[run][1], step))

    def against(games: dict[str, list[float] | None], one: str, other: str) -> dict | None:
        if games.get(one) is None or games.get(other) is None:
            return None
        gap, error = difference(games[one], games[other])
        return {"gap": gap, "error": error, "games": len(games[other])}

    # The screens of the thirteen.
    screens: dict[str, Games] = {}
    for deck in thirteen:
        found = Games(f"the screen of {deck}")
        for run in runs:
            found.add(directory / "screens" / f"{run}.{deck}.json", steps(run))
        screens[deck] = found
    on_screens: dict[str, list[float] | None] = {}
    for run in runs:
        games: list[float] | None = []
        for found in screens.values():
            told = three(found, run)
            if told is None:
                games = None
                break
            games.extend(told)
        on_screens[run] = games
    result["screens"] = {run: mean(games) for run, games in on_screens.items() if games}
    result["by_screen"] = {
        deck: {run: mean(told) for run in runs if (told := three(found, run))}
        for deck, found in screens.items()
    }
    found = against(on_screens, r3, HALF)
    if found is not None:
        readings["thirteen"] = {**found, "verdict": kept_rule(found["gap"], found["error"])}
    for run in (r1, r2):
        if (told := against(on_screens, run, HALF)) is not None:
            beside[f"the screens at {run} against half"] = told

    # The eighteen other decks of thirty-one, and the forty new ones.
    kinds = {"old": "the eighteen other decks of thirty-one", "new": "the forty new decks"}
    played: dict[str, dict[str, list[float] | None]] = {}
    decks_of: dict[str, list[str]] = {}
    for kind, what in kinds.items():
        found = Games(what)
        for run in runs:
            found.add(directory / kind / f"{run}.json", steps(run))
        found.add(directory / kind / f"{COPY}.json", {COPY: copied})
        found.add(directory / kind / f"{PILOT}.json", None)
        games = {run: three(found, run) for run in runs}
        games[COPY] = found.games.get(COPY)
        games[PILOT] = found.games.get(PILOT)
        played[kind] = games
        result[kind] = {run: mean(told) for run, told in games.items() if told}
        if found.deals is not None:
            decks_of[kind] = [
                case.split("-vs-")[0] for case, count in found.deals[1] for _ in range(count)
            ]
    old, new = played["old"], played["new"]
    found = against(old, r2, ALL)
    if found is not None:
        readings["eighteen"] = {**found, "verdict": kept_rule(found["gap"], found["error"])}
    found = against(new, r3, PILOT)
    if found is not None:
        readings["forty"] = {
            **found,
            "verdict": verdict(found["gap"], found["error"], MARK, "over", "under"),
        }
    pairs = {
        f"the eighteen at {r1} against all": (old, r1, ALL),
        f"the eighteen at {r3} against all": (old, r3, ALL),
        f"the eighteen at {r3} against their pilots": (old, r3, PILOT),
        f"the forty at {r1} against their pilots": (new, r1, PILOT),
        f"the forty at {r2} against their pilots": (new, r2, PILOT),
        f"the forty at {r3} against all, which never trained on them": (new, r3, ALL),
        "all on the forty against their pilots": (new, ALL, PILOT),
        "the copy on the forty against their pilots": (new, COPY, PILOT),
    }
    for name, (games, one, other) in pairs.items():
        if (told := against(games, one, other)) is not None:
            beside[name] = told
    result["by_deck"] = {}
    for kind, playing in decks_of.items():
        for name in sorted(set(playing)):
            places = [index for index, deck in enumerate(playing) if deck == name]
            row: dict[str, Any] = {"kind": kind}
            for run in (PILOT, ALL, COPY, r1, r2, r3):
                told = played[kind].get(run)
                if told is not None:
                    row[run] = mean([told[index] for index in places])
            result["by_deck"][name] = row
    for run, metrics in ((NETWORK, NETWORK), (HALF, HALF), (ALL, ALL)):
        told = figures(directory / f"{metrics}.metrics.jsonl", step)
        if told is not None:
            result.setdefault("figures", {})[run] = told
    return result


def show(result: dict[str, Any]) -> str:
    r1, r2, r3 = (f"{NETWORK}-{point}" for point in result["points"])
    lines = ["The thirteen screens together, each run by its last three checkpoints there:"]
    for run, rate in result.get("screens", {}).items():
        lines.append(f"  {run:<24}{percent(rate):>8}")
    for kind, what in (
        ("old", "The eighteen other decks of thirty-one, against the thirteen decks' pilots:"),
        ("new", "The forty new decks, against the thirteen decks' pilots:"),
    ):
        if result.get(kind):
            lines.append(what)
            for run, rate in result[kind].items():
                lines.append(f"  {run:<24}{percent(rate):>8}")
    if result["beside"]:
        lines.append("Beside the readings, the same games:")
        for name, found in result["beside"].items():
            lines.append(f"  {name:<58}{points(found['gap'], found['error']):>14}")
    if result.get("by_deck"):
        runs = (PILOT, ALL, COPY, r1, r2, r3)
        lines.append("Deck by deck, against the thirteen decks' pilots:")
        lines.append(f"  {'':18}" + "".join(f"{run[-11:]:>12}" for run in runs))
        for name, row in result["by_deck"].items():
            lines.append(f"  {name:<18}" + "".join(f"{percent(row.get(run)):>12}" for run in runs))
    if result.get("by_screen"):
        lines.append("Screen by screen:")
        for deck, row in result["by_screen"].items():
            lines.append(
                f"  {deck:<14}" + "".join(f"{run}: {percent(rate)}  " for run, rate in row.items())
            )
    if "figures" in result:
        lines.append("The runs' own figures over their last updates: an update, duels an hour.")
        for run, found in result["figures"].items():
            lines.append(
                f"  {run:<24}{found['update_seconds']:>6.1f} s{found['duels_an_hour']:>10,.0f}"
            )
    readings = result["readings"]
    lines.append("The readings:")
    for key, label in (
        ("thirteen", f"1. the thirteen screens, {r3} against half"),
        ("eighteen", f"2. the eighteen other decks of thirty-one, {r2} against all"),
        ("forty", f"3. the forty new decks, {r3} against their pilots"),
    ):
        found = readings.get(key)
        if found is None:
            lines.append(f"  {label}: not measured")
        else:
            lines.append(
                f"  {label}: {points(found['gap'], found['error'])} over {found['games']:,} games:"
                f" {found['verdict']}"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("directory", type=Path)
    parser.add_argument("--thirteen", nargs="+", default=list(DECKS))
    parser.add_argument("--points", nargs=3, type=int, default=[1920, 4440, 10560])
    parser.add_argument("--copied", type=int, default=71 * 240, help="the network's copy's updates")
    parser.add_argument("--half-copied", type=int, default=3120)
    parser.add_argument("--all-copied", type=int, default=7440)
    parser.add_argument("--control-point", type=int, default=1920)
    parser.add_argument("--step", type=int, default=120, help="the updates between checkpoints")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = report(
        args.directory,
        args.thirteen,
        args.points,
        args.copied,
        args.half_copied,
        args.all_copied,
        args.control_point,
        args.step,
    )
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(show(result))


if __name__ == "__main__":
    main()
