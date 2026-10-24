"""The readings of thirty-one decks (experiments/2026-10-18-thirty-one-decks.md).

    uv run python tools/thirty_one_decks_report.py analysis/thirty-one-decks \\
        --output analysis/thirty-one-decks/report.json

The directory is what `PARTS=measure` of
`experiments/commands/thirty-one-decks.sh` leaves: each arm (plus-a, the
decks of training and the nine of `--half-a`; plus-b, the same with those of
`--half-b`; all, the thirty-one) and the control (half of the experiment on
thirteen decks) by its checkpoints after the last three `--step` of
`--updates` updates of PPO, on the screen of each deck of training
(`screens/RUN.DECK.json`) and on the decks outside them
(`unseen/RUN.json`); each one's copy at its last checkpoint and the decks'
own pilots on those decks too (`unseen/RUN-copy.json`, `unseen/pilot.json`);
and each one's own figures (`RUN.metrics.jsonl`, the control's over as many
updates as the arms').

A game is read by the share of a run's three checkpoints that win it, a won
game a win (a drawn one is not). A deck outside the decks of training is
held out by the arm of the other half, and trained on by its own half's arm
and by all.

The readings, fixed on 2026-10-18 before anything was built, each by an
arm's checkpoints after 1,680, 1,800 and 1,920 updates of PPO, game by game:

1. More decks help the decks never trained on: plus-a on the decks of
   half B and plus-b on those of half A, the eighteen together, against
   the control on the same games: better at 2 points and two standard
   errors more, worse at as many fewer, otherwise not told apart.
2. The decks of training kept: each arm against the control over the
   thirteen screens: equivalent when the gap less two standard errors is
   above -2 points, worse when the gap and two standard errors are below
   -2 points, otherwise not decided.
3. A network of all the decks on the eighteen it now trains on: all
   against those decks' own pilots on the same games: over at 2 points and
   two standard errors more, under at as many fewer, otherwise not told
   apart.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from card_descriptions_report import kept_rule, played_by
from ppo_head_report import difference
from search_leads_report import percent, points
from thirteen_decks_report import DECKS, MARK, Games, figures, mean, points_of, verdict

ARMS = ("plus-a", "plus-b", "all")
CONTROL = "control"
PILOT = "pilot"
HALF_A = (
    "arcana", "draconic-might", "harpie", "karakuri", "pyramid", "quickdraw-plant",
    "six-samurai", "tele-dad", "x-saber",
)  # fmt: skip
HALF_B = (
    "countdown", "destiny-hero", "fortune-lady", "gravekeeper", "machina", "ojama", "toon",
    "verdict", "watt",
)  # fmt: skip
NAMES = {
    "plus-a": "plus-a: thirteen and half A",
    "plus-b": "plus-b: thirteen and half B",
    "all": "all: the thirty-one",
    CONTROL: "the control, half: thirteen",
    PILOT: "the decks' own pilots",
    "held": "the arm that held the deck out",
    "trained": "the arm of 22 that trained on it",
}


def report(
    directory: Path,
    decks: list[str],
    half_a: list[str],
    half_b: list[str],
    copy: int = 240,
    updates: int = 1920,
    step: int = 120,
) -> dict[str, Any]:
    end = points_of(updates, step)
    copied = {
        "plus-a": (len(decks) + len(half_a)) * copy,
        "plus-b": (len(decks) + len(half_b)) * copy,
        "all": (len(decks) + len(half_a) + len(half_b)) * copy,
        CONTROL: len(decks) * copy,
    }
    runs = (*ARMS, CONTROL)
    result: dict[str, Any] = {
        "decks": decks,
        "half_a": half_a,
        "half_b": half_b,
        "end": end,
        "readings": {},
    }

    def steps(run: str) -> dict[str, int]:
        return {f"{run}@{point}": copied[run] + point for point in end}

    # The screens of the decks of training.
    screens: dict[str, Games] = {}
    for deck in decks:
        found = Games(f"the screen of {deck}")
        for run in runs:
            found.add(directory / "screens" / f"{run}.{deck}.json", steps(run))
        screens[deck] = found

    def together(run: str) -> list[float] | None:
        games: list[float] = []
        for found in screens.values():
            told = found.three(run, end)
            if told is None:
                return None
            games.extend(told)
        return games

    on_screens = {run: together(run) for run in runs}
    result["screens"] = {run: mean(games) for run, games in on_screens.items() if games}
    result["by_screen"] = {
        deck: {run: mean(told) for run in runs if (told := found.three(run, end))}
        for deck, found in screens.items()
    }
    readings = result["readings"]
    for arm in ARMS:
        if on_screens[arm] is not None and on_screens[CONTROL] is not None:
            gap, error = difference(on_screens[arm], on_screens[CONTROL])
            readings.setdefault("kept", {})[arm] = {
                "gap": gap,
                "error": error,
                "games": len(on_screens[CONTROL]),
                "verdict": kept_rule(gap, error),
            }

    # The decks outside training.
    unseen = Games("the decks outside training")
    for run in runs:
        unseen.add(directory / "unseen" / f"{run}.json", steps(run))
        unseen.add(directory / "unseen" / f"{run}-copy.json", {f"{run}-copy": copied[run]})
    unseen.add(directory / "unseen" / f"{PILOT}.json", None)
    if unseen.deals is None:
        return result
    playing = played_by(unseen)
    strange = sorted(set(playing) - set(half_a) - set(half_b))
    if strange:
        raise SystemExit(f"Decks outside training in neither half: {', '.join(strange)}")
    outside = {run: unseen.three(run, end) for run in runs}
    outside |= {f"{run}-copy": unseen.games.get(f"{run}-copy") for run in runs}
    outside[PILOT] = unseen.games.get(PILOT)

    def by_half(a_games: list[float] | None, b_games: list[float] | None) -> list[float] | None:
        """Each game of a deck of half A read from `a_games`, of half B
        from `b_games`."""
        if a_games is None or b_games is None:
            return None
        return [
            a_games[index] if deck in half_a else b_games[index]
            for index, deck in enumerate(playing)
        ]

    # Held out: a deck of half A by plus-b, one of half B by plus-a; trained
    # on: the other way round.
    outside["held"] = by_half(outside["plus-b"], outside["plus-a"])
    outside["trained"] = by_half(outside["plus-a"], outside["plus-b"])
    outside["held-copy"] = by_half(outside["plus-b-copy"], outside["plus-a-copy"])
    outside["trained-copy"] = by_half(outside["plus-a-copy"], outside["plus-b-copy"])
    result["unseen"] = {name: mean(games) for name, games in outside.items() if games}

    control, pilot = outside[CONTROL], outside[PILOT]
    if outside["held"] is not None and control is not None:
        gap, error = difference(outside["held"], control)
        readings["held"] = {
            "gap": gap,
            "error": error,
            "games": len(control),
            "verdict": verdict(gap, error, MARK, "better", "worse"),
        }
    if outside["all"] is not None and pilot is not None:
        gap, error = difference(outside["all"], pilot)
        readings["all"] = {
            "gap": gap,
            "error": error,
            "games": len(pilot),
            "verdict": verdict(gap, error, MARK, "over", "under"),
        }

    # Beside the readings.
    beside: dict[str, Any] = {}
    pairs = {
        "held against the control, its copies": ("held-copy", f"{CONTROL}-copy"),
        "trained against the pilots": ("trained", PILOT),
        "trained against held": ("trained", "held"),
        "held against the pilots": ("held", PILOT),
        "control against the pilots": (CONTROL, PILOT),
        "all against trained": ("all", "trained"),
    }
    for name, (one, other) in pairs.items():
        if outside.get(one) is not None and outside.get(other) is not None:
            gap, error = difference(outside[one], outside[other])
            beside[name] = {"gap": gap, "error": error}
    for half, members in (("half A", half_a), ("half B", half_b)):
        places = [index for index, deck in enumerate(playing) if deck in members]
        if outside["held"] is not None and control is not None and places:
            gap, error = difference(
                [outside["held"][index] for index in places],
                [control[index] for index in places],
            )
            beside[f"held against the control, {half}"] = {"gap": gap, "error": error}
    result["beside"] = beside
    result["by_deck"] = {}
    for name in sorted(set(playing)):
        places = [index for index, deck in enumerate(playing) if deck == name]
        row = {"half": "A" if name in half_a else "B"}
        for run in (PILOT, CONTROL, "held", "trained", "all"):
            if outside.get(run) is not None:
                row[run] = mean([outside[run][index] for index in places])
        result["by_deck"][name] = row
    for run in runs:
        found = figures(directory / f"{run}.metrics.jsonl", step)
        if found is not None:
            result.setdefault("figures", {})[run] = found
    return result


def show(result: dict[str, Any]) -> str:
    end = ", ".join(map(str, result["end"]))
    lines = [
        "The decks outside the thirteen of training, each against their pilots: each network",
        f"by its checkpoints after {end} updates of PPO, its copy at its last checkpoint.",
    ]
    unseen = result.get("unseen", {})
    for run in (CONTROL, "held", "trained", "all", PILOT):
        if run in unseen:
            copy = unseen.get(f"{run}-copy")
            lines.append(
                f"  {NAMES[run]:<36}{percent(unseen[run]):>8}"
                + ("" if copy is None else f"   its copy {percent(copy)}")
            )
    if result.get("beside"):
        lines.append("Beside the readings, the same games:")
        for name, found in result["beside"].items():
            lines.append(f"  {name:<46}{points(found['gap'], found['error']):>14}")
    if result.get("by_deck"):
        lines.append(
            "Deck by deck: its half, its pilot, the control, the arm that held it out, the arm"
        )
        lines.append("of 22 that trained on it, and all.")
        for name, row in result["by_deck"].items():
            lines.append(
                f"  {name:<16}{row['half']:>3}"
                + "".join(
                    f"{percent(row.get(run)):>9}"
                    for run in (PILOT, CONTROL, "held", "trained", "all")
                )
            )
    if result.get("screens"):
        lines.append(f"The {len(result['decks'])} screens of the decks of training together:")
        for run, rate in result["screens"].items():
            lines.append(f"  {NAMES[run]:<36}{percent(rate):>8}")
    if "figures" in result:
        lines.append("The runs' own figures over their last updates: an update, duels an hour.")
        for run, found in result["figures"].items():
            lines.append(
                f"  {NAMES[run]:<36}{found['update_seconds']:>6.1f} s"
                f"{found['duels_an_hour']:>10,.0f}"
            )
    readings = result["readings"]
    lines.append("The readings:")
    if "held" in readings:
        found = readings["held"]
        lines.append(
            "  1. the arm that held each deck out against the control, the eighteen together: "
            f"{points(found['gap'], found['error'])} over {found['games']:,} games: "
            f"{found['verdict']}"
        )
    else:
        lines.append("  1. the decks held out: not measured")
    for arm in ARMS:
        found = readings.get("kept", {}).get(arm)
        label = f"  2. {arm} against the control on the screens"
        if found is None:
            lines.append(f"{label}: not measured")
        else:
            lines.append(
                f"{label}: {points(found['gap'], found['error'])} over {found['games']:,} games: "
                f"{found['verdict']}"
            )
    if "all" in readings:
        found = readings["all"]
        lines.append(
            "  3. all against the pilots of the eighteen: "
            f"{points(found['gap'], found['error'])} over {found['games']:,} games: "
            f"{found['verdict']}"
        )
    else:
        lines.append("  3. all against the pilots of the eighteen: not measured")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("directory", type=Path)
    parser.add_argument("--decks", nargs="+", default=list(DECKS))
    parser.add_argument("--half-a", nargs="+", default=list(HALF_A))
    parser.add_argument("--half-b", nargs="+", default=list(HALF_B))
    parser.add_argument("--copy", type=int, default=240, help="a copy's updates for each deck")
    parser.add_argument("--updates", type=int, default=1920, help="an arm's updates of PPO")
    parser.add_argument("--step", type=int, default=120, help="the updates between checkpoints")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = report(
        args.directory,
        args.decks,
        args.half_a,
        args.half_b,
        args.copy,
        args.updates,
        args.step,
    )
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(show(result))


if __name__ == "__main__":
    main()
