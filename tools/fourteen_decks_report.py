"""The readings of the network of fourteen decks
(experiments/2026-10-26-fourteen-decks.md).

    uv run python tools/fourteen_decks_report.py analysis/fourteen-decks \\
        --decks blackwing,cyber,lightsworn,infernity,gishki,burn,crystal,dragunity,gladiator,heroes,monarch,morphtronic,rock-block,spellcaster

The directory is what `PARTS=measure`, and `recipe-measure` where the recipe
arm is made, of `experiments/commands/fourteen-decks.sh` leave. A pair
`STEM.plays.json` and `STEM.held.json` is a suite played with the first
network named in the learner's seat and then the second; a single
`POINT.json` is a suite of every ordered pairing, the learner's network
against `all71` on the other side.

- `teachers/DECK`, `taught-DECK/POINT`: each teacher, and the copy after
  POINT updates, against `all71` in each mirror; `copy-quick/0.json`: the
  copy's last checkpoint on 64 games of every ordered pairing;
- `ARM-pairings/POINT.json`: an arm's last three checkpoints on every
  ordered pairing, 1,024 games each; `ARM-quick/POINT.json`: every
  checkpoint it keeps, on 64 games of each;
- `heads-ARM/DECK`: an arm's last checkpoint against each champion in its
  mirror;
- `pilots/ARM.json`, `pilots/all71.json`: an arm's last checkpoint and
  `all71`, each deck against each of the thirteen decks' pilots;
- `four-decks/ARM`: an arm's last checkpoint against the network of four
  decks on its pairings, both ways;
- `taught.metrics.jsonl`, `ARM.metrics.jsonl`.

The readings, fixed on 2026-10-26 before anything was built, of the selfplay
arm and, where it is made, the recipe arm:

1. The mirrors are kept: head to head against each champion in its mirror,
   by the last checkpoint, every deal from both seats: kept when the mean
   over the fourteen mirrors is at least 48% by two standard errors and no
   mirror is at most 45% by two standard errors, lost when the mean is at
   most 48% by two standard errors or a mirror at most 45%, and otherwise
   not decided.
2. It beats all71 on every pairing: each pairing's share against `all71` by
   the last three checkpoints, a deal's games at the three together (a
   mirror from both seats, two decks the mean of their two seatings): it
   beats all71 when every pairing is over 50% by two standard errors, falls
   short when one is at most 50% by two standard errors, and is otherwise
   not decided.
3. Where the recipe arm is made, the selfplay arm against it by the mean
   over the pairings of the share against `all71`, by their last three
   checkpoints, the deals paired: as good when the two are within 1 point
   of each other by two standard errors, worse when the selfplay arm is at
   least 1 point under by two standard errors, and otherwise not decided.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from four_decks_report import averaged, compared, over_points, pairing_games, quick_curve
from shorter_copy_report import verdict as within
from specialist_distillation_report import shares, spent, stems
from two_decks_report import error, mean, paired, told

ERRORS = 2.0
# Reading 1: the mean over the mirrors, and the floor of each one.
MIRRORS = 0.48
FLOOR = 0.45
# Reading 2: each pairing over an even share.
EVEN = 0.50
ARMS = ("selfplay", "recipe", "quarter", "half")
# The arms that give the mirrors a share of their duels, each read against
# the selfplay arm.
MIRRORED = ("quarter", "half")
# The decks of the network of four decks.
FOUR = ["blackwing", "cyber", "lightsworn", "infernity"]


def joined(errors: list[float]) -> float:
    """The standard error of a mean of measurements made apart."""
    return math.sqrt(sum(value**2 for value in errors)) / len(errors)


def kept_mirrors(each: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Reading 1, of each mirror's share head to head and its error."""
    centre = mean([found["share"] for found in each.values()])
    error_ = joined([found["error"] for found in each.values()])
    under = [
        deck for deck, found in each.items() if found["share"] + ERRORS * found["error"] <= FLOOR
    ]
    if under or centre + ERRORS * error_ <= MIRRORS:
        word = "lost"
    elif centre - ERRORS * error_ >= MIRRORS:
        word = "kept"
    else:
        word = "not decided"
    return {"mean": centre, "error": error_, "under the floor": under, "verdict": word}


def beats(each: dict[str, dict[str, Any]]) -> str:
    """Reading 2, of each pairing's share against `all71` and its error."""
    if any(found["share"] + ERRORS * found["error"] <= EVEN for found in each.values()):
        return "falls short"
    if all(found["share"] - ERRORS * found["error"] > EVEN for found in each.values()):
        return "beats all71"
    return "not decided"


def outcomes_of(path: Path) -> dict[str, list[str]]:
    """How each game of each case of a suite ended."""
    if not path.is_file():
        raise SystemExit(f"Missing {path}")
    (record,) = json.loads(path.read_text(encoding="utf8"))["records"]
    return {name: case["outcomes"] for name, case in record["evaluation_suite"].items()}


def head_to_head(stem: Path) -> dict[str, Any]:
    """The first network against the second on the pairings of four decks,
    each deal played both ways: its share of each pairing and their mean."""
    plays = outcomes_of(stem.with_name(f"{stem.name}.plays.json"))
    held = outcomes_of(stem.with_name(f"{stem.name}.held.json"))
    # The second network plays the held games in the learner's seat: the
    # first wins what it loses.
    deals = {
        name: [
            (float(one == "win") + float(other == "loss")) / 2
            for one, other in zip(plays[name], held[name], strict=True)
        ]
        for name in plays
    }
    games = pairing_games(deals, FOUR)
    return {
        "mean": averaged(games),
        "error": joined([error(values) for values in games.values()]),
        "by pairing": {f"{one}|{other}": mean(values) for (one, other), values in games.items()},
    }


def against_pilots(path: Path, decks: list[str]) -> dict[str, Any]:
    """Each deck's share against the thirteen decks' pilots, and the mean
    over the decks; and every game's result, in the suite's order."""
    played = outcomes_of(path)
    by_deck = {
        deck: mean(
            [
                float(game == "win")
                for name, games in played.items()
                if name.split("-vs-")[0] == deck
                for game in games
            ]
        )
        for deck in decks
    }
    return {
        "mean": mean(list(by_deck.values())),
        "decks": by_deck,
        "games": [float(game == "win") for name in sorted(played) for game in played[name]],
    }


def arm_report(
    directory: Path, arm: str, decks: list[str]
) -> tuple[dict[str, Any], dict[tuple[str, str], list[float]]]:
    """An arm's readings 1 and 2 and what is told beside them, and each
    pairing's games against `all71` for reading 3."""
    heads = {deck: told(shares(directory / f"heads-{arm}" / deck)) for deck in decks}
    folder = directory / f"{arm}-pairings"
    points = sorted(int(path.stem) for path in folder.glob("*.json"))
    if len(points) != 3:
        raise SystemExit(f"{folder} holds {len(points)} checkpoints, not the last three")
    games = over_points(folder, points, decks)
    each = {pairing: told(values) for pairing, values in games.items()}
    weakest = min(each, key=lambda pairing: each[pairing]["share"])
    found: dict[str, Any] = {
        "points": points,
        "mirrors": {**kept_mirrors(heads), "each": heads},
        "pairings": {
            "verdict": beats(each),
            "mean": averaged(games),
            "weakest": "|".join(weakest),
            "each": {f"{one}|{other}": shares_ for (one, other), shares_ in each.items()},
        },
        "curve": quick_curve(directory / "copy-quick", decks)
        + quick_curve(directory / f"{arm}-quick", decks),
        "hours": spent(directory, arm),
    }
    pilots = directory / "pilots"
    if (pilots / f"{arm}.json").is_file() and (pilots / "all71.json").is_file():
        ours = against_pilots(pilots / f"{arm}.json", decks)
        theirs = against_pilots(pilots / "all71.json", decks)
        found["pilots"] = {
            "mean": ours["mean"],
            "decks": ours["decks"],
            "all71": theirs["mean"],
            "all71 decks": theirs["decks"],
            **paired(ours["games"], theirs["games"]),
        }
    if (directory / "four-decks" / f"{arm}.held.json").is_file():
        found["four decks"] = head_to_head(directory / "four-decks" / arm)
    return found, games


def report(directory: Path, decks: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {"arms": {}}
    games = {}
    for arm in ARMS:
        if (directory / f"heads-{arm}").is_dir():
            result["arms"][arm], games[arm] = arm_report(directory, arm, decks)
    if not result["arms"]:
        raise SystemExit(f"No arm is measured in {directory}")
    if "selfplay" in games and "recipe" in games:
        found = compared(games["selfplay"], games["recipe"])
        result["selfplay against recipe"] = {
            **found,
            "verdict": within(found["gap"], found["error"]),
        }
    for arm in MIRRORED:
        if "selfplay" in games and arm in games:
            found = compared(games[arm], games["selfplay"])
            result[f"{arm} against selfplay"] = {
                **found,
                "verdict": within(found["gap"], found["error"]),
            }
    teachers = {deck: shares(directory / "teachers" / deck) for deck in decks}
    result["copy"] = {
        deck: [
            {
                "updates": point,
                "share": mean(shares(directory / f"taught-{deck}" / str(point))),
                "teacher": mean(teachers[deck]),
                **paired(shares(directory / f"taught-{deck}" / str(point)), teachers[deck]),
            }
            for point in stems(directory / f"taught-{deck}")
        ]
        for deck in decks
    }
    if (directory / "heads-taught").is_dir():
        result["copy head to head"] = copy_head_to_head(directory, decks)
    result["hours"] = {"the copy": spent(directory, "taught")}
    return result


def copy_head_to_head(directory: Path, decks: list[str]) -> dict[str, Any]:
    """The copy's last checkpoint head to head against each champion in its
    mirror, on the deals the arms played them: each share and their mean, and
    each arm less the copy, deal by deal."""
    arms = [arm for arm in ARMS if (directory / f"heads-{arm}").is_dir()]
    each = {}
    for deck in decks:
        copy = shares(directory / "heads-taught" / deck)
        each[deck] = told(copy)
        for arm in arms:
            each[deck][f"{arm} less the copy"] = paired(
                shares(directory / f"heads-{arm}" / deck), copy
            )
    found: dict[str, Any] = {
        "mean": mean([one["share"] for one in each.values()]),
        "error": joined([one["error"] for one in each.values()]),
        "each": each,
    }
    for arm in arms:
        gaps = [one[f"{arm} less the copy"] for one in each.values()]
        found[f"{arm} less the copy"] = {
            "gap": mean([gap["gap"] for gap in gaps]),
            "error": joined([gap["error"] for gap in gaps]),
        }
    return found


def number(value: float | None) -> str:
    return "not measured" if value is None else f"{value:.2f} h"


def shown(result: dict[str, Any], decks: list[str]) -> str:
    lines = []
    for arm, found in result["arms"].items():
        mirrors, pairings_ = found["mirrors"], found["pairings"]
        lowest = min(mirrors["each"], key=lambda deck: mirrors["each"][deck]["share"])
        weakest = pairings_["each"][pairings_["weakest"]]
        lines += [
            f"The {arm} arm, by its checkpoints after {', '.join(map(str, found['points']))} updates:",
            (
                f"  1. The mirrors: {mirrors['verdict']}. Head to head against each champion, the "
                f"mean {mirrors['mean']:.1%} ± {mirrors['error']:.1%}, the lowest {lowest} "
                f"{mirrors['each'][lowest]['share']:.1%} ± {mirrors['each'][lowest]['error']:.1%}"
            ),
            (
                f"  2. Against all71: {pairings_['verdict']}. The mean over the pairings "
                f"{pairings_['mean']:.1%}, the weakest {pairings_['weakest'].replace('|', ' against ')} "
                f"{weakest['share']:.1%} ± {weakest['error']:.1%}"
            ),
        ]
    if "selfplay against recipe" in result:
        found = result["selfplay against recipe"]
        lines.append(
            f"3. The selfplay arm against the recipe arm: {found['verdict']}, "
            f"{100 * found['gap']:+.1f} ± {100 * found['error']:.1f} points over the pairings against "
            "all71, the deals paired"
        )
    for arm in MIRRORED:
        if f"{arm} against selfplay" in result:
            found = result[f"{arm} against selfplay"]
            lines.append(
                f"4. The {arm} arm against the selfplay arm: {found['verdict']}, "
                f"{100 * found['gap']:+.1f} ± {100 * found['error']:.1f} points over the pairings "
                "against all71, the deals paired"
            )
    lines.append("Beside them:")
    for arm, found in result["arms"].items():
        heads = ", ".join(
            f"{deck} {each['share']:.1%}" for deck, each in found["mirrors"]["each"].items()
        )
        lines.append(f"  The {arm} arm head to head against each champion: {heads}")
        lines.append(f"  The {arm} arm against all71, each pairing by its last three checkpoints:")
        each = found["pairings"]["each"]
        for place, one in enumerate(decks):
            told_ = ", ".join(
                f"{other} {each[f'{one}|{other}']['share']:.1%}" for other in decks[place:]
            )
            lines.append(f"    {one}: {told_}")
        curve = ", ".join(
            f"{point['updates']}: {point['mean']:.1%} ({point['weakest']} "
            f"{point['shares'][point['weakest']]:.0%})"
            for point in found["curve"]
        )
        lines.append(f"  The {arm} arm along its updates, the mean (the weakest): {curve}")
        if "pilots" in found:
            pilots = found["pilots"]
            by_deck = ", ".join(
                f"{deck} {share:.1%} ({pilots['all71 decks'][deck]:.1%})"
                for deck, share in pilots["decks"].items()
            )
            lines.append(
                f"  The {arm} arm against the thirteen decks' pilots {pilots['mean']:.1%}, all71 "
                f"{pilots['all71']:.1%} ({100 * pilots['gap']:+.1f} ± {100 * pilots['error']:.1f} "
                f"points, the games paired); each deck (all71's): {by_deck}"
            )
        if "four decks" in found:
            four = found["four decks"]
            by_pairing = ", ".join(
                f"{pairing.replace('|', ' against ')} {share:.1%}"
                for pairing, share in four["by pairing"].items()
            )
            lines.append(
                f"  The {arm} arm head to head against the network of four decks on its pairings: "
                f"{four['mean']:.1%} ± {four['error']:.1%}; {by_pairing}"
            )
    if "copy head to head" in result:
        found = result["copy head to head"]
        arms = [arm for arm in ARMS if f"{arm} less the copy" in found]

        def gaps(one: dict[str, Any]) -> str:
            told_ = ", ".join(
                f"{arm} {100 * one[f'{arm} less the copy']['gap']:+.1f}" for arm in arms
            )
            return f" ({told_})" if told_ else ""

        each = ", ".join(
            f"{deck} {one['share']:.1%}{gaps(one)}" for deck, one in found["each"].items()
        )
        line = (
            f"  The copy head to head against each champion, the mean {found['mean']:.1%} ± "
            f"{found['error']:.1%}"
        )
        for arm in arms:
            gap = found[f"{arm} less the copy"]
            line += (
                f"; the {arm} arm less the copy {100 * gap['gap']:+.1f} ± "
                f"{100 * gap['error']:.1f} points"
            )
        if arms:
            line += ", the deals paired"
        lines.append(f"{line}: {each}")
    for deck, points in result["copy"].items():
        told_ = ", ".join(
            f"{point['updates']}: {point['share']:.1%} ({100 * point['gap']:+.1f})"
            for point in points
        )
        lines.append(
            f"  The copy in the {deck} mirror against all71, over its teacher's "
            f"{points[0]['teacher']:.1%}: {told_}"
        )
    hours = [f"the copy {number(result['hours']['the copy'])}"] + [
        f"{arm} {number(found['hours'])}" for arm, found in result["arms"].items()
    ]
    lines.append(f"  Hours: {'; '.join(hours)}")
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
    print(shown(result, decks))


if __name__ == "__main__":
    main()
