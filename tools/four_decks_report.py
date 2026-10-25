"""The readings of the network of four decks
(experiments/2026-10-24-four-decks.md, its part 2).

    uv run python tools/four_decks_report.py analysis/four-decks/network \\
        --decks blackwing,cyber,lightsworn,infernity \\
        --distilled analysis/specialist-distillation/cross --far 1920

The directory is what `PARTS=measure` of `experiments/commands/four-decks.sh`
leaves. A pair `STEM.plays.json` and `STEM.held.json` is a suite of one deck
against another, the first network named in the learner's seat and then the
second; a single `POINT.json` is a suite of every ordered pairing, the
learner's network against `all71` on the other side.

- `teachers/DECK`, `taught-DECK/POINT`, `taught-curve-DECK/POINT`: each
  teacher, and the copy after POINT updates, against `all71` in each mirror;
- `uniform-pairings/POINT.json`, `weighted-pairings/POINT.json`: each arm's
  last three checkpoints on every ordered pairing, 1,024 games each;
- `uniform-quick/POINT.json`, `weighted-weights/POINT.json`: every
  checkpoint of each arm on 128 games of each, the weighted arm's with the
  shares and weights made of them (`POINT.shares.json`);
- `uniform-cross/FAR`: the uniform arm after FAR updates in Blackwing against
  Cyber against `all71`, both ways; `--distilled` is where specialist
  distillation's cross pairings were measured the same way (`240`);
- `heads-WHO/DECK`: the copy's and each arm's last checkpoint against each
  champion in its mirror;
- `taught.metrics.jsonl`, `uniform.metrics.jsonl`, `weighted.metrics.jsonl`.

The readings, fixed on 2026-10-25 before anything was built:

1. The copy keeps each champion's mirror: for each deck, the copy's share
   against `all71` in that deck's mirror, by its last three checkpoints, a
   deal's games at the three together, less its teacher's on the same deals:
   it keeps the champion's mirror when the gap less two standard errors is
   above -2 points, loses strength when the gap plus two standard errors is
   below -2, and is otherwise not decided.
2. Weighting by weakness pays: the weighted arm against the uniform arm, by
   their last three checkpoints, by the mean over the pairings of the
   learner's share against `all71` (a mirror from both seats, two decks the
   mean of their two seatings), the deals paired: it pays when the weighted
   arm is at least 1 point over by two standard errors, costs when it is at
   least 1 point under, and is otherwise not told apart.
3. Four decks against two, for one pairing: Blackwing against Cyber against
   `all71`, the uniform arm after FAR updates against specialist
   distillation's cross pairings after 240, the deals paired: sharing four
   decks helps a pairing when the uniform arm is at least 2 points over by
   two standard errors, hurts it when at least 2 points under, and is
   otherwise not told apart.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from specialist_distillation_report import keeps, shares, spent, stems, together
from two_decks_report import mean, paired, told

ERRORS = 2.0
# Reading 2: so many points between the arms; reading 3: between a network
# of four decks and one of two.
PAYS = 0.01
HELPS = 0.02
ARMS = ("uniform", "weighted")


def cases(path: Path) -> dict[str, list[float]]:
    """Each case of a suite of every ordered pairing: the learner's result in
    each of its games."""
    if not path.is_file():
        raise SystemExit(f"Missing {path}")
    (record,) = json.loads(path.read_text(encoding="utf8"))["records"]
    return {
        name: [float(game == "win") for game in case["outcomes"]]
        for name, case in record["evaluation_suite"].items()
    }


def pairings(decks: list[str]) -> list[tuple[str, str]]:
    """The pairings of the decks, each once, a deck against itself among
    them, in the decks' order."""
    return [(one, other) for place, one in enumerate(decks) for other in decks[place:]]


def pairing_games(
    played: dict[str, list[float]], decks: list[str]
) -> dict[tuple[str, str], list[float]]:
    """Each pairing's games: a mirror's own, two decks' the two seatings'
    games of a deal together."""
    found = {}
    for one, other in pairings(decks):
        names = [f"{one}-vs-{other}-checkpoint"]
        if one != other:
            names.append(f"{other}-vs-{one}-checkpoint")
        missing = [name for name in names if name not in played]
        if missing:
            raise SystemExit(f"The pairings measured hold no {', '.join(missing)}")
        found[(one, other)] = [
            mean(list(games)) for games in zip(*(played[n] for n in names), strict=True)
        ]
    return found


def over_points(
    folder: Path, points: list[int], decks: list[str]
) -> dict[tuple[str, str], list[float]]:
    """Each pairing's games at the points, a deal's games at all of them
    together."""
    found = [pairing_games(cases(folder / f"{point}.json"), decks) for point in points]
    return {
        pairing: [mean(list(deal)) for deal in zip(*(each[pairing] for each in found), strict=True)]
        for pairing in found[0]
    }


def averaged(games: dict[tuple[str, str], list[float]]) -> float:
    """The mean over the pairings of each pairing's share."""
    return mean([mean(values) for values in games.values()])


def compared(
    ours: dict[tuple[str, str], list[float]], theirs: dict[tuple[str, str], list[float]]
) -> dict[str, float]:
    """The mean over the pairings of our share less theirs, deal by deal
    within each pairing, and its standard error."""
    gaps, variances = [], []
    for pairing, values in ours.items():
        difference = [one - other for one, other in zip(values, theirs[pairing], strict=True)]
        centre = mean(difference)
        gaps.append(centre)
        spread = sum((value - centre) ** 2 for value in difference) / (len(difference) - 1)
        variances.append(spread / len(difference))
    return {"gap": mean(gaps), "error": math.sqrt(sum(variances)) / len(gaps)}


def verdict(gap: float, error_: float, margin: float, words: tuple[str, str, str]) -> str:
    if gap - ERRORS * error_ >= margin:
        return words[0]
    if gap + ERRORS * error_ <= -margin:
        return words[1]
    return words[2]


def quick_curve(folder: Path, decks: list[str]) -> list[dict[str, Any]]:
    """An arm's mean and weakest pairing on the 128 games of each, at each of
    its points measured so."""
    found = []
    for path in sorted(folder.glob("*.json"), key=lambda path: int(path.name.split(".")[0])):
        if not path.name.split(".")[0].isdigit() or path.name.count(".") > 1:
            continue
        games = pairing_games(cases(path), decks)
        each = {f"{one}|{other}": mean(values) for (one, other), values in games.items()}
        weakest = min(each, key=each.__getitem__)
        found.append(
            {"updates": int(path.stem), "mean": averaged(games), "weakest": weakest, "shares": each}
        )
    return found


def report(directory: Path, decks: list[str], distilled: Path, far: int) -> dict[str, Any]:
    result: dict[str, Any] = {"mirrors": {}, "beside": {}}
    teachers = {deck: shares(directory / "teachers" / deck) for deck in decks}
    for deck in decks:
        folder = directory / f"taught-{deck}"
        copy = together(folder, stems(folder))
        found = paired(copy, teachers[deck])
        result["mirrors"][deck] = {
            "copy": told(copy)["share"],
            "its teacher": told(teachers[deck])["share"],
            **found,
            "verdict": keeps(found["gap"], found["error"]),
        }
    arms = {}
    for arm in ARMS:
        folder = directory / f"{arm}-pairings"
        points = sorted(int(path.stem) for path in folder.glob("*.json"))
        arms[arm] = over_points(folder, points, decks)
    weighting = compared(arms["weighted"], arms["uniform"])
    result["weighting"] = {
        **weighting,
        "uniform": averaged(arms["uniform"]),
        "weighted": averaged(arms["weighted"]),
        "verdict": verdict(
            weighting["gap"], weighting["error"], PAYS, ("pays", "costs", "not told apart")
        ),
        "by pairing": {
            f"{one}|{other}": {arm: mean(arms[arm][(one, other)]) for arm in ARMS}
            for one, other in pairings(decks)
        },
    }
    ours = shares(directory / "uniform-cross" / str(far))
    theirs = shares(distilled / "240")
    sharing = paired(ours, theirs)
    result["sharing"] = {
        **sharing,
        "four decks": told(ours)["share"],
        "two decks": told(theirs)["share"],
        "verdict": verdict(
            sharing["gap"], sharing["error"], HELPS, ("helps", "hurts", "not told apart")
        ),
    }
    zero = directory / "weighted-weights" / "0.json"
    curves = {}
    for arm, folder in (
        ("uniform", directory / "uniform-quick"),
        ("weighted", directory / "weighted-weights"),
    ):
        points = quick_curve(folder, decks)
        if zero.is_file() and arm == "uniform":
            points = quick_curve(zero.parent, decks)[:1] + points
        curves[arm] = points
    result["beside"]["curves"] = curves
    result["beside"]["weights"] = {
        int(path.name.split(".")[0]): json.loads(path.read_text(encoding="utf8"))["weights"]
        for path in sorted((directory / "weighted-weights").glob("*.shares.json"))
    }
    copy = {}
    for deck in decks:
        points = []
        for folder in (directory / f"taught-curve-{deck}", directory / f"taught-{deck}"):
            for point in stems(folder) if folder.is_dir() else []:
                found = shares(folder / str(point))
                points.append(
                    {
                        "updates": point,
                        **told(found),
                        "over its teacher": paired(found, teachers[deck]),
                    }
                )
        copy[deck] = sorted(points, key=lambda point: point["updates"])
    result["beside"]["the copy along its updates"] = copy
    result["beside"]["head to head"] = {
        who: {deck: told(shares(directory / f"heads-{who}" / deck))["share"] for deck in decks}
        for who in ("taught", *ARMS)
        if (directory / f"heads-{who}").is_dir()
    }
    measuring = sum(
        float(json.loads(path.read_text(encoding="utf8")).get("seconds", 0.0))
        for path in (directory / "weighted-weights").glob("*.json")
        if path.name.count(".") == 1
    )
    result["hours"] = {
        **{name: spent(directory, name) for name in ("taught", *ARMS)},
        "the weighted arm measuring": measuring / 3600,
    }
    return result


def number(value: float | None) -> str:
    return "not measured" if value is None else f"{value:.2f} h"


def shown(result: dict[str, Any]) -> str:
    lines = ["1. The copy keeps each champion's mirror, against all71, less its teacher:"]
    for deck, found in result["mirrors"].items():
        lines.append(
            f"   {deck}: {found['verdict']}. The copy {found['copy']:.1%}, its teacher "
            f"{found['its teacher']:.1%}: {100 * found['gap']:+.1f} ± {100 * found['error']:.1f} "
            "points, the deals paired"
        )
    weighting = result["weighting"]
    lines.append(
        f"2. Weighting by weakness: {weighting['verdict']}. The mean over the pairings against "
        f"all71: weighted {weighting['weighted']:.1%}, uniform {weighting['uniform']:.1%}: "
        f"{100 * weighting['gap']:+.1f} ± {100 * weighting['error']:.1f} points, the deals paired"
    )
    sharing = result["sharing"]
    lines.append(
        f"3. Four decks against two, Blackwing against Cyber at as many of its games: "
        f"{sharing['verdict']}. Four decks {sharing['four decks']:.1%}, two "
        f"{sharing['two decks']:.1%}: {100 * sharing['gap']:+.1f} ± "
        f"{100 * sharing['error']:.1f} points"
    )
    lines.append("Beside them:")
    lines.append("  Each pairing against all71 by the arms' last three checkpoints:")
    for pairing, found in weighting["by pairing"].items():
        lines.append(
            f"    {pairing.replace('|', ' against '):<28} uniform {found['uniform']:.1%}, "
            f"weighted {found['weighted']:.1%}"
        )
    for arm, points in result["beside"]["curves"].items():
        told_ = ", ".join(
            f"{point['updates']}: {point['mean']:.1%} ({point['weakest']} "
            f"{point['shares'][point['weakest']]:.0%})"
            for point in points
        )
        lines.append(f"  The {arm} arm along its updates, the mean (the weakest): {told_}")
    for point, weights in result["beside"]["weights"].items():
        heaviest = sorted(weights.items(), key=lambda item: -item[1])[:3]
        lines.append(
            f"  Weights before update {point}: "
            + ", ".join(f"{pair} {weight:.1%}" for pair, weight in heaviest)
        )
    for deck, points in result["beside"]["the copy along its updates"].items():
        told_ = ", ".join(
            f"{point['updates']}: {point['share']:.1%} ({100 * point['over its teacher']['gap']:+.1f})"
            for point in points
        )
        lines.append(f"  The copy along its updates, the {deck} mirror, over its teacher: {told_}")
    for who, found in result["beside"]["head to head"].items():
        told_ = ", ".join(f"{deck} {share:.1%}" for deck, share in found.items())
        lines.append(f"  Head to head against the champions, {who}: {told_}")
    spent_ = "; ".join(f"{name} {number(value)}" for name, value in result["hours"].items())
    lines.append(f"  Hours: {spent_}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--decks", required=True)
    parser.add_argument("--distilled", type=Path, required=True)
    parser.add_argument("--far", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    decks = [deck.strip() for deck in args.decks.split(",") if deck.strip()]
    result = report(args.directory, decks, args.distilled, args.far)
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf8")
    print(shown(result))


if __name__ == "__main__":
    main()
