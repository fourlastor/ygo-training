"""The checks of the seventy-one decks' PPO as it trains
(experiments/2026-10-28-seventy-one-champions.md).

    uv run python tools/seventy_one_checks.py suites --decks blackwing,cyber,... \\
        --suites eval_suites --games 128 --into analysis/seventy-one-champions/checks/suites
    uv run python tools/seventy_one_checks.py read analysis/seventy-one-champions/checks \\
        --decks blackwing,cyber,...
    uv run python tools/seventy_one_checks.py pairings analysis/seventy-one-champions/checks \\
        --met blackwing,lightsworn,...
    uv run python tools/seventy_one_checks.py along analysis/seventy-one-champions \\
        --decks blackwing,cyber,... --fourteen analysis/fourteen-decks \\
        --fourteen-decks blackwing,cyber,...
    uv run python tools/seventy_one_checks.py mean analysis/seventy-one-champions/checks \\
        --decks blackwing,cyber,... --against 5760
    uv run python tools/seventy_one_checks.py means analysis/seventy-one-champions/checks \\
        --decks blackwing,cyber,... --windows 160-1360,1600-2800,3040-4240,4480-5680

`suites` writes each deck's mirror suite against a checkpoint with its games
set to so many: the deals of every check in the mirrors.

`read` reads what `PARTS=copy-check` and `PARTS=ppo` of
`experiments/commands/seventy-one-champions.sh` leave in the checks' folder:
for the copy, `copy/`, and for each check,
`POINT/` by the updates of PPO it was made after, a pair `DECK.plays.json`
and `DECK.held.json` for each deck, the network head to head against that
deck's teacher in its mirror, and `pairings.json`, it against `all71` on the
check's pairings.

The rule, fixed on 2026-10-28 before anything was built: PPO stops when the
mean over the mirrors head to head, or the mean over the pairings against
`all71`, is under the copy's by at least 1 point by two standard errors, the
deals paired, the same of the two at two checks in a row. `read` says so of
the last check and ends with 3.

`pairings` tells the games against `all71` of the copy and of each check by
how many of a pairing's two decks `--met` names, the decks whose pilots the
copy played: both, one or neither, and the mirrors apart. Each group holds both
seats of its pairings, so that what a deck is worth cancels: told by the
opponent's deck alone, the groups would differ by their decks and not only by
the network. Beside the rule, and no part of it.

`along` reads what `PARTS=along` leaves in `along/`, `RUN/POINT/`: how the
mirrors moved along PPO, on the checks' deals, and along the fourteen decks'
arms, on the deals they were read on, each less its copy's. Beside the rule,
and no part of it.

`mean` reads what `PARTS=mean` leaves in `mean/` of the checks' folder: the
mean of PPO's weights, measured as a check, less PPO's check `--against` on the
same deals. Fixed on 2026-10-31 before it was played: the mirrors are raised when
at least 1 point over it by two standard errors, and not raised when at most 1
point over it by two standard errors; where the pairings against `all71` are
under it by at least 1 point by two standard errors the mean has broken the
network, and the mirrors say nothing.

`means` reads what `PARTS=means` leaves in the checks' folder: the mean of
PPO's weights in each of its windows, `mean-WINDOW/`, against the copy, with the
noise of its updates out as a single check's cannot be. Fixed on 2026-10-31
before they were played: the mirrors are rising when those of the last window
are over the second's by two standard errors, the deals paired, and not rising
otherwise. The first is left out of that, the first updates' fall being in it.
Told beside: the last window against `mean/`, how long a mean has to be.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fourteen_decks_report import joined, outcomes_of
from specialist_distillation_report import shares
from two_decks_report import error, mean, paired

ERRORS = 2.0
# Under the copy by at least so much, by two standard errors.
MARGIN = 0.01
# The checks in a row that stop PPO.
IN_A_ROW = 2
MEASURES = ("mirrors", "pairings")
STOPPED = 3
# A check's pairings, by how many of their two decks were met.
MIRRORS, BOTH, ONE, NEITHER = "mirrors", "both", "one", "neither"
TOLD = {MIRRORS: "the mirrors", BOTH: "both met", ONE: "one met", NEITHER: "neither met"}
# The fourteen decks' arms, and the updates they ended at.
ARMS = ("selfplay", "quarter", "half")
ARM_UPDATES = 4320


def suites(decks: list[str], source: Path, games: int, into: Path) -> None:
    """Each deck's mirror suite against a checkpoint, its games so many."""
    into.mkdir(parents=True, exist_ok=True)
    for deck in decks:
        suite = json.loads((source / f"{deck}-against-checkpoint.json").read_text(encoding="utf8"))
        suite["name"] = f"{deck}-check"
        for case in suite["cases"]:
            case["games"] = games
        (into / f"{deck}.json").write_text(json.dumps(suite, indent=2) + "\n", encoding="utf8")


def won(path: Path) -> list[float]:
    """Each game against `all71` won or not, in the order of its cases' names."""
    played = outcomes_of(path)
    return [float(game == "win") for name in sorted(played) for game in played[name]]


def mirrors_against(folder: Path, copy: Path, decks: list[str]) -> dict[str, Any]:
    """Each mirror head to head against the copy's on the same deals."""
    each = {deck: paired(shares(folder / deck), shares(copy / deck)) for deck in decks}
    return {
        "share": mean([mean(shares(folder / deck)) for deck in decks]),
        "gap": mean([one["gap"] for one in each.values()]),
        "error": joined([one["error"] for one in each.values()]),
        "each": each,
    }


def under(found: dict[str, Any]) -> bool:
    """Under the copy's by at least 1 point by two standard errors."""
    return found["gap"] + ERRORS * found["error"] <= -MARGIN


def check(folder: Path, copy: Path, decks: list[str]) -> dict[str, Any]:
    """A check against the copy's on the same deals: each mirror head to
    head, and the pairings against `all71`."""
    mirrors = mirrors_against(folder, copy, decks)
    ours = won(folder / "pairings.json")
    pairings = {"share": mean(ours), **paired(ours, won(copy / "pairings.json"))}
    for found in (mirrors, pairings):
        found["under"] = under(found)
    return {"mirrors": mirrors, "pairings": pairings}


def checks_of(directory: Path) -> list[int]:
    """The checks of a folder, by the updates of PPO each was made after."""
    return sorted(
        int(path.name) for path in directory.iterdir() if path.is_dir() and path.name.isdigit()
    )


def read(directory: Path, decks: list[str]) -> dict[str, Any]:
    copy = directory / "copy"
    points = checks_of(directory)
    checks = {point: check(directory / str(point), copy, decks) for point in points}
    # Where the rule was met, check by check, and whether the last check meets it.
    met = []
    for place, point in enumerate(points):
        for measure in MEASURES:
            row = points[max(0, place - IN_A_ROW + 1) : place + 1]
            if len(row) == IN_A_ROW and all(checks[one][measure]["under"] for one in row):
                met.append({"after": point, "by": measure})
    return {
        "copy": {
            "mirrors": mean([mean(shares(copy / deck)) for deck in decks]),
            "pairings": mean(won(copy / "pairings.json")),
        },
        "checks": checks,
        "met": met,
        "stop": bool(points) and any(one["after"] == points[-1] for one in met),
    }


def shown(result: dict[str, Any]) -> str:
    copy = result["copy"]
    lines = [
        "PPO's checks against the copy's, on the same deals:",
        (
            f"  The copy: the mirrors head to head {copy['mirrors']:.1%}, against all71 on the "
            f"check's pairings {copy['pairings']:.1%}"
        ),
    ]
    for point, found in result["checks"].items():
        told = []
        for measure in MEASURES:
            one = found[measure]
            told.append(
                f"the {measure} {one['share']:.1%} ({100 * one['gap']:+.1f} ± "
                f"{100 * one['error']:.1f} points{', under' if one['under'] else ''})"
            )
        lines.append(f"  After {point:,} updates: {', '.join(told)}")
    for one in result["met"]:
        lines.append(
            f"  The rule met after {one['after']:,} updates, by the {one['by']}: under the copy's "
            f"by at least 1 point by two standard errors at {IN_A_ROW} checks in a row"
        )
    lines.append("STOP." if result["stop"] else "Go on.")
    return "\n".join(lines)


def met_of(learner: str, opponent: str, met: set[str]) -> str:
    """How many of a pairing's two decks were met, the same from either seat."""
    if learner == opponent:
        return MIRRORS
    return (NEITHER, ONE, BOTH)[(learner in met) + (opponent in met)]


def against(path: Path, suite: dict[str, Any], met: set[str]) -> dict[str, dict[str, float]]:
    """A network's games against `all71` by how many of each pairing's two
    decks were met. A case's games are so many against each deck it names, in
    the order it names them."""
    played = outcomes_of(path)
    games: dict[str, list[float]] = {MIRRORS: [], BOTH: [], ONE: [], NEITHER: []}
    for case in suite["cases"]:
        count = int(case["games"])
        ended = played[case["name"]]
        names = [name.strip() for name in case["opponent"].split(",")]
        if len(ended) != count * len(names):
            raise SystemExit(f"{case['name']}: {len(ended)} games of {count * len(names)}")
        for place, deck in enumerate(names):
            group = met_of(case["learner"], deck, met)
            games[group] += [float(game == "win") for game in ended[place * count :][:count]]
    return {
        group: {"share": mean(won), "error": error(won), "games": len(won)}
        for group, won in games.items()
        if won
    }


def pairings(directory: Path, suite: dict[str, Any], met: set[str]) -> dict[str, Any]:
    """The copy's and each check's games against `all71` by how many of each
    pairing's two decks were met."""
    names = ["copy", *map(str, checks_of(directory))]
    return {name: against(directory / name / "pairings.json", suite, met) for name in names}


def shown_pairings(result: dict[str, Any]) -> str:
    lines = ["Against all71 on the check's pairings, both seats of each, by the decks met:"]
    for name, groups in result.items():
        told = ", ".join(
            f"{TOLD[group]} {one['share']:.1%} ± {one['error']:.1%} ({one['games']:,} games)"
            for group, one in groups.items()
        )
        lines.append(
            f"  {'The copy' if name == 'copy' else f'After {int(name):,} updates'}: {told}"
        )
    return "\n".join(lines)


def along(folder: Path, copy: Path, decks: list[str]) -> dict[int, dict[str, Any]]:
    """The mirrors after each point of a run in a folder, against its copy's
    on the same deals."""
    return {point: mirrors_against(folder / str(point), copy, decks) for point in checks_of(folder)}


def runs_along(
    directory: Path, fourteen: Path, decks: list[str], fourteen_decks: list[str]
) -> dict[str, dict[int, dict[str, Any]]]:
    """The mirrors along PPO, its checks among its points, over all its decks
    and over the fourteen; and along each of the fourteen decks' arms, where it
    ended among its points."""
    copy = directory / "checks" / "copy"
    runs = {}
    for name, told in (("PPO", decks), ("PPO, the fourteen decks", fourteen_decks)):
        points = {
            **along(directory / "along" / "ppo", copy, told),
            **along(directory / "checks", copy, told),
        }
        runs[name] = dict(sorted(points.items()))
    theirs = fourteen / "heads-taught"
    for arm in ARMS:
        runs[arm] = {
            **along(directory / "along" / arm, theirs, fourteen_decks),
            ARM_UPDATES: mirrors_against(fourteen / f"heads-{arm}", theirs, fourteen_decks),
        }
    return runs


def shown_along(runs: dict[str, dict[int, dict[str, Any]]]) -> str:
    lines = ["The mirrors head to head along each run, less its copy's on the same deals:"]
    for name, points in runs.items():
        lines.append(f"  {name}:")
        for point, one in points.items():
            lines.append(
                f"    After {point:,} updates: {one['share']:.1%} ({100 * one['gap']:+.1f} ± "
                f"{100 * one['error']:.1f} points{', under' if under(one) else ''})"
            )
    return "\n".join(lines)


def mean_of_weights(directory: Path, against: int, decks: list[str]) -> dict[str, Any]:
    """The mean of PPO's weights against its check `against`, by the rule of
    the mean."""
    found = check(directory / "mean", directory / str(against), decks)
    mirrors = found["mirrors"]
    if mirrors["gap"] - ERRORS * mirrors["error"] >= MARGIN:
        found["told"] = "raised"
    elif mirrors["gap"] + ERRORS * mirrors["error"] <= MARGIN:
        found["told"] = "not raised"
    else:
        found["told"] = "not told"
    return found


def shown_mean(found: dict[str, Any], against: int) -> str:
    lines = [f"The mean of PPO's weights against it after {against:,} updates, on the same deals:"]
    for measure in MEASURES:
        one = found[measure]
        lines.append(
            f"  The {measure} {one['share']:.1%} ({100 * one['gap']:+.1f} ± "
            f"{100 * one['error']:.1f} points)"
        )
    if found["pairings"]["under"]:
        lines.append("The pairings are under it: the mean has broken the network.")
    else:
        lines.append(f"The mirrors: {found['told']}.")
    return "\n".join(lines)


def means(directory: Path, windows: list[str], decks: list[str]) -> dict[str, Any]:
    """The mean of PPO's weights in each window against the copy, the last
    against the second by the rule of the means, and the last against `mean/`."""
    last, second = directory / f"mean-{windows[-1]}", directory / f"mean-{windows[1]}"
    rise = check(last, second, decks)
    return {
        "along": {
            window: check(directory / f"mean-{window}", directory / "copy", decks)
            for window in windows
        },
        "rise": rise,
        "rising": rise["mirrors"]["gap"] - ERRORS * rise["mirrors"]["error"] > 0,
        "against_the_mean": check(last, directory / "mean", decks),
    }


def told_of(found: dict[str, Any]) -> str:
    """A check's two measures, each with its gap."""
    return ", ".join(
        f"the {measure} {found[measure]['share']:.1%} ({100 * found[measure]['gap']:+.1f} ± "
        f"{100 * found[measure]['error']:.1f} points)"
        for measure in MEASURES
    )


def shown_means(result: dict[str, Any]) -> str:
    windows = list(result["along"])
    lines = ["The means of PPO's weights along it, against the copy's on the same deals:"]
    for window, found in result["along"].items():
        lines.append(f"  Updates {window}: {told_of(found)}")
    lines.append(f"Its last window against its second: {told_of(result['rise'])}")
    lines.append(f"Its last window against the longer mean: {told_of(result['against_the_mean'])}")
    lines.append(
        f"The mirrors of {windows[-1]} over those of {windows[1]}: "
        f"{'rising' if result['rising'] else 'not rising'}."
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    made = commands.add_parser("suites")
    made.add_argument("--decks", required=True)
    made.add_argument("--suites", type=Path, default=Path("eval_suites"))
    made.add_argument("--games", type=int, required=True)
    made.add_argument("--into", type=Path, required=True)
    reading = commands.add_parser("read")
    reading.add_argument("directory", type=Path)
    reading.add_argument("--decks", required=True)
    reading.add_argument("--output", type=Path)
    told = commands.add_parser("pairings")
    told.add_argument("directory", type=Path)
    told.add_argument("--met", required=True)
    told.add_argument(
        "--suite",
        type=Path,
        default=Path("eval_suites/seventy-one-decks-check-pairings-checkpoint.json"),
    )
    walked = commands.add_parser("along")
    walked.add_argument("directory", type=Path)
    walked.add_argument("--decks", required=True)
    walked.add_argument("--fourteen", type=Path, required=True)
    walked.add_argument("--fourteen-decks", required=True)
    averaged = commands.add_parser("mean")
    averaged.add_argument("directory", type=Path)
    averaged.add_argument("--decks", required=True)
    averaged.add_argument("--against", type=int, required=True)
    alongside = commands.add_parser("means")
    alongside.add_argument("directory", type=Path)
    alongside.add_argument("--decks", required=True)
    alongside.add_argument("--windows", required=True)
    args = parser.parse_args()
    if args.command == "pairings":
        suite = json.loads(args.suite.read_text(encoding="utf8"))
        met = {deck.strip() for deck in args.met.split(",") if deck.strip()}
        print(shown_pairings(pairings(args.directory, suite, met)))
        return
    decks = [deck.strip() for deck in args.decks.split(",") if deck.strip()]
    if args.command == "suites":
        suites(decks, args.suites, args.games, args.into)
        return
    if args.command == "along":
        fourteen_decks = [deck.strip() for deck in args.fourteen_decks.split(",") if deck.strip()]
        print(shown_along(runs_along(args.directory, args.fourteen, decks, fourteen_decks)))
        return
    if args.command == "mean":
        print(shown_mean(mean_of_weights(args.directory, args.against, decks), args.against))
        return
    if args.command == "means":
        print(shown_means(means(args.directory, args.windows.split(","), decks)))
        return
    result = read(args.directory, decks)
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(shown(result))
    if result["stop"]:
        sys.exit(STOPPED)


if __name__ == "__main__":
    main()
