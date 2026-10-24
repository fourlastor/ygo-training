"""The networks of the experiment on more decks beyond their own decks
(experiments/2026-10-14-more-decks.md, "Beyond their decks").

    uv run python tools/more_decks_beyond_report.py analysis/more-decks/beyond \\
        --sizes 1 2 3 5 8 13 --output analysis/more-decks/beyond/report.json

The directory is what `PARTS=beyond` of `experiments/commands/more-decks.sh`
leaves.  For each network, of the first SIZE decks (decks-NN): its copy at
its last checkpoint and its checkpoints after its last three `--step` of
`--updates` PPO updates playing the eighteen decks outside training
themselves against the pilots of the thirteen of training
(`unseen/decks-NN.json`), and those decks' own pilots on the same games
(`unseen/pilot.json`); and the network of the most decks against that of
the fewest in the mirror of the first deck, their checkpoints after as many
updates matched, each deal from either seat (`head-to-head/mirror-N.json`,
its own checkpoint held, and `mirror-N.back.json`, the other's).  A game of
the decks outside training is read by the share of a network's last three
checkpoints that win it, and a won game is a win (a drawn one is not); a
game of the mirror is the measured one's, a draw half a win.

The readings, fixed on 2026-10-16, each by a network's last three
checkpoints:

1. The network of the most decks against the decks' own pilots, on the
   decks outside training, game by game: over at 2 points and two
   standard errors more, under at 2 points and two standard errors fewer.
2. More decks help with a deck never trained on: the network of the most
   decks against that of the fewest, the same games, at 2 points and two
   standard errors more.
3. PPO carries over to a deck never trained on: the network of the most
   decks against its own copy, the same games, at 2 points and two
   standard errors more.
4. The network of the most decks against that of the fewest in the mirror,
   every deal from either seat, the three pairs together: worse when it
   wins at most 48% and two standard errors under half, better at 52% and
   two standard errors over.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from more_decks_report import COPY, END, PILOT, last_three, mean, network, outcomes, steps
from ppo_head_report import difference
from search_leads_report import percent, points
from self_play_report import pair, shares, spread, within

# The marks of the readings: so far over or under, from half, and in
# standard errors.
OVER = 0.02
HELP = 0.02
CARRIES = 0.02
FROM_HALF = 0.02
ERRORS = 2.0


def played(
    directory: Path, file: str, wanted: dict[str, int] | None
) -> tuple[dict[str, list[float]], Any]:
    """Checkpoints, or the pilots, on the games of the decks outside
    training, by what they are called (the pilots by PILOT), and what tells
    the games."""
    path = directory / "unseen" / f"{file}.json"
    if not path.is_file():
        return {}, None
    found = json.loads(path.read_text(encoding="utf8"))
    records = found["records"]
    if not records or any(
        "outcomes" not in case for record in records for case in record["evaluation_suite"].values()
    ):
        raise SystemExit(f"{path}: a measurement without how its games ended")
    for record in records:
        failed = [name for name, case in record["evaluation_suite"].items() if "error" in case]
        if failed:
            raise SystemExit(f"{path}: {', '.join(failed)} failed")
    cases = records[0]["evaluation_suite"]
    dealt = (found.get("seed"), tuple((case, cases[case].get("games")) for case in sorted(cases)))
    if wanted is None:
        return {PILOT: outcomes(cases)}, dealt
    by_step = {
        int(record["checkpoint_step"]): outcomes(record["evaluation_suite"]) for record in records
    }
    games = {}
    for name, step in wanted.items():
        if step not in by_step:
            raise SystemExit(f"{path}: no checkpoint after {step} updates in all")
        games[name] = by_step[step]
    return games, dealt


def by_deck(cases: list[str], games: list[float]) -> dict[str, list[float]]:
    """The games of each deck outside training: the case names are
    DECK-vs-OPPONENT, in the order `outcomes` reads them."""
    found: dict[str, list[float]] = {}
    for case, game in zip(cases, games, strict=True):
        found.setdefault(case.split("-vs-")[0], []).append(game)
    return found


def ordered_cases(directory: Path) -> list[str]:
    """Each game's case, in the order `outcomes` reads the games."""
    path = directory / "unseen" / f"{PILOT}.json"
    cases = json.loads(path.read_text(encoding="utf8"))["records"][0]["evaluation_suite"]
    return [case for case in sorted(cases) for _ in cases[case]["outcomes"]]


def verdict(gap: float, error: float, mark: float, over: str, under: str | None) -> str:
    if gap >= mark and gap >= ERRORS * error:
        return over
    if under is not None and gap <= -mark and -gap >= ERRORS * error:
        return under
    return "not told apart" if under is not None else "not shown"


def report(directory: Path, sizes: list[int], copy: int, updates: int, step: int) -> dict[str, Any]:
    result: dict[str, Any] = {"sizes": sizes, "read_by": [], "networks": {}, "readings": {}}
    pilots, dealt = played(directory, PILOT, None)
    if not pilots:
        raise SystemExit(f"{directory}/unseen/{PILOT}.json: the pilots have not played")
    networks: dict[int, dict[str, list[float]]] = {}
    for size in sizes:
        wanted = steps(size, copy, updates, step)
        games, told = played(directory, network(size), wanted)
        if not games:
            continue
        if told != dealt:
            raise SystemExit(f"{network(size)} played other games than the pilots")
        networks[size] = games
        result["networks"][network(size)] = {
            COPY: mean(games[COPY]),
            "three": mean(last_three(games, updates, step)),
            **{name: mean(games[name]) for name in games if name != COPY},
        }
    result["read_by"] = [updates - (END - 1 - index) * step for index in range(END)]
    result["pilots"] = mean(pilots[PILOT])
    most, fewest = max(networks, default=None), min(networks, default=None)
    if most is not None:
        three = last_three(networks[most], updates, step)
        gap, error = difference(three, pilots[PILOT])
        result["readings"]["pilots"] = {
            "gap": gap,
            "error": error,
            "verdict": verdict(gap, error, OVER, "over", "under"),
        }
        if fewest != most:
            gap, error = difference(three, last_three(networks[fewest], updates, step))
            result["readings"]["more decks"] = {
                "gap": gap,
                "error": error,
                "verdict": verdict(gap, error, HELP, "more decks help", None),
            }
        gap, error = difference(three, networks[most][COPY])
        result["readings"]["copy"] = {
            "gap": gap,
            "error": error,
            "verdict": verdict(gap, error, CARRIES, "PPO carries over", None),
        }
        cases = ordered_cases(directory)
        mine, theirs = by_deck(cases, three), by_deck(cases, pilots[PILOT])
        result["by_deck"] = {
            deck: {"pilot": mean(theirs[deck]), "three": mean(mine[deck])} for deck in mine
        }
    games = []
    for point in result["read_by"]:
        found = pair(directory, f"mirror-{point}")
        if found is None:
            games = []
            break
        games.extend(found["games"])
        result.setdefault("mirror", {})[str(point)] = mean(shares(found["games"]))
    if games:
        deals = shares(games)
        share, error = mean(deals), spread(deals)
        result["readings"]["mirror"] = {
            "share": share,
            "error": error,
            "games": len(games) * len(deals),
            "verdict": verdict(share - 0.5, error, FROM_HALF, "better", "worse"),
        }
    return result


def show(result: dict[str, Any]) -> str:
    sizes = result["sizes"]
    names = [network(size) for size in sizes if network(size) in result["networks"]]
    read_by = ", ".join(str(point) for point in result["read_by"])
    lines = [
        "The decks outside training, played by each network against the pilots of the decks of",
        f"training: its copy, its checkpoints after {read_by} updates of PPO, and the three",
        f"together. Those decks' own pilots win {percent(result['pilots'])} of the same games.",
        "  Network     Its copy    "
        + "  ".join(f"{p:>6}" for p in result["read_by"])
        + "  The three",
    ]
    for name in names:
        row = result["networks"][name]
        cells = "  ".join(f"{percent(row[str(p)]):>6}" for p in result["read_by"])
        lines.append(f"  {name:<10} {percent(row[COPY]):>9}    {cells}  {percent(row['three']):>9}")
    if "by_deck" in result:
        lines.append(
            "Deck by deck, the network of the most decks (the three) and the deck's own pilot."
        )
        for deck, row in sorted(result["by_deck"].items()):
            lines.append(
                f"  {deck:<16} {percent(row['three']):>7}  pilot {percent(row['pilot']):>7}"
            )
    if "mirror" in result:
        lines.append(
            "The network of the most decks against that of the fewest in the mirror, every deal "
            "from either seat, by its checkpoint:"
        )
        for point, share in result["mirror"].items():
            lines.append(f"  after {point}: {percent(share)}")
    readings = result["readings"]
    lines.append("The readings:")
    if "pilots" in readings:
        found = readings["pilots"]
        lines.append(
            f"  1. against the decks' own pilots: {points(found['gap'], found['error'])}: {found['verdict']}"
        )
    if "more decks" in readings:
        found = readings["more decks"]
        lines.append(
            f"  2. against the network of the fewest decks: {points(found['gap'], found['error'])}: "
            f"{found['verdict']}"
        )
    if "copy" in readings:
        found = readings["copy"]
        lines.append(
            f"  3. against its own copy: {points(found['gap'], found['error'])}: {found['verdict']}"
        )
    if "mirror" in readings:
        found = readings["mirror"]
        lines.append(
            f"  4. in the mirror: {within(found['share'], found['error'])} of {found['games']:,} games: "
            f"{found['verdict']}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("directory", type=Path)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 2, 3, 5, 8, 13])
    parser.add_argument("--copy", type=int, default=240)
    parser.add_argument("--updates", type=int, default=1920)
    parser.add_argument("--step", type=int, default=120)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = report(args.directory, args.sizes, args.copy, args.updates, args.step)
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(show(result))


if __name__ == "__main__":
    main()
