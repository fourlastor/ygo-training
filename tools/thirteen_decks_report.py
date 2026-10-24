"""The readings of thirteen decks on the new input
(experiments/2026-10-16-thirteen-decks.md).

    uv run python tools/thirteen_decks_report.py analysis/thirteen-decks \\
        --before analysis/more-decks --output analysis/thirteen-decks/report.json

The directory is what `PARTS=measure` of `experiments/commands/thirteen-decks.sh`
leaves: the copy at its last checkpoint and each arm (half, half-upgo,
itself) at its checkpoints after its last three `--step` of `--updates`
updates of PPO, half also after the three up to `--midway`, on the screen
of each deck (`screens/RUN.DECK.json`); the same but half's at midway on
the held-out deals (`heldout/RUN.json`) and on the decks outside training
(`unseen/RUN.json`); and itself against half, their checkpoints matched,
on every pairing of the decks, each in turn the one held
(`head-to-head/itself-N.json` and `itself-N.back.json`).  `--before` is
where the experiment on more decks left decks-13 and its copy on the same
screens (`screens/decks-13.DECK.json`) and the decks' own pilots
(`screens/pilot.DECK.json`), and the same beyond the decks of training
(`beyond/unseen/`).

A game of a screen is read by the share of a run's three checkpoints that
win it, and a won game is a win (a drawn one is not); a game of the
pairings is the measured one's, a draw half a win.

The readings, fixed on 2026-10-16 before any run, each by a run's
checkpoints after 3,600, 3,720 and 3,840 updates unless said, over the
thirteen screens together, game by game:

1. Nothing lost without the lists, under PPO: half after 1,680 to 1,920
   updates against decks-13 after as many: equivalent when the gap less two
   standard errors is above -2 points, worse when it wins at least 2 points
   and two standard errors fewer, otherwise not decided.  The copy against
   decks-13's beside it, without a mark.
2. More updates add: half against itself after 1,680 to 1,920 updates:
   gained at 2 points and two standard errors more.
3. UPGO: half-upgo against half: better at 2 points and two standard
   errors more, worse at as many fewer, otherwise not told apart.
4. Against itself alone, what the pilots say: itself against half, the
   same marks.
5. Against itself alone, head to head: itself against half on every
   pairing of the decks, each deal from either seat, the three pairs of
   checkpoints together: better when it wins at least 52% and two standard
   errors over half, worse at 48% and two standard errors under, otherwise
   not told apart.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from more_decks_report import outcomes
from ppo_head_report import difference
from search_leads_report import percent, points
from self_play_report import score, shares, spread, within

DECKS = (
    "blackwing", "lightsworn", "gishki", "burn", "crystal", "dragunity", "gladiator",
    "heroes", "infernity", "monarch", "morphtronic", "rock-block", "spellcaster",
)  # fmt: skip
ARMS = ("half", "half-upgo", "itself")
COPY = "copy"
PILOT = "pilot"
BEFORE = "decks-13"
# The marks: so far apart, from half, and in standard errors.
MARK = 0.02
FROM_HALF = 0.02
ERRORS = 2.0
# A run's checkpoints are read three at a time.
END = 3


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def points_of(updates: int, step: int) -> list[int]:
    """The updates of PPO of the three checkpoints read up to `updates`."""
    return [updates - (END - 1 - index) * step for index in range(END)]


def kept(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf8")) if path.is_file() else None


def by_step(path: Path, found: dict[str, Any]) -> dict[int, list[float]]:
    """A suite's games of each checkpoint, by its updates in all: whether
    each game was won, the cases in the order of their names."""
    games = {}
    for record in found["records"]:
        failed = [name for name, case in record["evaluation_suite"].items() if "error" in case]
        if failed:
            raise SystemExit(f"{path}: {', '.join(failed)} failed")
        if any("outcomes" not in case for case in record["evaluation_suite"].values()):
            raise SystemExit(f"{path}: a measurement without how its games ended")
        games[int(record.get("checkpoint_step") or 0)] = outcomes(record["evaluation_suite"])
    return games


def dealt(found: dict[str, Any]) -> Any:
    """What tells the games of a suite: its seed, its cases and their games."""
    cases = found["records"][0]["evaluation_suite"]
    return found.get("seed"), tuple((case, cases[case].get("games")) for case in sorted(cases))


class Games:
    """The games of one kind (a deck's screen, the decks outside training),
    each player's by what it is called, all of them the same games."""

    def __init__(self, what: str) -> None:
        self.what = what
        self.games: dict[str, list[float]] = {}
        self.deals: Any = None

    def add(self, path: Path, wanted: dict[str, int] | None) -> None:
        """The players of a file: the checkpoints of `wanted` by what they
        are called, or the pilot where none is."""
        found = kept(path)
        if found is None:
            return
        told = dealt(found)
        if self.deals is not None and told != self.deals:
            raise SystemExit(f"{path}: other games than the others of {self.what}")
        self.deals = told
        if wanted is None:
            self.games[PILOT] = outcomes(found["records"][0]["evaluation_suite"])
            return
        steps = by_step(path, found)
        for name, step in wanted.items():
            if step not in steps:
                raise SystemExit(f"{path}: no checkpoint after {step} updates in all")
            self.games[name] = steps[step]

    def three(self, prefix: str, points: list[int]) -> list[float] | None:
        """For each game, the share of the three checkpoints that win it."""
        names = [f"{prefix}@{point}" for point in points]
        if not all(name in self.games for name in names):
            return None
        return [sum(game) / END for game in zip(*(self.games[n] for n in names), strict=True)]


def verdict(gap: float, error: float, mark: float, over: str, under: str | None) -> str:
    if gap >= mark and gap >= ERRORS * error:
        return over
    if under is not None and gap <= -mark and -gap >= ERRORS * error:
        return under
    return "not told apart" if under is not None else "not shown"


def lost(gap: float, error: float) -> str:
    """Reading 1: equivalent, worse, or neither."""
    if gap <= -MARK and -gap >= ERRORS * error:
        return "worse"
    if gap - ERRORS * error > -MARK:
        return "equivalent"
    return "not decided"


def paired(directory: Path, point: int) -> list[list[float]] | None:
    """Itself against half on every pairing after `point` updates, each in
    turn the one held: for itself, its game of every deal from either seat,
    the deals in the order of the cases' names."""
    paths = [
        directory / "head-to-head" / f"itself-{point}{suffix}.json" for suffix in ("", ".back")
    ]
    forward, back = (kept(path) for path in paths)
    if forward is None or back is None:
        return None
    one, other = forward["records"], back["records"]
    if (
        len(one) != 1
        or len(other) != 1
        or forward["opponent_checkpoint"]["path"] != other[0].get("checkpoint")
        or back["opponent_checkpoint"]["path"] != one[0].get("checkpoint")
        or dealt(forward) != dealt(back)
    ):
        raise SystemExit(f"{paths[1]} is not {paths[0]} with the other checkpoint held")

    def told(path: Path, record: dict[str, Any]) -> list[float]:
        cases = record["evaluation_suite"]
        failed = [name for name, case in cases.items() if "error" in case]
        if failed:
            raise SystemExit(f"{path}: {', '.join(failed)} failed")
        return [score(outcome) for case in sorted(cases) for outcome in cases[case]["outcomes"]]

    return [told(paths[0], one[0]), [1.0 - game for game in told(paths[1], other[0])]]


def figures(path: Path, step: int) -> dict[str, float] | None:
    """A run's own figures over its last `step` updates: how long an update
    took, duels an hour, what it won against the pilots, and how often the
    seat that opens won against itself."""
    if not path.is_file():
        return None
    rows = [json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line]
    rows = rows[-step:]
    if not rows:
        return None
    seconds = mean(
        [float(row["wall_seconds"]) - float(row.get("evaluation_seconds", 0.0)) for row in rows]
    )
    found = {
        "update_seconds": seconds,
        "duels_an_hour": 3600 * mean([float(row["duels"]) for row in rows]) / seconds,
    }
    against = [
        row["other_seat_win_rate"]["pilot"]
        for row in rows
        if "pilot" in (row.get("other_seat_win_rate") or {})
    ]
    if against:
        found["against_pilots"] = mean([float(rate) for rate in against])
    first = [
        float(row["self_play_first_seat_win_rate"])
        for row in rows
        if "self_play_first_seat_win_rate" in row
    ]
    if first:
        found["first_seat_against_itself"] = mean(first)
    return found


def report(
    directory: Path,
    before: Path,
    decks: list[str],
    copy: int = 240,
    updates: int = 3840,
    midway: int = 1920,
    step: int = 120,
) -> dict[str, Any]:
    copied = len(decks) * copy
    end, middle = points_of(updates, step), points_of(midway, step)
    result: dict[str, Any] = {"decks": decks, "end": end, "midway": middle, "readings": {}}

    def steps(prefix: str, along: list[int]) -> dict[str, int]:
        return {f"{prefix}@{point}": copied + point for point in along}

    wanted = {
        COPY: {COPY: copied},
        "half": steps("half", middle) | steps("half", end),
        **{arm: steps(arm, end) for arm in ARMS[1:]},
    }
    # The screens: the runs, decks-13 and its copy, and the pilots.
    screens: dict[str, Games] = {}
    for deck in decks:
        found = Games(f"the screen of {deck}")
        for run, asked in wanted.items():
            found.add(directory / "screens" / f"{run}.{deck}.json", asked)
        found.add(
            before / "screens" / f"{BEFORE}.{deck}.json",
            {f"{BEFORE}-copy": copied} | steps(BEFORE, middle),
        )
        found.add(before / "screens" / f"{PILOT}.{deck}.json", None)
        screens[deck] = found

    def together(read: Any) -> list[float] | None:
        """The games of every screen, each read by `read`, or None where a
        screen lacks them."""
        games: list[float] = []
        for found in screens.values():
            told = read(found)
            if told is None:
                return None
            games.extend(told)
        return games

    def three(prefix: str, along: list[int]) -> list[float] | None:
        return together(lambda found: found.three(prefix, along))

    def alone(name: str) -> list[float] | None:
        return together(lambda found: found.games.get(name))

    read = {
        COPY: alone(COPY),
        f"{BEFORE}-copy": alone(f"{BEFORE}-copy"),
        f"{BEFORE}@midway": three(BEFORE, middle),
        "half@midway": three("half", middle),
        **{arm: three(arm, end) for arm in ARMS},
        PILOT: alone(PILOT),
    }
    result["screens"] = {name: mean(games) for name, games in read.items() if games}
    result["by_deck"] = {}
    for deck, found in screens.items():
        row = {}
        for name, games in {
            COPY: found.games.get(COPY),
            f"{BEFORE}-copy": found.games.get(f"{BEFORE}-copy"),
            f"{BEFORE}@midway": found.three(BEFORE, middle),
            "half@midway": found.three("half", middle),
            **{arm: found.three(arm, end) for arm in ARMS},
            PILOT: found.games.get(PILOT),
        }.items():
            if games:
                row[name] = mean(games)
        result["by_deck"][deck] = row

    readings = result["readings"]
    if read["half@midway"] is not None and read[f"{BEFORE}@midway"] is not None:
        gap, error = difference(read["half@midway"], read[f"{BEFORE}@midway"])
        readings["lists"] = {
            "gap": gap, "error": error, "games": len(read["half@midway"]), "verdict": lost(gap, error)
        }  # fmt: skip
        if read[COPY] is not None and read[f"{BEFORE}-copy"] is not None:
            gap, error = difference(read[COPY], read[f"{BEFORE}-copy"])
            readings["lists"]["copies"] = {"gap": gap, "error": error}
    if read["half"] is not None and read["half@midway"] is not None:
        gap, error = difference(read["half"], read["half@midway"])
        readings["longer"] = {
            "gap": gap,
            "error": error,
            "verdict": verdict(gap, error, MARK, "gained", None),
        }
    for arm, name in (("half-upgo", "upgo"), ("itself", "itself")):
        if read[arm] is not None and read["half"] is not None:
            gap, error = difference(read[arm], read["half"])
            readings[name] = {
                "gap": gap,
                "error": error,
                "verdict": verdict(gap, error, MARK, "better", "worse"),
            }
    games = []
    for point in end:
        found = paired(directory, point)
        if found is None:
            games = []
            break
        games.extend(found)
        result.setdefault("head_to_head", {})[str(point)] = mean(shares(found))
    if games:
        deals = shares(games)
        share, error = mean(deals), spread(deals)
        readings["head to head"] = {
            "share": share,
            "error": error,
            "games": len(games) * len(deals),
            "verdict": verdict(share - 0.5, error, FROM_HALF, "better", "worse"),
        }

    # Beside the readings: the held-out deals, the decks outside training,
    # and the runs' own figures.
    beside_points = {COPY: [None], **{arm: end for arm in ARMS}}
    for run, along in beside_points.items():
        found = kept(directory / "heldout" / f"{run}.json")
        if found is None:
            continue
        rates = {
            int(point["checkpoint_step"]): mean(
                [float(o == "win") for o in point["deals"]["outcomes"]]
            )
            for point in found["checkpoints"]
        }
        told = [rates.get(copied if point is None else copied + point) for point in along]
        if all(rate is not None for rate in told):
            result.setdefault("heldout", {})[run] = mean(told)
    unseen = Games("the decks outside training")
    for run in beside_points:
        unseen.add(
            directory / "unseen" / f"{run}.json",
            {COPY: copied} if run == COPY else steps(run, end),
        )
    unseen.add(
        before / "beyond" / "unseen" / f"{BEFORE}.json",
        {f"{BEFORE}-copy": copied} | steps(BEFORE, points_of(midway, step)),
    )
    unseen.add(before / "beyond" / "unseen" / f"{PILOT}.json", None)
    told = {
        COPY: unseen.games.get(COPY),
        f"{BEFORE}-copy": unseen.games.get(f"{BEFORE}-copy"),
        f"{BEFORE}@midway": unseen.three(BEFORE, middle),
        **{arm: unseen.three(arm, end) for arm in ARMS},
        PILOT: unseen.games.get(PILOT),
    }
    result["unseen"] = {name: mean(games) for name, games in told.items() if games}
    for run in (COPY, *ARMS):
        found = figures(directory / f"{run}.metrics.jsonl", step)
        if found is not None:
            result.setdefault("figures", {})[run] = found
    return result


# The players as a table of the decks heads them.
COLUMNS = {
    COPY: "copy",
    f"{BEFORE}-copy": "13's copy",
    f"{BEFORE}@midway": "decks-13",
    "half@midway": "half mid",
    "half": "half",
    "half-upgo": "half-upgo",
    "itself": "itself",
    PILOT: "pilots",
}
NAMES = {
    COPY: "the copy",
    f"{BEFORE}-copy": "decks-13's copy",
    f"{BEFORE}@midway": "decks-13",
    "half@midway": "half, midway",
    "half": "half",
    "half-upgo": "half-upgo",
    "itself": "itself",
    PILOT: "the pilots",
}


def show(result: dict[str, Any]) -> str:
    end = ", ".join(map(str, result["end"]))
    middle = ", ".join(map(str, result["midway"]))
    lines = [
        f"The screens of the {len(result['decks'])} decks together: each arm by its checkpoints after",
        f"{end} updates of PPO, half and decks-13 also after {middle}, the copies at their last",
        "checkpoint, the decks' own pilots.",
    ]
    for name, rate in result["screens"].items():
        lines.append(f"  {NAMES[name]:<16}{percent(rate):>8}")
    columns = [name for name in NAMES if any(name in row for row in result["by_deck"].values())]
    lines.append("Deck by deck.")
    lines.append("  " + f"{'':<13}" + "".join(f"{COLUMNS[name]:>10}" for name in columns))
    for deck, row in result["by_deck"].items():
        lines.append(
            f"  {deck:<13}"
            + "".join(f"{percent(row.get(name)) if name in row else '':>10}" for name in columns)
        )
    if "heldout" in result:
        lines.append("The held-out deals of Blackwing, by the same checkpoints but half's midway:")
        for run, rate in result["heldout"].items():
            lines.append(f"  {NAMES[run]:<16}{percent(rate):>8}")
    if result.get("unseen"):
        lines.append("The decks outside training, played against the pilots of training:")
        for name, rate in result["unseen"].items():
            lines.append(f"  {NAMES[name]:<16}{percent(rate):>8}")
    if "head_to_head" in result:
        lines.append(
            "Itself against half on every pairing, each deal from either seat, by the checkpoints:"
        )
        for point, share in result["head_to_head"].items():
            lines.append(f"  after {point}: {percent(share)}")
    if "figures" in result:
        lines.append(
            "The runs' own figures over their last updates: an update, duels an hour, against"
        )
        lines.append("the pilots, and the opening seat against itself.")
        for run, found in result["figures"].items():
            lines.append(
                f"  {NAMES[run]:<16}{found['update_seconds']:>6.1f} s {found['duels_an_hour']:>10,.0f}"
                f"{percent(found.get('against_pilots')):>9}{percent(found.get('first_seat_against_itself')):>9}"
            )
    readings = result["readings"]
    lines.append("The readings:")
    if "lists" in readings:
        found = readings["lists"]
        line = (
            f"  1. half against decks-13 after {middle}: {points(found['gap'], found['error'])} "
            f"over {found['games']:,} games: {found['verdict']}"
        )
        if "copies" in found:
            line += f" (the copies: {points(found['copies']['gap'], found['copies']['error'])})"
        lines.append(line)
    else:
        lines.append("  1. half against decks-13: not measured")
    labels = (
        ("longer", "2. half against itself midway"),
        ("upgo", "3. half-upgo against half"),
        ("itself", "4. itself against half"),
    )
    for key, label in labels:
        if key in readings:
            found = readings[key]
            lines.append(f"  {label}: {points(found['gap'], found['error'])}: {found['verdict']}")
        else:
            lines.append(f"  {label}: not measured")
    if "head to head" in readings:
        found = readings["head to head"]
        lines.append(
            f"  5. itself against half head to head: {within(found['share'], found['error'])} "
            f"of {found['games']:,} games: {found['verdict']}"
        )
    else:
        lines.append("  5. itself against half head to head: not measured")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("directory", type=Path)
    parser.add_argument("--before", type=Path, default=Path("analysis/more-decks"))
    parser.add_argument("--decks", nargs="+", default=list(DECKS))
    parser.add_argument("--copy", type=int, default=240, help="a copy's updates for each deck")
    parser.add_argument("--updates", type=int, default=3840, help="an arm's updates of PPO")
    parser.add_argument("--midway", type=int, default=1920, help="decks-13's updates of PPO")
    parser.add_argument("--step", type=int, default=120, help="the updates between checkpoints")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = report(
        args.directory, args.before, args.decks, args.copy, args.updates, args.midway, args.step
    )
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(show(result))


if __name__ == "__main__":
    main()
