"""The readings of one network for more decks, half of its duels against
itself (experiments/2026-10-14-more-decks.md).

    uv run python tools/more_decks_report.py analysis/more-decks \\
        --sizes 1 2 3 5 8 13 --output analysis/more-decks/report.json

The directory is what `PARTS=measure` of `experiments/commands/more-decks.sh`
leaves.  For each network, of the first SIZE decks of `--decks` (decks-NN):
its copy at its last checkpoint and its checkpoints after its last three
`--step` of `--updates` PPO updates, on the held-out deals
(`heldout/decks-NN.json`) and on the screen of each of its decks
(`screens/decks-NN.DECK.json`), with how every game ended; and each deck's
pilot on its screen (`screens/pilot.DECK.json`).  A game is read by the
share of a network's last three checkpoints that win it, and a won game is
a win (a drawn one is not).

Beside the readings, without a mark: each network's other checkpoints of
its PPO on the held-out deals (`heldout/decks-NN.curve.json`), how fast it
learns, by its updates and by the seats of the first deck it had played by
then (its runs' figures, `decks-NN.metrics.jsonl`): a network of N decks
plays the first deck in about an Nth of its seats.

The readings, fixed on 2026-10-13, each by a network's last three
checkpoints:

1. What more decks cost Blackwing.  Each network of more decks against the
   network of the first deck alone, on the held-out deals, deal by deal: it
   is worse when it wins at least 2 points and two standard errors fewer.
   Each is read on its own.
2. The network of all the decks gains on its copy.  Its three checkpoints
   against its copy's last checkpoint, over the screens of all its decks
   together, game by game: it has gained when it wins at least 2 points and
   two standard errors more.
3. The network of all the decks is over the pilots.  Its three checkpoints
   against each deck's own pilot on that deck's screen, all the decks
   together, game by game: over at 2 points and two standard errors more,
   under at 2 points and two standard errors fewer.
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

# The marks of the readings: so far under or over, and in standard errors.
WORSE = 0.02
GAINED = 0.02
OVER = 0.02
ERRORS = 2.0
# The checkpoints a network is read by: its last ones.
END = 3
PILOT = "pilot"
COPY = "copy"


def network(size: int) -> str:
    return f"decks-{size:02d}"


def steps(size: int, copy: int, updates: int, step: int) -> dict[str, int]:
    """The updates in all of a network's checkpoints that are measured: its
    copy's last, and its last three of PPO, by what they are called."""
    copied = size * copy
    found = {COPY: copied}
    for point in range(updates - (END - 1) * step, updates + 1, step):
        found[str(point)] = copied + point
    return found


def outcomes(cases: dict[str, Any]) -> list[float]:
    """Whether each game was won, the cases in the order of their names."""
    return [float(outcome == "win") for case in sorted(cases) for outcome in cases[case]["outcomes"]]


def held(directory: Path, size: int, wanted: dict[str, int]) -> tuple[dict[str, list[float]], Any]:
    """A network's checkpoints on the held-out deals, by what they are
    called, and what tells the deals."""
    path = directory / "heldout" / f"{network(size)}.json"
    if not path.is_file():
        return {}, None
    found = json.loads(path.read_text(encoding="utf8"))
    by_step = {
        int(point["checkpoint_step"]): [float(o == "win") for o in point["deals"]["outcomes"]]
        for point in found["checkpoints"]
    }
    games = {}
    for name, step in wanted.items():
        if step not in by_step:
            raise SystemExit(f"{path}: no checkpoint after {step} updates in all")
        games[name] = by_step[step]
    return games, (found.get("fingerprint"), found.get("games"))


def screened(directory: Path, file: str, wanted: dict[str, int] | None) -> tuple[dict[str, list[float]], Any]:
    """Checkpoints, or the pilot, on a deck's screen, by what they are called
    (the pilot by PILOT), and what tells the games."""
    path = directory / "screens" / f"{file}.json"
    if not path.is_file():
        return {}, None
    found = json.loads(path.read_text(encoding="utf8"))
    records = found["records"]
    if not records or any(
        "outcomes" not in case for record in records for case in record["evaluation_suite"].values()
    ):
        raise SystemExit(f"{path}: a measurement without how its games ended")
    cases = records[0]["evaluation_suite"]
    dealt = (found.get("seed"), tuple((case, cases[case].get("games")) for case in sorted(cases)))
    if wanted is None:
        return {PILOT: outcomes(cases)}, dealt
    by_step = {int(record["checkpoint_step"]): outcomes(record["evaluation_suite"]) for record in records}
    games = {}
    for name, step in wanted.items():
        if step not in by_step:
            raise SystemExit(f"{path}: no checkpoint after {step} updates in all")
        games[name] = by_step[step]
    return games, dealt


def last_three(games: dict[str, list[float]], updates: int, step: int) -> list[float]:
    """For each game, the share of a network's last three checkpoints that
    win it."""
    names = [str(point) for point in range(updates - (END - 1) * step, updates + 1, step)]
    return [sum(game) / END for game in zip(*(games[name] for name in names), strict=True)]


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def curve(directory: Path, size: int, copy: int, step: int) -> dict[int, float]:
    """A network's other checkpoints of its PPO on the held-out deals, by
    its updates of PPO: what each wins of them."""
    path = directory / "heldout" / f"{network(size)}.curve.json"
    if not path.is_file():
        return {}
    found = json.loads(path.read_text(encoding="utf8"))
    copied = size * copy
    return {
        int(point["checkpoint_step"]) - copied: mean([float(o == "win") for o in point["deals"]["outcomes"]])
        for point in found["checkpoints"]
        if (int(point["checkpoint_step"]) - copied) % step == 0
    }


def seats(directory: Path, size: int, deck: str) -> dict[int, float]:
    """The seats of `deck` a network had played after each of its updates of
    PPO, by its runs' figures: against the pilots and against itself."""
    path = directory / f"{network(size)}.metrics.jsonl"
    if not path.is_file():
        return {}
    taken, played = {}, 0.0
    rows = [json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line.strip()]
    for update, row in enumerate(rows, start=1):
        if size == 1:
            # One deck: every duel against a pilot is one seat of it, and
            # every duel against itself two.
            told = row.get("other_seat_duels") or {}
            played += float(told.get("pilot", 0.0)) + 2.0 * float(told.get("self", 0.0))
        else:
            played += float((row.get("learner_deck_duels") or {}).get(deck, 0.0))
            played += float((row.get("self_play_deck_seats") or {}).get(deck, 0.0))
        taken[update] = played
    return taken


def report(
    directory: Path,
    sizes: list[int],
    decks: list[str],
    copy: int = 240,
    updates: int = 1920,
    step: int = 120,
) -> dict[str, Any]:
    if sorted(sizes) != sizes or sizes[0] != 1 or sizes[-1] > len(decks):
        raise SystemExit("--sizes runs up from 1, at most as many as --decks")
    result: dict[str, Any] = {"sizes": sizes, "decks": decks, "read_by": [], "networks": {}}
    read_by = list(range(updates - (END - 1) * step, updates + 1, step))
    result["read_by"] = read_by

    # The held-out deals.
    deals = set()
    heldout: dict[int, list[float]] = {}
    print(
        "The held-out deals of the first deck: each network's copy, its checkpoints after "
        + ", ".join(map(str, read_by))
        + " updates of PPO, and the last three together."
    )
    print(f"  {'Network':<10}{'Its copy':>10}" + "".join(f"{point:>8}" for point in read_by) + f"{'The three':>11}")
    for size in sizes:
        wanted = steps(size, copy, updates, step)
        games, dealt = held(directory, size, wanted)
        if not games:
            print(f"  {network(size):<10} not measured")
            continue
        deals.add(dealt)
        heldout[size] = last_three(games, updates, step)
        result["networks"].setdefault(network(size), {})["heldout"] = {
            name: mean(values) for name, values in games.items()
        } | {"three": mean(heldout[size])}
        print(
            f"  {network(size):<10}{percent(mean(games[COPY])):>10}"
            + "".join(f"{percent(mean(games[str(point)])):>8}" for point in read_by)
            + f"{percent(mean(heldout[size])):>11}"
        )
    if len(deals) > 1:
        raise SystemExit(f"{directory}/heldout: the networks are not measured on the same deals")
    print()

    # How fast each learns, beside the readings.
    curves: dict[int, dict[int, float]] = {}
    for size in sizes:
        told = result["networks"].get(network(size), {}).get("heldout")
        along = curve(directory, size, copy, step)
        if told is None or not along:
            continue
        along[0] = told[COPY]
        for point in read_by:
            along[point] = told[str(point)]
        curves[size] = dict(sorted(along.items()))
    if curves:
        points_of = sorted({point for along in curves.values() for point in along})
        print(f"How fast each learns: the held-out deals of {decks[0]}, by its updates of PPO (0 is its copy).")
        print(f"  {'Updates':>8}" + "".join(f"{network(size):>10}" for size in curves))
        for point in points_of:
            print(
                f"  {point:>8}"
                + "".join(f"{percent(curves[size].get(point)) if point in curves[size] else '':>10}" for size in curves)
            )
        print(f"The same by the seats of {decks[0]} it had played by then, in thousands.")
        for size, along in curves.items():
            taken = seats(directory, size, decks[0])
            line = f"  {network(size):<10}"
            for point, rate in along.items():
                played = 0.0 if point == 0 else taken.get(point)
                if played is None:
                    continue
                line += f"  {played / 1000:5.0f}k {percent(rate)}"
            print(line)
            result["networks"][network(size)]["curve"] = {
                str(point): {"rate": rate, "seats": 0.0 if point == 0 else taken.get(point)}
                for point, rate in along.items()
            }
        print()

    # The screens.
    screens: dict[str, dict[str, Any]] = {}
    for deck in decks[: sizes[-1]]:
        dealt_here = set()
        found: dict[str, Any] = {}
        pilot, dealt = screened(directory, f"{PILOT}.{deck}", None)
        if pilot:
            found[PILOT] = pilot[PILOT]
            dealt_here.add(dealt)
        for size in sizes:
            if size <= decks.index(deck):
                continue
            games, dealt = screened(directory, f"{network(size)}.{deck}", steps(size, copy, updates, step))
            if not games:
                continue
            dealt_here.add(dealt)
            found[network(size)] = {COPY: games[COPY], "three": last_three(games, updates, step)}
        if len(dealt_here) > 1:
            raise SystemExit(f"{directory}/screens: {deck} is not measured on the same games throughout")
        screens[deck] = found
    print("Each deck's screen: its pilot, and every network of the deck, its copy / its last three.")
    print(f"  {'Deck':<13}{'Pilot':>8}" + "".join(f"{network(size):>16}" for size in sizes))
    for deck, found in screens.items():
        line = f"  {deck:<13}{percent(mean(found[PILOT])) if PILOT in found else '-':>8}"
        for size in sizes:
            told = found.get(network(size))
            line += f"{'' if told is None else percent(mean(told[COPY])) + ' / ' + percent(mean(told['three'])):>16}"
        print(line)
        result.setdefault("screens", {})[deck] = {
            name: (mean(told) if name == PILOT else {key: mean(values) for key, values in told.items()})
            for name, told in found.items()
        }
    print()

    readings: dict[str, Any] = {}
    print(f"The readings, by each network's checkpoints after {', '.join(map(str, read_by))} updates of PPO:")
    # 1. What more decks cost the first deck.
    first = heldout.get(sizes[0])
    costs = {}
    for size in sizes[1:]:
        if first is None or size not in heldout:
            print(f"  1. {network(size)} on {decks[0]}, beside {network(sizes[0])}: not measured")
            continue
        gap, error = difference(heldout[size], first)
        worse = gap <= -WORSE and gap <= -ERRORS * error
        costs[network(size)] = {"gap": gap, "error": error, "worse": worse}
        print(
            f"  1. {network(size)} on {decks[0]}, beside {network(sizes[0])}: "
            f"{'worse' if worse else 'not shown worse'} ({points(gap, error)} on the held-out deals)"
        )
    readings["cost"] = costs
    # 2 and 3. The network of all the decks, over its screens together.
    largest = sizes[-1]
    name = network(largest)
    together = [deck for deck in decks[:largest] if name in screens.get(deck, {})]
    if largest != len(decks) or len(together) < largest:
        print(f"  2. {name} on its copy: not measured")
        print(f"  3. {name} on the pilots: not measured")
    else:
        three = [game for deck in together for game in screens[deck][name]["three"]]
        copied = [game for deck in together for game in screens[deck][name][COPY]]
        gap, error = difference(three, copied)
        gained = gap >= GAINED and gap >= ERRORS * error
        readings["gained"] = {"gap": gap, "error": error, "gained": gained, "games": len(three)}
        print(
            f"  2. {name} has gained on its copy: {'yes' if gained else 'not shown'} "
            f"({points(gap, error)} over its {largest} screens together, {len(three):,} games)"
        )
        if all(PILOT in screens[deck] for deck in together):
            pilots = [game for deck in together for game in screens[deck][PILOT]]
            gap, error = difference(three, pilots)
            over = gap >= OVER and gap >= ERRORS * error
            under = gap <= -OVER and gap <= -ERRORS * error
            readings["pilots"] = {"gap": gap, "error": error, "over": over, "under": under}
            verdict = "yes" if over else "no: under them" if under else "not shown"
            print(
                f"  3. {name} is over the decks' own pilots: {verdict} "
                f"({points(gap, error)} over the {largest} screens together)"
            )
        else:
            print(f"  3. {name} on the pilots: not measured")
    result["readings"] = readings
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 2, 3, 5, 8, 13])
    parser.add_argument(
        "--decks",
        nargs="+",
        default=[
            "blackwing", "lightsworn", "gishki", "burn", "crystal", "dragunity", "gladiator",
            "heroes", "infernity", "monarch", "morphtronic", "rock-block", "spellcaster",
        ],
    )  # fmt: skip
    parser.add_argument("--copy", type=int, default=240, help="a copy's updates for each deck")
    parser.add_argument("--updates", type=int, default=1920, help="a network's updates of PPO")
    parser.add_argument("--step", type=int, default=120, help="the updates between its checkpoints")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    found = report(args.directory, args.sizes, args.decks, args.copy, args.updates, args.step)
    if args.output is not None:
        args.output.write_text(json.dumps(found, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
