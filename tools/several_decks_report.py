"""The readings of one network of several decks beside a network of each.

    uv run python tools/several_decks_report.py analysis/several-decks \\
        --decks blackwing lightsworn gishki --updates 200 \\
        --output analysis/several-decks/report.json

The directory is what `experiments/commands/several-decks.sh` leaves.  For
each deck a network of its own that imitated the deck's pilot
(`alone-DECK.metrics.jsonl`, `alone-DECK.run.json`), and one network that
imitated the pilots of all the decks, each in an equal share of its games
(`together.*`), for as many updates as there are decks times those of a
deck's own network.  On each deck's screen, with how every game ended:
the deck's pilot (`screens/DECK.pilot.json`), its own network
(`screens/DECK.alone.json`) and the network of all the decks
(`screens/DECK.together.json`), which is set beside the deck's own after
as many games of the deck.

A deck is read game by game of its screen, by the share of a network's last
checkpoints at which it wins a game, and only when the networks have had the
updates of `--updates`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from learner_report import metrics, reading, suite
from ppo_head_report import difference
from search_leads_report import percent, points

ALONE = "alone"
TOGETHER = "together"
PILOT = "pilot"
# The checkpoints a network is read by: its last ones.
READ = 2
# The updates of a deck's own network that a figure of its learning is of.
BEFORE = 10
# A network is worse than another when it is this far under it, and by this
# many standard errors.
WORSE = 0.02
ERRORS = 2.0


def said(directory: Path, name: str) -> dict[str, Any]:
    """What a run was started with."""
    path = directory / f"{name}.run.json"
    return json.loads(path.read_text(encoding="utf8")) if path.exists() else {}


def by_update(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """A run's rows by the updates it had had, the row's own among them."""
    return {int(row["version"]) + 1: row for row in rows}


def screened(directory: Path, name: str) -> tuple[dict[int | None, list[float]], Any]:
    """A measurement on a deck's screen: for each checkpoint, by the updates
    it had (None for the pilot), whether it won each game, the cases in the
    order of their names; and what tells the games, to see two measurements
    are of the same ones."""
    path = directory / "screens" / f"{name}.json"
    if not path.exists():
        return {}, None
    found = json.loads(path.read_text(encoding="utf8"))
    games: dict[int | None, list[float]] = {}
    for record in found["records"]:
        cases = record["evaluation_suite"]
        if any("outcomes" not in cases[case] for case in cases):
            continue
        step = record.get("checkpoint_step")
        games[None if step is None else int(step)] = [
            float(outcome == "win") for case in sorted(cases) for outcome in cases[case]["outcomes"]
        ]
    cases = found["records"][0]["evaluation_suite"] if found["records"] else {}
    dealt = (found.get("seed"), tuple((case, cases[case].get("games")) for case in sorted(cases)))
    return games, dealt


def over_checkpoints(games: dict[int | None, list[float]], steps: list[int]) -> list[float]:
    """For each game, the share of the checkpoints at which it is won."""
    return [sum(game) / len(game) for game in zip(*(games[step] for step in steps), strict=True)]


def agreement(
    rows: dict[int, dict[str, Any]], deck: str | None, until: int, updates: int
) -> float | None:
    """How often the first choice is the pilot's, on games new to the
    network, over the `updates` updates that end at `until`: of all its
    decisions, or of those of one deck."""
    agreed = decisions = 0.0
    for update in range(max(1, until - updates + 1), until + 1):
        row = rows.get(update)
        if row is None:
            continue
        if deck is None:
            rate, count = row.get("imitation_model_teacher_agreement"), row.get("samples")
        else:
            rate = row.get("imitation_model_teacher_agreement_by_deck", {}).get(deck)
            count = row.get("imitation_samples_by_deck", {}).get(deck)
        if rate is None or not count:
            continue
        agreed += float(rate) * float(count)
        decisions += float(count)
    return agreed / decisions if decisions else None


def gap(scores: dict[str, list[float]], first: str, second: str) -> tuple[float, float] | None:
    """The mean of `first - second` over the games, and its standard error."""
    if first not in scores or second not in scores:
        return None
    return difference(scores[first], scores[second])


def told_of(found: tuple[float, float] | None) -> str:
    return "not measured" if found is None else points(*found)


def report(directory: Path, decks: list[str], updates: int | None = None) -> dict[str, Any]:
    count = len(decks)
    rows = {deck: by_update(metrics(directory, f"{ALONE}-{deck}")) for deck in decks}
    together = by_update(metrics(directory, TOGETHER))
    told = {
        name: said(directory, name) for name in [*(f"{ALONE}-{deck}" for deck in decks), TOGETHER]
    }
    result: dict[str, Any] = {"decks": {}, "differences": {}, "readings": {}}

    print(f"One network of {count} decks ({', '.join(decks)}) beside a network of each.")
    starts = {name: found.get("initial_model_fingerprint") for name, found in told.items() if found}
    if starts:
        same = len(set(starts.values())) == 1 and len(starts) == count + 1
        result["same_start"] = same
        print("  every network starts from the same first network: " + ("yes" if same else "NO"))
    named = told[TOGETHER].get("learners")
    if named is not None:
        result["together_plays"] = named
        print(
            "  the network of all plays "
            + ", ".join(named)
            + (
                ": the decks read here"
                if sorted(named) == sorted(decks)
                else ": NOT the decks read"
            )
        )
    played: dict[str, float] = {}
    for row in together.values():
        for deck, games in row.get("learner_deck_duels", {}).items():
            played[deck] = played.get(deck, 0.0) + float(games)
    if played:
        result["together_games"] = played
        print(
            f"  in its {len(together)} updates it has had "
            + ", ".join(f"{int(played.get(deck, 0)):,} games of {deck}" for deck in decks)
        )
    for deck in decks:
        if rows[deck]:
            duels = sum(float(row.get("duels", 0)) for row in rows[deck].values())
            print(
                f"  the network of {deck} alone has had {int(duels):,} games of it in its "
                f"{len(rows[deck])} updates"
            )
    # The Blackwing pilot's copy made before, where its figures are there:
    # the same lesson on the same deals should be the same run.
    before = by_update(metrics(directory, "before"))
    first = next((deck for deck in decks if deck == "blackwing" and rows[deck]), None)
    if before and first:
        shared = sorted(set(before) & set(rows[first]))
        alike = bool(shared) and all(
            before[update].get("rollout_training_fingerprint")
            == rows[first][update].get("rollout_training_fingerprint")
            for update in shared
        )
        result["blackwing_as_before"] = alike
        print(
            f"  the network of blackwing alone is taught on the games of the copy made before, "
            f"over their {len(shared)} updates in common: " + ("yes" if alike else "NO")
        )
        # And, where it is, the same network: it then wins the same games.
        played = [
            (suite(rows[first][update]), suite(before[update]))
            for update in shared
            if "evaluation_suite" in rows[first][update] and "evaluation_suite" in before[update]
        ]
        if played:
            result["blackwing_suite_as_before"] = all(here == there for here, there in played)
            print(
                f"    and wins of its screen what that copy won, at their {len(played)} "
                "evaluations in common: "
                + (
                    "yes"
                    if result["blackwing_suite_as_before"]
                    else "no ("
                    + ", ".join(f"{percent(here)} for {percent(there)}" for here, there in played)
                    + ")"
                )
            )

    readings: dict[str, bool | None] = {}
    for deck in decks:
        alone, alone_dealt = screened(directory, f"{deck}.{ALONE}")
        both, both_dealt = screened(directory, f"{deck}.{TOGETHER}")
        pilot, pilot_dealt = screened(directory, f"{deck}.{PILOT}")
        found: dict[str, Any] = {"rows": []}
        result["decks"][deck] = found
        scores: dict[str, list[float]] = {}
        print()
        rate = sum(pilot[None]) / len(pilot[None]) if None in pilot else None
        found["pilot"] = rate
        games = len(pilot[None]) if None in pilot else None
        print(
            f"{deck}: its pilot wins {percent(rate)} of its screen"
            + ("" if games is None else f" ({games:,} games)")
            + "."
        )
        dealt = {name for name in (alone_dealt, both_dealt, pilot_dealt) if name is not None}
        found["same_games"] = len(dealt) <= 1
        if len(dealt) > 1:
            print("  the measurements are NOT of the same games: nothing is set beside another")
            readings[f"holds_{deck}"] = None
            continue
        # A point is so many updates of the deck's own network, and as many
        # games of the deck in the network of all: its updates times the decks.
        steps = sorted(step for step in alone if step is not None and count * step in both)
        duels = int(next(iter(rows[deck].values())).get("duels", 0)) if rows[deck] else 0
        print("  After so many games of the deck: how often the first choice is the pilot's on")
        print(
            f"  games new to the network, over the {BEFORE} updates of the deck's own before, and"
        )
        print("  the games of the screen it wins.")
        print(
            f"  {'games':>9}{'updates':>12}{'agrees alone':>14}{'together':>10}"
            f"{'wins alone':>12}{'together':>10}"
        )
        shown = steps or sorted(update for update, row in rows[deck].items() if "evaluation" in row)
        for step in shown:
            line = {
                "updates": step,
                "games": step * duels,
                "agrees_alone": agreement(rows[deck], None, step, BEFORE),
                "agrees_together": agreement(together, deck, count * step, count * BEFORE),
                "wins_alone": sum(alone[step]) / len(alone[step]) if step in alone else None,
                "wins_together": (
                    sum(both[count * step]) / len(both[count * step])
                    if count * step in both
                    else None
                ),
            }
            found["rows"].append(line)
            print(
                f"  {line['games']:>9,}{f'{step} / {count * step}':>12}"
                f"{percent(line['agrees_alone']):>14}{percent(line['agrees_together']):>10}"
                f"{percent(line['wins_alone']):>12}{percent(line['wins_together']):>10}"
            )
        read = steps[-READ:]
        complete = (
            bool(steps) and updates is not None and steps[-1] == updates and len(read) == READ
        )
        found["read"] = read if complete else []
        if not complete:
            reached = f"as far as {steps[-1]} updates" if steps else "at no update"
            print(
                f"  measured {reached}"
                + ("" if updates is None else f" of {updates}")
                + ": the deck is not read"
            )
            readings[f"holds_{deck}"] = None
            continue
        scores[ALONE] = over_checkpoints(alone, read)
        scores[TOGETHER] = over_checkpoints(both, [count * step for step in read])
        if None in pilot:
            scores[PILOT] = pilot[None]
        gaps = {
            (first, second): gap(scores, first, second)
            for first, second in ((TOGETHER, ALONE), (ALONE, PILOT), (TOGETHER, PILOT))
        }
        for (first, second), found_gap in gaps.items():
            if found_gap is not None:
                result["differences"][f"{deck}: {first} over {second}"] = found_gap
        print(
            "  Game by game, by the checkpoints after "
            + " and ".join(str(step) for step in read)
            + " updates of the deck's own network:"
        )
        held = gaps[TOGETHER, ALONE]
        print(f"    the network of all over the deck's own: {told_of(held)}")
        print(f"    the deck's own over the pilot: {told_of(gaps[ALONE, PILOT])}")
        print(f"    the network of all over the pilot: {told_of(gaps[TOGETHER, PILOT])}")
        readings[f"holds_{deck}"] = (
            None if held is None else not (held[0] <= -WORSE and held[0] < -ERRORS * held[1])
        )

    each = [readings.get(f"holds_{deck}") for deck in decks]
    readings["holds_every_deck"] = None if None in each else all(each)
    result["readings"] = readings
    print()
    print("The readings:")
    mark = f"worse is {WORSE * 100:.0f} points and {ERRORS:.0f} standard errors under"
    for deck in decks:
        held = result["differences"].get(f"{deck}: {TOGETHER} over {ALONE}")
        print(
            reading(
                f"the network of all the decks plays {deck} no worse than the network of "
                f"{deck} alone ({told_of(held)}; {mark})",
                readings[f"holds_{deck}"],
            )
        )
    print(
        reading(
            f"one network holds the {count} decks as well as a network each",
            readings["holds_every_deck"],
        )
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--decks", nargs="+", required=True, help="the decks, as the runs name them"
    )
    parser.add_argument(
        "--updates", type=int, help="the updates a deck's own network was to have, for the readings"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    found = report(args.directory, args.decks, args.updates)
    if args.output:
        args.output.write_text(json.dumps(found, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
