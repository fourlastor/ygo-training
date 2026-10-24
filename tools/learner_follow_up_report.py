"""The readings of the follow-up to the experiment on the learner itself
(experiments/2026-10-08-value-head-and-board.md): the head that predicts and
the best copy in one network.

    uv run python tools/learner_follow_up_report.py analysis/learner-follow-up \\
        --output analysis/learner-follow-up/report.json

The directory is what `experiments/commands/learner-follow-up.sh` leaves: the
four runs' `NAME.metrics.jsonl` and `heldout/NAME.json`, each with the run's
checkpoints after the same updates.  The runs are the board by the positions
a game's result is taught at: `once` and `once-few` on the pooled board,
`once-cards` and `once-cards-few` on the board as its cards.

A copy is read over all the checkpoints measured, deal by deal: the share of
them at which a run wins a deal, against the same of another run.  One
checkpoint of a run differs from the next by as much as the runs differ.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from learner_report import column, mean_error, measured, metrics, reading, share, suite
from search_leads_report import percent, points, wins

RUNS = ("once", "once-few", "once-cards", "once-cards-few")
# The updates a value head's fit is taken over: a run's last ones.
LAST = 40
# The marks of the readings.
PREDICTS = 0.30
BETTER = 0.02
ERRORS = 2.0


def over_checkpoints(found: dict[int, dict[str, Any]], steps: list[int]) -> list[float]:
    """For each deal, the share of the checkpoints at which the run wins it."""
    won = [wins(found[step]["deals"]) for step in steps]
    return [sum(deal) / len(deal) for deal in zip(*won, strict=True)]


def difference(first: list[float], second: list[float]) -> tuple[float, float]:
    """The mean of `first - second` over the deals, and its standard error."""
    if len(first) != len(second) or len(first) < 2:
        raise ValueError(f"{len(first)} deals against {len(second)}: not the same deals")
    centre, error = mean_error([a - b for a, b in zip(first, second)])
    assert centre is not None and error is not None
    return centre, error


def report(directory: Path) -> dict[str, Any]:
    rows = {name: metrics(directory, name) for name in RUNS}
    held = {name: measured(directory, "heldout", name) for name in RUNS}
    result: dict[str, Any] = {"runs": {}, "differences": {}, "readings": {}}

    # The runs are read against each other only on the same games.
    prints = {
        name: [row.get("rollout_decision_fingerprint") for row in rows[name]] for name in RUNS
    }
    shared = min((len(found) for found in prints.values() if found), default=0)
    same_games = shared > 0 and all(
        found[:shared] == prints["once"][:shared] for found in prints.values() if found
    )
    result["same_games"] = same_games
    print(
        f"The runs have the same games over the {shared} updates they share: "
        + ("yes" if same_games else "NO, and nothing below compares like with like")
    )

    print()
    print("The value heads: what each explains of the results of games new to it")
    print("over its last updates, its correlation with them, and at its last checkpoint")
    print("its values against the worlds of the held-out games.")
    print(
        f"{'run':<16}{'updates':>9}{'fit':>18}{'correlation':>13}{'tells':>8}{'stands':>8}{'suite':>8}"
    )
    for name in RUNS:
        count = len(rows[name])
        updates = range(max(1, count - LAST + 1), count + 1)
        fit, error = mean_error(column(rows[name], "imitation_value_explained_variance", updates))
        correlation, _ = mean_error(column(rows[name], "imitation_value_correlation", updates))
        played = [suite(row) for row in rows[name] if "evaluation_suite" in row][-2:]
        told = None
        if held[name]:
            told = (held[name][max(held[name])]["answer_key"]["tried"] or {}).get("value")
        result["runs"][name] = {
            "updates": count,
            "fit": fit,
            "fit_error": error,
            "correlation": correlation,
            "suite": sum(played) / len(played) if played else None,
            "value": told,
        }
        print(
            f"{name:<16}"
            f"{f'{updates.start}-{count}' if count else '-':>9}"
            f"{share(fit, error):>18}"
            f"{share(correlation):>13}"
            f"{percent(told['told'] if told else None):>8}"
            f"{percent(told['explained'] if told else None):>8}"
            f"{percent(result['runs'][name]['suite']):>8}"
        )

    steps = (
        sorted(set.intersection(*(set(found) for found in held.values())))
        if all(held.values())
        else []
    )
    result["checkpoints"] = steps
    print()
    if not steps:
        print("The copies: no checkpoints measured for all four runs.")
        scores: dict[str, list[float]] = {}
    else:
        print("The copies: held-out deals won after each number of updates, and over all of them.")
        print(f"{'run':<16}" + "".join(f"{step:>8}" for step in steps) + f"{'all':>8}")
        scores = {name: over_checkpoints(held[name], steps) for name in RUNS}
        for name in RUNS:
            each = [sum(wins(held[name][step]["deals"])) / len(scores[name]) for step in steps]
            result["runs"][name]["deals"] = dict(zip(steps, each))
            result["runs"][name]["deals_all"] = sum(scores[name]) / len(scores[name])
            print(
                f"{name:<16}"
                + "".join(f"{percent(value):>8}" for value in each)
                + f"{percent(result['runs'][name]['deals_all']):>8}"
            )
        paired_with_pilot = held["once"][steps[-1]]["deals"].get("paired", {}).get("pilot")
        if paired_with_pilot:
            pilot = paired_with_pilot["both"] + paired_with_pilot["pilot_only"]
            print(f"  on those deals the pilot wins {percent(pilot / len(scores['once']))}")

    def over(first: str, second: str) -> tuple[float, float] | None:
        if not scores:
            return None
        found = difference(scores[first], scores[second])
        result["differences"][f"{first} over {second}"] = found
        return found

    few_pooled = over("once-few", "once")
    few_cards = over("once-cards-few", "once-cards")
    cards_all = over("once-cards", "once")
    cards_few = over("once-cards-few", "once-few")
    cards = None
    if scores:
        # The board as its cards against the pooled board, whichever way the
        # value head is taught: both pairs, deal by deal.
        cards = difference(
            [(a + b) / 2 for a, b in zip(scores["once-cards"], scores["once-cards-few"])],
            [(a + b) / 2 for a, b in zip(scores["once"], scores["once-few"])],
        )
        result["differences"]["cards over pooled"] = cards
        print()
        print("Deal by deal, over the checkpoints:")
        for text, found in (
            ("four positions on the pooled board (once-few over once)", few_pooled),
            ("four positions on the cards (once-cards-few over once-cards)", few_cards),
            ("the cards at every position (once-cards over once)", cards_all),
            ("the cards at four positions (once-cards-few over once-few)", cards_few),
            ("the cards, both ways of teaching the head", cards),
        ):
            assert found is not None
            print(f"  {text}: {points(*found)}")

    def no_worse(found: tuple[float, float] | None) -> bool | None:
        return None if found is None else found[0] >= -ERRORS * found[1]

    both = result["runs"]["once-cards-few"]["fit"]
    readings = {
        "both_predicts": None if both is None else both >= PREDICTS,
        "both_copies": no_worse(few_cards),
        "few_copies": no_worse(few_pooled),
        "cards_help": None
        if cards is None
        else cards[0] >= BETTER and cards[0] > ERRORS * cards[1],
    }
    result["readings"] = readings
    print()
    print("The readings:")
    print(
        reading(
            f"with both, the value head predicts (once-cards-few explains {share(both)}, "
            f"the mark {PREDICTS})",
            readings["both_predicts"],
        )
    )
    print(
        reading(
            "with both, the copy is no worse than with the cards alone "
            f"({points(*few_cards) if few_cards else '-'})",
            readings["both_copies"],
        )
    )
    print(
        reading(
            "on the pooled board four positions cost the copy nothing "
            f"({points(*few_pooled) if few_pooled else '-'})",
            readings["few_copies"],
        )
    )
    print(
        reading(
            f"the board as its cards helps the copy ({points(*cards) if cards else '-'}, "
            f"the mark {100 * BETTER:.0f} points and {ERRORS:.0f} standard errors)",
            readings["cards_help"],
        )
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = report(arguments.directory)
    if arguments.output:
        arguments.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
