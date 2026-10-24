"""The readings of the experiment on PPO with a head that predicts
(experiments/2026-10-09-ppo-with-a-head-that-predicts.md).

    uv run python tools/ppo_head_report.py analysis/ppo-head \\
        --output analysis/ppo-head/report.json

The directory is what `experiments/commands/ppo-head.sh` leaves: the arms'
`NAME.metrics.jsonl`, `heldout/NAME.json` with each arm's checkpoints after
the same updates, and `heldout/start.json`, the checkpoints of the run they
all start from.  The arms are PPO from one checkpoint without an anchor,
by how the value head is taught: `as-before` (four passes, PPO's return at
every transaction), `one-pass`, `head-kept` (one pass, the duel's result at
four transactions of a seat), `head-kept-plain` (the same, an answer's
credit the result minus the head's estimate) and `head-kept-anchored` (as
`head-kept`, with the pilot's answers as an anchor).

A copy is read deal by deal: the share of an arm's later checkpoints at
which it wins a deal, against the share of its start's checkpoints.  The
first checkpoint of an arm is left out of that: what PPO does to a copy
takes updates to show.  One checkpoint of a run differs from the next by as
much as the runs differ.
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

ARMS = ("as-before", "one-pass", "head-kept", "head-kept-plain", "head-kept-anchored")
START = "start"
# The arm the readings are of, and the one PPO was before.
KEPT = "head-kept"
BEFORE = "as-before"
ANCHORED = "head-kept-anchored"
# The updates a value head's fit is taken over, and a rate's ends.
LAST = 40
ENDS = 10
# The marks of the readings.
PREDICTS = 0.30
BETTER = 0.02
ERRORS = 2.0
# The decisions a first choice is set beside the pilot's at, by their kind.
KINDS = ("Chain", "Idle", "Battle")


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


def ends(rows: list[dict[str, Any]], key: str) -> tuple[float | None, float | None]:
    """A figure over a run's first updates and over its last."""
    count = len(rows)
    width = min(ENDS, count)
    first, _ = mean_error(column(rows, key, range(1, width + 1)))
    last, _ = mean_error(column(rows, key, range(count - width + 1, count + 1)))
    return first, last


def plain(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def left(entry: dict[str, Any], kind: str | None = None) -> float | None:
    """Where the search kept the pilot's answer, how often the model's first
    choice is another: at all such decisions, or at those of one kind."""
    key = entry["answer_key"]
    found = key if kind is None else key.get("by_question", {}).get(kind)
    return None if not found else found.get("kept_left_rate")


def report(directory: Path) -> dict[str, Any]:
    rows = {name: metrics(directory, name) for name in ARMS}
    held = {name: measured(directory, "heldout", name) for name in ARMS}
    start = measured(directory, "heldout", START)
    result: dict[str, Any] = {"runs": {}, "differences": {}, "readings": {}}

    # Every arm starts from the same model with the same seed: its first
    # update's games are the same, whatever the update then does.  The
    # fingerprint of an update's decisions holds the pilot's answers where
    # the pilot is asked, so the anchored arm is set beside the others by
    # what it has in common with them.
    firsts = {name: found[0] for name, found in rows.items() if found}
    prints = {
        name: row.get("rollout_decision_fingerprint")
        for name, row in firsts.items()
        if name != ANCHORED
    }
    same = len(prints) > 1 and len(set(prints.values())) == 1
    result["same_first_games"] = same
    print(
        f"The first update of the {len(prints)} arms without an anchor has the same games: "
        + ("yes" if same else "NO" if len(prints) > 1 else "one arm only")
    )
    if ANCHORED in firsts and prints:
        other = firsts[next(iter(prints))]
        alike = all(
            firsts[ANCHORED].get(key) == other.get(key) for key in ("samples", "learner_win_rate")
        )
        result["anchored_first_games_alike"] = alike
        print(
            "  the anchored arm's, whose fingerprint holds the pilot's answers too, has as many "
            "decisions and wins: " + ("yes" if alike else "NO")
        )

    print()
    print("The value heads under PPO: what each explains of the results of its own games")
    print("before it trains on them, over its last updates; the entropy of its answers,")
    print("the games it wins while it samples them, both at its first and its last updates;")
    print("how far an update moves it; and the suite at its last two evaluations.")
    print(
        f"{'arm':<20}{'updates':>9}{'fit':>18}{'correlation':>13}{'entropy':>14}"
        f"{'wins':>16}{'step KL':>9}{'suite':>8}"
    )
    for name in ARMS:
        count = len(rows[name])
        updates = range(max(1, count - LAST + 1), count + 1)
        fit, error = mean_error(column(rows[name], "duel_outcome_explained_variance", updates))
        correlation, _ = mean_error(column(rows[name], "duel_outcome_correlation", updates))
        moved, _ = mean_error(column(rows[name], "duel_approximate_kl", updates))
        entropy = ends(rows[name], "duel_entropy")
        sampled = ends(rows[name], "learner_win_rate")
        played = [suite(row) for row in rows[name] if "evaluation_suite" in row][-2:]
        result["runs"][name] = {
            "updates": count,
            "fit": fit,
            "fit_error": error,
            "correlation": correlation,
            "entropy": entropy,
            "sampled_wins": sampled,
            "step_kl": moved,
            "suite": sum(played) / len(played) if played else None,
        }
        print(
            f"{name:<20}"
            f"{f'{updates.start}-{count}' if count else '-':>9}"
            f"{share(fit, error):>18}"
            f"{share(correlation):>13}"
            f"{f'{plain(entropy[0])} to {plain(entropy[1])}':>14}"
            f"{f'{percent(sampled[0])} to {percent(sampled[1])}':>16}"
            f"{plain(moved, 3):>9}"
            f"{percent(result['runs'][name]['suite']):>8}"
        )

    # An arm's checkpoints by the updates of PPO they had: its steps count on
    # from the start's last.
    began = max(start) if start else 0
    measured_arms = [name for name in ARMS if held[name]]
    steps = (
        sorted(set.intersection(*(set(held[name]) for name in measured_arms)))
        if measured_arms
        else []
    )
    later = steps[1:] if len(steps) > 1 else steps
    result["checkpoints"] = {
        "start": sorted(start),
        "arms": [step - began for step in steps],
        "read": [step - began for step in later],
    }
    scores: dict[str, list[float]] = {}
    print()
    if not start or not steps:
        print("The copies: the start or the arms are not measured.")
    else:
        scores[START] = over_checkpoints(start, sorted(start))
        deals = len(scores[START])
        print("The copies: held-out deals won.  The start by its own updates, an arm by the")
        print("updates of PPO it had; `read` is what a copy is read by: all of the start's")
        print(f"checkpoints, an arm's after {', '.join(str(step - began) for step in later)}.")
        print(f"{START:<20}" + "".join(f"{step:>8}" for step in sorted(start)) + f"{'read':>8}")
        each = [sum(wins(start[step]["deals"])) / deals for step in sorted(start)]
        result["runs"][START] = {
            "deals": dict(zip(sorted(start), each)),
            "deals_read": sum(scores[START]) / deals,
        }
        print(
            f"{'':<20}"
            + "".join(f"{percent(value):>8}" for value in each)
            + f"{percent(result['runs'][START]['deals_read']):>8}"
        )
        print(f"{'arm':<20}" + "".join(f"{step - began:>8}" for step in steps) + f"{'read':>8}")
        for name in measured_arms:
            scores[name] = over_checkpoints(held[name], later)
            each = [sum(wins(held[name][step]["deals"])) / deals for step in steps]
            result["runs"][name]["deals"] = dict(zip([step - began for step in steps], each))
            result["runs"][name]["deals_read"] = sum(scores[name]) / deals
            print(
                f"{name:<20}"
                + "".join(f"{percent(value):>8}" for value in each)
                + f"{percent(result['runs'][name]['deals_read']):>8}"
            )
        paired_with_pilot = start[max(start)]["deals"].get("paired", {}).get("pilot")
        if paired_with_pilot:
            pilot = paired_with_pilot["both"] + paired_with_pilot["pilot_only"]
            print(f"  on those deals the pilot wins {percent(pilot / deals)}")

    def over(first: str, second: str) -> tuple[float, float] | None:
        if first not in scores or second not in scores:
            return None
        found = difference(scores[first], scores[second])
        result["differences"][f"{first} over {second}"] = found
        return found

    def said(found: tuple[float, float] | None) -> str:
        return "not measured" if found is None else points(*found)

    print()
    print("Deal by deal, over the checkpoints a copy is read by:")
    against_start = {name: over(name, START) for name in ARMS}
    for name in ARMS:
        print(f"  {name} over its start: {said(against_start[name])}")
    lesson = over(KEPT, BEFORE)
    print(f"  {KEPT} over {BEFORE}: {said(lesson)}")
    print(f"  {KEPT} over one-pass: {said(over(KEPT, 'one-pass'))}")
    print(f"  {KEPT} over head-kept-plain: {said(over(KEPT, 'head-kept-plain'))}")
    print(f"  {ANCHORED} over {KEPT}: {said(over(ANCHORED, KEPT))}")

    if start and steps:
        print()
        print("Where the search kept the pilot's answer, how often the first choice is another:")
        print("at all those decisions and by their kind, at the start's last checkpoint and at")
        print("each arm's.")
        print(f"{'':<20}{'all':>8}" + "".join(f"{kind:>8}" for kind in KINDS))
        for name, entry in [(START, start[max(start)])] + [
            (name, held[name][steps[-1]]) for name in measured_arms
        ]:
            rates = [left(entry)] + [left(entry, kind) for kind in KINDS]
            result["runs"][name]["leaves_the_pilot"] = dict(zip(("all", *KINDS), rates))
            print(f"{name:<20}" + "".join(f"{percent(rate):>8}" for rate in rates))

    def no_worse(found: tuple[float, float] | None) -> bool | None:
        return None if found is None else found[0] >= -ERRORS * found[1]

    def better(found: tuple[float, float] | None) -> bool | None:
        return None if found is None else found[0] >= BETTER and found[0] > ERRORS * found[1]

    fit = result["runs"][KEPT]["fit"]
    readings = {
        "head_predicts": None if fit is None else fit >= PREDICTS,
        "copy_holds": no_worse(against_start[KEPT]),
        "copy_gains": better(against_start[KEPT]),
        "anchored_copy_gains": better(against_start[ANCHORED]),
        "lesson_holds_it": better(lesson),
    }
    result["readings"] = readings
    mark = f"the mark {BETTER * 100:.0f} points and {ERRORS:.0f} standard errors"
    print()
    print("The readings:")
    print(
        reading(
            f"under PPO the head keeps predicting ({KEPT} explains {share(fit)}, "
            f"the mark {PREDICTS})",
            readings["head_predicts"],
        )
    )
    print(
        reading(
            f"PPO no longer undoes the copy ({KEPT} over its start {said(against_start[KEPT])})",
            readings["copy_holds"],
        )
    )
    print(
        reading(
            f"PPO adds to the copy ({said(against_start[KEPT])}, {mark})", readings["copy_gains"]
        )
    )
    print(
        reading(
            f"with the anchor PPO adds to the copy ({said(against_start[ANCHORED])}, {mark})",
            readings["anchored_copy_gains"],
        )
    )
    print(
        reading(
            f"the head's lesson is what holds it ({KEPT} over {BEFORE} {said(lesson)}, {mark})",
            readings["lesson_holds_it"],
        )
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    found = report(args.directory)
    if args.output:
        args.output.write_text(json.dumps(found, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
