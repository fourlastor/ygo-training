"""The readings of the experiment on the learner itself: the value head and
the board (experiments/2026-10-08-value-head-and-board.md).

    uv run python tools/learner_report.py analysis/learner \\
        --output analysis/learner/report.json

The directory is what `experiments/commands/learner-value-board.sh` leaves:
every run's `NAME.metrics.jsonl`, `heldout/NAME.json` and
`validation/NAME.json` from `evaluate-search`, and `references.json`.  The
runs are `again` (the seed's recipe), `once`, `once-few`, `once-reading` and
`once-cards`, the two passes with the lead head on `once` and `once-cards`,
and the existing seed (`seed-value`, `heldout/seed.json`).

A value head's fit is the share of the results it explains at the first
steps of the transactions of an update's games before it trains on them:
games new to it.  The readings are applied as the record fixes them.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from search_leads_report import paired, percent, points, wins

# The runs, and how many of a run's last updates its fit is taken over: those
# of `again` are four passes each, the optimizer steps of four of the others.
RUNS = ("again", "once", "once-few", "once-reading", "once-cards")
LAST = {"again": 10}
LAST_ONCE = 40
# The marks of the readings.
PREDICTS = 0.20
READS = 0.30
MORE = 0.05
BETTER = 0.02
ERRORS = 2.0
# The updates `once-few` is read against `once` over.
FEW = (81, 120)


def mean_error(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    centre = sum(values) / len(values)
    if len(values) < 2:
        return centre, None
    spread = sum((value - centre) ** 2 for value in values) / (len(values) - 1)
    return centre, math.sqrt(spread / len(values))


def share(value: float | None, error: float | None = None) -> str:
    if value is None:
        return "-"
    return f"{value:+.3f}" + ("" if error is None else f" ± {error:.3f}")


def metrics(directory: Path, name: str) -> list[dict[str, Any]]:
    path = directory / f"{name}.metrics.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line]


def column(rows: list[dict[str, Any]], key: str, updates: range) -> list[float]:
    """A figure of a run's updates, counted from 1 at its first."""
    if not rows:
        return []
    first = min(int(row["version"]) for row in rows)
    return [
        float(row[key])
        for row in rows
        if int(row["version"]) - first + 1 in updates and row.get(key) is not None
    ]


def last(rows: list[dict[str, Any]], name: str) -> range:
    count = len(rows)
    return range(max(1, count - LAST.get(name, LAST_ONCE) + 1), count + 1)


def suite(row: dict[str, Any]) -> float:
    cases = row["evaluation_suite"].values()
    return sum(case["wins"] for case in cases) / sum(case["games"] for case in cases)


def measured(directory: Path, rows: str, name: str) -> dict[int, dict[str, Any]]:
    """A run's checkpoints on a set of logged games, by the updates they had."""
    path = directory / rows / f"{name}.json"
    if not path.exists():
        return {}
    found = json.loads(path.read_text(encoding="utf8"))["checkpoints"]
    return {int(entry["checkpoint_step"]): entry for entry in found}


def against(mine: dict[str, Any], theirs: dict[str, Any]) -> tuple[float, float]:
    return paired(wins(mine["deals"]), wins(theirs["deals"]))


def reading(text: str, met: bool | None) -> str:
    verdict = "not measured" if met is None else "yes" if met else "no"
    return f"  {text}: {verdict}"


def report(directory: Path) -> dict[str, Any]:
    found: dict[str, Any] = {"runs": {}, "readings": {}}
    references_path = directory / "references.json"
    result = None
    if references_path.exists():
        references = json.loads(references_path.read_text(encoding="utf8"))
        result = references["worlds"]["result"]
        found["references"] = {
            "result": result,
            "networks": [
                {key: network[key] for key in ("target", "opponent", "of_one_result")}
                for network in references["networks"]
            ],
        }
        print(
            f"Of one result a perfect predictor explains {percent(result)} at the "
            f"{references['decisions']} opening decisions of the {references['games']} held-out games."
        )
        for network in references["networks"]:
            lesson = "the games' results" if network["target"] == "result" else "the worlds' rates"
            sees = "with" if network["opponent"] else "without"
            print(
                f"  a small network taught {lesson}, {sees} the opponent's deck: "
                f"{percent(network['of_one_result'])}"
            )
        print()

    # The value heads.
    print("The value heads: what each explains of the results of games new to it (its fit),")
    print("its correlation with them, and on the held-out logged games its values against")
    print("the worlds: the correlation, what it tells of what can be told, what its values")
    print("explain as they stand, and what that is of one result.")
    print(
        f"{'run':<14}{'updates':>8}{'fit':>17}{'correlation':>13}"
        f"{'worlds':>9}{'tells':>8}{'stands':>8}{'of a result':>13}"
    )
    seed_rows = metrics(directory, "seed-value")
    rows_of = {name: metrics(directory, name) for name in RUNS}
    held = {name: measured(directory, "heldout", name) for name in ("seed", *RUNS)}

    def value_row(name: str, fit: Any, correlation: Any, updates: str, entry: Any) -> None:
        value = None if entry is None else (entry["answer_key"]["tried"] or {}).get("value")
        told = None if value is None else value["told"]
        of_result = None if told is None or value["result"] is None else told * value["result"]
        print(
            f"{name:<14}{updates:>8}{share(*fit):>17}{share(correlation):>13}"
            f"{share(None if value is None else value['correlation']):>9}"
            f"{percent(told):>8}{percent(None if value is None else value['explained']):>8}"
            f"{percent(of_result):>13}"
        )
        found["runs"].setdefault(name, {}).update(
            {"fit": fit[0], "fit_error": fit[1], "correlation": correlation, "worlds": value}
        )

    if seed_rows:
        own = seed_rows[0]
        value_row(
            "seed",
            (own.get("imitation_value_explained_variance"), None),
            own.get("imitation_value_correlation"),
            "its own",
            max(held["seed"].items(), default=(0, None))[1],
        )
    fits: dict[str, float | None] = {}
    for name in RUNS:
        rows = rows_of[name]
        if not rows:
            continue
        over = last(rows, name)
        fit = mean_error(column(rows, "imitation_value_explained_variance", over))
        correlation = mean_error(column(rows, "imitation_value_correlation", over))[0]
        fits[name] = fit[0]
        value_row(
            name,
            fit,
            correlation,
            f"{over.start}-{over.stop - 1}",
            max(held[name].items(), default=(0, None))[1],
        )
    print()

    # The copies.
    print("The copies: agreement with the pilot on games new to them over the same updates,")
    print("the suite at the last two evaluations, and the held-out deals won at the last")
    print("checkpoint measured, with what that has over `again` deal by deal.")
    print(f"{'run':<14}{'agreement':>11}{'suite':>9}{'deals won':>11}{'against again':>17}")
    control = max(held["again"].items(), default=(0, None))[1]
    for name in ("seed", *RUNS):
        rows = rows_of.get(name, [])
        entry = max(held[name].items(), default=(0, None))[1]
        agreement = (
            mean_error(column(rows, "imitation_model_teacher_agreement", last(rows, name)))[0]
            if rows
            else None
        )
        evaluated = [suite(row) for row in rows if "evaluation_suite" in row][-2:]
        difference = (
            None if entry is None or control is None or name == "again" else against(entry, control)
        )
        print(
            f"{name:<14}{percent(agreement):>11}"
            f"{percent(sum(evaluated) / len(evaluated) if evaluated else None):>9}"
            f"{percent(None if entry is None else entry['deals']['win_rate']):>11}"
            f"{'' if difference is None else points(*difference):>17}"
        )
        found["runs"].setdefault(name, {}).update(
            {
                "agreement": agreement,
                "suite": sum(evaluated) / len(evaluated) if evaluated else None,
                "deals": None if entry is None else entry["deals"]["win_rate"],
                "against_again": difference,
            }
        )
    if control is not None:
        references = control["deals"]["references"]
        print(
            "  on those deals: "
            + ", ".join(
                f"{name} {percent(reference['win_rate'])}" for name, reference in references.items()
            )
        )
    print()

    # The lead heads.
    leads: dict[str, float | None] = {}
    print("The lead heads after one pass through the logged games: what each explains of")
    print("the measured leads before an update, over the second half of the pass, and its")
    print("correlation with them on the validation games.")
    for name in ("once-leads", "once-cards-leads"):
        rows = metrics(directory, name)
        if not rows:
            continue
        half = range(len(rows) // 2 + 1, len(rows) + 1)
        fit = mean_error(column(rows, "imitation_lead_explained_variance", half))
        entry = max(measured(directory, "validation", name).items(), default=(0, None))[1]
        head = None if entry is None else (entry["answer_key"]["tried"] or {}).get("head")
        correlation = None if head is None else head["correlation"]
        leads[name] = fit[0]
        print(f"  {name:<18}{share(*fit):>17}{share(correlation):>10}")
        found["runs"][name] = {"fit": fit[0], "fit_error": fit[1], "correlation": correlation}
    print()

    # The readings.
    def over_same(first: str, second: str, updates: range) -> tuple[float | None, float | None]:
        mine = column(rows_of.get(first, []), "imitation_value_explained_variance", updates)
        theirs = column(rows_of.get(second, []), "imitation_value_explained_variance", updates)
        if not mine or len(mine) != len(theirs):
            return None, None
        return mean_error([a - b for a, b in zip(mine, theirs, strict=True)])

    def deals(first: str, second: str) -> tuple[float, float] | None:
        mine = max(held[first].items(), default=(0, None))[1]
        theirs = max(held[second].items(), default=(0, None))[1]
        return None if mine is None or theirs is None else against(mine, theirs)

    once = fits.get("once")
    shaping = (
        over_same("once", "once-reading", last(rows_of["once"], "once"))
        if rows_of["once"]
        else (None, None)
    )
    fewer = over_same("once-few", "once", range(FEW[0], FEW[1] + 1))
    copy = deals("once", "again")
    board = deals("once-cards", "once")
    cards_value = (
        over_same("once-cards", "once", last(rows_of["once"], "once"))
        if rows_of["once"]
        else (None, None)
    )
    cards_leads = (
        None
        if leads.get("once-leads") is None or leads.get("once-cards-leads") is None
        else leads["once-cards-leads"] - leads["once-leads"]
    )
    readings = {
        "predicts": None if once is None else once >= PREDICTS,
        "reads": None if once is None else once >= READS,
        "copy_no_worse": None if copy is None else copy[0] >= -ERRORS * copy[1],
        "copy_better": None if copy is None else copy[0] >= BETTER and copy[0] > ERRORS * copy[1],
        "must_shape": None if shaping[0] is None else shaping[0] >= MORE,
        "fewer_positions": None if fewer[0] is None else fewer[0] >= MORE,
        "board_helps_copy": (
            None if board is None else board[0] >= BETTER and board[0] > ERRORS * board[1]
        ),
        "board_helps_heads": (
            None
            if cards_value[0] is None and cards_leads is None
            else (cards_value[0] is not None and cards_value[0] >= MORE)
            or (cards_leads is not None and cards_leads >= MORE)
        ),
    }
    found["readings"] = readings
    found["differences"] = {
        "once_against_reading": shaping,
        "few_against_once": fewer,
        "once_against_again": copy,
        "cards_against_once": board,
        "cards_value_against_once": cards_value,
        "cards_leads_against_once": cards_leads,
    }
    print("The readings:")
    print(
        reading(
            f"the value head predicts (once explains {share(once)}, the mark {PREDICTS})",
            readings["predicts"],
        )
    )
    print(reading(f"it reads the position (the mark {READS})", readings["reads"]))
    print(
        reading(
            "the copy is no worse for it (once against again on the held-out deals: "
            f"{'-' if copy is None else points(*copy)})",
            readings["copy_no_worse"],
        )
    )
    print(reading("the copy is better (2 points and two standard errors)", readings["copy_better"]))
    print(
        reading(
            f"the head must shape what it reads (once over once-reading: {share(*shaping)}, "
            f"the mark {MORE})",
            readings["must_shape"],
        )
    )
    print(
        reading(
            f"fewer positions of a game are needed (once-few over once, updates {FEW[0]} to "
            f"{FEW[1]}: {share(*fewer)})",
            readings["fewer_positions"],
        )
    )
    print(
        reading(
            "the board as its cards helps the copy (once-cards against once: "
            f"{'-' if board is None else points(*board)})",
            readings["board_helps_copy"],
        )
    )
    print(
        reading(
            f"it helps the heads (value {share(*cards_value)}, leads {share(cards_leads)}, "
            f"the mark {MORE})",
            readings["board_helps_heads"],
        )
    )
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    found = report(args.directory)
    if args.output:
        args.output.write_text(json.dumps(found, indent=2, sort_keys=True), encoding="utf8")


if __name__ == "__main__":
    main()
