"""The readings of the experiment on the model without the deck lists
(`experiments/2026-10-15-without-deck-lists.md`), on what the measuring part
of `experiments/commands/without-deck-lists.sh` leaves:

    uv run python tools/deck_lists_report.py analysis/without-deck-lists

Each network that reads less is set beside its control, game by game on the
same deals. The gap is it less the control, with its standard error: the
list can go (EQUIVALENT) when the gap less two standard errors is above
-MARK, and it stays (WORSE) when the network that reads less wins at least
MARK and two standard errors fewer; otherwise the reading is not decided.

1. The other player's list, on the decks of training: `blackwing-own-list`
   against `decks-01-copy` on the held-out deals.
2. The same two outside training: on the eighteen decks outside training.
3. Both lists, in a network of three: `three-no-lists` against
   `decks-03-copy` over the screens of its three decks together.
4. The seat's own earlier answers, on the decks of training:
   `blackwing-no-history` against `decks-01-copy` on the held-out deals.
5. The same two outside training.

Told beside them without a mark: each gap where it is over, and the network
of three on the held-out deals and outside training. With `--runs`, the
runs' own figures as the measuring part copies them (NAME.run.json,
NAME.metrics.jsonl): whether each copy began as the same network as its
control and played the same games in every update, which is what makes the
two differ only in what their models read, and how often each answers as
its pilots at its last update.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

MARK = 0.02
ERRORS = 2.0
OWN_LIST = ("blackwing-own-list", "decks-01-copy")
NO_LISTS = ("three-no-lists", "decks-03-copy")
NO_HISTORY = ("blackwing-no-history", "decks-01-copy")
THREE = ("blackwing", "lightsworn", "gishki")


def heldout(path: Path) -> tuple[list[float], Any]:
    """A checkpoint's games on the held-out deals, and what tells the deals."""
    found = json.loads(path.read_text(encoding="utf8"))
    (point,) = found["checkpoints"]
    return [float(outcome == "win") for outcome in point["deals"]["outcomes"]], found.get(
        "fingerprint"
    )


def suite(path: Path) -> tuple[list[float], Any]:
    """A checkpoint's games on a suite, its cases in the order of their
    names, and what tells the games."""
    found = json.loads(path.read_text(encoding="utf8"))
    (record,) = found["records"]
    cases = record["evaluation_suite"]
    games = [
        float(outcome == "win") for case in sorted(cases) for outcome in cases[case]["outcomes"]
    ]
    return games, (
        found.get("seed"),
        sorted(cases),
        [cases[case]["games"] for case in sorted(cases)],
    )


def paired(ours: list[float], theirs: list[float]) -> tuple[float, float]:
    """The mean of ours less theirs, game by game, and its standard error."""
    if len(ours) != len(theirs) or len(ours) < 2:
        raise SystemExit("the two are not measured on the same games")
    gaps = [a - b for a, b in zip(ours, theirs, strict=True)]
    mean = sum(gaps) / len(gaps)
    spread = sum((gap - mean) ** 2 for gap in gaps) / (len(gaps) - 1)
    return mean, (spread / len(gaps)) ** 0.5


def verdict(gap: float, error: float) -> str:
    if gap - ERRORS * error > -MARK:
        return "equivalent"
    if gap <= -MARK and gap <= -ERRORS * error:
        return "worse"
    return "not decided"


def compared(
    out: Path, kind: str, ours: str, theirs: str, files: list[str]
) -> dict[str, Any] | None:
    """Ours beside theirs over the files named, their games laid end to end;
    none where one of them is not measured."""
    read = heldout if kind == "heldout" else suite
    games: dict[str, list[float]] = {ours: [], theirs: []}
    for file in files:
        paths = {name: out / kind / f"{name}{file}.json" for name in games}
        if not all(path.is_file() for path in paths.values()):
            return None
        mine, told = read(paths[ours])
        other, also = read(paths[theirs])
        if told != also:
            raise SystemExit(
                f"{paths[ours]} and {paths[theirs]} are not measured on the same games"
            )
        games[ours] += mine
        games[theirs] += other
    gap, error = paired(games[ours], games[theirs])
    return {
        "games": len(games[ours]),
        ours: sum(games[ours]) / len(games[ours]),
        theirs: sum(games[theirs]) / len(games[theirs]),
        "gap": gap,
        "error": error,
        "verdict": verdict(gap, error),
    }


def alike(runs: Path) -> dict[str, Any]:
    """Each copy beside its control by their runs' own figures: the network
    they began as, the games of each update, and how often each answers as
    the pilots it imitates at its last update."""
    found: dict[str, Any] = {}
    for ours, theirs in (OWN_LIST, NO_LISTS, NO_HISTORY):
        files = {
            name: (runs / f"{name}.run.json", runs / f"{name}.metrics.jsonl")
            for name in (ours, theirs)
        }
        if not all(run.is_file() and metrics.is_file() for run, metrics in files.values()):
            found[ours] = None
            continue
        start = {
            name: json.loads(run.read_text(encoding="utf8"))["initial_model_fingerprint"]
            for name, (run, _) in files.items()
        }
        rows = {
            name: [
                json.loads(line)
                for line in metrics.read_text(encoding="utf8").splitlines()
                if line.strip()
            ]
            for name, (_, metrics) in files.items()
        }
        games = {
            name: [row.get("rollout_decision_fingerprint") for row in rows[name]] for name in rows
        }
        found[ours] = {
            "control": theirs,
            "same_start": start[ours] == start[theirs],
            "updates": len(games[ours]),
            "other_games": sum(a != b for a, b in zip(games[ours], games[theirs], strict=False))
            + abs(len(games[ours]) - len(games[theirs])),
            "agreement": {
                name: rows[name][-1].get("imitation_model_teacher_agreement") for name in rows
            },
        }
    return found


def report(out: Path, runs: Path | None = None) -> dict[str, Any]:
    screens = [f".{deck}" for deck in THREE]
    readings = {
        "1. the other player's list, on the decks of training": compared(
            out, "heldout", *OWN_LIST, [""]
        ),
        "2. the other player's list, outside training": compared(out, "unseen", *OWN_LIST, [""]),
        "3. both lists, in a network of three, on its screens": compared(
            out, "screens", *NO_LISTS, screens
        ),
        "4. its own earlier answers, on the decks of training": compared(
            out, "heldout", *NO_HISTORY, [""]
        ),
        "5. its own earlier answers, outside training": compared(out, "unseen", *NO_HISTORY, [""]),
    }
    beside = {
        "both lists, in a network of three, on the held-out deals": compared(
            out, "heldout", *NO_LISTS, [""]
        ),
        "both lists, in a network of three, outside training": compared(
            out, "unseen", *NO_LISTS, [""]
        ),
    }
    print("Each network that reads less, less its control, game by game on the same deals.")
    print(
        f"Equivalent when the gap less {ERRORS:g} standard errors is above -{100 * MARK:g} points;"
    )
    print(f"worse at {100 * MARK:g} points and {ERRORS:g} standard errors fewer.")
    for title, found in [*readings.items(), *beside.items()]:
        if found is None:
            print(f"{title}: not measured")
            continue
        names = [name for name in found if name not in ("games", "gap", "error", "verdict")]
        rates = ", ".join(f"{name} {100 * found[name]:.1f}%" for name in names)
        told = found["verdict"] if title[0].isdigit() else "told without a mark"
        print(
            f"{title}: {100 * found['gap']:+.1f} ± {100 * found['error']:.1f} over {found['games']:,} games "
            f"({rates}): {told}"
        )
    checked = alike(runs) if runs is not None else {}
    for ours, found in checked.items():
        if found is None:
            print(f"{ours}: its run or its control's is not there to compare")
            continue
        theirs = found["control"]
        agreement = found["agreement"]
        print(
            f"{ours} began as {theirs} did: {'yes' if found['same_start'] else 'NO'}; "
            f"updates of other games: {found['other_games']} of {found['updates']}; "
            f"answers as its pilots at its last update {100 * agreement[ours]:.1f}% "
            f"({theirs} {100 * agreement[theirs]:.1f}%)"
        )
    return {"readings": readings, "beside": beside, "runs": checked}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out", type=Path, help="the folder the measuring part writes")
    parser.add_argument(
        "--runs", type=Path, help="the runs' own figures, NAME.run.json and NAME.metrics.jsonl"
    )
    parser.add_argument("--output", type=Path, help="the readings as JSON")
    args = parser.parse_args()
    found = report(args.out, args.runs)
    if args.output is not None:
        args.output.write_text(json.dumps(found, indent=2) + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
