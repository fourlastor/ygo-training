"""The seventy-one's copy twice as wide (experiments/2026-11-02-expanded-pool.md, its
section beside the batches).

    uv run python tools/seventy_one_wide_report.py analysis/seventy-one-wide \\
        --reference analysis/seventy-one-champions --decks "$DECKS" --half 5680 --copy 11360

The directory is what `experiments/commands/seventy-one-wide.sh` leaves: `heads/` with
`copy.plays.json`, `wide` in the learner's seat against `seventy-one-champions`' copy
(`taught`) on the pairings of that record's checks, and `copy.held.json`, the same deals
the other way round; `checks/wide/` and `checks/taught/`, each copy as those checks
measured `taught`, a pair `DECK.plays.json` and `DECK.held.json` for each deck, the copy
head to head against that deck's teacher in its mirror, and `pairings.json`, it against
`all71` on the checks' pairings; `teachers/`, a pair for each deck, `wide` head to head
against its teacher on every deal of its mirror's suite; `wide.metrics.jsonl`, the
copy's metrics. The reference is `seventy-one-champions`' measurements: its copy's
metrics (`taught.metrics.jsonl`) and its copy head to head against each teacher on the
same deals (`heads-taught/`).

The reading, fixed on 2026-11-14 before anything was built, with the marks of batch six's
head to head and of the fourteen decks made from scratch: `wide` against `taught` head to
head is **stronger** when it wins at least 51% of the deals by two standard errors,
**weaker** when at most 49% by two, otherwise **not decided**. PPO goes on from `wide`
only where it is stronger.

Told beside it, without a mark: the checks against `taught`'s deal by deal, the mirrors
and the pairings against `all71`; the teachers head to head, deal by deal against
`taught`'s; the share of the teachers' answers at the decisions imitated, each deck alike,
update by update against `taught`'s on the same games, over the first half, where only
the width differs, over the second, and over the last twenty updates a deck; the seconds
an update and the hours.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from copy_width_rate_report import against as fitted
from copy_width_rate_report import seconds, window_share
from copy_width_rate_report import shares as answers
from fourteen_tokens_report import heads, reading
from seventy_one_checks import check, won
from specialist_distillation_report import shares
from two_decks_report import mean, paired, told

# The updates of one a deck, a stretch of the fit's errors.
STRETCH = 71
# The last updates a deck the fit is told over, at the copy's end.
LATELY = 20


def checks(directory: Path, decks: list[str]) -> dict[str, Any]:
    """Each copy as the checks measured `taught`, and `wide` less `taught` deal by deal:
    the mirrors head to head against the teachers, each deck alike, and the pairings
    against `all71`."""
    ours, theirs = directory / "checks" / "wide", directory / "checks" / "taught"
    found = check(ours, theirs, decks)
    found["mirrors"]["taught"] = mean([mean(shares(theirs / deck)) for deck in decks])
    found["pairings"]["taught"] = mean(won(theirs / "pairings.json"))
    for told_ in found.values():
        told_.pop("under", None)
    return found


def teachers(directory: Path, reference: Path, decks: list[str]) -> dict[str, Any]:
    """`wide` and `taught` head to head against each teacher on every deal of its
    mirror's suite, and `wide` less `taught` deal by deal."""
    ours: list[float] = []
    theirs: list[float] = []
    for deck in decks:
        one = shares(directory / "teachers" / deck)
        other = shares(reference / "heads-taught" / deck)
        if len(one) != len(other):
            raise SystemExit(f"{deck}: the two copies were played on other deals")
        ours.extend(one)
        theirs.extend(other)
    return {"wide": told(ours), "taught": told(theirs), "less taught": paired(ours, theirs)}


def fit(
    directory: Path,
    reference: Path,
    decks: list[str],
    half: int,
    copy: int,
    stretch: int = STRETCH,
    lately: int = LATELY,
) -> dict[str, Any]:
    """The share of the teachers' answers, each deck alike, update by update against
    `taught`'s on the same games, over each window, its error from the means of stretches
    of ``stretch`` updates."""
    ours = answers(directory / "wide.metrics.jsonl")
    theirs = answers(reference / "taught.metrics.jsonl")
    windows = {
        "first half": (1, half),
        "second half": (half + 1, copy),
        "lately": (copy - lately * len(decks) + 1, copy),
    }
    found: dict[str, Any] = {}
    for name, (first, last) in windows.items():
        told_ = fitted(ours, theirs, first, last, stretch, decks)
        told_["window"] = [first, last]
        told_["wide"] = window_share(ours, first, last, decks)
        told_["taught"] = window_share(theirs, first, last, decks)
        found[name] = told_
    return found


def spent(path: Path) -> dict[str, float | None]:
    told_ = seconds(path)
    return {
        "seconds an update": mean(told_) if told_ else None,
        "hours": sum(told_) / 3600 if told_ else None,
    }


def report(
    directory: Path,
    reference: Path,
    decks: list[str],
    half: int,
    copy: int,
    stretch: int = STRETCH,
    lately: int = LATELY,
) -> dict[str, Any]:
    head = told(heads(directory / "heads"))
    return {
        "head to head": head,
        "reading": reading(head["share"], head["error"]),
        "checks": checks(directory, decks),
        "teachers": teachers(directory, reference, decks),
        "fit": fit(directory, reference, decks, half, copy, stretch, lately),
        "lately": lately,
        "runs": {
            "wide": spent(directory / "wide.metrics.jsonl"),
            "taught": spent(reference / "taught.metrics.jsonl"),
        },
    }


def percent(share: float, error: float | None = None) -> str:
    if error is None:
        return f"{100 * share:.1f}%"
    return f"{100 * share:.1f}% ± {100 * error:.1f}%"


def points(found: dict[str, float], places: int = 1) -> str:
    return f"{100 * found['gap']:+.{places}f} ± {100 * found['error']:.{places}f} points"


def shown(found: dict[str, Any]) -> str:
    head = found["head to head"]
    lines = [
        (
            f"1. The wide copy {found['reading']}: wide wins "
            f"{percent(head['share'], head['error'])} head to head against taught "
            f"({head['deals']:,} deals)"
        ),
        "Beside it:",
    ]
    mirrors, pairings = found["checks"]["mirrors"], found["checks"]["pairings"]
    lines.append(
        f"  The checks, against each teacher in its mirror: wide {percent(mirrors['share'])}, "
        f"taught {percent(mirrors['taught'])}, {points(mirrors)} deal by deal"
    )
    lines.append(
        f"  The checks, against all71 on their pairings: wide {percent(pairings['share'])}, "
        f"taught {percent(pairings['taught'])}, {points(pairings)} deal by deal"
    )
    each = found["teachers"]
    lines.append(
        f"  Against each teacher in its mirror, {each['wide']['deals']:,} deals: wide "
        f"{percent(each['wide']['share'], each['wide']['error'])}, taught "
        f"{percent(each['taught']['share'], each['taught']['error'])}; "
        f"wide less taught {points(each['less taught'])} deal by deal"
    )
    told_of = {"first half": "the first half", "second half": "the second half"}
    for name, told_ in found["fit"].items():
        first, last = told_["window"]
        where = told_of.get(name, f"the last {found['lately']} updates a deck")
        lines.append(
            f"  The teachers' answers over {where} (updates {first:,} to {last:,}): "
            f"wide {100 * told_['wide']:.2f}%, taught {100 * told_['taught']:.2f}%, "
            f"{points(told_, 2)} update by update"
        )
    for run, spent_ in found["runs"].items():
        if spent_["hours"] is not None:
            lines.append(
                f"  {run}: {spent_['seconds an update']:.1f} s an update, "
                f"{spent_['hours']:.2f} hours"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--decks", required=True, help="the seventy-one, comma-separated")
    parser.add_argument("--half", type=int, required=True, help="the update the rate falls after")
    parser.add_argument("--copy", type=int, required=True, help="the copy's updates")
    parser.add_argument(
        "--stretch", type=int, default=STRETCH, help="updates in a stretch of the fit's errors"
    )
    parser.add_argument(
        "--lately", type=int, default=LATELY, help="the last updates a deck the fit is told over"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    found = report(
        args.directory,
        args.reference,
        args.decks.split(","),
        args.half,
        args.copy,
        args.stretch,
        args.lately,
    )
    if args.output is not None:
        args.output.write_text(json.dumps(found, indent=2) + "\n", encoding="utf8")
    print(shown(found))


if __name__ == "__main__":
    main()
