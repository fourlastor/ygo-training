"""The copy's width and rate at seventy-one decks (experiments/2026-11-02-expanded-pool.md,
its section beside the batches).

    uv run python tools/copy_width_rate_report.py analysis/copy-width-rate \\
        --window 711,1420 --decks "$DECKS" \\
        --reference analysis/seventy-one-champions/taught.metrics.jsonl \\
        --fourteen analysis/fourteen-decks/taught.metrics.jsonl

The directory holds `RUN.metrics.jsonl` of `control`, `wide`, `narrow` and `slow`, as
`experiments/commands/copy-width-rate.sh` leaves them. A row is an update (`version`, from
0), with the share of the teachers' answers the network picked at the decisions it
imitated, by deck (`imitation_model_teacher_agreement_by_deck`), and how many there were
(`imitation_samples_by_deck`). An update's share is its decks' mean, each deck alike, as
`seventy-one-champions`' record counts.

The readings, fixed on 2026-11-12 before anything was built: each of `wide`, `narrow` and
`slow` against `control` over the window, update by update on the same games, by the mean
difference and two standard errors of its means over the window's stretches. It **fits
better** when at least 0.25 points over by two standard errors, **fits worse** when at
least 0.25 under by two, otherwise it is **not a lever at this length**: half of the gap
between the two copies over ten to twenty updates a deck. Agreed on 2026-11-12, before
the results: where any of the three is not a lever, the copies go on to forty updates a
deck and are read again over twenty to forty (updates 1,421 to 2,840), at half the gap
there, 0.38 points (``--mark 0.0038``).

Told beside them: on the fourteen decks, each run beside the fourteen decks' copy at the
same updates a deck; `control` against `seventy-one-champions`' copy update by update;
the seconds an update and the hours. With ``--heads``, each run's last checkpoint head to
head against the teacher of each of the fourteen decks in its mirror, every deal from both
seats, as `experiments/commands/copy-width-rate-heads.sh` leaves them
(``HEADS/DECK/RUN.plays.json`` and ``.held.json``): each run's share of the deals, and each
arm less `control` deal by deal.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from specialist_distillation_report import shares as deal_shares
from two_decks_report import paired
from two_decks_report import told as told_shares

MARK = 0.0025
ERRORS = 2.0
ARMS = ("wide", "narrow", "slow")
RUNS = ("control", *ARMS)
FOURTEEN = [
    "blackwing",
    "cyber",
    "lightsworn",
    "infernity",
    "gishki",
    "burn",
    "crystal",
    "dragunity",
    "gladiator",
    "heroes",
    "monarch",
    "morphtronic",
    "rock-block",
    "spellcaster",
]

# An update's shares by deck, by its number counted from 1.
Shares = dict[int, dict[str, float]]


def shares(path: Path) -> Shares:
    """Each update's share of the teachers' answers by deck, the decks that had decisions."""
    if not path.is_file():
        raise SystemExit(f"Missing {path}")
    found: Shares = {}
    for line in path.read_text(encoding="utf8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        agreed = row.get("imitation_model_teacher_agreement_by_deck") or {}
        samples = row.get("imitation_samples_by_deck") or {}
        found[int(row["version"]) + 1] = {
            deck: float(share) for deck, share in agreed.items() if samples.get(deck, 0) > 0
        }
    return found


def seconds(path: Path) -> list[float]:
    if not path.is_file():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line]
    return [float(row["wall_seconds"]) for row in rows if "wall_seconds" in row]


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def update_share(by_deck: dict[str, float], decks: list[str] | None = None) -> float | None:
    told = [share for deck, share in by_deck.items() if decks is None or deck in decks]
    return mean(told) if told else None


def window_share(found: Shares, first: int, last: int, decks: list[str] | None = None) -> float:
    """The window's share, each deck alike: each deck's mean over the window, their mean."""
    per: dict[str, list[float]] = {}
    for update in range(first, last + 1):
        for deck, share in found.get(update, {}).items():
            if decks is None or deck in decks:
                per.setdefault(deck, []).append(share)
    if not per:
        raise SystemExit(f"No decision of the decks named from update {first} to {last}")
    return mean([mean(values) for values in per.values()])


def against(
    ours: Shares, theirs: Shares, first: int, last: int, stretch: int, decks: list[str] | None
) -> dict[str, Any]:
    """Our share less theirs, update by update on the same games: its mean over the
    window, and its error from the means of the window's stretches of ``stretch``
    updates, the last of them what is left."""
    gaps: list[float] = []
    for update in range(first, last + 1):
        if update not in ours or update not in theirs:
            raise SystemExit(f"Update {update} is missing from a run: the two are read alike")
        one, other = update_share(ours[update], decks), update_share(theirs[update], decks)
        if one is None or other is None:
            continue
        gaps.append(one - other)
    if len(gaps) < 2 * stretch:
        raise SystemExit(f"The window holds {len(gaps)} updates, fewer than two stretches")
    pieces = [mean(gaps[at : at + stretch]) for at in range(0, len(gaps), stretch)]
    centre = mean(pieces)
    error = math.sqrt(sum((piece - centre) ** 2 for piece in pieces) / (len(pieces) - 1)) / (
        math.sqrt(len(pieces))
    )
    return {"gap": mean(gaps), "error": error, "updates": len(gaps), "stretches": len(pieces)}


def reading(gap: float, error: float, mark: float = MARK) -> str:
    if gap - ERRORS * error >= mark:
        return "fits better"
    if gap + ERRORS * error <= -mark:
        return "fits worse"
    return "not a lever at this length"


def heads(folder: Path, decks: list[str] = FOURTEEN) -> dict[str, Any]:
    """Each run head to head against the decks' teachers: its share of every deal of every
    deck, each deal's two games together, and each arm's less `control`'s deal by deal."""
    found: dict[str, list[float]] = {run: [] for run in RUNS}
    for deck in decks:
        for run in RUNS:
            found[run].extend(deal_shares(folder / deck / run))
    each: dict[str, Any] = {run: told_shares(found[run]) for run in RUNS}
    for arm in ARMS:
        each[arm]["less control"] = paired(found[arm], found["control"])
    return each


def report(
    directory: Path,
    first: int,
    last: int,
    stretch: int,
    decks: list[str] | None,
    reference: Path | None = None,
    fourteen: Path | None = None,
    mark: float = MARK,
    against_teachers: Path | None = None,
) -> dict[str, Any]:
    runs = {run: shares(directory / f"{run}.metrics.jsonl") for run in RUNS}
    found: dict[str, Any] = {
        "window": [first, last],
        "mark": mark,
        "control": window_share(runs["control"], first, last, decks),
        "arms": {},
    }
    for arm in ARMS:
        told = against(runs[arm], runs["control"], first, last, stretch, decks)
        told["share"] = window_share(runs[arm], first, last, decks)
        told["reading"] = reading(told["gap"], told["error"], mark)
        found["arms"][arm] = told
    # The fourteen decks, each run beside the fourteen decks' copy at the same updates a
    # deck: a run of N decks is past update U where that copy is past U * 14 / N.
    count = (
        len(decks) if decks else len({deck for told in runs["control"].values() for deck in told})
    )
    beside: dict[str, Any] = {run: window_share(runs[run], first, last, FOURTEEN) for run in RUNS}
    if fourteen is not None:
        beside["fourteen decks' copy"] = window_share(
            shares(fourteen),
            (first - 1) * len(FOURTEEN) // count + 1,
            last * len(FOURTEEN) // count,
            FOURTEEN,
        )
    found["the fourteen decks"] = beside
    if reference is not None:
        theirs = shares(reference)
        found["control against the reference"] = against(
            runs["control"], theirs, 1, max(runs["control"]), stretch, decks
        )
    if against_teachers is not None:
        found["against the teachers"] = heads(against_teachers)
    found["hours"] = {
        run: {
            "seconds an update": mean(told) if told else None,
            "hours": sum(told) / 3600 if told else None,
        }
        for run in RUNS
        for told in [seconds(directory / f"{run}.metrics.jsonl")]
    }
    return found


def points(value: float) -> str:
    return f"{100 * value:+.2f}"


def shown(found: dict[str, Any]) -> str:
    first, last = found["window"]
    lines = [
        (
            f"Over updates {first:,} to {last:,}, each deck alike: control picks the "
            f"teachers' answers at {100 * found['control']:.2f}%"
        )
    ]
    for number, arm in enumerate(ARMS, 1):
        told = found["arms"][arm]
        lines.append(
            f"{number}. {arm} {told['reading']}: {100 * told['share']:.2f}%, "
            f"{points(told['gap'])} ± {100 * told['error']:.2f} points against control "
            f"({told['updates']:,} updates, {told['stretches']} stretches)"
        )
    lines.append("Beside them:")
    beside = found["the fourteen decks"]
    parts = [f"{run} {100 * share:.2f}%" for run, share in beside.items()]
    lines.append(f"  On the fourteen decks: {', '.join(parts)}")
    if "control against the reference" in found:
        told = found["control against the reference"]
        lines.append(
            f"  control less seventy-one-champions' copy, update by update: "
            f"{points(told['gap'])} ± {100 * told['error']:.2f} points over {told['updates']:,}"
        )
    if "against the teachers" in found:
        each = found["against the teachers"]
        lines.append(
            f"  Head to head against the fourteen decks' teachers in their mirrors, "
            f"{each['control']['deals']:,} deals from both seats: "
            + ", ".join(
                f"{run} {100 * each[run]['share']:.1f}% ± {100 * each[run]['error']:.1f}%"
                for run in RUNS
            )
        )
        lines.append(
            "    less control, deal by deal: "
            + ", ".join(
                f"{arm} {points(each[arm]['less control']['gap'])} "
                f"± {100 * each[arm]['less control']['error']:.2f} points"
                for arm in ARMS
            )
        )
    for run, spent in found["hours"].items():
        if spent["hours"] is not None:
            lines.append(
                f"  {run}: {spent['seconds an update']:.1f} s an update, {spent['hours']:.2f} hours"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--window", required=True, help="the first and last updates read")
    parser.add_argument("--stretch", type=int, default=71)
    parser.add_argument("--decks", help="the decks read, comma-separated: every deck if none")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--fourteen", type=Path)
    parser.add_argument("--mark", type=float, default=MARK, help="the reading's mark, a share")
    parser.add_argument("--heads", type=Path, help="the runs head to head against the teachers")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    first, last = (int(value) for value in args.window.split(","))
    decks = None if not args.decks else args.decks.split(",")
    found = report(
        args.directory,
        first,
        last,
        args.stretch,
        decks,
        args.reference,
        args.fourteen,
        args.mark,
        args.heads,
    )
    if args.output is not None:
        args.output.write_text(json.dumps(found, indent=2) + "\n", encoding="utf8")
    print(shown(found))


if __name__ == "__main__":
    sys.exit(main())
