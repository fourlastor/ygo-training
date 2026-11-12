"""The board as its cards from scratch, at fourteen decks
(experiments/2026-11-02-expanded-pool.md, its section beside the batches).

    uv run python tools/fourteen_tokens_report.py analysis/fourteen-tokens \\
        --last 4000,4240,4320 --along copy,1360,2800 \\
        --quarter analysis/fourteen-decks/quarter-quick

The directory is what `experiments/commands/fourteen-tokens.sh` leaves: in each folder
of `heads/` (`last`, `copy` and a point of PPO) a file `POINT.plays.json` for each of its
points, `tokens` in the learner's seat against `pooled`, and `POINT.held.json`, the same
deals the other way round; `pooled-all71/POINT.json` and `tokens-all71/POINT.json`, each
arm against `all71`; `RUN.metrics.jsonl` for `pooled-copy`, `tokens-copy`, `pooled` and
`tokens`. A deal is a game of the suite: its games at a folder's points are read
together, as the screens read theirs.

The reading, fixed on 2026-11-12 before anything was built, with the marks of batch six's
head to head: `tokens` against `pooled` at the last three checkpoints is **stronger** when
it wins at least 51% of the deals by two standard errors, **weaker** when at most 49% by
two, otherwise **not decided**.

Told beside it, without a mark: the head to head of the copies and along PPO; each arm
against `all71` and the two deal by deal; `pooled` against the fourteen decks' `quarter`
arm against `all71` on the same deals (QUARTER, its files at the same points); the
copies' imitation, the seconds an update and the hours.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fourteen_decks_report import ERRORS
from pool_arms_report import outcomes
from two_decks_report import mean, paired, told

STRONGER = 0.51
WEAKER = 0.49
ARMS = ("pooled", "tokens")
# The updates a copy's imitation is told over, its last.
LATELY = 80


def reading(share: float, error_: float) -> str:
    if share - ERRORS * error_ >= STRONGER:
        return "stronger"
    if share + ERRORS * error_ <= WEAKER:
        return "weaker"
    return "not decided"


def averaged(columns: list[list[float]], where: str) -> list[float]:
    """Each deal's mean over the points, the deals in the same order at each."""
    if not columns:
        raise SystemExit(f"Nothing is measured in {where}")
    if len({len(column) for column in columns}) != 1:
        raise SystemExit(f"{where}: the points play other deals")
    return [mean(list(deal)) for deal in zip(*columns, strict=True)]


def heads(folder: Path) -> list[float]:
    """tokens' share of each deal, its game in the learner's seat and the same deal with
    pooled there, which it wins where pooled loses, at each of the folder's points
    together. A draw is a game neither wins."""
    columns = []
    for plays in sorted(folder.glob("*.plays.json")):
        held = plays.with_name(plays.name.replace(".plays.", ".held."))
        won = [float(game == "win") for game in outcomes(plays)]
        lost = [float(game == "loss") for game in outcomes(held)]
        if len(won) != len(lost):
            raise SystemExit(f"{plays.name} and {held.name} play other deals")
        columns.append([(one + other) / 2 for one, other in zip(won, lost, strict=True)])
    return averaged(columns, str(folder))


def against(folder: Path, points: list[int]) -> list[float]:
    """The share of each deal won by the network in the learner's seat, at the points
    together."""
    return averaged(
        [[float(game == "win") for game in outcomes(folder / f"{point}.json")] for point in points],
        str(folder),
    )


def rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line]


def spent(directory: Path, run: str) -> dict[str, float | None]:
    found = rows(directory / f"{run}.metrics.jsonl")
    seconds = [float(row["wall_seconds"]) for row in found if "wall_seconds" in row]
    told_: dict[str, float | None] = {
        "seconds an update": mean(seconds) if seconds else None,
        "hours": sum(seconds) / 3600 if seconds else None,
    }
    if run.endswith("-copy"):
        lately = found[-LATELY:]
        for key in ("imitation_loss", "imitation_model_teacher_agreement"):
            values = [float(row[key]) for row in lately if key in row]
            told_[key] = mean(values) if values else None
    return told_


def report(
    directory: Path, last: list[int], along: list[str], quarter: Path | None = None
) -> dict[str, Any]:
    head = told(heads(directory / "heads" / "last"))
    found: dict[str, Any] = {
        "head to head": head,
        "reading": reading(head["share"], head["error"]),
        "along": {name: told(heads(directory / "heads" / name)) for name in along},
    }
    shares = {arm: against(directory / f"{arm}-all71", last) for arm in ARMS}
    found["against all71"] = {arm: told(shares[arm]) for arm in ARMS}
    found["tokens less pooled against all71"] = paired(shares["tokens"], shares["pooled"])
    if quarter is not None:
        theirs = against(quarter, last)
        found["quarter against all71"] = told(theirs)
        found["pooled less quarter against all71"] = paired(shares["pooled"], theirs)
    found["runs"] = {run: spent(directory, run) for run in ("pooled-copy", "tokens-copy", *ARMS)}
    return found


def percent(found: dict[str, float]) -> str:
    return f"{100 * found['share']:.1f}% ± {100 * found['error']:.1f}%"


def points(found: dict[str, float]) -> str:
    return f"{100 * found['gap']:+.1f} ± {100 * found['error']:.1f} points"


def shown(found: dict[str, Any], last: list[int]) -> str:
    head = found["head to head"]
    at = ", ".join(f"{point:,}" for point in last)
    lines = [
        (
            f"1. The board as its cards from scratch {found['reading']}: tokens wins "
            f"{percent(head)} head to head against pooled at {at} updates of PPO "
            f"({head['deals']:,} deals)"
        ),
        "Beside it:",
    ]
    for name, told_ in found["along"].items():
        where = "at the copies" if name == "copy" else f"after {int(name):,} updates of PPO"
        lines.append(f"  Head to head {where}: {percent(told_)}")
    for arm in ARMS:
        lines.append(f"  {arm} against all71: {percent(found['against all71'][arm])}")
    lines.append(
        f"  tokens less pooled against all71, deal by deal: "
        f"{points(found['tokens less pooled against all71'])}"
    )
    if "quarter against all71" in found:
        lines.append(
            f"  The fourteen decks' quarter arm against all71: "
            f"{percent(found['quarter against all71'])}; pooled less it, deal by deal: "
            f"{points(found['pooled less quarter against all71'])}"
        )
    for run, spent_ in found["runs"].items():
        parts = []
        if spent_.get("imitation_loss") is not None:
            parts.append(f"imitation loss {spent_['imitation_loss']:.3f}")
        if spent_.get("imitation_model_teacher_agreement") is not None:
            parts.append(f"agreement {spent_['imitation_model_teacher_agreement']:.1%}")
        if spent_["seconds an update"] is not None:
            parts.append(f"{spent_['seconds an update']:.1f} s an update")
        if spent_["hours"] is not None:
            parts.append(f"{spent_['hours']:.2f} hours")
        if parts:
            lines.append(f"  {run}: {', '.join(parts)}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--last", required=True, help="the last three points of PPO")
    parser.add_argument("--along", default="", help="copy and points of PPO before them")
    parser.add_argument("--quarter", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    last = [int(point) for point in args.last.split(",")]
    along = [name for name in args.along.split(",") if name]
    found = report(args.directory, last, along, args.quarter)
    if args.output is not None:
        args.output.write_text(json.dumps(found, indent=2) + "\n", encoding="utf8")
    print(shown(found, last))


if __name__ == "__main__":
    main()
