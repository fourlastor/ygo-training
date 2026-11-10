"""Batch six's two arms side by side (experiments/2026-11-02-expanded-pool.md).

    uv run python tools/pool_arms_report.py analysis/expanded-pool/batch-six-pooled \\
        analysis/expanded-pool/batch-six-switched analysis/expanded-pool/batch-six-arms \\
        --points 3840,7680

POOLED and SWITCHED are what `experiments/commands/pool-batch-six.sh` leaves for each arm,
ARMS what `experiments/commands/pool-batch-six-arms.sh` leaves: at each point
`arms.plays.json`, `switched` in the learner's seat against `pooled`, and
`arms.held.json`, the same deals the other way round.

Reading 4, fixed on 2026-11-10 before anything was built: `switched` head to head against
`pooled` at the last point, a deal's two games together: the switch **pays** when it wins
at least 51% of the deals by two standard errors, **harms** when at most 49% by two,
otherwise it is **not shown**.

Told beside it, without a mark: the head to head at the points before; the arms'
differences deal by deal on the measures they share at the last point, `switched` less
`pooled` (the new decks against the start, the old decks, the next decks, the pilots, the
mirrors against their champions); and whether `switched`'s start played the pilots' games
at 0 as `pooled`'s did, one by one, as a network given the board as its cards with
`board_fusion` at zero must.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mirror_pairs
from first_batch_report import cases, games
from fourteen_decks_report import ERRORS
from two_decks_report import paired, told

PAYS = 0.51
HARMS = 0.49
# The measures both arms play on the same deals, by their files at a point.
SHARED = (
    ("pairings", "the new decks against the start"),
    ("old", "the old decks against the start"),
    ("held-out", "the next decks against the start"),
    ("pilots", "the new decks against the pilots"),
)


def outcomes(path: Path) -> list[str]:
    """How each game of a file ended for the network in the learner's seat, its cases
    in the order of their names."""
    if not path.is_file():
        raise SystemExit(f"Missing {path}")
    (record,) = json.loads(path.read_text(encoding="utf8"))["records"]
    suite = record["evaluation_suite"]
    return [game for name in sorted(suite) for game in suite[name]["outcomes"]]


def deals(folder: Path) -> list[float]:
    """switched's share of each deal: its game in the learner's seat, and the same deal
    with pooled there, which it wins where pooled loses."""
    won = [float(game == "win") for game in outcomes(folder / "arms.plays.json")]
    lost = [float(game == "loss") for game in outcomes(folder / "arms.held.json")]
    if len(won) != len(lost):
        raise SystemExit(f"{folder}: the two ways round play other deals")
    return [(one + other) / 2 for one, other in zip(won, lost, strict=True)]


def reading(share: float, error_: float) -> str:
    if share - ERRORS * error_ >= PAYS:
        return "pays"
    if share + ERRORS * error_ <= HARMS:
        return "harms"
    return "not shown"


def same_games(pooled: Path, switched: Path) -> dict[str, int]:
    """How many of the games of one file the other played to the same end, in order."""
    ours = json.loads(pooled.read_text(encoding="utf8"))["records"][0]["evaluation_suite"]
    theirs = json.loads(switched.read_text(encoding="utf8"))["records"][0]["evaluation_suite"]
    if sorted(ours) != sorted(theirs):
        raise SystemExit(f"{pooled} and {switched} play other cases")
    same = total = 0
    for name in ours:
        one, other = ours[name]["outcomes"], theirs[name]["outcomes"]
        if len(one) != len(other):
            raise SystemExit(f"{name}: {len(one)} games against {len(other)}")
        same += sum(a == b for a, b in zip(one, other, strict=True))
        total += len(one)
    return {"same": same, "games": total}


def report(pooled: Path, switched: Path, arms: Path, points: list[int]) -> dict[str, Any]:
    last = points[-1]
    head = {point: told(deals(arms / str(point))) for point in points}
    found: dict[str, Any] = {
        "head to head": head,
        "switch": reading(head[last]["share"], head[last]["error"]),
        "differences": {},
    }
    for part, _ in SHARED:
        ours, theirs = pooled / str(last) / f"{part}.json", switched / str(last) / f"{part}.json"
        if ours.is_file() and theirs.is_file():
            found["differences"][part] = paired(games(cases(theirs)), games(cases(ours)))
    ours, theirs = pooled / str(last) / "mirrors", switched / str(last) / "mirrors"
    if ours.is_dir() and theirs.is_dir():
        moved = mirror_pairs.against(mirror_pairs.shares(ours), mirror_pairs.shares(theirs))
        found["differences"]["mirrors"] = {
            "gap": moved["difference"],
            "error": moved["error"],
            "lower": moved["lower"],
            "decks": moved["decks"],
        }
    ours, theirs = pooled / "0" / "pilots.json", switched / "0" / "pilots.json"
    if ours.is_file() and theirs.is_file():
        found["start"] = same_games(ours, theirs)
    return found


def shown(found: dict[str, Any], points: list[int]) -> str:
    last = points[-1]
    head = found["head to head"][last]
    lines = [
        (
            f"4. The switch {found['switch']}: switched wins {100 * head['share']:.1f}% ± "
            f"{100 * head['error']:.1f}% head to head against pooled at {last:,} updates "
            f"({head['deals']:,} deals)"
        ),
        "Beside it:",
    ]
    for point in points[:-1]:
        told_ = found["head to head"][point]
        lines.append(
            f"  At {point:,} updates: {100 * told_['share']:.1f}% ± {100 * told_['error']:.1f}%"
        )
    labels = dict(SHARED)
    labels["mirrors"] = "the mirrors against their champions, a draw half a win"
    for part, gap in found["differences"].items():
        lines.append(
            f"  switched less pooled, deal by deal, {labels[part]}: "
            f"{100 * gap['gap']:+.1f} ± {100 * gap['error']:.1f} points"
            + (f", lower on {gap['lower']} of {gap['decks']}" if "lower" in gap else "")
        )
    if "start" in found:
        start = found["start"]
        lines.append(
            f"  switched's start against the pilots at 0: {start['same']:,} of "
            f"{start['games']:,} games ended as pooled's did"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pooled", type=Path)
    parser.add_argument("switched", type=Path)
    parser.add_argument("arms", type=Path)
    parser.add_argument("--points", required=True)
    args = parser.parse_args()
    points = [int(point) for point in args.points.split(",")]
    found = report(args.pooled, args.switched, args.arms, points)
    (args.arms / "report.json").write_text(json.dumps(found, indent=1), encoding="utf8")
    print(shown(found, points))


if __name__ == "__main__":
    main()
