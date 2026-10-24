"""Where a network stands in the Blackwing mirror on the curve of the network of
Blackwing alone (decks-01 of experiments/2026-10-14-more-decks.md, its section
on the mirror along the training of Blackwing alone).

    uv run python tools/blackwing_mirror_curve.py analysis/blackwing-mirror-curve/decks-13

The directory is what `experiments/commands/blackwing-mirror-curve.sh` leaves
for a network: for each of its checkpoints and each point of decks-01's PPO,
the mirror's deals with the checkpoint playing and decks-01 at that point in
the other seat (`STEM.plays-POINT.json`), and again with the seats the other
way round (`STEM.held-POINT.json`). At each point, the network's share of
those games, a won game a win (a drawn one is not), its checkpoints' and
seats' games of a deal taken together, with the error of the mean over deals:
over decks-01 there when the share less two standard errors is above 50%,
under it when the share and two standard errors are below, otherwise level.
Where it stands is between the last point it is over and the first it is
under. With `--keeps` and `--little`, the reading of a copy of decks-01
(the copy of the specialist's fourth, fixed on 2026-10-20 before any run):
it keeps its teacher's strength in the mirror when it is level with or over
decks-01 after `--keeps` updates (720), keeps little of it when it is under
decks-01 after `--little` (120), and some otherwise.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

CASE = "blackwing-vs-checkpoint"
# decks-01's games of the mirror an update of its PPO: 260 against itself and
# 20 against the Blackwing pilot.
MIRROR_AN_UPDATE = 280


def outcomes(path: Path) -> list[str]:
    (record,) = json.loads(path.read_text(encoding="utf8"))["records"]
    return record["evaluation_suite"][CASE]["outcomes"]


def shares(directory: Path) -> dict[int, dict[str, Any]]:
    """The network's share of the mirror at each point both seats are played."""
    found: dict[int, list[list[float]]] = {}
    for plays in sorted(directory.glob("*.plays-*.json")):
        stem, point = plays.name.removesuffix(".json").rsplit(".plays-", 1)
        held = directory / f"{stem}.held-{point}.json"
        if not held.exists():
            continue
        won = [outcome == "win" for outcome in outcomes(plays)]
        # decks-01 plays these: the network wins what it loses.
        won_held = [outcome == "loss" for outcome in outcomes(held)]
        if len(won) != len(won_held):
            raise SystemExit(f"{plays.name} and {held.name} play other deals")
        found.setdefault(int(point), []).append(
            [(a + b) / 2 for a, b in zip(won, won_held, strict=True)]
        )
    result = {}
    for point, checkpoints in sorted(found.items()):
        if len({len(deals) for deals in checkpoints}) != 1:
            raise SystemExit(f"The checkpoints at {point} play other deals")
        deals = [sum(told) / len(told) for told in zip(*checkpoints, strict=True)]
        share = sum(deals) / len(deals)
        error = math.sqrt(
            sum((deal - share) ** 2 for deal in deals) / (len(deals) - 1) / len(deals)
        )
        if share - 2 * error > 0.5:
            verdict = "over"
        elif share + 2 * error < 0.5:
            verdict = "under"
        else:
            verdict = "level"
        result[point] = {
            "share": share,
            "error": error,
            "verdict": verdict,
            "checkpoints": len(checkpoints),
            "games": 2 * len(deals) * len(checkpoints),
        }
    return result


def stands(found: dict[int, dict[str, Any]]) -> str:
    """Where on the curve the network plays the mirror, in words."""
    points = sorted(found)
    level = [point for point in points if found[point]["verdict"] == "level"]
    over = [point for point in points if found[point]["verdict"] == "over"]
    under = [point for point in points if found[point]["verdict"] == "under"]
    if level:
        named = ", ".join(f"{point:,}" for point in level)
        return f"level with decks-01 after {named} updates of its PPO"
    if not under:
        return f"over decks-01 at every point measured, to {points[-1]:,} updates"
    if not over:
        return f"under decks-01 at every point measured, from {points[0]:,} updates"
    last_over, first_under = max(over), min(under)
    if last_over < first_under:
        return (
            f"between decks-01 after {last_over:,} and after {first_under:,} updates of its PPO"
            f" ({MIRROR_AN_UPDATE * last_over:,} and {MIRROR_AN_UPDATE * first_under:,}"
            " games of the mirror)"
        )
    return "over and under decks-01 by turns: no single place on the curve"


def kept(found: dict[int, dict[str, Any]], keeps: int, little: int) -> str:
    """How much of decks-01's strength in the mirror a copy of it keeps."""
    for point in (keeps, little):
        if point not in found:
            return f"not measured: no games against decks-01 after {point:,} updates"
    if found[keeps]["verdict"] in ("level", "over"):
        return "it keeps its teacher's strength in the mirror"
    if found[little]["verdict"] == "under":
        return "it keeps little of its teacher's strength in the mirror"
    return "it keeps some of its teacher's strength in the mirror"


def show(found: dict[int, dict[str, Any]], name: str, reading: str | None = None) -> str:
    lines = [
        f"{name} against decks-01 in the Blackwing mirror, by decks-01's updates of PPO:",
        f"  {'updates':>8} {'its mirror games':>17} {'share':>8} {'error':>6}  verdict",
    ]
    for point, told in found.items():
        lines.append(
            f"  {point:>8,} {MIRROR_AN_UPDATE * point:>17,} {told['share']:>8.1%}"
            f" {told['error'] * 100:>6.1f}  {told['verdict']}"
        )
    if found:
        lines.append(f"It stands {stands(found)}.")
    if reading is not None:
        lines.append(f"The reading: {reading}.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--keeps", type=int, help="the point a copy is level with to keep it")
    parser.add_argument("--little", type=int, help="the point a copy under keeps little")
    args = parser.parse_args()
    if (args.keeps is None) != (args.little is None):
        parser.error("--keeps and --little go together")
    found = shares(args.directory)
    if args.output is not None:
        args.output.write_text(json.dumps(found, indent=2) + "\n", encoding="utf8")
    reading = None if args.keeps is None else kept(found, args.keeps, args.little)
    print(show(found, args.directory.name, reading))


if __name__ == "__main__":
    main()
