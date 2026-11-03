"""Two runs of a batch of the expanded pool read against each other at one point
(experiments/2026-11-02-expanded-pool.md, the card identities): the batch with its
identity tables at ten times the learning rate against the batch as it ran.

    uv run python tools/pool_batch_compare.py analysis/expanded-pool/batch-one-fast \\
        analysis/expanded-pool/batch-one-4020 --point 4020

Each directory is what `experiments/commands/pool-batch.sh` leaves. Both start from the
same frozen network on the same games at 0, so that the difference of the two gains is
the difference of the two networks' shares, deal by deal.

The readings, fixed on 2026-11-03 before anything was built:

1. It learns faster: the first run's gain against the frozen network less the second's:
   faster when at least +1.0 point by two standard errors, not faster when at most +0.3
   points by two standard errors, otherwise not decided.
2. The old decks are no worse: the first run's head-to-head share on the old decks'
   pairings less the second's: no worse when at least -1.0 point by two standard
   errors, worse when at most -1.0 point by two standard errors, otherwise not decided.

Told beside them, without a mark: each run's gain, the mirrors against their champions,
the next decks trained on nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from first_batch_report import cases, games
from fourteen_decks_report import ERRORS
from specialist_distillation_report import shares
from two_decks_report import mean, paired, told

# Reading 1.
FASTER = 0.01
NOT_FASTER = 0.003
# Reading 2.
NO_WORSE = -0.01


def faster(gap: float, error: float) -> str:
    if gap - ERRORS * error >= FASTER:
        return "faster"
    if gap + ERRORS * error <= NOT_FASTER:
        return "not faster"
    return "not decided"


def no_worse(gap: float, error: float) -> str:
    if gap - ERRORS * error >= NO_WORSE:
        return "no worse"
    if gap + ERRORS * error <= NO_WORSE:
        return "worse"
    return "not decided"


def mirrors(directory: Path, point: int) -> float | None:
    found = sorted((directory / str(point) / "mirrors").glob("*.held.json"))
    if not found:
        return None
    return mean([told(shares(path.with_name(path.name.split(".")[0])))["share"] for path in found])


def compared(first: Path, second: Path, point: int) -> dict[str, Any]:
    start = games(cases(first / "0" / "pairings.json"))
    if start != games(cases(second / "0" / "pairings.json")):
        raise SystemExit("the two runs were not read against the same games at 0")
    ours = games(cases(first / str(point) / "pairings.json"))
    theirs = games(cases(second / str(point) / "pairings.json"))
    gain = paired(ours, theirs)
    old = paired(
        games(cases(first / str(point) / "old.json")),
        games(cases(second / str(point) / "old.json")),
    )
    found: dict[str, Any] = {
        "gain": gain,
        "faster": faster(gain["gap"], gain["error"]),
        "old": old,
        "old decks": no_worse(old["gap"], old["error"]),
        "gains": {
            "first": paired(ours, start)["gap"],
            "second": paired(theirs, start)["gap"],
        },
        "mirrors": {"first": mirrors(first, point), "second": mirrors(second, point)},
    }
    held = [directory / str(point) / "held-out.json" for directory in (first, second)]
    if all(path.is_file() for path in held):
        found["held out"] = paired(games(cases(held[0])), games(cases(held[1])))
    return found


def shown(found: dict[str, Any], point: int) -> str:
    gain, old = found["gain"], found["old"]
    lines = [
        (
            f"1. Learning: {found['faster']}, {100 * gain['gap']:+.1f} ± {100 * gain['error']:.1f} "
            f"points at {point:,} updates ({100 * found['gains']['first']:+.1f} against "
            f"{100 * found['gains']['second']:+.1f} over the frozen network)"
        ),
        (
            f"2. The old decks: {found['old decks']}, {100 * old['gap']:+.1f} ± "
            f"{100 * old['error']:.1f} points head to head"
        ),
        "Beside them:",
    ]
    first, second = found["mirrors"]["first"], found["mirrors"]["second"]
    if first is not None and second is not None:
        lines.append(
            f"  The mirrors against their champions: {100 * first:.1f}% against {100 * second:.1f}%"
        )
    if "held out" in found:
        held = found["held out"]
        lines.append(
            f"  The next decks, trained on nothing: {100 * held['gap']:+.1f} ± {100 * held['error']:.1f} points"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--point", type=int, required=True)
    args = parser.parse_args()
    found = compared(args.first, args.second, args.point)
    (args.first / "compare.json").write_text(json.dumps(found, indent=1), encoding="utf8")
    print(shown(found, args.point))


if __name__ == "__main__":
    main()
