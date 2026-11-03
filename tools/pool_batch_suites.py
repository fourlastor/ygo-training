"""The suites of one batch of the expanded pool (experiments/2026-11-02-expanded-pool.md).

    uv run python tools/pool_batch_suites.py one --new game-0013:game-0062 \\
        --held-out game-0063:game-0074

The batch's old decks are the seventy-one decks of the champions' list and every
game deck numbered before its first new one. Into `eval_suites/pool-batch-NAME-*.json`:
each new deck against the frozen network at each old deck (`pairings`), every ordered
pairing of the old decks head to head against it (`old-pairings`), the next decks,
trained on nothing, as the new ones (`held-out`), and the new decks against the
thirteen decks' pilots (`against-pilots`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIST = ROOT / "experiments/commands/champion-recordings.list"
THIRTEEN = (
    "blackwing,lightsworn,gishki,burn,crystal,dragunity,gladiator,heroes,infernity,"
    "monarch,morphtronic,rock-block,spellcaster"
)
# Games of each pairing, from both seats.
GAMES = {"pairings": 32, "old-pairings": 8, "held-out": 32, "against-pilots": 64}


def numbered(span: str) -> list[str]:
    """`game-0013:game-0062`: the game decks from the first to the last."""
    first, last = (int(part.rsplit("-", 1)[1]) for part in span.split(":"))
    return [f"game-{number:04d}" for number in range(first, last + 1)]


def seventy_one(path: Path = LIST) -> list[str]:
    lines = path.read_text(encoding="utf8").splitlines()
    return [line.split()[0] for line in lines if line.strip() and not line.startswith("#")]


def suite(name: str, learners: list[str], opponents: list[str] | None, games: int) -> dict:
    """Each learner against the frozen network at each of `opponents`, or against
    the pilots where none are named."""
    cases = []
    for learner in learners:
        if opponents is None:
            case = {"name": f"{learner}-vs-pilots", "learner": learner, "opponent": THIRTEEN}
        else:
            case = {
                "name": f"{learner}-vs-each-checkpoint",
                "learner": learner,
                "opponent": ",".join(opponents),
                "opponent_policy": "checkpoint",
            }
        cases.append({**case, "games": games})
    return {"name": name, "primary": cases[0]["name"], "cases": cases}


def suites(batch: str, new: list[str], held_out: list[str]) -> dict[str, dict]:
    first = int(new[0].rsplit("-", 1)[1])
    old = (
        seventy_one() + numbered(f"game-0001:game-{first - 1:04d}") if first > 1 else seventy_one()
    )
    return {
        f"pool-batch-{batch}-pairings-checkpoint": suite(
            f"pool-batch-{batch}-pairings-checkpoint", new, old, GAMES["pairings"]
        ),
        f"pool-batch-{batch}-old-pairings-checkpoint": suite(
            f"pool-batch-{batch}-old-pairings-checkpoint", old, old, GAMES["old-pairings"]
        ),
        f"pool-batch-{batch}-held-out-checkpoint": suite(
            f"pool-batch-{batch}-held-out-checkpoint", held_out, old, GAMES["held-out"]
        ),
        f"pool-batch-{batch}-against-pilots": suite(
            f"pool-batch-{batch}-against-pilots", new, None, GAMES["against-pilots"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("batch")
    parser.add_argument("--new", required=True)
    parser.add_argument("--held-out", required=True)
    parser.add_argument("--into", type=Path, default=ROOT / "eval_suites")
    args = parser.parse_args()
    for name, told in suites(args.batch, numbered(args.new), numbered(args.held_out)).items():
        (args.into / f"{name}.json").write_text(json.dumps(told, indent=2) + "\n", encoding="utf8")
        print(name, len(told["cases"]), "cases")


if __name__ == "__main__":
    main()
