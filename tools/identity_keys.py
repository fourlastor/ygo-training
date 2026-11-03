"""What a card the network never trained on costs at play, by its random identity
(experiments/2026-11-02-expanded-pool.md, the card identities beside the batches).

    uv run python tools/identity_keys.py make runs/expanded-pool/first-batch-mean/4320.pt \\
        --keys blank --output runs/identity-keys/blank.pt
    uv run python tools/identity_keys.py read analysis/identity-keys

A card's identity is a row of each of two tables, indexed by the low and the high 16 bits
of its code. A row no update moved is still its untrained value, which every run of the
project draws at seed 2012: a random key the network learnt nothing about. `make`
rewrites those rows only: `blank` to zero, `mean` to the mean of the rows training moved,
`redraw` afresh from the untrained distribution (`--seed`). Row 0 of each table, the code
of a card the seat cannot see, is never touched.

`read` takes the games `experiments/commands/identity-keys.sh` leaves: in `as-is/` the
network against itself, in `plays/ARM/` the network with ARM's keys at the decks it never
trained on against itself as it is, in `faces/ARM/` the network as it is at those decks
against ARM. Each arm is read against the mean of the random keys (`as-is` and the
`redraw` arms), the deals paired.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TABLES = ("card_identity.low.weight", "card_identity.high.weight")
SEED = 2012
# Reading 1 and 2: blank keys against the random ones.
HELPS = 0.01
ERRORS = 2.0


def untrained() -> dict[str, torch.Tensor]:
    """The identity tables as every run of the project begins them."""
    from ygo_training.model import PolicyValueNet

    torch.manual_seed(SEED)
    return {
        name: value.detach().clone()
        for name, value in PolicyValueNet().named_parameters()
        if name in TABLES
    }


def rewritten(
    state: dict[str, torch.Tensor],
    keys: str,
    seed: int = 0,
    start: dict[str, torch.Tensor] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    """The state with every identity row no update moved rewritten, and how many."""
    start = untrained() if start is None else start
    generator = torch.Generator().manual_seed(seed)
    new, counts = dict(state), {}
    for name in TABLES:
        table = state[name]
        still = (table == start[name]).all(dim=1)
        still[0] = False
        if still.float().mean() < 0.5:
            raise SystemExit(f"{name}: most rows moved; this is not the network seed {SEED} began")
        trained = ~still
        trained[0] = False
        made = table.clone()
        if keys == "blank":
            made[still] = 0.0
        elif keys == "mean":
            made[still] = table[trained].mean(dim=0)
        elif keys == "redraw":
            made[still] = torch.randn(
                (int(still.sum()), table.shape[1]), generator=generator, dtype=table.dtype
            )
        else:
            raise SystemExit(f"keys are blank, mean or redraw, not {keys}")
        new[name] = made
        counts[name] = int(still.sum())
    return new, counts


def make(network: Path, keys: str, seed: int, output: Path) -> None:
    from ygo_training import checkpoints

    model, payload = checkpoints.load_inference_model(network)
    state, counts = rewritten(model.state_dict(), keys, seed)
    model.load_state_dict(state)
    metadata = {**payload["metadata"], "identity_keys": {"keys": keys, "seed": seed, **counts}}
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoints.save(
        output, model, torch.optim.Adam(model.parameters()), int(payload["step"]), metadata
    )
    print(f"{output}: {keys}, rows rewritten {counts}")


def games(folder: Path, faces: bool) -> dict[tuple, float]:
    """The arm's share of each game of a folder, by deal: the network's own where it
    plays the decks, the other seat's where it faces them."""
    found = {}
    for path in sorted(folder.glob("*.jsonl")):
        for line in path.read_text(encoding="utf8").splitlines():
            game = json.loads(line)
            won = float(game["over"]["winner"] == game["seat"])
            found[(path.stem, game["other"], game["deal"], game["seat"])] = (
                1.0 - won if faces else won
            )
    return found


def against(arm: dict[tuple, float], reference: dict[tuple, float]) -> dict[str, float]:
    keys = sorted(set(arm) & set(reference))
    if len(keys) != len(reference):
        raise SystemExit("an arm was played on other deals than the reference")
    gaps = [arm[key] - reference[key] for key in keys]
    centre = sum(gaps) / len(gaps)
    spread = math.sqrt(sum((gap - centre) ** 2 for gap in gaps) / (len(gaps) - 1))
    return {"gap": centre, "error": spread / math.sqrt(len(gaps)), "games": len(gaps)}


def verdict(gap: float, error: float) -> str:
    if gap - ERRORS * error >= HELPS:
        return "help"
    if gap + ERRORS * error <= -HELPS:
        return "hurt"
    return "not decided"


def read(directory: Path) -> dict[str, dict]:
    found = {}
    for side, faces in (("plays", False), ("faces", True)):
        arms = {"as-is": games(directory / "as-is", faces)}
        arms.update(
            {path.name: games(path, faces) for path in sorted((directory / side).iterdir())}
        )
        draws = [arms[name] for name in arms if name == "as-is" or name.startswith("redraw")]
        reference = {key: sum(draw[key] for draw in draws) / len(draws) for key in draws[0]}
        told = {"draws": len(draws), "reference": sum(reference.values()) / len(reference)}
        for name, arm in arms.items():
            told[name] = {"share": sum(arm.values()) / len(arm), **against(arm, reference)}
            if name in ("blank", "mean"):
                told[name]["verdict"] = verdict(told[name]["gap"], told[name]["error"])
                each = defaultdict(list)
                for key, expected in reference.items():
                    each[key[0]].append(arm[key] - expected)
                told[name]["decks"] = {deck: sum(v) / len(v) for deck, v in sorted(each.items())}
        found[side] = told
    return found


def shown(found: dict[str, dict]) -> str:
    lines = []
    for number, (side, words) in enumerate(
        (("plays", "playing unseen cards"), ("faces", "facing unseen cards")), start=1
    ):
        told = found[side]
        blank = told["blank"]
        lines.append(
            f"{number}. Blank keys {blank['verdict']}, {words}: {100 * blank['gap']:+.1f} ± "
            f"{100 * blank['error']:.1f} points against the random keys "
            f"({100 * told['reference']:.1f}%, {told['draws']} draws), {blank['games']:,} games"
        )
    lines.append("Beside them:")
    for side in ("plays", "faces"):
        told = found[side]
        mean = told["mean"]
        draws = ", ".join(
            f"{name} {100 * told[name]['share']:.1f}%"
            for name in told
            if name == "as-is" or name.startswith("redraw")
        )
        lines.append(
            f"  {side}: mean keys {mean['verdict']}, {100 * mean['gap']:+.1f} ± {100 * mean['error']:.1f};"
            f" the random draws {draws}"
        )
        decks = told["blank"]["decks"]
        lines.append(
            "    blank deck by deck: " + ", ".join(f"{d} {100 * g:+.1f}" for d, g in decks.items())
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parts = parser.add_subparsers(dest="part", required=True)
    made = parts.add_parser("make")
    made.add_argument("network", type=Path)
    made.add_argument("--keys", required=True, choices=("blank", "mean", "redraw"))
    made.add_argument("--seed", type=int, default=0)
    made.add_argument("--output", type=Path, required=True)
    told = parts.add_parser("read")
    told.add_argument("directory", type=Path)
    args = parser.parse_args()
    if args.part == "make":
        make(args.network, args.keys, args.seed, args.output)
    else:
        found = read(args.directory)
        (args.directory / "report.json").write_text(json.dumps(found, indent=1), encoding="utf8")
        print(shown(found))


if __name__ == "__main__":
    main()
