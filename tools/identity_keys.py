"""What a card the network never trained on costs at play, by its random identity
(experiments/2026-11-02-expanded-pool.md, the card identities beside the batches).

    uv run python tools/identity_keys.py make runs/expanded-pool/first-batch-mean/4320.pt \\
        --keys blank --output runs/identity-keys/blank.pt
    uv run python tools/identity_keys.py read analysis/identity-keys
    uv run python tools/identity_keys.py make runs/expanded-pool/batch-one-mean/4020.pt \\
        --begun runs/expanded-pool/first-batch-mean/4320.pt --keys initial \\
        --output runs/identity-rows/control/initial.pt
    uv run python tools/identity_keys.py rows analysis/identity-rows runs/identity-rows

A card's identity is a row of each of two tables, indexed by the low and the high 16 bits
of its code. A row no update moved is still its untrained value, which every run of the
project draws at seed 2012: a random key the network learnt nothing about. `make`
rewrites those rows only: `blank` to zero, `mean` to the mean of the rows training moved,
`redraw` afresh from the untrained distribution (`--seed`). Row 0 of each table, the code
of a card the seat cannot see, is never touched.

With `--begun`, the network a run began from, `make` rewrites the rows of the cards the
run met first instead: those it moved where that network still had them untrained. They
are put back to their untrained values (`initial`), what the run learnt of those cards
then kept outside their rows, or drawn afresh (`redraw`), the keys the rest of the
network learnt to read gone too.

`read` takes the games `experiments/commands/identity-keys.sh` leaves: in `as-is/` the
network against itself, in `plays/ARM/` the network with ARM's keys at the decks it never
trained on against itself as it is, in `faces/ARM/` the network as it is at those decks
against ARM. Each arm is read against the mean of the random keys (`as-is` and the
`redraw` arms), the deals paired; beside it, what a key's luck is worth: how far the draws
lie apart beyond what their games tell.

`rows` takes the games `experiments/commands/identity-rows.sh` leaves: in `start/` the
network a run began from against itself, in `NAME/ARM/` the network NAME names, as it is
(`as-is`) and with its rows rewritten (`initial`, `redraw`), at the run's new decks against
that start. Each network's gain is read against the start's own games and each arm against
the network as it is, the deals paired, beside the rows rewritten (from the arms in the
second directory) and how far the run moved them.
"""

from __future__ import annotations

import argparse
import itertools
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


def put_back(
    state: dict[str, torch.Tensor],
    begun: dict[str, torch.Tensor],
    keys: str,
    seed: int = 0,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """The state with the rows of the cards a run met first, those it moved where the
    network it began from (`begun`) had them untrained, put back to their untrained
    values (`initial`) or drawn afresh (`redraw`); how many, how far the run moved them
    on average, and the average length of their untrained keys."""
    start = untrained()
    generator = torch.Generator().manual_seed(seed)
    new, told = dict(state), {}
    for name in TABLES:
        table = state[name]
        met = (begun[name] == start[name]).all(dim=1) & (table != start[name]).any(dim=1)
        met[0] = False
        made = table.clone()
        if keys == "initial":
            made[met] = start[name][met]
        elif keys == "redraw":
            made[met] = torch.randn(
                (int(met.sum()), table.shape[1]), generator=generator, dtype=table.dtype
            )
        else:
            raise SystemExit(f"with --begun keys are initial or redraw, not {keys}")
        new[name] = made
        told[name] = int(met.sum())
        told[f"{name} moved"] = float((table[met] - start[name][met]).norm(dim=1).mean())
        told[f"{name} key"] = float(start[name][met].norm(dim=1).mean())
    return new, told


def make(network: Path, keys: str, seed: int, output: Path, begun: Path | None = None) -> None:
    from ygo_training import checkpoints

    model, payload = checkpoints.load_inference_model(network)
    if begun is None:
        state, counts = rewritten(model.state_dict(), keys, seed)
    else:
        start, _ = checkpoints.load_inference_model(begun)
        state, counts = put_back(model.state_dict(), start.state_dict(), keys, seed)
        counts = {**counts, "begun": str(begun)}
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


def luck(draws: list[dict[tuple, float]]) -> dict[str, float]:
    """What a random key's luck is worth: the spread of the draws' shares beyond what their
    games tell apart, from each two paired on their deals."""
    pairs = [against(first, second) for first, second in itertools.combinations(draws, 2)]
    excess = sum(pair["gap"] ** 2 - pair["error"] ** 2 for pair in pairs) / len(pairs)
    spread = math.sqrt(max(0.0, excess / 2))
    return {
        "spread": spread,
        "widest": max(abs(pair["gap"]) for pair in pairs),
        "reference": spread / math.sqrt(len(draws)),
    }


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
        told = {
            "draws": len(draws),
            "reference": sum(reference.values()) / len(reference),
            "luck": luck(draws),
        }
        for name, arm in arms.items():
            told[name] = {"share": sum(arm.values()) / len(arm), **against(arm, reference)}
            if name in ("blank", "mean"):
                told[name]["verdict"] = verdict(told[name]["gap"], told[name]["error"])
                # The reading's error is the games'; beside it, with the luck of the draws
                # the reference was made of.
                told[name]["with luck"] = math.hypot(told[name]["error"], told["luck"]["reference"])
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
        drawn = told["luck"]
        lines.append(
            f"    a key's luck: {100 * drawn['spread']:.1f} points a draw, draws up to "
            f"{100 * drawn['widest']:.1f} apart; with it blank ± {100 * told['blank']['with luck']:.1f}"
        )
        decks = told["blank"]["decks"]
        lines.append(
            "    blank deck by deck: " + ", ".join(f"{d} {100 * g:+.1f}" for d, g in decks.items())
        )
    return "\n".join(lines)


def rows(directory: Path, arms: Path) -> dict[str, dict]:
    """Each network's gain over the start its run began from, and what its arms cost of
    it, from the games `experiments/commands/identity-rows.sh` leaves; the rows rewritten
    as the network's `initial` arm in `arms` tells them."""
    start = games(directory / "start", faces=False)
    found: dict[str, dict] = {"start": {"share": sum(start.values()) / len(start)}}
    for folder in sorted(path for path in directory.iterdir() if (path / "as-is").is_dir()):
        as_is = games(folder / "as-is", faces=False)
        told = {"share": sum(as_is.values()) / len(as_is), "gain": against(as_is, start)}
        for arm in ("initial", "redraw"):
            arm_games = games(folder / arm, faces=False)
            told[arm] = {
                "share": sum(arm_games.values()) / len(arm_games),
                **against(arm_games, as_is),
            }
        payload = torch.load(
            arms / folder.name / "initial.pt", map_location="cpu", weights_only=False
        )
        told["rows"] = payload["metadata"]["identity_keys"]
        found[folder.name] = told
    return found


def shown_rows(found: dict[str, dict]) -> str:
    lines = [f"The start against itself: {100 * found['start']['share']:.1f}%"]
    for name, told in found.items():
        if name == "start":
            continue
        gain, made = told["gain"], told["rows"]
        moved = "; ".join(
            f"{made[table]:,} rows of the {table.split('.')[1]} table, moved"
            f" {made[f'{table} moved']:.2f} against keys of {made[f'{table} key']:.2f}"
            for table in TABLES
        )
        lines.append(
            f"{name}: {100 * told['share']:.1f}%, {100 * gain['gap']:+.1f} ± {100 * gain['error']:.1f}"
            f" points over the start, {gain['games']:,} games; {moved}"
        )
        for arm, words in (
            ("initial", "put back to their untrained values"),
            ("redraw", "drawn afresh"),
        ):
            cost = told[arm]
            lines.append(
                f"  the rows of the cards it met first {words}: {100 * cost['gap']:+.1f} ±"
                f" {100 * cost['error']:.1f} points, {-cost['gap'] / gain['gap']:.0%} of its gain"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parts = parser.add_subparsers(dest="part", required=True)
    made = parts.add_parser("make")
    made.add_argument("network", type=Path)
    made.add_argument("--keys", required=True, choices=("blank", "mean", "redraw", "initial"))
    made.add_argument("--seed", type=int, default=0)
    made.add_argument("--begun", type=Path)
    made.add_argument("--output", type=Path, required=True)
    told = parts.add_parser("read")
    told.add_argument("directory", type=Path)
    told_rows = parts.add_parser("rows")
    told_rows.add_argument("directory", type=Path)
    told_rows.add_argument("arms", type=Path)
    args = parser.parse_args()
    if args.part == "make":
        if args.keys == "initial" and args.begun is None:
            parser.error("rows are put back to their untrained values with --begun")
        make(args.network, args.keys, args.seed, args.output, args.begun)
    elif args.part == "rows":
        found = rows(args.directory, args.arms)
        (args.directory / "report.json").write_text(json.dumps(found, indent=1), encoding="utf8")
        print(shown_rows(found))
    else:
        found = read(args.directory)
        (args.directory / "report.json").write_text(json.dumps(found, indent=1), encoding="utf8")
        print(shown(found))


if __name__ == "__main__":
    main()
