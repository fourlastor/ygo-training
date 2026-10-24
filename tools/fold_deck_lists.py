"""A checkpoint of format 14 as one of format 15, for the duels of one pairing
of decks: the network of the mirror before the deck lists went played by
today's code (beside one of today's, for one).

    uv run python tools/fold_deck_lists.py runs/more-decks/decks-01/checkpoints/model-*.pt \\
        --own blackwing --opponent blackwing --old-checkout CHECKOUT \\
        --output-dir runs/more-decks/decks-01/blackwing-mirror

A network of format 14 reads two deck lists, its seat's own and the other
seat's, each as one vector, the mean of its cards through an encoder of its
own, given to the state's first layer beside the rest. A deck's arrays are
built once and shared by its decisions, so in every duel of the same two
decks the two vectors are the same at every decision: what they add to that
layer is a constant, folded here into its bias, and the columns that read them
go with the encoder. The checkpoint that comes out is that network for those
duels, to the rounding of a float, and nothing for any other: it says so in
its metadata (`folded_deck_lists`), and its name should too.

The vectors are worked out by the old network at an old revision, which still
builds the lists' arrays: `--old-checkout` is a checkout at 05d27141 or before
(format 13 or 14), built, with its own environment (`.venv`), whose Python
this runs them with. A list the network left out (`--leave-out`) is the zero
vector it was given. The checkpoint is written with today's code, a fresh
optimizer beside it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ygo_training import checkpoints
from ygo_training.checkpoints import value_fingerprint
from ygo_training.model import PolicyValueNet

OLD_FORMATS = (13, 14)
LISTS = ("own-deck", "opponent-deck")
ENCODER = "deck_card_encoder."
FIRST = "state.0."

# Run by the old checkout's Python, from its root: the two vectors of each
# checkpoint for the decks named, through the old network's own encoder, from
# the arrays the old code builds for a list.
VECTORS = """
import json
import sys

import torch

from tests.views import view
from ygo_training import checkpoints
from ygo_training.checkpoints import load_inference_model
from ygo_training.environment import Environment
from ygo_training.packed import deck_arrays, pack_decision
from ygo_training.tensorize import batch_many_packed

own, opponent, output, *paths = sys.argv[1:]
if checkpoints.FORMAT_VERSION not in (13, 14):
    raise SystemExit(f"this checkout writes format {checkpoints.FORMAT_VERSION}, not 13 or 14")
environment = Environment(".")
lists = [deck_arrays(environment.decklist(deck)) for deck in (own, opponent)]
batch = batch_many_packed([pack_decision(view(), *lists)], "cpu")
found = {}
for path in paths:
    model, _ = load_inference_model(path, device="cpu")
    model.eval()
    with torch.no_grad():
        found[path] = {
            "own-deck": model._deck(batch, "self_deck")[0].clone(),
            "opponent-deck": model._deck(batch, "opponent_deck")[0].clone(),
            "leave_out": list(model.leave_out),
        }
torch.save({"vectors": found, "cards": [len(arrays["card"]) for arrays in lists]}, output)
"""


def vectors(
    checkout: Path, own: str, opponent: str, paths: list[Path]
) -> tuple[dict[str, dict[str, Any]], str]:
    """The two vectors of each checkpoint, by its path, and the checkout's
    revision."""
    python = checkout / ".venv" / "bin" / "python"
    if not python.exists():
        raise SystemExit(f"{checkout} has no environment of its own ({python})")
    revision = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    with tempfile.TemporaryDirectory() as scratch:
        output = Path(scratch) / "vectors.pt"
        subprocess.run(
            [str(python), "-c", VECTORS, own, opponent, str(output), *map(str, paths)],
            cwd=checkout,
            check=True,
        )
        found = torch.load(output, map_location="cpu", weights_only=False)
    return found["vectors"], revision


def fold(
    old: dict[str, torch.Tensor], own: torch.Tensor, opponent: torch.Tensor
) -> dict[str, torch.Tensor]:
    """The state of a format-15 model for a format-14 one's, the two lists'
    vectors folded into the bias of the state's first layer."""
    card = old["placed_card_encoder.0.weight"].shape[0]
    width = old[f"{ENCODER}0.weight"].shape[0]
    if own.shape != (width,) or opponent.shape != (width,):
        raise ValueError(f"A list's vector has {width} numbers")
    weight = old[f"{FIRST}weight"]
    lists = slice(card, card + 2 * width)
    added = weight[:, lists].double() @ torch.cat((own, opponent)).double()
    folded = {name: value for name, value in old.items() if not name.startswith(ENCODER)}
    folded[f"{FIRST}weight"] = torch.cat((weight[:, :card], weight[:, lists.stop :]), dim=1)
    folded[f"{FIRST}bias"] = (old[f"{FIRST}bias"].double() + added).to(weight.dtype)
    return folded


def converted(
    payload: dict[str, Any], told: dict[str, Any], source: dict[str, Any]
) -> tuple[PolicyValueNet, dict[str, Any]]:
    """Today's model of an old checkpoint for the duels its vectors are of,
    and the metadata it is written with."""
    if payload.get("format_version") not in OLD_FORMATS:
        raise SystemExit(f"{source['checkpoint']} is of format {payload.get('format_version')}")
    if "lead.0.weight" not in payload["model"]:
        raise SystemExit(f"{source['checkpoint']} has no lead head: format 13 is not folded")
    metadata = dict(payload.get("metadata", {}))
    left = set(told["leave_out"])
    width = payload["model"][f"{ENCODER}0.weight"].shape[0]
    own, opponent = (
        torch.zeros(width) if part in left else told[part].to(torch.float32) for part in LISTS
    )
    leave_out = tuple(part for part in told["leave_out"] if part not in LISTS)
    model = PolicyValueNet(
        board_representation=str(metadata.get("board_representation")), leave_out=leave_out
    )
    model.load_state_dict(fold(payload["model"], own, opponent))
    for name in ("policy_input_schema", "policy_input_schema_digest"):
        metadata.pop(name, None)
    metadata["leave_out"] = list(leave_out)
    metadata["folded_deck_lists"] = {
        **source,
        "format_version": payload["format_version"],
        "model_fingerprint": value_fingerprint(payload["model"]),
        "vectors_fingerprint": value_fingerprint({"own": own, "opponent": opponent}),
        "lists_left_out": sorted(left & set(LISTS)),
    }
    return model, metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--own", required=True, help="the deck of the network's own seat")
    parser.add_argument("--opponent", required=True, help="the deck of the other seat")
    parser.add_argument("--old-checkout", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    paths = [path.resolve() for path in args.checkpoints]
    found, revision = vectors(args.old_checkout.resolve(), args.own, args.opponent, paths)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        source = {
            "checkpoint": str(path),
            "own": args.own,
            "opponent": args.opponent,
            "revision": revision,
        }
        model, metadata = converted(payload, found[str(path)], source)
        output = args.output_dir / path.name
        optimizer = torch.optim.Adam(model.parameters())
        checkpoints.save(output, model, optimizer, int(payload["step"]), metadata)
        print(json.dumps({"written": str(output), "step": payload["step"]}))


if __name__ == "__main__":
    main()
