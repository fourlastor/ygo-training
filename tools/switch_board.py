"""A network trained with its board pooled, given the board as its cards from
where it stands (experiments/2026-11-02-expanded-pool.md, the second session
of the screen of the network under PPO).

    uv run python tools/switch_board.py \\
        runs/network-screen/cyber/checkpoints/model-00959.pt \\
        --output runs/network-screen/cyber-switched.begun.pt

Both shapes own the same weights, and the pooled board never reads those of
the attention. With the board as its cards each answer attends to the cards
on the field and adds what it reads through `board_fusion`, set here to zero:
the network answers as the one it came from until training moves it. What
comes out has the checkpoint's step, its optimizer's state and its random
states, and names the checkpoint it was switched from (`switched_from`). A
run begins from it with --resume and --board-representation
individual-tokens.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ygo_training.checkpoints import load_inference_model
from ygo_training.config import left_out
from ygo_training.schema import model_policy_input_schema, schema_digest

POOLED = "mean-pool"
TOKENS = "individual-tokens"
FUSION = "board_fusion.weight"


def switched(payload: dict[str, Any], source: str) -> dict[str, Any]:
    """The checkpoint ``payload`` of a pooled board as one of the board as its
    cards, its fusion of what an answer reads at zero."""
    metadata = payload.get("metadata", {})
    if metadata.get("board_representation") != POOLED:
        raise SystemExit(
            f"{source} reads its board as {metadata.get('board_representation')!r}, "
            f"not {POOLED!r}: there is nothing to switch"
        )
    state = dict(payload["model"])
    if FUSION not in state:
        raise SystemExit(f"{source} holds no {FUSION}")
    state[FUSION] = torch.zeros_like(state[FUSION])
    schema = model_policy_input_schema(
        board_representation=TOKENS, leave_out=left_out(metadata.get("leave_out", ()))
    )
    return {
        **payload,
        "model": state,
        "metadata": {
            **metadata,
            "board_representation": TOKENS,
            "policy_input_schema": schema,
            "policy_input_schema_digest": schema_digest(schema),
            "switched_from": source,
        },
    }


def switch(source: Path, output: Path) -> dict[str, Any]:
    """Write ``source`` switched to ``output`` with its description beside it,
    each whole under another name first, and check that it loads as a network
    of the board as its cards with every weight but the fusion's as it was."""
    payload = torch.load(source, map_location="cpu", weights_only=False)
    result = switched(payload, source.name)
    output.parent.mkdir(parents=True, exist_ok=True)
    described = output.with_suffix(output.suffix + ".json")
    unfinished = output.with_name(output.name + ".part")
    torch.save(result, unfinished)
    described.write_text(
        json.dumps(
            {
                "format_version": result["format_version"],
                "step": result["step"],
                **result["metadata"],
            },
            indent=2,
        ),
        encoding="utf8",
    )
    os.replace(unfinished, output)
    model, loaded = load_inference_model(output)
    if model.board_representation != TOKENS or loaded["step"] != payload["step"]:
        raise SystemExit(f"{output} does not load as the network switched")
    for name, value in model.state_dict().items():
        expected = torch.zeros_like(value) if name == FUSION else payload["model"][name]
        if not torch.equal(value, expected.to(value.device)):
            raise SystemExit(f"{output} holds another {name} than {source}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = switch(args.checkpoint, args.output)
    print(
        f"{args.output}: {args.checkpoint.name} at step {result['step']} with the board as "
        "its cards, board_fusion at zero"
    )


if __name__ == "__main__":
    main()
