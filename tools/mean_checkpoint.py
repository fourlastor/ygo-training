"""The mean of a run's checkpoints, weight by weight, as a checkpoint to play:
what they share stays, and little of what the noise of their updates left in
each (experiments/2026-10-28-seventy-one-champions.md).

    uv run python tools/mean_checkpoint.py \\
        runs/seventy-one-champions/ppo/checkpoints/model-14159.pt ... \\
        --output runs/seventy-one-champions/ppo-mean/model-17039.pt

What comes out has the last checkpoint's step and metadata, says what it is the
mean of (`mean_of`), and has a fresh optimizer beside it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ygo_training import checkpoints


def meaned(states: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Each weight's mean over the states, taken in doubles. What is not a
    float no update moves, and has to be the same in all of them."""
    first = states[0]
    if any(state.keys() != first.keys() for state in states):
        raise SystemExit("The checkpoints do not hold the same weights")
    mean = {}
    for name, value in first.items():
        if value.is_floating_point():
            mean[name] = torch.stack([state[name].double() for state in states]).mean(dim=0)
            mean[name] = mean[name].to(value.dtype)
        elif all(torch.equal(state[name], value) for state in states):
            mean[name] = value
        else:
            raise SystemExit(f"{name} is not a float and differs between the checkpoints")
    return mean


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model, payload = checkpoints.load_inference_model(args.checkpoints[-1])
    states = [
        torch.load(path, map_location="cpu", weights_only=False)["model"]
        for path in args.checkpoints
    ]
    model.load_state_dict(meaned(states))
    metadata = {**payload["metadata"], "mean_of": [path.name for path in args.checkpoints]}
    optimizer = torch.optim.Adam(model.parameters())
    checkpoints.save(args.output, model, optimizer, int(payload["step"]), metadata)
    print(f"{args.output}: the mean of {len(states)} checkpoints")


if __name__ == "__main__":
    main()
