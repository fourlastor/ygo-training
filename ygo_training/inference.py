"""How a network's answer is drawn, and a network asked outside the pool."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from .environment import Environment

_UINT64_MASK = (1 << 64) - 1


def _seed_uniform(seed: int) -> float:
    """Map an integer seed to a stable uniform using SplitMix64."""
    value = (int(seed) + 0x9E3779B97F4A7C15) & _UINT64_MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
    value ^= value >> 31
    return ((value >> 11) & ((1 << 53) - 1)) / float(1 << 53)


def _sample_with_seeds(
    logits: torch.Tensor,
    seeds: list[int],
    temperature: float,
) -> torch.Tensor:
    """Sample each row independently of batch membership and arrival order."""
    if len(seeds) != logits.shape[0]:
        raise ValueError("Each sampled inference request requires one sample seed.")
    probabilities = torch.softmax(logits / temperature, dim=-1)
    uniforms = torch.tensor(
        [_seed_uniform(seed) for seed in seeds],
        dtype=probabilities.dtype,
        device=probabilities.device,
    )
    chosen = (probabilities.cumsum(dim=-1) < uniforms.unsqueeze(-1)).sum(dim=-1)
    # The running sum can end a digit under 1 and the uniform, as a float32,
    # reach 1: the count then runs past the answers.  The last answer with a
    # chance takes it, not the last column, which in a row padded beside a
    # longer one is no answer of the decision.
    possible = probabilities > 0
    last = possible.shape[-1] - 1 - possible.flip(-1).to(torch.int64).argmax(dim=-1)
    return torch.minimum(chosen, last)


def scored(
    model: torch.nn.Module,
    environment: Environment,
    views: list[dict[str, Any]],
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """What ``model`` makes of decisions it is shown outside the pool (at a
    table of EDOPro, in a duel read again): the scores of their answers and
    their values, the views packed into one batch by the library."""
    batch = {
        name: torch.from_numpy(array).to(device)
        for name, array in environment.batch(views).items()
    }
    with torch.no_grad():
        return model(batch)
