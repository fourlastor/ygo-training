from __future__ import annotations

import torch

from ygo_training import inference
from ygo_training.inference import _sample_with_seeds


def test_seeded_sampling_is_independent_of_batch_order_and_membership() -> None:
    logits = torch.tensor(
        [
            [0.1, 0.2, 0.3],
            [1.0, -1.0, 0.0],
            [-0.5, 0.5, 0.25],
        ]
    )
    seeds = [11, 22, 33]

    together = _sample_with_seeds(logits, seeds, 1.0)
    reordered = _sample_with_seeds(logits[[2, 0]], [33, 11], 1.0)
    alone = _sample_with_seeds(logits[1:2], [22], 1.0)

    assert torch.equal(reordered, together[[2, 0]])
    assert torch.equal(alone, together[1:2])


def test_a_draw_at_the_top_of_its_range_stays_among_the_answers_offered(monkeypatch) -> None:
    """The running sum of three chances ends a digit under 1 here, and the
    largest uniform a seed gives is 1 as a float32: the draw must not run on
    into the columns a longer decision of the batch pads the row with."""
    absent = float("-inf")
    logits = torch.tensor(
        [
            [0.3, -1.2, 0.9, absent, absent, absent, absent, absent],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            [0.3, absent, 0.9, -1.2, absent, absent, absent, absent],
        ]
    )
    assert float(torch.softmax(logits, dim=-1).cumsum(-1)[0, -1]) < 1.0
    monkeypatch.setattr(inference, "_seed_uniform", lambda seed: (2**53 - 1) / 2**53)

    assert _sample_with_seeds(logits, [1, 2, 3], 1.0).tolist() == [2, 7, 3]

    # Anywhere else in its range a draw is the count it was.
    monkeypatch.setattr(inference, "_seed_uniform", lambda seed: seed / 10)
    uniforms = torch.tensor([[0.0], [0.6], [0.9]])
    counted = (torch.softmax(logits, dim=-1).cumsum(-1) < uniforms).sum(-1)
    assert _sample_with_seeds(logits, [0, 6, 9], 1.0).tolist() == counted.tolist() == [0, 4, 2]
