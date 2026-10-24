"""UPGO and the PPO clip, two of the settings of ygo-agent's trainer that
this one can take (`train --upgo`, `--clip`).

UPGO is held to a transcription of ygo-agent's own loop (`truncated_gae`
and `truncated_gae_loop` of `ygoai/rl/jax`, read at `26293f8`), line for
line in 32-bit numbers, on a seat's transactions as an update here has
them: every one the same seat's, rewarded at the last only.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ygo_training.__main__ import DEFAULT_LEARNER
from ygo_training.cli_parser import build_parser
from ygo_training.config import TrainerConfig
from ygo_training.ppo import (
    PpoConfig,
    ppo_loss_components,
    transaction_advantages,
    transaction_upgo_advantages,
)


def theirs(
    rewards: list[float], values: list[float], dones: list[float], gamma: float, gae_lambda: float
) -> tuple[list[float], list[float]]:
    """ygo-agent's `truncated_gae` with `upgo`, the next value 0 as a whole
    duel leaves it and every step the same player's: (targets, advantages)."""
    single = np.float32
    lastgaelam = single(0)
    next_value = last_return = next_q = single(0)
    next_main = True
    gaes, upgo_returns = [], []
    for cur_value, next_done, reward in reversed(
        list(zip(map(single, values), map(bool, dones), map(single, rewards), strict=True))
    ):
        main = True
        lastgaelam = single(0) if next_done else lastgaelam
        next_value = single(0) if next_done else next_value
        sign = single(1) if main == next_main else single(-1)
        lastgaelam = lastgaelam * sign
        next_value = next_value * sign
        discount = single(gamma) * (single(1) - single(next_done))
        delta = reward + discount * next_value - cur_value
        lastgaelam = delta + discount * single(gae_lambda) * lastgaelam
        last_return = last_return * sign
        next_q = next_q * sign
        last_return = reward + discount * (last_return if next_q >= next_value else next_value)
        next_q = reward + discount * next_value
        next_value, next_main = cur_value, main
        gaes.append(lastgaelam)
        upgo_returns.append(last_return)
    gaes.reverse()
    upgo_returns.reverse()
    value = [single(entry) for entry in values]
    targets = [v + a for v, a in zip(value, gaes, strict=True)]
    advantages = [a + (r - v) for a, r, v in zip(gaes, upgo_returns, value, strict=True)]
    return [float(t) for t in targets], [float(a) for a in advantages]


@pytest.mark.parametrize("gae_lambda", [0.95, 1.0])
def test_upgo_is_ygo_agents_to_the_bit(gae_lambda):
    generator = np.random.default_rng(5)
    for length in (1, 2, 3, 7, 40, 90):
        for _ in range(20):
            values = generator.uniform(-1, 1, length).astype(np.float32).tolist()
            rewards = [0.0] * length
            rewards[-1] = float(generator.choice([-1.0, 1.0]))
            dones = [0.0] * length
            dones[-1] = 1.0
            advantages, returns = transaction_advantages(
                rewards, values, dones, gae_lambda=gae_lambda
            )
            targets, expected = theirs(rewards, values, dones, 1.0, gae_lambda)
            # The head's target is what it was, and the advantage theirs.
            assert returns == targets
            assert transaction_upgo_advantages(advantages, rewards, values, dones) == expected


def test_upgo_follows_the_return_while_the_play_after_went_as_expected():
    # The head thought 0.0, then 0.5, then 0.2, and the duel was won. The
    # last transaction's UPGO return is the result; the one before follows
    # it, the result being over what the head thought there; the first
    # stops at 0.5, the play after it having gone worse than expected.
    values, rewards, dones = [0.0, 0.5, 0.2], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]
    advantages, _ = transaction_advantages(rewards, values, dones, gae_lambda=1.0)
    added = [
        upgo - plain
        for upgo, plain in zip(
            transaction_upgo_advantages(advantages, rewards, values, dones), advantages, strict=True
        )
    ]
    assert added == pytest.approx([0.5, 0.5, 0.8])


def test_the_clip_and_upgo_are_the_trainers_to_set():
    parser = build_parser(
        handlers={"train": print, "status": print}, default_learner=DEFAULT_LEARNER
    )
    args = parser.parse_args(["train", "--clip", "0.25", "--upgo"])
    assert (args.clip, args.upgo) == (0.25, True)
    args = parser.parse_args(["train"])
    assert (args.clip, args.upgo) == (0.2, False)
    assert TrainerConfig(clip_epsilon=0.25, upgo=True).clip_epsilon == 0.25
    with pytest.raises(ValueError, match="clip_epsilon"):
        TrainerConfig(clip_epsilon=0.0)


def test_the_clip_bounds_the_ratio_the_loss_follows():
    # One answer, its chance up by 22% since the rollout: under a clip of
    # 0.2 the gain of a positive advantage stops at 1.2, under 0.25 it does
    # not.
    logits = torch.tensor([[0.45, 0.0]])
    old = torch.log_softmax(torch.tensor([[0.0, 0.0]]), dim=-1)[0, 0]
    ratio = torch.exp(torch.log_softmax(logits, dim=-1)[0, 0] - old)
    assert 1.2 < float(ratio) < 1.25
    arguments = (
        logits,
        torch.tensor([0]),
        old.reshape(1),
        torch.tensor([1.0]),
        torch.tensor([0.0]),
        torch.tensor([0.0]),
    )
    tight = ppo_loss_components(*arguments, PpoConfig(clip_epsilon=0.2, entropy_coefficient=0.0))
    loose = ppo_loss_components(*arguments, PpoConfig(clip_epsilon=0.25, entropy_coefficient=0.0))
    assert float(tight["clip_fraction"]) == 1.0 and float(loose["clip_fraction"]) == 0.0
    assert float(tight["policy"]) == pytest.approx(-1.2)
    assert float(loose["policy"]) == pytest.approx(-float(ratio))
