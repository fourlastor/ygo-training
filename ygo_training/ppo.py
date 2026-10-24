from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


@dataclass
class PpoConfig:
    clip_epsilon: float = 0.2
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.5
    max_gradient_norm: float = 1.0
    rollout_temperature: float = 1.0


def generalized_advantage(
    rewards: Tensor, values: Tensor, dones: Tensor, gamma: float = 1.0, gae_lambda: float = 1.0
) -> tuple[Tensor, Tensor]:
    advantages = torch.zeros_like(rewards)
    accumulator = torch.zeros((), device=rewards.device)
    for i in range(len(rewards) - 1, -1, -1):
        next_value = values[i + 1] if i + 1 < len(values) else torch.zeros((), device=values.device)
        delta = rewards[i] + gamma * next_value * (1 - dones[i]) - values[i]
        accumulator = delta + gamma * gae_lambda * (1 - dones[i]) * accumulator
        advantages[i] = accumulator
    return advantages, advantages + values


def transaction_advantages(
    rewards: list[float],
    values: list[float],
    dones: list[float],
    gamma: float = 1.0,
    gae_lambda: float = 1.0,
) -> tuple[list[float], list[float]]:
    """What :func:`generalized_advantage` gives for the same numbers, worked
    out on the processor one 32-bit number at a time.

    The tensors' loop makes a dozen small operations for every transaction.
    On a graphics card each of them is started and waited for, and every
    result read back is one more wait: seconds an update, for arithmetic a
    processor does in milliseconds.  Every operation here is the one the
    tensors make, on the same 32-bit numbers and in the same order, so the
    results are theirs to the last bit; a test holds the two together.
    """
    single = np.float32
    count = len(rewards)
    reward = [single(value) for value in rewards]
    value = [single(entry) for entry in values]
    going_on = [single(1) - single(done) for done in dones]
    discount, decay, zero = single(gamma), single(gamma * gae_lambda), single(0)
    advantages = [zero] * count
    accumulator = zero
    for i in range(count - 1, -1, -1):
        next_value = value[i + 1] if i + 1 < count else zero
        delta = reward[i] + discount * next_value * going_on[i] - value[i]
        accumulator = delta + decay * going_on[i] * accumulator
        advantages[i] = accumulator
    return (
        [float(advantage) for advantage in advantages],
        [float(advantage + entry) for advantage, entry in zip(advantages, value, strict=True)],
    )


def transaction_upgo_advantages(
    advantages: list[float],
    rewards: list[float],
    values: list[float],
    dones: list[float],
    gamma: float = 1.0,
) -> list[float]:
    """The advantages of :func:`transaction_advantages` with UPGO added, as
    ygo-agent's trainer adds it (`truncated_gae` with `upgo`, read at
    `26293f8`): to each transaction's advantage, its UPGO return less the
    head's estimate there.

    The UPGO return follows the seat's return on through the transaction
    after for as long as that one went at least as well as the head
    expected, its reward and the head's estimate after it at least the
    head's estimate before; where it went worse it stops and takes that
    estimate.  Theirs turns the numbers round where the other seat moves
    next; a seat's transactions here are all its own.  The same 32-bit
    numbers in the same order as theirs, and the head's target is left as
    it is, as theirs is.
    """
    single = np.float32
    count = len(rewards)
    reward = [single(value) for value in rewards]
    value = [single(entry) for entry in values]
    going_on = [single(1) - single(done) for done in dones]
    discount, zero = single(gamma), single(0)
    upgo_return = next_q = zero
    upgo = [zero] * count
    for i in range(count - 1, -1, -1):
        next_value = (value[i + 1] if i + 1 < count else zero) * going_on[i]
        step = discount * going_on[i]
        followed = upgo_return if next_q >= next_value else next_value
        upgo_return = reward[i] + step * followed
        next_q = reward[i] + step * next_value
        upgo[i] = single(advantages[i]) + (upgo_return - value[i])
    return [float(entry) for entry in upgo]


def ppo_loss(
    logits: Tensor,
    actions: Tensor,
    old_log_probs: Tensor,
    advantages: Tensor,
    returns: Tensor,
    values: Tensor,
    config: PpoConfig,
) -> Tensor:
    return ppo_loss_components(
        logits,
        actions,
        old_log_probs,
        advantages,
        returns,
        values,
        config,
    )["total"]


def ppo_loss_components(
    logits: Tensor,
    actions: Tensor,
    old_log_probs: Tensor,
    advantages: Tensor,
    returns: Tensor,
    values: Tensor,
    config: PpoConfig,
) -> dict[str, Tensor]:
    """Return the exact scalar terms used by the production PPO objective."""
    if config.rollout_temperature <= 0.0:
        raise ValueError("rollout_temperature must be positive")
    # Rollout sampling and PPO must use the same distribution.  A lower
    # temperature keeps stochastic exploration valid while making it much more
    # likely to visit the argmax branches used at deployment.
    distribution = torch.distributions.Categorical(logits=logits / config.rollout_temperature)
    new_log_probs = distribution.log_prob(actions)
    log_ratio = new_log_probs - old_log_probs
    ratio = log_ratio.exp()
    policy = -torch.minimum(
        ratio * advantages,
        ratio.clamp(1 - config.clip_epsilon, 1 + config.clip_epsilon) * advantages,
    ).mean()
    value = torch.nn.functional.mse_loss(values, returns)
    entropy = distribution.entropy().mean()
    weighted_value = config.value_coefficient * value
    weighted_entropy = -config.entropy_coefficient * entropy
    approximate_kl = ((ratio - 1.0) - log_ratio).mean()
    clip_fraction = ((ratio - 1.0).abs() > config.clip_epsilon).to(dtype=torch.float32).mean()
    return {
        "policy": policy,
        "value": value,
        "weighted_value": weighted_value,
        "entropy": entropy,
        "weighted_entropy": weighted_entropy,
        "approximate_kl": approximate_kl,
        "clip_fraction": clip_fraction,
        "total": policy + weighted_value + weighted_entropy,
    }


def joint_ppo_loss_components(
    logits: Tensor,
    actions: Tensor,
    old_log_probs: Tensor,
    group_ids: Tensor,
    root_positions: Tensor,
    advantages: Tensor,
    returns: Tensor,
    values: Tensor,
    config: PpoConfig,
    value_roots: Tensor | None = None,
) -> dict[str, Tensor]:
    """PPO over variable-length transactions of conditional policy choices.

    ``value_roots`` names, among the transactions, those whose value the
    head is taught: all of them when left out.  Every transaction keeps its
    part in the policy's loss whatever it says.
    """
    if config.rollout_temperature <= 0.0:
        raise ValueError("rollout_temperature must be positive")
    distribution = torch.distributions.Categorical(logits=logits / config.rollout_temperature)
    step_log_probs = distribution.log_prob(actions)
    groups = int(root_positions.numel())
    new_joint = torch.zeros(groups, dtype=step_log_probs.dtype, device=logits.device)
    old_joint = torch.zeros_like(new_joint)
    joint_entropy = torch.zeros_like(new_joint)
    new_joint.index_add_(0, group_ids, step_log_probs)
    old_joint.index_add_(0, group_ids, old_log_probs)
    joint_entropy.index_add_(0, group_ids, distribution.entropy())
    log_ratio = new_joint - old_joint
    ratio = log_ratio.exp()
    policy = -torch.minimum(
        ratio * advantages,
        ratio.clamp(1 - config.clip_epsilon, 1 + config.clip_epsilon) * advantages,
    ).mean()
    if value_roots is None:
        value = torch.nn.functional.mse_loss(values[root_positions], returns)
    elif value_roots.numel():
        value = torch.nn.functional.mse_loss(
            values[root_positions][value_roots], returns[value_roots]
        )
    else:
        # None of the batch's transactions carries a target.
        value = torch.zeros((), dtype=values.dtype, device=values.device)
    entropy = joint_entropy.mean()
    weighted_value = config.value_coefficient * value
    weighted_entropy = -config.entropy_coefficient * entropy
    approximate_kl = ((ratio - 1.0) - log_ratio).mean()
    clip_fraction = ((ratio - 1.0).abs() > config.clip_epsilon).float().mean()
    return {
        "policy": policy,
        "value": value,
        "weighted_value": weighted_value,
        "entropy": entropy,
        "weighted_entropy": weighted_entropy,
        "approximate_kl": approximate_kl,
        "clip_fraction": clip_fraction,
        "total": policy + weighted_value + weighted_entropy,
    }
