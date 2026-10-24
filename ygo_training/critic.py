"""A critic that sees the hidden cards, in training only
(experiments/2026-10-22-privileged-critic.md).

It reads two packings of a decision: the seat's view, exactly as the policy
has it with its history, and the same decision with nothing hidden but the
Decks' order (``DuelPool.batch(whole=True)``).  Seeing both, it knows what
the seat knows as well as the truth, which keeps the policy's gradient
unbiased (Baisero and Amato, 2022, on unbiased asymmetric reinforcement
learning under partial observability: a critic of the hidden state alone can
be biased, one of the history and the state is not).

It is never what plays.  The policy's input is the seat's view alone; a
checkpoint for play holds the policy, and this critic is kept beside it in
training (decided on 2026-10-22: a critic that sees the hidden cards is
allowed when used in training only, so long as the model plays without
seeing them).
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .model import PolicyValueNet
from .optimization import PpoOptimizer
from .rewards import RewardModel


class SightedCritic(nn.Module):
    """A duel's result for the seat, told from its view and from the duel
    with nothing hidden: a reader of each, the policy's own network in
    shape, and a value of their two states together."""

    def __init__(self, seat: PolicyValueNet, whole: PolicyValueNet):
        super().__init__()
        self.seat = seat
        self.whole = whole
        hidden = seat.hidden_dim
        self.value = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1), nn.Tanh()
        )

    @classmethod
    def from_policy(cls, policy: PolicyValueNet, *, warm: bool = False) -> SightedCritic:
        """A critic whose two readers begin as the policy's network, which
        already reads the cards.  Its value begins anew, or with ``warm`` as
        the policy's value head: what it reads of the seat's view is the
        head's, what it reads of the whole duel nothing yet, so that it
        begins by telling each decision's value as the head does."""
        critic = cls(copy.deepcopy(policy), copy.deepcopy(policy))
        if warm:
            hidden = policy.hidden_dim
            first, last = critic.value[0], critic.value[2]
            with torch.no_grad():
                first.weight.zero_()
                first.weight[:, :hidden].copy_(policy.value[0].weight)
                first.bias.copy_(policy.value[0].bias)
                last.weight.copy_(policy.value[2].weight)
                last.bias.copy_(policy.value[2].bias)
        return critic

    def forward(self, seat: dict[str, Tensor], whole: dict[str, Tensor]) -> Tensor:
        """The value of each decision: ``seat`` as the policy is given it,
        ``whole`` the same decisions with nothing hidden."""
        seen, _, _ = self.seat.encode_state(seat)
        known, _, _ = self.whole.encode_state(whole)
        return self.value(torch.cat((seen, known), dim=-1)).squeeze(-1)


@dataclass(frozen=True, eq=False)
class Root:
    """The first decision of a transaction an update of PPO trains on: its
    step, its duel among the update's, its seat, and how the duel ended for
    that seat."""

    step: dict[str, Any]
    duel: int
    seat: int
    result: float


def transaction_roots(trajectories: list[dict[str, Any]], rewards: RewardModel) -> list[Root]:
    """The first decision of each transaction an update of PPO trains on, of
    the seats `PpoOptimizer.prepare_samples` learns from: both of a duel
    that ended and of one against itself cut short, the learner's of
    another cut short; each with its seat's result, which a duel cut short
    gives as `RewardModel.cutoff`."""
    found = []
    for duel, trajectory in enumerate(trajectories):
        cutoff = not trajectory["terminal"]
        both = not cutoff or trajectory.get("other") == "self"
        for seat in (0, 1) if both else (int(trajectory["learner_player"]),):
            result = rewards.cutoff() if cutoff else rewards.terminal(trajectory["winner"], seat)
            opened: set[object] = set()
            for index, step in enumerate(trajectory["steps"]):
                if step["player"] != seat:
                    continue
                key = PpoOptimizer._transaction_key(duel, index, step)
                if key not in opened:
                    opened.add(key)
                    found.append(Root(step, duel, seat, float(result)))
    return found


def drawn(found: list[Any], positions: int, generator: random.Random) -> list[Any]:
    """``positions`` of each seat's transactions in a duel, drawn at random,
    in an order drawn too: what the critic is taught after an update's games.
    A duel's result is one outcome, and its positions tell it again and
    again (the value head taught every one of them learnt the results by
    heart, taught four did not: experiments/2026-10-08-value-head-and-board.md)."""
    by_seat: dict[tuple[int, int], list[Any]] = {}
    for root in found:
        by_seat.setdefault((root.duel, root.seat), []).append(root)
    taken = []
    for seat in by_seat.values():
        taken.extend(seat if len(seat) <= positions else generator.sample(seat, positions))
    generator.shuffle(taken)
    return taken
