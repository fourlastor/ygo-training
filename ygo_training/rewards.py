"""Outcome rewards independent from rollout collection and PPO math."""

from __future__ import annotations

from abc import ABC, abstractmethod


def terminal_outcome_reward(winner: int | None, player: int) -> float:
    """Return the deck-agnostic duel objective from one player's perspective."""
    if winner not in (0, 1):
        return -0.5
    return 1.0 if player == winner else -1.0


class RewardModel(ABC):
    """Maps duel outcomes and progress signals to learner rewards."""

    @abstractmethod
    def terminal(self, winner: int | None, player: int) -> float:
        """Reward a terminal outcome from one player's perspective."""

    @abstractmethod
    def cutoff(self) -> float:
        """Reward a learner-side decision-limit cutoff."""


class OutcomeRewardModel(RewardModel):
    """Optimize only duel outcomes; avoid deck-specific proxy objectives."""

    def terminal(self, winner: int | None, player: int) -> float:
        return terminal_outcome_reward(winner, player)

    def cutoff(self) -> float:
        return -1.0
