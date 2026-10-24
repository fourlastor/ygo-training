"""Small interfaces for interchangeable training frontends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Trainer(ABC):
    """Lifecycle shared by multiprocess PPO and future trainer backends."""

    @abstractmethod
    def train_update(
        self, checkpoint=None, step: int = 1, *, finish: bool = True, ahead: bool = False
    ) -> dict[str, Any]:
        """Collect rollouts, optimize the policy, and optionally checkpoint.

        Unfinished, the checkpoint is written and left for the caller to give
        its name, once what has to come before that is done. With ``ahead``
        another update follows, whose games may be begun before this one's
        pass.
        """

    @abstractmethod
    def close(self, force: bool = False) -> None:
        """Release worker processes and native engine resources."""
