"""A head that learns how cards work, in training only
(experiments/2026-09-08-engine-transition-prediction.md).

At the decision that ends a transaction of the seat, it tells from the
policy network's own card tokens, state and chosen answer what the seat
will see at the horizon: where each card of the decision's state is, the
change of each one's ATK and DEF on the field, and of each seat's life
points and cards in each location (`crates/ygo-env/src/follow.rs`, the
labels; `DuelPool.labels`).  Its gradient reaches the network: what it is
for is to shape what the network learns of the cards.  It is never what
plays: it is kept beside a checkpoint, and nothing that plays reads it.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from .model import PolicyValueNet

# The places a card can be found at (`follow::PLACES`): on its own seat's
# side, then the other's, and out of the duel.
PLACES = 23
# The locations each seat's cards are counted in (`follow::COUNTED`).
COUNTED = 7
# What `DuelPool.labels` gives, under these names in a minibatch.
LABELS = ("labelled", "card_place", "card_change", "life_change", "count_change")
PREFIX = "label_"


class CardHead(nn.Module):
    """Of a decision and its chosen answer: each card's place at the horizon
    and the change of its ATK and DEF, and the change of each seat's life
    points and counted cards."""

    def __init__(self, policy: PolicyValueNet):
        super().__init__()
        hidden = policy.hidden_dim
        card = policy.card_representation_dim
        self.card = nn.Sequential(
            nn.Linear(card + 2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, PLACES + 2)
        )
        self.board = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2 + 2 * COUNTED)
        )

    def forward(
        self, state: Tensor, tokens: Tensor, answer: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """The places' logits and the changes of each card (``tokens``, its
        decision's ``state`` and chosen ``answer`` beside each), and the life
        points' and counts' changes."""
        both = torch.cat((state, answer), dim=-1)
        each = torch.cat((tokens, both.unsqueeze(1).expand(-1, tokens.shape[1], -1)), dim=-1)
        card = self.card(each)
        board = self.board(both)
        return card[..., :PLACES], card[..., PLACES:], board[..., :2], board[..., 2:]


class _Scaled(torch.autograd.Function):
    """The tensor as it is, its gradient scaled on the way back."""

    @staticmethod
    def forward(ctx: Any, tensor: Tensor, weight: float) -> Tensor:
        ctx.weight = weight
        return tensor.view_as(tensor)

    @staticmethod
    def backward(ctx: Any, gradient: Tensor) -> tuple[Tensor, None]:
        return gradient * ctx.weight, None


def scaled(tensor: Tensor, weight: float) -> Tensor:
    """What the head reads of the network, the gradient that goes back into
    the network at ``weight`` of the head's own: the head learns as much
    whatever the weight, and at 0 the network is as without it."""
    return _Scaled.apply(tensor, weight)


def chosen_answers(action: Tensor, chosen: Tensor) -> Tensor:
    """Of each decision, the representation of the answer it chose."""
    return action[torch.arange(action.shape[0], device=action.device), chosen]


def card_loss(
    head: CardHead,
    state: Tensor,
    tokens: Tensor,
    answer: Tensor,
    labels: dict[str, Tensor],
) -> tuple[Tensor, dict[str, float]] | None:
    """The head's loss over the decisions that have labels: the places by
    cross-entropy, the changes by squared error, each the mean over what it
    is of, and their sum; with what each came to.  None where no decision
    has one."""
    taught = labels["labelled"].bool()
    if not bool(taught.any()):
        return None
    width = min(tokens.shape[1], labels["card_place"].shape[1])
    places, changes, life, counts = head(state[taught], tokens[taught][:, :width], answer[taught])
    place = labels["card_place"][taught][:, :width]
    known = place >= 0
    parts: dict[str, Tensor] = {}
    if bool(known.any()):
        parts["places"] = nn.functional.cross_entropy(places[known], place[known])
    change = labels["card_change"][taught][:, :width]
    finite = torch.isfinite(change)
    if bool(finite.any()):
        parts["changes"] = nn.functional.mse_loss(changes[finite], change[finite])
    parts["life"] = nn.functional.mse_loss(life, labels["life_change"][taught])
    parts["counts"] = nn.functional.mse_loss(counts, labels["count_change"][taught])
    total = torch.stack(list(parts.values())).sum()
    told = {f"card_head_{name}_loss": float(value.detach()) for name, value in parts.items()}
    told["card_head_decisions"] = float(taught.sum())
    return total, told


def labels_of(batch: dict[str, Any]) -> dict[str, Tensor] | None:
    """The labels a minibatch carries beside the model's input, if any."""
    if f"{PREFIX}labelled" not in batch:
        return None
    return {name: batch[f"{PREFIX}{name}"] for name in LABELS}
