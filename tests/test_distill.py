"""The old decks held to a frozen network's answers in PPO's updates
(experiments/2026-11-02-expanded-pool.md, the batch again in two arms)."""

from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")
from tests.test_cli import parser
from tests.views import SHOWN, step
from ygo_training.__main__ import trainer_config
from ygo_training.config import TrainerConfig
from ygo_training.model import PolicyValueNet
from ygo_training.optimization import PpoOptimizer
from ygo_training.rewards import OutcomeRewardModel

DECKS = ["fresh", "old", "older"]
HELD = ["old", "older"]


def config(**changed) -> TrainerConfig:
    settings = {
        "learner_deck": "fresh",
        "learner_decks": DECKS,
        "opponent_decks": ["burn"],
        "distill_reference": "frozen.pt",
        "distill_weight": 1.0,
        "distill_decks": HELD,
        "ppo_epochs": 1,
        "minibatch": 64,
        **changed,
    }
    return TrainerConfig(**settings)


def optimizer(model, reference, lr: float = 1e-3, second=None, **changed) -> PpoOptimizer:
    if second is not None:
        changed = {
            "distill_second_reference": "champions.pt",
            "distill_second_decks": ["old"],
            **changed,
        }
    made = PpoOptimizer(
        model,
        torch.optim.Adam(model.parameters(), lr=lr),
        config(**changed),
        OutcomeRewardModel(),
        "cpu",
    )
    # The steps' decisions are kept where the tests' views are.
    made.kept = SHOWN
    made.reference = reference
    made.second_reference = second
    return made


def shaken(model) -> PolicyValueNet:
    """A copy of the model whose answers are not its own."""
    other = copy.deepcopy(model)
    with torch.no_grad():
        for parameter in other.parameters():
            parameter.add_(0.5 * torch.randn_like(parameter))
    return other


def duel(one: str, two: str, decisions: int = 6) -> dict:
    """A duel of the network against itself, ``one`` in seat 0 and ``two`` in seat 1."""
    return {
        "terminal": True,
        "learner_player": 0,
        "winner": 0,
        "other": "self",
        "learner_deck": one,
        "opponent_deck": two,
        "steps": [step(chosen=index % 2, player=index % 2) for index in range(decisions)],
    }


@pytest.fixture
def model() -> PolicyValueNet:
    torch.manual_seed(2012)
    return PolicyValueNet(embedding_dim=8, hidden_dim=16)


def test_the_decisions_of_a_duel_between_two_held_decks_are_held(model):
    held = optimizer(model, copy.deepcopy(model))
    assert held._distilled(duel("old", "older")) and held._distilled(duel("older", "older"))
    assert not held._distilled(duel("fresh", "old")) and not held._distilled(duel("old", "fresh"))
    # Where no reference is given, nothing is held.
    assert not optimizer(model, None)._distilled(duel("old", "older"))


def test_against_its_own_copy_a_held_decision_diverges_by_nothing(model):
    metrics = optimizer(model, copy.deepcopy(model)).update(
        [duel("old", "older"), duel("fresh", "old")]
    )
    assert metrics["distill_decisions"] == 6.0
    assert metrics["distill_divergence"] == pytest.approx(0.0, abs=1e-6)
    # The term is the reference's own cross-entropy then, summed over the held
    # decisions, over the minibatch's transactions.
    assert metrics["distill_loss"] > 0.0


def test_a_duel_with_a_fresh_deck_adds_nothing(model):
    metrics = optimizer(model, copy.deepcopy(model)).update([duel("fresh", "old")])
    assert metrics["distill_decisions"] == 0.0 and metrics["distill_loss"] == 0.0


def test_the_term_brings_the_model_back_towards_the_reference(model):
    reference = copy.deepcopy(model)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.5 * torch.randn_like(parameter))
    held = optimizer(model, reference, lr=0.01, distill_weight=10.0, entropy_coefficient=0.0)
    first = held.update([duel("old", "older")])["distill_divergence"]
    for _ in range(30):
        last = held.update([duel("old", "older")])["distill_divergence"]
    assert first > 0.0 and last < first / 2


def test_its_settings_go_together(model):
    with pytest.raises(ValueError, match="go together"):
        config(distill_reference=None)
    with pytest.raises(ValueError, match="go together"):
        config(distill_weight=0.0)
    with pytest.raises(ValueError, match="go together"):
        config(distill_decks=None)
    with pytest.raises(ValueError, match="non-negative"):
        config(distill_weight=-1.0)
    with pytest.raises(ValueError, match="among the learner's decks"):
        config(distill_decks=["old", "burn"])
    assert (
        config(distill_reference=None, distill_weight=0.0, distill_decks=None).distill_weight == 0.0
    )


def test_the_command_line_names_them():
    args = parser().parse_args(
        [
            "train",
            "--learner",
            "fresh,old,older",
            "--opponents",
            "burn",
            "--distill-reference",
            "frozen.pt",
            "--distill-weight",
            "1",
            "--distill-decks",
            "old,older",
        ]
    )
    made = trainer_config(args)
    assert (made.distill_reference, made.distill_weight, made.distill_decks) == (
        "frozen.pt",
        1.0,
        ["old", "older"],
    )
    assert made.distill_second_reference is None and made.distill_second_decks is None


# Batch four: the seventy-one held to seventy-one-champions in their duels with
# each other, the other old decks to the start.


def test_a_duel_between_two_of_the_narrower_decks_is_held_to_the_second_network(model):
    held = optimizer(model, copy.deepcopy(model), second=copy.deepcopy(model))
    assert held._held_to(duel("old", "old")) == 1
    assert held._held_to(duel("old", "older")) == 0
    assert held._held_to(duel("older", "older")) == 0
    assert held._held_to(duel("fresh", "old")) is None
    # Without a second network, they are all held to the first.
    assert optimizer(model, copy.deepcopy(model))._held_to(duel("old", "old")) == 0


def test_each_decision_is_asked_of_the_network_its_duel_is_held_to(model):
    # The first network answers as the model does, the second does not: only the
    # decisions of the duel between two of the narrower decks diverge.
    held = optimizer(model, copy.deepcopy(model), second=shaken(model))
    metrics = held.update([duel("old", "old", decisions=4), duel("old", "older")])
    assert metrics["distill_decisions"] == 10.0
    assert metrics["distill_second_decisions"] == 4.0
    assert metrics["distill_second_divergence"] > 0.0
    assert metrics["distill_divergence"] == pytest.approx(
        metrics["distill_second_divergence"] * 4 / 10
    )
    # The other way round, only the duel held to the first network diverges.
    held = optimizer(model, shaken(model), second=copy.deepcopy(model))
    metrics = held.update([duel("old", "old", decisions=4), duel("old", "older")])
    assert metrics["distill_second_divergence"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["distill_divergence"] > 0.0


def test_without_a_second_network_the_term_is_as_it_was(model):
    reference = shaken(model)
    one = optimizer(copy.deepcopy(model), reference).update([duel("old", "older")])
    two = optimizer(copy.deepcopy(model), reference, second=copy.deepcopy(model)).update(
        [duel("old", "older")]
    )
    assert "distill_second_decisions" not in one
    assert two["distill_second_decisions"] == 0.0
    for name in ("distill_loss", "distill_divergence", "distill_decisions"):
        assert two[name] == one[name]


def test_the_term_brings_the_narrower_decks_towards_the_second_network(model):
    second = copy.deepcopy(model)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.5 * torch.randn_like(parameter))
    held = optimizer(
        model,
        shaken(model),
        lr=0.01,
        second=second,
        distill_weight=10.0,
        entropy_coefficient=0.0,
    )
    first = held.update([duel("old", "old")])["distill_second_divergence"]
    for _ in range(30):
        last = held.update([duel("old", "old")])["distill_second_divergence"]
    assert first > 0.0 and last < first / 2


def test_the_second_network_goes_with_its_decks_among_the_held_ones():
    with pytest.raises(ValueError, match="go together"):
        config(distill_second_reference="champions.pt")
    with pytest.raises(ValueError, match="go together"):
        config(distill_second_decks=["old"])
    with pytest.raises(ValueError, match="among the distill_decks"):
        config(distill_second_reference="champions.pt", distill_second_decks=["fresh"])
    made = config(distill_second_reference="champions.pt", distill_second_decks=["old"])
    assert made.distill_second_decks == ["old"]


def test_the_command_line_names_the_second_network():
    args = parser().parse_args(
        [
            "train",
            "--learner",
            "fresh,old,older",
            "--opponents",
            "burn",
            "--distill-reference",
            "frozen.pt",
            "--distill-weight",
            "1",
            "--distill-decks",
            "old,older",
            "--distill-second-reference",
            "champions.pt",
            "--distill-second-decks",
            "old",
        ]
    )
    made = trainer_config(args)
    assert (made.distill_second_reference, made.distill_second_decks) == ("champions.pt", ["old"])
