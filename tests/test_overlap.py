"""The next update's games played beside an update's pass (`train --overlap`,
experiments/2026-10-23-overlap.md): by a second thread they are the updates
the same games played in turn before the pass give, to the bit, and the
weights they are played with are those from before the pass."""

from __future__ import annotations

from pathlib import Path

import pytest

from ygo_training.checkpoints import value_fingerprint
from ygo_training.config import TrainerConfig
from ygo_training.environment import library_path
from ygo_training.ppo_trainer import PpoTrainer

ROOT = Path(__file__).resolve().parents[1]


def _built() -> bool:
    try:
        library_path(ROOT)
    except FileNotFoundError:
        return False
    return True


def _config(overlap: str) -> TrainerConfig:
    return TrainerConfig(
        workers=2,
        duels=4,
        duels_in_flight=8,
        rollout_estimates="kept",
        pool_threads=2,
        pool_slots=4,
        pool_groups=2,
        device="cpu",
        seed=45,
        learner_deck="blackwing",
        learner_decks=["blackwing", "lightsworn"],
        opponent_decks=["blackwing", "lightsworn"],
        self_play_duels=2,
        minibatch=64,
        overlap=overlap,
    )


# What of an update must not hang on who played its games, or when.
SAME = (
    "rollout_decision_fingerprint",
    "rollout_training_fingerprint",
    "duels",
    "ppo_loss",
    "duel_approximate_kl",
    "duel_clip_fraction",
)


def _updates(overlap: str, count: int) -> tuple[list[dict[str, float]], str]:
    import torch

    torch.set_num_threads(2)
    with PpoTrainer(ROOT, _config(overlap)) as trainer:
        updates = [
            trainer.train_update(None, step, ahead=step < count) for step in range(1, count + 1)
        ]
        weights = value_fingerprint(
            {name: tensor.tolist() for name, tensor in trainer.model.state_dict().items()}
        )
    return updates, weights


@pytest.mark.integration
@pytest.mark.skipif(not _built(), reason="the environment is not built: cargo build --release")
def test_the_games_beside_the_pass_are_those_played_in_turn_before_it():
    beside, beside_weights = _updates("thread", 4)
    in_turn, in_turn_weights = _updates("in-turn", 4)
    for one, other in zip(beside, in_turn, strict=True):
        assert {name: one.get(name) for name in SAME} == {name: other.get(name) for name in SAME}
    assert beside_weights == in_turn_weights
    # Played beside the pass, the next update's games wait for nothing of it.
    assert all("overlap_wait_seconds" in update for update in beside[1:])
    assert all(update["overlap_lag"] == 1.0 for update in beside[1:])
    assert "overlap_wait_seconds" not in beside[0]


@pytest.mark.integration
@pytest.mark.skipif(not _built(), reason="the environment is not built: cargo build --release")
def test_the_games_beside_the_pass_are_played_by_the_weights_from_before_it():
    beside, _ = _updates("in-turn", 3)
    after, _ = _updates("", 3)
    # The first update's games are the same whoever plays the next.
    assert beside[0]["rollout_training_fingerprint"] == after[0]["rollout_training_fingerprint"]
    # The second's are played by the weights from before the first pass,
    # where without the overlap they are played by those it left.
    assert beside[1]["rollout_training_fingerprint"] != after[1]["rollout_training_fingerprint"]


def test_the_games_beside_the_pass_go_with_duels_in_flight_alone():
    with pytest.raises(ValueError, match="overlap must be one of"):
        TrainerConfig(overlap="beside")
    with pytest.raises(ValueError, match="are duels in flight"):
        TrainerConfig(overlap="thread")
    in_flight = {"duels": 4, "duels_in_flight": 8, "rollout_estimates": "kept", "pool_slots": 16}
    with pytest.raises(ValueError, match="of updates of PPO"):
        TrainerConfig(overlap="thread", imitation_only_updates=2, **in_flight)
    with pytest.raises(ValueError, match="a copy of the model"):
        TrainerConfig(overlap="thread", inference_graphs=8, **in_flight)
    assert TrainerConfig(overlap="in-turn", **in_flight).overlap == "in-turn"
