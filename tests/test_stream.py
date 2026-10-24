"""Duels kept in flight from one update to the next (`pool.Stream`, `train
--duels-in-flight`): by the same weights they are the duels the same games
played to their end are, an update is the next duels to end, and what an
update has trained on is let go of."""

from __future__ import annotations

from pathlib import Path

import pytest

from ygo_training.config import TrainerConfig
from ygo_training.environment import library_path
from ygo_training.pool import DuelPool, Stream, sampled_games
from ygo_training.ppo_trainer import PpoTrainer
from ygo_training.schedules import rollout_games

ROOT = Path(__file__).resolve().parents[1]


def _built() -> bool:
    try:
        library_path(ROOT)
    except FileNotFoundError:
        return False
    return True


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _built(), reason="the environment is not built: cargo build --release"),
]

OPPONENTS = ["monarch", "heroes", "burn", "gladiator", "lightsworn", "blackwing"]
# Rollouts of six games each, as the updates of a run would have them.
GAMES = 6


def block(index: int) -> tuple[list[dict], int]:
    seed = (40 + index) * 1024
    return rollout_games(seed, 2, [OPPONENTS[(game + index) % 6] for game in range(GAMES)]), seed


@pytest.fixture(scope="module")
def model():
    import torch

    torch.set_num_threads(2)
    from ygo_training.model import PolicyValueNet

    torch.manual_seed(45)
    return PolicyValueNet().eval()


def played(trajectory: dict) -> tuple:
    """A duel as it was played: its game and every answer of it."""
    return (
        trajectory["episode_index"],
        trajectory["worker"],
        trajectory["learner_player"],
        trajectory["opponent_deck"],
        trajectory["winner"],
        trajectory["turns"],
        tuple(
            (step["chosen"], step["player"], step["transaction"]) for step in trajectory["steps"]
        ),
    )


def streamed(model, updates: int) -> tuple[list[list[dict]], list[dict]]:
    """The updates a stream of two rollouts in flight hands over, the same
    weights answering every one, and what it held after each."""
    handed, held = [], []
    with DuelPool(ROOT, threads=2, slots=6, groups=2) as pool:
        stream = Stream(pool, "cpu", learner="blackwing", limit=4096, temperature=1.0)
        for index in range(updates):
            trajectories, asked = stream.play(model, block, index, 1, GAMES)
            handed.append(trajectories)
            stream.trained()
            kept = sum(len(rows) for rows in stream.rows.values())
            held.append(
                {"asked": asked, "kept": pool.kept, "rows": kept, "games": len(stream.games)}
            )
    return handed, held


def test_duels_kept_in_flight_are_the_games_played_to_their_end(model):
    handed, held = streamed(model, 3)
    assert [len(trajectories) for trajectories in handed] == [GAMES] * 3
    # Each update hands over duels that ended, all of them as they ended.
    for trajectories in handed:
        assert all(trajectory["steps"] for trajectory in trajectories)
    # What ended and was trained on is let go of; what is in flight, or ended
    # and not yet handed over, is kept.
    for after in held:
        assert after["kept"] == after["rows"] and after["games"] > 0
    assert held[-1]["asked"]["stream_in_flight"] > 0
    assert held[-1]["asked"]["stream_oldest"] >= 1

    # The same weights, the same games played to their end: the same duels,
    # but where a draw falls on the last digit a batch of other decisions
    # leaves on a probability.
    alone = []
    for index in range(4):
        games, seed = block(index)
        with DuelPool(ROOT, threads=2, slots=6) as pool:
            trajectories, _ = sampled_games(
                pool,
                model,
                "cpu",
                learner="blackwing",
                games=games,
                limit=4096,
                rollout_seed=seed,
                temperature=1.0,
            )
        alone.extend(played(trajectory) for trajectory in trajectories)
    ours = [played(trajectory) for trajectories in handed for trajectory in trajectories]
    assert len(set(ours)) == len(ours) == 3 * GAMES
    assert sum(duel not in alone for duel in ours) <= 2


def test_a_stream_hands_over_the_same_updates_every_time(model):
    once, _ = streamed(model, 2)
    again, _ = streamed(model, 2)
    assert once == again


def test_duels_in_flight_are_asked_for_what_they_need():
    with pytest.raises(ValueError, match="a multiple of duels"):
        TrainerConfig(duels=4, duels_in_flight=6, rollout_estimates="kept", pool_slots=16)
    with pytest.raises(ValueError, match="twice them at least"):
        TrainerConfig(duels=4, duels_in_flight=4, rollout_estimates="kept", pool_slots=16)
    with pytest.raises(ValueError, match="rollout_estimates kept"):
        TrainerConfig(duels=4, duels_in_flight=8, rollout_estimates="recomputed", pool_slots=16)
    assert TrainerConfig(duels=4, duels_in_flight=8, pool_slots=16).rollout_estimates == "kept"
    with pytest.raises(ValueError, match="as many slots"):
        TrainerConfig(duels=4, duels_in_flight=8, rollout_estimates="kept", pool_slots=4)
    with pytest.raises(ValueError, match="rollout_estimates must be one of"):
        TrainerConfig(rollout_estimates="fresh")


def test_an_update_of_duels_in_flight_trains_on_the_next_to_end():
    import torch

    torch.set_num_threads(2)
    config = TrainerConfig(
        workers=2,
        duels=4,
        duels_in_flight=8,
        rollout_estimates="kept",
        pool_threads=2,
        pool_slots=4,
        device="cpu",
        seed=45,
        learner_deck="blackwing",
        learner_decks=["blackwing", "lightsworn"],
        opponent_decks=["blackwing", "lightsworn"],
        self_play_duels=2,
        minibatch=64,
    )
    with PpoTrainer(ROOT, config) as trainer:
        first = trainer.train_update(None, 1)
        second = trainer.train_update(None, 2)
        assert first["duels"] == second["duels"] == 4.0
        assert first["recomputed_rollout_estimates"] == 0.0
        assert second["stream_in_flight"] > 0
        with pytest.raises(ValueError, match="not evaluated as it goes"):
            trainer.evaluate_games(
                "blackwing", [{"opponent": "monarch", "learner_player": 0, "seed": 1}]
            )
