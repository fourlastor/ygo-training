"""A deck list read from a file in the learner's seat (`evaluate-suite
--learner-list`): the Blackwing list read from its own file plays the games
of `blackwing`, and a list no pilot was written for is played by the pilot
of the learner's deck, or by the model with that pilot alongside."""

from __future__ import annotations

from pathlib import Path

import pytest

from ygo_training.environment import library_path, pool
from ygo_training.orchestration import evaluation_assignments
from ygo_training.pool import DuelPool, evaluate_games

ROOT = Path(__file__).resolve().parents[1]
VARIANT = ROOT / "data/decks/darker-than-black.ydk"


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


@pytest.fixture(scope="module")
def model():
    import torch

    torch.set_num_threads(2)
    from ygo_training.model import PolicyValueNet

    torch.manual_seed(45)
    return PolicyValueNet().eval()


def games(learner_list: Path | None = None, model=None) -> list[dict]:
    """Blackwing's games against Monarch and itself, three each, by its pilot
    or, given a model, by the model."""
    with DuelPool(ROOT, threads=2, slots=6) as duels:
        return evaluate_games(
            duels,
            model,
            "cpu",
            learner="blackwing",
            assignments=evaluation_assignments(["monarch", "blackwing"], 3, 2012),
            limit=4096,
            pilot=model is None,
            learner_list=None if learner_list is None else str(learner_list),
        )


def test_the_blackwing_list_from_its_file_plays_the_games_of_blackwing():
    own = ROOT / "native/ygo-policies/decks" / f"{pool(ROOT)['blackwing']}.ydk"
    assert games(own) == games()


def test_a_list_no_pilot_was_written_for_is_played_by_its_decks_pilot(model):
    played = games(VARIANT)
    assert played != games()
    assert len(played) == 6 and all(game["terminal"] for game in played)
    # The model in the seat, Blackwing's pilot asked alongside it.
    asked = games(VARIANT, model)
    assert len(asked) == 6
    assert sum(game["pilot_labels"] for game in asked) > 0
