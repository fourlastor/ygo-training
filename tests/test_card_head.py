"""`ygo_training/card_head.py`: a head that learns how cards work, taught in
PPO's own minibatches from the labels the pool keeps, and kept beside each
checkpoint."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

from ygo_training.card_head import COUNTED, PLACES, CardHead, card_loss, chosen_answers
from ygo_training.checkpoints import value_fingerprint
from ygo_training.config import TrainerConfig
from ygo_training.environment import library_path
from ygo_training.model import PolicyValueNet
from ygo_training.ppo_trainer import PpoTrainer

ROOT = Path(__file__).resolve().parents[1]


def _built() -> bool:
    try:
        library_path(ROOT)
    except FileNotFoundError:
        return False
    return True


built = pytest.mark.skipif(
    not _built(), reason="the environment is not built: cargo build --release"
)


def labels(rows: int, width: int) -> dict[str, torch.Tensor]:
    """Labels of `rows` decisions of `width` cards, every other one labelled,
    its last card lost and its changes known for its first card alone."""
    place = torch.randint(0, PLACES, (rows, width))
    place[:, -1] = -1
    change = torch.full((rows, width, 2), float("nan"))
    change[:, 0] = 0.25
    return {
        "labelled": torch.arange(rows) % 2 == 0,
        "card_place": place,
        "card_change": change,
        "life_change": torch.zeros(rows, 2),
        "count_change": torch.zeros(rows, 2 * COUNTED),
    }


def test_the_head_is_taught_where_there_are_labels_and_only_there():
    torch.manual_seed(3)
    policy = PolicyValueNet(embedding_dim=8, hidden_dim=16)
    head = CardHead(policy)
    rows, width = 6, 5
    state = torch.randn(rows, policy.hidden_dim, requires_grad=True)
    tokens = torch.randn(rows, width + 2, policy.card_representation_dim)
    action = torch.randn(rows, 3, policy.hidden_dim)
    answer = chosen_answers(action, torch.tensor([0, 1, 2, 0, 1, 2]))
    assert torch.equal(answer[4], action[4, 1])
    places, changes, life, counts = head(state, tokens, answer)
    assert places.shape == (rows, width + 2, PLACES) and changes.shape == (rows, width + 2, 2)
    assert life.shape == (rows, 2) and counts.shape == (rows, 2 * COUNTED)
    told = labels(rows, width)
    found = card_loss(head, state, tokens, answer, told)
    assert found is not None
    loss, parts = found
    assert parts["card_head_decisions"] == 3
    assert set(parts) >= {"card_head_places_loss", "card_head_changes_loss"}
    loss.backward()
    # The gradient reaches what the head reads, of the labelled decisions alone.
    assert state.grad is not None
    assert state.grad[1::2].abs().sum() == 0 and state.grad[::2].abs().sum() > 0
    nothing = {**told, "labelled": torch.zeros(rows, dtype=torch.bool)}
    assert card_loss(head, state, tokens, answer, nothing) is None


def test_the_parts_of_the_network_give_its_answers_as_they_are():
    torch.manual_seed(4)
    policy = PolicyValueNet(embedding_dim=8, hidden_dim=16)
    from test_critic import decisions

    seat, _ = decisions()
    logits, value = policy(seat)
    same, again, state, tokens, action = policy.parts(seat)
    assert torch.equal(logits, same) and torch.equal(value, again)
    assert state.shape == (len(value), policy.hidden_dim)
    assert tokens.shape[:2] == seat["state_card"].shape
    assert action.shape[:2] == seat["action_mask"].shape


def config(**more: Any) -> TrainerConfig:
    """A small run whose games the model plays, against pilots and itself."""
    return TrainerConfig(
        workers=2,
        duels=4,
        device="cpu",
        seed=23,
        opponent_decks=["monarch", "gishki"],
        self_play_duels=2,
        pool_threads=2,
        pool_slots=2,
        minibatch=32,
        ppo_epochs=1,
        **more,
    )


@built
@pytest.mark.integration
def test_an_update_with_the_head_at_no_weight_is_the_update_without_it():
    """The head reads the network's own pass and leaves the games alone: at
    weight 0 the network is changed as without it, bit for bit, and the head
    is taught all the same; with a weight its gradient reaches the network."""
    torch.set_num_threads(2)
    with (
        PpoTrainer(ROOT, config()) as plain,
        PpoTrainer(ROOT, config(card_head=True, card_head_weight=0.0)) as weightless,
        PpoTrainer(ROOT, config(card_head=True, card_head_weight=1.0)) as weighted,
    ):
        for trainer in (plain, weightless, weighted):
            trainer.begin_card_head()
        assert plain.card_head is None and weightless.card_head is not None
        begun = [parameter.detach().clone() for parameter in weightless.card_head.parameters()]
        first = plain.train_update()
        second = weightless.train_update()
        third = weighted.train_update()
        assert first["rollout_decision_fingerprint"] == second["rollout_decision_fingerprint"]
        assert first["rollout_decision_fingerprint"] == third["rollout_decision_fingerprint"]
        assert value_fingerprint(plain.model.state_dict()) == value_fingerprint(
            weightless.model.state_dict()
        )
        assert value_fingerprint(plain.model.state_dict()) != value_fingerprint(
            weighted.model.state_dict()
        )
        assert "card_head_places_loss" not in first
        assert second["card_head_places_loss"] > 0 and second["card_head_decisions"] > 10
        assert any(
            not torch.equal(before, after)
            for before, after in zip(begun, weightless.card_head.parameters(), strict=True)
        )


@built
@pytest.mark.integration
def test_a_run_with_the_head_keeps_it_beside_its_checkpoints_and_goes_on_from_it(tmp_path):
    run = tmp_path / "run"
    common = [
        "--device", "cpu", "--workers", "2", "--pool-threads", "2", "--pool-slots", "2",
        "--duels", "2", "--ppo-epochs", "1", "--minibatch", "64", "--seed", "46",
        "--opponents", "monarch", "--eval-every", "0", "--run", str(run),
        "--card-head", "--card-head-weight", "0.5",
    ]  # fmt: skip

    def train(*more: str) -> None:
        subprocess.run(
            [sys.executable, "-m", "ygo_training", "train", *common, *more],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=600,
            check=True,
        )

    train("--updates", "2")
    checkpoints = run / "checkpoints"
    assert sorted(path.name for path in checkpoints.glob("*.pt")) == [
        "card-head-00000.pt",
        "card-head-00001.pt",
        "model-00000.pt",
        "model-00001.pt",
    ]
    held = torch.load(checkpoints / "model-00001.pt", map_location="cpu", weights_only=False)
    assert not any(name.startswith(("card.", "board.")) for name in held["model"])
    assert json.loads((run / "run.json").read_text())["card_head"] == {
        "weight": 0.5,
        "learning_rate": 3e-4,
        "begun": None,
    }
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert all(row["card_head_decisions"] > 0 for row in rows)
    train("--continue", "--updates", "1")
    saved = torch.load(checkpoints / "card-head-00002.pt", map_location="cpu", weights_only=False)
    assert saved["step"] == 3 and saved["optimizer"]["state"]
