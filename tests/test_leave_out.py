"""A model that leaves parts of its input out (`train --leave-out`): the same
network, given nothing where a part was, so that a copy that reads less
begins as the same network as one that reads everything and is taught on
the very games it is (experiments/2026-10-15-without-deck-lists.md).
Since format 15 the seat's own earlier answers are the one part: the deck
lists it was made for are out of the input."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest
import torch

from tests.views import ACTIVATE, MONSTER, PASS, batched, choice, named, view
from ygo_training.checkpoints import load, load_inference_model, save, value_fingerprint
from ygo_training.cli_parser import _parts_left_out
from ygo_training.config import LEAVE_OUT, TrainerConfig
from ygo_training.model import PolicyValueNet
from ygo_training.ppo_trainer import PpoTrainer
from ygo_training.schema import model_policy_input_schema

ROOT = Path(__file__).resolve().parents[1]
# A seat's earlier answers: none, and two.
FRESH = view()
ANSWERED = view(
    history=[
        {"code": MONSTER, "location": 2, "kind": ACTIVATE, "description": 0},
        {"code": 0, "location": 0, "kind": PASS, "description": 0},
    ],
    choices=[choice(ACTIVATE, named(-1, MONSTER)), choice(PASS)],
)


def answered(model: PolicyValueNet, shown: dict) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        return model(batched(shown))


def test_every_model_begins_as_the_same_network():
    begun = set()
    for parts in ((), *((part,) for part in LEAVE_OUT), LEAVE_OUT):
        torch.manual_seed(3)
        begun.add(value_fingerprint(PolicyValueNet(leave_out=parts).state_dict()))
    assert len(begun) == 1


@pytest.mark.parametrize("parts", [(), ("history",)])
def test_a_part_left_out_does_not_change_the_answers(parts):
    torch.manual_seed(3)
    model = PolicyValueNet(leave_out=parts).eval()
    logits, value = answered(model, ANSWERED)
    changed, worth = answered(model, ANSWERED | {"history": []})
    unchanged = torch.equal(changed, logits) and torch.equal(worth, value)
    assert unchanged is ("history" in parts)
    assert answered(model, FRESH)[0].shape == (1, 2)


def test_a_model_that_reads_everything_answers_as_before():
    torch.manual_seed(3)
    model = PolicyValueNet().eval()
    assert model.leave_out == ()
    batch = batched(ANSWERED)
    with torch.no_grad():
        state, _, _ = model.encode_state(batch)
        expected = model.state(
            torch.cat(
                (
                    model._masked_mean(
                        model._placed_cards(batch, "state_"),
                        model._present(batch["state_lengths"], batch["state_card"].shape[1]),
                    ),
                    model.phase_embedding(batch["phase"]),
                    model.decision_kind_embedding(batch["decision_kind"]),
                    model.hint_embedding(batch["decision_hint"]),
                    batch["scalars"],
                    (
                        model.context_encoder(model._placed_cards(batch, "context_"))
                        * batch["context_mask"].unsqueeze(-1)
                    ).flatten(1),
                    model._history(batch),
                ),
                dim=-1,
            )
        )
    assert torch.equal(state, expected)


def test_a_checkpoint_keeps_what_its_model_leaves_out(tmp_path):
    torch.manual_seed(3)
    model = PolicyValueNet(leave_out=("history",))
    path = tmp_path / "model-00001.pt"
    save(path, model, torch.optim.Adam(model.parameters()), 1, {})
    # Loaded into a model built as ever, it reads what it was taught to.
    other = PolicyValueNet()
    payload = load(path, other)
    assert other.leave_out == ("history",)
    assert payload["metadata"]["leave_out"] == ["history"]
    other.eval()
    assert torch.equal(answered(other, ANSWERED)[0], answered(other, ANSWERED | {"history": []})[0])
    assert load_inference_model(path)[0].leave_out == ("history",)
    with pytest.raises(ValueError, match="leaves out history of its input, expected nothing"):
        load(path, PolicyValueNet(), expected_leave_out=())
    load(path, PolicyValueNet(), expected_leave_out=("history",))


def test_a_checkpoint_from_before_reads_everything(tmp_path):
    torch.manual_seed(3)
    model = PolicyValueNet()
    path = tmp_path / "model-00001.pt"
    save(path, model, torch.optim.Adam(model.parameters()), 1, {})
    payload = torch.load(path, weights_only=False)
    assert payload["metadata"]["leave_out"] == []
    del payload["metadata"]["leave_out"]
    torch.save(payload, path)
    other = PolicyValueNet(leave_out=("history",))
    load(path, other)
    assert other.leave_out == ()


def test_a_model_that_reads_everything_is_described_as_before():
    before = model_policy_input_schema(board_representation="mean-pool")
    assert "leave_out" not in before
    assert model_policy_input_schema(board_representation="mean-pool", leave_out=()) == before
    assert model_policy_input_schema(board_representation="mean-pool", leave_out=("history",))[
        "leave_out"
    ] == ["history"]


def test_only_parts_of_the_input_are_left_out():
    with pytest.raises(ValueError, match="leaves out only"):
        PolicyValueNet(leave_out=("opponent-deck",))
    with pytest.raises(ValueError, match="leave_out names each"):
        TrainerConfig(leave_out=("history", "history"))
    assert TrainerConfig(leave_out=("history",)).leave_out == ("history",)
    assert _parts_left_out(" history ") == ("history",)
    # The deck lists are no part of the input any more.
    with pytest.raises(argparse.ArgumentTypeError, match="name each of"):
        _parts_left_out("history, opponent-deck")


def test_a_checkpoint_that_leaves_parts_out_is_not_exported(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "export_checkpoint", ROOT / "tools/export_checkpoint.py"
    )
    assert spec is not None and spec.loader is not None
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    model = PolicyValueNet(leave_out=("history",))
    path = tmp_path / "model-00001.pt"
    save(path, model, torch.optim.Adam(model.parameters()), 1, {})
    with pytest.raises(ValueError, match="leaves out history of its input"):
        tool.export(path, tmp_path / "exported", tmp_path / "decisions.json")


def test_an_imitating_update_plays_the_same_games_whatever_the_model_reads():
    """Its pilots play them: a copy that reads less begins as the same
    network and is taught on the very games of one that reads everything."""
    torch.set_num_threads(2)

    def imitating(parts: tuple[str, ...]) -> TrainerConfig:
        return TrainerConfig(
            workers=3,
            duels=8,
            device="cpu",
            seed=71,
            opponent_decks=["monarch", "heroes", "burn", "gladiator", "lightsworn"],
            imitation_only_updates=3,
            imitation_value_weight=0.5,
            imitation_value_positions=4,
            ppo_epochs=1,
            minibatch=48,
            pool_threads=3,
            pool_slots=2,
            leave_out=parts,
        )

    with (
        PpoTrainer(ROOT, imitating(())) as everything,
        PpoTrainer(ROOT, imitating(("history",))) as less,
    ):
        assert less.model.leave_out == ("history",)
        assert less.initial_model_fingerprint == everything.initial_model_fingerprint
        ours, theirs = less.train_update(), everything.train_update()
        assert ours["rollout_decision_fingerprint"] == theirs["rollout_decision_fingerprint"]
        assert ours["rollout_training_fingerprint"] == theirs["rollout_training_fingerprint"]
        assert ours["leave_out"] == ["history"] and "leave_out" not in theirs
        # Taught on the same games, told less: another network after it.
        assert value_fingerprint(less.model.state_dict()) != value_fingerprint(
            everything.model.state_dict()
        )
