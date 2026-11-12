"""How wide the network is (`train --embedding-dim --hidden-dim`): a checkpoint of
other widths than every network had says so, and whatever plays it builds the
network it says; one of those widths is described as before."""

from __future__ import annotations

import pytest
import torch

from tests.test_cli import parser
from tests.test_leave_out import ANSWERED, answered
from ygo_training.__main__ import _told_as
from ygo_training.checkpoints import (
    WIDTHS,
    load,
    load_inference_model,
    network_widths,
    save,
    saved_network_widths,
)
from ygo_training.config import TrainerConfig
from ygo_training.model import PolicyValueNet

WIDE = {"embedding_dim": 192, "hidden_dim": 384}


def test_every_network_until_now_was_as_wide_as_the_model_builds_one():
    assert WIDTHS == {"embedding_dim": 96, "hidden_dim": 192}
    assert network_widths(PolicyValueNet()) == WIDTHS
    assert TrainerConfig().embedding_dim == 96 and TrainerConfig().hidden_dim == 192


def test_a_wide_checkpoint_says_how_wide_it_is_and_plays_as_saved(tmp_path):
    torch.manual_seed(5)
    model = PolicyValueNet(**WIDE).eval()
    path = tmp_path / "model-00001.pt"
    save(path, model, torch.optim.Adam(model.parameters()), 1, {})
    metadata = torch.load(path, weights_only=False)["metadata"]
    assert saved_network_widths(metadata) == WIDE
    loaded, _ = load_inference_model(path)
    assert network_widths(loaded) == WIDE
    assert torch.equal(answered(loaded, ANSWERED)[0], answered(model, ANSWERED)[0])


def test_a_network_of_the_widths_until_now_is_described_as_before(tmp_path):
    model = PolicyValueNet()
    path = tmp_path / "model-00001.pt"
    save(path, model, torch.optim.Adam(model.parameters()), 1, {})
    metadata = torch.load(path, weights_only=False)["metadata"]
    assert "embedding_dim" not in metadata and "hidden_dim" not in metadata
    assert saved_network_widths(metadata) == WIDTHS


def test_a_checkpoint_is_loaded_only_into_a_network_as_wide(tmp_path):
    model = PolicyValueNet(embedding_dim=48, hidden_dim=96)
    path = tmp_path / "model-00001.pt"
    save(path, model, torch.optim.Adam(model.parameters()), 1, {})
    with pytest.raises(ValueError, match="build it as the checkpoint says"):
        load(path, PolicyValueNet())
    load(path, PolicyValueNet(embedding_dim=48, hidden_dim=96))


def test_a_measure_builds_its_network_as_wide_as_the_checkpoint(tmp_path):
    model = PolicyValueNet(**WIDE)
    path = tmp_path / "model-00001.pt"
    save(path, model, torch.optim.Adam(model.parameters()), 1, {})
    assert _told_as([path]) == WIDE
    plain = tmp_path / "model-00002.pt"
    save(plain, PolicyValueNet(), torch.optim.Adam(PolicyValueNet().parameters()), 2, {})
    assert _told_as([plain]) == WIDTHS


def test_training_is_told_its_widths():
    args = parser().parse_args(["train", "--embedding-dim", "48", "--hidden-dim", "96"])
    assert (args.embedding_dim, args.hidden_dim) == (48, 96)
    plain = parser().parse_args(["train"])
    assert (plain.embedding_dim, plain.hidden_dim) == (96, 192)


@pytest.mark.parametrize("widths", [{"embedding_dim": 1}, {"hidden_dim": 0}])
def test_a_network_is_two_wide_in_its_cards_and_one_in_a_decision_at_least(widths):
    with pytest.raises(ValueError, match="embedding_dim is 2 at least"):
        TrainerConfig(**widths)
