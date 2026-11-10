"""tools/switch_board.py: a network trained with its board pooled, given the
board as its cards with `board_fusion` at zero, answers as it did, and goes on
training in its new shape."""

import sys
from pathlib import Path

import pytest
import torch

from tests.test_packed import DECISIONS
from tests.views import PASS, batched, card, choice, hidden, view
from ygo_training.checkpoints import load, load_inference_model, save
from ygo_training.model import PolicyValueNet

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from switch_board import FUSION, switch

FIELD = [
    view(cards=[hidden(), card(attack=attack)], choices=[choice(PASS), choice(22)])
    for attack in (100, 3000)
]


def _pooled(tmp_path: Path) -> Path:
    torch.manual_seed(7)
    model = PolicyValueNet(board_representation="mean-pool")
    optimizer = torch.optim.Adam(model.parameters())
    # An update, so that the optimizer holds a state as a run's does.
    logits, values = model(batched(*DECISIONS))
    (logits[torch.isfinite(logits)].sum() + values.sum()).backward()
    optimizer.step()
    path = tmp_path / "pooled.pt"
    save(path, model, optimizer, 960, {})
    return path


def test_a_switched_network_answers_as_it_did(tmp_path):
    source = _pooled(tmp_path)
    switch(source, tmp_path / "switched.pt")
    pooled, _ = load_inference_model(source)
    tokens, payload = load_inference_model(tmp_path / "switched.pt")
    assert tokens.board_representation == "individual-tokens"
    assert payload["step"] == 960
    assert payload["metadata"]["switched_from"] == "pooled.pt"
    for decisions in (DECISIONS, FIELD):
        batch = batched(*decisions)
        with torch.no_grad():
            for theirs, ours in zip(pooled(batch), tokens(batch), strict=True):
                assert torch.equal(theirs, ours)
    # What it reads of the cards is there to be learned: moved from zero, the
    # fusion changes the answers where cards are on the field.
    with torch.no_grad():
        getattr(tokens, FUSION.removesuffix(".weight")).weight.normal_()
        batch = batched(*FIELD)
        assert not torch.equal(pooled(batch)[0], tokens(batch)[0])


def test_a_switched_network_goes_on_training_in_its_shape(tmp_path):
    source = _pooled(tmp_path)
    switch(source, tmp_path / "switched.pt")
    saved = torch.load(source, map_location="cpu", weights_only=False)["optimizer"]
    model = PolicyValueNet(board_representation="individual-tokens")
    optimizer = torch.optim.Adam(model.parameters())
    payload = load(
        tmp_path / "switched.pt",
        model,
        optimizer,
        expected_board_representation="individual-tokens",
    )
    assert payload["step"] == 960
    assert len(optimizer.state) == len(saved["state"]) > 0
    # A run of the pooled board does not take it.
    with pytest.raises(ValueError, match="Checkpoint board representation"):
        load(
            tmp_path / "switched.pt",
            PolicyValueNet(),
            expected_board_representation="mean-pool",
        )


def test_only_a_pooled_board_is_switched(tmp_path):
    switch(_pooled(tmp_path), tmp_path / "switched.pt")
    with pytest.raises(SystemExit, match="nothing to switch"):
        switch(tmp_path / "switched.pt", tmp_path / "again.pt")
