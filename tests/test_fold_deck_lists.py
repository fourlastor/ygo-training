"""`tools/fold_deck_lists.py`: a checkpoint of format 14 as one of format 15,
its two deck lists' vectors folded into the state's first layer."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from ygo_training import checkpoints
from ygo_training.model import PolicyValueNet

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
spec = importlib.util.spec_from_file_location("fold_deck_lists", ROOT / "tools/fold_deck_lists.py")
tool = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(tool)


def old_state(seed: int = 5) -> dict[str, torch.Tensor]:
    """A model of today's with what format 14 had more: the lists' encoder,
    and the two lists' columns in the state's first layer after the board's."""
    torch.manual_seed(seed)
    state = PolicyValueNet().state_dict()
    card = state["placed_card_encoder.0.weight"].shape[0]
    width = state["card_identity.low.weight"].shape[1] * 2
    weight = state["state.0.weight"]
    lists = torch.randn(weight.shape[0], 2 * width) * 0.05
    old = dict(state)
    old["state.0.weight"] = torch.cat((weight[:, :card], lists, weight[:, card:]), dim=1)
    old["deck_card_encoder.0.weight"] = torch.randn(width, card) * 0.05
    old["deck_card_encoder.0.bias"] = torch.randn(width) * 0.05
    return old


def test_the_folded_layer_gives_what_the_old_one_did_with_the_two_vectors():
    old = old_state()
    width = old["deck_card_encoder.0.weight"].shape[0]
    card = old["placed_card_encoder.0.weight"].shape[0]
    own, opponent = torch.rand(width), torch.rand(width)
    folded = tool.fold(old, own, opponent)
    assert not any(name.startswith("deck_card_encoder.") for name in folded)
    rest = old["state.0.weight"].shape[1] - card - 2 * width
    inputs = torch.randn(7, card + rest)
    before = torch.nn.functional.linear(
        torch.cat(
            (inputs[:, :card], own.expand(7, -1), opponent.expand(7, -1), inputs[:, card:]), dim=1
        ),
        old["state.0.weight"],
        old["state.0.bias"],
    )
    after = torch.nn.functional.linear(inputs, folded["state.0.weight"], folded["state.0.bias"])
    assert torch.allclose(after, before, atol=1e-5)
    model = PolicyValueNet()
    model.load_state_dict(folded)


def test_a_converted_checkpoint_is_written_and_read_by_todays_code(tmp_path):
    old = old_state()
    width = old["deck_card_encoder.0.weight"].shape[0]
    own, opponent = torch.rand(width), torch.rand(width)
    payload = {
        "format_version": 14,
        "step": 2160,
        "model": old,
        "metadata": {"board_representation": "mean-pool", "duels": 520.0},
    }
    told = {"own-deck": own, "opponent-deck": opponent, "leave_out": []}
    source = {"checkpoint": "old.pt", "own": "blackwing", "opponent": "blackwing"}
    model, metadata = tool.converted(payload, told, source)
    assert metadata["folded_deck_lists"]["own"] == "blackwing"
    assert metadata["folded_deck_lists"]["format_version"] == 14
    path = tmp_path / "model-02159.pt"
    checkpoints.save(path, model, torch.optim.Adam(model.parameters()), 2160, metadata)
    mirror = {("blackwing", "blackwing")}
    loaded, written = checkpoints.load_inference_model(path, pairings=mirror)
    assert written["format_version"] == checkpoints.FORMAT_VERSION
    assert written["step"] == 2160
    assert checkpoints.folded_pairing(written["metadata"]) == ("blackwing", "blackwing")
    for name, value in tool.fold(old, own, opponent).items():
        assert torch.equal(loaded.state_dict()[name], value), name


@pytest.mark.parametrize(
    "pairings",
    [None, {("blackwing", "monarch")}, {("blackwing", "blackwing"), ("monarch", "blackwing")}],
)
def test_a_converted_checkpoint_plays_its_pairing_alone(tmp_path, pairings):
    old = old_state()
    width = old["deck_card_encoder.0.weight"].shape[0]
    payload = {"format_version": 14, "step": 1, "model": old, "metadata": {}}
    payload["metadata"]["board_representation"] = "mean-pool"
    told = {"own-deck": torch.rand(width), "opponent-deck": torch.rand(width), "leave_out": []}
    source = {"checkpoint": "old.pt", "own": "blackwing", "opponent": "blackwing"}
    model, metadata = tool.converted(payload, told, source)
    path = tmp_path / "model-00000.pt"
    checkpoints.save(path, model, torch.optim.Adam(model.parameters()), 1, metadata)
    with pytest.raises(ValueError, match="folded for blackwing against blackwing"):
        checkpoints.load_inference_model(path, pairings=pairings)


def test_a_checkpoint_not_folded_loads_wherever_it_plays(tmp_path):
    torch.manual_seed(5)
    model = PolicyValueNet()
    path = tmp_path / "model-00000.pt"
    checkpoints.save(path, model, torch.optim.Adam(model.parameters()), 1, {})
    for pairings in (None, {("blackwing", "monarch"), ("monarch", "blackwing")}):
        checkpoints.load_inference_model(path, pairings=pairings)


def test_a_list_left_out_is_folded_as_the_nothing_it_was_given():
    old = old_state()
    width = old["deck_card_encoder.0.weight"].shape[0]
    own, opponent = torch.rand(width), torch.rand(width)
    payload = {"format_version": 14, "step": 1, "model": old, "metadata": {}}
    payload["metadata"]["board_representation"] = "mean-pool"
    told = {"own-deck": own, "opponent-deck": opponent, "leave_out": ["opponent-deck"]}
    model, metadata = tool.converted(payload, told, {"checkpoint": "old.pt"})
    assert metadata["leave_out"] == []
    assert metadata["folded_deck_lists"]["lists_left_out"] == ["opponent-deck"]
    expected = tool.fold(old, own, torch.zeros(width))["state.0.bias"]
    assert torch.equal(model.state_dict()["state.0.bias"], expected)


def test_a_checkpoint_of_another_format_is_refused():
    payload = {"format_version": 15, "step": 1, "model": {}, "metadata": {}}
    with pytest.raises(SystemExit):
        tool.converted(payload, {"leave_out": []}, {"checkpoint": "new.pt"})
