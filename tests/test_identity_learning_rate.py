"""The cards' identity tables learning at a multiple of the learning rate, and two runs of a
batch read against each other (`tools/pool_batch_compare.py`)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from ygo_training import checkpoints
from ygo_training.model import PolicyValueNet
from ygo_training.ppo_trainer import parameter_groups

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "pool_batch_compare", ROOT / "tools/pool_batch_compare.py"
)
compare = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(compare)


@pytest.fixture(scope="module")
def model() -> PolicyValueNet:
    torch.manual_seed(2012)
    return PolicyValueNet()


def test_at_one_the_weights_are_one_group_as_they_always_were(model):
    (group,) = parameter_groups(model, 1e-4, 1.0)
    assert group["params"] == list(model.parameters())
    optimizer = torch.optim.Adam(parameter_groups(model, 1e-4, 1.0), lr=1e-4)
    assert len(optimizer.param_groups) == 1 and optimizer.param_groups[0]["lr"] == 1e-4


def test_the_identity_tables_learn_apart_at_their_multiple(model):
    optimizer = torch.optim.Adam(parameter_groups(model, 1e-4, 10.0), lr=1e-4)
    rest, identity = optimizer.param_groups
    names = {id(value): name for name, value in model.named_parameters()}
    assert sorted(names[id(value)] for value in identity["params"]) == [
        "card_identity.high.weight",
        "card_identity.low.weight",
    ]
    assert identity["lr"] == pytest.approx(1e-3) and identity["lr_scale"] == 10.0
    assert rest["lr"] == 1e-4 and rest["lr_scale"] == 1.0
    assert len(rest["params"]) + len(identity["params"]) == len(list(model.parameters()))


def test_a_fresh_optimizer_of_one_group_begins_a_run_of_two(model, tmp_path):
    path = tmp_path / "mean.pt"
    checkpoints.save(path, model, torch.optim.Adam(model.parameters()), 37440, {})
    optimizer = torch.optim.Adam(parameter_groups(model, 1e-4, 10.0), lr=1e-4)
    checkpoints.load(path, model, optimizer)
    assert [group["lr"] for group in optimizer.param_groups] == [1e-4, pytest.approx(1e-3)]


def test_an_optimizer_that_learnt_is_not_dropped_for_another_make(model, tmp_path):
    path = tmp_path / "trained.pt"
    trained = torch.optim.Adam(model.parameters(), lr=1e-4)
    weight = next(model.value.parameters())
    weight.grad = torch.ones_like(weight)
    trained.step()
    model.zero_grad(set_to_none=True)
    checkpoints.save(path, model, trained, 37441, {})
    optimizer = torch.optim.Adam(parameter_groups(model, 1e-4, 10.0), lr=1e-4)
    with pytest.raises(ValueError):
        checkpoints.load(path, model, optimizer)


def written(path: Path, wins: dict[str, float], games: int = 400) -> None:
    suite = {}
    for name, share in wins.items():
        won = round(share * games)
        suite[name] = {"outcomes": ["win"] * won + ["loss"] * (games - won)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"records": [{"evaluation_suite": suite}]}), encoding="utf8")


def test_two_runs_are_read_against_each_other_on_the_same_games(tmp_path):
    for name, gain, old in (("fast", 0.12, 0.50), ("control", 0.08, 0.50)):
        written(tmp_path / name / "0/pairings.json", {"a-vs-each": 0.2})
        written(tmp_path / name / "4020/pairings.json", {"a-vs-each": 0.2 + gain})
        written(tmp_path / name / "4020/old.json", {"old-vs-each": old}, games=4000)
    found = compare.compared(tmp_path / "fast", tmp_path / "control", 4020)
    assert found["gain"]["gap"] == pytest.approx(0.04)
    assert found["faster"] == "faster"
    assert found["old decks"] == "no worse"
    assert found["gains"]["first"] == pytest.approx(0.12)
    assert "1. Learning: faster" in compare.shown(found, 4020)
    written(tmp_path / "control" / "0/pairings.json", {"a-vs-each": 0.3})
    with pytest.raises(SystemExit):
        compare.compared(tmp_path / "fast", tmp_path / "control", 4020)


@pytest.mark.parametrize(
    ("gap", "error", "told"),
    [(0.02, 0.004, "faster"), (0.01, 0.004, "not decided"), (0.0, 0.001, "not faster")],
)
def test_reading_one_asks_two_standard_errors(gap, error, told):
    assert compare.faster(gap, error) == told


@pytest.mark.parametrize(
    ("gap", "error", "told"),
    [(0.0, 0.004, "no worse"), (-0.01, 0.004, "not decided"), (-0.02, 0.004, "worse")],
)
def test_reading_two_asks_two_standard_errors(gap, error, told):
    assert compare.no_worse(gap, error) == told
