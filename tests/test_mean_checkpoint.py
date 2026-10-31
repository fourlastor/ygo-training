"""`tools/mean_checkpoint.py`: the mean of a run's checkpoints, weight by
weight, as a checkpoint to play."""

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
spec = importlib.util.spec_from_file_location("mean_checkpoint", ROOT / "tools/mean_checkpoint.py")
tool = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(tool)


def saved(path: Path, seed: int, step: int) -> dict[str, torch.Tensor]:
    """A checkpoint of a network of its own seed, and that network's weights."""
    torch.manual_seed(seed)
    model = PolicyValueNet()
    checkpoints.save(path, model, torch.optim.Adam(model.parameters()), step, {"duels": 520.0})
    return model.state_dict()


def test_the_mean_is_each_weights_mean_and_plays_as_a_checkpoint(tmp_path, monkeypatch):
    paths = [tmp_path / f"model-{step:05d}.pt" for step in (239, 479, 719)]
    states = [
        saved(path, seed, step + 1)
        for seed, (path, step) in enumerate(zip(paths, (239, 479, 719), strict=True))
    ]
    output = tmp_path / "mean" / "model-00719.pt"
    arguments = ["mean_checkpoint.py", *map(str, paths), "--output", str(output)]
    monkeypatch.setattr(sys, "argv", arguments)

    tool.main()

    loaded, payload = checkpoints.load_inference_model(output)
    for name, value in loaded.state_dict().items():
        expected = (
            states[0][name].double() + states[1][name].double() + states[2][name].double()
        ) / 3
        assert torch.allclose(value.double(), expected, atol=1e-6), name
    assert payload["step"] == 720
    assert payload["metadata"]["duels"] == 520.0
    assert payload["metadata"]["mean_of"] == [path.name for path in paths]
    assert (output.parent / "model-00719.pt.json").exists()


def test_a_weight_keeps_its_type_and_what_no_update_moves_is_kept():
    first = {"weight": torch.tensor([1.0, 2.0], dtype=torch.float32), "codes": torch.tensor([7, 8])}
    second = {
        "weight": torch.tensor([2.0, 5.0], dtype=torch.float32),
        "codes": torch.tensor([7, 8]),
    }

    mean = tool.meaned([first, second])

    assert mean["weight"].dtype == torch.float32
    assert mean["weight"].tolist() == [1.5, 3.5]
    assert mean["codes"].tolist() == [7, 8]


def test_checkpoints_of_other_networks_have_no_mean():
    weight = torch.zeros(2)
    with pytest.raises(SystemExit, match="the same weights"):
        tool.meaned([{"weight": weight}, {"other": weight}])
    with pytest.raises(SystemExit, match="codes is not a float"):
        tool.meaned([{"codes": torch.tensor([1])}, {"codes": torch.tensor([2])}])
