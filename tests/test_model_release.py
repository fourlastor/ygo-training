"""`tools/model_release.py`: the files of a network's release."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

from tests.views import PASS, choice, view
from ygo_training.checkpoints import load_inference_model, save, value_fingerprint
from ygo_training.model import PolicyValueNet

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
SPEC = importlib.util.spec_from_file_location("model_release", ROOT / "tools/model_release.py")
assert SPEC is not None and SPEC.loader is not None
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)


def made(tmp_path: Path) -> tuple[PolicyValueNet, Path, Path]:
    torch.manual_seed(5)
    model = PolicyValueNet(board_representation="mean-pool")
    checkpoint = tmp_path / "model-00002.pt"
    save(checkpoint, model, torch.optim.Adam(model.parameters()), 2, {})
    vectors = tmp_path / "decisions.json"
    shown = [view(), view(choices=[choice(PASS)])]
    vectors.write_text(json.dumps({"decisions": [{"view": each} for each in shown]}))
    return model, checkpoint, vectors


def test_a_release_holds_the_network_for_play_its_export_and_their_digests(tmp_path):
    model, checkpoint, vectors = made(tmp_path)
    output = tmp_path / "release"

    files = tool.release("tiny-net", checkpoint, output, vectors)

    names = ["tiny-net.pt", "tiny-net.safetensors", "tiny-net.json"]
    names += ["tiny-net.outputs.json", "SHA256SUMS"]
    assert [path.name for path in files] == names
    kept = torch.load(output / "tiny-net.pt", map_location="cpu", weights_only=False)
    assert sorted(kept) == ["format_version", "metadata", "model", "step"]
    loaded, _ = load_inference_model(output / "tiny-net.pt")
    assert value_fingerprint(loaded.state_dict()) == value_fingerprint(model.state_dict())
    described = json.loads((output / "tiny-net.json").read_text())
    assert described["checkpoint"]["checkpoint"] == "tiny-net.pt"
    assert described["checkpoint"]["step"] == 2
    sums = (output / "SHA256SUMS").read_text().splitlines()
    assert sums == [
        f"{hashlib.sha256((output / name).read_bytes()).hexdigest()}  {name}" for name in names[:-1]
    ]


@pytest.mark.parametrize("name", ["All71", "all 71", "-all71", "all71-", "model/all71"])
def test_a_name_that_is_no_tag_is_refused(tmp_path, name):
    _, checkpoint, vectors = made(tmp_path)
    with pytest.raises(SystemExit, match="lowercase words joined by dashes"):
        tool.release(name, checkpoint, tmp_path / "release", vectors)


def test_a_checkpoint_without_its_weights_is_refused(tmp_path):
    _, checkpoint, vectors = made(tmp_path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    del payload["metadata"]
    torch.save(payload, checkpoint)
    with pytest.raises(SystemExit, match="holds no metadata"):
        tool.release("tiny-net", checkpoint, tmp_path / "release", vectors)
