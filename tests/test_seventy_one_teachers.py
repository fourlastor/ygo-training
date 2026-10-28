"""`tools/seventy_one_teachers.py`: the teachers of the network of the
seventy-one decks held to the record before its copy is made."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from ygo_training.checkpoints import load_inference_model, save, value_fingerprint
from ygo_training.model import PolicyValueNet

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "seventy_one_teachers", ROOT / "tools/seventy_one_teachers.py"
)
tool = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(tool)


def run(folder: Path, seed: int) -> Path:
    """A run's last checkpoint, of a network drawn from `seed`."""
    torch.manual_seed(seed)
    model = PolicyValueNet(board_representation="mean-pool").eval()
    path = folder / f"run-{seed}" / "model-00959.pt"
    save(path, model, torch.optim.Adam(model.parameters()), 960, {})
    return path


def recorded(records: Path, deck: str, checkpoint: Path) -> None:
    """A recording of `deck` made with the network at `checkpoint`."""
    model, _ = load_inference_model(checkpoint, device="cpu")
    about = {"teacher": {"fingerprint": value_fingerprint(model.state_dict())}}
    (records / deck).mkdir(parents=True)
    (records / deck / "about.json").write_text(json.dumps(about))


def figure(champions: Path, deck: str, share: float, error: float, verdict: str) -> None:
    (champions / deck).mkdir(parents=True)
    found = {deck: {"share": share, "error": error, "deals": 2048, "verdict": verdict}}
    (champions / deck / "report.json").write_text(json.dumps(found))


def test_each_deck_is_told_by_its_figure_and_a_release_deck_without_one(tmp_path):
    records, champions = tmp_path / "records", tmp_path / "champions"
    one, other = run(tmp_path, 1), run(tmp_path, 2)
    recorded(records, "burn", one)
    recorded(records, "cyber", other)
    # Not told apart from all71, but not under it: copied all the same.
    figure(champions, "burn", 0.509, 0.005, "not told apart")
    listing = tmp_path / "list"
    listing.write_text(f"# the decks\nburn {one}\ncyber {other}\n")

    told = tool.checked(tool.listed(listing), records, champions, ("cyber",))

    assert told == [
        "burn: 50.9% ± 0.5% against all71 in its mirror, not told apart",
        "cyber: of the champions' release",
    ]


def test_a_recording_of_another_network_is_refused(tmp_path):
    records, champions = tmp_path / "records", tmp_path / "champions"
    recorded(records, "burn", run(tmp_path, 1))
    figure(champions, "burn", 0.6, 0.005, "a champion")

    with pytest.raises(SystemExit, match="burn was recorded with another network"):
        tool.checked([("burn", run(tmp_path, 2))], records, champions, ())


def test_a_run_under_all71_is_refused(tmp_path):
    records, champions = tmp_path / "records", tmp_path / "champions"
    checkpoint = run(tmp_path, 1)
    recorded(records, "burn", checkpoint)
    figure(champions, "burn", 0.48, 0.009, "not told apart")

    with pytest.raises(SystemExit, match="under all71 in its mirror, 48.0% ± 0.9%"):
        tool.checked([("burn", checkpoint)], records, champions, ())


def test_a_deck_without_a_figure_or_a_recording_is_refused(tmp_path):
    records, champions = tmp_path / "records", tmp_path / "champions"
    checkpoint = run(tmp_path, 1)
    recorded(records, "burn", checkpoint)

    with pytest.raises(SystemExit, match="burn's run is told by its figure"):
        tool.checked([("burn", checkpoint)], records, champions, ("cyber",))
    with pytest.raises(SystemExit, match="No recording of gusto"):
        tool.checked([("gusto", checkpoint)], records, champions, ())
