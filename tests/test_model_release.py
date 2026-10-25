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

    files = tool.release({"tiny-net": checkpoint}, output, vectors)

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


def test_a_release_of_several_networks_names_each_ones_files_and_sums_them_all(tmp_path):
    _, checkpoint, vectors = made(tmp_path)
    torch.manual_seed(6)
    model = PolicyValueNet(board_representation="mean-pool")
    other = tmp_path / "model-00003.pt"
    save(other, model, torch.optim.Adam(model.parameters()), 3, {})
    output = tmp_path / "release"

    files = tool.release({"one-deck": checkpoint, "another-deck": other}, output, vectors)

    suffixes = (".pt", ".safetensors", ".json", ".outputs.json")
    names = [f"{name}{suffix}" for name in ("one-deck", "another-deck") for suffix in suffixes]
    assert [path.name for path in files] == [*names, "SHA256SUMS"]
    sums = (output / "SHA256SUMS").read_text().splitlines()
    assert [line.split("  ")[1] for line in sums] == names
    described = json.loads((output / "another-deck.json").read_text())
    assert described["checkpoint"]["checkpoint"] == "another-deck.pt"
    assert described["checkpoint"]["step"] == 3


def test_a_release_is_of_one_checkpoint_or_of_the_networks_named():
    assert tool.networks_of("all71", Path("a.pt"), []) == {"all71": Path("a.pt")}
    named = tool.networks_of("champions", None, ["burn=b.pt", "gishki=g.pt"])
    assert named == {"burn": Path("b.pt"), "gishki": Path("g.pt")}
    for checkpoint, given, said in (
        (Path("a.pt"), ["burn=b.pt"], "one checkpoint, or"),
        (None, [], "one checkpoint, or"),
        (None, ["burn"], "NAME=CHECKPOINT"),
        (None, ["=b.pt"], "NAME=CHECKPOINT"),
        (None, ["burn=b.pt", "burn=c.pt"], "named twice"),
    ):
        with pytest.raises(SystemExit, match=said):
            tool.networks_of("champions", checkpoint, given)


@pytest.mark.parametrize("name", ["All71", "all 71", "-all71", "all71-", "model/all71"])
def test_a_name_that_is_no_tag_is_refused(tmp_path, name):
    _, checkpoint, vectors = made(tmp_path)
    with pytest.raises(SystemExit, match="lowercase words joined by dashes"):
        tool.release({name: checkpoint}, tmp_path / "release", vectors)


def test_a_checkpoint_without_its_weights_is_refused(tmp_path):
    _, checkpoint, vectors = made(tmp_path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    del payload["metadata"]
    torch.save(payload, checkpoint)
    with pytest.raises(SystemExit, match="holds no metadata"):
        tool.release({"tiny-net": checkpoint}, tmp_path / "release", vectors)


def test_a_release_is_published_with_its_files_titled_and_described_by_its_card(
    tmp_path, monkeypatch
):
    files = [tmp_path / "tiny-net.pt", tmp_path / "SHA256SUMS"]
    card = "# tiny-net: a network\n\nA network\nof one deck ([its record](../experiments/r.md)).\n"
    ran = []

    def recorded(argv, **_):
        notes = Path(argv[argv.index("--notes-file") + 1]).read_text(encoding="utf8")
        ran.append((argv, notes))

    monkeypatch.setattr(tool.subprocess, "run", recorded)
    tool.publish("tiny-net", files, card, "abc123", "owner/repo")

    ((argv, notes),) = ran
    assert argv[:6] == ["gh", "release", "create", "model/tiny-net", str(files[0]), str(files[1])]
    told = argv[6:]
    assert told[:6] == [
        "--repo",
        "owner/repo",
        "--target",
        "abc123",
        "--title",
        "tiny-net: a network",
    ]
    assert notes == (
        "A network of one deck "
        "([its record](https://github.com/owner/repo/blob/abc123/experiments/r.md)).\n"
    )


@pytest.mark.parametrize(
    ("url", "repository"),
    [
        ("git@github.com:fourlastor/ygo-training.git", "fourlastor/ygo-training"),
        ("https://github.com/fourlastor/ygo-training", "fourlastor/ygo-training"),
        ("ssh://git@github.com/fourlastor/ygo-training.git", "fourlastor/ygo-training"),
    ],
)
def test_the_repository_is_origins(url, repository):
    assert tool.repository_of(url) == repository


def test_a_remote_not_on_github_is_refused():
    with pytest.raises(SystemExit, match="no repository of GitHub"):
        tool.repository_of("daniele@luna:/home/daniele/ygo-training/ygo")


def test_a_network_without_a_card_is_not_published():
    with pytest.raises(SystemExit, match="No card at"):
        tool.publishing("no-such-network", "HEAD")
