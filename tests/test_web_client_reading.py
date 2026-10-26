"""`tools/web_client_reading.py`: a duel the web client saved, read again
beside other answers, and played on from the person's decisions."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

from tests.test_web_client_duels import needs_the_environment, network, saved_by_a_page
from ygo_training.checkpoints import save
from ygo_training.environment import Environment
from ygo_training.model import PolicyValueNet

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "web_client_reading", ROOT / "tools/web_client_reading.py"
)
assert SPEC is not None and SPEC.loader is not None
tool = importlib.util.module_from_spec(SPEC)
# Registered as an import would be, so that its workers find its functions.
sys.modules["web_client_reading"] = tool
SPEC.loader.exec_module(tool)


@pytest.fixture(scope="module")
def environment():
    with Environment(ROOT) as opened:
        yield opened


def kept_as_a_file(tmp_path: Path, saved: dict) -> Path:
    path = tmp_path / "claudi-oh-duel.json"
    path.write_text(json.dumps({"format": 1, "replay": json.dumps(saved)}), encoding="utf8")
    return path


def checkpoint_of(tmp_path: Path, model: torch.nn.Module) -> Path:
    path = tmp_path / "model-00001.pt"
    save(path, model, torch.optim.Adam(model.parameters()), 1, {})
    return path


@pytest.mark.integration
@needs_the_environment
def test_a_duel_played_on_from_a_cut_is_the_persons_to_it_and_then_the_answer_named(
    environment,
):
    model = network(3)
    saved, record = saved_by_a_page(environment, model, seat=0)
    # Played whole with the person's seat handed over, it is the duel saved:
    # the person's answers as recorded, the network's as the checkpoint gives them.
    views, over = tool.played(environment, model, saved)
    assert over["record"]["responses"] == record["responses"]
    assert views and all(int(view["player"]) == 1 for view in views)

    found = tool.departures(environment, model, views)
    assert found, "the pilot and the network part at a command of the pilot's"
    branch = found[0]
    cut = branch["answer"]
    _, theirs = tool.played(
        environment, model, saved, cut=cut, answer="person", shown=branch["view"]
    )
    _, its = tool.played(environment, model, saved, cut=cut, answer="network")
    # Both are the duel saved up to the cut; there one gives the person's
    # answer and the other the checkpoint's, which is another.
    assert theirs["record"]["responses"][: cut + 1] == record["responses"][: cut + 1]
    assert its["record"]["responses"][:cut] == record["responses"][:cut]
    assert its["record"]["responses"][cut] != record["responses"][cut]
    # A playout of a seed is the same playout every time.
    once = tool.played(environment, model, saved, cut=cut, answer="network", sampled=7)[1]
    again = tool.played(environment, model, saved, cut=cut, answer="network", sampled=7)[1]
    assert once["record"]["responses"] == again["record"]["responses"]
    with pytest.raises(ValueError, match="named by the view"):
        tool.played(environment, model, saved, cut=cut, answer="person")


@pytest.mark.integration
@needs_the_environment
def test_both_readings_are_said(environment, tmp_path, capsys):
    # A checkpoint is loaded at the network's own sizes.
    torch.manual_seed(5)
    model = PolicyValueNet(board_representation="mean-pool").eval()
    saved, _ = saved_by_a_page(environment, model, seat=1)
    path = kept_as_a_file(tmp_path, saved)
    checkpoint = checkpoint_of(tmp_path, model)

    tool.beside(argparse.Namespace(checkpoint=checkpoint, saved=[path], other=checkpoint))
    said = capsys.readouterr().out
    told = [line for line in said.splitlines() if "decisions of the network" in line]
    assert len(told) == 1
    # The same checkpoint beside itself answers every decision alike.
    decisions = int(told[0].split()[0])
    assert f"the other checkpoint's in {decisions}" in told[0]
    assert " NET [" in said

    tool.what_if(
        argparse.Namespace(
            checkpoint=checkpoint, saved=[path], playouts=2, every_kind=False, workers=2
        )
    )
    said = capsys.readouterr().out
    assert "played whole, the duel recorded" in said
    assert " | yours " in said and "/2 (" in said
