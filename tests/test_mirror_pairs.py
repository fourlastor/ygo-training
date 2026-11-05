"""`tools/mirror_pairs.py`: two networks' mirrors paired deal by deal."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def loaded(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"tools/{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


tool = loaded("mirror_pairs")


def measured(directory: Path, deck: str, plays: list[str], held: list[str]) -> None:
    directory.mkdir(exist_ok=True)
    for kind, outcomes in (("plays", plays), ("held", held)):
        record = {"records": [{"evaluation": {"outcomes": outcomes}}]}
        (directory / f"{deck}.{kind}.json").write_text(json.dumps(record))


def test_a_network_shares_its_own_deals_and_the_other_side_of_the_champions(tmp_path):
    measured(tmp_path, "agents", ["win", "draw"], ["win", "loss"])
    assert tool.shares(tmp_path) == {"agents": [1.0, 0.5, 0.0, 1.0]}


def test_two_networks_are_paired_deal_by_deal_over_the_decks_both_measured(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    measured(first, "agents", ["win", "win"], ["loss", "loss"])
    measured(first, "alien", ["loss", "loss"], ["win", "win"])
    measured(first, "zombie", ["win", "win"], ["win", "win"])
    measured(second, "agents", ["loss", "win"], ["loss", "win"])
    measured(second, "alien", ["loss", "draw"], ["win", "win"])
    paired = tool.against(tool.shares(first), tool.shares(second))
    assert paired["decks"] == 2 and paired["lower"] == 1
    # Differences -1, 0, 0, -1 on agents and 0, +0.5, 0, 0 on alien.
    assert paired["difference"] == pytest.approx(-1.5 / 8)
    assert paired["by deck"] == [(-0.5, "agents"), (0.125, "alien")]


def test_networks_measured_on_different_deals_are_refused(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    measured(first, "agents", ["win"], ["loss"])
    measured(second, "agents", ["win", "win"], ["loss", "loss"])
    with pytest.raises(ValueError, match="agents"):
        tool.against(tool.shares(first), tool.shares(second))
