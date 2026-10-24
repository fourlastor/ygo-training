"""`tools/matchups_report.py`: Blackwing against another deck, head to head,
on measurements written as `experiments/commands/blackwing-matchups.sh`
leaves them."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
spec = importlib.util.spec_from_file_location("matchups_report", ROOT / "tools/matchups_report.py")
tool = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(tool)

DEALS = 400


def measured(path: Path, deck: str, won: dict[str, int]) -> None:
    """Each network of WON playing Blackwing against DECK, winning the first
    so many deals (a draw after them, then losses)."""
    records = []
    for network, wins in won.items():
        outcomes = ["win"] * wins + ["draw"] + ["loss"] * (DEALS - wins - 1)
        checkpoint = f"/somewhere/runs/more-decks/{network}/blackwing-against-{deck}/model-1.pt"
        case = {f"blackwing-vs-{deck}-checkpoint": {"games": DEALS, "outcomes": outcomes}}
        records.append({"checkpoint": checkpoint, "evaluation_suite": case})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"records": records}))


def test_the_networks_beside_decks_01_over_the_decks_together(tmp_path):
    measured(tmp_path / "decks-05" / "burn.json", "burn", {"decks-01": 200, "decks-05": 260})
    measured(tmp_path / "decks-05" / "gishki.json", "gishki", {"decks-01": 100, "decks-05": 140})
    measured(tmp_path / "decks-13" / "burn.json", "burn", {"decks-01": 240, "decks-05": 200})
    measured(tmp_path / "decks-13" / "gishki.json", "gishki", {"decks-01": 120, "decks-05": 80})
    result = tool.report(tmp_path)
    against = result["decks-05"]
    assert against["games"] == 2 * DEALS
    assert against["shares"]["burn"]["decks-05"] == pytest.approx(0.65)
    assert against["together"]["decks-01"] == pytest.approx(0.375)
    found = against["beside"]["decks-05"]
    assert found["gap"] == pytest.approx(0.125)
    assert found["by_deck"]["gishki"]["gap"] == pytest.approx(0.10)
    assert against["reading"] == "the matchup's own games count"
    assert result["decks-13"]["beside"]["decks-05"]["gap"] == pytest.approx(-0.10)
    assert result["decks-13"]["reading"] == "the deck's other games count for more"
    assert "The reading: the matchup's own games count." in tool.show(result)


def test_a_small_gap_is_not_told_apart(tmp_path):
    measured(tmp_path / "decks-05" / "burn.json", "burn", {"decks-01": 200, "decks-05": 204})
    assert tool.report(tmp_path)["decks-05"]["reading"] == "not told apart"


def test_the_pilots_are_set_beside_without_a_reading(tmp_path):
    measured(tmp_path / "pilots" / "burn.json", "burn", {"decks-01": 300, "decks-05": 200})
    result = tool.report(tmp_path)
    assert result["pilots"]["beside"]["decks-05"]["gap"] == pytest.approx(-0.25)
    assert "reading" not in result["pilots"]
    assert "Blackwing against the decks' pilots" in tool.show(result)
