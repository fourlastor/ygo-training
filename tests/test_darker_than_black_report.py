"""`tools/darker_than_black_report.py`: each network's edge over Blackwing's
pilot with Blackwing's list and with Darker Than Black, on files written as
`experiments/commands/darker-than-black.sh` leaves them."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
spec = importlib.util.spec_from_file_location(
    "darker_than_black_report", ROOT / "tools/darker_than_black_report.py"
)
tool = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(tool)

GAMES = 400
STEPS = {"half": [4800, 4920, 5040], "all": [9120, 9240, 9360]}


def cases(won: int, games: int = GAMES) -> dict:
    """The mirror and Monarch, the first `won` games of each won."""
    told = ["win"] * won + ["loss"] * (games - won)
    return {
        "blackwing-vs-blackwing": {"games": games, "outcomes": told},
        "blackwing-vs-monarch": {"games": games, "outcomes": told},
    }


def write(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"seed": 2012, "records": records}))


def measurements(folder: Path, won: dict[str, dict[str, int]], checkpoints: int = 3) -> None:
    """Each player's wins of each pairing's games, with each list."""
    for listed, players in won.items():
        write(
            folder / f"pilot.{listed}.json",
            [{"checkpoint_label": "pilot", "evaluation_suite": cases(players["pilot"])}],
        )
        for network in tool.NETWORKS:
            write(
                folder / f"{network}.{listed}.json",
                [
                    {"checkpoint_step": step, "evaluation_suite": cases(players[network])}
                    for step in STEPS[network][:checkpoints]
                ],
            )


WON = {
    "blackwing": {"pilot": 200, "half": 320, "all": 280},
    "darker-than-black": {"pilot": 120, "half": 160, "all": 220},
}


def test_an_edge_is_read_on_the_same_games_with_each_list(tmp_path):
    measurements(tmp_path, WON)
    result = tool.report(tmp_path)
    assert result["rates"]["blackwing"]["pilot"] == pytest.approx(0.5)
    assert result["rates"]["darker-than-black"]["half"] == pytest.approx(0.4)
    edges = result["edges"]["half"]
    assert edges["blackwing"]["gap"] == pytest.approx(0.3)
    assert edges["darker-than-black"]["gap"] == pytest.approx(0.1)
    # half keeps a third of its edge: lost; all keeps more than all of it.
    assert result["readings"]["half"]["gap"] == pytest.approx(-0.2)
    assert result["readings"]["half"]["verdict"] == "lost"
    assert result["readings"]["all"]["gap"] == pytest.approx(0.05)
    assert result["readings"]["all"]["verdict"] == "kept"
    assert result["readings"]["all"]["games"] == 2 * GAMES
    assert result["mirror"]["pilot"]["darker-than-black"] == pytest.approx(0.3)
    assert result["mirror"]["all"]["blackwing"]["games"] == GAMES
    assert result["by_opponent"]["monarch"]["half"]["blackwing"] == pytest.approx(0.3)
    shown = tool.show(result)
    assert "half: thirteen decks" in shown and "-20.0" in shown and "lost" in shown


def test_files_of_other_games_or_checkpoints_are_refused(tmp_path):
    measurements(tmp_path, WON, checkpoints=2)
    with pytest.raises(SystemExit, match="2 checkpoints, not 3"):
        tool.report(tmp_path)
    measurements(tmp_path, WON)
    write(
        tmp_path / "all.darker-than-black.json",
        [
            {"checkpoint_step": step, "evaluation_suite": cases(10, games=20)}
            for step in STEPS["all"]
        ],
    )
    with pytest.raises(SystemExit, match="other games than the others"):
        tool.report(tmp_path)
