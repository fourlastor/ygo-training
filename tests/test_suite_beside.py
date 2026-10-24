"""Measurements of one suite set beside each other game by game."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("suite_beside", ROOT / "tools/suite_beside.py")
assert SPEC is not None and SPEC.loader is not None
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)


def measured(path: Path, players: dict[str, list[int]], *, seed: int = 2012) -> Path:
    """A file of records: each player wins the first games of two cases of
    ten, as many as it is given for each."""
    records = [
        {
            "checkpoint_label": name,
            "evaluation_suite": {
                case: {"outcomes": ["win"] * count + ["loss"] * (10 - count)}
                for case, count in zip(("b-vs-x", "a-vs-y"), won, strict=True)
            },
        }
        for name, won in players.items()
    ]
    path.write_text(json.dumps({"seed": seed, "records": records}), encoding="utf8")
    return path


def test_players_of_the_same_deals_are_told_by_the_games_one_wins_alone(tmp_path, capsys):
    pilot = measured(tmp_path / "pilot.json", {"pilot": [6, 5]})
    models = measured(tmp_path / "models.json", {"model-1": [5, 5], "model-2": [8, 7]})

    found = tool.report([pilot, models], ["pilot", "start", "end"])

    assert found["cases"]["a-vs-y"] == {"pilot": 0.5, "start": 0.5, "end": 0.7}
    assert found["all"] == {"pilot": 0.55, "start": 0.5, "end": 0.75}
    # Each over the first, and over the one before it.
    assert list(found["differences"]) == ["start over pilot", "end over pilot", "end over start"]
    assert found["differences"]["end over pilot"][0] == pytest.approx(0.2)
    # Four games of twenty are the end's alone: the error of a share of them.
    assert found["differences"]["end over pilot"][1] == pytest.approx(0.0918, abs=1e-4)
    said = capsys.readouterr().out
    assert "a-vs-y" in said and "Game by game, of 20:" in said
    assert "end over start: +25.0 ± " in said
    # Without names a record goes by its own.
    assert list(tool.report([pilot, models])["all"]) == ["pilot", "model-1", "model-2"]


def test_records_of_other_games_or_without_them_are_refused(tmp_path):
    pilot = measured(tmp_path / "pilot.json", {"pilot": [6, 5]})
    other = measured(tmp_path / "other.json", {"model": [5, 5]}, seed=7)
    with pytest.raises(SystemExit, match="not the same games"):
        tool.report([pilot, other])
    with pytest.raises(SystemExit, match="2 names for 1 records"):
        tool.report([pilot], ["a", "b"])
    counted = tmp_path / "counted.json"
    counted.write_text(
        json.dumps(
            {
                "seed": 2012,
                "records": [{"checkpoint_label": "old", "evaluation_suite": {"a": {"wins": 3}}}],
            }
        ),
        encoding="utf8",
    )
    with pytest.raises(SystemExit, match="does not tell how each game"):
        tool.report([counted])
