"""The report of the experiment on the learner itself: the readings."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("learner_report", ROOT / "tools/learner_report.py")
assert SPEC is not None and SPEC.loader is not None
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)


def run(directory: Path, name: str, fits: list[float], *, agreement: float = 0.95) -> None:
    """A run's figures: one row an update, with the value head's fit there."""
    rows = [
        {
            "version": update,
            "imitation_value_explained_variance": fit,
            "imitation_value_correlation": 0.5,
            "imitation_model_teacher_agreement": agreement,
            **(
                {"evaluation_suite": {"a": {"wins": 60, "games": 100}}}
                if (update + 1) % 40 == 0
                else {}
            ),
        }
        for update, fit in enumerate(fits)
    ]
    (directory / f"{name}.metrics.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf8"
    )


def measured(directory: Path, name: str, won: int, *, step: int = 360, told: float = 0.5) -> None:
    """A run's last checkpoint on 100 held-out deals of which it wins the first ``won``."""
    entry: dict[str, Any] = {
        "checkpoint_step": step,
        "deals": {
            "win_rate": won / 100,
            "outcomes": ["win"] * won + ["loss"] * (100 - won),
            "references": {"pilot": {"win_rate": 0.6}, "search": {"win_rate": 0.72}},
        },
        "answer_key": {
            "tried": {
                "value": {"correlation": 0.6, "told": told, "explained": 0.3, "result": 0.4},
                "head": {"correlation": 0.4},
            }
        },
    }
    folder = directory / "heldout"
    folder.mkdir(exist_ok=True)
    (folder / f"{name}.json").write_text(json.dumps({"checkpoints": [entry]}), encoding="utf8")


def learner(tmp_path: Path, *, once: float, reading: float, few: float, cards: float) -> Path:
    run(tmp_path, "again", [-0.1] * 90)
    run(tmp_path, "once", [once] * 360)
    run(tmp_path, "once-few", [few] * 120)
    run(tmp_path, "once-reading", [reading] * 360)
    run(tmp_path, "once-cards", [cards] * 360)
    for name, won in (("seed", 55), ("again", 56), ("once", 57), ("once-cards", 62)):
        measured(tmp_path, name, won)
    return tmp_path


def test_the_readings_are_applied_as_the_record_fixes_them(tmp_path, capsys):
    found = tool.report(learner(tmp_path, once=0.32, reading=0.25, few=0.33, cards=0.38))

    readings = found["readings"]
    # 0.32 of the results of new games: it predicts, and reads the position.
    assert (readings["predicts"], readings["reads"]) == (True, True)
    # It explains 0.07 more than the head that only reads, and the head on
    # the cards 0.06 more than it; four positions a game add 0.01.
    assert (readings["must_shape"], readings["fewer_positions"]) == (True, False)
    assert readings["board_helps_heads"] is True
    # One deal more than `again` of 100 is no worse and no better; five more
    # for the cards than `once` are more than two standard errors.
    assert (readings["copy_no_worse"], readings["copy_better"]) == (True, False)
    assert readings["board_helps_copy"] is True
    assert found["runs"]["again"]["fit"] == -0.1
    assert found["runs"]["once"]["against_again"][0] == 0.01
    printed = capsys.readouterr().out
    assert "the value head predicts (once explains +0.320, the mark 0.2): yes" in printed
    row = next(line for line in printed.splitlines() if line.startswith("once-cards "))
    # Its last 40 updates, and of one result half of the 40% there is to tell.
    assert row.split()[1] == "321-360" and row.split()[-1] == "20.0%"


def test_a_head_under_the_marks_and_a_copy_that_loses_deals(tmp_path):
    directory = learner(tmp_path, once=0.15, reading=0.15, few=0.25, cards=0.15)
    measured(directory, "once", 40)

    readings = tool.report(directory)["readings"]

    assert (readings["predicts"], readings["reads"]) == (False, False)
    assert (readings["must_shape"], readings["fewer_positions"]) == (False, True)
    # Sixteen deals fewer than `again` of 100 are more than two standard errors.
    assert readings["copy_no_worse"] is False
    assert readings["board_helps_heads"] is False


def test_what_was_not_run_is_not_read(tmp_path):
    run(tmp_path, "once", [0.25] * 360)

    readings = tool.report(tmp_path)["readings"]

    assert readings["predicts"] is True and readings["reads"] is False
    assert readings["copy_no_worse"] is None and readings["board_helps_copy"] is None
    assert readings["must_shape"] is None and readings["fewer_positions"] is None
