"""The report of the follow-up on the learner itself: the readings."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "learner_follow_up_report", ROOT / "tools/learner_follow_up_report.py"
)
assert SPEC is not None and SPEC.loader is not None
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)

STEPS = (240, 280, 320, 360)


def run(directory: Path, name: str, fit: float, *, games: str = "same") -> None:
    """A run's figures over 360 updates, with the value head's fit there."""
    rows = [
        {
            "version": update,
            "imitation_value_explained_variance": fit,
            "imitation_value_correlation": 0.5,
            "rollout_decision_fingerprint": f"{games}-{update}",
            **(
                {"evaluation_suite": {"a": {"wins": 58, "games": 100}}}
                if (update + 1) % 40 == 0
                else {}
            ),
        }
        for update in range(360)
    ]
    (directory / f"{name}.metrics.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf8"
    )


def measured(directory: Path, name: str, won: list[int]) -> None:
    """A run's four checkpoints on 200 held-out deals: after each it wins as
    many deals as ``won`` says, the first ones."""
    entries = [
        {
            "checkpoint_step": step,
            "deals": {
                "outcomes": ["win"] * count + ["loss"] * (200 - count),
                "paired": {"pilot": {"both": 100, "pilot_only": 22}},
            },
            "answer_key": {
                "tried": {
                    "value": {"correlation": 0.8, "told": 0.7, "explained": 0.6, "result": 0.4}
                }
            },
        }
        for step, count in zip(STEPS, won, strict=True)
    ]
    folder = directory / "heldout"
    folder.mkdir(exist_ok=True)
    (folder / f"{name}.json").write_text(json.dumps({"checkpoints": entries}), encoding="utf8")


def follow_up(tmp_path: Path, *, both: float, cards: list[int], cards_few: list[int]) -> Path:
    run(tmp_path, "once", 0.18)
    run(tmp_path, "once-few", 0.33)
    run(tmp_path, "once-cards", 0.17)
    run(tmp_path, "once-cards-few", both)
    measured(tmp_path, "once", [112, 116, 114, 114])
    measured(tmp_path, "once-few", [113, 115, 114, 115])
    measured(tmp_path, "once-cards", cards)
    measured(tmp_path, "once-cards-few", cards_few)
    return tmp_path


def test_the_readings_are_applied_as_the_record_fixes_them(tmp_path, capsys):
    directory = follow_up(
        tmp_path, both=0.31, cards=[122, 124, 124, 126], cards_few=[124, 124, 126, 126]
    )

    found = tool.report(directory)

    assert found["same_games"] is True
    assert found["checkpoints"] == list(STEPS)
    # 0.31 of the results of new games with both, where the mark is 0.30.
    assert found["readings"]["both_predicts"] is True
    # Four positions take nothing from either copy.
    assert (found["readings"]["both_copies"], found["readings"]["few_copies"]) == (True, True)
    # Ten deals of 200 more at every checkpoint, whichever way the head is taught.
    assert found["readings"]["cards_help"] is True
    assert round(found["differences"]["cards over pooled"][0], 4) == 0.0519
    assert found["runs"]["once-cards-few"]["deals"][360] == 0.63
    assert round(found["runs"]["once"]["deals_all"], 3) == 0.57
    printed = capsys.readouterr().out
    assert "the same games over the 360 updates they share: yes" in printed
    assert "on those deals the pilot wins 61.0%" in printed
    row = next(line for line in printed.splitlines() if line.startswith("once-cards-few "))
    assert row.split()[1:3] == ["321-360", "+0.310"]


def test_a_head_under_its_mark_and_cards_that_do_not_help(tmp_path):
    directory = follow_up(
        tmp_path, both=0.22, cards=[114, 116, 113, 115], cards_few=[100, 102, 101, 100]
    )

    readings = tool.report(directory)["readings"]

    assert readings["both_predicts"] is False
    # Seven points fewer deals with both than with the cards alone.
    assert readings["both_copies"] is False
    assert readings["few_copies"] is True
    assert readings["cards_help"] is False


def test_what_is_not_there_is_not_read(tmp_path, capsys):
    run(tmp_path, "once", 0.18)
    run(tmp_path, "once-few", 0.33, games="other")
    measured(tmp_path, "once", [112, 116, 114, 114])

    found = tool.report(tmp_path)

    assert found["same_games"] is False
    assert found["checkpoints"] == []
    assert set(found["readings"].values()) == {None}
    printed = capsys.readouterr().out
    assert "NO, and nothing below compares like with like" in printed
    assert "the board as its cards helps the copy" in printed and "not measured" in printed
