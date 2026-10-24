"""The report of the copies of a specialist: the readings."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "specialist_copy_report", ROOT / "tools/specialist_copy_report.py"
)
assert SPEC is not None and SPEC.loader is not None
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)

# 200 held-out deals: the pilot wins the first 120, the teacher the first 150.
DEALS = 200
PILOT = 120
TEACHER = 150
TAUGHT = 2280
POINTS = [40, 80, 120, 160, 200]


def run(
    directory: Path,
    name: str,
    updates: int,
    *,
    teacher: dict | None = None,
    start: str = "as built",
    games: str = "same",
    suite: int = 60,
) -> None:
    """A run's figures over its updates, from scratch."""
    rows = [
        {
            "version": update - 1,
            "duels": 520,
            "samples": 40_000,
            "learner_win_rate": 0.74 if teacher else 0.6,
            "imitation_model_teacher_agreement": 0.5 + update / 500,
            "imitation_value_explained_variance": 0.3,
            "rollout_decision_fingerprint": f"{games}-{update}",
            **(
                {
                    "imitation_departures": 6_000,
                    "imitation_model_departure_agreement": update / 400,
                    "imitation_model_departure_pilot_agreement": 0.5 - update / 500,
                    "imitation_teacher_divergence": 1.0 / update,
                    "imitation_teacher_entropy": 0.11,
                }
                if teacher
                else {}
            ),
            **(
                {"evaluation_suite": {"a": {"wins": suite + update // 40, "games": 100}}}
                if update % 40 == 0
                else {}
            ),
        }
        for update in range(1, updates + 1)
    ]
    (directory / f"{name}.metrics.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf8"
    )
    (directory / f"{name}.run.json").write_text(
        json.dumps({"initial_model_fingerprint": start, "imitation_teacher": teacher}),
        encoding="utf8",
    )


def measured(
    directory: Path, file: str, steps: list[int], won: list[int], games: str = "held"
) -> None:
    """Checkpoints on the held-out deals: after each the model wins as many
    deals as ``won`` says, the first ones, as the pilot wins its own."""
    entries = [
        {
            "checkpoint_step": step,
            "deals": {
                "outcomes": ["win"] * count + ["loss"] * (DEALS - count),
                "references": {
                    "pilot": {"outcomes": ["win"] * PILOT + ["loss"] * (DEALS - PILOT)},
                    "search": {"win_rate": 0.72},
                },
            },
        }
        for step, count in zip(steps, won, strict=True)
    ]
    folder = directory / "heldout"
    folder.mkdir(exist_ok=True)
    (folder / f"{file}.json").write_text(
        json.dumps({"checkpoints": entries, "fingerprint": games, "games": DEALS}),
        encoding="utf8",
    )


def copies(
    tmp_path: Path,
    *,
    choice: list[int],
    distribution: list[int],
    points: list[int] = POINTS,
    teacher_step: int = TAUGHT,
) -> Path:
    """Two copies of a teacher, the pilot's copy beside them, and the teacher."""
    for name, won in (("choice", choice), ("distribution", distribution)):
        teacher = {
            "path": "runs/long/model-02279.pt",
            "step": TAUGHT,
            "fingerprint": "the teacher",
            "targets": name,
        }
        run(tmp_path, name, points[-1], teacher=teacher)
        measured(tmp_path, name, points, won)
    run(tmp_path, "pilot-copy", 360, suite=40)
    measured(tmp_path, "pilot-copy", POINTS, [80, 96, 104, 110, 112])
    measured(tmp_path, "pilot-copy-later", [240, 280, 320, 360], [116, 116, 118, 118])
    measured(tmp_path, "teacher", [teacher_step], [TEACHER])
    return tmp_path


def test_the_readings_are_applied_as_they_were_fixed(tmp_path, capsys):
    directory = copies(
        tmp_path, choice=[100, 120, 130, 136, 140], distribution=[104, 126, 136, 142, 146]
    )
    measured(tmp_path, "line.one-pass", [390, 480], [124, 130])
    measured(tmp_path, "line.one-pass-long.1920", [TAUGHT], [TEACHER])
    run(tmp_path, "teacher", TAUGHT, suite=17)

    found = tool.report(directory, updates=200)

    assert found["teacher"]["step"] == TAUGHT and found["one_teacher"] is True
    assert found["same_games"] is True and found["measured_on_other_games"] == []
    assert found["copies"]["choice"]["starts_as_the_pilots_copy"] is True
    assert found["teacher_games"]["games"] == 200 * 520
    assert found["teacher_games"]["win_rate"] == pytest.approx(0.74)
    assert found["teacher_games"]["leaves_its_pilot"] == pytest.approx(0.15)
    assert found["checkpoints"] == {"measured": POINTS, "read": [160, 200]}
    # A copy is read by its last two checkpoints: 69% of the deals, where the
    # pilot wins 60%, its copy 55.5% after as many updates, the teacher 75%.
    assert found["copies"]["choice"]["deals_read"] == pytest.approx(0.69)
    assert found["copies"]["pilot-copy"]["deals_read"] == pytest.approx(0.555)
    assert (found["pilot"], found["teacher_deals"]) == (pytest.approx(0.6), pytest.approx(0.75))
    differences = found["differences"]
    assert differences["choice over pilot"][0] == pytest.approx(0.09)
    assert differences["choice over pilot-copy"][0] == pytest.approx(0.135)
    assert differences["choice over pilot-copy-later"][0] == pytest.approx(0.105)
    assert differences["choice over teacher"][0] == pytest.approx(-0.06)
    # Halfway between the pilot and the teacher is 67.5% of the deals.
    assert differences["choice over mark"][0] == pytest.approx(0.015)
    assert differences["distribution over choice"][0] == pytest.approx(0.03)
    assert differences["teacher over pilot"][0] == pytest.approx(0.15)
    assert found["readings"] == {
        "past_the_pilot": True,
        "holds_its_share": True,
        "distribution_teaches_more": True,
    }
    # While they learn: the ten updates that end at a point.
    learned = found["copies"]["choice"]["learning"]
    assert [row["updates"] for row in learned] == POINTS
    assert learned[-1]["agrees"] == pytest.approx(0.5 + 195.5 / 500)
    assert learned[-1]["divergence"] == pytest.approx(sum(1 / u for u in range(191, 201)) / 10)
    assert found["copies"]["pilot-copy"]["learning"][0]["divergence"] is None
    # The suite as the runs played it, and the teacher's own.
    assert found["copies"]["choice"]["suite"][200] == pytest.approx(0.65)
    assert found["copies"]["pilot-copy"]["suite"][360] == pytest.approx(0.49)
    assert found["teacher_suite"] == pytest.approx(0.74)
    # The line the teacher came from had what a copy has after so many updates.
    assert found["copies"]["choice"]["line_reached_after"] == TAUGHT
    assert found["line"][480] == pytest.approx(0.65)
    said = capsys.readouterr().out
    assert "choice over the pilot: +9.0 ± " in said
    assert "on those deals the pilot wins 60.0%, the pilot with its search 72.0%" in said
    assert "the teacher 75.0%" in said
    assert "a copy of the specialist is past the pilot" in said and ": yes" in said
    json.dumps(found)


def test_a_copy_under_its_marks_is_said_to_be(tmp_path):
    # 61% of the deals: a point over the pilot, and under halfway to the
    # teacher; the other copy wins as many.
    directory = copies(
        tmp_path, choice=[100, 110, 118, 122, 122], distribution=[100, 110, 118, 122, 122]
    )

    found = tool.report(directory, updates=200)

    assert found["differences"]["choice over pilot"][0] == pytest.approx(0.01)
    assert found["readings"] == {
        "past_the_pilot": False,
        "holds_its_share": False,
        "distribution_teaches_more": False,
    }


def test_a_copy_cut_short_is_shown_and_not_read(tmp_path, capsys):
    directory = copies(
        tmp_path, choice=[100, 120, 130], distribution=[104, 126, 136], points=POINTS[:3]
    )

    found = tool.report(directory, updates=200)

    assert found["checkpoints"] == {"measured": [40, 80, 120], "read": []}
    assert found["copies"]["choice"]["deals"][120] == pytest.approx(0.65)
    assert found["differences"] == {}
    assert set(found["readings"].values()) == {None}
    assert "measured as far as 120 updates of 200: they are not read" in capsys.readouterr().out


def test_a_teacher_measured_at_other_updates_is_not_set_beside_its_copies(tmp_path, capsys):
    directory = copies(
        tmp_path,
        choice=[100, 120, 130, 136, 140],
        distribution=[104, 126, 136, 142, 146],
        teacher_step=2160,
    )

    found = tool.report(directory, updates=200)

    assert "teacher_deals" not in found
    assert found["readings"]["past_the_pilot"] is True
    assert found["readings"]["holds_its_share"] is None
    assert "it is not set beside the copies" in capsys.readouterr().out


def test_what_was_measured_on_other_logged_games_is_left_out(tmp_path, capsys):
    """A copy is set beside nothing that played other deals."""
    directory = copies(
        tmp_path, choice=[100, 120, 130, 136, 140], distribution=[104, 126, 136, 142, 146]
    )
    measured(tmp_path, "pilot-copy-later", [240, 360], [116, 118], games="others")
    measured(tmp_path, "line.one-pass", [390, 480], [124, 130], games="others")
    measured(tmp_path, "line.one-pass-long.1920", [TAUGHT], [TEACHER])

    found = tool.report(directory, updates=200)

    assert found["measured_on_other_games"] == ["pilot-copy-later"]
    assert "pilot-copy-later" not in found["copies"]
    assert "choice over pilot-copy-later" not in found["differences"]
    assert sorted(found["line"]) == [*POINTS, TAUGHT]
    assert found["readings"]["past_the_pilot"] is True
    assert "left out: pilot-copy-later" in capsys.readouterr().out


def test_runs_with_nothing_measured_are_reported_as_far_as_they_go(tmp_path, capsys):
    teacher = {"path": "model.pt", "step": TAUGHT, "fingerprint": "the teacher"}
    run(tmp_path, "choice", 12, teacher=teacher)
    run(tmp_path, "distribution", 9, teacher=teacher, games="other")

    found = tool.report(tmp_path, updates=200)

    assert found["same_games"] is False
    assert found["copies"]["choice"]["learning"][0]["updates"] == 12
    assert set(found["readings"].values()) == {None}
    assert "not measured on the held-out games" in capsys.readouterr().out
    assert tool.report(tmp_path / "nothing")["readings"]["past_the_pilot"] is None
