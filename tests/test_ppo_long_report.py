"""The report of the follow-up on PPO in one pass: the readings."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("ppo_long_report", ROOT / "tools/ppo_long_report.py")
assert SPEC is not None and SPEC.loader is not None
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)

# The start has had 360 updates; the pilot wins the first 120 of 200 deals.
BEGAN = 360
DEALS = 200
PILOT = 120


def run(
    directory: Path,
    name: str,
    updates: range,
    *,
    games: str = "same",
    seed: int = 2012,
    rate: float = 3e-4,
    suite_every: int = 30,
) -> None:
    """A run's figures over those updates, counted from the start's last."""
    rows = [
        {
            "version": BEGAN + update - 1,
            "duels": 520,
            "duel_outcome_explained_variance": 0.36,
            "duel_outcome_correlation": 0.6,
            "duel_approximate_kl": 0.02,
            "duel_entropy": 0.1,
            "learner_win_rate": 0.5 + update / 10_000,
            "rollout_decision_fingerprint": f"{games}-{update}",
            **(
                {"evaluation_suite": {"a": {"wins": 60, "games": 100}}}
                if update % suite_every == 0
                else {}
            ),
        }
        for update in updates
    ]
    (directory / f"{name}.metrics.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf8"
    )
    (directory / f"{name}.run.json").write_text(
        json.dumps({"training_seed": seed, "learning_rate": rate}), encoding="utf8"
    )


def measured(directory: Path, file: str, steps: list[int], won: list[int]) -> None:
    """Checkpoints on the 200 held-out deals: after each the model wins as
    many deals as ``won`` says, the first ones, as the pilot wins its own."""
    entries = [
        {
            "checkpoint_step": step,
            "deals": {
                "outcomes": ["win"] * count + ["loss"] * (DEALS - count),
                "paired": {
                    "pilot": {
                        "both": min(count, PILOT),
                        "model_only": max(0, count - PILOT),
                        "pilot_only": max(0, PILOT - count),
                        "neither": DEALS - max(count, PILOT),
                    }
                },
                "references": {
                    "pilot": {"outcomes": ["win"] * PILOT + ["loss"] * (DEALS - PILOT)},
                    "search": {"win_rate": 0.72},
                },
            },
            "answer_key": {
                "kept_left_rate": 0.15,
                "by_question": {kind: {"kept_left_rate": 0.05} for kind in tool.KINDS},
            },
        }
        for step, count in zip(steps, won, strict=True)
    ]
    folder = directory / "heldout"
    folder.mkdir(exist_ok=True)
    (folder / f"{file}.json").write_text(json.dumps({"checkpoints": entries}), encoding="utf8")


def follow_up(
    tmp_path: Path, *, seed: list[int], long: dict[int, int], rate: list[int] | None = None
) -> Path:
    """The first run and its start as they were, a second run, one at another
    rate, and the long run's checkpoints after the updates of ``long``."""
    short = [BEGAN + update for update in (30, 60, 90, 120)]
    measured(tmp_path, "start", [240, 280, 320, 360], [116, 116, 118, 118])
    run(tmp_path, "one-pass", range(1, 121))
    measured(tmp_path, "one-pass", short, [118, 122, 126, 130])
    run(tmp_path, "one-pass-seed", range(1, 121), games="other", seed=2013)
    measured(tmp_path, "one-pass-seed", short, seed)
    run(tmp_path, "one-pass-rate", range(1, 121), rate=1e-4)
    measured(tmp_path, "one-pass-rate", short, rate or [117, 118, 119, 120])
    if long:
        run(tmp_path, "one-pass-long", range(121, max(long) + 1), suite_every=120)
    for update, count in long.items():
        measured(tmp_path, f"one-pass-long.{update:04d}", [BEGAN + update], [count])
    return tmp_path


def test_the_readings_are_applied_as_the_record_fixes_them(tmp_path, capsys):
    directory = follow_up(
        tmp_path, seed=[118, 121, 124, 127], long={240: 132, 360: 136, 480: 138, 600: 140}
    )

    found = tool.report(directory, long=600)

    assert (found["rate_same_first_games"], found["seed_other_first_games"]) == (True, True)
    assert found["runs"]["one-pass-seed"]["training_seed"] == 2013
    assert found["runs"]["one-pass-rate"]["learning_rate"] == 1e-4
    assert found["checkpoints"]["read"] == [60, 90, 120]
    # The second run wins 62% of the deals where its start wins 58.5%.
    assert found["differences"]["one-pass-seed over start"][0] == pytest.approx(0.035)
    assert found["readings"]["second_run_gains"] is True
    # The long run is read by its last three checkpoints: 69% of the deals,
    # where it had 65% after its 120 updates and the pilot has 60%.
    assert found["long"]["read"] == [360, 480, 600]
    assert found["long"]["deals_read"] == pytest.approx(0.69)
    centre, error = found["differences"]["end over after"]
    assert (centre, error) == (pytest.approx(0.04), pytest.approx(0.01306, abs=1e-5))
    assert found["differences"]["end over pilot"][0] == pytest.approx(0.09)
    assert found["readings"] == {
        "second_run_gains": True,
        "long_holds": True,
        "long_adds": True,
        "past_the_pilot": True,
    }
    # The whole run in rows: the first run's checkpoints, then the long one's.
    rows = found["long"]["checkpoints"]
    assert [row["updates"] for row in rows] == [30, 60, 90, 120, 240, 360, 480, 600]
    assert rows[4]["games"] == 240 * 520
    # A checkpoint against the pilot from the count of who won each deal.
    assert rows[3]["over_the_pilot"][0] == pytest.approx(0.05)
    assert rows[0]["over_the_pilot"][0] == pytest.approx(-0.01)
    # The games won while sampling, over the updates since the row before.
    assert rows[4]["sampled_wins"] == pytest.approx(0.5 + (121 + 240) / 2 / 10_000)
    assert (rows[3]["suite"], rows[4]["suite"]) == (0.6, 0.6)
    printed = capsys.readouterr().out
    assert "The first update of one-pass-rate has the games of one-pass: yes" in printed
    assert "its games drawn with seed 2013" in printed
    assert "on those deals the pilot wins 60.0%" in printed
    assert "the pilot with its search wins 72.0% of those deals" in printed
    assert "its checkpoints after 360, 480, 600 updates (69.0% of the deals)" in printed


def test_a_run_that_holds_without_adding_and_a_second_that_does_not_gain(tmp_path):
    directory = follow_up(
        tmp_path, seed=[116, 117, 118, 117], long={240: 130, 360: 131, 480: 129, 600: 130}
    )

    found = tool.report(directory, long=600)

    # A sixth of a point over the start is not a gain.
    assert found["readings"]["second_run_gains"] is False
    assert found["differences"]["end over after"][0] == pytest.approx(0.0)
    assert (found["readings"]["long_holds"], found["readings"]["long_adds"]) == (True, False)
    # Five points over the pilot all the same.
    assert found["readings"]["past_the_pilot"] is True


def test_a_long_run_that_comes_undone(tmp_path):
    directory = follow_up(
        tmp_path, seed=[118, 121, 124, 127], long={240: 126, 360: 118, 480: 112, 600: 108}
    )

    readings = tool.report(directory, long=600)["readings"]

    assert (readings["long_holds"], readings["long_adds"]) == (False, False)
    assert readings["past_the_pilot"] is False


def test_a_long_run_measured_part_of_the_way_is_not_read(tmp_path, capsys):
    directory = follow_up(tmp_path, seed=[118, 121, 124, 127], long={240: 132, 360: 136, 480: 138})

    found = tool.report(directory, long=600)

    assert found["long"]["measured"] == [30, 60, 90, 120, 240, 360, 480]
    assert found["readings"]["second_run_gains"] is True
    assert (found["readings"]["long_holds"], found["readings"]["long_adds"]) == (None, None)
    assert found["readings"]["past_the_pilot"] is None
    printed = capsys.readouterr().out
    assert "measured as far as 480 updates of 600: its readings are not taken" in printed
    assert "PPO goes on adding (not measured" in printed


def test_the_short_runs_alone(tmp_path, capsys):
    directory = follow_up(tmp_path, seed=[118, 121, 124, 127], long={})

    found = tool.report(directory, long=600)

    assert found["readings"]["second_run_gains"] is True
    assert found["readings"]["long_adds"] is None
    # At their rate the copy wins half a point more than its start, and
    # 3.5 points fewer than the first run.
    assert found["differences"]["one-pass-rate over one-pass"][0] == pytest.approx(-0.035)
    assert "no checkpoint of one-pass-long is measured" in capsys.readouterr().out


def test_a_long_run_set_after_another_runs_updates_is_not_read(tmp_path, capsys):
    """A folder that pairs a long run with the first updates of another run
    (the script, before it took a run's name, made one so): the two
    descriptions tell it, and the long run's readings are not taken."""
    folder = follow_up(
        tmp_path,
        seed=[118, 121, 124, 127],
        long={240: 132, 360: 136, 480: 138, 600: 140},
    )
    # The long run went on from the run at the other rate.
    run(folder, "one-pass-long", range(121, 601), rate=1e-4, suite_every=120)

    found = tool.report(folder, long=600)

    assert found["long"]["one_run"] is False
    assert found["readings"]["long_holds"] is None and found["readings"]["long_adds"] is None
    assert found["readings"]["past_the_pilot"] is None
    printed = capsys.readouterr().out
    assert "NOT ONE RUN: one-pass-long does not go on from one-pass" in printed
    assert "a learning rate of 0.0003 and 0.0001" in printed
    assert "its readings are not taken" in printed
    # Other games tell it as well.
    run(folder, "one-pass-long", range(121, 601), seed=2013, suite_every=120)
    assert tool.report(folder, long=600)["long"]["one_run"] is False
    # The run it does go on from is read.
    run(folder, "one-pass-long", range(121, 601), suite_every=120)
    again = tool.report(folder, long=600)
    assert again["long"]["one_run"] is True and again["readings"]["long_adds"] is True


def test_another_run_made_long_is_set_beside_the_first_long_run(tmp_path, capsys):
    """A second long run goes on from another run than `one-pass`: it is read
    as the first was, and beside it after the same updates."""
    (tmp_path / "first").mkdir()
    first = follow_up(
        tmp_path / "first",
        seed=[118, 121, 124, 127],
        long={240: 132, 360: 136, 480: 138, 600: 140},
    )
    again = tmp_path / "again"
    again.mkdir()
    short = [BEGAN + update for update in (30, 60, 90, 120)]
    measured(again, "start", [240, 280, 320, 360], [116, 116, 118, 118])
    run(again, "one-pass-rate", range(1, 121), rate=1e-4)
    measured(again, "one-pass-rate", short, [120, 126, 130, 132])
    run(again, "one-pass-rate-long", range(121, 601), rate=1e-4, suite_every=120)
    for update, count in {240: 138, 360: 144, 480: 146, 600: 148}.items():
        measured(again, f"one-pass-rate-long.{update:04d}", [BEGAN + update], [count])

    found = tool.report(again, long=600, first="one-pass-rate", beside=first)

    # The run as one: the rate's 120 updates, then its continuation.
    assert found["long"]["measured"] == [30, 60, 90, 120, 240, 360, 480, 600]
    assert found["long"]["deals_read"] == pytest.approx(0.73)
    assert found["differences"]["end over after"][0] == pytest.approx(0.07)
    # Beside the first long run after the same updates, and at the end: 73%
    # of the deals where the first wins 69%.
    assert found["beside"]["updates"] == [30, 60, 90, 120, 240, 360, 480, 600]
    assert found["beside"]["over"][600][0] == pytest.approx(0.04)
    assert found["beside"]["over"][30][0] == pytest.approx(0.01)
    assert found["differences"]["end over the first long run"][0] == pytest.approx(0.04)
    assert found["readings"] == {
        "second_run_gains": None,
        "long_holds": True,
        "long_adds": True,
        "past_the_pilot": True,
        "over_the_first_long_run": True,
    }
    printed = capsys.readouterr().out
    assert "The long run: one-pass-rate and one-pass-rate-long" in printed
    assert "Beside the first long run" in printed
    assert "run long, it ends over the first long run (+4.0 ± " in printed
    # Without the first long run beside it, it is read for itself.
    alone = tool.report(again, long=600, first="one-pass-rate")
    assert "over_the_first_long_run" not in alone["readings"] and "beside" not in alone
    # A second run that ends where the first did is not over it.
    for update, count in {360: 136, 480: 138, 600: 140}.items():
        measured(again, f"one-pass-rate-long.{update:04d}", [BEGAN + update], [count])
    level = tool.report(again, long=600, first="one-pass-rate", beside=first)
    assert level["differences"]["end over the first long run"][0] == pytest.approx(0.0)
    assert level["readings"]["over_the_first_long_run"] is False
