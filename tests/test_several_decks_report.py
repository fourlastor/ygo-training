"""The report of one network of several decks beside a network of each."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "several_decks_report", ROOT / "tools/several_decks_report.py"
)
assert SPEC is not None and SPEC.loader is not None
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)

DECKS = ["blackwing", "lightsworn"]
# A screen of two cases of 100 games: a pilot wins the first 60 of each.
GAMES = 100
PILOT = 60
POINTS = [40, 80, 120, 160, 200]


def run(
    directory: Path,
    name: str,
    updates: int,
    decks: list[str],
    *,
    start: str = "one",
    suite: int | None = None,
) -> None:
    """A run's figures over its updates, from scratch: a network of one deck,
    or of several in equal shares."""
    share = 520 // len(decks)
    rows = []
    for update in range(1, updates + 1):
        row = {
            "version": update - 1,
            "duels": 520.0,
            "samples": 40_000.0,
            "imitation_model_teacher_agreement": 0.9,
            "rollout_training_fingerprint": f"{name}-{update}",
        }
        if suite is not None and update % 40 == 0:
            row["evaluation_suite"] = {"a": {"wins": suite + update // 40, "games": 100}}
        if len(decks) > 1:
            row["learner_deck_duels"] = {deck: float(share) for deck in decks}
            row["imitation_samples_by_deck"] = {deck: 20_000.0 for deck in decks}
            row["imitation_model_teacher_agreement_by_deck"] = {
                deck: 0.88 + place / 100 for place, deck in enumerate(decks)
            }
        rows.append(row)
    (directory / f"{name}.metrics.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf8"
    )
    (directory / f"{name}.run.json").write_text(
        json.dumps({"initial_model_fingerprint": start, "learners": decks}), encoding="utf8"
    )


def screen(
    directory: Path, file: str, steps: list[int | None], won: list[int], *, seed: int = 2012
) -> None:
    """A measurement on a screen of two cases: after each checkpoint the
    network wins the first games of each case, as many as ``won`` says."""
    records = [
        {
            "checkpoint_step": step,
            "evaluation_suite": {
                case: {
                    "games": GAMES,
                    "wins": count,
                    "outcomes": ["win"] * count + ["loss"] * (GAMES - count),
                }
                for case in ("b-case", "a-case")
            },
        }
        for step, count in zip(steps, won, strict=True)
    ]
    folder = directory / "screens"
    folder.mkdir(exist_ok=True)
    (folder / f"{file}.json").write_text(
        json.dumps({"seed": seed, "records": records}), encoding="utf8"
    )


def measured(tmp_path: Path, *, together: dict[str, list[int]], points: list[int] = POINTS) -> Path:
    """Two decks: each alone, both together for twice the updates, and the
    pilots, on the screens."""
    for deck in DECKS:
        run(tmp_path, f"alone-{deck}", points[-1], [deck])
        screen(tmp_path, f"{deck}.pilot", [None], [PILOT])
        screen(tmp_path, f"{deck}.alone", points, [40, 48, 52, 54, 56][: len(points)])
        screen(tmp_path, f"{deck}.together", [2 * point for point in points], together[deck])
    run(tmp_path, "together", 2 * points[-1], DECKS)
    return tmp_path


def test_a_deck_is_read_beside_its_own_network_after_as_many_of_its_games(tmp_path, capsys):
    directory = measured(
        tmp_path,
        together={"blackwing": [38, 47, 52, 53, 56], "lightsworn": [30, 40, 44, 46, 48]},
    )
    run(tmp_path, "before", 300, ["blackwing"])
    (tmp_path / "before.metrics.jsonl").write_text(
        (tmp_path / "alone-blackwing.metrics.jsonl").read_text(encoding="utf8"), encoding="utf8"
    )

    found = tool.report(directory, DECKS, updates=200)

    assert found["same_start"] is True and found["blackwing_as_before"] is True
    assert "blackwing_suite_as_before" not in found
    assert found["together_plays"] == DECKS
    assert found["together_games"] == {"blackwing": 400 * 260.0, "lightsworn": 400 * 260.0}
    blackwing = found["decks"]["blackwing"]
    assert blackwing["pilot"] == pytest.approx(0.6) and blackwing["read"] == [160, 200]
    # A row is a number of the deck's games: 200 updates of its own network,
    # 400 of the network of both.
    last = blackwing["rows"][-1]
    assert (last["updates"], last["games"]) == (200, 104_000)
    assert (last["wins_alone"], last["wins_together"]) == (pytest.approx(0.56), pytest.approx(0.56))
    assert last["agrees_alone"] == pytest.approx(0.9)
    assert last["agrees_together"] == pytest.approx(0.88)
    assert found["decks"]["lightsworn"]["rows"][-1]["agrees_together"] == pytest.approx(0.89)
    # Blackwing: 54.5% together where its own network has 55%, over the last
    # two checkpoints; Lightsworn: 47% where its own has 55%.
    differences = found["differences"]
    assert differences["blackwing: together over alone"][0] == pytest.approx(-0.005)
    assert differences["blackwing: alone over pilot"][0] == pytest.approx(-0.05)
    assert differences["lightsworn: together over alone"][0] == pytest.approx(-0.08)
    assert differences["lightsworn: together over pilot"][0] == pytest.approx(-0.13)
    assert found["readings"] == {
        "holds_blackwing": True,
        "holds_lightsworn": False,
        "holds_every_deck": False,
    }
    said = capsys.readouterr().out
    assert "blackwing: its pilot wins 60.0% of its screen (200 games)." in said
    assert "one network holds the 2 decks as well as a network each: no" in said
    json.dumps(found)


def test_the_blackwing_network_is_checked_against_the_copy_made_before(tmp_path, capsys):
    """The same lesson on the same deals is the same network where the two
    ways of playing an update's games give the same numbers."""
    directory = measured(
        tmp_path,
        together={"blackwing": [38, 47, 52, 53, 56], "lightsworn": [30, 40, 44, 46, 48]},
    )
    run(tmp_path, "alone-blackwing", 200, ["blackwing"], suite=50)
    for name, suite, same in (("alone-blackwing", 50, True), ("other", 51, False)):
        rows = (tmp_path / "alone-blackwing.metrics.jsonl").read_text(encoding="utf8")
        if not same:
            run(tmp_path, "other", 360, ["blackwing"], suite=suite)
            rows = (tmp_path / "other.metrics.jsonl").read_text(encoding="utf8")
        (tmp_path / "before.metrics.jsonl").write_text(rows, encoding="utf8")

        found = tool.report(directory, DECKS, updates=200)

        assert found["blackwing_as_before"] is same
        assert found["blackwing_suite_as_before"] is same
    assert "evaluations in common: no (51.0% for 52.0%" in capsys.readouterr().out


def test_one_network_that_plays_every_deck_as_its_own_holds_them(tmp_path):
    directory = measured(
        tmp_path,
        together={"blackwing": [40, 48, 52, 55, 57], "lightsworn": [40, 48, 52, 53, 55]},
    )

    found = tool.report(directory, DECKS, updates=200)

    # A point over on one deck, a point under on the other: neither is worse.
    assert found["differences"]["blackwing: together over alone"][0] == pytest.approx(0.01)
    assert found["differences"]["lightsworn: together over alone"][0] == pytest.approx(-0.01)
    assert found["readings"] == {
        "holds_blackwing": True,
        "holds_lightsworn": True,
        "holds_every_deck": True,
    }


def test_networks_cut_short_are_shown_and_not_read(tmp_path, capsys):
    directory = measured(
        tmp_path,
        together={"blackwing": [38, 47, 52], "lightsworn": [30, 40, 44]},
        points=POINTS[:3],
    )

    found = tool.report(directory, DECKS, updates=200)

    assert found["decks"]["blackwing"]["read"] == []
    assert len(found["decks"]["blackwing"]["rows"]) == 3
    assert found["differences"] == {}
    assert set(found["readings"].values()) == {None}
    assert "measured as far as 120 updates of 200: the deck is not read" in capsys.readouterr().out


def test_measurements_of_other_games_are_not_set_beside_each_other(tmp_path, capsys):
    directory = measured(
        tmp_path,
        together={"blackwing": [38, 47, 52, 53, 56], "lightsworn": [30, 40, 44, 46, 48]},
    )
    screen(tmp_path, "lightsworn.together", [2 * p for p in POINTS], [30, 40, 44, 46, 48], seed=7)

    found = tool.report(directory, DECKS, updates=200)

    assert found["decks"]["lightsworn"]["same_games"] is False
    assert found["readings"] == {
        "holds_blackwing": True,
        "holds_lightsworn": None,
        "holds_every_deck": None,
    }
    assert "NOT of the same games" in capsys.readouterr().out


def test_runs_with_nothing_measured_are_reported_as_far_as_they_go(tmp_path, capsys):
    run(tmp_path, "alone-blackwing", 12, ["blackwing"])
    run(tmp_path, "together", 5, DECKS, start="another")

    found = tool.report(tmp_path, DECKS, updates=200)

    assert found["same_start"] is False
    assert set(found["readings"].values()) == {None}
    assert "measured at no update of 200: the deck is not read" in capsys.readouterr().out
    assert tool.report(tmp_path / "nothing", DECKS)["readings"]["holds_every_deck"] is None
