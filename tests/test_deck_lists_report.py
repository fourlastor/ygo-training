"""`tools/deck_lists_report.py`: the readings of the experiment on the model
without the deck lists, on measurements written as the measuring part of
`experiments/commands/without-deck-lists.sh` leaves them."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "deck_lists_report", ROOT / "tools/deck_lists_report.py"
)
tool = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(tool)


def won(count: int, of: int) -> list[str]:
    """Games of which the first so many are won."""
    return ["win"] * count + ["loss"] * (of - count)


def heldout(out: Path, name: str, count: int, fingerprint: str = "deals") -> None:
    """A checkpoint that wins the first `count` of 200 held-out deals."""
    (out / "heldout").mkdir(parents=True, exist_ok=True)
    (out / "heldout" / f"{name}.json").write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "games": 200,
                "checkpoints": [{"deals": {"outcomes": won(count, 200)}}],
            }
        )
    )


def suite(out: Path, kind: str, file: str, count: int, seed: int = 2012) -> None:
    """A checkpoint on a suite of two cases of 50 games, the first `count`
    of the hundred won."""
    (out / kind).mkdir(parents=True, exist_ok=True)
    games = won(count, 100)
    cases = {
        "a-vs-x": {"games": 50, "outcomes": games[:50]},
        "a-vs-y": {"games": 50, "outcomes": games[50:]},
    }
    (out / kind / f"{file}.json").write_text(
        json.dumps({"seed": seed, "records": [{"evaluation_suite": cases}]})
    )


def measured(out: Path, own: int = 120, outside: int = 60, three: int = 55) -> Path:
    """The controls win 120 of the 200 deals, 60 of the hundred games outside
    training and 55 of each screen's hundred; the networks that read less
    win what is given (the network of three 120 and 60 beside them)."""
    heldout(out, "decks-01-copy", 120)
    heldout(out, "blackwing-own-list", own)
    heldout(out, "decks-03-copy", 120)
    heldout(out, "three-no-lists", 120)
    heldout(out, "blackwing-no-history", 120)
    for name, count in (
        ("decks-01-copy", 60),
        ("blackwing-own-list", outside),
        ("decks-03-copy", 60),
        ("three-no-lists", 60),
        ("blackwing-no-history", 60),
    ):
        suite(out, "unseen", name, count)
    for deck in tool.THREE:
        suite(out, "screens", f"decks-03-copy.{deck}", 55)
        suite(out, "screens", f"three-no-lists.{deck}", three)
    return out


def test_the_readings_are_taken_by_their_marks(tmp_path, capsys):
    found = tool.report(measured(tmp_path, own=110, outside=57, three=55))
    readings = found["readings"]
    # Ten deals fewer of 200: 5 points, by more than two standard errors.
    first = readings["1. the other player's list, on the decks of training"]
    assert first["gap"] == pytest.approx(-0.05)
    assert first["error"] == pytest.approx((0.05 * 0.95 / 199) ** 0.5)
    assert first["verdict"] == "worse"
    # Three games fewer of a hundred: past the mark but within two standard
    # errors, and too uncertain to say the list can go.
    assert readings["2. the other player's list, outside training"]["verdict"] == "not decided"
    # The same games won on each of three screens: equivalent, over 300.
    third = readings["3. both lists, in a network of three, on its screens"]
    assert (third["gap"], third["games"], third["verdict"]) == (0.0, 300, "equivalent")
    assert found["beside"]["both lists, in a network of three, on the held-out deals"]["gap"] == 0.0
    assert (
        readings["4. its own earlier answers, on the decks of training"]["verdict"] == "equivalent"
    )
    assert readings["5. its own earlier answers, outside training"]["games"] == 100
    told = capsys.readouterr().out
    assert "1. the other player's list, on the decks of training: -5.0 ± 1.5 over 200 games" in told
    assert "blackwing-own-list 55.0%, decks-01-copy 60.0%): worse" in told
    assert "a network of three, outside training: +0.0 ± 0.0 over 100 games" in told
    assert "decks-03-copy 60.0%): told without a mark" in told


def test_a_network_over_its_control_is_equivalent(tmp_path):
    found = tool.report(measured(tmp_path, own=126))
    first = found["readings"]["1. the other player's list, on the decks of training"]
    assert first["gap"] == pytest.approx(0.03) and first["verdict"] == "equivalent"


def test_what_is_not_there_is_not_read_and_other_games_are_refused(tmp_path, capsys):
    partial = measured(tmp_path / "partial")
    (partial / "screens" / "three-no-lists.gishki.json").unlink()
    found = tool.report(partial)
    assert found["readings"]["3. both lists, in a network of three, on its screens"] is None
    assert (
        "3. both lists, in a network of three, on its screens: not measured"
        in capsys.readouterr().out
    )
    other = measured(tmp_path / "other")
    heldout(other, "blackwing-own-list", 120, fingerprint="other deals")
    with pytest.raises(SystemExit, match="not measured on the same games"):
        tool.report(other)
    seeded = measured(tmp_path / "seeded")
    suite(seeded, "unseen", "blackwing-own-list", 60, seed=2013)
    with pytest.raises(SystemExit, match="not measured on the same games"):
        tool.report(seeded)


def test_the_runs_say_whether_a_copy_began_and_played_as_its_control(tmp_path, capsys):
    out = measured(tmp_path / "out")
    runs = tmp_path / "runs"
    runs.mkdir()

    def run(name: str, start: str, games: list[str], agreement: float) -> None:
        (runs / f"{name}.run.json").write_text(json.dumps({"initial_model_fingerprint": start}))
        rows = [
            {"rollout_decision_fingerprint": game, "imitation_model_teacher_agreement": agreement}
            for game in games
        ]
        (runs / f"{name}.metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

    run("decks-01-copy", "start", ["a", "b", "c"], 0.95)
    run("blackwing-own-list", "start", ["a", "b", "c"], 0.94)
    run("decks-03-copy", "start", ["a", "b", "c"], 0.95)
    run("three-no-lists", "other start", ["a", "x", "c"], 0.95)
    run("blackwing-no-history", "start", ["a", "b"], 0.93)
    found = tool.report(out, runs)["runs"]
    # One update fewer than its control: an update of other games.
    assert found["blackwing-no-history"]["other_games"] == 1
    assert found["blackwing-own-list"]["same_start"] is True
    assert found["blackwing-own-list"]["other_games"] == 0
    assert found["three-no-lists"]["same_start"] is False
    assert found["three-no-lists"]["other_games"] == 1
    told = capsys.readouterr().out
    assert (
        "blackwing-own-list began as decks-01-copy did: yes; updates of other games: 0 of 3; "
        in told
    )
    assert "answers as its pilots at its last update 94.0% (decks-01-copy 95.0%)" in told
    assert "three-no-lists began as decks-03-copy did: NO; updates of other games: 1 of 3" in told
