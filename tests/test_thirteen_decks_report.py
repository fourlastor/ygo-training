"""`tools/thirteen_decks_report.py`: the five readings of thirteen decks on
the new input, on measurements written as `PARTS=measure` of
`experiments/commands/thirteen-decks.sh` and the experiment on more decks
leave them."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
spec = importlib.util.spec_from_file_location(
    "thirteen_decks_report", ROOT / "tools/thirteen_decks_report.py"
)
tool = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(tool)

DECKS = ["blackwing", "monarch"]
# A copy of 2 updates a deck, arms of 6 of PPO with a checkpoint every one,
# set beside decks-13 after its 3.
SHAPE = {"copy": 2, "updates": 6, "midway": 3, "step": 1}
COPIED = len(DECKS) * SHAPE["copy"]
END, MIDDLE = [4, 5, 6], [1, 2, 3]
CHECKPOINT = "runs/thirteen-decks/{}/checkpoints/model-{:05d}.pt"


def won(count: int, of: int) -> list[str]:
    """Games of which the first so many are won."""
    return ["win"] * count + ["loss"] * (of - count)


def cases(deck: str, count: int, games: int = 100) -> dict:
    """A deck's screen, two cases of half its games: the first `count` won."""
    told = won(count, games)
    half = games // 2
    return {
        f"{deck}-vs-burn": {"games": half, "outcomes": told[:half]},
        f"{deck}-vs-heroes": {"games": half, "outcomes": told[half:]},
    }


def measured(path: Path, wins: dict[int, int] | None, deck: str, seed: int = 2012) -> None:
    """A file of evaluate-suite: the checkpoints of `wins` (updates in all ->
    games won), or the pilot where it is None."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if wins is None:
        records = [{"checkpoint_label": "pilot", "evaluation_suite": cases(deck, 50)}]
    else:
        records = [
            {"checkpoint_step": step, "evaluation_suite": cases(deck, count)}
            for step, count in wins.items()
        ]
    path.write_text(json.dumps({"seed": seed, "records": records}))


def screens(folder: Path, before: Path, **wins: int) -> None:
    """Every deck's screen, each player winning as many of its 100 games:
    the copies, half midway and at its end, the other arms at their end,
    decks-13 after the updates half is set beside it at."""
    for deck in DECKS:
        measured(folder / "screens" / f"copy.{deck}.json", {COPIED: wins["copy"]}, deck)
        measured(
            folder / "screens" / f"half.{deck}.json",
            {COPIED + point: wins["midway"] for point in MIDDLE}
            | {COPIED + point: wins["half"] for point in END},
            deck,
        )
        for arm in ("half-upgo", "itself"):
            measured(
                folder / "screens" / f"{arm}.{deck}.json",
                {COPIED + point: wins[arm] for point in END},
                deck,
            )
        measured(
            before / "screens" / f"decks-13.{deck}.json",
            {COPIED: wins["decks13_copy"]} | {COPIED + point: wins["decks13"] for point in MIDDLE},
            deck,
        )
        measured(before / "screens" / f"pilot.{deck}.json", None, deck)


def against(folder: Path, point: int, many: int, deals: int = 100) -> None:
    """Itself against half after `point` updates on every pairing: in its
    own seat it wins the first `many` deals, and held, the first `many`."""
    mine = CHECKPOINT.format("itself", COPIED + point - 1)
    theirs = CHECKPOINT.format("half", COPIED + point - 1)
    pairings = [
        "blackwing-vs-blackwing",
        "blackwing-vs-monarch",
        "monarch-vs-blackwing",
        "monarch-vs-monarch",
    ]
    for suffix, measured_, held, outcomes in (
        ("", mine, theirs, won(many, deals)),
        (".back", theirs, mine, ["loss"] * many + ["win"] * (deals - many)),
    ):
        quarter = deals // 4
        path = folder / "head-to-head" / f"itself-{point}{suffix}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "seed": 2012,
                    "opponent_checkpoint": {"path": held},
                    "records": [
                        {
                            "checkpoint": measured_,
                            "checkpoint_step": COPIED + point,
                            "evaluation_suite": {
                                name: {
                                    "games": quarter,
                                    "outcomes": outcomes[quarter * place : quarter * (place + 1)],
                                }
                                for place, name in enumerate(pairings)
                            },
                        }
                    ],
                }
            )
        )


def read(folder: Path, before: Path) -> dict:
    return tool.report(folder, before, DECKS, **SHAPE)


def test_the_five_readings(tmp_path):
    folder, before = tmp_path / "out", tmp_path / "before"
    screens(
        folder,
        before,
        copy=40,
        decks13_copy=42,
        midway=55,
        decks13=55,
        half=60,
        **{"half-upgo": 70, "itself": 50},
    )
    for point in END:
        against(folder, point, many=40)
    result = read(folder, before)
    readings = result["readings"]
    # 1. The same games won: equivalent, the copies set beside without a mark.
    assert readings["lists"]["gap"] == 0.0 and readings["lists"]["verdict"] == "equivalent"
    assert readings["lists"]["games"] == 200
    assert readings["lists"]["copies"]["gap"] == pytest.approx(-0.02)
    # 2. Five points more of 200 games: gained.
    assert readings["longer"]["gap"] == pytest.approx(0.05)
    assert readings["longer"]["verdict"] == "gained"
    # 3. and 4. Ten more, ten fewer.
    assert readings["upgo"]["gap"] == pytest.approx(0.1) and readings["upgo"]["verdict"] == "better"
    assert readings["itself"]["gap"] == pytest.approx(-0.1)
    assert readings["itself"]["verdict"] == "worse"
    # 5. Of each deal's six games itself wins all or none: 40% of them.
    found = readings["head to head"]
    assert found["share"] == pytest.approx(0.4) and found["games"] == 600
    assert found["verdict"] == "worse"
    assert result["head_to_head"] == {
        "4": pytest.approx(0.4),
        "5": pytest.approx(0.4),
        "6": pytest.approx(0.4),
    }
    assert result["screens"]["pilot"] == pytest.approx(0.5)
    assert result["by_deck"]["monarch"]["half"] == pytest.approx(0.6)
    told = tool.show(result)
    assert "1. half against decks-13 after 1, 2, 3: +0.0 ± 0.0 over 200 games: equivalent" in told
    assert "5. itself against half head to head: 40.0% ± " in told


@pytest.mark.parametrize(
    ("half", "verdict"),
    # Twenty points under; one, with an error that leaves it open.
    [(35, "worse"), (54, "not decided")],
)
def test_without_the_lists_is_worse_or_not_decided(tmp_path, half, verdict):
    folder, before = tmp_path / "out", tmp_path / "before"
    screens(
        folder,
        before,
        copy=40,
        decks13_copy=40,
        midway=half,
        decks13=55,
        half=60,
        **{"half-upgo": 60, "itself": 60},
    )
    readings = read(folder, before)["readings"]
    assert readings["lists"]["verdict"] == verdict
    assert readings["upgo"]["verdict"] == readings["itself"]["verdict"] == "not told apart"


def test_what_is_not_measured_is_told_so(tmp_path):
    folder, before = tmp_path / "out", tmp_path / "missing"
    for deck in DECKS:
        measured(folder / "screens" / f"copy.{deck}.json", {COPIED: 40}, deck)
    result = read(folder, before)
    assert result["readings"] == {} and result["screens"] == {"copy": pytest.approx(0.4)}
    told = tool.show(result)
    assert "1. half against decks-13: not measured" in told
    assert "5. itself against half head to head: not measured" in told


def test_other_games_are_refused(tmp_path):
    folder, before = tmp_path / "out", tmp_path / "before"
    screens(
        folder,
        before,
        copy=40,
        decks13_copy=40,
        midway=50,
        decks13=50,
        half=50,
        **{"half-upgo": 50, "itself": 50},
    )
    measured(before / "screens" / "pilot.monarch.json", None, "monarch", seed=2013)
    with pytest.raises(SystemExit, match="other games than the others"):
        read(folder, before)


def test_beside_the_readings(tmp_path):
    folder, before = tmp_path / "out", tmp_path / "before"
    (folder / "heldout").mkdir(parents=True)
    (folder / "heldout" / "half.json").write_text(
        json.dumps(
            {
                "checkpoints": [
                    {"checkpoint_step": COPIED + point, "deals": {"outcomes": won(rate, 10)}}
                    for point, rate in zip(END, (5, 6, 7), strict=True)
                ]
            }
        )
    )
    rows = [
        {
            "duels": 520.0,
            "wall_seconds": 26.0,
            "other_seat_win_rate": {"pilot": 0.6, "self": 0.5},
            "self_play_first_seat_win_rate": 0.55,
        }
    ] * 3
    (folder / "half.metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = read(folder, before)
    assert result["heldout"] == {"half": pytest.approx(0.6)}
    figures = result["figures"]["half"]
    assert figures["update_seconds"] == pytest.approx(26.0)
    assert figures["duels_an_hour"] == pytest.approx(72_000)
    assert figures["against_pilots"] == pytest.approx(0.6)
    assert figures["first_seat_against_itself"] == pytest.approx(0.55)
    assert "half" in tool.show(result)
