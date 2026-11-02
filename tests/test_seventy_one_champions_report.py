"""`tools/seventy_one_champions_report.py`: the readings of the network of
the seventy-one decks, on measurements written as
`experiments/commands/seventy-one-champions.sh` leaves them (three decks
here, for brevity)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from tests.test_four_decks_report import DEALS, DECKS, pair, played
from tests.test_seventy_one_checks import checks

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
spec = importlib.util.spec_from_file_location(
    "seventy_one_champions_report", ROOT / "tools/seventy_one_champions_report.py"
)
tool = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(tool)

PILOTS = ["blackwing", "burn"]


def won(wins: int) -> list[str]:
    return ["win"] * wins + ["loss"] * (DEALS - wins)


def against_each(path: Path, wins: dict[str, int]) -> None:
    """Each deck in the learner's seat against all71 playing each of the
    decks in one case, winning `wins[deck]` of each opponent's games."""
    cases = {}
    for one, share in wins.items():
        cases[f"{one}-vs-each-checkpoint"] = {
            "games": DEALS * len(DECKS),
            "outcomes": won(share) * len(DECKS),
            "by_opponent": {other: {"games": DEALS, "wins": share} for other in DECKS},
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"records": [{"evaluation_suite": cases}]}))


def measured(out: Path, *, heads: dict[str, int] | None = None) -> Path:
    """The copy measured as the script measures it: head to head against
    each teacher at `heads` (48 where not named), against all71 in each
    mirror at 44 and then 46, and against the pilots at 70 to all71's 60."""
    heads = heads or {}
    for deck in DECKS:
        pair(out / "heads-taught" / deck, heads.get(deck, 48))
        for point, share in ((4, 44), (8, 46)):
            pair(out / f"taught-{deck}" / str(point), share)
    for who, wins in (("taught", 70), ("all71", 60)):
        played(
            out / "pilots" / f"{who}.json",
            {f"{deck}-vs-{pilot}": won(wins) for deck in DECKS for pilot in PILOTS},
        )
    return out


def test_the_teachers_are_kept_lost_or_not_decided(tmp_path):
    def read(name: str, heads: dict[str, int]) -> dict:
        out = measured(tmp_path / name, heads=heads)
        return tool.report(out, DECKS, tmp_path / "none", tmp_path / "none")["teachers"]

    assert read("kept", dict.fromkeys(DECKS, 50))["verdict"] == "kept"
    # One mirror under the floor is lost, whatever the mean.
    under = read("under", {"cyber": 40})
    assert under["under the floor"] == ["cyber"] and under["verdict"] == "lost"
    level = read("level", {})
    assert level["mean"] == pytest.approx(0.48) and level["verdict"] == "not decided"


def test_each_mirror_is_told_beside_its_teachers_figure_along_the_copy(tmp_path):
    champions, teachers = tmp_path / "champions", tmp_path / "teachers"
    (champions / "blackwing").mkdir(parents=True)
    figure = {"share": 0.6, "error": 0.005, "deals": 2048, "verdict": "a champion"}
    (champions / "blackwing" / "report.json").write_text(json.dumps({"blackwing": figure}))
    # Of the champions' release, as the fourteen decks measured it.
    pair(teachers / "cyber", 55)

    result = tool.report(measured(tmp_path / "out"), DECKS, champions, teachers)

    against = result["against all71"]
    assert against["each"]["blackwing"]["teacher"]["from"] == "its making"
    assert against["each"]["blackwing"]["gap"] == pytest.approx(0.46 - 0.60)
    assert against["each"]["cyber"]["teacher"]["from"] == "the fourteen decks"
    assert against["each"]["cyber"]["gap"] == pytest.approx(0.46 - 0.55)
    assert against["unknown"] == ["lightsworn"]
    assert against["gap"] == pytest.approx((0.46 - 0.60 + 0.46 - 0.55) / 2)
    assert [point["updates"] for point in result["curve"]] == [4, 8]
    assert result["curve"][0]["mean"] == pytest.approx(0.44)
    assert result["pilots"]["mean"] == pytest.approx(0.70)
    assert result["pilots"]["gap"] == pytest.approx(0.10)
    assert result["hours"] is None

    shown = tool.shown(result, DECKS)
    assert "1. The teachers: not decided" in shown
    assert "No teacher's figure: lightsworn" in shown
    assert "The copy took not measured." in shown
    assert "PPO" not in shown


def test_ppo_is_read_by_its_mirrors_its_decks_against_all71_and_the_copy(tmp_path):
    out = measured(tmp_path / "out")
    for deck in DECKS:
        pair(out / "heads-ppo" / deck, 50)
    # Against all71 PPO wins 60 of a pairing's hundred games and the copy 55,
    # Cyber in the learner's seat 25 fewer each; the suite in two parts.
    for part, learners in enumerate((DECKS[:2], DECKS[2:]), start=1):
        for who, wins in (("ppo", 60), ("taught", 55)):
            against_each(
                out / "pairings" / f"{who}-{part}.json",
                {one: wins - 25 * (one == "cyber") for one in learners},
            )
    played(
        out / "pilots" / "ppo.json",
        {f"{deck}-vs-{pilot}": won(75) for deck in DECKS for pilot in PILOTS},
    )
    checks(out / "checks", (2880, 50, 52))

    result = tool.report(out, DECKS, tmp_path / "none", tmp_path / "none")
    found = result["ppo"]

    assert found["mirrors"]["verdict"] == "kept"
    against = found["against all71"]
    assert against["weakest"] == "cyber"
    # 35 of each hundred in the learner's seat, 60 of those it is played against.
    assert against["each"]["cyber"]["share"] == pytest.approx((3 * 35 + 2 * 60) / 500)
    assert against["verdict"] == "falls short"
    assert found["against the copy"]["gap"] == pytest.approx(0.05)
    assert found["against the copy"]["verdict"] == "pays"
    # 50 head to head where the copy had 48, on the same deals.
    lifted = found["mirrors against the copy"]
    assert lifted["gap"] == pytest.approx(0.02)
    assert lifted["verdict"] == "not decided"
    assert found["mirrors against all71"]["ppo"] == pytest.approx((0.60 + 0.35 + 0.60) / 3)
    assert found["mirrors against all71"]["taught"] == pytest.approx((0.55 + 0.30 + 0.55) / 3)
    assert found["pilots"]["mean"] == pytest.approx(0.75)
    assert found["pilots"]["against the copy"]["gap"] == pytest.approx(0.05)
    assert found["pilots"]["against all71"]["gap"] == pytest.approx(0.15)
    assert list(found["checks"]["checks"]) == [2880]
    shown = tool.shown(result, DECKS)
    assert "PPO, by the mean of its last thirteen kept checkpoints:" in shown
    assert "2. Against all71: falls short" in shown
    assert "3. Against the copy: pays, +5.0" in shown
    assert "4. The mirrors against the copy: not decided, +2.0" in shown
    assert "Each mirror less the copy's: blackwing +2.0" in shown
    assert "Its check after 2,880 updates against the copy's: the mirrors +0.0" in shown


def test_a_deck_is_read_from_both_seats_so_that_what_it_is_worth_cancels():
    # Level with all71, Cyber the weak deck: it wins 30 of a hundred against
    # the others whoever plays it, and they 70 against it.
    def wins(learner: str, opponent: str) -> int:
        if learner == opponent:
            return 50
        return {"cyber": 30}.get(learner, 70 if opponent == "cyber" else 50)

    cases = {
        f"{one}-vs-each-checkpoint": {
            "by_opponent": {other: {"games": DEALS, "wins": wins(one, other)} for other in DECKS}
        }
        for one in DECKS
    }

    found = tool.both_seats(cases, DECKS)

    assert all(one["share"] == pytest.approx(0.5) for one in found.values())
    assert found["cyber"]["deals"] == (2 * len(DECKS) - 1) * DEALS


@pytest.mark.parametrize(
    ("gap", "error", "verdict"),
    [(0.03, 0.005, "pays"), (0.005, 0.002, "does not pay"), (0.012, 0.004, "not decided")],
)
def test_ppo_pays_by_a_point_over_the_copy(gap, error, verdict):
    assert tool.pays(gap, error) == verdict


@pytest.mark.parametrize(
    ("gap", "error", "verdict"),
    [(0.03, 0.005, "raised"), (0.005, 0.002, "not raised"), (0.012, 0.004, "not decided")],
)
def test_ppo_raises_the_mirrors_by_a_point_over_the_copy(gap, error, verdict):
    assert tool.raised(gap, error) == verdict
