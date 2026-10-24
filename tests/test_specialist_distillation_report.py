"""`tools/specialist_distillation_report.py`: the readings of specialist
distillation, on measurements written as
`experiments/commands/specialist-distillation.sh` leaves them."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
spec = importlib.util.spec_from_file_location(
    "specialist_distillation_report", ROOT / "tools/specialist_distillation_report.py"
)
tool = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(tool)

DEALS = 100


def played(path: Path, outcomes: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    case = {"games": len(outcomes), "outcomes": outcomes}
    path.write_text(json.dumps({"records": [{"evaluation_suite": {"some-vs-checkpoint": case}}]}))


def pair(stem: Path, share: int) -> None:
    """A pair in which the first network wins `share` of the deals' games
    in all: `share` deals in the learner's seat, and as many in the other."""
    played(stem.with_name(f"{stem.name}.plays.json"), ["win"] * share + ["loss"] * (DEALS - share))
    lost = DEALS - share
    played(stem.with_name(f"{stem.name}.held.json"), ["win"] * lost + ["loss"] * share)


def run(folder: Path, name: str, pairs: str, hours: list[float]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    told = {"learners": ["blackwing", "cyber"], "self_play_duels": 260}
    if pairs != "all":
        told["self_play_pairs"] = pairs
    (folder / f"{name}.run.json").write_text(json.dumps(told))
    rows = [json.dumps({"wall_seconds": 3600 * each}) for each in hours]
    (folder / f"{name}.metrics.jsonl").write_text("\n".join(rows) + "\n")


def measured(
    tmp_path: Path,
    *,
    copy: tuple[int, int] = (50, 50),
    ours: tuple[int, ...] = (50, 55, 60, 64, 66, 67, 68),
) -> tuple[Path, Path, Path]:
    """The run's measurements, the network of both's and the Cyber
    champion's: the copy's share against all71 in each mirror, the teachers'
    at 50%, and the two curves in Blackwing against Cyber, the network of
    both's last three at 64%."""
    directory, both, cyber = tmp_path / "ours", tmp_path / "both", tmp_path / "cyber"
    for point in (320, 400, 480):
        pair(directory / "taught-blackwing" / str(point), copy[0])
        pair(directory / "taught-cyber" / str(point), copy[1])
        pair(directory / "teacher-blackwing" / str(point), 47)
        pair(directory / "teacher-cyber" / str(point), 45)
    for name in ("blackwing-720", "cyber-720"):
        pair(directory / "check" / name, 50)
    for point, share in zip(range(0, 1441, 240), ours, strict=True):
        pair(directory / "cross" / str(point), share)
        if point:
            pair(directory / "cross-blackwing" / str(point), 49)
            pair(directory / "cross-cyber" / str(point), 51)
    for point, share in zip(range(0, 2881, 480), (29, 55, 58, 61, 64, 64, 64), strict=True):
        pair(both / "cross" / str(point), share)
    run(directory, "taught", "mirrors", [0.25] * 4)
    # An update's row each, 7.5 seconds of the clock.
    run(directory, "cross", "cross", [3.0 / 1440] * 1440)
    run(both, "both", "all", [1.0] * 6)
    for name, hours in (("both-copy", [0.5]), ("blackwing-copy", [0.25]), ("blackwing", [1.5])):
        (both / f"{name}.metrics.jsonl").write_text(
            "\n".join(json.dumps({"wall_seconds": 3600 * each}) for each in hours) + "\n"
        )
    cyber.mkdir(parents=True, exist_ok=True)
    for name, hours in (("cyber-old-copy", [0.25]), ("cyber-old", [2.0])):
        (cyber / f"{name}.metrics.jsonl").write_text(
            "\n".join(json.dumps({"wall_seconds": 3600 * each}) for each in hours) + "\n"
        )
    return directory, both, cyber


def test_the_copy_keeps_a_champions_mirror_or_loses_strength(tmp_path: Path) -> None:
    result = tool.report(*measured(tmp_path, copy=(60, 30)))
    blackwing, cyber = result["mirrors"]["Blackwing"], result["mirrors"]["Cyber"]
    assert (blackwing["copy"], blackwing["its teacher"]) == (pytest.approx(0.6), 0.5)
    assert blackwing["gap"] == pytest.approx(0.1)
    assert blackwing["verdict"] == "keeps"
    assert cyber["gap"] == pytest.approx(-0.2) and cyber["verdict"] == "loses strength"
    assert cyber["teacher"] == "cyber-old"
    # The rule at its edges: -2 points, by two standard errors.
    assert tool.keeps(-0.01, 0.004) == "keeps"
    assert tool.keeps(-0.01, 0.006) == "not decided"
    assert tool.keeps(-0.04, 0.009) == "loses strength"
    assert tool.keeps(-0.04, 0.011) == "not decided"
    head = result["beside"]["Cyber, head to head"]
    assert head["share"] == pytest.approx(0.45)


def test_the_games_of_the_two_decks_it_needs_beside_the_network_of_boths(tmp_path: Path) -> None:
    cross = tool.report(*measured(tmp_path))["cross"]
    assert cross["reference"] == pytest.approx(0.64)
    # Its checkpoint after 720 updates is the first at the network of both's
    # last three; 480, four points under, is not within two.
    assert (cross["updates"], cross["games"]) == (720, 720 * 260)
    assert cross["verdict"] == "closed after 187,200 games of the two decks against each other"
    assert [point["within"] for point in cross["curve"]] == [False] * 3 + [True] * 4
    # The network of both, two decks against each other in 130 of its duels
    # an update, by the same rule.
    assert (cross["updates of both"], cross["games of both"]) == (1920, 1920 * 130)
    assert [point["games"] for point in cross["curve of both"]] == [
        point * 130 for point in range(0, 2881, 480)
    ]


def test_a_gap_not_closed_within_the_network_of_boths_games(tmp_path: Path) -> None:
    cross = tool.report(*measured(tmp_path, ours=(50, 52, 54, 56, 58, 60, 61)))["cross"]
    assert cross["games"] is None and cross["updates"] is None
    assert cross["verdict"] == "not closed within the network of both's games"


def test_the_hours_of_the_way_to_the_reading_and_of_the_whole(tmp_path: Path) -> None:
    result = tool.report(*measured(tmp_path))
    hours = result["hours"]
    assert (hours["Blackwing alone"], hours["cyber-old"]) == (1.75, 2.25)
    assert (hours["taught"], hours["cross"]) == (1.0, 3.0)
    # Met after 720 updates of cross: the first half of its hours.
    assert hours["cross, to the second reading"] == 1.5
    assert hours["the whole way, the champions one after the other"] == 1.75 + 2.25 + 1.0 + 1.5
    assert hours["the whole way, the champions at once"] == 2.25 + 1.0 + 1.5
    assert hours["the network of both"] == 6.5
    shown = tool.shown(result)
    assert "1. The copy keeps" in shown and "187,200" in shown and "249,600" in shown


def test_each_mirror_along_the_cross_pairings_is_told_over_its_teacher(tmp_path: Path) -> None:
    along = tool.report(*measured(tmp_path))["beside"]["each mirror along cross"]
    assert [point["updates"] for point in along["Blackwing"]] == list(range(240, 1441, 240))
    assert along["Cyber"][0]["over its teacher"]["gap"] == pytest.approx(0.01)


def test_the_games_played_again_end_as_those_measured_before(tmp_path: Path) -> None:
    directory, both, cyber = measured(tmp_path)
    pair(directory / "check" / "cross-1920", 64)
    pair(both / "all71-alone" / "720", 50)
    pair(cyber / "cyber-old" / "mirror" / "model-00959", 50)
    found = tool.check(directory, both, cyber)
    assert [name for name, _, _ in found] == [
        f"{name}.{side}"
        for name in ("cross-1920", "blackwing-720", "cyber-720")
        for side in ("plays", "held")
    ]
    assert all(same == games == DEALS for _, same, games in found)
    pair(cyber / "cyber-old" / "mirror" / "model-00959", 48)
    assert [same for _, same, _ in tool.check(directory, both, cyber)][-2:] == [98, 98]
