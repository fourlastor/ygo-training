"""`tools/seventy_one_wide_report.py`: the seventy-one's copy twice as wide."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def loaded(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"tools/{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


tool = loaded("seventy_one_wide_report")
DECKS = ["blackwing", "cyber"]
HALF, COPY = 4, 8


def written(path: Path, suite: dict[str, list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    told = {name: {"outcomes": games, "by_opponent": {}} for name, games in suite.items()}
    path.write_text(json.dumps({"records": [{"evaluation_suite": told}]}), encoding="utf8")


def games(share: float, deals: int) -> tuple[list[str], list[str]]:
    """The first network wins a share of the deals: its games in the learner's seat, and
    the second network's there, which loses what the first wins."""
    won = round(share * deals)
    return ["win"] * won + ["loss"] * (deals - won), ["loss"] * won + ["win"] * (deals - won)


def pair(stem: Path, share: float, deals: int, cases: int = 1) -> None:
    """A pair of files, the deals over so many cases."""
    plays, held = games(share, deals)
    size = deals // cases
    for suffix, played in ((".plays.json", plays), (".held.json", held)):
        split = {f"case-{at:02d}": played[at * size : (at + 1) * size] for at in range(cases)}
        written(stem.with_name(stem.name + suffix), split)


def against_all71(path: Path, share: float, deals: int = 400) -> None:
    won = round(share * deals)
    outcomes = ["win"] * won + ["loss"] * (deals - won)
    written(path, {"a-vs-each": outcomes[: deals // 2], "b-vs-each": outcomes[deals // 2 :]})


def metrics(path: Path, agreement: dict[int, float], seconds: float) -> None:
    rows = [
        {
            "version": version,
            "imitation_model_teacher_agreement_by_deck": dict.fromkeys(DECKS, share),
            "imitation_samples_by_deck": dict.fromkeys(DECKS, 100),
            "wall_seconds": seconds,
        }
        for version, share in agreement.items()
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf8")


def measured(tmp_path: Path, share: float) -> tuple[Path, Path]:
    out, reference = tmp_path / "out", tmp_path / "reference"
    pair(out / "heads/copy", share, 20000, cases=4)
    for deck in DECKS:
        pair(out / "checks/wide" / deck, 0.48, 100)
        pair(out / "checks/taught" / deck, 0.46, 100)
        pair(out / "teachers" / deck, 0.49, 400)
        pair(reference / "heads-taught" / deck, 0.47, 400)
    against_all71(out / "checks/wide/pairings.json", 0.47)
    against_all71(out / "checks/taught/pairings.json", 0.45)
    # wide over taught by a point in the first half and by two in the second.
    metrics(
        out / "wide.metrics.jsonl",
        {version: 0.95 + (0.01 if version < HALF else 0.02) for version in range(COPY)},
        12.0,
    )
    metrics(reference / "taught.metrics.jsonl", dict.fromkeys(range(COPY), 0.95), 11.0)
    return out, reference


@pytest.mark.parametrize(
    ("share", "verdict"), [(0.53, "stronger"), (0.505, "not decided"), (0.47, "weaker")]
)
def test_the_wide_copy_is_read_head_to_head_against_the_copy(tmp_path, share, verdict):
    out, reference = measured(tmp_path, share)
    found = tool.report(out, reference, DECKS, HALF, COPY, stretch=1, lately=1)
    assert found["reading"] == verdict
    assert found["head to head"]["share"] == pytest.approx(share)
    # The deals of every case of the checks' pairings, each played both ways.
    assert found["head to head"]["deals"] == 20000
    assert f"1. The wide copy {verdict}" in tool.shown(found)


def test_the_checks_and_the_teachers_are_told_deal_by_deal(tmp_path):
    out, reference = measured(tmp_path, 0.5)
    found = tool.report(out, reference, DECKS, HALF, COPY, stretch=1, lately=1)
    mirrors, pairings = found["checks"]["mirrors"], found["checks"]["pairings"]
    assert mirrors["share"] == pytest.approx(0.48)
    assert mirrors["taught"] == pytest.approx(0.46)
    assert mirrors["gap"] == pytest.approx(0.02)
    assert "under" not in mirrors
    assert pairings["share"] == pytest.approx(0.47)
    assert pairings["taught"] == pytest.approx(0.45)
    assert pairings["gap"] == pytest.approx(0.02)
    teachers = found["teachers"]
    assert teachers["wide"]["share"] == pytest.approx(0.49)
    assert teachers["taught"]["share"] == pytest.approx(0.47)
    assert teachers["wide"]["deals"] == 800
    assert teachers["less taught"]["gap"] == pytest.approx(0.02)


def test_the_fit_is_told_by_halves_and_at_the_end(tmp_path):
    out, reference = measured(tmp_path, 0.5)
    found = tool.report(out, reference, DECKS, HALF, COPY, stretch=1, lately=1)
    fit = found["fit"]
    assert fit["first half"]["window"] == [1, HALF]
    assert fit["first half"]["gap"] == pytest.approx(0.01)
    assert fit["second half"]["window"] == [HALF + 1, COPY]
    assert fit["second half"]["gap"] == pytest.approx(0.02)
    # One update a deck: the last two.
    assert fit["lately"]["window"] == [COPY - 1, COPY]
    assert fit["lately"]["wide"] == pytest.approx(0.97)
    assert fit["lately"]["taught"] == pytest.approx(0.95)
    assert found["runs"]["wide"]["seconds an update"] == pytest.approx(12.0)
    assert found["runs"]["taught"]["hours"] == pytest.approx(8 * 11.0 / 3600)
    assert "the last 1 updates a deck (updates 7 to 8)" in tool.shown(found)


def test_the_copies_are_read_on_the_same_deals(tmp_path):
    out, reference = measured(tmp_path, 0.5)
    pair(reference / "heads-taught" / "cyber", 0.47, 200)
    with pytest.raises(SystemExit, match="cyber: the two copies were played on other deals"):
        tool.report(out, reference, DECKS, HALF, COPY, stretch=1, lately=1)


def test_an_update_missing_from_a_copy_is_not_read(tmp_path):
    out, reference = measured(tmp_path, 0.5)
    metrics(reference / "taught.metrics.jsonl", dict.fromkeys(range(COPY - 1), 0.95), 11.0)
    with pytest.raises(SystemExit, match="Update 8 is missing"):
        tool.report(out, reference, DECKS, HALF, COPY, stretch=1, lately=1)
