"""`tools/more_decks_report.py`: the three readings of the experiment on one
network for more decks, on measurements written as the measuring part of
`experiments/commands/more-decks.sh` leaves them."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("more_decks_report", ROOT / "tools/more_decks_report.py")
tool = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(tool)

DECKS = ["blackwing", "lightsworn", "gishki"]
SIZES = [1, 2, 3]
# A copy of 2 updates a deck, then 6 of PPO with a checkpoint every 2.
SHAPE = {"copy": 2, "updates": 6, "step": 2}


def won(count: int, of: int) -> list[str]:
    """Games of which the first so many are won."""
    return ["win"] * count + ["loss"] * (of - count)


def steps(size: int) -> dict[str, int]:
    return tool.steps(size, SHAPE["copy"], SHAPE["updates"], SHAPE["step"])


def heldout(folder: Path, size: int, copy: int, three: int, fingerprint: str = "deals") -> None:
    """A network on 200 held-out deals: its copy wins the first `copy`, each
    of its last three checkpoints the first `three`."""
    (folder / "heldout").mkdir(parents=True, exist_ok=True)
    wins = {tool.COPY: copy} | {name: three for name in steps(size) if name != tool.COPY}
    (folder / "heldout" / f"{tool.network(size)}.json").write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "games": 200,
                "checkpoints": [
                    {"checkpoint_step": step, "deals": {"outcomes": won(wins[name], 200)}}
                    for name, step in steps(size).items()
                ],
            }
        )
    )


def suite(deck: str, count: int) -> dict:
    """A deck's screen of two cases of 50 games, the first `count` of the
    hundred won."""
    games = won(count, 100)
    return {
        f"{deck}-vs-burn": {"games": 50, "outcomes": games[:50]},
        f"{deck}-vs-monarch": {"games": 50, "outcomes": games[50:]},
    }


def screen(folder: Path, size: int, deck: str, copy: int, three: int, seed: int = 2012) -> None:
    (folder / "screens").mkdir(parents=True, exist_ok=True)
    wins = {tool.COPY: copy} | {name: three for name in steps(size) if name != tool.COPY}
    (folder / "screens" / f"{tool.network(size)}.{deck}.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "records": [
                    {"checkpoint_step": step, "evaluation_suite": suite(deck, wins[name])}
                    for name, step in steps(size).items()
                ],
            }
        )
    )


def pilot(folder: Path, deck: str, count: int) -> None:
    (folder / "screens").mkdir(parents=True, exist_ok=True)
    (folder / "screens" / f"pilot.{deck}.json").write_text(
        json.dumps({"seed": 2012, "records": [{"checkpoint_label": "pilot", "evaluation_suite": suite(deck, count)}]})
    )


def measured(folder: Path, cost: tuple[int, int] = (110, 118), largest: tuple[int, int] = (60, 70), pilots: int = 65) -> Path:
    """Blackwing's network wins 120 of 200 held-out deals, the others what
    `cost` says; the network of all three decks wins `largest` (its copy,
    its last three) of each screen's hundred games, where each pilot wins
    `pilots`."""
    heldout(folder, 1, 100, 120)
    for size, three in zip(SIZES[1:], cost, strict=True):
        heldout(folder, size, 100, three)
    for size in SIZES:
        for deck in DECKS[:size]:
            screen(folder, size, deck, *(largest if size == 3 else (50, 55)))
    for deck in DECKS:
        pilot(folder, deck, pilots)
    return folder


def test_the_readings_are_taken_by_their_marks(tmp_path, capsys):
    found = tool.report(measured(tmp_path), SIZES, DECKS, **SHAPE)
    assert found["read_by"] == [2, 4, 6]
    cost = found["readings"]["cost"]
    # Ten deals fewer of 200 at each of the last three: 5 points, by more
    # than two standard errors; two fewer: one point, under the mark.
    assert cost["decks-02"]["gap"] == pytest.approx(-0.05)
    assert cost["decks-02"]["error"] == pytest.approx((0.05 * 0.95 / 199) ** 0.5)
    assert cost["decks-02"]["worse"] is True
    assert cost["decks-03"]["gap"] == pytest.approx(-0.01) and cost["decks-03"]["worse"] is False
    gained = found["readings"]["gained"]
    assert gained["gap"] == pytest.approx(0.10) and gained["games"] == 300
    assert gained["gained"] is True
    over = found["readings"]["pilots"]
    assert over["gap"] == pytest.approx(0.05) and (over["over"], over["under"]) == (True, False)
    told = capsys.readouterr().out
    assert "1. decks-02 on blackwing, beside decks-01: worse (-5.0 ± 1.5 on the held-out deals)" in told
    assert "1. decks-03 on blackwing, beside decks-01: not shown worse (-1.0" in told
    assert "2. decks-03 has gained on its copy: yes (+10.0" in told and "300 games" in told
    assert "3. decks-03 is over the decks' own pilots: yes (+5.0" in told
    # The tables: a network's copy and its last three, and each deck's pilot.
    assert found["networks"]["decks-01"]["heldout"]["copy"] == pytest.approx(0.5)
    assert found["networks"]["decks-01"]["heldout"]["three"] == pytest.approx(0.6)
    assert found["screens"]["gishki"]["pilot"] == pytest.approx(0.65)
    assert found["screens"]["gishki"]["decks-03"] == {"copy": 0.6, "three": 0.7}
    assert "decks-01" not in found["screens"]["gishki"]


def test_a_network_under_its_copy_and_its_pilots_is_told_so(tmp_path):
    found = tool.report(measured(tmp_path, largest=(70, 69), pilots=75), SIZES, DECKS, **SHAPE)
    gained = found["readings"]["gained"]
    assert gained["gap"] == pytest.approx(-0.01) and gained["gained"] is False
    pilots = found["readings"]["pilots"]
    assert pilots["gap"] == pytest.approx(-0.06) and (pilots["over"], pilots["under"]) == (False, True)


def test_what_is_not_there_is_not_read_and_what_differs_is_refused(tmp_path, capsys):
    # The network of all the decks not measured yet: the first reading only.
    partial = measured(tmp_path / "partial")
    for path in (partial / "screens").glob("decks-03.*.json"):
        path.unlink()
    (partial / "heldout" / "decks-03.json").unlink()
    found = tool.report(partial, SIZES, DECKS, **SHAPE)
    assert set(found["readings"]["cost"]) == {"decks-02"}
    assert "gained" not in found["readings"] and "pilots" not in found["readings"]
    told = capsys.readouterr().out
    assert "  decks-03   not measured\n" in told
    assert "2. decks-03 on its copy: not measured" in told
    # Held-out deals that are not the same deals.
    other = measured(tmp_path / "other")
    heldout(other, 2, 100, 110, fingerprint="other deals")
    with pytest.raises(SystemExit, match="not measured on the same deals"):
        tool.report(other, SIZES, DECKS, **SHAPE)
    # A screen of other games.
    seeded = measured(tmp_path / "seeded")
    screen(seeded, 3, "blackwing", 60, 70, seed=2013)
    with pytest.raises(SystemExit, match="blackwing is not measured on the same games throughout"):
        tool.report(seeded, SIZES, DECKS, **SHAPE)
    # A measurement without the checkpoint asked for.
    missing = measured(tmp_path / "missing")
    with pytest.raises(SystemExit, match="decks-01.json: no checkpoint after 10 updates in all"):
        tool.report(missing, SIZES, DECKS, copy=2, updates=8, step=2)
    with pytest.raises(SystemExit, match="runs up from 1"):
        tool.report(missing, [2, 3], DECKS, **SHAPE)


def test_how_fast_each_learns_is_told_by_updates_and_by_seats(tmp_path, capsys):
    """Beside the readings: each network's other checkpoints of its PPO on
    the held-out deals, by its updates and by the seats of the first deck it
    had played by then."""
    shape = {"copy": 2, "updates": 10, "step": 2}
    for size, wins in ((1, {2: 110, 4: 115}), (2, {2: 104, 4: 108})):
        wanted = tool.steps(size, **shape)
        (tmp_path / "heldout").mkdir(exist_ok=True)
        (tmp_path / "heldout" / f"{tool.network(size)}.json").write_text(
            json.dumps(
                {
                    "fingerprint": "deals",
                    "games": 200,
                    "checkpoints": [
                        {"checkpoint_step": step, "deals": {"outcomes": won(100 if name == tool.COPY else 120, 200)}}
                        for name, step in wanted.items()
                    ],
                }
            )
        )
        (tmp_path / "heldout" / f"{tool.network(size)}.curve.json").write_text(
            json.dumps(
                {
                    "fingerprint": "deals",
                    "games": 200,
                    "checkpoints": [
                        {"checkpoint_step": size * 2 + point, "deals": {"outcomes": won(count, 200)}}
                        for point, count in wins.items()
                    ],
                }
            )
        )
        # An update: 26 duels against the pilots and 26 against itself.
        if size == 1:
            row = {"other_seat_duels": {"pilot": 26.0, "self": 26.0}}
        else:
            row = {
                "other_seat_duels": {"pilot": 26.0, "self": 26.0},
                "learner_deck_duels": {"blackwing": 13.0, "lightsworn": 13.0},
                "self_play_deck_seats": {"blackwing": 26.0, "lightsworn": 26.0},
            }
        (tmp_path / f"{tool.network(size)}.metrics.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for _ in range(10))
        )
    found = tool.report(tmp_path, [1, 2], ["blackwing", "lightsworn"], **shape)
    one = found["networks"]["decks-01"]["curve"]
    assert {point: told["rate"] for point, told in one.items()} == {
        "0": 0.5, "2": 0.55, "4": 0.575, "6": 0.6, "8": 0.6, "10": 0.6,
    }  # fmt: skip
    # One deck: a seat against each pilot and two against itself, 78 an
    # update; of two decks, half of them.
    assert one["4"]["seats"] == 4 * 78
    assert found["networks"]["decks-02"]["curve"]["4"]["seats"] == 4 * 39
    told = capsys.readouterr().out
    assert "How fast each learns: the held-out deals of blackwing, by its updates of PPO (0 is its copy)." in told
    assert "         2     55.0%     52.0%" in told
    assert "The same by the seats of blackwing it had played by then, in thousands." in told
