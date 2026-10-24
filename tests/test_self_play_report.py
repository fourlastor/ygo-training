"""What the base of the self-play experiment says of its folder."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from ygo_training.evaluation_suite import against_checkpoint, load_evaluation_suite

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "self_play_report", ROOT / "tools/self_play_report.py"
)
assert SPEC is not None and SPEC.loader is not None
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)

START = "/runs/long/checkpoints/model-02279.pt"


def measured(label: str, step: int | None, wins: int, checkpoint: str | None = None) -> dict:
    record = {
        "checkpoint_label": label,
        "evaluation_suite": {tool.AGAINST: {"wins": wins, "games": 2048}},
    }
    if step is not None:
        record.update(checkpoint_step=step, checkpoint=checkpoint or f"/runs/long/{label}.pt")
    return record


def update(version: int, duels: int, won: float, evaluation: tuple[int, int] | None = None) -> dict:
    """A row of an exploiter's metrics: its duels against the start, and
    with an evaluation what it wins against the start and against a pilot."""
    row = {
        "version": version,
        "duels": float(duels),
        "other_seat_duels": {"checkpoint": float(duels)},
        "other_seat_win_rate": {"checkpoint": won},
    }
    if evaluation is not None:
        row["evaluation_suite"] = {
            "blackwing-vs-blackwing": {"wins": evaluation[1], "games": 128},
            "blackwing-vs-monarch": {"wins": 64, "games": 128},
            tool.AGAINST: {"wins": evaluation[0], "games": 2048},
        }
    return row


def folder(tmp_path: Path) -> Path:
    held = {"path": START, "step": 2280, "fingerprint": "f"}
    (tmp_path / "head-to-head").mkdir()
    (tmp_path / "head-to-head" / "start.json").write_text(
        json.dumps(
            {
                "opponent_checkpoint": held,
                "records": [
                    measured("model-01319", 1320, 820),
                    measured("model-02279", 2280, 1030, START),
                    measured("model-00839", 840, 717),
                ],
            }
        )
    )
    (tmp_path / "head-to-head" / "pilot.json").write_text(
        json.dumps({"opponent_checkpoint": held, "records": [measured("pilot", None, 512)]})
    )
    rows = {
        2012: [
            update(2280, 520, 0.50),
            update(2281, 520, 0.54, (1100, 100)),
            update(2282, 520, 0.56),
            update(2283, 520, 0.60, (1200, 96)),
        ],
        2013: [update(2280, 520, 0.48), update(2281, 520, 0.52, (1126, 90))],
    }
    for seed, updates in rows.items():
        (tmp_path / f"exploiter-{seed}.metrics.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in updates)
        )
        (tmp_path / f"exploiter-{seed}.run.json").write_text(
            json.dumps({"training_seed": seed, "opponent_checkpoint": held, "learning_rate": 1e-4})
        )
    return tmp_path


def test_the_base_is_told_from_its_folder(tmp_path, capsys):
    found = tool.report(folder(tmp_path))
    # The pilot, then the checkpoints in the order of their updates.
    assert [(player["player"], player["itself"]) for player in found["head_to_head"]] == [
        ("pilot", False),
        ("model-00839", False),
        ("model-01319", False),
        ("model-02279", True),
    ]
    one = found["exploiters"]["exploiter-2012"]
    assert (one["seed"], one["victim"], one["learning_rate"]) == (2012, START, 1e-4)
    assert one["points"] == [
        {
            "updates": 2,
            "games": 1040,
            "against": [1100, 2048],
            "pilots": [164, 256],
            "sampled": pytest.approx(0.52),
        },
        {
            "updates": 4,
            "games": 2080,
            "against": [1200, 2048],
            "pilots": [160, 256],
            # Of the duels since the evaluation before, not of all.
            "sampled": pytest.approx(0.58),
        },
    ]
    assert one["last"] == {"evaluations": 2, "rate": pytest.approx(2300 / 4096)}
    assert found["exploiters"]["exploiter-2013"]["last"]["rate"] == pytest.approx(1126 / 2048)
    assert (one["of"], one["victim_step"]) == (tool.START, 2280)
    assert found["apart"] == {tool.START: pytest.approx(2300 / 4096 - 1126 / 2048)}
    # Nothing of the arms is there, and no exploiter with every deal from
    # either seat: no reading is spoken of.
    assert "readings" not in found and "exploited" not in found and "found_again" not in found
    told = capsys.readouterr().out
    assert "the pilot" in told and "25.0% ± 1.0" in told
    assert "after 2280 updates in all (the start itself)" in told and "50.3% ± 1.1" in told
    assert "exploiter-2012, of the start (training seed 2012" in told
    assert "The exploiters of the start end 1.2 points apart." in told
    assert "leaves less for an exploiter" not in told


def test_a_measurement_that_failed_is_not_read(tmp_path):
    base = folder(tmp_path)
    path = base / "head-to-head" / "pilot.json"
    kept = json.loads(path.read_text())
    kept["records"][0]["evaluation_suite"][tool.AGAINST] = {"error": "the duel failed"}
    path.write_text(json.dumps(kept))
    with pytest.raises(SystemExit, match="pilot failed: the duel failed"):
        tool.report(base)


def test_the_suites_against_a_checkpoint_deal_the_screens_mirror():
    """The case against a held checkpoint is the same 2,048 deals in the
    suite of its own and beside the screen, and its first are the deals of
    the screen's mirror against the pilot."""
    alone = load_evaluation_suite(ROOT / "eval_suites/blackwing-against-checkpoint.json")
    beside = load_evaluation_suite(
        ROOT / "eval_suites/ygo-policies-blackwing-screen-and-checkpoint.json"
    )
    screen = load_evaluation_suite(ROOT / "eval_suites/ygo-policies-blackwing-screen.json")
    assert [against_checkpoint(case) for case in alone["cases"]] == [True]
    assert beside["cases"][:-1] == screen["cases"] and beside["primary"] == screen["primary"]
    held = beside["cases"][-1]
    assert against_checkpoint(held) and not any(map(against_checkpoint, screen["cases"]))
    assert {key: value for key, value in held.items() if key != "seed_index"} == alone["cases"][0]
    # The place a case is dealt from: its own, or the one it names.
    assert held["seed_index"] == 0 == beside["cases"].index(screen["cases"][0])
    assert (screen["cases"][0]["learner"], screen["cases"][0]["opponent"]) == (
        held["learner"],
        held["opponent"],
    )


RUNS = "/card/runs/self-play"


def of(arm: str, updates: int) -> str:
    """The checkpoint of an arm after so many updates of its own."""
    return f"{RUNS}/{arm}/checkpoints/model-{2279 + updates:05d}.pt"


def deals(won: int, of_: int = 200) -> list[str]:
    """Held-out deals of which the first so many are won."""
    return ["win"] * won + ["loss"] * (of_ - won)


def heldout(folder: Path, name: str, outcomes: list[str], fingerprint: str = "deals") -> None:
    (folder / "heldout").mkdir(parents=True, exist_ok=True)
    (folder / "heldout" / f"{name}.json").write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "games": len(outcomes),
                "checkpoints": [{"deals": {"outcomes": outcomes}}],
            }
        )
    )


def evaluated(version: int, pilots: int, against: int) -> dict:
    """A row of an arm's metrics with its evaluation."""
    return {
        "version": version,
        "duels": 520.0,
        "evaluation_suite": {
            "blackwing-vs-blackwing": {"wins": pilots, "games": 128},
            tool.AGAINST: {"wins": against, "games": 2048},
        },
    }


def mirror(folder: Path, forward: list[list[str]], back: list[list[str]]) -> None:
    """With-itself's three checkpoints against pilots-alone's last held, and
    that one against each of the three held."""
    (folder / "head-to-head").mkdir(parents=True, exist_ok=True)
    held = {"path": of(tool.PILOTS, 6), "step": 2286}
    (folder / "head-to-head" / f"{tool.ITSELF}.json").write_text(
        json.dumps(
            {
                "seed": 2012,
                "suite": "against.json",
                "opponent_checkpoint": held,
                "records": [
                    {
                        "checkpoint": of(tool.ITSELF, updates),
                        "evaluation_suite": {tool.AGAINST: {"outcomes": outcomes}},
                    }
                    for updates, outcomes in zip((2, 4, 6), forward, strict=True)
                ],
            }
        )
    )
    for updates, outcomes in zip((2, 4, 6), back, strict=True):
        (folder / "head-to-head" / f"{tool.PILOTS}.{updates:04d}.json").write_text(
            json.dumps(
                {
                    "seed": 2012,
                    "suite": "against.json",
                    "opponent_checkpoint": {"path": of(tool.ITSELF, updates)},
                    "records": [
                        {
                            "checkpoint": of(tool.PILOTS, 6),
                            "evaluation_suite": {tool.AGAINST: {"outcomes": outcomes}},
                        }
                    ],
                }
            )
        )


def arms(folder: Path, mine: tuple[int, int, int] = (114, 114, 114)) -> Path:
    """The start wins 120 of 200 held-out deals, pilots-alone 124 at each of
    its checkpoints, and with-itself what is given at its last three."""
    heldout(folder, "start", deals(120))
    for updates, won in zip((2, 4, 6), mine, strict=True):
        heldout(folder, f"{tool.ITSELF}.{updates:04d}", deals(won))
        heldout(folder, f"{tool.PILOTS}.{updates:04d}", deals(124))
    for name in tool.ARMS:
        (folder / f"{name}.metrics.jsonl").write_text(
            "".join(
                json.dumps(row) + "\n"
                for row in (
                    {"version": 2280, "duels": 520.0},
                    evaluated(2281, 96, 1100),
                    {"version": 2282, "duels": 520.0},
                    evaluated(2283, 100, 1200),
                )
            )
        )
    return folder


def test_self_play_has_lost_against_the_pilots_at_two_points_and_two_errors(tmp_path, capsys):
    """Six deals of 200 fewer than the start at each of the last three
    checkpoints: 3 points under, by more than two standard errors."""
    found = tool.report(arms(tmp_path), 6, 2)
    assert found["arms"]["last"] == [2, 4, 6] and found["arms"]["start"] == pytest.approx(0.6)
    kept = found["readings"]["kept"]
    assert kept["rate"] == pytest.approx(0.57) and kept["gap"] == pytest.approx(-0.03)
    # Six deals differ by one and the others by nothing.
    assert kept["error"] == pytest.approx((0.03 * 0.97 / 199) ** 0.5)
    assert kept["lost"] is True
    assert kept["pilots_alone"][0] == pytest.approx(0.02)
    assert kept["between"][0] == pytest.approx(-0.05)
    assert "mirror" not in found["readings"]
    told = capsys.readouterr().out
    assert "1. Self-play keeps what it has against the pilots: no (57.0%" in told
    assert "-3.0 ± 1.2 on the start" in told
    assert "2. Self-play makes the better player of the mirror: not measured" in told
    # The arms' figures: the suite against the pilots and the mirror against the start.
    assert "75.0%" in told and "78.1%" in told and "58.6%" in told


def test_self_play_keeps_what_is_under_either_mark(tmp_path):
    # Three deals fewer: a point and a half, under the two points.
    assert (
        tool.report(arms(tmp_path / "a", (117, 117, 117)), 6, 2)["readings"]["kept"]["lost"]
        is False
    )
    # Three points under on the average, and not by two standard errors:
    # one checkpoint wins other deals than the start does.
    scattered = tmp_path / "b"
    arms(scattered)
    heldout(scattered, f"{tool.ITSELF}.0002", ["loss"] * 80 + ["win"] * 120)
    heldout(scattered, f"{tool.ITSELF}.0004", ["loss"] * 80 + ["win"] * 120)
    heldout(scattered, f"{tool.ITSELF}.0006", ["win"] * 102 + ["loss"] * 98)
    kept = tool.report(scattered, 6, 2)["readings"]["kept"]
    assert kept["gap"] == pytest.approx(-0.03) and kept["error"] > 0.015
    assert kept["lost"] is False
    # More than the start is no loss.
    assert (
        tool.report(arms(tmp_path / "c", (130, 130, 130)), 6, 2)["readings"]["kept"]["lost"]
        is False
    )


def test_with_itself_is_under_the_control_at_a_point_and_two_errors(tmp_path, capsys):
    """Pilots-alone wins 124 of 200 held-out deals at each checkpoint.  Ten
    deals fewer at each of with-itself's last three: 5 points under, by more
    than two standard errors."""
    under = tool.report(arms(tmp_path / "under"), 6, 2)["readings"]["under"]
    assert under["gap"] == pytest.approx(-0.05)
    # Ten deals differ by one and the others by nothing.
    assert under["error"] == pytest.approx((0.05 * 0.95 / 199) ** 0.5)
    assert (under["under"], under["over"]) == (True, False)
    told = capsys.readouterr().out
    assert "Against the pilots, With itself is under Pilots alone: yes (-5.0 ± 1.5 on it" in told
    # One deal fewer: half a point, under the mark of one.
    close = tool.report(arms(tmp_path / "close", (123, 123, 123)), 6, 2)["readings"]["under"]
    assert close["gap"] == pytest.approx(-0.005)
    assert (close["under"], close["over"]) == (False, False)
    # Two points under on the average, and not by two standard errors: it
    # wins other deals than the control does.
    scattered = tmp_path / "scattered"
    arms(scattered)
    for updates in (2, 4, 6):
        heldout(scattered, f"{tool.ITSELF}.{updates:04d}", ["loss"] * 80 + ["win"] * 120)
    apart = tool.report(scattered, 6, 2)["readings"]["under"]
    assert apart["gap"] == pytest.approx(-0.02) and apart["error"] > 0.05
    assert (apart["under"], apart["over"]) == (False, False)
    assert "is under Pilots alone: not shown (-2.0 ± " in capsys.readouterr().out
    # Ten deals more: over the control, by the same marks.
    over = tool.report(arms(tmp_path / "over", (134, 134, 134)), 6, 2)["readings"]["under"]
    assert over["gap"] == pytest.approx(0.05)
    assert (over["under"], over["over"]) == (False, True)
    assert "is under Pilots alone: no: over it (+5.0 ± 1.5 on it" in capsys.readouterr().out
    assert tool.UNDER == 0.01 and tool.ERRORS == 2.0


def test_an_arm_is_read_only_when_it_is_whole_and_on_the_same_deals(tmp_path, capsys):
    # Read for more updates than it has had: its last checkpoints are not there.
    found = tool.report(arms(tmp_path / "short"), 8, 2)
    assert found["readings"] == {}
    assert "keeps what it has against the pilots: not measured" in capsys.readouterr().out
    other = arms(tmp_path / "other")
    heldout(other, f"{tool.ITSELF}.0006", deals(114), fingerprint="other deals")
    with pytest.raises(SystemExit, match="not of the same held-out games"):
        tool.report(other, 6, 2)


def test_the_mirror_is_every_deal_from_either_seat(tmp_path, capsys):
    """Ten deals.  Measured, with-itself wins the first seven at each of its
    checkpoints; held, it loses the first four to pilots-alone and wins the
    other six.  A deal is its share of the six games."""
    forward = [["win"] * 7 + ["loss"] * 3] * 3
    back = [["win"] * 4 + ["loss"] * 6] * 3
    mirror(arms(tmp_path / "ten"), forward, back)
    played = tool.report(tmp_path / "ten", 6, 2)["readings"]["mirror"]
    assert (played["deals"], played["games"]) == (10, 60)
    # Deals 0-3 one of two, deals 4-6 both, deals 7-9 one of two.
    assert played["share"] == pytest.approx(0.65)
    assert played["error"] == pytest.approx(((7 * 0.15**2 + 3 * 0.35**2) / 9 / 10) ** 0.5)
    # Measured it opens the even deals (0, 2, 4 and 6 won) and plays second
    # in the odd ones (1, 3 and 5 won); held it opens the odd ones (5, 7 and
    # 9 won) and plays second in the even ones (4, 6 and 8 won).
    assert played["opening"] == pytest.approx((4 + 3) / 10)
    assert played["second"] == pytest.approx((3 + 3) / 10)
    # Fifteen points over half, a hair under two standard errors of ten deals.
    assert (played["better"], played["worse"]) == (False, False)
    assert "of the mirror: not shown (65.0% ± 7.6 of 60 games" in capsys.readouterr().out
    # The same shares of a hundred deals are past both marks.
    forward = [["win"] * 70 + ["loss"] * 30] * 3
    back = [["win"] * 40 + ["loss"] * 60] * 3
    mirror(arms(tmp_path / "hundred"), forward, back)
    played = tool.report(tmp_path / "hundred", 6, 2)["readings"]["mirror"]
    assert played["share"] == pytest.approx(0.65) and played["error"] == pytest.approx(
        0.02303, abs=1e-4
    )
    assert (played["better"], played["worse"]) == (True, False)
    assert "2. Self-play makes the better player of the mirror: yes (65.0% ± 2.3" in (
        capsys.readouterr().out
    )


def test_the_mirror_is_better_or_worse_only_past_both_marks(tmp_path):
    def read(folder: Path, forward: list[str], back: list[str]) -> dict:
        mirror(arms(folder), [forward] * 3, [back] * 3)
        return tool.report(folder, 6, 2)["readings"]["mirror"]

    # 51% of the games, however surely: under the two points.
    level = read(tmp_path / "a", ["win"] * 51 + ["loss"] * 49, ["loss"] * 51 + ["win"] * 49)
    assert level["share"] == pytest.approx(0.51) and level["error"] == pytest.approx(
        0.0502, abs=1e-3
    )
    assert (level["better"], level["worse"]) == (False, False)
    # 55% of ten deals: past the two points and not by two standard errors.
    few = read(tmp_path / "b", ["win"] * 6 + ["loss"] * 4, ["loss"] * 5 + ["win"] * 5)
    assert few["share"] == pytest.approx(0.55) and few["error"] > 0.025
    assert (few["better"], few["worse"]) == (False, False)
    # 40% of two hundred deals, the same in either seat: worse.
    under = read(tmp_path / "c", ["win"] * 80 + ["loss"] * 120, ["loss"] * 80 + ["win"] * 120)
    assert under["share"] == pytest.approx(0.4)
    assert (under["better"], under["worse"]) == (False, True)
    # A drawn game is half a win for each.
    drawn = read(tmp_path / "d", ["draw"] * 10, ["draw"] * 10)
    assert drawn["share"] == pytest.approx(0.5) and drawn["error"] == 0.0


def test_the_mirror_is_read_only_of_the_two_arms_last_checkpoints(tmp_path):
    forward = [["win", "loss"]] * 3
    # One of the three is missing: not measured.
    partial = arms(tmp_path / "partial")
    mirror(partial, forward, forward)
    (partial / "head-to-head" / f"{tool.PILOTS}.0004.json").unlink()
    assert "mirror" not in tool.report(partial, 6, 2)["readings"]
    # Another checkpoint measured than the one held the other way round.
    crossed = arms(tmp_path / "crossed")
    mirror(crossed, forward, forward)
    path = crossed / "head-to-head" / f"{tool.PILOTS}.0006.json"
    told = json.loads(path.read_text())
    told["records"][0]["checkpoint"] = of(tool.PILOTS, 4)
    path.write_text(json.dumps(told))
    with pytest.raises(SystemExit, match="on the deals of"):
        tool.report(crossed, 6, 2)
    # Not the two arms at all.
    swapped = arms(tmp_path / "swapped")
    mirror(swapped, forward, forward)
    path = swapped / "head-to-head" / f"{tool.ITSELF}.json"
    told = json.loads(path.read_text())
    told["opponent_checkpoint"]["path"] = of(tool.ITSELF, 6)
    path.write_text(json.dumps(told))
    with pytest.raises(SystemExit, match="is not with-itself's checkpoints against pilots-alone's"):
        tool.report(swapped, 6, 2)


def test_the_decks_outside_training_are_told_game_by_game(tmp_path, capsys):
    def record(checkpoint: str, first: list[str], second: list[str]) -> dict:
        return {
            "checkpoint": checkpoint,
            "evaluation_suite": {"b": {"outcomes": second}, "a": {"outcomes": first}},
        }

    folder = arms(tmp_path)
    won, lost = ["win"] * 4, ["loss"] * 4
    (folder / "unseen.json").write_text(
        json.dumps(
            {
                "records": [
                    record("/card/runs/long/checkpoints/model-02279.pt", won, lost),
                    *(record(of(tool.ITSELF, updates), won, won) for updates in (2, 4, 6)),
                    *(record(of(tool.PILOTS, updates), won, lost) for updates in (2, 4)),
                    record(of(tool.PILOTS, 6), lost, lost),
                ]
            }
        )
    )
    outside = tool.report(folder, 6, 2)["unseen"]
    assert (outside["games"], outside["start"]) == (8, 0.5)
    assert outside[tool.ITSELF] == 1.0 and outside[tool.PILOTS] == pytest.approx(1 / 3)
    assert outside["over_start"][tool.ITSELF][0] == pytest.approx(0.5)
    assert outside["between"][0] == pytest.approx(2 / 3)
    assert "The decks outside training, 8 games: the start 50.0%" in capsys.readouterr().out


def exploiters_of(folder: Path, victim: str, step: int, label: str = "") -> None:
    """An exploiter of that checkpoint, which has had so many updates, for
    each of two training seeds: its description and its figures."""
    for seed in (2012, 2013):
        name = f"exploiter-{label}{seed}"
        (folder / f"{name}.run.json").write_text(
            json.dumps(
                {
                    "training_seed": seed,
                    "opponent_checkpoint": {"path": victim, "step": step},
                    "learning_rate": 1e-4,
                }
            )
        )
        (folder / f"{name}.metrics.jsonl").write_text(
            "".join(
                json.dumps(row) + "\n"
                for row in (update(step, 520, 0.5), update(step + 1, 520, 0.55, (1100, 100)))
            )
        )


def after(name: str, step: int, updates: int) -> str:
    """The checkpoint of a run that began as one of so many updates, after
    so many of its own."""
    return f"{RUNS}/{name}/checkpoints/model-{step + updates - 1:05d}.pt"


def paired(
    folder: Path, name: str, one: str, step: int, other: str, forward: list[str], back: list[str]
) -> None:
    """A checkpoint, which has had so many updates, measured against another
    held, and that one measured against it held."""
    head = folder / "head-to-head"
    head.mkdir(parents=True, exist_ok=True)
    for suffix, measured_, held, outcomes in (
        ("", {"checkpoint": one, "checkpoint_step": step}, other, forward),
        (".back", {"checkpoint": other}, one, back),
    ):
        (head / f"{name}{suffix}.json").write_text(
            json.dumps(
                {
                    "seed": 2012,
                    "suite": "against.json",
                    "opponent_checkpoint": {"path": held},
                    "records": [
                        {**measured_, "evaluation_suite": {tool.AGAINST: {"outcomes": outcomes}}}
                    ],
                }
            )
        )


def beats(folder: Path, name: str, one: str, step: int, other: str, won: int, of_: int) -> None:
    """The first checkpoint wins the first so many of the deals against the
    other, from either seat."""
    paired(folder, name, one, step, other, deals(won, of_), deals(of_ - won, of_)[::-1])


def exploited(folder: Path, mine: tuple[int, int], theirs: tuple[int, int], of_: int = 100) -> Path:
    """Two exploiters of each arm's checkpoint after 6 updates, each 6
    updates with a checkpoint every 2: at each of its last three an
    exploiter wins the first so many of the deals from either seat."""
    arms(folder)
    for arm, shares in ((tool.ITSELF, mine), (tool.PILOTS, theirs)):
        exploiters_of(folder, of(arm, 6), 2286, f"{arm}-")
        for seed, won in zip((2012, 2013), shares, strict=True):
            name = f"exploiter-{arm}-{seed}"
            for point in (2, 4, 6):
                beats(
                    folder,
                    f"{name}.{point:04d}",
                    after(name, 2286, point),
                    2286 + point,
                    of(arm, 6),
                    won,
                    of_,
                )
    return folder


def test_an_exploiter_of_a_longer_arm_is_known_in_a_folder_named_by_the_arms_length(tmp_path):
    """The exploiters of an arm's checkpoint after a longer arm have run
    folders of their own, `exploiter-ARM-SEED.LENGTH`: they are the
    exploiters named, and a folder named otherwise is not."""
    folder = exploited(tmp_path / "longer", (60, 62), (70, 72))
    name = f"exploiter-{tool.ITSELF}-2012"
    for point in (2, 4, 6):
        beats(
            folder, f"{name}.{point:04d}", after(f"{name}.1920", 2286, point), 2286 + point,
            of(tool.ITSELF, 6), 60, 100,
        )  # fmt: skip
    found = tool.report(folder, 6, 2, 6, 2)
    assert found["exploited"]["victims"][tool.ITSELF]["rate"] == pytest.approx(0.61)
    other = exploited(tmp_path / "other", (60, 62), (70, 72))
    beats(other, f"{name}.0006", after(f"{name}.again", 2286, 6), 2292, of(tool.ITSELF, 6), 60, 100)
    with pytest.raises(SystemExit, match=f"{name}.0006.json is not {name} after 6 updates"):
        tool.report(other, 6, 2, 6, 2)
    beats(other, f"{name}.0006", after(f"{name}0.6", 2286, 6), 2292, of(tool.ITSELF, 6), 60, 100)
    with pytest.raises(SystemExit, match=f"{name}.0006.json is not {name} after 6 updates"):
        tool.report(other, 6, 2, 6, 2)


def test_self_play_leaves_less_at_three_points_two_errors_and_each_under_each(tmp_path, capsys):
    """A hundred deals.  With-itself's exploiters win 60 and 62 of them from
    either seat, pilots-alone's 70 and 72: ten points fewer."""
    found = tool.report(exploited(tmp_path, (60, 62), (70, 72)), 6, 2, 6, 2)
    told = found["exploited"]
    assert told["last"] == [2, 4, 6]
    mine = told["victims"][tool.ITSELF]
    assert mine["exploiters"][f"exploiter-{tool.ITSELF}-2012"]["rate"] == pytest.approx(0.60)
    assert mine["exploiters"][f"exploiter-{tool.ITSELF}-2013"]["rate"] == pytest.approx(0.62)
    assert mine["rate"] == pytest.approx(0.61)
    assert told["victims"][tool.PILOTS]["rate"] == pytest.approx(0.71)
    read = found["readings"]["exploited"]
    assert (read[tool.ITSELF], read[tool.PILOTS]) == (pytest.approx(0.61), pytest.approx(0.71))
    assert read["gap"] == pytest.approx(-0.10)
    # Of the hundred deals, 88 fall alike to the two arms' exploiters, 4
    # differ by half a game and 8 by a whole one.
    assert read["error"] == pytest.approx((8.0 / 99 / 100) ** 0.5)
    assert (read["each_under"], read["each_over"]) == (True, False)
    assert (read["less"], read["more"]) == (True, False)
    said = capsys.readouterr().out
    assert "their checkpoints after 2, 4, 6 updates and their victim" in said
    assert "of With itself's last checkpoint:" in said and "61.0% ± " in said
    assert "3. Self-play leaves less for an exploiter: yes (With itself's exploiters win 61.0%" in (
        said
    )
    assert "Pilots alone's 71.0%: -10.0 ± 2.8; each of its own under each of the other's" in said
    # The same the other way round is more.
    read = tool.report(exploited(tmp_path / "more", (70, 72), (60, 62)), 6, 2, 6, 2)["readings"]
    assert (read["exploited"]["less"], read["exploited"]["more"]) == (False, True)
    assert "leaves less for an exploiter: no: more" in capsys.readouterr().out


def test_self_play_is_not_shown_to_leave_less_under_any_of_the_marks(tmp_path, capsys):
    def read(name: str, *given) -> dict:
        return tool.report(exploited(tmp_path / name, *given), 6, 2, 6, 2)["readings"]["exploited"]

    # Two points fewer of a thousand deals, by more than two standard errors
    # and each under each: under the three points.
    near = read("near", (680, 690), (700, 710), 1000)
    assert near["gap"] == pytest.approx(-0.02) and near["gap"] < -2 * near["error"]
    assert (near["each_under"], near["less"], near["more"]) == (True, False, False)
    # Four and a half points fewer by more than two standard errors, and one
    # of its exploiters wins more than both of the other's.
    capsys.readouterr()
    mixed = read("mixed", (60, 72), (70, 71))
    assert mixed["gap"] == pytest.approx(-0.045) and mixed["gap"] < -2 * mixed["error"]
    assert (mixed["each_under"], mixed["each_over"], mixed["less"]) == (False, False, False)
    said = capsys.readouterr().out
    assert "3. Self-play leaves less for an exploiter: not shown" in said
    assert "-4.5 ± 1.6; its own not all on one side of the other's" in said
    # Ten points fewer of ten deals, each under each: one standard error.
    few = read("few", (5, 5), (6, 6), 10)
    assert few["gap"] == pytest.approx(-0.1) and few["error"] == pytest.approx(0.1)
    assert (few["each_under"], few["less"]) == (True, False)


def test_the_third_reading_is_taken_only_of_whole_exploiters_of_the_arms_last(tmp_path, capsys):
    # One checkpoint of one exploiter is not measured from either seat.
    partial = exploited(tmp_path / "partial", (60, 62), (70, 72))
    (partial / "head-to-head" / f"exploiter-{tool.ITSELF}-2013.0004.back.json").unlink()
    found = tool.report(partial, 6, 2, 6, 2)
    assert "exploited" not in found["readings"]
    assert list(found["exploited"]["victims"][tool.ITSELF]["exploiters"]) == [
        f"exploiter-{tool.ITSELF}-2012"
    ]
    assert "3. Self-play leaves less for an exploiter: not measured" in capsys.readouterr().out
    # The arms alone: the reading is named and not measured.
    assert "exploited" not in tool.report(arms(tmp_path / "arms"), 6, 2, 6, 2)["readings"]
    assert "3. Self-play leaves less for an exploiter: not measured" in capsys.readouterr().out
    # Read for an arm of other updates than its exploiters' victim has had.
    with pytest.raises(SystemExit, match="is not of With itself's last checkpoint after 8"):
        tool.report(exploited(tmp_path / "other", (60, 62), (70, 72)), 8, 2, 6, 2)
    # A pair that is another checkpoint than the exploiter's with its victim.
    crossed = exploited(tmp_path / "crossed", (60, 62), (70, 72))
    name = f"exploiter-{tool.PILOTS}-2012"
    beats(crossed, f"{name}.0006", after(name, 2286, 4), 2290, of(tool.PILOTS, 6), 70, 100)
    with pytest.raises(SystemExit, match=f"{name}.0006.json is not {name} after 6 updates"):
        tool.report(crossed, 6, 2, 6, 2)
    victim = exploited(tmp_path / "victim", (60, 62), (70, 72))
    beats(victim, f"{name}.0006", after(name, 2286, 6), 2292, of(tool.PILOTS, 4), 70, 100)
    with pytest.raises(SystemExit, match=f"{name}.0006.json is not {name} after 6 updates"):
        tool.report(victim, 6, 2, 6, 2)
    # The two files of a pair are not of the same two checkpoints.
    torn = exploited(tmp_path / "torn", (60, 62), (70, 72))
    path = torn / "head-to-head" / f"{name}.0002.back.json"
    kept = json.loads(path.read_text())
    kept["opponent_checkpoint"]["path"] = after(name, 2286, 4)
    path.write_text(json.dumps(kept))
    with pytest.raises(SystemExit, match="0002.back.json is not .*0002.json with the other"):
        tool.report(torn, 6, 2, 6, 2)


def test_what_the_starts_exploiters_found_is_told_against_each_arm(tmp_path, capsys):
    """A hundred deals from either seat.  The start's exploiters win 70 and
    72 against the start; against with-itself's last checkpoint the start
    wins 40 and they 44 and 46, against pilots-alone's 45 and they 65 and
    67."""
    folder = arms(tmp_path)
    exploiters_of(folder, START, 2280)
    for seed, won in ((2012, 70), (2013, 72)):
        name = f"exploiter-{seed}"
        for point in (2, 4, 6):
            beats(
                folder,
                f"{name}.{point:04d}",
                after(name, 2280, point),
                2280 + point,
                START,
                won,
                100,
            )
    for arm, start, theirs in ((tool.ITSELF, 40, (44, 46)), (tool.PILOTS, 45, (65, 67))):
        beats(folder, f"start.at-{arm}", START, 2280, of(arm, 6), start, 100)
        for seed, won in zip((2012, 2013), theirs, strict=True):
            name = f"exploiter-{seed}"
            beats(folder, f"{name}.at-{arm}", after(name, 2280, 6), 2286, of(arm, 6), won, 100)
    found = tool.report(folder, 6, 2, 6, 2)
    # Against their own victim, every deal from either seat.
    assert found["exploited"]["victims"][tool.START]["rate"] == pytest.approx(0.71)
    assert "exploited" not in found["readings"]
    again = found["found_again"]
    assert again[tool.START]["exploiters"][0] == pytest.approx(0.71)
    assert again[tool.ITSELF]["start"][0] == pytest.approx(0.40)
    assert again[tool.ITSELF]["exploiters"][0] == pytest.approx(0.45)
    assert again[tool.ITSELF]["over"][0] == pytest.approx(0.05)
    # Four deals differ by a whole game and two by half of one.
    assert again[tool.ITSELF]["over"][1] == pytest.approx(
        ((4 * 0.95**2 + 2 * 0.45**2 + 94 * 0.05**2) / 99 / 100) ** 0.5
    )
    assert again[tool.PILOTS]["over"][0] == pytest.approx(0.21)
    said = capsys.readouterr().out
    assert "What the start's exploiters found, against each arm's last checkpoint" in said
    assert "the start itself" in said and "+21.0 ± " in said
    assert "With itself's last checkpoint" in said and "+5.0 ± 2.1" in said
    # The start measured with another checkpoint than its exploiters.
    name = "exploiter-2013"
    beats(
        folder, f"{name}.at-{tool.ITSELF}", after(name, 2280, 6), 2286, of(tool.ITSELF, 4), 46, 100
    )
    with pytest.raises(SystemExit, match=f"{name}.at-{tool.ITSELF}.json is not {name} after 6"):
        tool.report(folder, 6, 2, 6, 2)


def further(folder: Path, won: tuple[int, int, int], of_: int = 100, since: int = 2) -> Path:
    """An arm of 8 updates whose with-itself wins so many of the deals, from
    either seat, at each of its last three checkpoints against its own
    checkpoint after `since` updates."""
    arms(folder)
    for updates, wins in zip((4, 6, 8), won, strict=True):
        beats(
            folder,
            f"{tool.ITSELF}.{updates:04d}.since",
            of(tool.ITSELF, updates),
            2280 + updates,
            of(tool.ITSELF, since),
            wins,
            of_,
        )
        back = folder / "head-to-head" / f"{tool.ITSELF}.{updates:04d}.since.back.json"
        kept = json.loads(back.read_text())
        kept["records"][0]["checkpoint_step"] = 2280 + since
        back.write_text(json.dumps(kept))
    return folder


def test_with_itself_has_gained_on_itself_past_both_marks(tmp_path, capsys):
    """Its last three with its own earlier checkpoint, each in turn held:
    every deal from either seat, gained at 52% and two standard errors."""
    found = tool.report(further(tmp_path / "a", (60, 62, 58)), updates=8, step=2, since=2)
    gained = found["readings"]["gained"]
    assert gained["games"] == 600 and gained["deals"] == 100
    assert gained["share"] == pytest.approx(0.60)
    assert gained["better"] and not gained["worse"]
    assert "Since its checkpoint after 2 updates, With itself has gained on itself: yes" in (
        capsys.readouterr().out
    )
    # Over half, and under the 52% that says so.
    level = tool.report(further(tmp_path / "b", (51, 51, 51)), updates=8, step=2, since=2)
    assert not level["readings"]["gained"]["better"] and not level["readings"]["gained"]["worse"]
    assert "has gained on itself: not shown" in capsys.readouterr().out
    lost = tool.report(further(tmp_path / "c", (40, 42, 38)), updates=8, step=2, since=2)
    assert lost["readings"]["gained"]["worse"]
    # Not asked, not told; asked with a pair missing, not measured.
    assert "gained" not in tool.report(tmp_path / "a", updates=8, step=2)["readings"]
    (tmp_path / "a" / "head-to-head" / f"{tool.ITSELF}.0006.since.back.json").unlink()
    capsys.readouterr()
    assert "gained" not in tool.report(tmp_path / "a", updates=8, step=2, since=2)["readings"]
    assert "has gained on itself: not measured" in capsys.readouterr().out


def test_gaining_on_itself_is_read_only_of_with_itself_and_its_own_earlier_checkpoint(tmp_path):
    folder = further(tmp_path, (60, 62, 58))
    with pytest.raises(SystemExit, match="is not with-itself after 4 updates with its checkpoint after 4"):
        tool.report(folder, updates=8, step=2, since=4)
    beats(
        folder,
        f"{tool.ITSELF}.0008.since",
        of(tool.ITSELF, 8),
        2288,
        of(tool.PILOTS, 2),
        60,
        100,
    )
    with pytest.raises(SystemExit, match="is not with-itself after 8 updates"):
        tool.report(folder, updates=8, step=2, since=2)
