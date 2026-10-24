"""The search-leads report: the weight, the choice and the readings."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "search_leads_report", ROOT / "tools/search_leads_report.py"
)
assert SPEC is not None and SPEC.loader is not None
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)

SEED = 89


def reading(margin: float, gain: float, error: float = 0.01, changed: int = 10) -> dict[str, Any]:
    return {
        "margin": margin,
        "gain_per_game": gain,
        "gain_per_game_error": error,
        "changed": changed,
    }


def entry(
    update: int,
    won: int,
    *,
    games: int = 100,
    led: list[dict[str, Any]] | None = None,
    ruled: tuple[float, int] | None = None,
) -> dict[str, Any]:
    """A checkpoint as `evaluate-search` reports it: it wins the first `won`
    deals, and by its rule, at the margin of `ruled`, the first so many."""
    tried: dict[str, Any] = {
        "decisions": 50 * games,
        "pilot": 48 * games,
        "games": games,
        "lead_per_game": -0.02,
        "lead_per_game_error": 0.004,
        "lead_by_game": [-0.02] * games,
    }
    if led is not None:
        tried["led"] = led
        tried["head"] = {
            "alternatives": 120 * games,
            "correlation": 0.25,
            "highest": [{"share": 0.01, "lead": 0.04, "lead_error": 0.01}],
        }
    found: dict[str, Any] = {
        "checkpoint": f"/runs/arm/checkpoints/model-{SEED + update:05d}.pt",
        "checkpoint_label": f"arm/model-{SEED + update:05d}",
        "checkpoint_step": SEED + update,
        "version": SEED + update,
        "deals": {
            "win_rate": won / games,
            "outcomes": ["win"] * won + ["loss"] * (games - won),
            "references": {"pilot": {"win_rate": 0.60}, "search": {"win_rate": 0.72}},
        },
        "answer_key": {"tried": tried},
    }
    if ruled is not None:
        margin, wins = ruled
        found["led_deals"] = {
            "margin": margin,
            "win_rate": wins / games,
            "outcomes": ["win"] * wins + ["loss"] * (games - wins),
        }
    return found


def suite(wins: dict[str, int]) -> dict[str, Any]:
    return {"evaluation_suite": {name: {"games": 128, "wins": won} for name, won in wins.items()}}


def test_the_seed_is_held_against_its_pilot_on_the_whole_suite() -> None:
    pilot = suite({"blackwing-vs-burn": 110, "blackwing-vs-monarch": 40})
    model = suite({"blackwing-vs-burn": 100, "blackwing-vs-monarch": 36})
    reference, rate, games = report.seed_gap(pilot, model)
    assert games == 256
    assert reference - rate == pytest.approx(14 / 256)
    with pytest.raises(ValueError, match="same suite"):
        report.seed_gap(pilot, suite({"blackwing-vs-burn": 100}))
    model["evaluation_suite"]["blackwing-vs-burn"] = {"error": "no such deck"}
    with pytest.raises(ValueError, match="gave no result: blackwing-vs-burn"):
        report.seed_gap(pilot, model)


def test_the_weight_makes_the_two_losses_pull_alike_after_the_first_update() -> None:
    records = [
        {
            "version": 90,
            "imitation_shared_gradient_norm": 1.0,
            "imitation_lead_shared_gradient_norm": 0.5,
        },
        {
            "version": 91,
            "imitation_shared_gradient_norm": 0.6,
            "imitation_lead_shared_gradient_norm": 0.04,
        },
        {
            "version": 92,
            "imitation_shared_gradient_norm": 0.5,
            "imitation_lead_shared_gradient_norm": 0.05,
        },
        {
            "version": 93,
            "imitation_shared_gradient_norm": 0.4,
            "imitation_lead_shared_gradient_norm": 0.06,
        },
        {"version": 94, "samples": 1.0},
    ]
    weight, measured = report.lead_weight(records)
    assert weight == pytest.approx(1.5 / 0.15)
    assert [version for version, _, _ in measured] == [90, 91, 92, 93]


def test_too_few_updates_set_no_weight() -> None:
    records = [
        {
            "version": 90,
            "imitation_shared_gradient_norm": 1.0,
            "imitation_lead_shared_gradient_norm": 0.5,
        },
        {
            "version": 91,
            "imitation_shared_gradient_norm": 0.6,
            "imitation_lead_shared_gradient_norm": 0.04,
        },
    ]
    with pytest.raises(ValueError, match="too few"):
        report.lead_weight(records)


def test_the_choice_is_where_the_rule_gains_most() -> None:
    entries = [
        entry(16, 60, led=[reading(0.05, 0.010), reading(0.10, 0.020)]),
        entry(32, 60, led=[reading(0.05, 0.035), reading(0.10, 0.030)]),
        entry(64, 60, led=[reading(0.05, -0.010), reading(0.10, 0.015)]),
    ]
    choice = report.choose(entries, SEED)
    assert (choice["update"], choice["margin"], choice["version"]) == (32, 0.05, SEED + 32)
    assert choice["gain_per_game"] == 0.035
    assert choice["checkpoint"].endswith(f"model-{SEED + 32:05d}.pt")


def test_of_equal_gains_the_earlier_update_then_the_larger_margin() -> None:
    entries = [
        entry(32, 60, led=[reading(0.05, 0.02), reading(0.10, 0.02)]),
        entry(16, 60, led=[reading(0.05, 0.02), reading(0.10, 0.02)]),
    ]
    choice = report.choose(entries, SEED)
    assert (choice["update"], choice["margin"]) == (16, 0.10)


def test_a_rule_that_gains_nothing_has_no_choice() -> None:
    entries = [entry(16, 60, led=[reading(0.05, 0.0), reading(0.10, -0.01)])]
    assert report.choose(entries, SEED) is None


def test_the_paired_difference_counts_the_deals_only_one_won() -> None:
    first = [True] * 68 + [False] * 32
    second = [True] * 60 + [False] * 40
    difference, error = report.paired(first, second)
    assert difference == pytest.approx(0.08)
    assert error == pytest.approx((8 - 0.64) ** 0.5 / 100)
    with pytest.raises(ValueError, match="not the same run"):
        report.paired(first, second[:-1])


def test_the_validation_run_gives_every_arm_and_the_choices(capsys) -> None:
    arms = {
        "seed": [entry(0, 55)],
        "pilot": [entry(16, 58), entry(32, 59)],
        "search": [entry(16, 57), entry(32, 60)],
        "leads": [
            entry(16, 58, led=[reading(0.05, 0.01), reading(0.10, 0.02)]),
            entry(32, 59, led=[reading(0.05, 0.03), reading(0.10, 0.01)]),
        ],
        "leads-read": [
            entry(16, 58, led=[reading(0.05, -0.01), reading(0.10, 0.0)]),
            entry(32, 59, led=[reading(0.05, -0.02), reading(0.10, -0.01)]),
        ],
    }
    table, choices = report.validation(arms, SEED)
    assert (choices["leads"]["update"], choices["leads"]["margin"]) == (32, 0.05)
    assert choices["leads-read"] is None
    assert table["pilot"]["32"]["win_rate"] == 0.59
    assert table["leads"]["16"]["led"]["0.1"]["gain_per_game"] == 0.02
    assert table["leads"]["32"]["head"]["correlation"] == 0.25
    printed = capsys.readouterr().out
    assert "leads: chosen update 32, margin 5 points" in printed
    assert "leads-read: its rule gains nothing above zero" in printed


def held_out(gain: float, error: float, ruled: int) -> dict[str, Any]:
    """The readings of a leads arm chosen at update 32 and a margin of 10
    points, beside a pilot arm that wins 60 of 100 deals."""
    choices = {
        "leads": {"update": 32, "version": SEED + 32, "margin": 0.10},
        "leads-read": None,
    }
    arms = {
        "seed": [entry(0, 55)],
        "pilot": [entry(32, 60)],
        "search": [entry(32, 62)],
        "leads": [entry(32, 61, led=[reading(0.10, gain, error)], ruled=(0.10, ruled))],
    }
    return report.heldout(arms, choices, SEED)


def test_the_readings_ask_for_the_mark_and_for_the_standard_errors() -> None:
    # The search gains 12 points on these deals: the mark is 3.
    found = held_out(0.04, 0.01, 68)
    assert found["mark"] == pytest.approx(0.03)
    reading_ = found["readings"]["leads"]
    assert reading_["learnable"] and reading_["plays_better"]
    assert reading_["against_pilot"][0] == pytest.approx(0.08)
    assert reading_["against_own_first_choices"][0] == pytest.approx(0.07)
    assert reading_["search_against_pilot"][0] == pytest.approx(0.02)
    assert found["arms"]["leads"]["32"]["led_win_rate"] == 0.68

    # Over the mark, and within three standard errors of zero.
    assert not held_out(0.04, 0.02, 68)["readings"]["leads"]["learnable"]
    # Well above zero, and under the mark.
    assert not held_out(0.02, 0.002, 68)["readings"]["leads"]["learnable"]
    # Two deals more than the pilot arm: under the mark.
    assert not held_out(0.04, 0.01, 62)["readings"]["leads"]["plays_better"]
    # Three deals more: at the mark, and within two standard errors.
    assert not held_out(0.04, 0.01, 63)["readings"]["leads"]["plays_better"]
    # Four deals more: over the mark and over two standard errors.
    assert held_out(0.04, 0.01, 64)["readings"]["leads"]["plays_better"]


def test_a_search_that_gains_nothing_sets_no_mark(capsys) -> None:
    choices = {"leads": {"update": 32, "version": SEED + 32, "margin": 0.10}}
    arms = {
        "seed": [entry(0, 55)],
        "pilot": [entry(32, 60)],
        "leads": [entry(32, 61, led=[reading(0.10, 0.04)], ruled=(0.10, 68))],
    }
    for name in arms:
        arms[name][0]["deals"]["references"] = {
            "pilot": {"win_rate": 0.70},
            "search": {"win_rate": 0.66},
        }
    found = report.heldout(arms, choices, SEED)
    assert found["mark"] == pytest.approx(-0.01)
    assert not found["readings"]["leads"]["learnable"]
    assert not found["readings"]["leads"]["plays_better"]
    assert "sets no mark" in capsys.readouterr().out


def test_a_held_out_result_at_another_choice_is_refused() -> None:
    choices = {"leads": {"update": 32, "version": SEED + 32, "margin": 0.10}}
    arms = {
        "seed": [entry(0, 55)],
        "pilot": [entry(32, 60)],
        "leads": [entry(32, 61, led=[reading(0.05, 0.04)], ruled=(0.05, 68))],
    }
    with pytest.raises(ValueError, match="the choice is version"):
        report.heldout(arms, choices, SEED)
    arms["leads"] = [entry(16, 61, led=[reading(0.10, 0.04)], ruled=(0.10, 68))]
    with pytest.raises(ValueError, match="the choice is version"):
        report.heldout(arms, choices, SEED)


def test_two_arms_are_set_against_each_other_deal_by_deal() -> None:
    mine = {16: [1.0, 1.0, 0.0, 0.0], 32: [1.0, 0.0, 1.0, 0.0]}
    theirs = {16: [1.0, 0.0, 0.0, 0.0], 32: [0.0, 0.0, 1.0, 0.0], 64: [1.0, 1.0, 1.0, 1.0]}
    found = report.against(mine, theirs)
    assert sorted(found) == ["16", "32", "all"]
    assert found["16"][0] == found["32"][0] == found["all"][0] == 0.25
    # Over the updates a deal counts once: half, half, nothing and nothing.
    assert found["all"][1] == pytest.approx((1 / 12) ** 0.5 / 2)
    with pytest.raises(ValueError, match="same deals"):
        report.against({16: [1.0]}, {16: [1.0, 0.0]})


def apart(left: int, back: int) -> dict[str, Any]:
    def part(changed: int, gain: float) -> dict[str, Any]:
        return {
            "changed": changed,
            "unmeasured": 0,
            "ahead": changed // 4,
            "level": changed // 2,
            "behind": changed - changed // 4 - changed // 2,
            "gain_per_game": gain,
            "gain_per_game_error": 0.002,
        }

    return {
        "from_pilot": part(left, -0.008),
        "to_pilot": part(back, 0.02),
        "between_others": part(0, 0.0),
    }


def more(tmp_path: Path, files: dict[str, list[dict[str, Any]]]) -> Path:
    """The folder of the arms' script: what was read first, the choice, and
    `files` under `more`."""
    for folder in ("heldout", "more"):
        (tmp_path / folder).mkdir(exist_ok=True)
    first = {"seed": [entry(0, 55)], "pilot": [entry(32, 60)]}
    for name, entries in first.items():
        (tmp_path / "heldout" / f"{name}.json").write_text(json.dumps({"checkpoints": entries}))
    choice = {"leads": {"update": 32, "version": SEED + 32, "margin": 0.05}, "leads-read": None}
    (tmp_path / "choice.json").write_text(json.dumps(choice))
    for name, entries in files.items():
        (tmp_path / "more" / f"{name}.json").write_text(json.dumps({"checkpoints": entries}))
    return tmp_path


def test_the_held_out_run_at_every_update_sets_the_arms_against_the_pilot_arm(
    tmp_path, capsys
) -> None:
    rule = [{**reading(0.05, 0.012, 0.004, 150), "split": apart(120, 30)}]
    folder = more(
        tmp_path,
        {
            "pilot": [entry(16, 58), entry(32, 60)],
            "search": [entry(16, 61), entry(32, 64)],
            "leads": [entry(16, 59, led=rule), entry(32, 60, led=rule)],
        },
    )
    output = tmp_path / "report.json"

    report.more(argparse.Namespace(directory=folder, output=output))

    printed = capsys.readouterr().out
    assert "100 held-out deals, after the readings: pilot 60.0%, search 72.0%." in printed
    won = next(line for line in printed.splitlines() if line.split()[:1] == ["32"])
    assert won.split()[1:5] == ["60.0%", "64.0%", "(+4.0", "±"]
    assert "leads at update 32, its changes apart by where they start" in printed
    assert "120 (30/60/30) -0.8 ± 0.2" in printed and "30 (7/15/8) +2.0 ± 0.2" in printed
    found = json.loads(output.read_text())
    assert found["deals_won"]["search"]["all"]["against_pilot"][0] == pytest.approx(0.035)
    assert found["deals_won"]["pilot"]["16"] == {"value": 0.58}
    assert found["leads"]["search"]["32"]["against_pilot"][0] == pytest.approx(0.0)
    assert found["rule"]["leads"]["32"]["0.05"]["split"]["to_pilot"]["changed"] == 30


def test_the_report_written_beside_the_results_is_not_read_as_one(tmp_path, capsys) -> None:
    rule = [{**reading(0.05, 0.012, 0.004, 150), "split": apart(120, 30)}]
    folder = more(tmp_path, {"pilot": [entry(32, 60)], "leads": [entry(32, 60, led=rule)]})
    # Where the script writes it: the folder the command reads.
    output = folder / "more" / "report.json"

    report.more(argparse.Namespace(directory=folder, output=output))
    first = capsys.readouterr().out
    report.more(argparse.Namespace(directory=folder, output=output))

    assert capsys.readouterr().out == first


def test_what_was_read_first_must_come_out_the_same(tmp_path) -> None:
    folder = more(tmp_path, {"pilot": [entry(16, 58), entry(32, 59)]})
    with pytest.raises(SystemExit, match="does not play the games of heldout/pilot.json"):
        report.more(argparse.Namespace(directory=folder, output=None))


def test_the_changes_apart_need_a_measurement_that_holds_them(tmp_path) -> None:
    rule = [reading(0.05, 0.012)]
    folder = more(tmp_path, {"pilot": [entry(32, 60)], "leads": [entry(32, 60, led=rule)]})
    with pytest.raises(SystemExit, match="without the changes apart"):
        report.more(argparse.Namespace(directory=folder, output=None))
