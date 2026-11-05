from ygo_training.metrics import (
    append,
    evaluation_summary_for,
    format_evaluation_suite_status,
    format_status,
    read,
)


def test_status_formats_one_jsonl_record_per_model_version(tmp_path):
    path = tmp_path / "metrics.jsonl"
    append(
        path,
        {
            "version": 3,
            "samples": 42,
            "learner_win_rate": 0.5,
            "evaluation": {
                "win_rate": 0.75,
                "wins": 3,
                "losses": 1,
                "draws": 0,
                "by_opponent": {"Dingy Daredevil": {"win_rate": 0.75}},
            },
        },
    )
    rendered = format_status(read(path))
    assert "version" in rendered
    assert "3" in rendered
    assert "75.0%" in rendered


def test_status_omits_matchup_summary_column():
    record = {
        "version": 1,
        "samples": 1,
        "learner_win_rate": 0.0,
        "evaluation": {
            "win_rate": 0.5,
            "wins": 1,
            "losses": 1,
            "draws": 0,
            "by_opponent": {"Hard": {"win_rate": 0.0}, "Easy": {"win_rate": 1.0}},
        },
    }
    rendered = format_status([record])
    assert "matchup summary" not in rendered
    assert "Hard:0%" not in rendered


def test_status_displays_evaluation_cutoffs_as_a_rate():
    rendered = format_status(
        [
            {
                "version": 4,
                "evaluation": {
                    "games": 1024,
                    "wins": 545,
                    "losses": 421,
                    "draws": 58,
                    "cutoff_draws": 58,
                    "win_rate": 545 / 1024,
                },
            }
        ]
    )
    assert "cut E %" in rendered
    assert "5.7%" in rendered
    assert "     58" not in rendered


def test_status_includes_evaluation_deckout_loss_and_win_counts():
    record = {
        "version": 1,
        "duels": 2,
        "samples": 1,
        "learner_win_rate": 0.5,
        "learner_deckout_losses": 1,
        "learner_deckout_wins": 0,
        "evaluation": {
            "win_rate": 0.5,
            "wins": 1,
            "losses": 1,
            "draws": 0,
            "deckout_losses": 1,
            "deckout_wins": 0,
            "by_opponent": {},
        },
    }
    rendered = format_status([record])
    assert rendered.count("1/0") == 1


def test_status_shows_recorded_runtime_breakdown_and_handles_old_records():
    record = {
        "version": 1,
        "seconds": 63.0,
        "ppo_update_seconds": 12.6,
        "evaluation_seconds": 48.0,
        "wall_seconds": 124.0,
    }
    rendered = format_status([record])
    assert "63s" in rendered
    assert "13s" in rendered
    assert "48s" in rendered
    assert "124s" in rendered
    assert "  - " in format_status([{"version": 2}])


def test_status_includes_total_learner_reward():
    rendered = format_status([{"version": 1, "learner_total_reward": 37.25}])
    assert "reward" in rendered
    assert "37.2" in rendered


def test_status_distinguishes_pending_evaluation_from_training_cutoff_steps():
    rendered = format_status(
        [{"version": 1, "cutoff_loss_trajectories": 1, "cutoff_loss_steps": 146}]
    )
    assert "cut train T/S" in rendered
    assert "1/146" in rendered
    assert "0/0            evaluation pending" not in rendered


def test_read_merges_a_later_posthoc_evaluation_patch(tmp_path):
    path = tmp_path / "metrics.jsonl"
    append(path, {"version": 7, "samples": 42, "learner_win_rate": 0.5})
    append(
        path,
        {
            "version": 7,
            "evaluation_seconds": 3.0,
            "evaluation": {"win_rate": 0.75, "wins": 3, "losses": 1, "draws": 0, "by_opponent": {}},
        },
    )
    records = read(path)
    assert len(records) == 1
    assert records[0]["samples"] == 42
    assert records[0]["evaluation"]["wins"] == 3


def test_evaluation_summary_prefers_pooled_suite_result():
    record = {
        "evaluation": {"wins": 3, "losses": 1, "draws": 0, "win_rate": 0.75},
        "evaluation_suite": {
            "first": {"wins": 3, "losses": 1, "draws": 0},
            "second": {"wins": 1, "losses": 2, "draws": 1},
            "failed": {"error": "not completed"},
        },
    }

    summary = evaluation_summary_for(record)

    assert summary["wins"] == 4
    assert summary["losses"] == 3
    assert summary["draws"] == 1
    assert summary["games"] == 8
    assert summary["win_rate"] == 0.5


def test_evaluation_suite_status_shows_pooled_and_per_matchup_rates():
    rendered = format_evaluation_suite_status(
        [
            {
                "version": 29,
                "evaluation_suite": {
                    "blackwing-vs-amazoness": {"wins": 3, "losses": 1, "draws": 0},
                    "blackwing-vs-draconic": {"wins": 1, "losses": 2, "draws": 1},
                    "failed": {"error": "not completed"},
                },
            }
        ]
    )

    assert "version" in rendered
    assert "aggregate" in rendered
    assert "blackwing-vs-amazoness" in rendered
    assert "blackwing-vs-draconic" in rendered
    assert "4" in rendered
    assert "3" in rendered
    assert "1" in rendered
    assert "50.0%" in rendered
    assert "75.0%" in rendered
    assert "25.0%" in rendered
    assert "failed" not in rendered


def test_optimization_status_shows_imitation_and_ppo_health():
    from ygo_training.metrics import format_optimization_status

    rendered = format_optimization_status(
        [
            {
                "version": 9,
                "warmup": 1.0,
                "samples": 60555,
                "learner_win_rate": 0.708,
                "imitation_loss": 0.25,
                "imitation_model_teacher_agreement": 0.958,
                "imitation_value_explained_variance": 0.076,
                "imitation_lead_loss": 0.0312,
                "imitation_lead_explained_variance": 0.041,
            },
            {
                "version": 10,
                "samples": 58130,
                "learner_win_rate": 0.615,
                "duel_explained_variance": 0.555,
                "duel_approximate_kl": 0.066,
                "duel_clip_fraction": 0.125,
                "duel_entropy": 0.418,
                "duel_value": 0.17,
            },
        ]
    )
    header, imitation, ppo = rendered.splitlines()
    assert "value EV" in header and "approx KL" in header
    assert imitation.split()[:2] == ["9", "imitation"]
    assert "0.076" in imitation and "95.8%" in imitation and "0.250" in imitation
    assert header.split("  ")[-2:] == ["lead loss", "lead EV"]
    assert imitation.split()[-2:] == ["0.0312", "0.041"] and ppo.split()[-2:] == ["-", "-"]
    assert ppo.split()[:2] == ["10", "ppo"]
    assert "0.555" in ppo and "0.0660" in ppo and "0.418" in ppo and "0.170" in ppo


def test_optimization_status_tolerates_runs_without_value_warmup():
    from ygo_training.metrics import format_optimization_status

    row = format_optimization_status([{"version": 0, "warmup": 1.0, "samples": 1}]).splitlines()[1]
    assert row.split()[:2] == ["0", "imitation"]
    assert row.split()[4] == "-"


def test_optimization_status_accumulates_end_of_update_kl():
    from ygo_training.metrics import OPTIMIZATION_COLUMNS, format_optimization_status

    rows = [
        line.split()
        for line in format_optimization_status(
            [
                {"version": 9, "warmup": 1.0, "samples": 1},
                {"version": 10, "samples": 1, "duel_final_approximate_kl": 0.02},
                {"version": 11, "samples": 1, "duel_final_approximate_kl": 0.03},
            ]
        ).splitlines()[1:]
    ]
    # "value EV" and similar headings contain spaces; data cells do not.
    cumulative = OPTIMIZATION_COLUMNS.index("cum KL")
    assert [row[cumulative] for row in rows] == ["-", "0.020", "0.050"]


def test_optimization_status_omits_cumulative_kl_for_older_runs():
    from ygo_training.metrics import OPTIMIZATION_COLUMNS, format_optimization_status

    row = (
        format_optimization_status([{"version": 10, "samples": 1, "duel_approximate_kl": 0.03}])
        .splitlines()[1]
        .split()
    )
    assert row[OPTIMIZATION_COLUMNS.index("approx KL")] == "0.0300"
    assert row[OPTIMIZATION_COLUMNS.index("final KL")] == "-"
    assert row[OPTIMIZATION_COLUMNS.index("cum KL")] == "-"
