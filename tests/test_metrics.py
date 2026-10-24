from ygo_training.evaluation import summarize_answer_key, summarize_deals
from ygo_training.metrics import (
    _value_lines,
    append,
    evaluation_summary_for,
    format_evaluation_suite_status,
    format_search_evaluation,
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


def test_a_search_evaluation_shows_the_models_answers_by_the_searchs_worlds():
    def decision(model, leads, weights, **fields):
        return {
            "question": "Idle",
            "outright": False,
            "taken": 0,
            "leads": leads,
            "model": model,
            "weights": weights,
            **fields,
        }

    watched = {
        "decisions": 3,
        "recorded_deviations": 0,
        "examined_known": True,
        "questions": {"Idle": {"kept_pilot": 1, "kept_other": 2, "examined_other": 2}},
        "deviations": [],
        "tried": [
            decision(0, [0.0, 0.25], [0.9, 0.1]),
            decision(1, [0.0, -0.125], [0.2, 0.8]),
            decision(1, [0.0, None], [0.4, 0.6], outright=True),
        ],
    }
    result = {"opponent": "monarch", "learner_player": 0, "outcome": "win", "terminal": True}
    record = {
        "checkpoint_label": "run/model-00001",
        "deals": summarize_deals(
            [{**result, "win_reason": 1, "decisions": 3, "turns": 2}], {"pilot": ["win"]}
        ),
        "answer_key": summarize_answer_key([watched]),
    }

    lines = format_search_evaluation(record).splitlines()

    header = next(line for line in lines if "3 examined" in line)
    assert header.split()[2:] == [
        *("others", "a", "game", "outright", "untried", "tried", "ahead", "level"),
        *("behind", "lead", "sum", "a", "game"),
    ]
    first = next(line for line in lines if "first choice" in line)
    assert first.split()[2:8] == ["2", "2.00", "1", "0", "1", "0"]
    assert "-12.5" in first
    assert any("at 0.1 of the pilot's" in line and "+6.2" in line for line in lines)
    assert "any other answer the search tried (2): +6.2" in lines[-1]

    # Without the search's log the table ends with the answer key.
    del watched["tried"]
    record["answer_key"] = summarize_answer_key([watched])
    assert "examined" not in format_search_evaluation(record).splitlines()[-1]


def test_a_search_evaluation_shows_what_following_the_lead_head_is_worth():
    def decision(model, leads, predicted, **fields):
        return {
            "question": "Idle",
            "outright": False,
            "taken": 0,
            "leads": leads,
            "model": model,
            "weights": [0.9, 0.1],
            "predicted": predicted,
            **fields,
        }

    watched = {
        "decisions": 3,
        "recorded_deviations": 0,
        "examined_known": True,
        "questions": {"Idle": {"kept_pilot": 2, "kept_other": 1, "examined_other": 1}},
        "deviations": [],
        "tried": [
            # A lead of 12 points predicted for an answer that is 25 ahead.
            decision(0, [0.0, 0.25], [0.0, 0.12]),
            # Its first choice is another answer where the pilot's wins outright.
            decision(1, [0.0, -0.5], [0.0, 0.02], outright=True),
            decision(0, [0.0, 0.0], [0.0, -0.3]),
        ],
    }
    result = {"opponent": "monarch", "learner_player": 0, "outcome": "win", "terminal": True}
    deals = summarize_deals(
        [{**result, "win_reason": 1, "decisions": 3, "turns": 2}], {"pilot": ["win"]}
    )
    record = {
        "checkpoint_label": "run/model-00001",
        "deals": deals,
        "led_deals": {"margin": 0.1, "changed": 4, **deals},
        "answer_key": summarize_answer_key([watched]),
    }

    lines = format_search_evaluation(record).splitlines()

    assert "  following its lead head by 10 points: 1/1 (100.0%), 4 first choices left" in lines
    assert any(
        "of the first choices tried, 1 where the pilot's answer wins outright: -50.0 a game"
        in line
        for line in lines
    )
    header = next(line for line in lines if "following the lead head" in line)
    assert header.split()[4:] == ["changed", "a", "game", "others", "sum", "a", "game", "gain", "a", "game"]
    by_ten = next(line for line in lines if line.strip().startswith("by 10 points"))
    # One first choice left, for an answer 25 points ahead; the other answer
    # it gives is its own first choice, 50 behind.
    assert by_ten.split()[3:6] == ["1", "1.00", "2.00"]
    assert by_ten.split()[6:] == ["-25.0", "+25.0"]
    # The same change by where it starts: it leaves the pilot's answer.
    apart = next(index for index, line in enumerate(lines) if "the changes apart" in line)
    titles = ["from", "the", "pilot's", "answer", "back", "to", "it", "between", "others"]
    assert lines[apart].split()[3:] == titles
    assert lines[apart + 2].split()[3:] == ["1:", "+25.0", "0:", "+0.0", "0:", "+0.0"]
    assert "the head over 3 alternatives" in lines[-1] and "correlate at" in lines[-1]



def test_a_search_evaluation_shows_what_the_value_head_tells():
    value = {
        "decisions": 1200,
        "elsewhere": 30,
        "correlation": 0.512,
        "told": 0.5,
        "explained": 0.2,
        "result": 0.4,
        "by_turn": [
            {"turns": [1, 2], "told": 0.1, "result": 0.26},
            {"turns": [11, None], "told": 0.5, "result": 0.6},
        ],
    }

    lines = _value_lines(value)

    assert lines[0] == (
        "  the value head at 1200 examined decisions that open a transaction (30 others left out):"
    )
    assert "correlate at +0.512" in lines[1]
    assert lines[1].endswith("it tells 50.0% (20.0% as its values stand)")
    assert lines[2].endswith("a perfect predictor explains 40.0% there, this head 20.0%")
    assert lines[3].endswith("1-2 10.0% of 26.0%, 11+ 50.0% of 60.0%")
    # A head that says the same everywhere tells nothing that can be put in a share.
    assert "it tells - (" in _value_lines({**value, "told": None, "by_turn": []})[1]
