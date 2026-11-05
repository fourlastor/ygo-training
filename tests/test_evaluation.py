from ygo_training.evaluation import summarize_results


def game(**fields):
    return {
        "opponent": "monarch",
        "learner_player": 0,
        "outcome": "win",
        "terminal": True,
        "win_reason": 1,
        "turns": 10,
        "decisions": 40,
        **fields,
    }


def test_a_case_is_summarized_overall_by_opponent_and_by_seat():
    summary = summarize_results(
        [
            game(),
            game(learner_player=1, outcome="loss", win_reason=2, decisions=60, turns=20),
            game(opponent="heroes", outcome="draw"),
            game(opponent="heroes", learner_player=1, outcome="draw", terminal=False),
        ]
    )

    assert (summary["games"], summary["wins"], summary["losses"], summary["draws"]) == (4, 1, 1, 2)
    assert summary["win_rate"] == 0.25
    assert summary["cutoff_draws"] == 1 and summary["terminal_draws"] == 1
    assert summary["deckout_losses"] == 1 and summary["deckout_wins"] == 0
    assert summary["mean_decisions"] == 45.0
    assert summary["by_opponent"]["monarch"]["win_rate"] == 0.5
    assert summary["by_opponent"]["heroes"]["draws"] == 2
    assert summary["by_player"]["0"]["wins"] == 1
    assert summary["by_player"]["1"]["mean_turns"] == 15.0


def test_the_learners_answers_and_its_agreement_with_the_pilot_are_pooled():
    summary = summarize_results(
        [
            game(
                learner_answers={"Chain:Pass": 3, "Idle:EndTurn": 1},
                pilot_labels=4,
                pilot_agreements=3,
            ),
            game(
                learner_answers={"Chain:Pass": 2, "Chain:Activate": 2},
                pilot_labels=4,
                pilot_agreements=1,
            ),
        ]
    )

    assert list(summary["learner_answers"].items()) == [
        ("Chain:Pass", 5),
        ("Chain:Activate", 2),
        ("Idle:EndTurn", 1),
    ]
    assert summary["pilot_agreement"] == 0.5
    # A pilot's own games carry no labels to agree with.
    assert "pilot_agreement" not in summarize_results([game()])


def test_only_the_seats_that_played_are_listed():
    summary = summarize_results([game(learner_player=1)])
    assert list(summary["by_player"]) == ["1"]
