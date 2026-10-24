import random

import pytest

from ygo_training.evaluation import (
    _told,
    summarize_answer_key,
    summarize_deals,
    summarize_results,
    summarize_tried,
)


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


def deviation(reading, *, question="Idle", pilot="EndTurn", search="Activate", **fields):
    return {
        "question": question,
        "pilot": pilot,
        "search": search,
        "reading": reading,
        "search_probability": 0.5 if reading == "search" else 0.1,
        "pilot_probability": 0.3,
        **fields,
    }


def watched(questions, deviations=(), *, examined_known=False, recorded=None):
    return {
        "decisions": sum(sum(counts.values()) for counts in questions.values()),
        "recorded_deviations": len(deviations) if recorded is None else recorded,
        "examined_known": examined_known,
        "questions": questions,
        "deviations": list(deviations),
    }


def test_the_answer_key_reads_deviations_and_kept_decisions_apart():
    key = summarize_answer_key(
        [
            watched(
                {
                    "Idle": {"left_search": 1, "left_pilot": 1, "kept_pilot": 6, "kept_other": 2},
                    "Chain": {"left_other": 1, "kept_pilot": 10, "unread": 2},
                },
                [
                    deviation("search"),
                    deviation("pilot"),
                    deviation("other", question="Chain", pilot="Pass"),
                ],
            ),
            watched({"Idle": {"left_search": 1, "kept_pilot": 2}}, [deviation("search")]),
        ]
    )

    assert (key["games"], key["decisions"], key["unread"]) == (2, 26, 2)
    assert key["recorded_deviations"] == key["deviations"] == 4
    assert (key["search"], key["pilot"], key["other"]) == (2, 1, 1)
    assert (key["search_rate"], key["pilot_rate"]) == (0.5, 0.25)
    assert (key["kept"], key["kept_left"], key["kept_left_rate"]) == (20, 2, 0.1)
    assert key["mean_search_probability"] == pytest.approx(0.3)
    assert key["mean_pilot_probability"] == pytest.approx(0.3)
    assert key["by_question"]["Idle"]["search_rate"] == pytest.approx(2 / 3)
    assert key["by_question"]["Chain"]["search_rate"] == 0.0
    assert key["by_question"]["Chain"]["kept_left_rate"] == 0.0
    assert list(key["by_change"]) == ["Idle: EndTurn -> Activate", "Chain: Pass -> Activate"]
    assert key["by_change"]["Idle: EndTurn -> Activate"] == {
        "deviations": 3,
        "search": 2,
        "pilot": 1,
        "other": 0,
    }
    assert len(key["deviation_list"]) == 4
    # Without the search's log nothing is said of what it examined.
    assert "examined_kept" not in key and "examined_kept" not in key["by_question"]["Idle"]


def test_the_answer_key_splits_what_the_search_examined_when_it_is_known():
    counts = {"kept_pilot": 8, "kept_other": 2, "examined_pilot": 3, "examined_other": 1}
    key = summarize_answer_key([watched({"Idle": counts}, examined_known=True)])

    assert (key["examined_kept"], key["examined_kept_left"]) == (4, 1)
    assert key["examined_kept_left_rate"] == 0.25
    # No deviation: nothing to give a rate of.
    assert key["search_rate"] is None and key["mean_search_probability"] is None
    mixed = summarize_answer_key(
        [watched({"Idle": counts}, examined_known=True), watched({"Idle": {"kept_pilot": 1}})]
    )
    assert "examined_kept" not in mixed


def decision(model, leads, weights, *, taken=0, outright=False, question="Idle", predicted=None):
    """One decision a search examined, with a model's answer among what it
    tried and, when its lead head was asked, what that predicts for each."""
    return {
        "question": question,
        "outright": outright,
        "taken": taken,
        "leads": leads,
        "model": model,
        "weights": weights,
        "predicted": predicted,
    }


def worlds(generator, worth, count=8):
    """The mean result of an answer over worlds it wins with ``worth``'s odds."""
    return (
        sum(1.0 if generator.random() < (1.0 + worth) / 2.0 else -1.0 for _ in range(count)) / count
    )


def test_a_value_head_is_read_against_what_is_real_in_the_worlds_rates():
    generator = random.Random(7)
    worths = [generator.uniform(-0.9, 0.9) for _ in range(40_000)]
    rates = [worlds(generator, worth) for worth in worths]

    # A head that knows every decision's worth: the worlds' own noise, which
    # it cannot know, is not held against it.
    knowing = _told([(worth, rate, 8) for worth, rate in zip(worths, rates, strict=True)])
    assert knowing["told"] == pytest.approx(1.0, abs=0.03)
    assert knowing["explained"] == pytest.approx(1.0, abs=0.03)
    assert knowing["correlation"] < 0.95
    # Worths spread evenly from -0.9 to 0.9 have a variance of 0.27, and one
    # result around a worth r a variance of 1 - r^2.
    assert knowing["real"] == pytest.approx(0.27, abs=0.01)
    assert knowing["result"] == pytest.approx(0.27, abs=0.01)
    # A head twice as sure as it should be ranks as well and stands worse.
    sure = _told([(2.0 * worth, rate, 8) for worth, rate in zip(worths, rates, strict=True)])
    assert sure["told"] == pytest.approx(knowing["told"])
    assert sure["explained"] == pytest.approx(0.0, abs=0.05)
    # A head that says the same of every decision, or anything at all.
    assert _told([(0.3, rate, 8) for rate in rates])["told"] is None
    blind = _told([(generator.uniform(-1.0, 1.0), rate, 8) for rate in rates])
    assert blind["told"] == pytest.approx(0.0, abs=0.01)
    assert _told([(0.0, 1.0, 1), (0.0, -1.0, 1)]) is None


def test_the_value_head_is_read_where_it_is_trained_and_by_turn():
    generator = random.Random(11)

    def examined(turn, *, start=True):
        worth = generator.uniform(-0.9, 0.9)
        return {
            **decision(0, [0.0, 0.0], [0.5, 0.5]),
            "turn": turn,
            "value": worth,
            "start": start,
            "rate": worlds(generator, worth),
            "worlds": 8,
        }

    watched = [
        {
            "tried": [
                *(examined(1 + place % 12) for place in range(6_000)),
                # A decision inside a transaction, where the head has no target.
                *({**examined(3, start=False), "value": -1.0} for _ in range(500)),
                # And one whose worlds the log does not hold.
                {**examined(3), "rate": None},
            ]
        }
    ]

    value = summarize_tried(watched)["value"]

    assert (value["decisions"], value["elsewhere"]) == (6_000, 500)
    assert value["told"] == pytest.approx(1.0, abs=0.08)
    assert [part["turns"] for part in value["by_turn"]] == [
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 10],
        [11, None],
    ]
    assert [part["decisions"] for part in value["by_turn"]] == [1000, 1000, 1000, 2000, 1000]
    # Without a value there is nothing to read.
    assert "value" not in summarize_tried([{"tried": [decision(0, [0.0, 0.0], [0.5, 0.5])]}])


def test_a_models_answers_are_worth_what_the_search_measured_for_them():
    first = {
        "tried": [
            # The pilot's answer.
            decision(0, [0.0, 0.25], [0.9, 0.1]),
            # An alternative that did better, which the search took too.
            decision(1, [0.0, 0.25], [0.3, 0.7], taken=1),
            # One that did worse.
            decision(2, [0.0, 0.0, -0.5], [0.2, 0.1, 0.7], question="Chain"),
            # Another answer where the pilot's won every world: nothing was tried.
            decision(1, [0.0, None], [0.4, 0.6], outright=True),
            # An answer past the twelfth, which the search did not try.
            decision(None, [0.0, 0.125], [0.5, 0.1]),
        ],
        "tried_unread": 1,
    }
    # The first choice is a like card of neither heavier alternative: among
    # what the search tried, the pilot's answer weighs more.
    second = {"tried": [decision(1, [0.0, 0.0, None], [0.6, 0.35, 0.05])]}

    tried = summarize_tried([first, second])

    assert (tried["games"], tried["decisions"], tried["unread"]) == (2, 6, 1)
    assert (tried["pilot"], tried["outright"], tried["untried"], tried["tried"]) == (1, 1, 1, 3)
    assert (tried["ahead"], tried["level"], tried["behind"]) == (1, 1, 1)
    assert tried["lead"] == pytest.approx(-0.25 / 3)
    assert tried["lead_by_game"] == [-0.25, 0.0]
    assert tried["lead_per_game"] == pytest.approx(-0.125)
    assert tried["lead_per_game_error"] == pytest.approx(0.125)
    # The search's own choices, and any other answer it tried.
    assert (tried["search"]["tried"], tried["search"]["lead"]) == (1, 0.25)
    assert tried["search"]["pilot"] == 5 and "lead_by_game" not in tried["search"]
    assert tried["any_other"]["tried"] == 6
    assert tried["any_other"]["lead"] == pytest.approx((0.25 + 0.25 - 0.5 + 0.125) / 6)
    assert tried["by_question"]["Chain"]["behind"] == 1
    assert tried["by_question"]["Idle"]["decisions"] == 5
    # Taking the heaviest alternative at lower and lower bars.
    bars = {bar["ratio"]: bar for bar in tried["bars"]}
    assert (bars[1.0]["tried"], bars[1.0]["outright"], bars[1.0]["pilot"]) == (2, 1, 3)
    assert bars[1.0]["lead"] == pytest.approx(-0.125)
    assert (bars[0.5]["tried"], bars[0.5]["level"]) == (3, 1)
    assert (bars[0.2]["tried"], bars[0.2]["ahead"]) == (4, 2)
    assert (bars[0.1]["tried"], bars[0.1]["pilot"]) == (5, 0)
    assert bars[0.1]["lead"] == pytest.approx((0.25 - 0.5 + 0.0 + 0.125 + 0.25) / 5)

    # A run that logged nothing has nothing to read.
    assert summarize_tried([{"tried": []}, {}]) is None
    key = summarize_answer_key([watched({"Idle": {"kept_pilot": 1}})])
    assert key["tried"] is None


def test_a_full_log_also_values_the_answers_given_where_the_pilots_wins_outright():
    game = {
        "tried": [
            # Another answer where the pilot's won every world: with a full
            # log it was tried there, and lost half of them.
            decision(1, [0.0, -0.5], [0.4, 0.6], outright=True),
            decision(1, [0.0, 0.25], [0.3, 0.7]),
            decision(0, [0.0, 0.0], [0.9, 0.1], outright=True),
        ]
    }

    tried = summarize_tried([game])

    assert (tried["pilot"], tried["outright"], tried["tried"]) == (1, 0, 2)
    assert (tried["tried_outright"], tried["lead_outright_per_game"]) == (1, -0.5)
    assert tried["lead_per_game"] == pytest.approx(-0.25)
    # No model was asked for its leads: nothing is said of following them.
    assert "led" not in tried and "head" not in tried
    assert tried["examined_otherwise"] == 0
    assert summarize_tried([{**game, "examined_otherwise": 2}])["examined_otherwise"] == 2


def test_a_model_that_follows_its_lead_head_is_read_at_every_margin():
    first = {
        "tried": [
            # It predicts 12 points for an alternative that is 25 ahead.
            decision(0, [0.0, 0.25, -0.5], [0.8, 0.1, 0.1], predicted=[0.0, 0.12, -0.2]),
            # And 7 points for one that is 25 behind.
            decision(0, [0.0, -0.25], [0.7, 0.3], predicted=[0.01, 0.08]),
            # Its first choice is another answer already, 50 behind: the head
            # leads it back to the pilot's.
            decision(2, [0.0, 0.0, -0.5], [0.2, 0.1, 0.7], predicted=[0.3, 0.0, 0.0]),
            # Nothing is predicted to lead here.
            decision(0, [0.0, 0.5], [0.9, 0.1], predicted=[0.0, -0.1]),
        ]
    }
    second = {"tried": [decision(None, [0.0, 0.125], [0.5, 0.1], predicted=[0.0, 0.9])]}

    tried = summarize_tried([first, second])

    led = {part["margin"]: part for part in tried["led"]}
    assert sorted(led) == [0.05, 0.1, 0.15, 0.2, 0.3]
    # By 5 points: the first three decisions change their answer.
    assert (led[0.05]["changed"], led[0.05]["unmeasured"]) == (3, 0)
    assert led[0.05]["gain_by_game"] == pytest.approx([0.25 - 0.25 + 0.5, 0.0])
    assert led[0.05]["gain_per_game"] == pytest.approx(0.25)
    assert (led[0.05]["pilot"], led[0.05]["tried"], led[0.05]["untried"]) == (2, 2, 1)
    assert led[0.05]["lead_by_game"] == pytest.approx([0.0, 0.0])
    # By 10: the second no longer does, and by 30 only the third.
    assert (led[0.1]["changed"], led[0.1]["gain_by_game"][0]) == (2, pytest.approx(0.75))
    assert (led[0.3]["changed"], led[0.3]["gain_by_game"][0]) == (1, pytest.approx(0.5))
    # A first choice the search did not try is left as it is.
    assert all(part["untried"] == 1 for part in tried["led"])

    # What the head knows: its predictions over the pilot's against the leads.
    head = tried["head"]
    assert head["alternatives"] == 7
    assert -1.0 <= head["correlation"] <= 1.0
    best = head["highest"][0]
    assert (best["alternatives"], best["predicted"], best["lead"]) == (1, 0.9, 0.125)
    assert head["highest"][-1]["alternatives"] == 2


def test_the_rules_changes_are_counted_apart_by_where_they_start():
    first = {
        "tried": [
            # From the pilot's answer: to one 25 ahead, to one level with it,
            # and to one the search has no lead for.
            decision(0, [0.0, 0.25], [0.9, 0.1], predicted=[0.0, 0.2]),
            decision(0, [0.0, 0.0], [0.9, 0.1], predicted=[0.0, 0.2]),
            decision(0, [0.0, None], [0.9, 0.1], predicted=[0.0, 0.2]),
            # From another answer, 50 behind, back to the pilot's.
            decision(1, [0.0, -0.5], [0.1, 0.9], predicted=[0.2, 0.0]),
        ]
    }
    second = {
        "tried": [
            # From one other answer to another: 25 behind to 12.5 ahead, and
            # 12.5 ahead to 25 behind.
            decision(1, [0.0, -0.25, 0.125], [0.1, 0.8, 0.1], predicted=[0.0, 0.0, 0.2]),
            decision(2, [0.0, -0.25, 0.125], [0.1, 0.1, 0.8], predicted=[0.0, 0.2, 0.0]),
        ]
    }

    rule = summarize_tried([first, second])["led"][0]

    assert (rule["margin"], rule["changed"], rule["unmeasured"]) == (0.05, 6, 1)
    split = rule["split"]
    assert sorted(split) == ["between_others", "from_pilot", "to_pilot"]
    left = split["from_pilot"]
    assert (left["changed"], left["unmeasured"]) == (3, 1)
    assert (left["ahead"], left["level"], left["behind"]) == (1, 1, 0)
    assert left["gain_per_game"] == pytest.approx(0.125)
    back = split["to_pilot"]
    assert (back["changed"], back["ahead"], back["gain_per_game"]) == (1, 1, pytest.approx(0.25))
    between = split["between_others"]
    assert (between["changed"], between["ahead"], between["behind"]) == (2, 1, 1)
    assert between["gain_per_game"] == pytest.approx(0.0)
    # The parts add up to the whole.
    assert sum(part["changed"] for part in split.values()) == rule["changed"]
    assert sum(part["gain_per_game"] for part in split.values()) == pytest.approx(
        rule["gain_per_game"]
    )
    assert all("gain_by_game" not in part for part in split.values())


def test_games_on_a_runs_deals_are_set_beside_the_pilots_and_the_searchs():
    results = [
        game(),
        game(outcome="loss"),
        game(opponent="heroes", outcome="loss"),
        game(opponent="heroes", outcome="draw"),
    ]
    summary = summarize_deals(
        results,
        {"pilot": ["win", "win", "loss", "loss"], "search": ["win", "win", "win", "draw"]},
    )

    assert (summary["games"], summary["wins"]) == (4, 1)
    assert summary["outcomes"] == ["win", "loss", "loss", "draw"]
    assert summary["references"]["pilot"]["win_rate"] == 0.5
    assert summary["references"]["search"]["wins"] == 3
    assert summary["references"]["search"]["draws"] == 1
    assert summary["references"]["search"]["by_opponent"]["heroes"]["win_rate"] == 0.5
    assert summary["references"]["pilot"]["outcomes"] == ["win", "win", "loss", "loss"]
    assert summary["paired"]["pilot"] == {
        "both": 1,
        "model_only": 0,
        "pilot_only": 1,
        "neither": 2,
    }
    assert summary["paired"]["search"]["search_only"] == 2
    with pytest.raises(ValueError, match="pilot has 1 outcomes for 4 deals"):
        summarize_deals(results, {"pilot": ["win"]})
