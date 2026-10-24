from collections import Counter

import pytest

from ygo_training.config import TrainerConfig
from ygo_training.orchestration import Coordinator
from ygo_training.records import RecordedGame
from ygo_training.rewards import OutcomeRewardModel
from ygo_training.schedules import (
    OTHER_SEATS,
    allocate_balanced_matchups,
    allocate_balanced_opponents,
    allocate_episodes,
    allocate_matchup_seats,
    allocate_other_seats,
    episode_offsets,
    rollout_games,
)

NOT_ASKED = {"prompts": 0.0, "batches": 0.0, "max_batch": 0.0, "inference_seconds": 0.0}


def ended(games):
    """Games as they come back played: each won by the learner's seat."""
    return [
        {
            "terminal": True,
            "winner": int(game["learner_player"]),
            "learner_player": int(game["learner_player"]),
            "win_reason": 1,
            "turns": 3,
            "opponent_deck": str(game["opponent"]),
            "departures": 0,
        }
        for game in games
    ]


def coordinator(config, **played):
    return Coordinator(
        config=config,
        reward_model=OutcomeRewardModel(),
        opponents=["monarch"],
        pilot_games=played.pop("pilot_games", ended),
        learner_games=played.pop(
            "learner_games", lambda games, seed, labels: (ended(games), NOT_ASKED)
        ),
        **played,
    )


def test_episodes_are_distributed_across_workers():
    assert allocate_episodes(4, 6) == [2, 2, 1, 1]
    assert allocate_episodes(20, 512) == [26] * 12 + [25] * 8
    assert allocate_episodes(1, 1) == [1]
    assert allocate_episodes(4, 0) == [0, 0, 0, 0]
    assert allocate_episodes(4, 12) == [3, 3, 3, 3]


def test_episode_offsets_map_local_worker_indices_to_one_global_stream():
    counts = allocate_episodes(4, 10)
    assert counts == [3, 3, 2, 2]
    assert episode_offsets(counts) == [0, 3, 6, 8]


def test_balanced_opponents_are_exact_deterministic_and_shuffled():
    opponents = ["Blackwing", "Amazoness", "Gemini", "Fabled"]

    first = allocate_balanced_opponents(opponents, 512, seed=2012)
    second = allocate_balanced_opponents(opponents, 512, seed=2012)

    assert first == second
    assert first != opponents * 128
    assert {opponent: first.count(opponent) for opponent in opponents} == {
        opponent: 128 for opponent in opponents
    }


def test_a_learner_of_several_decks_meets_every_opponent_with_each_as_often():
    learners = ["blackwing", "monarch"]
    opponents = ["burn", "heroes", "gishki"]

    first = allocate_balanced_matchups(learners, opponents, 60, seed=2012)

    assert first == allocate_balanced_matchups(learners, opponents, 60, seed=2012)
    assert first != allocate_balanced_matchups(learners, opponents, 60, seed=2013)
    assert {pair: first.count(pair) for pair in set(first)} == {
        (learner, opponent): 10 for learner in learners for opponent in opponents
    }
    assert first[:6] != [(learner, opponent) for learner in learners for opponent in opponents]
    # What does not divide is spread a game at most.
    uneven = allocate_balanced_matchups(learners, opponents, 20, seed=2012)
    assert sorted(uneven.count(pair) for pair in set(uneven)) == [3, 3, 3, 3, 4, 4]


def test_the_games_of_a_rollout_name_the_learners_deck_only_where_it_plays_several():
    opponents = ["burn", "heroes", "gishki", "burn", "heroes", "gishki", "burn"]
    learners = ["blackwing", "monarch", "blackwing", "monarch", "blackwing", "monarch", "monarch"]

    plain = rollout_games(5 * 1024, 3, opponents)
    several = rollout_games(5 * 1024, 3, opponents, learners)

    assert all("learner" not in game for game in plain)
    assert [game["learner"] for game in several] == learners
    # The same deals, seats and seeds whichever deck is played in them.
    assert [{**game, "learner": None} for game in several] == [
        {**game, "learner": None} for game in plain
    ]


def test_rollout_metrics_tell_the_decks_of_a_learner_apart():
    subject = coordinator(TrainerConfig(workers=1, duels=2))
    game = {"terminal": True, "win_reason": 1, "turns": 8, "opponent_deck": "burn"}
    won = {**game, "winner": 0, "learner_player": 0}
    lost = {**game, "winner": 1, "learner_player": 0}
    one = [{**won, "learner_deck": "blackwing"}, {**lost, "learner_deck": "blackwing"}]

    alone = subject._rollout_metrics(one, 0, 0, 0, 0.0, 1.0, 0.0, 0)
    both = subject._rollout_metrics(
        [*one, {**won, "learner_deck": "monarch"}], 0, 0, 0, 0.0, 1.0, 0.0, 0
    )

    assert "learner_deck_duels" not in alone and "learner_deck_win_rate" not in alone
    assert both["learner_deck_duels"] == {"blackwing": 2.0, "monarch": 1.0}
    assert both["learner_deck_win_rate"] == {"blackwing": 0.5, "monarch": 1.0}
    assert both["learner_win_rate"] == pytest.approx(2 / 3)


def test_resume_indices_are_owned_by_coordinator_schedule():
    subject = coordinator(TrainerConfig())
    subject.set_rollout_index(12)
    subject.set_training_schedule_index(7)

    assert subject.schedule.rollout_index == 12
    assert subject.schedule.update_index == 7


def test_rollout_metrics_treat_focused_cutoff_as_loss():
    config = TrainerConfig()
    subject = coordinator(config)
    trajectories = [
        {
            "learner_player": 0,
            "terminal": False,
            "winner": None,
            "win_reason": None,
            "turns": 10,
            "opponent_deck": "monarch",
        }
    ]

    metrics = subject._rollout_metrics(trajectories, 0, 0, 0, 0.0, 1.0, 0.0, 0)

    assert metrics["learner_total_reward"] == pytest.approx(-1.0)
    assert metrics["learner_mean_reward"] == pytest.approx(-1.0)


def test_rollout_metrics_report_per_opponent_duel_counts():
    subject = coordinator(TrainerConfig())
    trajectories = [
        {
            "learner_player": 0,
            "terminal": True,
            "winner": 0,
            "win_reason": 0,
            "turns": 3,
            "matchup": "random",
            "opponent_deck": opponent,
        }
        for opponent in ("Fabled", "Blackwing", "Fabled")
    ]

    metrics = subject._rollout_metrics(trajectories, 0, 0, 0, 0.0, 1.0, 0.0, 0)

    assert metrics["opponent_deck_duels"] == {
        "Blackwing": 1.0,
        "Fabled": 2.0,
    }


def test_imitation_only_updates_hand_out_recorded_games_when_given():
    games = [
        RecordedGame("blackwing", "monarch", index % 2, 100 + index, 0, f"d{index}", (), None, "{}")
        for index in range(6)
    ]
    given = []

    def replayed(handed):
        given.append(("records", handed))
        return ended(handed)

    def drawn(handed, seed, labels):
        given.append(("games", handed))
        return ended(handed), NOT_ASKED

    subject = coordinator(
        TrainerConfig(
            workers=2, duels=4, seed=3, imitation_only_updates=2, imitation_records="rows.jsonl"
        ),
        learner_games=drawn,
        replayed_games=replayed,
        recorded=games,
    )

    def handed_out():
        given.clear()
        _, metrics = subject.collect()
        ((kind, handed),) = given
        return kind, handed, metrics

    kind, first, metrics = handed_out()
    assert kind == "records" and len(first) == 4
    assert (metrics["recorded_games"], metrics["recorded_pass"]) == (6.0, 0.0)
    # Each game keeps its place in the update, and the share it is dealt in
    # is the one a game played now has there.
    assert [game["index"] for game in first] == [0, 1, 2, 3]
    assert [game["episode_index"] for game in first] == [0, 1, 2, 3]
    assert [game["worker"] for game in first] == [0, 0, 1, 1]
    kind, second, metrics = handed_out()
    seeds = [game["seed"] for game in first + second]
    # The first pass ends in the second update, which goes on with the next.
    assert sorted(seeds[:6]) == [100, 101, 102, 103, 104, 105]
    assert metrics["recorded_pass"] == 0.0
    # Once the imitation-only updates are over the learner plays new duels.
    kind, third, metrics = handed_out()
    assert kind == "games" and len(third) == 4
    assert all("record" not in game for game in third)
    assert "recorded_games" not in metrics
    # A continued run goes on where it stood in the games.
    subject.set_training_schedule_index(1)
    kind, again, metrics = handed_out()
    assert kind == "records" and again == second


def test_a_learner_of_several_decks_meets_itself_with_any_two_of_them():
    """Of an update's duels, those against itself have one of the learner's
    decks in each seat, every pair as often as any other; the others are its
    decks against the pilots, every pairing as often as any other; the two
    kinds shuffled among each other, and the same every time."""
    learners = ["blackwing", "lightsworn", "gishki"]
    opponents = ["monarch", "burn", "heroes", "blackwing"]
    owns, others, seats = allocate_matchup_seats(learners, opponents, 60, itself=36, seed=7)
    assert (owns, others, seats) == allocate_matchup_seats(
        learners, opponents, 60, itself=36, seed=7
    )
    assert len(owns) == len(others) == len(seats) == 60
    assert Counter(seats) == {"self": 36, "pilot": 24}
    # Against itself: each of the nine pairs four times, so that a seat has
    # each deck as often, and two decks meet twice as often as one itself.
    itself = Counter((own, other) for own, other, seat in zip(owns, others, seats) if seat == "self")
    assert itself == {(own, other): 4 for own in learners for other in learners}
    # Against the pilots: each of the twelve pairings twice.
    piloted = Counter((own, other) for own, other, seat in zip(owns, others, seats) if seat != "self")
    assert piloted == {(own, other): 2 for own in learners for other in opponents}
    # Not one kind after the other, and other duels for another seed.
    assert seats != sorted(seats) and seats != sorted(seats, reverse=True)
    assert allocate_matchup_seats(learners, opponents, 60, itself=36, seed=8) != (owns, others, seats)
    # To within a game where the duels do not share out evenly.
    uneven = allocate_matchup_seats(learners, opponents, 31, itself=11, seed=7)
    against = Counter(zip(uneven[0], uneven[1], uneven[2]))
    assert {count for (_, _, seat), count in against.items() if seat == "self"} <= {1, 2}
    assert sum(count for (_, _, seat), count in against.items() if seat == "self") == 11
    assert {count for (_, _, seat), count in against.items() if seat == "pilot"} <= {1, 2}
    # None against itself is the learner's decks against the pilots, and
    # every duel may be against itself.
    assert set(allocate_matchup_seats(learners, opponents, 24, itself=0, seed=7)[2]) == {"pilot"}
    whole = allocate_matchup_seats(learners, opponents, 18, itself=18, seed=7)
    assert set(whole[2]) == {"self"} and set(whole[1]) == set(learners)
    with pytest.raises(ValueError, match="of the update's duels"):
        allocate_matchup_seats(learners, opponents, 8, itself=9, seed=7)
    with pytest.raises(ValueError, match="of the update's duels"):
        allocate_matchup_seats(learners, opponents, 8, itself=-1, seed=7)

    games = rollout_games(3 * 1024, 2, others, owns, seats)
    assert [game["learner"] for game in games] == owns
    assert [game["opponent"] for game in games] == others
    assert [game["other"] for game in games] == seats


def test_a_learner_of_several_decks_meets_itself_in_its_mirrors_alone():
    """With `mirrors` the duels against itself have one deck in both seats,
    every deck as often as any other; the duels against the pilots are the
    same, in the same places, as without it."""
    learners = ["blackwing", "lightsworn", "gishki"]
    opponents = ["monarch", "burn", "heroes", "blackwing"]
    owns, others, seats = allocate_matchup_seats(
        learners, opponents, 60, itself=36, seed=7, mirrors=True
    )
    itself = Counter(
        (own, other) for own, other, seat in zip(owns, others, seats) if seat == "self"
    )
    assert itself == {(deck, deck): 12 for deck in learners}
    every = allocate_matchup_seats(learners, opponents, 60, itself=36, seed=7)
    assert every[2] == seats
    assert [(own, other) for own, other, seat in zip(*every) if seat == "pilot"] == [
        (own, other) for own, other, seat in zip(owns, others, seats) if seat == "pilot"
    ]
    # An update of the network of two decks: 260 of its 520 duels.
    two = allocate_matchup_seats(
        ["blackwing", "cyber"], opponents, 520, itself=260, seed=2012, mirrors=True
    )
    assert Counter((own, other) for own, other, seat in zip(*two) if seat == "self") == {
        ("blackwing", "blackwing"): 130,
        ("cyber", "cyber"): 130,
    }
    # To within a game where the duels do not share out evenly.
    uneven = allocate_matchup_seats(learners, opponents, 31, itself=11, seed=7, mirrors=True)
    mirrored = Counter(own for own, other, seat in zip(*uneven) if seat == "self" and own == other)
    assert sum(mirrored.values()) == 11 and set(mirrored.values()) <= {3, 4}


def test_an_updates_duels_are_shared_out_by_who_sits_in_the_other_seat():
    """The pilots' duels as they are allocated, the duels against itself and
    against a held checkpoint in the mirror, shuffled among each other and
    the same every time."""
    pilots = allocate_balanced_opponents(["monarch", "burn", "heroes"], 6, seed=7)
    decks, seats = allocate_other_seats(pilots, "blackwing", itself=4, checkpoint=2, seed=7)
    assert (decks, seats) == allocate_other_seats(
        pilots, "blackwing", itself=4, checkpoint=2, seed=7
    )
    assert len(decks) == len(seats) == 12 and set(seats) == set(OTHER_SEATS)
    assert sorted(zip(decks, seats)) == sorted(
        [(deck, "pilot") for deck in pilots]
        + [("blackwing", "self")] * 4
        + [("blackwing", "checkpoint")] * 2
    )
    # Not one kind after the other, and the pilots' duels in their own order.
    assert seats != sorted(seats) and seats != sorted(seats, reverse=True)
    assert [deck for deck, seat in zip(decks, seats) if seat == "pilot"] != sorted(pilots)
    assert allocate_other_seats(pilots, "blackwing", itself=4, checkpoint=2, seed=8)[1] != seats
    # None but the pilots': nothing is added.
    assert (
        allocate_other_seats(pilots, "blackwing", itself=0, checkpoint=0, seed=7)[1]
        == ["pilot"] * 6
    )
    with pytest.raises(ValueError, match="must not be negative"):
        allocate_other_seats(pilots, "blackwing", itself=-1, checkpoint=0, seed=7)

    games = rollout_games(3 * 1024, 2, decks, None, seats)
    assert [game["other"] for game in games] == seats
    assert [game["opponent"] for game in games] == decks
    # The deals and the seats are those of the same duels with pilots throughout.
    plain = rollout_games(3 * 1024, 2, decks)
    assert all("other" not in game for game in plain)
    assert [{key: value for key, value in game.items() if key != "other"} for game in games] == (
        plain
    )
