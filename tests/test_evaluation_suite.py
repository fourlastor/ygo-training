import json

import pytest

from ygo_training.__main__ import _run_evaluation_suite
from ygo_training.evaluation_suite import (
    against_checkpoint,
    joined_evaluation_suites,
    load_evaluation_suite,
    wilson_interval,
)


class Trainer:
    """A stand-in for the trainer: it records the cases it is asked to play."""

    def __init__(self, fail: bool = False, held: object = None):
        self.decks = {"blackwing": "Blackwing Assassin", "monarch": "Emperor, Arise!"}
        self.fail = fail
        self.played = []
        # The checkpoint held for the other seat, and the cases played against it.
        self.held = held
        self.against_it = []
        # The deck list each case's learner played in place of its deck's own.
        self.lists = []

    def evaluate_pool(
        self, learner, opponents, games, seed, *, pilot=False, checkpoint=False, learner_list=None
    ):
        if self.fail:
            raise RuntimeError("the duel failed")
        self.played.append((learner, opponents, games, seed, pilot))
        self.lists.append(learner_list)
        if checkpoint:
            self.against_it.append((learner, opponents, games, seed, pilot))
        return [
            {
                "opponent": opponent,
                "learner_player": index % 2,
                "outcome": "win" if opponent == opponents[0] else "loss",
                "terminal": True,
                "win_reason": 1,
                "turns": 8,
                "decisions": 30,
            }
            for opponent in opponents
            for index in range(games)
        ]


def suite():
    return {
        "name": "test",
        "primary": "mirror",
        "cases": [
            {"name": "mirror", "learner": "blackwing", "opponent": "blackwing", "games": 4},
            {"name": "monarch", "learner": "blackwing", "opponent": "monarch", "games": 2},
        ],
    }


def test_each_case_is_played_on_seeds_of_its_own():
    trainer = Trainer()

    result = _run_evaluation_suite(trainer, suite(), seed=2012)

    assert trainer.played == [
        ("blackwing", ["blackwing"], 4, 2012, False),
        ("blackwing", ["monarch"], 2, 10_002_012, False),
    ]
    assert result["mirror"]["games"] == 4 and result["monarch"]["win_rate"] == 1.0


def test_a_list_from_a_file_is_played_in_every_case():
    trainer = Trainer()

    _run_evaluation_suite(trainer, suite(), seed=2012, learner_list="variant.ydk")

    assert trainer.lists == ["variant.ydk", "variant.ydk"]
    assert [played[:4] for played in trainer.played] == [
        ("blackwing", ["blackwing"], 4, 2012),
        ("blackwing", ["monarch"], 2, 10_002_012),
    ]


def test_suites_joined_for_a_learner_of_several_decks_keep_each_decks_own_deals():
    other = {
        "name": "other",
        "primary": "burn",
        "cases": [
            {"name": "burn", "learner": "monarch", "opponent": "blackwing", "games": 2},
            {"name": "second", "learner": "monarch", "opponent": "monarch", "games": 2},
        ],
    }
    trainer = Trainer()

    joined = joined_evaluation_suites([suite(), other])
    result = _run_evaluation_suite(trainer, joined, seed=2012)

    assert (joined["name"], joined["primary"]) == ("test+other", "mirror")
    assert [case["seed_index"] for case in joined["cases"]] == [0, 1, 0, 1]
    # A deck's games are the ones its own suite deals it.
    alone = Trainer()
    _run_evaluation_suite(alone, other, seed=2012)
    assert (
        trainer.played[2:]
        == alone.played
        == [
            ("monarch", ["blackwing"], 2, 2012, False),
            ("monarch", ["monarch"], 2, 10_002_012, False),
        ]
    )
    assert list(result) == ["mirror", "monarch", "burn", "second"]
    with pytest.raises(ValueError, match="unique"):
        joined_evaluation_suites([suite(), suite()])
    with pytest.raises(ValueError, match="negative seed_index"):
        joined_evaluation_suites([{**other, "cases": [{**other["cases"][0], "seed_index": -1}]}])


def test_a_case_tells_how_each_game_ended_only_when_asked():
    """A training run's rows hold the counts; a measurement holds every game,
    for two players of the same deals to be set beside each other."""
    trainer = Trainer()
    counted = _run_evaluation_suite(trainer, suite(), seed=2012)
    each = _run_evaluation_suite(trainer, suite(), seed=2012, outcomes=True)
    assert all("outcomes" not in case for case in counted.values())
    assert each["mirror"]["outcomes"] == ["win"] * 4 and each["monarch"]["outcomes"] == ["win"] * 2
    assert {name: {**case, "outcomes": None} for name, case in each.items()} == {
        name: {**case, "outcomes": None} for name, case in counted.items()
    }


def test_the_pilots_can_play_the_suite_as_the_reference():
    trainer = Trainer()
    _run_evaluation_suite(trainer, suite(), seed=2012, pilot=True)
    assert all(played[4] for played in trainer.played)


def test_evaluation_suite_names_fail_fast_case():
    with pytest.raises(RuntimeError, match="Evaluation case 'mirror' failed"):
        _run_evaluation_suite(Trainer(fail=True), suite(), seed=2012)


def test_checkpoint_screen_can_retain_case_errors():
    result = _run_evaluation_suite(
        Trainer(fail=True),
        suite(),
        seed=2012,
        continue_on_case_error=True,
    )

    assert result == {
        name: {"error": "the duel failed", "error_type": "RuntimeError", "games_requested": games}
        for name, games in (("mirror", 4), ("monarch", 2))
    }


def test_a_case_of_several_opponents_plays_them_all_at_once():
    several = suite()
    several["cases"][1]["opponent"] = "monarch, blackwing"
    trainer = Trainer()

    result = _run_evaluation_suite(trainer, several, seed=2012, outcomes=True)

    # So many games against each, in one go: the pool deals each opponent
    # seeds of its own.
    assert trainer.played[1] == ("blackwing", ["monarch", "blackwing"], 2, 10_002_012, False)
    found = result["monarch"]
    assert found["games"] == 4
    assert {name: tally["wins"] for name, tally in found["by_opponent"].items()} == {
        "monarch": 2,
        "blackwing": 0,
    }
    assert found["outcomes"] == ["win", "win", "loss", "loss"]


def test_a_case_names_decks_of_the_pool():
    unknown = suite()
    unknown["cases"][0]["opponent"] = "no-such-deck"
    with pytest.raises(ValueError, match="Unknown deck"):
        _run_evaluation_suite(Trainer(), unknown, seed=2012)


def test_wilson_interval_penalizes_small_uncertain_samples():
    small = wilson_interval(7, 10)
    large = wilson_interval(700, 1000)
    assert small[0] < large[0] < 0.7 < large[1] < small[1]


def load(tmp_path, content):
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(content))
    return load_evaluation_suite(path)


def test_a_suite_names_its_primary_case(tmp_path):
    assert load(tmp_path, suite())["primary"] == "mirror"
    unnamed = suite()
    del unnamed["primary"]
    assert load(tmp_path, unnamed)["primary"] == "mirror"
    unnamed["primary"] = "absent"
    with pytest.raises(ValueError, match="Unknown primary"):
        load(tmp_path, unnamed)


def test_suite_loader_rejects_what_cannot_be_played(tmp_path):
    odd = suite()
    odd["cases"][0]["games"] = 3
    with pytest.raises(ValueError, match="even game count"):
        load(tmp_path, odd)
    repeated = suite()
    repeated["cases"][1]["name"] = "mirror"
    with pytest.raises(ValueError, match="unique"):
        load(tmp_path, repeated)
    with pytest.raises(ValueError, match="non-empty cases"):
        load(tmp_path, {"cases": []})
    # Suites written for the old engine named how opponents were played.
    legacy = suite()
    legacy["cases"][0]["opponent_policy"] = "random"
    with pytest.raises(ValueError, match="unsupported policy"):
        load(tmp_path, legacy)
    legacy["cases"][0]["opponent_policy"] = "ygo-policies"
    legacy["cases"][0]["deck_mode"] = "declared"
    assert load(tmp_path, legacy)["cases"][0]["games"] == 4


def held():
    """A suite with a case of the mirror against a checkpoint held for the
    other seat, beside one against the deck's pilot."""
    return {
        "name": "test",
        "primary": "mirror",
        "cases": [
            {"name": "mirror", "learner": "blackwing", "opponent": "blackwing", "games": 4},
            {
                "name": "held",
                "learner": "blackwing",
                "opponent": "blackwing",
                "opponent_policy": "checkpoint",
                "games": 6,
            },
        ],
    }


def test_a_case_is_played_against_the_checkpoint_that_is_held(tmp_path):
    """The case says its opponent is a checkpoint and not which: the
    measurement names it.  It is dealt as any case is, on seeds of its own."""
    suite_ = load(tmp_path, held())
    assert against_checkpoint(suite_["cases"][1]) and not against_checkpoint(suite_["cases"][0])
    trainer = Trainer(held=object())
    result = _run_evaluation_suite(trainer, suite_, seed=2012)
    assert trainer.played == [
        ("blackwing", ["blackwing"], 4, 2012, False),
        ("blackwing", ["blackwing"], 6, 10_002_012, False),
    ]
    assert trainer.against_it == [("blackwing", ["blackwing"], 6, 10_002_012, False)]
    assert result["held"]["games"] == 6
    # The pilot of the learner's deck can be measured against it too.
    reference = Trainer(held=object())
    _run_evaluation_suite(reference, suite_, seed=2012, pilot=True)
    assert reference.against_it == [("blackwing", ["blackwing"], 6, 10_002_012, True)]


def test_a_case_against_a_checkpoint_needs_one_held():
    with pytest.raises(ValueError, match="--opponent-checkpoint"):
        _run_evaluation_suite(Trainer(), held(), seed=2012)
    # Not an error of one case, which a screen of checkpoints would keep.
    with pytest.raises(ValueError, match="--opponent-checkpoint"):
        _run_evaluation_suite(Trainer(), held(), seed=2012, continue_on_case_error=True)
