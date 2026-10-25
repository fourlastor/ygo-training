import pytest

from ygo_training.config import (
    BOARD_REPRESENTATIONS,
    GAE_UNIT,
    GOLDEN_TRAINING_PROTOCOL,
    TrainerConfig,
)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("workers", 0),
        ("workers", 65),
        ("duels", 0),
        ("duels", 1025),
        ("learning_rate", 0.0),
        ("rollout_temperature", 0.0),
        ("duel_gae_lambda", -0.1),
        ("duel_gae_lambda", 1.1),
        ("board_representation", "unknown"),
        ("board_representation", "legacy-mean-pool"),
        ("entropy_coefficient", -0.1),
        ("target_kl", 0.0),
        ("learner_deck", ""),
        ("opponent_decks", []),
        ("imitation_only_updates", -1),
        ("teacher_imitation_decay_updates", -1),
        ("pool_threads", 0),
    ],
)
def test_trainer_config_rejects_invalid_values(field, value):
    with pytest.raises(ValueError):
        TrainerConfig(**{field: value})


def test_outcome_only_credit_assignment_defaults_to_full_trace():
    assert TrainerConfig().duel_gae_lambda == 1.0


def test_transactions_are_the_only_gae_unit():
    assert GAE_UNIT == "transactions"
    with pytest.raises(TypeError):
        TrainerConfig(gae_unit="causal-transaction")


def test_both_board_representations_are_explicit():
    assert BOARD_REPRESENTATIONS == ("mean-pool", "individual-tokens")
    assert TrainerConfig().board_representation == "mean-pool"
    assert (
        TrainerConfig(board_representation="individual-tokens").board_representation
        == "individual-tokens"
    )


def test_duel_count_must_cover_every_worker():
    with pytest.raises(ValueError, match="duels must be at least workers"):
        TrainerConfig(workers=4, duels=3)
    assert TrainerConfig(workers=4, duels=4).duels == 4


def test_the_cli_defaults_are_recorded():
    assert GOLDEN_TRAINING_PROTOCOL.workers == 16
    assert GOLDEN_TRAINING_PROTOCOL.duels == 512
    assert GOLDEN_TRAINING_PROTOCOL.ppo_epochs == 1
    assert GOLDEN_TRAINING_PROTOCOL.minibatch == 256
    assert GOLDEN_TRAINING_PROTOCOL.learning_rate == pytest.approx(3e-4)
    assert GOLDEN_TRAINING_PROTOCOL.duel_gae_lambda == pytest.approx(0.95)


def test_the_learner_meets_every_other_deck_unless_told():
    assert TrainerConfig().opponent_decks is None
    assert TrainerConfig(opponent_decks=["monarch"]).opponent_decks == ["monarch"]


def test_imitation_only_updates_and_teacher_imitation_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        TrainerConfig(imitation_only_updates=2, teacher_imitation_weight=0.1)
    assert TrainerConfig(imitation_only_updates=2).imitation_only_updates == 2
    assert TrainerConfig(teacher_imitation_weight=0.1).teacher_imitation_weight == 0.1


def test_a_zero_policy_weight_needs_teacher_labels():
    with pytest.raises(ValueError, match="requires teacher imitation"):
        TrainerConfig(policy_loss_weight=0.0)
    dagger = TrainerConfig(policy_loss_weight=0.0, teacher_imitation_weight=1.0)
    assert dagger.policy_loss_weight == 0.0


def test_thinned_labels_need_teacher_imitation():
    with pytest.raises(ValueError, match="requires teacher imitation"):
        TrainerConfig(teacher_label_fraction=0.5)
    with pytest.raises(ValueError, match="above 0"):
        TrainerConfig(teacher_imitation_weight=0.1, teacher_label_fraction=0.0)
    thinned = TrainerConfig(teacher_imitation_weight=0.1, teacher_label_fraction=0.5)
    assert thinned.teacher_label_fraction == 0.5


def test_imitation_value_weight_needs_imitation_only_updates():
    with pytest.raises(ValueError, match="imitation-only"):
        TrainerConfig(imitation_value_weight=0.5)
    with pytest.raises(ValueError, match="non-negative"):
        TrainerConfig(imitation_only_updates=1, imitation_value_weight=-1.0)
    assert TrainerConfig(imitation_only_updates=1, imitation_value_weight=0.5)


def test_a_checkpoint_is_imitated_only_in_imitation_updates():
    with pytest.raises(ValueError, match="requires imitation-only updates"):
        TrainerConfig(imitation_teacher="model.pt", pool_threads=1)
    with pytest.raises(ValueError, match="mutually exclusive"):
        TrainerConfig(
            imitation_only_updates=1,
            imitation_teacher="model.pt",
            imitation_records="rows.jsonl",
            pool_threads=1,
        )
    with pytest.raises(ValueError, match="choice"):
        TrainerConfig(
            imitation_only_updates=1,
            imitation_teacher="model.pt",
            imitation_teacher_targets="logits",
            pool_threads=1,
        )
    with pytest.raises(ValueError, match="requires an imitation_teacher"):
        TrainerConfig(imitation_only_updates=1, imitation_teacher_targets="distribution")
    config = TrainerConfig(imitation_only_updates=1, imitation_teacher="model.pt", pool_threads=1)
    assert config.imitation_teacher_targets == "choice"
    assert TrainerConfig().imitation_teacher is None


def test_a_learner_of_several_decks_imitates_a_checkpoint_for_each_of_them():
    several = {
        "opponent_decks": ["burn"],
        "learner_decks": ["blackwing", "cyber"],
        "imitation_only_updates": 1,
        "pool_threads": 1,
    }
    teachers = {"blackwing": "a.pt", "cyber": "b.pt"}
    config = TrainerConfig(imitation_teachers=teachers, **several)
    assert (config.imitation_teacher, config.teachers) == (None, teachers)
    for wrong in ({"blackwing": "a.pt"}, {**teachers, "burn": "c.pt"}, {**teachers, "cyber": ""}):
        with pytest.raises(ValueError, match="for each learner deck"):
            TrainerConfig(imitation_teachers=wrong, **several)
    with pytest.raises(ValueError, match="one learner deck"):
        TrainerConfig(imitation_teacher="a.pt", **several)
    with pytest.raises(ValueError, match="requires imitation-only updates"):
        TrainerConfig(imitation_teachers=teachers, **{**several, "imitation_only_updates": 0})
    # A learner of one deck names its teacher either way, and not both.
    alone = {"imitation_only_updates": 1, "pool_threads": 1}
    assert TrainerConfig(imitation_teacher="a.pt", **alone).teachers == {"blackwing": "a.pt"}
    assert TrainerConfig(imitation_teachers={"blackwing": "a.pt"}, **alone).teachers == {
        "blackwing": "a.pt"
    }
    with pytest.raises(ValueError, match="mutually exclusive"):
        TrainerConfig(imitation_teacher="a.pt", imitation_teachers={"blackwing": "a.pt"}, **alone)
    assert TrainerConfig().teachers == {}


def test_the_weights_of_duels_against_itself_name_pairs_of_the_learners_decks():
    several = {
        "duels": 8,
        "self_play_duels": 4,
        "pool_threads": 1,
        "opponent_decks": ["burn"],
        "learner_decks": ["blackwing", "cyber"],
    }
    weights = {"blackwing>cyber": 2.0, "cyber>cyber": 0.0}
    config = TrainerConfig(self_play_weights=weights, **several)
    assert config.pair_weights == {("blackwing", "cyber"): 2.0, ("cyber", "cyber"): 0.0}
    assert TrainerConfig(**several).pair_weights is None
    for wrong, told in (
        ({"blackwing>burn": 1.0}, "pairs of the learner's decks"),
        ({"blackwing-cyber": 1.0}, "pairs of the learner's decks"),
        ({"blackwing>cyber": -1.0}, "not below zero"),
        ({"blackwing>cyber": 0.0}, "not all zero"),
    ):
        with pytest.raises(ValueError, match=told):
            TrainerConfig(self_play_weights=wrong, **several)
    with pytest.raises(ValueError, match="one or the other"):
        TrainerConfig(self_play_weights=weights, self_play_pairs="mirrors", **several)
    with pytest.raises(ValueError, match="duels against itself of several decks"):
        TrainerConfig(self_play_weights=weights, **{**several, "self_play_duels": 0})


def test_a_learner_of_several_decks_plays_against_named_opponents():
    several = {"learner_decks": ["blackwing", "monarch"]}
    with pytest.raises(ValueError, match="require opponent_decks"):
        TrainerConfig(pool_threads=1, **several)
    with pytest.raises(ValueError, match="first of learner_decks"):
        TrainerConfig(pool_threads=1, opponent_decks=["burn"], learner_decks=["monarch", "burn"])
    with pytest.raises(ValueError, match="each once"):
        TrainerConfig(learner_decks=["blackwing", "blackwing"])
    with pytest.raises(ValueError, match="each once"):
        TrainerConfig(learner_decks=[])
    with pytest.raises(ValueError, match="one learner deck"):
        TrainerConfig(
            pool_threads=1,
            opponent_decks=["burn"],
            imitation_only_updates=1,
            imitation_teacher="model.pt",
            **several,
        )
    with pytest.raises(ValueError, match="one learner deck"):
        TrainerConfig(
            pool_threads=1,
            opponent_decks=["burn"],
            imitation_only_updates=1,
            imitation_records="rows.jsonl",
            **several,
        )
    config = TrainerConfig(pool_threads=1, opponent_decks=["burn"], **several)
    assert config.learners == ["blackwing", "monarch"]
    assert TrainerConfig(learner_deck="monarch").learners == ["monarch"]
    assert TrainerConfig(learner_decks=["blackwing"]).learners == ["blackwing"]


def test_recorded_games_are_imitated_only_in_imitation_updates():
    with pytest.raises(ValueError, match="requires imitation-only updates"):
        TrainerConfig(imitation_records="rows.jsonl")
    with pytest.raises(ValueError, match="search or baseline"):
        TrainerConfig(
            imitation_only_updates=1, imitation_records="rows.jsonl", imitation_game="pilot"
        )
    with pytest.raises(ValueError, match="requires imitation_records"):
        TrainerConfig(imitation_only_updates=1, imitation_games=10)
    with pytest.raises(ValueError, match="at least 1"):
        TrainerConfig(imitation_only_updates=1, imitation_records="rows.jsonl", imitation_games=0)
    config = TrainerConfig(imitation_only_updates=1, imitation_records="rows.jsonl")
    assert (config.imitation_game, config.imitation_games) == ("search", None)


def test_a_network_in_the_other_seat_plays_some_of_an_updates_duels():
    """Duels against itself and against a held checkpoint are of an update's
    duels.  A held checkpoint plays one deck; against itself a learner plays
    as many as it has."""
    with pytest.raises(ValueError, match="requires an opponent_checkpoint"):
        TrainerConfig(duels=8, opponent_checkpoint_duels=4, pool_threads=1)
    with pytest.raises(ValueError, match="at most duels"):
        TrainerConfig(duels=8, self_play_duels=9, pool_threads=1)
    with pytest.raises(ValueError, match="at most duels"):
        TrainerConfig(
            duels=8,
            self_play_duels=5,
            opponent_checkpoint="model.pt",
            opponent_checkpoint_duels=4,
            pool_threads=1,
        )
    with pytest.raises(ValueError, match="must not be negative"):
        TrainerConfig(duels=8, self_play_duels=-1, pool_threads=1)
    several = {"opponent_decks": ["burn"], "learner_decks": ["blackwing", "monarch"]}
    with pytest.raises(ValueError, match="a checkpoint in the other seat plays one learner deck"):
        TrainerConfig(duels=8, opponent_checkpoint="model.pt", pool_threads=1, **several)
    with pytest.raises(ValueError, match="a checkpoint in the other seat plays one learner deck"):
        TrainerConfig(
            duels=8,
            self_play_duels=4,
            opponent_checkpoint="model.pt",
            opponent_checkpoint_duels=2,
            pool_threads=1,
            **several,
        )
    itself = TrainerConfig(duels=8, self_play_duels=4, pool_threads=1, **several)
    assert (itself.self_play_duels, itself.learners) == (4, ["blackwing", "monarch"])
    # Any two of its decks meet, each deck itself alone, or two different
    # decks alone, which a learner of one deck does not have.
    assert itself.self_play_pairs == "all"
    mirrored = TrainerConfig(
        duels=8, self_play_duels=4, self_play_pairs="mirrors", pool_threads=1, **several
    )
    assert mirrored.self_play_pairs == "mirrors"
    crossed = TrainerConfig(
        duels=8, self_play_duels=4, self_play_pairs="cross", pool_threads=1, **several
    )
    assert crossed.self_play_pairs == "cross"
    with pytest.raises(ValueError, match="two different decks"):
        TrainerConfig(duels=8, self_play_duels=4, self_play_pairs="cross", pool_threads=1)
    with pytest.raises(ValueError, match="self_play_pairs must be one of"):
        TrainerConfig(duels=8, self_play_duels=4, self_play_pairs="rings", pool_threads=1)
    with pytest.raises(ValueError, match="duels against itself: self_play_duels"):
        TrainerConfig(duels=8, self_play_pairs="mirrors", pool_threads=1, **several)
    # Every duel of an update may be against a network, and a checkpoint may
    # be held for the suite alone.
    whole = TrainerConfig(
        duels=8, opponent_checkpoint="model.pt", opponent_checkpoint_duels=8, pool_threads=1
    )
    assert (whole.self_play_duels, whole.opponent_checkpoint_duels) == (0, 8)
    assert (
        TrainerConfig(opponent_checkpoint="model.pt", pool_threads=1).opponent_checkpoint_duels == 0
    )
    plain = TrainerConfig()
    assert (plain.self_play_duels, plain.opponent_checkpoint, plain.opponent_checkpoint_duels) == (
        0,
        None,
        0,
    )
