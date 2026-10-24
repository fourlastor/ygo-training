"""A decision as the arrays the policy reads, as the library builds them,
held to what the trainer says of them (``schema.py``)."""

import numpy as np
import pytest
import torch

from tests.views import (
    ACTIVATE,
    ATTACK,
    GRAVEYARD,
    HAND,
    MONSTER,
    OTHER_MONSTER,
    PASS,
    SELECT_CARDS,
    SPELL,
    SPELL_TRAP_ZONE,
    TOGGLE,
    TRAP,
    batched,
    card,
    choice,
    hidden,
    library,
    named,
    packed,
    view,
)
from ygo_training.environment import DuelError
from ygo_training.model import PolicyValueNet
from ygo_training.schema import (
    CARD_STATUS,
    CATEGORY_COUNT,
    CONTEXT_SLOTS,
    MAX_ACTION_MEMBERS,
    MAX_CARDS,
    MAX_HISTORY,
    NUMERIC_COUNT,
    PACKED_ARRAY_SCHEMA,
    SCALARS,
    packed_policy_input_schema_digest,
    validate_packed_policy_input,
)

STATUS = {name: index for index, name in enumerate(CARD_STATUS)}
SCALAR = {name: index for index, name in enumerate(SCALARS)}
SLOT = {name: index for index, name in enumerate(CONTEXT_SLOTS)}


def combat(value: int) -> float:
    """An ATK or DEF as the input scales it."""
    return -1.0 if value < 0 else min(value, 10_000) / 4_000


def test_a_packed_decision_satisfies_the_schema():
    validate_packed_policy_input(packed())
    busy = packed(
        cards=[card(), hidden(), card(SPELL, location=HAND, position=0)],
        choices=[
            choice(ACTIVATE, named(2, SPELL, location=HAND, position=0)),
            choice(TOGGLE, named(0), members=[named(0), named(1, 0, controller=1)]),
            choice(PASS),
        ],
        history=[{"kind": ACTIVATE, "code": SPELL, "location": HAND, "description": 0}],
    )
    validate_packed_policy_input(busy)
    assert set(busy) == set(PACKED_ARRAY_SCHEMA)


def test_the_schema_refuses_what_it_does_not_describe():
    item = packed()
    with pytest.raises(ValueError, match="extra"):
        validate_packed_policy_input({**item, "state_sequence": np.zeros(1, dtype=np.int64)})
    with pytest.raises(ValueError, match="dtype"):
        validate_packed_policy_input({**item, "scalars": item["scalars"].astype(np.float64)})
    assert len(packed_policy_input_schema_digest()) == 64


def test_a_hidden_card_has_a_place_and_no_identity():
    item = packed(cards=[hidden(attacked=True, targeted=True), card()])

    assert item["state_card"][0] == 0
    assert not item["state_categories"][0].any()
    assert not item["state_numeric"][0].any()
    assert item["state_controller"][0] == 1
    assert item["state_position"][0] == 8
    # What happens to it is seen all the same.
    status = item["state_status"][0]
    assert status[STATUS["known"]] == 0.0
    assert status[STATUS["attacked"]] == 1.0
    assert status[STATUS["targeted"]] == 1.0
    assert item["state_status"][1][STATUS["known"]] == 1.0


def test_a_card_is_shown_with_its_printed_data():
    """Blackwing Armed Wing, named by its code alone: a Level 6 Synchro Effect
    monster with 2300 ATK and 1000 DEF."""
    item = packed(choices=[choice(TOGGLE, named(-1, 76913983, location=1)), choice(PASS)])
    categories, numeric = item["action_categories"][0], item["action_numeric"][0]
    assert (len(categories), len(numeric)) == (CATEGORY_COUNT, NUMERIC_COUNT)
    assert categories[0] == 1  # a monster
    assert categories[3] == 6  # its level
    assert numeric[0] == pytest.approx(combat(2300)) and numeric[1] == pytest.approx(combat(1000))
    assert numeric[3] == 1.0  # an Effect monster
    assert numeric[6] == 1.0  # a Synchro monster


def test_a_card_is_shown_as_it_is_now():
    item = packed(
        cards=[card(attack=2300, defense=700, level=5, counters=3, battles=1, can_attack=True)]
    )
    # Blackwing - Blizzard the Far North as it is printed: 1300 ATK, 0 DEF.
    printed = packed(cards=[card(attack=1300, defense=0)])

    assert item["state_card"][0] == MONSTER
    assert tuple(item["state_categories"][0]) == tuple(printed["state_categories"][0])
    assert item["state_numeric"][0][0] == pytest.approx(combat(2300))
    assert item["state_numeric"][0][1] == pytest.approx(combat(700))
    assert printed["state_numeric"][0][0] == pytest.approx(combat(1300))
    # The rest of the printed data stays.
    assert tuple(item["state_numeric"][0][2:]) == tuple(printed["state_numeric"][0][2:])
    status = item["state_status"][0]
    assert status[STATUS["level"]] == pytest.approx(5 / 12)
    assert status[STATUS["counters"]] == pytest.approx(0.3)
    assert status[STATUS["battles"]] == pytest.approx(1 / 3)
    assert status[STATUS["can_attack"]] == 1.0


def test_a_choice_names_the_card_of_the_view_or_only_its_code():
    boosted = card(attack=3000)
    item = packed(
        cards=[boosted],
        choices=[
            choice(ATTACK, named(0)),
            # A card that is not among the view's: a search in the Deck.
            choice(TOGGLE, named(-1, OTHER_MONSTER, location=1, position=0)),
            # The place now holds another card.
            choice(TOGGLE, named(0, OTHER_MONSTER)),
            choice(PASS),
        ],
    )

    assert item["action_numeric"][0][0] == pytest.approx(combat(3000))
    assert item["action_status"][0][STATUS["known"]] == 1.0
    # Blackwing - Shura the Blue Flame as it is printed: 1800 ATK, 1200 DEF.
    printed = packed(cards=[card(OTHER_MONSTER, attack=1800, defense=1200)])["state_numeric"][0]
    for row in (1, 2):
        assert item["action_card"][row] == OTHER_MONSTER
        assert tuple(item["action_numeric"][row]) == pytest.approx(printed)
    assert item["action_location"][1] == 1
    assert list(item["action_scalars"][:, 0]) == [1.0, 1.0, 1.0, 0.0]
    assert not item["action_card"][3] and not item["action_numeric"][3].any()


def test_the_members_of_an_answer_keep_their_order_and_values():
    item = packed(
        cards=[card(), card(OTHER_MONSTER)],
        choices=[
            choice(TOGGLE, named(1, OTHER_MONSTER), members=[named(0, value=2), named(1, OTHER_MONSTER, value=4)]),
            choice(PASS),
        ],
    )

    assert item["action_member_card"].shape == (2, 2)
    assert list(item["action_member_card"][0]) == [MONSTER, OTHER_MONSTER]
    assert list(item["action_member_value"][0]) == [2, 4]
    assert list(item["action_member_mask"][0]) == [True, True]
    assert item["action_member_order"][0][1] == pytest.approx(1 / MAX_ACTION_MEMBERS)
    assert not item["action_member_mask"][1].any()
    assert item["action_scalars"][0][4] == pytest.approx(2 / 8)


def test_the_context_names_what_the_decision_is_about():
    attacker = card(OTHER_MONSTER, attacking=True, attack=1800)
    target = hidden(attacked=True)
    item = packed(
        cards=[attacker, target, card(TRAP, location=SPELL_TRAP_ZONE, chained=True)],
        chain=[
            {"code": SPELL, "controller": 1, "source": -1, "targets": []},
            {"code": TRAP, "controller": 0, "source": 2, "targets": [1]},
        ],
        events=[{"card": -1, "code": MONSTER, "controller": 1, "location": HAND}],
        acting=named(0, OTHER_MONSTER),
        question={**view()["question"], "subject": SPELL},
    )

    mask = item["context_mask"]
    assert mask.all()
    cards = item["context_card"]
    assert cards[SLOT["chain_last"]] == TRAP
    assert cards[SLOT["chain_previous"]] == SPELL
    assert item["context_controller"][SLOT["chain_previous"]] == 1
    assert cards[SLOT["attacker"]] == OTHER_MONSTER
    assert cards[SLOT["attack_target"]] == 0
    assert item["context_status"][SLOT["attack_target"]][STATUS["attacked"]] == 1.0
    assert cards[SLOT["event"]] == MONSTER
    assert cards[SLOT["subject"]] == SPELL
    assert cards[SLOT["acting"]] == OTHER_MONSTER
    assert item["context_numeric"][SLOT["acting"]][0] == pytest.approx(combat(1800))
    assert not packed()["context_mask"].any()


def test_the_scalars_are_the_seats_own_view():
    item = packed(
        life_points=[4000, 6000],
        turn=7,
        turn_player=1,
        piles=[[30, 25], [3, 6], [9, 12], [1, 2], [15, 14]],
        summon_used=True,
        coin_toss=[2, 1],
        step=3,
        chain=[{"code": SPELL, "controller": 1, "source": -1, "targets": []}],
        question={**view()["question"], "forced": True, "minimum": 1, "maximum": 2},
    )
    scalars = item["scalars"]

    assert scalars[SCALAR["own_life_points"]] == pytest.approx(0.5)
    assert scalars[SCALAR["opponent_life_points"]] == pytest.approx(0.75)
    assert scalars[SCALAR["turn_player"]] == -1.0
    assert scalars[SCALAR["own_deck"]] == pytest.approx(30 / 60)
    assert scalars[SCALAR["opponent_hand"]] == pytest.approx(0.6)
    assert scalars[SCALAR["own_hand"]] == pytest.approx(0.3)
    assert scalars[SCALAR["opponent_graveyard"]] == pytest.approx(12 / 60)
    assert scalars[SCALAR["own_banished"]] == pytest.approx(1 / 60)
    assert scalars[SCALAR["opponent_extra"]] == pytest.approx(14 / 15)
    assert scalars[SCALAR["summon_used"]] == 1.0
    assert scalars[SCALAR["chain_length"]] == pytest.approx(1 / 16)
    assert scalars[SCALAR["forced"]] == 1.0
    assert scalars[SCALAR["minimum"]] == pytest.approx(0.1)
    assert scalars[SCALAR["coin_heads"]] == pytest.approx(2 / 3)
    assert scalars[SCALAR["step"]] == pytest.approx(0.3)
    assert packed(turn_player=-1)["scalars"][SCALAR["turn_player"]] == 0.0


def test_long_lists_are_cut_to_the_declared_limits():
    many = packed(
        cards=[card(location=GRAVEYARD, position=0)] * (MAX_CARDS + 5),
        history=[{"kind": PASS, "code": 0, "location": 0, "description": 0}] * (MAX_HISTORY + 3),
    )
    assert many["state_length"][0] == MAX_CARDS
    assert many["history_length"][0] == MAX_HISTORY
    empty = packed()
    assert empty["state_length"][0] == 0 and len(empty["state_card"]) == 1
    assert empty["history_length"][0] == 0 and len(empty["history_card"]) == 1


def test_no_deck_list_is_in_the_input():
    """Format 15 dropped both (experiments/2026-10-15-without-deck-lists.md):
    of the decks the input holds only how many cards each pile has."""
    assert not [key for key in packed() if "_deck" in key]
    assert not [key for key in PACKED_ARRAY_SCHEMA if "_deck" in key]


def _model(representation: str = "mean-pool") -> PolicyValueNet:
    torch.manual_seed(7)
    return PolicyValueNet(embedding_dim=16, hidden_dim=32, board_representation=representation).eval()


DECISIONS = [
    view(),
    view(
        cards=[card(), hidden(), card(SPELL, location=HAND, position=0)],
        choices=[choice(ACTIVATE, named(2, SPELL, location=HAND)), choice(ATTACK, named(0)), choice(PASS)],
        history=[{"kind": ACTIVATE, "code": SPELL, "location": HAND, "description": 0}] * 3,
    ),
    view(
        cards=[card(OTHER_MONSTER), card()],
        question={**view()["question"], "kind": SELECT_CARDS},
        choices=[
            choice(TOGGLE, named(index, code), members=[named(index, code)])
            for index, code in ((0, OTHER_MONSTER), (1, MONSTER))
        ]
        + [choice(22), choice(PASS)],
    ),
]


@pytest.mark.parametrize("representation", ["mean-pool", "individual-tokens"])
def test_a_decision_scores_the_same_alone_as_in_a_batch(representation):
    model = _model(representation)
    batch = batched(*DECISIONS)
    with torch.no_grad():
        logits, values = model(batch)
        assert logits.shape == (3, 4)
        assert torch.isinf(logits[~batch["action_mask"]]).all()
        assert torch.isfinite(logits[batch["action_mask"]]).all()
        for index, item in enumerate(DECISIONS):
            alone_logits, alone_values = model(batched(item))
            answers = alone_logits.shape[1]
            assert torch.allclose(logits[index, :answers], alone_logits[0], atol=1e-5)
            assert torch.allclose(values[index], alone_values[0], atol=1e-5)


@pytest.mark.parametrize("representation", ["mean-pool", "individual-tokens"])
def test_the_order_of_the_cards_does_not_matter(representation):
    model = _model(representation)
    cards = [card(), card(OTHER_MONSTER, attack=1800), hidden()]
    choices = [choice(ATTACK, named(1, OTHER_MONSTER)), choice(PASS)]
    turned = [cards[2], cards[0], cards[1]]
    turned_choices = [choice(ATTACK, named(2, OTHER_MONSTER)), choice(PASS)]
    with torch.no_grad():
        first = model(batched(view(cards=cards, choices=choices)))
        second = model(batched(view(cards=turned, choices=turned_choices)))
    assert torch.allclose(first[0], second[0], atol=1e-5)
    assert torch.allclose(first[1], second[1], atol=1e-5)


def test_tokens_let_an_answer_see_each_card_on_the_field():
    choices = [choice(PASS), choice(22)]
    weak = view(cards=[hidden(), card(attack=100)], choices=choices)
    strong = view(cards=[hidden(), card(attack=3000)], choices=choices)
    with torch.no_grad():
        scores = {
            representation: [model(batched(item))[0] for item in (weak, strong)]
            for representation in ("mean-pool", "individual-tokens")
            for model in [_model(representation)]
        }
    assert not torch.allclose(*scores["individual-tokens"])
    # Both arms own the same parameters.
    counts = {
        representation: sum(p.numel() for p in _model(representation).parameters())
        for representation in scores
    }
    assert counts["mean-pool"] == counts["individual-tokens"]


def test_what_cannot_be_a_batch_is_refused():
    with pytest.raises(DuelError, match="empty"):
        library().batch([])
    with pytest.raises(DuelError, match="at least one answer"):
        library().batch([view(choices=[])])
