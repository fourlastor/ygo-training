"""The arrays the policy reads, described: their names, types and shapes,
the limits and vocabularies they obey, and the canonical description a
checkpoint records.

The arrays are built by the crates of ``native/ygo-model-input``
(``input/src/pack.rs``), the one definition of the model's input: for the
duels of the library's pool, and for any decision the library is handed
(``Environment.pack``, ``Environment.batch``).  Python builds none of them.
What is here tells the model how wide its inputs are, and a checkpoint
which input it was trained on, without relying on a Python class or a
version number; ``tests/test_packed.py`` holds it to what the library
builds.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

# Cards a seat can be shown at once: both hands, fields, Graveyards and
# banished piles, and its own Extra Deck.
MAX_CARDS = 160
MAX_HISTORY = 32
MAX_ACTION_MEMBERS = 64
CARD_COMPONENT_VOCAB = 1 << 16
LOCATION_VOCAB = 16
POSITION_VOCAB = 16
PHASE_VOCAB = 16
DECISION_KIND_VOCAB = 16
HINT_VOCAB = 32
ACTION_KIND_VOCAB = 32

# What is known of one card where it lies, beside its printed data.
CARD_STATUS = (
    "known",
    "level",
    "can_attack",
    "battles",
    "counters",
    "coin_tails",
    "coin_heads",
    "selected",
    "attacking",
    "attacked",
    "event",
    "chained",
    "targeted",
)
CARD_STATUS_COUNT = len(CARD_STATUS)

# The cards a decision is about, each in a slot of its own.
CONTEXT_SLOTS = (
    "chain_last",
    "chain_previous",
    "attacker",
    "attack_target",
    "event",
    "subject",
    "acting",
)
CONTEXT_COUNT = len(CONTEXT_SLOTS)

SCALARS = (
    "own_life_points",
    "opponent_life_points",
    "turn",
    "chain_length",
    "opponent_hand",
    "own_deck",
    "opponent_deck",
    "bias",
    "turn_player",
    "own_extra",
    "opponent_extra",
    "own_hand",
    "own_graveyard",
    "opponent_graveyard",
    "own_banished",
    "opponent_banished",
    "summon_used",
    "minimum",
    "maximum",
    "forced",
    "triggers",
    "coin_heads",
    "coin_tails",
    "step",
)
SCALAR_COUNT = len(SCALARS)

ACTION_SCALARS = (
    "has_card",
    "required",
    "value_low",
    "value_high",
    "members",
    "effect_index",
    "names_an_effect",
    "bias",
)
ACTION_SCALAR_COUNT = len(ACTION_SCALARS)

HISTORY_SCALARS = ("effect_index", "bias")
HISTORY_SCALAR_COUNT = len(HISTORY_SCALARS)

# A card's printed data, from its row of the card database: its kind,
# attribute, race, level and up to four set codes; then ATK, DEF, and
# independent monster, Spell and Trap subtype flags.
CATEGORY_COUNT = 8
NUMERIC_COUNT = 17

# Version 2 is built from the view of a ygo-policies seat: what a pilot is
# given at the same decision, with players relative to the seat.  Version 3
# (2026-10-16) is version 2 without the two deck lists, which a copy was
# found to play as well without (experiments/2026-10-15-without-deck-lists.md).
POLICY_INPUT_SCHEMA_VERSION = 3


def _array(dtype: str, *shape: int | str) -> dict[str, Any]:
    return {"dtype": dtype, "shape": list(shape)}


def _cards(prefix: str, *shape: int | str) -> dict[str, dict[str, Any]]:
    """A card where it lies: what every card of the input is made of."""
    return {
        f"{prefix}card": _array("int64", *shape),
        f"{prefix}location": _array("int64", *shape),
        f"{prefix}position": _array("int64", *shape),
        f"{prefix}controller": _array("int64", *shape),
        f"{prefix}categories": _array("int64", *shape, CATEGORY_COUNT),
        f"{prefix}numeric": _array("float32", *shape, NUMERIC_COUNT),
        f"{prefix}status": _array("float32", *shape, len(CARD_STATUS)),
    }


PACKED_ARRAY_SCHEMA: dict[str, dict[str, Any]] = {
    "state_length": _array("int64", 1),
    **_cards("state_", "cards"),
    **_cards("context_", len(CONTEXT_SLOTS)),
    "context_mask": _array("bool", len(CONTEXT_SLOTS)),
    "phase": _array("int64", 1),
    "decision_kind": _array("int64", 1),
    "decision_hint": _array("int64", 1),
    "scalars": _array("float32", len(SCALARS)),
    "action_kind": _array("int64", "actions"),
    **_cards("action_", "actions"),
    "action_scalars": _array("float32", "actions", len(ACTION_SCALARS)),
    **_cards("action_member_", "actions", "members"),
    "action_member_value": _array("int64", "actions", "members"),
    "action_member_order": _array("float32", "actions", "members"),
    "action_member_mask": _array("bool", "actions", "members"),
    "history_card": _array("int64", "history"),
    "history_location": _array("int64", "history"),
    "history_kind": _array("int64", "history"),
    "history_scalars": _array("float32", "history", len(HISTORY_SCALARS)),
    "history_length": _array("int64", 1),
}


def canonical_schema_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def schema_digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_schema_json(value).encode("utf8")).hexdigest()


def packed_policy_input_schema() -> dict[str, Any]:
    """Return the representation-independent packed-array contract."""

    return {
        "schema_version": POLICY_INPUT_SCHEMA_VERSION,
        "arrays": {
            key: {"dtype": spec["dtype"], "shape": list(spec["shape"])}
            for key, spec in PACKED_ARRAY_SCHEMA.items()
        },
        "fields": {
            "card_status": list(CARD_STATUS),
            "context": list(CONTEXT_SLOTS),
            "scalars": list(SCALARS),
            "action_scalars": list(ACTION_SCALARS),
            "history_scalars": list(HISTORY_SCALARS),
        },
        "limits": {
            "cards": MAX_CARDS,
            "history": MAX_HISTORY,
            "action_members": MAX_ACTION_MEMBERS,
        },
        "vocabularies": {
            "card_component": CARD_COMPONENT_VOCAB,
            "action_kind": ACTION_KIND_VOCAB,
            "decision_kind": DECISION_KIND_VOCAB,
            "hint": HINT_VOCAB,
            "location": LOCATION_VOCAB,
            "position": POSITION_VOCAB,
            "phase": PHASE_VOCAB,
        },
        "invariants": {
            "source": "the view of a ygo-policies seat",
            "players": "relative to the seat",
            "hidden_card_identity": "absent",
            "zone_number": "absent",
            "combat_values": "current, feature-specific-normalization",
        },
    }


def packed_policy_input_schema_digest() -> str:
    return schema_digest(packed_policy_input_schema())


def model_policy_input_schema(
    *, board_representation: str, leave_out: tuple[str, ...] = ()
) -> dict[str, Any]:
    """Describe how one model consumes the canonical packed arrays: of a
    model that reads all of them, as it was described before a model could
    leave parts out (`TrainerConfig.leave_out`)."""

    packed = packed_policy_input_schema()
    described = {
        "schema_version": POLICY_INPUT_SCHEMA_VERSION,
        "packed_schema_digest": schema_digest(packed),
        "packed_schema": packed,
        "board_representation": str(board_representation),
    }
    if leave_out:
        described["leave_out"] = list(leave_out)
    return described


def model_policy_input_schema_digest(
    *, board_representation: str, leave_out: tuple[str, ...] = ()
) -> str:
    return schema_digest(
        model_policy_input_schema(board_representation=board_representation, leave_out=leave_out)
    )


def validate_packed_policy_input(packed: Mapping[str, Any]) -> None:
    """Fail clearly when a packed record does not satisfy the declared schema."""

    import numpy as np

    expected_keys = set(PACKED_ARRAY_SCHEMA)
    actual_keys = set(packed)
    if actual_keys != expected_keys:
        raise ValueError(
            "Packed policy input fields differ from schema: "
            f"missing={sorted(expected_keys - actual_keys)}, "
            f"extra={sorted(actual_keys - expected_keys)}"
        )
    dimensions: dict[str, int] = {}
    for key, spec in PACKED_ARRAY_SCHEMA.items():
        value = packed[key]
        if not isinstance(value, np.ndarray):
            raise TypeError(f"Packed field {key!r} is not a NumPy array")
        if str(value.dtype) != spec["dtype"]:
            raise ValueError(
                f"Packed field {key!r} has dtype {value.dtype}, expected {spec['dtype']}"
            )
        shape = spec["shape"]
        if value.ndim != len(shape):
            raise ValueError(
                f"Packed field {key!r} has rank {value.ndim}, expected {len(shape)}"
            )
        for axis, expected in enumerate(shape):
            actual = int(value.shape[axis])
            if isinstance(expected, int):
                if actual != expected:
                    raise ValueError(
                        f"Packed field {key!r} axis {axis} is {actual}, expected {expected}"
                    )
            elif expected in dimensions and actual != dimensions[expected]:
                raise ValueError(
                    f"Packed dimension {expected!r} is inconsistent at {key!r}: "
                    f"{actual} != {dimensions[expected]}"
                )
            else:
                dimensions[expected] = actual
