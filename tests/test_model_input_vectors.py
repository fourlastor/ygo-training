"""The decisions the shared input crate checks itself against."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from ygo_training.environment import Environment, library_path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "model_input_vectors", ROOT / "tools/model_input_vectors.py"
)
assert SPEC is not None and SPEC.loader is not None
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)

VECTORS = ROOT / "native/ygo-model-input/input/vectors/decisions.json"


def _built() -> bool:
    try:
        library_path(ROOT)
    except FileNotFoundError:
        return False
    return True


def test_a_digest_is_fnv_of_the_bytes():
    # The published values of FNV-1a, 64 bits, for nothing and for "a".
    assert tool.digest(np.zeros(0, dtype=np.int64)) == "cbf29ce484222325"
    assert tool.digest(np.frombuffer(b"a", dtype=np.uint8)) == "af63dc4c8601ec8c"
    one = np.asarray([[1.5, 0.0]], dtype=np.float32)
    assert tool.told({"scalars": one, "mask": one != 0}) == [
        ["scalars", "float32", [1, 2], tool.digest(one)],
        ["mask", "bool", [1, 2], tool.digest(np.asarray([1, 0], dtype=np.uint8))],
    ]


def test_one_decision_of_each_kind_is_kept_and_its_cards_are_found():
    by_hand = tool.by_hand()
    crowded, _, unseen = by_hand
    # The same question about the same sort of thing is kept once.
    assert tool.kept([*by_hand, *by_hand]) == by_hand
    # A card the seat cannot see has no code, and is no card to look up.
    assert tool.codes(unseen) == {0}
    assert {tool.SPELL, tool.TRAP, tool.MONSTER, 4_007} <= tool.codes(crowded)


@pytest.mark.skipif(not _built(), reason="the environment is not built: cargo build --release")
def test_the_shared_vectors_are_what_the_library_gives():
    """The file the shared crate is held to is the library's own word on
    its decisions: a change of the input without the file written again
    fails here, and there.  The duels are not played again, so the file
    outlives a move of the pilots."""
    if not VECTORS.is_file():
        pytest.skip("the submodule is not checked out: git submodule update --init --recursive")
    kept = json.loads(VECTORS.read_text(encoding="utf8"))
    assert len(kept["decisions"]) > 100
    with Environment(ROOT) as environment:
        packed, batch = tool.made(environment, [decision["view"] for decision in kept["decisions"]])
    again = "the input has changed: write the vectors again (docs/commands.md)"
    for index, (decision, arrays) in enumerate(zip(kept["decisions"], packed, strict=True)):
        assert tool.told(arrays) == decision["arrays"], f"decision {index}: {again}"
    assert tool.told(batch) == kept["batch"], again


@pytest.mark.integration
@pytest.mark.skipif(not _built(), reason="the environment is not built: cargo build --release")
def test_the_vectors_are_written_the_same_twice():
    first = tool.vectors(1)
    assert len(first["decisions"]) > 20 and first["checkpoint_format"] == tool.FORMAT_VERSION
    assert json.dumps(first) == json.dumps(tool.vectors(1))
    # Every card the decisions of real duels name comes with its row of the
    # database (those made by hand name codes of no card).
    rows = {row[0] for row in first["cards"]}
    real = [decision["view"] for decision in first["decisions"][: -len(tool.by_hand())]]
    assert tool.codes(real) - {0} <= rows
