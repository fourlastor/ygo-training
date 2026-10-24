"""`tools/card_use.py`: what a network does with each card of the list it
plays, beside the pilot asked alongside it, at the decisions of its seat."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from ygo_training.environment import library_path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
spec = importlib.util.spec_from_file_location("card_use", ROOT / "tools/card_use.py")
tool = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(tool)


def _built() -> bool:
    try:
        library_path(ROOT)
    except FileNotFoundError:
        return False
    return True


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _built(), reason="the environment is not built: cargo build --release"),
]

SUITE = {
    "name": "small",
    "primary": "blackwing-vs-monarch",
    "cases": [
        {"name": "blackwing-vs-monarch", "learner": "blackwing", "opponent": "monarch", "games": 4}
    ],
}


@pytest.fixture(scope="module")
def model():
    import torch

    torch.set_num_threads(2)
    from ygo_training.model import PolicyValueNet

    torch.manual_seed(45)
    return PolicyValueNet().eval()


def test_each_card_of_the_list_is_counted_where_an_answer_uses_it(model):
    result = tool.measure(
        model,
        SUITE,
        learner_list=ROOT / "data/decks/darker-than-black.ydk",
        trained=["blackwing", "monarch"],
        seed=2012,
        device="cpu",
        threads=2,
        slots=4,
    )
    assert result["games"] == 4
    cards = result["cards"]
    assert cards["Blackwing - Bora the Spear"] == tool.OWN
    # Pot of Avarice is Monarch's, and no deck of the two holds Mistral.
    assert cards["Pot of Avarice"] == tool.TRAINED
    assert cards["Blackwing - Mistral the Silver Shield"] == tool.NEVER
    rows = result["counts"]
    assert rows and all(row["card"] in cards for row in rows)
    for row in rows:
        assert row["offered"] >= row["labelled"] >= row["pilot"]
        assert row["offered"] >= row["network"]
    used = [row for row in rows if row["question"] in tool.QUESTIONS and row["kind"] in tool.USES]
    assert sum(row["offered"] for row in used if row["group"] == tool.OWN) > 0
    assert "never trained on" in tool.summary(result)
