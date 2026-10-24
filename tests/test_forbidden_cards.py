"""Answers barred to a model in an evaluation (`pool.evaluate_games`,
`forbid`), and `tools/forbidden_cards.py`, which plays a suite's deals as they
are and with a group of the list's cards barred."""

from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from ygo_training.environment import library_path
from ygo_training.orchestration import evaluation_assignments
from ygo_training.pool import DuelPool, evaluate_games

ROOT = Path(__file__).resolve().parents[1]
VARIANT = ROOT / "data/decks/darker-than-black.ydk"
sys.path.insert(0, str(ROOT / "tools"))
spec = importlib.util.spec_from_file_location("forbidden_cards", ROOT / "tools/forbidden_cards.py")
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


@pytest.fixture(scope="module")
def model():
    import torch

    torch.set_num_threads(2)
    from ygo_training.model import PolicyValueNet

    torch.manual_seed(45)
    return PolicyValueNet().eval()


def games(duels, model, forbid=None, observe=None):
    return evaluate_games(
        duels,
        model,
        "cpu",
        learner="blackwing",
        assignments=evaluation_assignments(["monarch"], 4, 2012),
        limit=4096,
        forbid=forbid,
        observe=observe,
    )


def test_a_barred_answer_is_taken_only_where_nothing_else_is_offered(model):
    with DuelPool(ROOT, threads=2, slots=4) as duels:
        # The kind of answer the model takes most where it has a choice.
        kinds = Counter()

        def counted(round_, mine, chosen):
            for row in mine.tolist():
                if round_.arrays["action_mask"][row].sum() > 1:
                    kinds[int(round_.arrays["action_kind"][row, chosen[row]])] += 1

        games(duels, model, observe=counted)
        barred = kinds.most_common(1)[0][0]
        taken = []

        def forbid(round_, mine):
            return round_.arrays["action_kind"][mine] == barred

        def observe(round_, mine, chosen):
            for row in mine.tolist():
                kinds_ = round_.arrays["action_kind"][row]
                others = round_.arrays["action_mask"][row] & (kinds_ != barred)
                taken.append((int(kinds_[chosen[row]]) == barred, bool(others.any())))

        results = games(duels, model, forbid=forbid, observe=observe)
    assert taken and not any(was_barred and others for was_barred, others in taken)
    assert sum(result["barred"] for result in results) > 0


def test_a_group_of_the_lists_cards_is_barred_on_the_same_deals(model):
    suite = {
        "name": "small",
        "primary": "blackwing-vs-monarch",
        "cases": [
            {
                "name": "blackwing-vs-monarch",
                "learner": "blackwing",
                "opponent": "monarch",
                "games": 4,
            }
        ],
    }
    result = tool.measure(
        model,
        suite,
        learner_list=VARIANT,
        trained=["blackwing"],
        barred=[tool.NEVER],
        seed=2012,
        device="cpu",
        threads=2,
        slots=4,
    )
    conditions = result["conditions"]
    assert list(conditions) == [tool.AS_IT_IS, tool.NEVER]
    assert len(conditions[tool.AS_IT_IS]["won"]) == len(conditions[tool.NEVER]["won"]) == 4
    assert "Mind Control" in conditions[tool.NEVER]["cards"]
    assert conditions[tool.AS_IT_IS]["barred"] == 0
    assert "never trained on" in tool.summary(result)
    assert np.isfinite(sum(conditions[tool.NEVER]["won"]))
